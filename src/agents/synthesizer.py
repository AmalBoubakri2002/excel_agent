# src/agents/synthesizer.py
# Agent 6 – Synthétiseur
# Transforme les résultats bruts en réponse claire pour l'utilisateur

import json
import re
from langchain_ollama import OllamaLLM

from src.models import (
    AnalysisResult, AnalysisStatus,
    FinalResponse, MappingResult, LoadedData,
)
from src.utils.prompts import build_synthesizer_prompt


# ─────────────────────────────────────────────────────────────
# FORMATAGE DES MÉTRIQUES POUR LE PROMPT
# ─────────────────────────────────────────────────────────────

def _format_statistical_metrics(metrics: dict) -> str:
    """Formate les stats descriptives pour le prompt."""
    lines = []
    for col, stats in metrics.items():
        if not isinstance(stats, dict) or "error" in stats:
            continue
        lines.append(f"Colonne '{col}' :")
        lines.append(f"  - Nombre de valeurs : {stats.get('count', 'N/A')}")
        lines.append(f"  - Moyenne           : {stats.get('mean', 'N/A')}")
        lines.append(f"  - Médiane           : {stats.get('median', 'N/A')}")
        lines.append(f"  - Écart-type        : {stats.get('std', 'N/A')}")
        lines.append(f"  - Min / Max         : {stats.get('min')} / {stats.get('max')}")
        is_normal = stats.get("is_normal")
        if is_normal is not None:
            lines.append(
                f"  - Distribution      : "
                f"{'normale' if is_normal else 'non normale'} "
                f"(p={stats.get('p_normal')})"
            )
    return "\n".join(lines) if lines else "Aucune statistique disponible"


def _format_correlation_metrics(metrics: dict) -> str:
    """Formate les corrélations pour le prompt."""
    pairs = metrics.get("pairs", [])
    if not pairs:
        return metrics.get("error", "Aucune paire de corrélation calculée")

    lines = [f"Colonnes analysées : {metrics.get('columns_analyzed', [])}"]
    lines.append("")
    for p in pairs[:5]:  # max 5 paires pour ne pas surcharger le prompt
        sig = "✓ significative" if p.get("significant") else "✗ non significative"
        lines.append(
            f"  {p['col1']} ↔ {p['col2']} : "
            f"r={p['pearson_r']} (p={p['p_value']}) "
            f"— {p['interpretation']} [{sig}]"
        )
    return "\n".join(lines)


def _format_regression_metrics(metrics: dict) -> str:
    """Formate les résultats de régression pour le prompt."""
    if "error" in metrics:
        return f"Erreur : {metrics['error']}"

    lines = [
        f"Variable cible      : {metrics.get('target_col')}",
        f"Variables utilisées : {metrics.get('feature_cols')}",
        f"R² (qualité)        : {metrics.get('r2_score')} "
        f"({'excellent' if metrics.get('r2_score', 0) > 0.8 else 'correct' if metrics.get('r2_score', 0) > 0.5 else 'faible'})",
        f"RMSE (erreur moy)   : {metrics.get('rmse')}",
        f"MAE                 : {metrics.get('mae')}",
        f"Échantillon train   : {metrics.get('n_train')} lignes",
        f"Coefficients        : {metrics.get('coefficients')}",
    ]
    return "\n".join(lines)


def _format_clustering_metrics(metrics: dict) -> str:
    """Formate les résultats de clustering pour le prompt."""
    if "error" in metrics:
        return f"Erreur : {metrics['error']}"

    lines = [
        f"Nombre de clusters  : {metrics.get('n_clusters')}",
        f"Score de silhouette : {metrics.get('silhouette_score')}",
        f"Interprétation      : {metrics.get('interpretation')}",
        "",
        "Caractéristiques par cluster :",
    ]
    for cluster in metrics.get("cluster_stats", []):
        lines.append(
            f"  Cluster {cluster['cluster']} "
            f"({cluster['size']} éléments) :"
        )
        for k, v in cluster.items():
            if k not in {"cluster", "size"}:
                lines.append(f"    {k} : {v}")
    return "\n".join(lines)


def _format_aggregation_metrics(metrics: dict) -> str:
    """Formate les résultats d'agrégation pour le prompt."""
    lines = [f"Total lignes : {metrics.get('total_rows', 'N/A')}"]
    for key, val in metrics.items():
        if key == "total_rows":
            continue
        if key.endswith("_distribution"):
            col_name = key.replace("_distribution", "")
            lines.append(f"\nDistribution de '{col_name}' :")
            if isinstance(val, dict):
                for cat, count in list(val.items())[:10]:
                    lines.append(f"  {cat} : {count}")
    return "\n".join(lines)


def _format_metrics_for_prompt(result: AnalysisResult) -> str:
    """
    Routeur : formate les métriques selon le type d'analyse.
    """
    atype   = result.analysis_type
    metrics = result.metrics

    if "error" in metrics:
        return f"Erreur lors de l'analyse : {metrics['error']}"

    if atype == "statistical":
        return _format_statistical_metrics(metrics)
    elif atype == "correlation":
        return _format_correlation_metrics(metrics)
    elif atype == "regression":
        return _format_regression_metrics(metrics)
    elif atype == "clustering":
        return _format_clustering_metrics(metrics)
    else:
        return _format_aggregation_metrics(metrics)


# ─────────────────────────────────────────────────────────────
# FALLBACK SANS LLM
# ─────────────────────────────────────────────────────────────

def _build_fallback_response(
    question: str,
    result: AnalysisResult,
    loaded: LoadedData,
) -> FinalResponse:
    """
    Génère une réponse de secours sans LLM.
    Utilisé si Ollama échoue ou retourne une réponse vide.
    """
    metrics      = result.metrics
    key_metrics  : dict[str, str] = {}
    answer_parts : list[str]      = []

    if result.analysis_type == "statistical":
        for col, stats in metrics.items():
            if not isinstance(stats, dict) or "error" in stats:
                continue
            mean = stats.get("mean", "N/A")
            std  = stats.get("std",  "N/A")
            key_metrics[f"Moyenne ({col})"]    = str(mean)
            key_metrics[f"Écart-type ({col})"] = str(std)
            key_metrics[f"Min ({col})"]        = str(stats.get("min"))
            key_metrics[f"Max ({col})"]        = str(stats.get("max"))
            answer_parts.append(
                f"La moyenne de {col} est {mean} "
                f"(écart-type : {std}, "
                f"min : {stats.get('min')}, "
                f"max : {stats.get('max')})."
            )

    elif result.analysis_type == "correlation":
        pairs = metrics.get("pairs", [])
        for p in pairs[:3]:
            key_metrics[f"r ({p['col1']}↔{p['col2']})"] = str(p["pearson_r"])
            answer_parts.append(
                f"{p['col1']} et {p['col2']} : "
                f"{p['interpretation']} "
                f"(r={p['pearson_r']}, p={p['p_value']})."
            )

    elif result.analysis_type == "regression":
        r2 = metrics.get("r2_score", "N/A")
        key_metrics["R²"]   = str(r2)
        key_metrics["RMSE"] = str(metrics.get("rmse"))
        answer_parts.append(
            f"Le modèle de régression obtient un R²={r2}. "
            f"{metrics.get('interpretation', '')}"
        )

    elif result.analysis_type == "clustering":
        k   = metrics.get("n_clusters", "N/A")
        sil = metrics.get("silhouette_score", "N/A")
        key_metrics["Clusters"]   = str(k)
        key_metrics["Silhouette"] = str(sil)
        answer_parts.append(metrics.get("interpretation", ""))

    else:
        total = metrics.get("total_rows", "N/A")
        key_metrics["Total lignes"] = str(total)
        for key, val in metrics.items():
            if key.endswith("_distribution") and isinstance(val, dict):
                col = key.replace("_distribution", "")
                for cat, count in list(val.items())[:5]:
                    key_metrics[f"{col} – {cat}"] = str(count)
                    answer_parts.append(f"{cat} : {count} occurrence(s)")

    answer = " ".join(answer_parts) if answer_parts else \
        "Analyse effectuée. Consultez les métriques pour les détails."

    return FinalResponse(
        answer       = answer,
        data_summary = (
            f"Basé sur {loaded.n_rows} lignes, "
            f"feuille '{loaded.sheet_name}'"
        ),
        key_metrics  = key_metrics,
        warnings     = result.warnings,
        suggestions  = ["Explorez d'autres colonnes pour approfondir l'analyse."],
    )


# ─────────────────────────────────────────────────────────────
# AGENT PRINCIPAL
# ─────────────────────────────────────────────────────────────

def run_synthesizer(
    result: AnalysisResult,
    mapping: MappingResult,
    loaded: LoadedData,
    original_question: str,
) -> FinalResponse:
    """
    Point d'entrée de l'Agent 6 – Synthétiseur.

    Reçoit l'AnalysisResult (Agent 5).
    Retourne une FinalResponse en langage naturel.
    """
    print(f"\n💬 Synthèse de la réponse...")

    # Cas d'échec total de l'analyse
    if result.status == AnalysisStatus.FAILED:
        return FinalResponse(
            answer       = f"L'analyse a échoué : {result.error_message}",
            data_summary = f"Feuille '{loaded.sheet_name}', {loaded.n_rows} lignes",
            key_metrics  = {},
            warnings     = result.warnings,
        )

    # ── Formater les métriques pour le prompt ────────────────
    metrics_summary = _format_metrics_for_prompt(result)
    data_summary    = (
        f"{loaded.n_rows} lignes analysées, "
        f"feuille '{loaded.sheet_name}', "
        f"colonnes : {list(loaded.dataframe.columns)}"
    )

    # ── Appel au LLM ─────────────────────────────────────────
    try:
        llm = OllamaLLM(
            model="qwen2.5:0.5b",
            temperature=0,
            format="json",
        )

        prompt = build_synthesizer_prompt(
            question        = original_question,
            analysis_type   = result.analysis_type,
            metrics_summary = metrics_summary,
            data_summary    = data_summary,
        )

        print("   ⏳ Appel à Ollama pour la synthèse...")
        raw       = llm.invoke(prompt)
        raw_clean = raw.strip()

        match = re.search(r'\{.*\}', raw_clean, re.DOTALL)
        if match:
            data = json.loads(match.group())
        else:
            data = json.loads(raw_clean)

        response = FinalResponse(
            answer       = data.get("answer", ""),
            data_summary = data_summary,
            key_metrics  = {
                str(k): str(v)
                for k, v in data.get("key_metrics", {}).items()
            },
            warnings     = result.warnings,
            suggestions  = data.get("suggestions", []),
        )

        if not response.answer.strip():
            raise ValueError("Réponse LLM vide")

        print("   ✅ Synthèse générée par LLM")
        return response

    except Exception as e:
        print(f"   ⚠️  LLM indisponible ({e}), utilisation du fallback")

    # ── Fallback déterministe ─────────────────────────────────
    print("   ✅ Synthèse générée par fallback")
    return _build_fallback_response(original_question, result, loaded)