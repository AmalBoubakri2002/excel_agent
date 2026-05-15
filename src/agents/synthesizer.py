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
    pairs = metrics.get("pairs", [])
    if not pairs:
        return metrics.get("error", "Aucune paire de corrélation calculée")

    lines = [f"Colonnes analysées : {metrics.get('columns_analyzed', [])}"]
    lines.append("")
    for p in pairs[:5]:
        sig = "✓ significative" if p.get("significant") else "✗ non significative"
        lines.append(
            f"  {p['col1']} ↔ {p['col2']} : "
            f"r={p['pearson_r']} (p={p['p_value']}) "
            f"— {p['interpretation']} [{sig}]"
        )
    return "\n".join(lines)


def _format_regression_metrics(metrics: dict) -> str:
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
# DÉTECTION DE TEMPLATE NON REMPLI
# ─────────────────────────────────────────────────────────────

def _is_template_response(data: dict, response: FinalResponse) -> bool:
    """
    Détecte si le LLM a retourné le template JSON sans le remplir.
    Exemples de faux positifs à détecter :
    - key_metrics = {"métrique 1": "valeur lisible"}
    - answer = "réponse claire et directe à la question en français"
    - suggestions = ["suggestion d'analyse complémentaire 1"]
    """
    # Vérifier les métriques clés
    for k, v in data.get("key_metrics", {}).items():
        if any(phrase in str(k).lower() for phrase in [
            "métrique", "metric", "key metric"
        ]):
            return True
        if any(phrase in str(v).lower() for phrase in [
            "valeur lisible", "lisible", "readable"
        ]):
            return True

    # Vérifier la réponse
    template_answer_phrases = [
        "réponse claire et directe",
        "réponse en langage naturel",
        "réponse directe à la question",
        "question en français",
        "clear and direct answer",
    ]
    if any(phrase in response.answer.lower() for phrase in template_answer_phrases):
        return True

    # Vérifier les suggestions
    for s in data.get("suggestions", []):
        if any(phrase in str(s).lower() for phrase in [
            "complémentaire 1",
            "complémentaire 2",
            "complementary 1",
        ]):
            return True

    # Réponse vide
    if not response.answer.strip():
        return True

    return False


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
    Utilisé si Ollama échoue ou retourne le template non rempli.
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
            is_normal = stats.get("is_normal")
            distrib = "suit une distribution normale" if is_normal else \
                      "ne suit pas une distribution normale"
            answer_parts.append(
                f"La moyenne de {col} est {mean} "
                f"(écart-type : {std}, "
                f"min : {stats.get('min')}, "
                f"max : {stats.get('max')}). "
                f"La distribution {distrib}."
            )

    elif result.analysis_type == "correlation":
        pairs = metrics.get("pairs", [])
        for p in pairs[:3]:
            key_metrics[f"r ({p['col1']}↔{p['col2']})"] = str(p["pearson_r"])
            sig_txt = "statistiquement significative" \
                      if p.get("significant") else "non significative"
            answer_parts.append(
                f"{p['col1']} et {p['col2']} : "
                f"{p['interpretation']} "
                f"(r={p['pearson_r']}, p={p['p_value']}, {sig_txt})."
            )

    elif result.analysis_type == "regression":
        r2  = metrics.get("r2_score", "N/A")
        rmse = metrics.get("rmse", "N/A")
        key_metrics["R²"]        = str(r2)
        key_metrics["RMSE"]      = str(rmse)
        key_metrics["Target"]    = str(metrics.get("target_col"))
        qualite = "excellent" if isinstance(r2, float) and r2 > 0.8 \
                  else "correct" if isinstance(r2, float) and r2 > 0.5 \
                  else "faible"
        answer_parts.append(
            f"Le modèle de régression prédit '{metrics.get('target_col')}' "
            f"avec un R²={r2} ({qualite}) et une erreur RMSE={rmse}. "
            f"{metrics.get('interpretation', '')}"
        )

    elif result.analysis_type == "clustering":
        k    = metrics.get("n_clusters", "N/A")
        sil  = metrics.get("silhouette_score", "N/A")
        key_metrics["Nombre de clusters"]   = str(k)
        key_metrics["Score de silhouette"]  = str(sil)
        answer_parts.append(metrics.get("interpretation", ""))

        for cluster in metrics.get("cluster_stats", [])[:3]:
            size = cluster.get("size", "?")
            means = {
                k: v for k, v in cluster.items()
                if k not in {"cluster", "size"}
            }
            means_str = ", ".join(
                f"{k.replace('_mean','')}={v}"
                for k, v in list(means.items())[:3]
            )
            key_metrics[f"Cluster {cluster['cluster']} ({size} pts)"] = means_str

    else:
        total = metrics.get("total_rows", "N/A")
        key_metrics["Total lignes"] = str(total)
        for key, val in metrics.items():
            if key.endswith("_distribution") and isinstance(val, dict):
                col = key.replace("_distribution", "")
                top = sorted(val.items(), key=lambda x: x[1], reverse=True)
                for cat, count in top[:5]:
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
        suggestions  = [
            "Explorez d'autres colonnes pour approfondir l'analyse.",
            "Essayez une analyse de corrélation entre les variables numériques.",
        ],
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
    Stratégie : LLM d'abord, fallback déterministe si LLM échoue
    ou retourne un template non rempli.
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

        # ── Garde-fou : détecter template non rempli ─────────
        if _is_template_response(data, response):
            raise ValueError("LLM a retourné le template non rempli — fallback")

        print("   ✅ Synthèse générée par LLM")
        return response

    except Exception as e:
        print(f"   ⚠️  LLM indisponible ou template détecté ({e}), fallback")

    # ── Fallback déterministe ─────────────────────────────────
    print("   ✅ Synthèse générée par fallback")
    return _build_fallback_response(original_question, result, loaded)