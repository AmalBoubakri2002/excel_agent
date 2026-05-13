import pandas as pd
import numpy as np

from src.models import (
    AnalysisResult, AnalysisStatus, AnalysisType,
    TransformPlan, MappingResult, ExcelStructure,
)
from src.tools.statistical  import run_statistical_analysis
from src.tools.correlation  import run_correlation_analysis
from src.tools.regression   import run_regression_analysis
from src.tools.clustering   import run_clustering_analysis
from src.tools.aggregation  import run_aggregation_analysis
from src.agents.loader      import _load_raw_dataframe


def _ensure_enough_columns(
    plan: TransformPlan,
    mapping: MappingResult,
    structure: ExcelStructure,
) -> pd.DataFrame:
    """
    Garde-fou final : si le DataFrame n'a qu'une colonne
    pour une analyse qui en nécessite plusieurs (correlation,
    regression, clustering), on recharge toutes les colonnes
    numériques de la feuille directement.

    Version corrigée :
    - exclusion des colonnes presque vides
    - évite les colonnes du type "Colonne_Vide"
    """
    df = plan.dataframe.copy()

    analysis_type = mapping.analysis_type

    needs_multiple = analysis_type in {
        AnalysisType.CORRELATION,
        AnalysisType.REGRESSION,
        AnalysisType.CLUSTERING,
    }

    numeric_count = len(
        df.select_dtypes(include=[np.number]).columns
    )

    if needs_multiple and numeric_count < 2:
        print(
            f"   ⚡ Rechargement automatique : "
            f"'{analysis_type.value}' nécessite ≥2 colonnes numériques"
        )

        # Fallback → première feuille pertinente
        sheet = structure.relevant_sheets[0]

        df_full = _load_raw_dataframe(
            structure.file_path,
            sheet.name,
            sheet.header_row_index,
        )

        # ─────────────────────────────────────────────
        # Colonnes numériques uniquement
        # + suppression des colonnes trop vides
        # ─────────────────────────────────────────────
        num_cols = df_full.select_dtypes(
            include=[np.number]
        ).columns.tolist()

        # Exclure les colonnes avec >50% de NaN
        num_cols = [
            c for c in num_cols
            if df_full[c].isna().mean() < 0.5
        ]

        if num_cols:
            df = df_full[num_cols].dropna()

            print(
                f"   ✅ Colonnes rechargées : {num_cols}"
            )
        else:
            print(
                "   ⚠️ Aucune colonne numérique exploitable trouvée"
            )

    return df


def run_analyst(
    plan: TransformPlan,
    mapping: MappingResult,
    structure: ExcelStructure,
) -> AnalysisResult:
    """
    Point d'entrée de l'Agent 5 – Analyste.

    Reçoit le TransformPlan (Agent 4).
    Sélectionne et exécute la bonne analyse.
    Retourne un AnalysisResult.
    """

    analysis_type = mapping.analysis_type
    warnings = list(plan.warnings)

    print(
        f"\n🔬 Analyse en cours : "
        f"{analysis_type.value}..."
    )

    # ─────────────────────────────────────────────
    # Garde-fou : recharger si colonnes insuffisantes
    # ─────────────────────────────────────────────
    df = _ensure_enough_columns(
        plan,
        mapping,
        structure,
    )

    try:

        # ====================================================
        # ROUTAGE VERS LA BONNE ANALYSE
        # ====================================================

        if analysis_type == AnalysisType.STATISTICAL:

            metrics = run_statistical_analysis(df)

            result_table = [
                {
                    "colonne": col,
                    **stats
                }
                for col, stats in metrics.items()
                if isinstance(stats, dict)
                and "error" not in stats
            ]

        elif analysis_type == AnalysisType.CORRELATION:

            metrics = run_correlation_analysis(df)

            result_table = metrics.get(
                "pairs",
                []
            )

        elif analysis_type == AnalysisType.REGRESSION:

            metrics = run_regression_analysis(
                df,
                target_col=plan.target_column,
                feature_cols=plan.feature_columns,
            )

            result_table = [metrics]

        elif analysis_type == AnalysisType.CLUSTERING:

            metrics = run_clustering_analysis(df)

            result_table = metrics.get(
                "cluster_stats",
                []
            )

        elif analysis_type in {
            AnalysisType.AGGREGATION,
            AnalysisType.CLASSIFICATION,
            AnalysisType.UNKNOWN,
        }:

            # Classification sans ML
            # → fallback agrégation intelligente

            group_col = next(
                (
                    col.column_name
                    for col in mapping.selected_columns
                    if col.role in {
                        "group_by",
                        "filter",
                    }
                ),
                ""
            )

            metrics = run_aggregation_analysis(
                df,
                group_by_col=group_col,
            )

            result_table = [
                {
                    "type": k,
                    "valeur": str(v)[:200]
                }
                for k, v in metrics.items()
            ]

        else:

            metrics = run_statistical_analysis(df)

            result_table = []

        # ====================================================
        # Vérification erreurs métriques
        # ====================================================

        if (
            isinstance(metrics, dict)
            and "error" in metrics
        ):

            print(
                f"   ⚠️ {metrics['error']}"
            )

            return AnalysisResult(
                status=AnalysisStatus.PARTIAL,
                analysis_type=analysis_type.value,
                metrics=metrics,
                result_table=result_table,
                warnings=warnings + [
                    metrics["error"]
                ],
            )

        print("   ✅ Analyse terminée")

        return AnalysisResult(
            status=AnalysisStatus.SUCCESS,
            analysis_type=analysis_type.value,
            metrics=metrics,
            result_table=result_table,
            warnings=warnings,
        )

    except Exception as e:

        print(
            f"   ❌ Erreur analyse : {e}"
        )

        return AnalysisResult(
            status=AnalysisStatus.FAILED,
            analysis_type=analysis_type.value,
            metrics={},
            warnings=warnings,
            error_message=str(e),
        )