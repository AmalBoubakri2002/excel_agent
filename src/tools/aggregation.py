import pandas as pd
import numpy as np


def run_aggregation_analysis(
    df: pd.DataFrame,
    group_by_col: str = "",
    agg_col: str = "",
) -> dict:
    """
    Effectue des agrégations sur le DataFrame.
    Si group_by_col est défini → groupby + stats.
    Sinon → distribution des colonnes catégorielles.
    """
    results: dict = {}

    # ── Distribution des colonnes texte ──────────────────────
    text_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in text_cols:
        counts = df[col].value_counts()
        results[f"{col}_distribution"] = {
            str(k): int(v) for k, v in counts.items()
        }

    # ── GroupBy si colonne spécifiée ─────────────────────────
    if group_by_col and group_by_col in df.columns:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            grouped = df.groupby(group_by_col)[numeric_cols].agg(
                ["count", "mean", "sum"]
            ).round(4)
            results["groupby"] = grouped.to_dict()

    # ── Comptage total ────────────────────────────────────────
    results["total_rows"] = len(df)

    return results