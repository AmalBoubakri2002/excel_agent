import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


def _interpret_correlation(r: float) -> str:
    """Interprète la force d'une corrélation."""
    abs_r = abs(r)
    if abs_r >= 0.8:
        direction = "positive" if r > 0 else "négative"
        return f"très forte corrélation {direction}"
    elif abs_r >= 0.6:
        direction = "positive" if r > 0 else "négative"
        return f"forte corrélation {direction}"
    elif abs_r >= 0.4:
        direction = "positive" if r > 0 else "négative"
        return f"corrélation modérée {direction}"
    elif abs_r >= 0.2:
        return "corrélation faible"
    else:
        return "pas de corrélation significative"


def run_correlation_analysis(df: pd.DataFrame) -> dict:
    """
    Calcule la matrice de corrélation de Pearson
    et les p-values associées pour toutes les paires de colonnes.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 2:
        return {
            "error": (
                f"Corrélation impossible : "
                f"seulement {len(numeric_cols)} colonne(s) numérique(s). "
                f"Il en faut au moins 2."
            ),
            "available_columns": numeric_cols,
        }

    # Matrice de corrélation Pearson
    corr_matrix = df[numeric_cols].corr(method="pearson")

    # P-values et corrélations par paire
    pairs = []
    for i, col1 in enumerate(numeric_cols):
        for col2 in numeric_cols[i+1:]:
            series1 = df[col1].dropna()
            series2 = df[col2].dropna()

            # Aligner les index
            common = series1.index.intersection(series2.index)
            s1, s2 = series1[common], series2[common]

            if len(common) < 3:
                continue

            r, p_value = scipy_stats.pearsonr(s1, s2)

            pairs.append({
                "col1"          : col1,
                "col2"          : col2,
                "pearson_r"     : round(float(r), 4),
                "p_value"       : round(float(p_value), 4),
                "significant"   : bool(p_value < 0.05),
                "interpretation": _interpret_correlation(r),
                "n_samples"     : len(common),
            })

    # Trier par corrélation absolue décroissante
    pairs.sort(key=lambda x: abs(x["pearson_r"]), reverse=True)

    return {
        "correlation_matrix": corr_matrix.round(4).to_dict(),
        "pairs"             : pairs,
        "n_columns"         : len(numeric_cols),
        "columns_analyzed"  : numeric_cols,
    }