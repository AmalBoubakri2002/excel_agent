import pandas as pd
import numpy as np
from scipy import stats as scipy_stats


def run_statistical_analysis(df: pd.DataFrame) -> dict:
    """
    Calcule les statistiques descriptives complètes
    pour toutes les colonnes numériques du DataFrame.
    """
    results = {}
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return {"error": "Aucune colonne numérique disponible"}

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) == 0:
            continue

        # Test de normalité (Shapiro-Wilk, max 5000 échantillons)
        sample = series if len(series) <= 5000 else series.sample(5000, random_state=42)
        try:
            _, p_normal = scipy_stats.shapiro(sample)
            is_normal = bool(p_normal > 0.05)
        except Exception:
            p_normal  = None
            is_normal = None

        results[col] = {
            "count"     : int(len(series)),
            "mean"      : round(float(series.mean()), 4),
            "median"    : round(float(series.median()), 4),
            "std"       : round(float(series.std()), 4),
            "min"       : round(float(series.min()), 4),
            "max"       : round(float(series.max()), 4),
            "q25"       : round(float(series.quantile(0.25)), 4),
            "q75"       : round(float(series.quantile(0.75)), 4),
            "skewness"  : round(float(series.skew()), 4),
            "kurtosis"  : round(float(series.kurtosis()), 4),
            "is_normal" : is_normal,
            "p_normal"  : round(float(p_normal), 4) if p_normal else None,
        }

    return results