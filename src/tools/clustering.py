import pandas as pd
import numpy as np
from sklearn.cluster  import KMeans
from sklearn.metrics  import silhouette_score
from sklearn.preprocessing import StandardScaler


def _find_optimal_k(X: np.ndarray, k_max: int = 6) -> int:
    """
    Trouve le k optimal via le score de silhouette.
    Teste k de 2 à k_max et retourne le meilleur.
    """
    best_k     = 2
    best_score = -1

    for k in range(2, min(k_max + 1, len(X))):
        try:
            km    = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = km.fit_predict(X)
            score  = silhouette_score(X, labels)
            if score > best_score:
                best_score = score
                best_k     = k
        except Exception:
            continue

    return best_k


def run_clustering_analysis(
    df: pd.DataFrame,
    n_clusters: int = 0,  # 0 = détection automatique
) -> dict:
    """
    Effectue un clustering KMeans sur les colonnes numériques.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) < 1:
        return {"error": "Aucune colonne numérique pour le clustering"}

    X_df  = df[numeric_cols].dropna()
    X     = StandardScaler().fit_transform(X_df)

    if len(X) < 4:
        return {"error": f"Pas assez de données : {len(X)} lignes (min 4)"}

    # Déterminer k
    k = n_clusters if n_clusters >= 2 else _find_optimal_k(X)

    # KMeans final
    km     = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X)

    sil_score = float(silhouette_score(X, labels)) if k > 1 else 0.0

    # Statistiques par cluster
    df_result          = X_df.copy()
    df_result["cluster"] = labels

    cluster_stats = []
    for cluster_id in sorted(df_result["cluster"].unique()):
        subset = df_result[df_result["cluster"] == cluster_id]
        stats  = {"cluster": int(cluster_id), "size": len(subset)}
        for col in numeric_cols:
            stats[f"{col}_mean"] = round(float(subset[col].mean()), 4)
        cluster_stats.append(stats)

    return {
        "n_clusters"      : k,
        "silhouette_score": round(sil_score, 4),
        "cluster_stats"   : cluster_stats,
        "columns_used"    : numeric_cols,
        "interpretation"  : (
            f"{k} clusters détectés "
            f"(silhouette={sil_score:.2f} — "
            f"{'bon' if sil_score > 0.5 else 'moyen' if sil_score > 0.25 else 'faible'})"
        ),
    }