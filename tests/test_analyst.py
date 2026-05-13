import pytest
import pandas as pd
import numpy as np

from src.tools.statistical import run_statistical_analysis
from src.tools.correlation import run_correlation_analysis
from src.tools.regression  import run_regression_analysis
from src.tools.clustering  import run_clustering_analysis
from src.tools.aggregation import run_aggregation_analysis


@pytest.fixture
def df_numeric():
    np.random.seed(42)
    return pd.DataFrame({
        "Temperature": np.random.normal(65, 10, 100),
        "Pression"   : np.random.normal(1.5, 0.3, 100),
        "Debit"      : np.random.normal(120, 20, 100),
    })


@pytest.fixture
def df_mixed():
    np.random.seed(42)
    return pd.DataFrame({
        "Temperature": np.random.normal(65, 10, 50),
        "Statut"     : np.random.choice(["Nominal", "Alerte", "Critique"], 50),
        "Zone"       : np.random.choice(["Zone_A", "Zone_B"], 50),
    })


# ── Tests statistiques ───────────────────────────────────────

def test_statistical_retourne_stats(df_numeric):
    result = run_statistical_analysis(df_numeric)
    assert "Temperature" in result
    assert "mean" in result["Temperature"]
    assert "std"  in result["Temperature"]
    assert result["Temperature"]["count"] == 100


def test_statistical_sans_colonnes_numeriques():
    df = pd.DataFrame({"Texte": ["a", "b", "c"]})
    result = run_statistical_analysis(df)
    assert "error" in result


# ── Tests corrélation ────────────────────────────────────────

def test_correlation_deux_colonnes(df_numeric):
    result = run_correlation_analysis(df_numeric)
    assert "pairs" in result
    assert len(result["pairs"]) > 0
    assert "pearson_r" in result["pairs"][0]


def test_correlation_une_seule_colonne():
    df = pd.DataFrame({"Temperature": [1.0, 2.0, 3.0]})
    result = run_correlation_analysis(df)
    assert "error" in result


def test_correlation_valeurs(df_numeric):
    # Toutes les corrélations sont entre -1 et 1
    result = run_correlation_analysis(df_numeric)
    for pair in result["pairs"]:
        assert -1.0 <= pair["pearson_r"] <= 1.0


# ── Tests régression ─────────────────────────────────────────

def test_regression_basique(df_numeric):
    result = run_regression_analysis(
        df_numeric,
        target_col   = "Temperature",
        feature_cols = ["Pression", "Debit"],
    )
    assert "r2_score" in result
    assert "rmse"     in result
    assert -1.0 <= result["r2_score"] <= 1.0


def test_regression_cible_manquante(df_numeric):
    result = run_regression_analysis(
        df_numeric,
        target_col   = "",
        feature_cols = ["Pression"],
    )
    assert "error" in result


# ── Tests clustering ─────────────────────────────────────────

def test_clustering_basique(df_numeric):
    result = run_clustering_analysis(df_numeric, n_clusters=3)
    assert result["n_clusters"] == 3
    assert "cluster_stats"    in result
    assert "silhouette_score" in result
    assert len(result["cluster_stats"]) == 3


def test_clustering_auto_k(df_numeric):
    result = run_clustering_analysis(df_numeric)
    assert result["n_clusters"] >= 2


# ── Tests agrégation ─────────────────────────────────────────

def test_aggregation_distribution(df_mixed):
    result = run_aggregation_analysis(df_mixed)
    assert "total_rows" in result
    assert result["total_rows"] == 50
    assert "Statut_distribution" in result


def test_aggregation_groupby(df_mixed):
    result = run_aggregation_analysis(df_mixed, group_by_col="Zone")
    assert "groupby" in result