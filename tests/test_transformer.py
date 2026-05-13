import pytest
import pandas as pd
import numpy as np

from src.agents.transformer import (
    _fill_missing_values,
    _encode_categoricals,
    _normalize_numerics,
    _identify_roles,
)
from src.models import (
    TransformType, AnalysisType,
    MappingResult, ColumnSelection, LoadedData,
    ExcelStructure, SheetInfo, ColumnInfo, ColumnType,
)


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def df_with_nulls():
    return pd.DataFrame({
        "Temperature": [65.0, None, 70.0, None, 80.0],
        "Statut":      ["Nominal", None, "Alerte", "Nominal", None],
    })


@pytest.fixture
def df_categorical():
    return pd.DataFrame({
        "Temperature": [65.0, 70.0, 80.0],
        "Statut":      ["Nominal", "Alerte", "Critique"],
    })


@pytest.fixture
def df_numeric():
    return pd.DataFrame({
        "Temperature": [65.0, 70.0, 80.0, 60.0],
        "Pression":    [1.5,  1.8,  1.2,  2.0],
    })


# ── Tests _fill_missing_values ───────────────────────────────

def test_fill_numeric_with_median(df_with_nulls):
    transforms, summary = [], []
    result = _fill_missing_values(df_with_nulls.copy(), transforms, summary)

    assert result["Temperature"].isna().sum() == 0
    # médiane de [65, 70, 80] = 70
    assert result["Temperature"].iloc[1] == pytest.approx(70.0)
    assert any(t.transform_type == TransformType.FILL_MEDIAN for t in transforms)


def test_fill_text_with_mode(df_with_nulls):
    transforms, summary = [], []
    result = _fill_missing_values(df_with_nulls.copy(), transforms, summary)

    assert result["Statut"].isna().sum() == 0
    assert any(t.transform_type == TransformType.FILL_MODE for t in transforms)


# ── Tests _encode_categoricals ───────────────────────────────

def test_encode_for_classification(df_categorical):
    transforms, summary = [], []
    result = _encode_categoricals(
        df_categorical.copy(), transforms, summary,
        AnalysisType.CLASSIFICATION
    )
    # Statut doit être numérique après encodage
    assert pd.api.types.is_numeric_dtype(result["Statut"])
    assert any(t.transform_type == TransformType.ENCODE_LABEL for t in transforms)


def test_no_encode_for_statistical(df_categorical):
    """Pour une analyse statistique, on n'encode pas."""
    transforms, summary = [], []
    result = _encode_categoricals(
        df_categorical.copy(), transforms, summary,
        AnalysisType.STATISTICAL
    )
    # Statut reste texte
    assert pd.api.types.is_object_dtype(result["Statut"])
    assert len(transforms) == 0


# ── Tests _normalize_numerics ────────────────────────────────

def test_normalize_for_regression(df_numeric):
    transforms, summary = [], []
    result = _normalize_numerics(
        df_numeric.copy(), transforms, summary,
        AnalysisType.REGRESSION, target_col="Temperature"
    )
    # Pression normalisée → moyenne ≈ 0
    assert abs(result["Pression"].mean()) < 1e-10
    # Temperature non normalisée (c'est la target)
    assert result["Temperature"].mean() != pytest.approx(0.0)
    assert any(t.transform_type == TransformType.NORMALIZE for t in transforms)


def test_no_normalize_for_statistical(df_numeric):
    """Pour une analyse statistique, on ne normalise pas."""
    transforms, summary = [], []
    result = _normalize_numerics(
        df_numeric.copy(), transforms, summary,
        AnalysisType.STATISTICAL, target_col=""
    )
    # Valeurs inchangées
    assert result["Temperature"].iloc[0] == pytest.approx(65.0)
    assert len(transforms) == 0


# ── Tests _identify_roles ────────────────────────────────────

def test_identify_roles_depuis_mapping(df_numeric):
    mapping = MappingResult(
        interpreted_question="test",
        analysis_type=AnalysisType.REGRESSION,
        selected_columns=[
            ColumnSelection(sheet_name="Mesures", column_name="Temperature",
                            role="target", reason=""),
            ColumnSelection(sheet_name="Mesures", column_name="Pression",
                            role="feature", reason=""),
        ],
        confidence=0.9,
    )
    loaded = LoadedData(
        dataframe=df_numeric, sheet_name="Mesures",
        n_rows=4, n_cols=2
    )
    target, features = _identify_roles(df_numeric, mapping, loaded)
    assert target   == "Temperature"
    assert "Pression" in features