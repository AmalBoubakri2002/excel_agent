import pytest
import pandas as pd
from src.agents.loader import (
    _select_columns,
    _apply_filter,
    _clean_dataframe,
    _resolve_sheet_for_mapping,
    run_loader,
)
from src.models import (
    ExcelStructure, SheetInfo, ColumnInfo, ColumnType,
    MappingResult, ColumnSelection, FilterCondition, AnalysisType,
)


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        "Temperature": [65.0, 70.0, None, 55.0, 80.0],
        "Pression":    [1.5,  1.8,  1.2,  None, 2.1],
        "Statut":      ["Nominal", "Alerte", "Nominal", "Critique", "Nominal"],
    })


@pytest.fixture
def simple_structure():
    cols = [
        ColumnInfo(name="Temperature", column_type=ColumnType.NUMERIC,
                   index=0, null_ratio=0.0, unique_count=5),
        ColumnInfo(name="Pression", column_type=ColumnType.NUMERIC,
                   index=1, null_ratio=0.0, unique_count=5),
        ColumnInfo(name="Statut", column_type=ColumnType.TEXT,
                   index=2, null_ratio=0.0, unique_count=3),
    ]
    sheet = SheetInfo(name="Mesures", n_rows=5, n_cols=3,
                      columns=cols, is_relevant=True)
    return ExcelStructure(
        file_path="data/samples/mesures_centrale.xlsx",
        file_name="mesures_centrale.xlsx",
        sheets=[sheet]
    )


@pytest.fixture
def simple_mapping():
    return MappingResult(
        interpreted_question="Moyenne de Temperature",
        analysis_type=AnalysisType.STATISTICAL,
        selected_columns=[
            ColumnSelection(sheet_name="Mesures", column_name="Temperature",
                            role="target", reason="colonne principale")
        ],
        filters=[],
        confidence=0.9,
    )


# ── Tests _select_columns ────────────────────────────────────

def test_select_columns_existantes(sample_df):
    warnings = []
    result = _select_columns(sample_df, ["Temperature", "Statut"], warnings)
    assert list(result.columns) == ["Temperature", "Statut"]
    assert len(warnings) == 0


def test_select_columns_absente_warning(sample_df):
    warnings = []
    result = _select_columns(sample_df, ["Temperature", "Inventee"], warnings)
    assert "Temperature" in result.columns
    assert len(warnings) == 1


def test_select_columns_toutes_absentes(sample_df):
    with pytest.raises(ValueError):
        _select_columns(sample_df, ["Col1", "Col2"], [])


# ── Tests _apply_filter ──────────────────────────────────────

def test_filter_egalite_texte(sample_df):
    cond = FilterCondition(column_name="Statut",
                           operator="==", value="Nominal")
    result = _apply_filter(sample_df, cond, [])
    assert len(result) == 3
    assert all(result["Statut"] == "Nominal")


def test_filter_superieur(sample_df):
    cond = FilterCondition(column_name="Temperature",
                           operator=">", value="65")
    result = _apply_filter(sample_df, cond, [])
    assert all(result["Temperature"] > 65)


def test_filter_contains(sample_df):
    cond = FilterCondition(column_name="Statut",
                           operator="contains", value="ale")
    result = _apply_filter(sample_df, cond, [])
    # "Alerte" et "Nominale" contiennent "ale" -> Alerte + Nominal(e?)
    assert len(result) >= 1


def test_filter_colonne_absente(sample_df):
    warnings = []
    cond = FilterCondition(column_name="Inexistante",
                           operator="==", value="x")
    result = _apply_filter(sample_df, cond, warnings)
    assert len(result) == len(sample_df)  # df inchangé
    assert len(warnings) == 1


# ── Tests _clean_dataframe ───────────────────────────────────

def test_clean_supprime_lignes_vides():
    df = pd.DataFrame({
        "A": [1.0, None, 3.0],
        "B": [4.0, None, 6.0],
    })
    warnings = []
    result = _clean_dataframe(df, warnings)
    assert len(result) == 2
    assert len(warnings) == 1  # warning "1 ligne vide supprimée"


# ── Tests _resolve_sheet ─────────────────────────────────────

def test_resolve_sheet_plus_frequente():
    mapping = MappingResult(
        interpreted_question="test",
        analysis_type=AnalysisType.STATISTICAL,
        selected_columns=[
            ColumnSelection(sheet_name="Mesures", column_name="Temp",
                            role="target", reason=""),
            ColumnSelection(sheet_name="Mesures", column_name="Pression",
                            role="feature", reason=""),
            ColumnSelection(sheet_name="Incidents", column_name="Zone",
                            role="filter", reason=""),
        ],
        confidence=0.9,
    )
    sheet = _resolve_sheet_for_mapping(mapping)
    assert sheet == "Mesures"


# ── Test d'intégration ───────────────────────────────────────

def test_run_loader_integration(simple_mapping, simple_structure):
    loaded = run_loader(simple_mapping, simple_structure)
    assert loaded.sheet_name == "Mesures"
    assert "Temperature" in loaded.dataframe.columns
    assert loaded.n_rows > 0
    assert loaded.column_roles["Temperature"] == "target"