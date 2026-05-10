import pytest
from unittest.mock import patch, MagicMock

from src.agents.interpreter import _extract_json, _validate_columns_exist, _parse_llm_response
from src.models import ExcelStructure, SheetInfo, ColumnInfo, ColumnType, AnalysisType


# ── Fixture : une structure Excel simple pour les tests ──────

@pytest.fixture
def simple_structure():
    """Structure Excel minimale pour les tests (sans appel LLM)."""
    col_temp = ColumnInfo(
        name="Temperature", column_type=ColumnType.NUMERIC,
        index=0, null_ratio=0.0, unique_count=100,
        mean=65.0, std=10.0, min_val=30.0, max_val=100.0
    )
    col_pression = ColumnInfo(
        name="Pression", column_type=ColumnType.NUMERIC,
        index=1, null_ratio=0.0, unique_count=80
    )
    col_statut = ColumnInfo(
        name="Statut", column_type=ColumnType.TEXT,
        index=2, null_ratio=0.0, unique_count=3
    )
    sheet = SheetInfo(
        name="Mesures", n_rows=200, n_cols=3,
        columns=[col_temp, col_pression, col_statut],
        is_relevant=True
    )
    return ExcelStructure(
        file_path="test.xlsx", file_name="test.xlsx",
        sheets=[sheet]
    )


# ── Tests de _extract_json ───────────────────────────────────

def test_extract_json_direct():
    """JSON propre parsé directement."""
    raw = '{"analysis_type": "statistical", "confidence": 0.9}'
    result = _extract_json(raw)
    assert result["analysis_type"] == "statistical"


def test_extract_json_avec_texte_parasite():
    """Le JSON est extrait même avec du texte autour."""
    raw = 'Voici ma réponse : {"analysis_type": "statistical"} fin.'
    result = _extract_json(raw)
    assert result["analysis_type"] == "statistical"


def test_extract_json_invalide():
    """Une réponse sans JSON lève une ValueError."""
    with pytest.raises(ValueError):
        _extract_json("Ceci n'est pas du JSON du tout.")


# ── Tests de _validate_columns_exist ────────────────────────

def test_validate_colonnes_valides(simple_structure):
    """Les colonnes qui existent passent la validation."""
    cols = [{"sheet_name": "Mesures", "column_name": "Temperature",
             "role": "target", "reason": "test"}]
    valid, warnings = _validate_columns_exist(cols, simple_structure)
    assert len(valid) == 1
    assert len(warnings) == 0


def test_validate_colonne_hallucinee(simple_structure):
    """Une colonne inventée est filtrée avec un warning."""
    cols = [{"sheet_name": "Mesures", "column_name": "ColonneInventee",
             "role": "target", "reason": "test"}]
    valid, warnings = _validate_columns_exist(cols, simple_structure)
    assert len(valid) == 0
    assert len(warnings) == 1


def test_validate_colonne_casse_insensible(simple_structure):
    """'temperature' doit matcher 'Temperature'."""
    cols = [{"sheet_name": "Mesures", "column_name": "temperature",
             "role": "target", "reason": "test"}]
    valid, warnings = _validate_columns_exist(cols, simple_structure)
    assert len(valid) == 1
    assert valid[0]["column_name"] == "Temperature"


def test_validate_feuille_hallucinee(simple_structure):
    """Une feuille inventée est ignorée."""
    cols = [{"sheet_name": "FeuilleInventee", "column_name": "Temperature",
             "role": "target", "reason": "test"}]
    valid, warnings = _validate_columns_exist(cols, simple_structure)
    assert len(valid) == 0


# ── Test de _parse_llm_response ──────────────────────────────

def test_parse_llm_response_complet(simple_structure):
    """Parse une réponse LLM complète et produit un MappingResult valide."""
    fake_response = '''{
        "interpreted_question": "Calculer la moyenne de Temperature",
        "analysis_type": "statistical",
        "selected_columns": [
            {
                "sheet_name": "Mesures",
                "column_name": "Temperature",
                "role": "target",
                "reason": "Colonne à analyser"
            }
        ],
        "filters": [],
        "confidence": 0.95,
        "needs_clarification": false,
        "clarification_question": ""
    }'''

    mapping = _parse_llm_response(fake_response, simple_structure)

    assert mapping.analysis_type == AnalysisType.STATISTICAL
    assert mapping.confidence == 0.95
    assert len(mapping.selected_columns) == 1
    assert mapping.selected_columns[0].column_name == "Temperature"
    assert not mapping.needs_clarification