# tests/test_synthesizer.py

import pytest
import pandas as pd

from src.agents.synthesizer import (
    _format_statistical_metrics,
    _format_correlation_metrics,
    _build_fallback_response,
)
from src.models import (
    AnalysisResult, AnalysisStatus,
    FinalResponse, MappingResult, AnalysisType,
    ColumnSelection, LoadedData,
)


# ── Fixtures ─────────────────────────────────────────────────

@pytest.fixture
def loaded_data():
    return LoadedData(
        dataframe  = pd.DataFrame({"Temperature": [65.0, 70.0, 80.0]}),
        sheet_name = "Mesures",
        n_rows     = 3,
        n_cols     = 1,
    )


@pytest.fixture
def mapping():
    return MappingResult(
        interpreted_question = "Moyenne Temperature",
        analysis_type        = AnalysisType.STATISTICAL,
        selected_columns     = [
            ColumnSelection(sheet_name="Mesures",
                            column_name="Temperature",
                            role="target", reason="")
        ],
        confidence = 0.9,
    )


@pytest.fixture
def stat_result():
    return AnalysisResult(
        status        = AnalysisStatus.SUCCESS,
        analysis_type = "statistical",
        metrics       = {
            "Temperature": {
                "count": 194, "mean": 64.79, "median": 65.43,
                "std": 11.1, "min": 33.56, "max": 97.64,
                "is_normal": True, "p_normal": 0.23,
            }
        },
    )


@pytest.fixture
def corr_result():
    return AnalysisResult(
        status        = AnalysisStatus.SUCCESS,
        analysis_type = "correlation",
        metrics       = {
            "pairs": [
                {
                    "col1": "Pression", "col2": "Debit",
                    "pearson_r": -0.02, "p_value": 0.79,
                    "significant": False,
                    "interpretation": "pas de corrélation significative",
                    "n_samples": 191,
                }
            ],
            "columns_analyzed": ["Temperature", "Pression", "Debit"],
        },
    )


# ── Tests formatage ──────────────────────────────────────────

def test_format_statistical(stat_result):
    text = _format_statistical_metrics(stat_result.metrics)
    assert "Temperature" in text
    assert "64.79"       in text
    assert "Moyenne"     in text


def test_format_correlation(corr_result):
    text = _format_correlation_metrics(corr_result.metrics)
    assert "Pression"   in text
    assert "Debit"      in text
    assert "-0.02"      in text


# ── Tests fallback ───────────────────────────────────────────

def test_fallback_statistical(stat_result, loaded_data):
    response = _build_fallback_response(
        "Quelle est la moyenne ?", stat_result, loaded_data
    )
    assert isinstance(response, FinalResponse)
    assert "64.79" in response.answer
    assert len(response.key_metrics) > 0


def test_fallback_correlation(corr_result, loaded_data):
    response = _build_fallback_response(
        "Y a-t-il une corrélation ?", corr_result, loaded_data
    )
    assert isinstance(response, FinalResponse)
    assert "Pression" in response.answer or "Debit" in response.answer


def test_fallback_failed_result(loaded_data):
    """Un résultat FAILED produit une réponse d'erreur."""
    failed = AnalysisResult(
        status        = AnalysisStatus.FAILED,
        analysis_type = "statistical",
        metrics       = {},
        error_message = "Pas assez de données",
    )
    from src.agents.synthesizer import run_synthesizer
    mapping = MappingResult(
        interpreted_question="test",
        analysis_type=AnalysisType.STATISTICAL,
        confidence=0.5,
    )
    response = run_synthesizer(failed, mapping, loaded_data, "question test")
    assert "échoué" in response.answer.lower() or "erreur" in response.answer.lower()


def test_final_response_structure(stat_result, loaded_data):
    """La réponse finale a bien toutes les propriétés attendues."""
    response = _build_fallback_response("test", stat_result, loaded_data)
    assert hasattr(response, "answer")
    assert hasattr(response, "data_summary")
    assert hasattr(response, "key_metrics")
    assert hasattr(response, "suggestions")
    assert hasattr(response, "warnings")
    assert "Mesures" in response.data_summary