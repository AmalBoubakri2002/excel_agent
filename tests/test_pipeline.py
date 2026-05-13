import pytest
from unittest.mock import patch, MagicMock
from src.models import (
    PipelineState, UserQuery, FinalResponse,
    ExcelStructure, MappingResult, AnalysisType,
    AnalysisResult, AnalysisStatus,
)
from src.pipeline.nodes import should_continue


# ── Tests de l'état ──────────────────────────────────────────

def test_pipeline_state_initial():
    """L'état initial a bien les champs None par défaut."""
    query = UserQuery(
        raw_question    = "Test question ?",
        excel_file_path = "test.xlsx"
    )
    state = PipelineState(query=query)
    assert state.structure   is None
    assert state.mapping     is None
    assert state.loaded      is None
    assert state.plan        is None
    assert state.result      is None
    assert state.response    is None
    assert state.errors      == []
    assert state.current_step == "start"


def test_should_continue_sans_erreur():
    """Sans erreur → should_continue retourne 'continue'."""
    query = UserQuery(
        raw_question    = "Test ?",
        excel_file_path = "test.xlsx"
    )
    state = PipelineState(query=query, current_step="inspector_done")
    assert should_continue(state) == "continue"


def test_should_continue_avec_erreur():
    """Avec erreur → should_continue retourne 'stop'."""
    query = UserQuery(
        raw_question    = "Test ?",
        excel_file_path = "test.xlsx"
    )
    state = PipelineState(
        query        = query,
        current_step = "error",
        errors       = ["Fichier introuvable"]
    )
    assert should_continue(state) == "stop"


def test_should_continue_erreur_dans_liste():
    """Même si current_step est ok, une erreur dans la liste → stop."""
    query = UserQuery(
        raw_question    = "Test ?",
        excel_file_path = "test.xlsx"
    )
    state = PipelineState(
        query        = query,
        current_step = "inspector_done",
        errors       = ["Quelque chose a planté"]
    )
    assert should_continue(state) == "stop"


# ── Test d'intégration du pipeline complet ───────────────────

def test_run_pipeline_integration():
    """Test d'intégration complet sur le vrai fichier de test."""
    from src.pipeline.runner import run_pipeline

    response = run_pipeline(
        excel_file_path = "data/samples/mesures_centrale.xlsx",
        question        = "Quelle est la moyenne de la température ?",
        thread_id       = "test_integration",
    )

    assert isinstance(response, FinalResponse)
    assert len(response.answer) > 10
    assert response.data_summary != ""