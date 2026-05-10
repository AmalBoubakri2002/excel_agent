# tests/test_inspector.py

import pytest
import pandas as pd
from src.agents.inspector import (
    run_inspector,
    _detect_column_type,
    _detect_header_row,
    _should_ignore_column,
)
from src.models import ColumnType


# ── Tests des fonctions utilitaires ──────────────────────────

def test_detect_type_numeric():
    s = pd.Series([1.0, 2.5, 3.1, None], name="Temperature")
    assert _detect_column_type(s) == ColumnType.NUMERIC


def test_detect_type_text():
    s = pd.Series(["Nominal", "Alerte", "Critique"], name="Statut")
    assert _detect_column_type(s) == ColumnType.TEXT


def test_detect_type_empty():
    s = pd.Series([None, None, None], name="Colonne_Vide")
    assert _detect_column_type(s) == ColumnType.EMPTY


def test_detect_type_identifier():
    s = pd.Series([1.0, 2.0, 3.0], name="Capteur_ID")
    assert _detect_column_type(s) == ColumnType.IDENTIFIER


def test_detect_header_row_titre_en_ligne_0():
    """Si la ligne 0 est un titre, l'en-tête doit être détecté en ligne 1."""
    df = pd.DataFrame([
        ["RAPPORT INCIDENTS", None, None, None],  # ligne 0 = titre
        ["ID", "Gravite", "Duree_h", "Zone"],      # ligne 1 = en-tête
        ["INC-001", 2, 4.5, "Zone_A"],
    ])
    assert _detect_header_row(df) == 1


def test_should_ignore_empty_column():
    s = pd.Series([None, None], name="Vide")
    assert _should_ignore_column(s, ColumnType.EMPTY, 1.0) is True


def test_should_ignore_unnamed_column():
    s = pd.Series([1, 2, 3], name="Unnamed: 3")
    assert _should_ignore_column(s, ColumnType.NUMERIC, 0.0) is True


# ── Tests de l'agent complet ──────────────────────────────────

def test_inspector_fichier_inexistant():
    with pytest.raises(FileNotFoundError):
        run_inspector("fichier_qui_nexiste_pas.xlsx")


def test_inspector_format_invalide(tmp_path):
    f = tmp_path / "test.csv"
    f.write_text("a,b,c")
    with pytest.raises(ValueError):
        run_inspector(str(f))


def test_inspector_sur_fichier_sample():
    """Test d'intégration sur le vrai fichier de test."""
    structure = run_inspector("data/samples/mesures_centrale.xlsx")

    # Le fichier a 3 feuilles
    assert len(structure.sheets) == 3

    # 2 feuilles pertinentes (Mesures + Incidents)
    assert len(structure.relevant_sheets) == 2

    # La feuille Mesures a bien une colonne ignorée (Colonne_Vide)
    mesures = next(s for s in structure.sheets if s.name == "Mesures")
    ignored = [c for c in mesures.columns if c.should_ignore]
    assert len(ignored) >= 1

    # La feuille Sommaire est non pertinente
    sommaire = next(s for s in structure.sheets if s.name == "Sommaire")
    assert sommaire.is_relevant is False