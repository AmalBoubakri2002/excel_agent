import pytest
from pydantic import ValidationError
from src.models import (
    UserQuery, AnalysisType,
    ExcelStructure, SheetInfo, ColumnInfo, ColumnType,
    MappingResult, ColumnSelection,
    AnalysisResult, FinalResponse, AnalysisStatus,
)



def test_user_query_valide():
    """Une requête correcte doit être créée sans erreur."""
    q = UserQuery(
        raw_question="Quelle est la moyenne de la colonne Température ?",
        excel_file_path="data/samples/mesures.xlsx"
    )
    assert q.analysis_type == AnalysisType.UNKNOWN
    assert q.language == "fr"


def test_user_query_question_trop_courte():
    """Une question trop courte doit lever une erreur."""
    with pytest.raises(ValidationError):
        UserQuery(raw_question="ab", excel_file_path="data/test.xlsx")



def test_excel_structure_sheets_pertinentes():
    """La propriété relevant_sheets ne retourne que les feuilles utiles."""
    col = ColumnInfo(
        name="Temp", column_type=ColumnType.NUMERIC,
        index=0, null_ratio=0.0, unique_count=100
    )
    sheet_ok = SheetInfo(name="Données", n_rows=100, n_cols=3,
                         columns=[col], is_relevant=True)
    sheet_ko = SheetInfo(name="Sommaire", n_rows=5, n_cols=2,
                         columns=[], is_relevant=False)

    structure = ExcelStructure(
        file_path="data/test.xlsx",
        file_name="test.xlsx",
        sheets=[sheet_ok, sheet_ko]
    )

    assert len(structure.relevant_sheets) == 1
    assert structure.relevant_sheets[0].name == "Données"
    assert structure.sheet_names == ["Données", "Sommaire"]


def test_column_info_null_ratio_invalide():
    """Un null_ratio > 1.0 doit lever une erreur."""
    with pytest.raises(ValidationError):
        ColumnInfo(
            name="Col", column_type=ColumnType.NUMERIC,
            index=0, null_ratio=1.5,  # invalide !
            unique_count=10
        )


def test_mapping_confidence_invalide():
    """Une confiance > 1.0 doit lever une erreur."""
    with pytest.raises(ValidationError):
        MappingResult(
            interpreted_question="Moyenne de Temp",
            analysis_type=AnalysisType.STATISTICAL,
            confidence=1.5  # invalide !
        )



def test_final_response_complete():
    """Une réponse finale correctement formée."""
    response = FinalResponse(
        answer="La moyenne de Température est 42.3°C",
        data_summary="Basé sur 1000 lignes, colonne Température",
        key_metrics={"Moyenne": "42.3°C", "Écart-type": "5.1°C"},
        suggestions=["Regardez aussi la corrélation avec Pression"]
    )
    assert "42.3" in response.answer
    assert len(response.suggestions) == 1