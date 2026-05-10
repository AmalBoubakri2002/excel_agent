from pydantic import BaseModel, Field
from .query import AnalysisType


class ColumnSelection(BaseModel):
    """
    Sélection d'une colonne spécifique pour l'analyse.
    """
    sheet_name: str  = Field(..., description="Nom de la feuille")
    column_name: str = Field(..., description="Nom de la colonne")

    reason: str = Field(
        ...,
        description="Justification de la sélection par l'agent"
    )

    role: str = Field(
        ...,
        description="Ex: 'target', 'feature', 'filter', 'group_by'"
    )


class FilterCondition(BaseModel):
    """
    Représente un filtre à appliquer sur les données.
    Ex : "Ne garder que les lignes où Statut == 'Actif'"
    """
    column_name: str = Field(..., description="Colonne sur laquelle filtrer")
    operator: str    = Field(..., description="Ex: '==', '>', '<', 'contains'")
    value: str       = Field(..., description="Valeur de comparaison")


class MappingResult(BaseModel):
    """
    Résultat de l'Agent 2 – Interpréteur.
    Fait le lien entre la question utilisateur et la structure Excel.
    """

    interpreted_question: str = Field(
        ...,
        description="Reformulation claire de ce que l'utilisateur veut"
    )

    analysis_type: AnalysisType = Field(
        ...,
        description="Type d'analyse à effectuer"
    )

    selected_columns: list[ColumnSelection] = Field(
        default_factory=list,
        description="Colonnes identifiées comme pertinentes"
    )
    filters: list[FilterCondition] = Field(
        default_factory=list,
        description="Conditions de filtrage des données"
    )

    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confiance de l'agent dans son interprétation"
    )

    needs_clarification: bool = Field(
        default=False,
        description="True si la requête est ambiguë"
    )

    clarification_question: str = Field(
        default="",
        description="Question à poser à l'utilisateur si ambiguïté"
    )