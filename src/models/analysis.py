from pydantic import BaseModel, Field
from typing import Any, Optional
from enum import Enum


class AnalysisStatus(str, Enum):
    SUCCESS = "success"
    PARTIAL = "partial"   
    FAILED  = "failed"


class AnalysisResult(BaseModel):
    """
    Résultat brut produit par l'Agent 5 – Analyste.
    Contient les chiffres, métriques, et résultats ML bruts.
    """
    status: AnalysisStatus = Field(..., description="Statut de l'analyse")

    analysis_type: str = Field(..., description="Type d'analyse effectuée")

    metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Métriques et résultats numériques"
    )

    result_table: Optional[list[dict[str, Any]]] = Field(
        default=None,
        description="Résultats sous forme de tableau (si applicable)"
    )

    warnings: list[str] = Field(
        default_factory=list,
        description="Ex: 'Trop peu de données pour être significatif'"
    )

    error_message: Optional[str] = Field(
        default=None,
        description="Message d'erreur si status == FAILED"
    )


class FinalResponse(BaseModel):
    """
    Réponse finale produite par l'Agent 6 – Synthétiseur.
    C'est ce que voit l'utilisateur.
    """

    answer: str = Field(
        ...,
        description="Réponse en langage naturel à la question posée"
    )

    data_summary: str = Field(
        ...,
        description="Ex: 'Analyse basée sur 1 245 lignes, colonne Température'"
    )

    key_metrics: dict[str, str] = Field(
        default_factory=dict,
        description="Ex: {'Moyenne': '42.3', 'Écart-type': '5.1'}"
    )

    warnings: list[str] = Field(default_factory=list)

    suggestions: list[str] = Field(
        default_factory=list,
        description="Ex: 'Vous pourriez aussi regarder la corrélation avec Y'"
    )