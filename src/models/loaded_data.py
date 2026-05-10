import pandas as pd
from pydantic import BaseModel, Field
from typing import Any


class LoadedData(BaseModel):
    """
    Résultat de l'Agent 3 – Chargeur.
    Contient le DataFrame prêt pour l'analyse + toutes les métadonnées.

    On ne stocke PAS le DataFrame dans Pydantic directement
    (Pydantic ne sait pas valider un DataFrame).
    On utilise model_config pour autoriser les types arbitraires.
    """
    model_config = {"arbitrary_types_allowed": True}

    # Le DataFrame principal chargé et nettoyé
    dataframe: pd.DataFrame = Field(
        ...,
        description="Données chargées et filtrées, prêtes pour l'analyse"
    )

    # Métadonnées sur ce qui a été chargé
    sheet_name: str  = Field(..., description="Feuille source")
    n_rows: int      = Field(..., description="Nombre de lignes chargées")
    n_cols: int      = Field(..., description="Nombre de colonnes chargées")

    # Colonnes disponibles avec leurs rôles
    # Ex: {"Temperature": "target", "Pression": "feature"}
    column_roles: dict[str, str] = Field(
        default_factory=dict,
        description="Rôle de chaque colonne dans l'analyse"
    )

    # Filtres qui ont été appliqués
    applied_filters: list[str] = Field(
        default_factory=list,
        description="Description des filtres appliqués"
    )

    # Avertissements sur la qualité des données
    warnings: list[str] = Field(
        default_factory=list,
        description="Problèmes détectés lors du chargement"
    )

    # Statistiques post-chargement
    null_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Nombre de valeurs nulles par colonne"
    )