import pandas as pd
from pydantic import BaseModel, Field
from enum import Enum
from typing import Any


class TransformType(str, Enum):
    NONE           = "none"            # pas de transformation
    NORMALIZE      = "normalize"       # normalisation min-max ou zscore
    ENCODE_LABEL   = "encode_label"    # encodage ordinal (ex: Nominal=0, Alerte=1)
    ENCODE_ONEHOT  = "encode_onehot"   # encodage one-hot
    FILL_MEDIAN    = "fill_median"     # remplir nulls par la médiane
    FILL_MODE      = "fill_mode"       # remplir nulls par le mode
    DROP_NULLS     = "drop_nulls"      # supprimer lignes avec nulls


class ColumnTransform(BaseModel):
    """Transformation appliquée à une colonne."""
    column_name    : str           = Field(..., description="Colonne transformée")
    transform_type : TransformType = Field(..., description="Type de transformation")
    params         : dict[str, Any] = Field(
        default_factory=dict,
        description="Paramètres de la transformation (ex: mean, std pour zscore)"
    )
    new_column_name: str = Field(
        default="",
        description="Nom de la colonne résultante (vide = même nom)"
    )


class TransformPlan(BaseModel):
    """
    Résultat de l'Agent 4 – Transformateur.
    Décrit toutes les transformations appliquées + le DataFrame final.
    """
    model_config = {"arbitrary_types_allowed": True}

    # DataFrame transformé, prêt pour l'analyse
    dataframe      : pd.DataFrame = Field(..., description="Données transformées")

    # Liste des transformations effectuées
    transformations: list[ColumnTransform] = Field(default_factory=list)

    # Colonnes finales et leurs rôles
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Colonnes features (X)"
    )
    target_column  : str = Field(
        default="",
        description="Colonne cible (y) si applicable"
    )

    # Résumé lisible des opérations
    summary        : list[str] = Field(
        default_factory=list,
        description="Description humaine des transformations"
    )
    warnings       : list[str] = Field(default_factory=list)