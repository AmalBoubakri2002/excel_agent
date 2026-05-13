import pandas as pd

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
)

from enum import Enum
from typing import Any


class TransformType(str, Enum):
    NONE           = "none"
    NORMALIZE      = "normalize"
    ENCODE_LABEL   = "encode_label"
    ENCODE_ONEHOT  = "encode_onehot"
    FILL_MEDIAN    = "fill_median"
    FILL_MODE      = "fill_mode"
    DROP_NULLS     = "drop_nulls"


class ColumnTransform(BaseModel):
    """Transformation appliquée à une colonne."""

    column_name: str = Field(
        ...,
        description="Colonne transformée"
    )

    transform_type: TransformType = Field(
        ...,
        description="Type de transformation"
    )

    params: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Paramètres de la transformation"
        )
    )

    new_column_name: str = Field(
        default="",
        description=(
            "Nom de la colonne résultante"
        )
    )


class TransformPlan(BaseModel):
    """
    Résultat de l'Agent 4 – Transformateur.

    Décrit toutes les transformations
    appliquées + le DataFrame final.
    """

    # Autoriser les DataFrame pandas
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    # ─────────────────────────────────────────────
    # DataFrame transformé
    # ─────────────────────────────────────────────
    dataframe: pd.DataFrame = Field(
        ...,
        description="Données transformées"
    )

    # ─────────────────────────────────────────────
    # Transformations effectuées
    # ─────────────────────────────────────────────
    transformations: list[ColumnTransform] = Field(
        default_factory=list
    )

    # ─────────────────────────────────────────────
    # Colonnes utilisées
    # ─────────────────────────────────────────────
    feature_columns: list[str] = Field(
        default_factory=list,
        description="Colonnes features (X)"
    )

    target_column: str = Field(
        default="",
        description="Colonne cible (y)"
    )

    # ─────────────────────────────────────────────
    # Résumé humain
    # ─────────────────────────────────────────────
    summary: list[str] = Field(
        default_factory=list,
        description=(
            "Description humaine "
            "des transformations"
        )
    )

    warnings: list[str] = Field(
        default_factory=list
    )

    # ─────────────────────────────────────────────
    # Correction LangGraph/msgpack
    # ─────────────────────────────────────────────
    def model_dump(self, *args, **kwargs):
        """
        Empêche msgpack de sérialiser
        directement le DataFrame.
        """

        data = super().model_dump(
            *args,
            **kwargs
        )

        # Remplacer le DataFrame
        # par un résumé léger
        data["dataframe"] = {
            "shape": list(self.dataframe.shape),
            "columns": self.dataframe.columns.tolist(),
        }

        return data