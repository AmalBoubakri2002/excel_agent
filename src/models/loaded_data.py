import pandas as pd

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
)


class LoadedData(BaseModel):
    """
    Résultat de l'Agent 3 – Chargeur.

    Contient le DataFrame prêt pour l'analyse
    + toutes les métadonnées.
    """

    # Autoriser les types arbitraires (DataFrame)
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )

    # ─────────────────────────────────────────────
    # DataFrame principal
    # ─────────────────────────────────────────────
    dataframe: pd.DataFrame = Field(
        ...,
        description=(
            "Données chargées et filtrées, "
            "prêtes pour l'analyse"
        )
    )

    # ─────────────────────────────────────────────
    # Métadonnées
    # ─────────────────────────────────────────────
    sheet_name: str = Field(
        ...,
        description="Feuille source"
    )

    n_rows: int = Field(
        ...,
        description="Nombre de lignes chargées"
    )

    n_cols: int = Field(
        ...,
        description="Nombre de colonnes chargées"
    )

    # ─────────────────────────────────────────────
    # Rôles des colonnes
    # ─────────────────────────────────────────────
    column_roles: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "Rôle de chaque colonne "
            "dans l'analyse"
        )
    )

    # ─────────────────────────────────────────────
    # Filtres appliqués
    # ─────────────────────────────────────────────
    applied_filters: list[str] = Field(
        default_factory=list,
        description=(
            "Description des filtres appliqués"
        )
    )

    # ─────────────────────────────────────────────
    # Warnings
    # ─────────────────────────────────────────────
    warnings: list[str] = Field(
        default_factory=list,
        description=(
            "Problèmes détectés "
            "lors du chargement"
        )
    )

    # ─────────────────────────────────────────────
    # Valeurs nulles
    # ─────────────────────────────────────────────
    null_counts: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Nombre de valeurs nulles "
            "par colonne"
        )
    )

    # ─────────────────────────────────────────────
    #  LangGraph/msgpack
    # ─────────────────────────────────────────────
    def model_dump(self, *args, **kwargs):
        """
        Empêche la sérialisation directe
        du DataFrame par msgpack.
        """

        data = super().model_dump(
            *args,
            **kwargs
        )

        data["dataframe"] = {
            "shape": list(self.dataframe.shape),
            "columns": self.dataframe.columns.tolist(),
        }

        return data