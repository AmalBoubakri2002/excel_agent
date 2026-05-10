from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional


class ColumnType(str, Enum):
    """
    Types de colonnes qu'on peut rencontrer dans un Excel métier.
    Plus précis que les types Pandas natifs.
    """
    NUMERIC     = "numeric"     
    TEXT        = "text"       
    DATE        = "date"        
    BOOLEAN     = "boolean"     
    MIXED       = "mixed"      
    EMPTY       = "empty"      
    IDENTIFIER  = "identifier"   


class ColumnInfo(BaseModel):
    """
    Informations détaillées sur UNE colonne d'une feuille Excel.
    """
    name: str = Field(..., description="Nom de la colonne (en-tête)")

    column_type: ColumnType = Field(
        ...,
        description="Type détecté automatiquement"
    )

    index: int = Field(..., ge=0) 

    null_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Ratio de valeurs manquantes (0=aucune, 1=tout vide)"
    )

    unique_count: int = Field(..., ge=0)

    mean: Optional[float] = Field(default=None, description="Moyenne (si numérique)")
    std: Optional[float]  = Field(default=None, description="Écart-type (si numérique)")
    min_val: Optional[float] = Field(default=None, description="Minimum (si numérique)")
    max_val: Optional[float] = Field(default=None, description="Maximum (si numérique)")

    should_ignore: bool = Field(
        default=False,
        description="True si la colonne est vide, inutile ou un artefact Excel"
    )


class SheetInfo(BaseModel):
    """
    Informations sur UNE feuille (onglet) du fichier Excel.
    """
    name: str = Field(..., description="Nom de l'onglet")

    n_rows: int = Field(..., ge=0, description="Nombre de lignes de données")
    n_cols: int = Field(..., ge=0, description="Nombre de colonnes")

    header_row_index: int = Field(
        default=0,
        ge=0,
        description="Index de la ligne d'en-tête (0 = première ligne)"
    )

    columns: list[ColumnInfo] = Field(
        default_factory=list,
        description="Informations sur chaque colonne"
    )

    is_relevant: bool = Field(
        default=True,
        description="False si la feuille est vide ou semble être un sommaire"
    )

    structure_notes: list[str] = Field(
        default_factory=list,
        description="Observations sur la structure de la feuille"
    )


class ExcelStructure(BaseModel):
    """
    Résultat complet de l'Agent 1 – Inspecteur.
    Décrit l'intégralité du fichier Excel analysé.
    """
    file_path: str = Field(..., description="Chemin du fichier analysé")
    file_name: str = Field(..., description="Nom du fichier uniquement")

    sheets: list[SheetInfo] = Field(
        default_factory=list,
        description="Liste de toutes les feuilles"
    )

    total_cells: int = Field(default=0, ge=0)

    warnings: list[str] = Field(
        default_factory=list,
        description="Problèmes détectés (cellules fusionnées, types mixtes...)"
    )


    @property
    def relevant_sheets(self) -> list[SheetInfo]:
        """Retourne uniquement les feuilles pertinentes."""
        return [s for s in self.sheets if s.is_relevant]

    @property
    def sheet_names(self) -> list[str]:
        """Retourne la liste des noms de feuilles."""
        return [s.name for s in self.sheets]