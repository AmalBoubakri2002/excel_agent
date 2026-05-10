import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from src.models import (
    ExcelStructure, SheetInfo, ColumnInfo, ColumnType
)

# FONCTIONS UTILITAIRES (privées, préfixées par _)=

def _detect_column_type(series: pd.Series) -> ColumnType:
    """
    Détecte le type sémantique d'une colonne Pandas.

    On ne se fie pas uniquement au dtype Pandas car :
    - Une colonne d'IDs peut être numérique mais ne pas être analysable
    - Une colonne peut être 'object' mais contenir des dates en texte
    - Une colonne peut mélanger types (mixed)
    """
    # Colonne entièrement vide
    if series.isna().all():
        return ColumnType.EMPTY

    # Ignorer les valeurs nulles pour l'analyse
    non_null = series.dropna()

    # Type date
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATE

    # Type booléen
    if pd.api.types.is_bool_dtype(series):
        return ColumnType.BOOLEAN

    # Type numérique
    if pd.api.types.is_numeric_dtype(series):
        # Heuristique : si le nom contient "id" ou "code"
        # et que les valeurs sont des entiers → c'est un identifiant
        col_name_lower = str(series.name).lower()
        if any(kw in col_name_lower for kw in ["_id", "id_", "code", "num"]):
            if series.dropna().apply(float.is_integer).all():
                return ColumnType.IDENTIFIER
        return ColumnType.NUMERIC

    # Type texte : vérifier si les valeurs sont homogènes
    if pd.api.types.is_object_dtype(series):
        # Essayer de convertir en date
        try:
            pd.to_datetime(non_null, format="mixed", dayfirst=True)
            return ColumnType.DATE
        except (ValueError, TypeError):
            pass

        # Essayer de convertir en numérique
        try:
            pd.to_numeric(non_null)
            return ColumnType.NUMERIC
        except (ValueError, TypeError):
            pass

        # Vérifier si c'est un mélange de types
        types_presents = set(type(v).__name__ for v in non_null)
        if len(types_presents) > 1:
            return ColumnType.MIXED

        return ColumnType.TEXT

    return ColumnType.MIXED


def _compute_column_stats(
    series: pd.Series,
    col_type: ColumnType
) -> dict:
    """
    Calcule les statistiques descriptives d'une colonne.
    Retourne un dict avec mean, std, min_val, max_val (None si non applicable).
    """
    stats = {"mean": None, "std": None, "min_val": None, "max_val": None}

    if col_type == ColumnType.NUMERIC:
        numeric = pd.to_numeric(series, errors="coerce").dropna()
        if len(numeric) > 0:
            stats["mean"]    = round(float(numeric.mean()), 4)
            stats["std"]     = round(float(numeric.std()), 4)
            stats["min_val"] = round(float(numeric.min()), 4)
            stats["max_val"] = round(float(numeric.max()), 4)

    return stats


def _should_ignore_column(
    series: pd.Series,
    col_type: ColumnType,
    null_ratio: float
) -> bool:
    """
    Détermine si une colonne doit être ignorée dans l'analyse.
    Critères :
    - Colonne entièrement vide
    - Plus de 90% de valeurs manquantes
    - Colonne avec un seul nom générique ('Unnamed')
    """
    if col_type == ColumnType.EMPTY:
        return True
    if null_ratio > 0.90:
        return True
    if str(series.name).startswith("Unnamed"):
        return True
    return False


def _detect_header_row(df_raw: pd.DataFrame) -> int:
    """
    Détecte la ligne d'en-tête réelle dans un DataFrame brut.

    Stratégie : on cherche la première ligne où la majorité des cellules
    sont des chaînes de caractères (c'est souvent l'en-tête).
    Si la ligne 0 ressemble à un titre (1 seule valeur non nulle),
    l'en-tête est probablement à la ligne 1 ou 2.

    Retourne l'index de la ligne d'en-tête (0-based).
    """
    for i, row in df_raw.head(5).iterrows():
        non_null_values = row.dropna()
        if len(non_null_values) == 0:
            continue

        # Si une seule cellule non nulle sur toute la ligne → titre
        if len(non_null_values) == 1:
            continue

        # Si la majorité des cellules sont des strings → c'est l'en-tête
        str_count = sum(isinstance(v, str) for v in non_null_values)
        if str_count / len(non_null_values) > 0.6:
            return int(i)

    return 0  # Par défaut, ligne 0


def _analyze_sheet(
    file_path: str,
    sheet_name: str
) -> SheetInfo:
    """
    Analyse complète d'une feuille Excel.
    Retourne un SheetInfo peuplé.
    """
    warnings: list[str] = []

    # ── Lecture brute sans suppositions sur l'en-tête ────────
    df_raw = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=None,      # on ne suppose PAS que la ligne 0 est l'en-tête
        nrows=10,         # on lit seulement les 10 premières lignes pour détecter
        engine="openpyxl"
    )

    # ── Détection de la ligne d'en-tête ──────────────────────
    header_row = _detect_header_row(df_raw)

    if header_row > 0:
        warnings.append(
            f"En-tête détecté à la ligne {header_row} "
            f"(et non ligne 0) — {header_row} ligne(s) ignorée(s)"
        )

    # ── Lecture complète avec le bon en-tête ─────────────────
    df = pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=header_row,
        engine="openpyxl"
    )

    n_rows, n_cols = df.shape

    # Feuille vide ou quasi-vide → à ignorer
    if n_rows < 2 or n_cols < 2:
        return SheetInfo(
            name=sheet_name,
            n_rows=n_rows,
            n_cols=n_cols,
            header_row_index=header_row,
            is_relevant=False,
            structure_notes=["Feuille trop petite, considérée comme non pertinente"]
        )

    # ── Analyse colonne par colonne ───────────────────────────
    columns_info: list[ColumnInfo] = []

    for idx, col_name in enumerate(df.columns):
        series = df[col_name]

        # Calculer le ratio de valeurs manquantes
        null_ratio = float(series.isna().mean())

        # Détecter le type
        col_type = _detect_column_type(series)

        # Calculer les stats si numérique
        stats = _compute_column_stats(series, col_type)

        # Faut-il ignorer cette colonne ?
        ignore = _should_ignore_column(series, col_type, null_ratio)

        # Nombre de valeurs uniques (sur valeurs non nulles)
        unique_count = int(series.dropna().nunique())

        col_info = ColumnInfo(
            name=str(col_name),
            column_type=col_type,
            index=idx,
            null_ratio=round(null_ratio, 4),
            unique_count=unique_count,
            should_ignore=ignore,
            **stats  # mean, std, min_val, max_val
        )
        columns_info.append(col_info)

    # ── Détection de colonnes à types mixtes ──────────────────
    mixed_cols = [c.name for c in columns_info if c.column_type == ColumnType.MIXED]
    if mixed_cols:
        warnings.append(f"Colonnes à types mixtes détectées : {mixed_cols}")

    # ── La feuille est-elle pertinente ? ─────────────────────
    # On considère qu'une feuille avec moins de 3 colonnes utiles
    # est probablement un sommaire ou une feuille de configuration
    useful_cols = [c for c in columns_info if not c.should_ignore]
    is_relevant = len(useful_cols) >= 2 and n_rows >= 5

    if not is_relevant:
        warnings.append(
            "Feuille considérée comme non pertinente "
            "(trop peu de colonnes utiles ou de lignes)"
        )

    return SheetInfo(
        name=sheet_name,
        n_rows=n_rows,
        n_cols=n_cols,
        header_row_index=header_row,
        columns=columns_info,
        is_relevant=is_relevant,
        structure_notes=warnings
    )

# AGENT PRINCIPAL

def run_inspector(file_path: str) -> ExcelStructure:
    """
    Point d'entrée de l'Agent 1 – Inspecteur Excel.

    Reçoit un chemin de fichier Excel.
    Retourne un ExcelStructure complet et validé par Pydantic.

    Raises:
        FileNotFoundError : si le fichier n'existe pas
        ValueError        : si le fichier n'est pas un Excel valide
    """
    # ── Validation du fichier ─────────────────────────────────
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Fichier introuvable : {file_path}")

    if path.suffix.lower() not in {".xlsx", ".xls", ".xlsm"}:
        raise ValueError(
            f"Format non supporté : {path.suffix}. "
            "Utilisez .xlsx, .xls ou .xlsm"
        )

    global_warnings: list[str] = []

    # ── Lecture de la liste des feuilles ──────────────────────
    xl_file   = pd.ExcelFile(file_path, engine="openpyxl")
    sheet_names = xl_file.sheet_names

    print(f"📂 Fichier : {path.name}")
    print(f"📋 Feuilles détectées : {sheet_names}")

    # ── Analyse de chaque feuille ─────────────────────────────
    sheets_info: list[SheetInfo] = []
    total_cells = 0

    for sheet_name in sheet_names:
        print(f"\n🔍 Analyse de la feuille : '{sheet_name}'...")
        sheet_info = _analyze_sheet(file_path, sheet_name)
        sheets_info.append(sheet_info)

        total_cells += sheet_info.n_rows * sheet_info.n_cols

        # Affichage résumé
        status = "✅ pertinente" if sheet_info.is_relevant else "⚠️  ignorée"
        print(f"   {status} — {sheet_info.n_rows} lignes × {sheet_info.n_cols} colonnes")

        for note in sheet_info.structure_notes:
            print(f"   ⚠️  {note}")

    # ── Construction de l'objet final ─────────────────────────
    structure = ExcelStructure(
        file_path=str(path.absolute()),
        file_name=path.name,
        sheets=sheets_info,
        total_cells=total_cells,
        warnings=global_warnings
    )

    print(f"\n✅ Inspection terminée : {len(structure.relevant_sheets)} feuille(s) pertinente(s)")
    return structure