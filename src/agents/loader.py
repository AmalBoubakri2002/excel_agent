import pandas as pd
import numpy as np
from pathlib import Path

from yaml import warnings

from src.models import (
    ExcelStructure, SheetInfo,
    MappingResult, FilterCondition,
    LoadedData,
)

def _get_sheet_info(
    sheet_name: str,
    structure: ExcelStructure
) -> SheetInfo:
    """
    Récupère les infos d'une feuille depuis la structure.
    Lève une ValueError si la feuille n'existe pas.
    """
    for sheet in structure.sheets:
        if sheet.name == sheet_name:
            return sheet
    raise ValueError(
        f"Feuille '{sheet_name}' introuvable dans la structure. "
        f"Feuilles disponibles : {structure.sheet_names}"
    )


def _load_raw_dataframe(
    file_path: str,
    sheet_name: str,
    header_row: int
) -> pd.DataFrame:
    """
    Charge le DataFrame brut depuis Excel avec le bon en-tête.
    """
    return pd.read_excel(
        file_path,
        sheet_name=sheet_name,
        header=header_row,
        engine="openpyxl"
    )


def _select_columns(
    df: pd.DataFrame,
    column_names: list[str],
    warnings: list[str]
) -> pd.DataFrame:
    """
    Sélectionne uniquement les colonnes demandées.
    Ignore silencieusement les colonnes absentes (avec warning).
    """
    available = set(df.columns)
    to_keep   = []

    for col in column_names:
        if col in available:
            to_keep.append(col)
        else:
            warnings.append(
                f"⚠️  Colonne '{col}' absente du DataFrame — ignorée"
            )

    if not to_keep:
        raise ValueError(
            "Aucune colonne valide à charger après filtrage. "
            f"Colonnes demandées : {column_names}. "
            f"Colonnes disponibles : {list(available)}"
        )

    return df[to_keep]

def _apply_filter(
    df: pd.DataFrame,
    condition: FilterCondition,
    warnings: list[str]
) -> pd.DataFrame:
    """
    Applique UN filtre sur le DataFrame.
    Si le filtre vide complètement le DataFrame, il est annulé.
    """
    col = condition.column_name
    op  = condition.operator
    val = condition.value

    if col not in df.columns:
        warnings.append(f"⚠️  Filtre ignoré : colonne '{col}' absente")
        return df

    df_original = df.copy()  # sauvegarde pour rollback

    try:
        if op == "==":
            try:
                df_filtered = df[df[col] == float(val)]
            except (ValueError, TypeError):
                df_filtered = df[df[col].astype(str) == val]

        elif op == "!=":
            try:
                df_filtered = df[df[col] != float(val)]
            except (ValueError, TypeError):
                df_filtered = df[df[col].astype(str) != val]

        elif op == ">":
            df_filtered = df[pd.to_numeric(df[col], errors="coerce") > float(val)]

        elif op == "<":
            df_filtered = df[pd.to_numeric(df[col], errors="coerce") < float(val)]

        elif op == ">=":
            df_filtered = df[pd.to_numeric(df[col], errors="coerce") >= float(val)]

        elif op == "<=":
            df_filtered = df[pd.to_numeric(df[col], errors="coerce") <= float(val)]

        elif op == "contains":
            df_filtered = df[df[col].astype(str).str.contains(
                val, case=False, na=False
            )]

        else:
            warnings.append(f"⚠️  Opérateur inconnu '{op}' — filtre ignoré")
            return df

    except Exception as e:
        warnings.append(f"⚠️  Erreur filtre ({col} {op} {val}) : {e}")
        return df

    # ── Garde-fou : rollback si le filtre vide tout ──────────
    if len(df_filtered) == 0:
        warnings.append(
            f"⚠️  Filtre '{col} {op} {val}' viderait toutes les données "
            f"— filtre annulé automatiquement"
        )
        return df_original  # on retourne le df AVANT le filtre

    return df_filtered

def _clean_dataframe(
    df: pd.DataFrame,
    warnings: list[str]
) -> pd.DataFrame:
    """
    Nettoyage basique du DataFrame :
    - Supprime les lignes entièrement vides
    - Reset de l'index
    - Signale les colonnes avec beaucoup de nulls
    """
    n_before = len(df)

    # Supprimer lignes entièrement vides
    df = df.dropna(how="all")
    df = df.reset_index(drop=True)

    n_after = len(df)
    if n_before != n_after:
        warnings.append(
            f"ℹ️  {n_before - n_after} ligne(s) entièrement vide(s) supprimée(s)"
        )

    # Signaler les colonnes avec > 20% de nulls
    for col in df.columns:
        null_ratio = df[col].isna().mean()
        if null_ratio > 0.2:
            warnings.append(
                f"⚠️  Colonne '{col}' : {null_ratio:.0%} de valeurs manquantes"
            )

    return df

def _resolve_sheet_for_mapping(
    mapping: MappingResult,
    structure: ExcelStructure
) -> str:
    """
    Détermine la feuille principale à charger.
    Si aucune colonne sélectionnée → prend la première feuille pertinente.
    """
    if not mapping.selected_columns:
        # Fallback : première feuille pertinente
        if structure.relevant_sheets:
            fallback = structure.relevant_sheets[0].name
            return fallback
        raise ValueError("Aucune feuille pertinente dans la structure")

    # Compter les occurrences de chaque feuille
    sheet_counts: dict[str, int] = {}
    for col in mapping.selected_columns:
        sheet_counts[col.sheet_name] = (
            sheet_counts.get(col.sheet_name, 0) + 1
        )

    return max(sheet_counts, key=lambda k: sheet_counts[k])

# ─────────────────────────────────────────────────────────────
# AGENT PRINCIPAL
# ─────────────────────────────────────────────────────────────

def run_loader(
    mapping: MappingResult,
    structure: ExcelStructure,
) -> LoadedData:
    """
    Point d'entrée de l'Agent 3 – Chargeur.

    Reçoit le MappingResult (Agent 2) et la structure (Agent 1).
    Retourne un LoadedData avec le DataFrame prêt pour l'analyse.
    """
    warnings: list[str] = []

    print(f"\n📥 Chargement des données...")

    # ── 1. Déterminer la feuille principale ──────────────────
    sheet_name = _resolve_sheet_for_mapping(mapping, structure)
    sheet_info = _get_sheet_info(sheet_name, structure)

    print(f"   Feuille cible : '{sheet_name}' "
          f"(en-tête ligne {sheet_info.header_row_index})")

    # ── 2. Charger le DataFrame brut ─────────────────────────
    df = _load_raw_dataframe(
        structure.file_path,
        sheet_name,
        sheet_info.header_row_index
    )
    print(f"   Dimensions brutes : {df.shape[0]} lignes × {df.shape[1]} colonnes")

    # ── 3. Sélectionner les colonnes pertinentes ─────────────
    # Colonnes demandées pour CETTE feuille uniquement
    cols_for_sheet = [
        col.column_name
        for col in mapping.selected_columns
        if col.sheet_name == sheet_name
    ]

    # Construire le mapping colonne → rôle
    col_roles = {
        col.column_name: col.role
        for col in mapping.selected_columns
        if col.sheet_name == sheet_name
    }

    if cols_for_sheet:
            df = _select_columns(df, cols_for_sheet, warnings)
    else:
    # Fallback : charger toutes les colonnes utiles de la feuille
            warnings.append(
        "ℹ️  Aucune colonne spécifiée par l'agent — "
        "chargement de toutes les colonnes utiles"
    )
    useful_cols = [
        c.name for c in sheet_info.columns
        if not c.should_ignore
    ]
    if useful_cols:
        df = _select_columns(df, useful_cols, warnings)
    # ── 4. Appliquer les filtres ──────────────────────────────
    applied_filters: list[str] = []

    for condition in mapping.filters:
        # Ne filtrer que sur les colonnes présentes dans le df
        if condition.column_name in df.columns:
            n_before = len(df)
            df = _apply_filter(df, condition, warnings)
            n_after  = len(df)

            filter_desc = (
                f"{condition.column_name} "
                f"{condition.operator} "
                f"'{condition.value}'"
            )
            applied_filters.append(filter_desc)

            if n_after < n_before:
                print(f"   🔎 Filtre appliqué : {filter_desc} "
                      f"({n_before} → {n_after} lignes)")

    # ── 5. Nettoyer le DataFrame ─────────────────────────────
    df = _clean_dataframe(df, warnings)

    # ── 6. Calculer les null counts finaux ───────────────────
    null_counts = {
        col: int(df[col].isna().sum())
        for col in df.columns
    }

    # ── 7. Affichage résumé ───────────────────────────────────
    print(f"   ✅ Données prêtes : "
          f"{df.shape[0]} lignes × {df.shape[1]} colonnes")

    for w in warnings:
        print(f"   {w}")

    return LoadedData(
        dataframe      = df,
        sheet_name     = sheet_name,
        n_rows         = df.shape[0],
        n_cols         = df.shape[1],
        column_roles   = col_roles,
        applied_filters= applied_filters,
        warnings       = warnings,
        null_counts    = null_counts,
    )