import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

from src.models import (
    ExcelStructure, MappingResult, LoadedData, AnalysisType,
    TransformPlan, ColumnTransform, TransformType, ColumnType,
)
from src.agents.loader import run_loader, _load_raw_dataframe, _select_columns


# ─────────────────────────────────────────────────────────────
# FONCTIONS DE TRANSFORMATION
# ─────────────────────────────────────────────────────────────

def _fill_missing_values(
    df: pd.DataFrame,
    transformations: list[ColumnTransform],
    summary: list[str],
) -> pd.DataFrame:
    """
    Remplit les valeurs manquantes :
    - Colonnes numériques → médiane
    - Colonnes texte/catégorie → mode
    """
    for col in df.columns:
        n_nulls = df[col].isna().sum()
        if n_nulls == 0:
            continue

        if pd.api.types.is_numeric_dtype(df[col]):
            median_val = df[col].median()
            df[col]    = df[col].fillna(median_val)
            transformations.append(ColumnTransform(
                column_name    = col,
                transform_type = TransformType.FILL_MEDIAN,
                params         = {"median": round(float(median_val), 4)},
            ))
            summary.append(
                f"'{col}' : {n_nulls} valeur(s) manquante(s) "
                f"remplacée(s) par la médiane ({median_val:.2f})"
            )
        else:
            mode_val = df[col].mode()
            if len(mode_val) > 0:
                df[col] = df[col].fillna(mode_val[0])
                transformations.append(ColumnTransform(
                    column_name    = col,
                    transform_type = TransformType.FILL_MODE,
                    params         = {"mode": str(mode_val[0])},
                ))
                summary.append(
                    f"'{col}' : {n_nulls} valeur(s) manquante(s) "
                    f"remplacée(s) par le mode ('{mode_val[0]}')"
                )

    return df


def _encode_categoricals(
    df: pd.DataFrame,
    transformations: list[ColumnTransform],
    summary: list[str],
    analysis_type: AnalysisType,
) -> pd.DataFrame:
    """
    Encode les colonnes catégorielles (texte) en valeurs numériques.

    - Pour classification/regression → LabelEncoder
    - Pour statistical/correlation   → on laisse tel quel
      (les analyses stats ne nécessitent pas d'encodage)
    """
    needs_encoding = analysis_type in {
        AnalysisType.REGRESSION,
        AnalysisType.CLASSIFICATION,
        AnalysisType.CLUSTERING,
    }

    if not needs_encoding:
        return df

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]):
            le        = LabelEncoder()
            encoded   = le.fit_transform(df[col].astype(str))
            classes   = list(le.classes_)
            df[col]   = encoded

            transformations.append(ColumnTransform(
                column_name    = col,
                transform_type = TransformType.ENCODE_LABEL,
                params         = {"classes": classes},
            ))
            summary.append(
                f"'{col}' encodé : {classes} → 0..{len(classes)-1}"
            )

    return df


def _normalize_numerics(
    df: pd.DataFrame,
    transformations: list[ColumnTransform],
    summary: list[str],
    analysis_type: AnalysisType,
    target_col: str,
) -> pd.DataFrame:
    """
    Normalise les colonnes numériques (z-score) pour les analyses ML.
    On ne normalise PAS la colonne target (pour garder les vraies valeurs).
    On ne normalise PAS pour les analyses statistiques
    (la moyenne d'une colonne normalisée n'a pas de sens métier).
    """
    needs_normalization = analysis_type in {
        AnalysisType.REGRESSION,
        AnalysisType.CLASSIFICATION,
        AnalysisType.CLUSTERING,
    }

    if not needs_normalization:
        return df

    for col in df.columns:
        # Ne pas normaliser la target ni les colonnes non numériques
        if col == target_col:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue

        mean_val = float(df[col].mean())
        std_val  = float(df[col].std())

        if std_val == 0:
            # Colonne constante → inutile de normaliser
            summary.append(f"'{col}' non normalisée (std=0, colonne constante)")
            continue

        df[col] = (df[col] - mean_val) / std_val

        transformations.append(ColumnTransform(
            column_name    = col,
            transform_type = TransformType.NORMALIZE,
            params         = {"mean": round(mean_val, 4),
                              "std" : round(std_val, 4)},
        ))
        summary.append(
            f"'{col}' normalisée (z-score) : "
            f"moy={mean_val:.2f}, std={std_val:.2f}"
        )

    return df


def _enrich_with_missing_columns(
    loaded: LoadedData,
    mapping: MappingResult,
    structure: ExcelStructure,
    warnings: list[str],
) -> pd.DataFrame:
    """
    Détecte et recharge les colonnes manquantes pour l'analyse.

    Problème typique avec le modèle 0.5b : pour une corrélation,
    il ne sélectionne parfois qu'UNE colonne au lieu de DEUX.
    Cette fonction détecte ce cas et recharge les colonnes manquantes
    directement depuis le fichier Excel.

    Ex: question "corrélation Pression/Debit" mais seul Debit est chargé
    → on recharge Pression depuis la feuille Mesures
    """
    df = loaded.dataframe.copy()

    # Colonnes demandées par l'agent mais absentes du DataFrame
    expected_cols = {
        col.column_name
        for col in mapping.selected_columns
        if col.sheet_name == loaded.sheet_name
    }
    present_cols  = set(df.columns)
    missing_cols  = expected_cols - present_cols

    if not missing_cols:
        return df  # rien à faire

    warnings.append(
        f"ℹ️  Colonnes manquantes détectées, rechargement : {missing_cols}"
    )

    # Trouver les infos de la feuille
    sheet_info = next(
        (s for s in structure.sheets if s.name == loaded.sheet_name),
        None
    )
    if sheet_info is None:
        return df

    # Recharger le DataFrame complet de la feuille
    df_full = _load_raw_dataframe(
        structure.file_path,
        loaded.sheet_name,
        sheet_info.header_row_index,
    )

    # Ajouter les colonnes manquantes
    for col in missing_cols:
        if col in df_full.columns:
            # Aligner les index avant de joindre
            df[col] = df_full[col].values[:len(df)]
            warnings.append(f"   ✅ Colonne '{col}' rechargée")
        else:
            warnings.append(f"   ⚠️  Colonne '{col}' introuvable dans la feuille")

    return df


def _identify_roles(
    df: pd.DataFrame,
    mapping: MappingResult,
    loaded: LoadedData,
) -> tuple[str, list[str]]:
    """
    Identifie la colonne target et les colonnes features.

    Priorité :
    1. Rôles définis dans le MappingResult
    2. Heuristique : la première colonne numérique est la target
    """
    target_col      = ""
    feature_columns = []

    # Chercher le rôle 'target' dans le mapping
    for col_sel in mapping.selected_columns:
        if "target" in col_sel.role and col_sel.column_name in df.columns:
            target_col = col_sel.column_name
            break

    # Toutes les autres colonnes numériques = features
    for col in df.columns:
        if col == target_col:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            feature_columns.append(col)

    # Si pas de target défini → heuristique
    if not target_col and feature_columns:
        target_col      = feature_columns[0]
        feature_columns = feature_columns[1:]

    return target_col, feature_columns


# ─────────────────────────────────────────────────────────────
# AGENT PRINCIPAL
# ─────────────────────────────────────────────────────────────

def run_transformer(
    loaded: LoadedData,
    mapping: MappingResult,
    structure: ExcelStructure,
) -> TransformPlan:
    """
    Point d'entrée de l'Agent 4 – Transformateur.

    Reçoit les données chargées (Agent 3) et le mapping (Agent 2).
    Retourne un TransformPlan avec le DataFrame prêt pour l'analyse.
    """
    print(f"\n⚙️  Transformation des données...")
    print(f"   Type d'analyse : {mapping.analysis_type.value}")

    transformations : list[ColumnTransform] = []
    summary         : list[str]             = []
    warnings        : list[str]             = []

    # ── 1. Enrichir avec les colonnes manquantes ─────────────
    df = _enrich_with_missing_columns(loaded, mapping, structure, warnings)

    # ── 2. Remplir les valeurs manquantes ────────────────────
    df = _fill_missing_values(df, transformations, summary)

    # ── 3. Identifier target et features ────────────────────
    target_col, feature_columns = _identify_roles(df, mapping, loaded)

    # ── 4. Encoder les catégories (si ML) ───────────────────
    df = _encode_categoricals(
        df, transformations, summary, mapping.analysis_type
    )

    # ── 5. Normaliser les numériques (si ML) ─────────────────
    df = _normalize_numerics(
        df, transformations, summary,
        mapping.analysis_type, target_col
    )

    # ── Affichage résumé ─────────────────────────────────────
    print(f"   ✅ Transformations appliquées : {len(transformations)}")
    for s in summary:
        print(f"      • {s}")
    for w in warnings:
        print(f"      {w}")

    if target_col:
        print(f"   🎯 Target   : {target_col}")
    if feature_columns:
        print(f"   📐 Features : {feature_columns}")
    print(f"   📊 DataFrame final : "
          f"{df.shape[0]} lignes × {df.shape[1]} colonnes")

    return TransformPlan(
        dataframe       = df,
        transformations = transformations,
        feature_columns = feature_columns,
        target_column   = target_col,
        summary         = summary,
        warnings        = warnings,
    )