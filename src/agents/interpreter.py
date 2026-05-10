import json
import re
from typing import Optional

from langchain_ollama import OllamaLLM

from src.models import (
    ExcelStructure,
    MappingResult,
    ColumnSelection,
    FilterCondition,
    AnalysisType,
)
from src.utils.prompts import build_interpreter_prompt, build_structure_summary


# ─────────────────────────────────────────────────────────────
# CONFIGURATION DU LLM
# ─────────────────────────────────────────────────────────────

def _get_llm() -> OllamaLLM:
    """
    Instancie le LLM Ollama.
    temperature=0 : réponses déterministes (pas de créativité,
    on veut du JSON précis, pas de la poésie).
    """
    return OllamaLLM(
        model="qwen2.5:0.5b",
        temperature=0,        # 0 = déterministe, reproductible
        format="json",        # force Ollama à produire du JSON
    )


# ─────────────────────────────────────────────────────────────
# FONCTIONS DE PARSING ET VALIDATION
# ─────────────────────────────────────────────────────────────

def _extract_json(raw_response: str) -> dict:
    """
    Extrait et parse le JSON de la réponse du LLM.

    Même avec format="json", le LLM peut parfois ajouter
    du texte parasite. On nettoie défensivement.
    """
    # Tentative 1 : parse direct
    try:
        return json.loads(raw_response.strip())
    except json.JSONDecodeError:
        pass

    # Tentative 2 : extraire le premier bloc JSON avec regex
    pattern = r'\{.*\}'
    match = re.search(pattern, raw_response, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass

    # Échec total
    raise ValueError(
        f"Impossible de parser la réponse du LLM en JSON.\n"
        f"Réponse brute : {raw_response[:200]}..."
    )


def _validate_columns_exist(
    selected_columns: list[dict],
    structure: ExcelStructure
) -> tuple[list[dict], list[str]]:
    """
    Vérifie que les colonnes choisies par le LLM existent vraiment.

    Le LLM peut halluciner des noms de colonnes — cette fonction
    est notre garde-fou principal contre ce problème.

    Retourne :
    - La liste des colonnes valides (hallucinations supprimées)
    - La liste des warnings pour colonnes introuvables
    """
    valid   = []
    warnings = []

    # Construire un index rapide : {sheet_name: [col_names]}
    sheet_columns: dict[str, list[str]] = {}
    for sheet in structure.relevant_sheets:
        sheet_columns[sheet.name] = [
            c.name for c in sheet.columns if not c.should_ignore
        ]

    for col_data in selected_columns:
        sheet_name = col_data.get("sheet_name", "")
        col_name   = col_data.get("column_name", "")

        # La feuille existe-t-elle ?
        if sheet_name not in sheet_columns:
            warnings.append(
                f"⚠️  Feuille hallucninée ignorée : '{sheet_name}'"
            )
            continue

        # La colonne existe-t-elle dans cette feuille ?
        if col_name not in sheet_columns[sheet_name]:
            # Tentative de correspondance approximative
            # (ex: "temperature" → "Temperature")
            col_lower  = col_name.lower()
            candidates = [
                c for c in sheet_columns[sheet_name]
                if c.lower() == col_lower
            ]
            if candidates:
                col_data["column_name"] = candidates[0]
                warnings.append(
                    f"ℹ️  Colonne '{col_name}' corrigée en '{candidates[0]}'"
                )
                valid.append(col_data)
            else:
                warnings.append(
                    f"⚠️  Colonne hallucinée ignorée : "
                    f"'{col_name}' dans '{sheet_name}'"
                )
        else:
            valid.append(col_data)

    return valid, warnings


def _parse_llm_response(
    raw_response: str,
    structure: ExcelStructure
) -> MappingResult:
    """
    Transforme la réponse brute du LLM en MappingResult validé.

    Étapes :
    1. Extraire le JSON
    2. Valider les colonnes (anti-hallucination)
    3. Construire le MappingResult Pydantic
    """
    data = _extract_json(raw_response)

    # ── Validation anti-hallucination ────────────────────────
    raw_columns = data.get("selected_columns", [])
    valid_columns, col_warnings = _validate_columns_exist(
        raw_columns, structure
    )

    # ── Construction des ColumnSelection ─────────────────────
    column_selections = []
    for col in valid_columns:
        try:
            column_selections.append(ColumnSelection(
                sheet_name  = col["sheet_name"],
                column_name = col["column_name"],
                role        = col.get("role", "feature"),
                reason      = col.get("reason", "")
            ))
        except Exception:
            continue  # ignorer les colonnes malformées

    # ── Construction des FilterCondition ─────────────────────
    filter_conditions = []
    for f in data.get("filters", []):
        try:
            filter_conditions.append(FilterCondition(
                column_name = f["column_name"],
                operator    = f["operator"],
                value       = str(f["value"])
            ))
        except Exception:
            continue

    # ── Récupération du type d'analyse ───────────────────────
    analysis_type_str = data.get("analysis_type", "unknown")
    try:
        analysis_type = AnalysisType(analysis_type_str)
    except ValueError:
        analysis_type = AnalysisType.UNKNOWN

    # ── Niveau de confiance ───────────────────────────────────
    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(1.0, confidence))  # clamp entre 0 et 1

    # ── Résultat final ────────────────────────────────────────
    result = MappingResult(
        interpreted_question  = data.get("interpreted_question", ""),
        analysis_type         = analysis_type,
        selected_columns      = column_selections,
        filters               = filter_conditions,
        confidence            = confidence,
        needs_clarification   = bool(data.get("needs_clarification", False)),
        clarification_question= data.get("clarification_question", "")
    )

    # Afficher les warnings de validation
    for w in col_warnings:
        print(f"   {w}")

    return result


# ─────────────────────────────────────────────────────────────
# AGENT PRINCIPAL
# ─────────────────────────────────────────────────────────────

def run_interpreter(
    question: str,
    structure: ExcelStructure
) -> MappingResult:
    """
    Point d'entrée de l'Agent 2 – Interpréteur.

    Reçoit la question utilisateur et la structure Excel (Agent 1).
    Retourne un MappingResult validé décrivant quoi analyser.
    """
    print(f"\n🧠 Interprétation de la question...")
    print(f"   Question : '{question}'")

    # ── Construire le résumé de structure pour le LLM ────────
    structure_summary = build_structure_summary(structure)

    # ── Construire le prompt ──────────────────────────────────
    prompt = build_interpreter_prompt(question, structure_summary)

    # ── Appel au LLM ─────────────────────────────────────────
    llm = _get_llm()

    print("   ⏳ Appel à Ollama (llama3.2:3b)...")
    raw_response = llm.invoke(prompt)
    print("   ✅ Réponse reçue")

    # ── Parser et valider la réponse ─────────────────────────
    mapping = _parse_llm_response(raw_response, structure)

    # ── Affichage du résultat ─────────────────────────────────
    print(f"\n   📊 Type d'analyse : {mapping.analysis_type.value}")
    print(f"   🎯 Confiance      : {mapping.confidence:.0%}")
    print(f"   📋 Colonnes sélectionnées :")
    for col in mapping.selected_columns:
        print(f"      • [{col.role}] {col.sheet_name}.{col.column_name}")
    if mapping.filters:
        print(f"   🔎 Filtres :")
        for f in mapping.filters:
            print(f"      • {f.column_name} {f.operator} '{f.value}'")
    if mapping.needs_clarification:
        print(f"   ❓ Clarification nécessaire : {mapping.clarification_question}")

    return mapping