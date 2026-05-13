import time
from src.models import PipelineState
from src.agents.inspector   import run_inspector
from src.agents.interpreter import run_interpreter
from src.agents.loader      import run_loader
from src.agents.transformer import run_transformer
from src.agents.analyst     import run_analyst
from src.agents.synthesizer import run_synthesizer


def _chrono(state: PipelineState, step: str, start: float) -> dict:
    """Enregistre la durée d'une étape dans l'état."""
    durations = dict(state.step_durations)
    durations[step] = round(time.time() - start, 2)
    return durations


# ── Nœud 1 ───────────────────────────────────────────────────

def node_inspector(state: PipelineState) -> dict:
    """
    Nœud Agent 1 : inspecte le fichier Excel.
    Retourne un dict partiel — LangGraph merge avec l'état existant.
    """
    start = time.time()
    print(f"\n{'─'*50}")
    print(f"[1/6] Inspection du fichier...")

    try:
        structure = run_inspector(state.query.excel_file_path)
        return {
            "structure"     : structure,
            "current_step"  : "inspector_done",
            "step_durations": _chrono(state, "inspector", start),
        }
    except Exception as e:
        return {
            "errors"      : state.errors + [f"Inspecteur : {e}"],
            "current_step": "error",
        }


# ── Nœud 2 ───────────────────────────────────────────────────

def node_interpreter(state: PipelineState) -> dict:
    """Nœud Agent 2 : interprète la question utilisateur."""
    start = time.time()
    print(f"\n[2/6] Interprétation de la question...")

    try:
        mapping = run_interpreter(
            state.query.raw_question,
            state.structure
        )
        return {
            "mapping"       : mapping,
            "current_step"  : "interpreter_done",
            "step_durations": _chrono(state, "interpreter", start),
        }
    except Exception as e:
        return {
            "errors"      : state.errors + [f"Interpréteur : {e}"],
            "current_step": "error",
        }


# ── Nœud 3 ───────────────────────────────────────────────────

def node_loader(state: PipelineState) -> dict:
    """Nœud Agent 3 : charge les données pertinentes."""
    start = time.time()
    print(f"\n[3/6] Chargement des données...")

    try:
        loaded = run_loader(state.mapping, state.structure)
        return {
            "loaded"        : loaded,
            "current_step"  : "loader_done",
            "step_durations": _chrono(state, "loader", start),
        }
    except Exception as e:
        return {
            "errors"      : state.errors + [f"Chargeur : {e}"],
            "current_step": "error",
        }


# ── Nœud 4 ───────────────────────────────────────────────────

def node_transformer(state: PipelineState) -> dict:
    """Nœud Agent 4 : transforme et prépare les données."""
    start = time.time()
    print(f"\n[4/6] Transformation des données...")

    try:
        plan = run_transformer(
            state.loaded,
            state.mapping,
            state.structure
        )
        return {
            "plan"          : plan,
            "current_step"  : "transformer_done",
            "step_durations": _chrono(state, "transformer", start),
        }
    except Exception as e:
        return {
            "errors"      : state.errors + [f"Transformateur : {e}"],
            "current_step": "error",
        }


# ── Nœud 5 ───────────────────────────────────────────────────

def node_analyst(state: PipelineState) -> dict:
    """Nœud Agent 5 : exécute l'analyse statistique ou ML."""
    start = time.time()
    print(f"\n[5/6] Analyse en cours...")

    try:
        result = run_analyst(
            state.plan,
            state.mapping,
            state.structure
        )
        return {
            "result"        : result,
            "current_step"  : "analyst_done",
            "step_durations": _chrono(state, "analyst", start),
        }
    except Exception as e:
        return {
            "errors"      : state.errors + [f"Analyste : {e}"],
            "current_step": "error",
        }


# ── Nœud 6 ───────────────────────────────────────────────────

def node_synthesizer(state: PipelineState) -> dict:
    """Nœud Agent 6 : synthétise la réponse finale."""
    start = time.time()
    print(f"\n[6/6] Synthèse de la réponse...")

    try:
        response = run_synthesizer(
            state.result,
            state.mapping,
            state.loaded,
            state.query.raw_question
        )
        return {
            "response"      : response,
            "current_step"  : "done",
            "step_durations": _chrono(state, "synthesizer", start),
        }
    except Exception as e:
        return {
            "errors"      : state.errors + [f"Synthétiseur : {e}"],
            "current_step": "error",
        }


# ── Condition de routage ──────────────────────────────────────

def should_continue(state: PipelineState) -> str:
    """
    Fonction de routage conditionnel.
    Si une erreur est détectée → on arrête le pipeline.
    Sinon → on continue vers l'étape suivante.
    """
    if state.current_step == "error" or state.errors:
        return "stop"
    return "continue"