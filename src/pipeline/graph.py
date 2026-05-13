from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.models import PipelineState
from src.pipeline.nodes import (
    node_inspector,
    node_interpreter,
    node_loader,
    node_transformer,
    node_analyst,
    node_synthesizer,
    should_continue,
)


def build_pipeline() -> StateGraph:
    """
    Construit et compile le graphe LangGraph du pipeline.

    Structure du graphe :
    START → inspector → interpreter → loader →
            transformer → analyst → synthesizer → END

    Chaque transition vérifie s'il y a eu une erreur.
    Si oui → END immédiat avec message d'erreur.
    """

    # ── Création du graphe avec notre état ────────────────────
    graph = StateGraph(PipelineState)

    # ── Ajout des nœuds ──────────────────────────────────────
    graph.add_node("inspector",   node_inspector)
    graph.add_node("interpreter", node_interpreter)
    graph.add_node("loader",      node_loader)
    graph.add_node("transformer", node_transformer)
    graph.add_node("analyst",     node_analyst)
    graph.add_node("synthesizer", node_synthesizer)

    # ── Point d'entrée ────────────────────────────────────────
    graph.set_entry_point("inspector")

    # ── Transitions conditionnelles ───────────────────────────
    # Après chaque nœud : si erreur → END, sinon → nœud suivant
    graph.add_conditional_edges(
        "inspector",
        should_continue,
        {"continue": "interpreter", "stop": END}
    )
    graph.add_conditional_edges(
        "interpreter",
        should_continue,
        {"continue": "loader", "stop": END}
    )
    graph.add_conditional_edges(
        "loader",
        should_continue,
        {"continue": "transformer", "stop": END}
    )
    graph.add_conditional_edges(
        "transformer",
        should_continue,
        {"continue": "analyst", "stop": END}
    )
    graph.add_conditional_edges(
        "analyst",
        should_continue,
        {"continue": "synthesizer", "stop": END}
    )

    # Dernier nœud → toujours END
    graph.add_edge("synthesizer", END)

    # ── Compilation ───────────────────────────────────────────
    # MemorySaver garde l'état en mémoire (utile pour debug)
    memory   = MemorySaver()
    compiled = graph.compile(checkpointer=memory)

    return compiled