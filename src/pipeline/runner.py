import time
from src.models import UserQuery, PipelineState, FinalResponse
from src.pipeline.graph import build_pipeline


def run_pipeline(
    excel_file_path: str,
    question: str,
    language: str = "fr",
    thread_id: str = "default",
) -> FinalResponse:
    """
    Point d'entrée unique du pipeline complet.

    Usage :
        from src.pipeline.runner import run_pipeline
        response = run_pipeline(
            excel_file_path = "data/samples/mesures_centrale.xlsx",
            question        = "Quelle est la moyenne de la température ?",
        )
        print(response.answer)

    Args:
        excel_file_path : chemin vers le fichier Excel
        question        : question en langage naturel
        language        : langue de la réponse (défaut: "fr")
        thread_id       : identifiant de session (pour le checkpointing)

    Returns:
        FinalResponse avec la réponse, les métriques et les suggestions
    """
    start_total = time.time()

    print(f"\n{'='*50}")
    print(f"  PIPELINE EXCEL AGENT")
    print(f"{'='*50}")
    print(f"  Fichier   : {excel_file_path}")
    print(f"  Question  : {question}")
    print(f"{'='*50}")

    # ── Construire l'état initial ─────────────────────────────
    query = UserQuery(
        raw_question    = question,
        excel_file_path = excel_file_path,
        language        = language,
    )
    initial_state = PipelineState(query=query)

    # ── Compiler et lancer le graphe ─────────────────────────
    pipeline = build_pipeline()

    config = {"configurable": {"thread_id": thread_id}}

    final_state_dict = pipeline.invoke(
        initial_state,
        config=config,
    )

    # ── Extraire l'état final ─────────────────────────────────
    # LangGraph retourne un dict — on reconstitue le PipelineState
    final_state = PipelineState(**final_state_dict)

    total_time = round(time.time() - start_total, 2)

    # ── Affichage du résumé de performance ───────────────────
    print(f"\n{'='*50}")
    print(f"  PIPELINE TERMINÉ en {total_time}s")
    print(f"{'='*50}")
    for step, duration in final_state.step_durations.items():
        bar = "█" * int(duration * 5)
        print(f"  {step:<15} {duration:>5.1f}s  {bar}")

    if final_state.errors:
        print(f"\n  Erreurs détectées :")
        for err in final_state.errors:
            print(f"  • {err}")

    # ── Retourner la réponse finale ───────────────────────────
    if final_state.response:
        return final_state.response

    # Fallback si le pipeline a planté avant la synthèse
    return FinalResponse(
        answer       = f"Erreur pipeline : {'; '.join(final_state.errors)}",
        data_summary = "Pipeline interrompu",
        key_metrics  = {},
        warnings     = final_state.errors,
    )