# demo_pipeline.py
# Pipeline complet Agent 1 → 2 → 3 → 4 → 5 → 6

from src.agents.inspector   import run_inspector
from src.agents.interpreter import run_interpreter
from src.agents.loader      import run_loader
from src.agents.transformer import run_transformer
from src.agents.analyst     import run_analyst
from src.agents.synthesizer import run_synthesizer

FILE = "data/samples/mesures_centrale.xlsx"

QUESTIONS = [
    "Quelle est la moyenne de la température ?",
    "Y a-t-il une corrélation entre la pression et le débit ?",
    "Combien d'incidents ont eu lieu en Zone_A ?",
]

def afficher_reponse_finale(reponse, question):
    print(f"\n{'='*60}")
    print(f"  RÉPONSE FINALE")
    print(f"{'='*60}")
    print(f"  ❓ Question   : {question}")
    print(f"  💬 Réponse    : {reponse.answer}")
    print(f"  📊 Données    : {reponse.data_summary}")
    if reponse.key_metrics:
        print(f"  📈 Métriques clés :")
        for k, v in reponse.key_metrics.items():
            print(f"     • {k} : {v}")
    if reponse.suggestions:
        print(f"  💡 Suggestions :")
        for s in reponse.suggestions:
            print(f"     → {s}")
    if reponse.warnings:
        print(f"  ⚠️  Avertissements : {len(reponse.warnings)}")


if __name__ == "__main__":
    # Agent 1 — une seule fois
    structure = run_inspector(FILE)

    for question in QUESTIONS:
        print(f"\n{'#'*60}")
        print(f"  QUESTION : {question}")
        print(f"{'#'*60}")

        try:
            mapping  = run_interpreter(question, structure)
            loaded   = run_loader(mapping, structure)
            plan     = run_transformer(loaded, mapping, structure)
            result   = run_analyst(plan, mapping, structure)
            reponse  = run_synthesizer(result, mapping, loaded, question)

            afficher_reponse_finale(reponse, question)

        except Exception as e:
            print(f"\n❌ Erreur pipeline : {e}")
            import traceback
            traceback.print_exc()