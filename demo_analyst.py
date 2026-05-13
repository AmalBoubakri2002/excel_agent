from src.agents.inspector   import run_inspector
from src.agents.interpreter import run_interpreter
from src.agents.loader      import run_loader
from src.agents.transformer import run_transformer
from src.agents.analyst     import run_analyst

FILE = "data/samples/mesures_centrale.xlsx"

QUESTIONS = [
    "Quelle est la moyenne de la température ?",
    "Y a-t-il une corrélation entre la pression et le débit ?",
    "Combien d'incidents ont eu lieu en Zone_A ?",
]

if __name__ == "__main__":
    structure = run_inspector(FILE)

    for question in QUESTIONS:
        print(f"\n{'='*60}")
        print(f"QUESTION : {question}")
        print(f"{'='*60}")

        mapping = run_interpreter(question, structure)
        loaded  = run_loader(mapping, structure)
        plan    = run_transformer(loaded, mapping, structure)
        result  = run_analyst(plan, mapping, structure)

        print(f"\n   📊 Statut    : {result.status.value}")
        print(f"   📈 Métriques :")
        for k, v in result.metrics.items():
            if isinstance(v, dict):
                print(f"      {k}:")
                for sk, sv in list(v.items())[:6]:
                    print(f"         {sk}: {sv}")
            else:
                print(f"      {k}: {v}")