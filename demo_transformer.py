from src.agents.inspector   import run_inspector
from src.agents.interpreter import run_interpreter
from src.agents.loader      import run_loader
from src.agents.transformer import run_transformer

FILE = "data/samples/mesures_centrale.xlsx"

QUESTIONS = [
    ("Quelle est la moyenne de la température ?",               "statistical"),
    ("Y a-t-il une corrélation entre la pression et le débit ?","correlation"),
    ("Peux-tu prédire la température à partir de la pression ?","regression"),
]

if __name__ == "__main__":
    # Agent 1 — une seule fois
    structure = run_inspector(FILE)

    for question, expected_type in QUESTIONS:
        print(f"\n{'='*60}")
        print(f"QUESTION : {question}")
        print(f"{'='*60}")

        mapping     = run_interpreter(question, structure)
        loaded      = run_loader(mapping, structure)
        plan        = run_transformer(loaded, mapping, structure)

        print(f"\n   Colonnes finales  : {list(plan.dataframe.columns)}")
        print(f"   Aperçu :")
        print(plan.dataframe.head(3).to_string(index=False))