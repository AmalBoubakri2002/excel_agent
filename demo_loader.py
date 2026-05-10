from src.agents.inspector   import run_inspector
from src.agents.interpreter import run_interpreter
from src.agents.loader      import run_loader

FILE = "data/samples/mesures_centrale.xlsx"

QUESTIONS = [
    "Quelle est la moyenne de la température ?",
    "Y a-t-il une corrélation entre la pression et le débit ?",
    "Combien d'incidents ont eu lieu en Zone_A ?",
]

if __name__ == "__main__":
    # Agent 1
    print("=" * 60)
    print("AGENT 1 – Inspection")
    print("=" * 60)
    structure = run_inspector(FILE)

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'='*60}")
        print(f"QUESTION {i} : {question}")
        print(f"{'='*60}")

        # Agent 2
        mapping = run_interpreter(question, structure)

        # Agent 3
        loaded = run_loader(mapping, structure)

        # Affichage du DataFrame résultant
        print(f"\n   Aperçu des données chargées :")
        print(f"   Colonnes : {list(loaded.dataframe.columns)}")
        print(f"   Rôles    : {loaded.column_roles}")
        print(loaded.dataframe.head(3).to_string(index=False))