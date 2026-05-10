# demo_interpreter.py
# Teste l'Agent 1 + Agent 2 ensemble sur plusieurs questions

from src.agents.inspector   import run_inspector
from src.agents.interpreter import run_interpreter

FILE = "data/samples/mesures_centrale.xlsx"

# Questions de test variées
QUESTIONS = [
    "Quelle est la moyenne de la température ?",
    "Y a-t-il une corrélation entre la pression et le débit ?",
    "Combien d'incidents critiques ont eu lieu en Zone_A ?",
    "Peux-tu prédire la température à partir de la pression et du débit ?",
]

if __name__ == "__main__":
    # Étape 1 : inspection (une seule fois pour toutes les questions)
    print("=" * 60)
    print("ÉTAPE 1 – Inspection du fichier Excel")
    print("=" * 60)
    structure = run_inspector(FILE)

    # Étape 2 : interprétation de chaque question
    print("\n" + "=" * 60)
    print("ÉTAPE 2 – Interprétation des questions")
    print("=" * 60)

    for i, question in enumerate(QUESTIONS, 1):
        print(f"\n{'─'*60}")
        print(f"Question {i} : {question}")
        print(f"{'─'*60}")
        try:
            mapping = run_interpreter(question, structure)
        except Exception as e:
            print(f"   ❌ Erreur : {e}")
            