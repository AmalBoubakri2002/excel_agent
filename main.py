import sys
from src.pipeline.runner import run_pipeline


def afficher_reponse(response, question: str):
    """Affichage formaté de la réponse finale."""
    print(f"\n{'╔' + '═'*58 + '╗'}")
    print(f"  RÉPONSE FINALE")
    print(f"{'╚' + '═'*58 + '╝'}")
    print(f"\n  Question : {question}")
    print(f"\n  Réponse  : {response.answer}")
    print(f"\n  Données  : {response.data_summary}")

    if response.key_metrics:
        print(f"\n  Métriques clés :")
        for k, v in response.key_metrics.items():
            print(f"    • {k} : {v}")

    if response.suggestions:
        print(f"\n  Suggestions :")
        for s in response.suggestions:
            print(f"    → {s}")

    if response.warnings:
        print(f"\n  Avertissements : {len(response.warnings)}")


# Questions de démonstration
DEMOS = [
    {
        "file"    : "data/samples/mesures_centrale.xlsx",
        "question": "Quelle est la moyenne de la température ?",
    },
    {
        "file"    : "data/samples/mesures_centrale.xlsx",
        "question": "Y a-t-il une corrélation entre la pression et le débit ?",
    },
    {
        "file"    : "data/samples/mesures_centrale.xlsx",
        "question": "Peux-tu faire un clustering des mesures ?",
    },
]


if __name__ == "__main__":
    # Mode interactif si argument fourni
    if len(sys.argv) == 3:
        file     = sys.argv[1]
        question = sys.argv[2]
        response = run_pipeline(file, question)
        afficher_reponse(response, question)

    # Mode démo sinon
    else:
        print("Mode démonstration — 3 questions sur le fichier de test\n")
        for demo in DEMOS:
            response = run_pipeline(
                excel_file_path = demo["file"],
                question        = demo["question"],
                thread_id       = demo["question"][:20],
            )
            afficher_reponse(response, demo["question"])
            print("\n" + "─" * 60)