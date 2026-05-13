def build_interpreter_prompt(
    question: str,
    excel_structure_summary: str
) -> str:
    """
    Construit le prompt pour l'Agent 2 – Interpréteur.

    Le prompt est conçu pour :
    - Donner au LLM le contexte exact du fichier Excel
    - Lui demander une réponse JSON strictement formatée
    - Éviter les hallucinations (colonnes inventées)
    - Forcer un raisonnement étape par étape
    """

    return f"""Tu es un expert en analyse de données. Tu reçois une question
d'un utilisateur et la structure d'un fichier Excel.
Ton rôle est d'identifier précisément quelles feuilles et colonnes
sont nécessaires pour répondre à la question.

STRUCTURE DU FICHIER EXCEL :
{excel_structure_summary}

QUESTION DE L'UTILISATEUR :
{question}

TYPES D'ANALYSES DISPONIBLES :
- statistical   : moyenne, médiane, écart-type, distribution
- correlation   : lien entre deux colonnes numériques
- regression    : prédire une valeur numérique (target + features)
- classification: prédire une catégorie (target + features)
- clustering    : regrouper les lignes automatiquement
- aggregation   : groupby, count, sum, filtre par catégorie

RÔLES POSSIBLES POUR LES COLONNES :
- target    : colonne à prédire ou analyser principalement
- feature   : colonne utilisée comme variable explicative
- group_by  : colonne de regroupement (ex: par Zone, par Statut)
- filter    : colonne sur laquelle appliquer un filtre
- date      : colonne temporelle pour tendances

RÈGLES IMPORTANTES :
1. N'utilise QUE les colonnes listées dans la structure ci-dessus
2. Choisis la feuille qui contient les données demandées :
   si la question parle d'incidents → utilise la feuille 'Incidents'
   si la question parle de mesures/capteurs → utilise 'Mesures'
3. Si la question est ambiguë, mets needs_clarification à true
4. Le champ confidence doit refléter ta certitude (0.0 à 1.0)
5. interpreted_question doit reformuler clairement l'intention

Réponds UNIQUEMENT avec un objet JSON valide, sans texte avant ou après,
sans balises markdown, sans explication. Exactement ce format :

{{
  "interpreted_question": "reformulation claire de la question",
  "analysis_type": "un des types listés ci-dessus",
  "selected_columns": [
    {{
      "sheet_name": "nom exact de la feuille",
      "column_name": "nom exact de la colonne",
      "role": "target|feature|group_by|filter|date",
      "reason": "pourquoi cette colonne est pertinente"
    }}
  ],
  "filters": [
    {{
      "column_name": "nom de la colonne",
      "operator": "==|>|<|>=|<=|contains",
      "value": "valeur du filtre"
    }}
  ],
  "confidence": 0.95,
  "needs_clarification": false,
  "clarification_question": ""
}}"""


def build_structure_summary(structure) -> str:
    """
    Convertit un ExcelStructure en texte lisible pour le LLM.

    On ne passe PAS l'objet Pydantic brut au LLM — on construit
    un résumé textuel compact et clair pour économiser les tokens.
    """

    lines = []

    lines.append(f"Fichier : {structure.file_name}")
    lines.append(
        f"Feuilles pertinentes : "
        f"{len(structure.relevant_sheets)}\n"
    )

    for sheet in structure.relevant_sheets:

        lines.append(
            f"FEUILLE : '{sheet.name}' "
            f"({sheet.n_rows} lignes)"
        )

        useful_cols = [
            c for c in sheet.columns
            if not c.should_ignore
        ]

        for col in useful_cols:

            stats_str = ""

            if col.mean is not None:
                stats_str = (
                    f" | moy={col.mean}, "
                    f"min={col.min_val}, "
                    f"max={col.max_val}"
                )

            lines.append(
                f"  - {col.name} "
                f"[{col.column_type.value}]"
                f"{stats_str}"
            )

        lines.append("")

    return "\n".join(lines)