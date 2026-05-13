# 🤖 Excel Agent

Un pipeline multi-agents intelligent pour analyser des fichiers Excel en langage naturel.  
Posez une question en français sur vos données — l'agent s'occupe du reste.

---

##  Table des matières

- [Présentation](#présentation)
- [Architecture](#architecture)
- [Agents & Pipeline](#agents--pipeline)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Structure du projet](#structure-du-projet)
- [Modèles de données](#modèles-de-données)
- [Outils d'analyse](#outils-danalyse)
- [Dépendances](#dépendances)
- [Tests](#tests)
- [Exemples de questions](#exemples-de-questions)

---

## Présentation

**Excel Agent** est un système multi-agents orchestré par [LangGraph](https://github.com/langchain-ai/langgraph) qui permet d'analyser des fichiers Excel (`.xlsx`, `.xls`, `.xlsm`) simplement en posant une question en langage naturel.

Le pipeline enchaîne automatiquement 6 agents spécialisés :

1. **Inspection** du fichier Excel (structure, types de colonnes, qualité)
2. **Interprétation** de la question via un LLM local (Ollama / Qwen 2.5)
3. **Chargement** intelligent des données pertinentes
4. **Transformation** (nettoyage, encodage, normalisation)
5. **Analyse** statistique ou ML (corrélation, régression, clustering…)
6. **Synthèse** de la réponse en langage naturel

### Fonctionnalités clés

-  Analyse en **langage naturel** (français)
-  Détection automatique du **type d'analyse** à effectuer
-  Support de **6 types d'analyses** : statistique, corrélation, régression, classification, clustering, agrégation
-  **Protection anti-hallucination** : validation des colonnes sélectionnées par le LLM
-  **Fallback déterministe** si le LLM est indisponible
-  Gestion robuste des **données manquantes** et des en-têtes décalés
-  **LLM 100% local** via Ollama (aucune donnée envoyée dans le cloud)

---

## Architecture

```
Question + Fichier Excel
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│                    PIPELINE LANGGRAPH                   │
│                                                         │
│  [1] Inspector ──► [2] Interpreter ──► [3] Loader       │
│                                             │           │
│  [6] Synthesizer ◄── [5] Analyst ◄── [4] Transformer   │
│                                                         │
└─────────────────────────────────────────────────────────┘
         │
         ▼
  Réponse en langage naturel + métriques + suggestions
```

Le pipeline utilise un `StateGraph` LangGraph avec transitions conditionnelles :  
en cas d'erreur à n'importe quelle étape, le pipeline s'arrête immédiatement et retourne un message d'erreur clair.

---

## Agents & Pipeline

### Agent 1 — Inspector (`src/agents/inspector.py`)

Inspecte le fichier Excel avant toute analyse.

**Responsabilités :**
- Lister toutes les feuilles (onglets)
- Détecter la ligne d'en-tête réelle (gère les titres au-dessus des données)
- Typer chaque colonne : `NUMERIC`, `TEXT`, `DATE`, `BOOLEAN`, `MIXED`, `EMPTY`, `IDENTIFIER`
- Calculer les statistiques de base (moyenne, min, max, ratio de nulls)
- Identifier les colonnes à ignorer (vides, colonnes `Unnamed`, >90% de nulls)
- Marquer les feuilles pertinentes vs. feuilles de sommaire/configuration

**Entrée :** chemin du fichier Excel  
**Sortie :** `ExcelStructure` (objet Pydantic)

---

### Agent 2 — Interpreter (`src/agents/interpreter.py`)

Interprète la question utilisateur en s'appuyant sur un LLM local.

**Responsabilités :**
- Construire un prompt structuré avec le résumé de la structure Excel
- Interroger Ollama (`qwen2.5:0.5b`, `temperature=0`, format JSON)
- Parser et valider la réponse JSON du LLM
- **Anti-hallucination** : vérifier que chaque colonne sélectionnée existe réellement, avec correction par correspondance insensible à la casse
- Déterminer le type d'analyse, les colonnes pertinentes, les filtres et le niveau de confiance

**Entrée :** question + `ExcelStructure`  
**Sortie :** `MappingResult`

---

### Agent 3 — Loader (`src/agents/loader.py`)

Charge les données pertinentes depuis le fichier Excel.

**Responsabilités :**
- Identifier la feuille principale à charger (via le mapping ou fallback)
- Charger uniquement les colonnes sélectionnées par l'Interpreter
- Appliquer les filtres (`==`, `!=`, `>`, `<`, `>=`, `<=`, `contains`) avec rollback automatique si un filtre vide tout le DataFrame
- Nettoyer le DataFrame (suppression des lignes entièrement vides)
- Signaler les colonnes avec un fort taux de valeurs manquantes

**Entrée :** `MappingResult` + `ExcelStructure`  
**Sortie :** `LoadedData`

---

### Agent 4 — Transformer (`src/agents/transformer.py`)

Prépare les données pour l'analyse.

**Responsabilités :**
- Détecter et recharger les colonnes manquantes (correctif pour les petits LLM)
- Remplir les valeurs manquantes : médiane pour les numériques, mode pour les catégories
- Encoder les colonnes catégorielles (LabelEncoder) pour les analyses ML
- Normaliser les numériques (z-score) pour régression / clustering / classification
- Identifier la colonne cible (target) et les features

**Entrée :** `LoadedData` + `MappingResult` + `ExcelStructure`  
**Sortie :** `TransformPlan`

---

### Agent 5 — Analyst (`src/agents/analyst.py`)

Exécute l'analyse appropriée selon le type détecté.

| Type d'analyse | Outil utilisé |
|---|---|
| `statistical` | Statistiques descriptives + test de normalité Shapiro-Wilk |
| `correlation` | Corrélation de Pearson + p-values |
| `regression` | Régression linéaire (scikit-learn) |
| `clustering` | K-Means avec détection automatique du k optimal (silhouette) |
| `aggregation` | GroupBy, distributions, comptages |
| `classification` | Fallback vers agrégation |

**Garde-fou** : si le DataFrame ne contient pas assez de colonnes numériques pour une analyse multi-variables (corrélation, régression, clustering), l'agent recharge automatiquement toutes les colonnes numériques exploitables de la feuille.

**Entrée :** `TransformPlan` + `MappingResult` + `ExcelStructure`  
**Sortie :** `AnalysisResult`

---

### Agent 6 — Synthesizer (`src/agents/synthesizer.py`)

Génère la réponse finale en langage naturel.

**Responsabilités :**
- Formater les métriques brutes de façon lisible selon le type d'analyse
- Interroger le LLM pour produire une réponse claire et contextualisée
- Proposer des suggestions d'analyses complémentaires
- **Fallback déterministe** : si Ollama est indisponible ou retourne une réponse vide, génère une réponse structurée à partir des métriques brutes

**Entrée :** `AnalysisResult` + `MappingResult` + `LoadedData` + question originale  
**Sortie :** `FinalResponse`

---

## Installation

### Prérequis

- Python 3.11+
- [Ollama](https://ollama.ai/) installé et lancé localement
- Le modèle `qwen2.5:0.5b` disponible dans Ollama

```bash
# Installer Ollama (Linux/Mac)
curl -fsSL https://ollama.ai/install.sh | sh

# Télécharger le modèle
ollama pull qwen2.5:0.5b
```

### Installation du projet

```bash
# Cloner le dépôt
git clone https://github.com/AmalBoubakri2002/excel_agent.git
cd excel_agent

# Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou : .venv\Scripts\activate  # Windows

# Installer les dépendances
pip install -r requirements.txt
```
---

## Utilisation

### Mode interactif (ligne de commande)

```bash
python main.py "data/samples/mesures_centrale.xlsx" "Quelle est la moyenne de la température ?"
```

### Mode démonstration

```bash
python main.py
```

Lance 3 questions de démonstration sur le fichier de test `data/samples/mesures_centrale.xlsx` :
- Moyenne de la température
- Corrélation entre la pression et le débit
- Clustering des mesures

### Utilisation depuis Python

```python
from src.pipeline.runner import run_pipeline

response = run_pipeline(
    excel_file_path="chemin/vers/mon_fichier.xlsx",
    question="Quelle colonne a la plus forte corrélation avec les ventes ?",
    language="fr",          # optionnel (défaut: "fr")
    thread_id="ma_session", # optionnel, pour le checkpointing LangGraph
)

print(response.answer)         # Réponse en langage naturel
print(response.data_summary)   # Résumé des données analysées
print(response.key_metrics)    # Métriques clés (dict)
print(response.suggestions)    # Suggestions d'analyses complémentaires
print(response.warnings)       # Avertissements éventuels
```

### Scripts de démonstration

Chaque agent peut être testé indépendamment :

```bash
python demo_inspector.py    # Tester l'inspection d'un fichier
python demo_interpreter.py  # Tester l'interprétation d'une question
python demo_loader.py       # Tester le chargement des données
python demo_transformer.py  # Tester les transformations
python demo_analyst.py      # Tester une analyse
python demo_pipeline.py     # Tester le pipeline complet
```

---

## Structure du projet

```
excel_agent/
├── main.py                        # Point d'entrée principal
├── requirements.txt               # Dépendances Python
├── test_setup.py                  # Vérification de l'installation
│
├── src/
│   ├── agents/                    # Les 6 agents spécialisés
│   │   ├── inspector.py           # Agent 1 : inspection du fichier Excel
│   │   ├── interpreter.py         # Agent 2 : interprétation via LLM
│   │   ├── loader.py              # Agent 3 : chargement des données
│   │   ├── transformer.py         # Agent 4 : transformation & nettoyage
│   │   ├── analyst.py             # Agent 5 : analyse statistique/ML
│   │   └── synthesizer.py         # Agent 6 : synthèse en langage naturel
│   │
│   ├── models/                    # Modèles Pydantic (contrats entre agents)
│   │   ├── query.py               # UserQuery, AnalysisType
│   │   ├── excel_structure.py     # ExcelStructure, SheetInfo, ColumnInfo
│   │   ├── mapping.py             # MappingResult, ColumnSelection, FilterCondition
│   │   ├── loaded_data.py         # LoadedData
│   │   ├── transform_plan.py      # TransformPlan, ColumnTransform
│   │   ├── analysis.py            # AnalysisResult, FinalResponse
│   │   └── pipeline_state.py      # PipelineState (état global LangGraph)
│   │
│   ├── pipeline/                  # Orchestration LangGraph
│   │   ├── graph.py               # Construction du StateGraph
│   │   ├── nodes.py               # Nœuds LangGraph (wrappeurs des agents)
│   │   └── runner.py              # run_pipeline() — point d'entrée unique
│   │
│   ├── tools/                     # Outils d'analyse (appelés par l'Analyst)
│   │   ├── statistical.py         # Statistiques descriptives
│   │   ├── correlation.py         # Corrélation de Pearson
│   │   ├── regression.py          # Régression linéaire
│   │   ├── clustering.py          # K-Means clustering
│   │   └── aggregation.py         # GroupBy & distributions
│   │
│   └── utils/
│       └── prompts.py             # Constructeurs de prompts LLM
│
├── tests/                         # Tests unitaires (pytest)
│   ├── test_inspector.py
│   ├── test_interpreter.py
│   ├── test_loader.py
│   ├── test_transformer.py
│   ├── test_analyst.py
│   ├── test_synthesizer.py
│   ├── test_pipeline.py
│   └── test_models.py
│
└── demo_*.py                      # Scripts de démonstration par agent
```

---

## Modèles de données

Tous les échanges entre agents utilisent des modèles **Pydantic** pour garantir la cohérence et la validation des données.

### `UserQuery`
Entrée utilisateur : question brute + chemin du fichier Excel.

### `ExcelStructure`
Résultat de l'Inspector : description complète du fichier (feuilles, colonnes, statistiques, anomalies).

### `MappingResult`
Résultat de l'Interpreter : type d'analyse, colonnes sélectionnées avec leurs rôles, filtres, niveau de confiance.

### `LoadedData`
Résultat du Loader : DataFrame pandas prêt à l'emploi + métadonnées (feuille source, filtres appliqués, null counts).

### `TransformPlan`
Résultat du Transformer : DataFrame transformé + liste des transformations appliquées + colonnes target/features.

### `AnalysisResult`
Résultat de l'Analyst : métriques brutes, tableau de résultats, statut (`success` / `partial` / `failed`).

### `FinalResponse`
Résultat final : réponse en langage naturel, métriques clés, suggestions, avertissements.

### `PipelineState`
État global partagé par tous les nœuds LangGraph, contenant toutes les sorties intermédiaires.

---

## Outils d'analyse

### `statistical.py` — Statistiques descriptives
Pour chaque colonne numérique :
- Count, moyenne, médiane, écart-type, min, max, Q25, Q75
- Asymétrie (skewness), aplatissement (kurtosis)
- Test de normalité Shapiro-Wilk (p-value + résultat booléen)

### `correlation.py` — Corrélation de Pearson
- Matrice de corrélation complète
- P-values pour chaque paire
- Interprétation textuelle automatique (très forte / forte / modérée / faible / aucune)
- Filtrage des corrélations significatives (p < 0.05)

### `regression.py` — Régression linéaire
- Split train/test 80/20
- R², RMSE, MAE
- Coefficients par feature
- Interprétation du R² (% de variance expliquée)

### `clustering.py` — K-Means
- Normalisation automatique (StandardScaler)
- Détection automatique du k optimal via score de silhouette (k=2 à 6)
- Statistiques par cluster (taille, moyennes par colonne)
- Interprétation de la qualité du clustering

### `aggregation.py` — Agrégation
- Distribution des colonnes catégorielles (value_counts)
- GroupBy + agrégats (count, mean, sum) sur les colonnes numériques

---

## Dépendances

```
# LLM & Orchestration
langchain==0.3.25
langchain-ollama==0.3.3
langgraph==0.4.3

# Data & ML
pandas==2.2.3
numpy>=1.26.4
scikit-learn==1.6.1
openpyxl==3.1.5
xlrd==2.0.1

# Validation
pydantic==2.11.4

# Tests & Qualité
pytest==8.3.5
black==25.1.0
ruff==0.11.9

# Utilitaires
python-dotenv==1.1.0
rich==14.0.0
```

---

## Tests

```bash
# Lancer tous les tests
pytest tests/ -v

# Un fichier de tests spécifique
pytest tests/test_inspector.py -v

# Avec rapport de couverture
pytest tests/ --cov=src --cov-report=term-missing
```

Les tests couvrent chaque agent individuellement ainsi que le pipeline complet :

| Fichier | Couverture |
|---|---|
| `test_inspector.py` | Détection de type, en-têtes, feuilles vides |
| `test_interpreter.py` | Parsing JSON, anti-hallucination, types d'analyse |
| `test_loader.py` | Sélection colonnes, filtres, rollback |
| `test_transformer.py` | Imputation, encodage, normalisation |
| `test_analyst.py` | Routage vers les bons outils |
| `test_synthesizer.py` | Fallback déterministe, formatage métriques |
| `test_pipeline.py` | Pipeline bout en bout |
| `test_models.py` | Validation Pydantic |

---

## Exemples de questions

```python
# Statistiques descriptives
"Quelle est la moyenne de la température ?"
"Donne-moi les statistiques de la colonne Pression"
"Quelle est la distribution des valeurs de débit ?"

# Corrélation
"Y a-t-il une corrélation entre la pression et le débit ?"
"Quelles colonnes sont les plus corrélées entre elles ?"

# Régression
"Peux-tu prédire le débit en fonction de la pression et de la température ?"

# Clustering
"Peux-tu faire un clustering des mesures ?"
"Identifie des groupes de comportements similaires"

# Agrégation
"Combien d'incidents par zone ?"
"Répartition des incidents par statut"
```

---
