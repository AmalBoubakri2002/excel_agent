
from pydantic import BaseModel, Field
from enum import Enum


class AnalysisType(str, Enum):
    """
    Types d'analyses que l'utilisateur peut demander.
    On utilise un Enum pour éviter les fautes de frappe
    et avoir une liste exhaustive des possibilités.
    """
    STATISTICAL   = "statistical"   
    CORRELATION   = "correlation"   
    REGRESSION    = "regression"     
    CLASSIFICATION = "classification" 
    CLUSTERING    = "clustering"    
    AGGREGATION   = "aggregation"    
    UNKNOWN       = "unknown"     


class UserQuery(BaseModel):
    """
    Modèle représentant la requête brute de l'utilisateur.
    C'est le point d'entrée de tout le pipeline.
    """

    raw_question: str = Field(
        ...,  # "..." signifie "champ obligatoire" en Pydantic
        description="La question brute posée par l'utilisateur",
        min_length=3,  
    )

    excel_file_path: str = Field(
        ...,
        description="Chemin absolu ou relatif vers le fichier Excel"
    )

    analysis_type: AnalysisType = Field(
        default=AnalysisType.UNKNOWN,
        description="Type d'analyse détecté automatiquement"
    )

    language: str = Field(
        default="fr",
        description="Langue de la réponse (fr, en...)"
    )