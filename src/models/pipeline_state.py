from typing import Optional, Any
from pydantic import BaseModel, Field
import pandas as pd

from src.models.query          import UserQuery
from src.models.excel_structure import ExcelStructure
from src.models.mapping        import MappingResult
from src.models.loaded_data    import LoadedData
from src.models.transform_plan import TransformPlan
from src.models.analysis       import AnalysisResult, FinalResponse


class PipelineState(BaseModel):
    """
    État global du pipeline — transmis de nœud en nœud par LangGraph.
    Chaque agent lit les champs dont il a besoin et remplit les siens.
    Les champs Optional sont None jusqu'à ce que l'agent correspondant s'exécute.
    """
    model_config = {"arbitrary_types_allowed": True}

    # ── Entrées utilisateur ───────────────────────────────────
    query: UserQuery = Field(..., description="Question + chemin fichier")

    # ── Sorties de chaque agent ───────────────────────────────
    structure  : Optional[ExcelStructure] = None   # Agent 1
    mapping    : Optional[MappingResult]  = None   # Agent 2
    loaded     : Optional[LoadedData]     = None   # Agent 3
    plan       : Optional[TransformPlan]  = None   # Agent 4
    result     : Optional[AnalysisResult] = None   # Agent 5
    response   : Optional[FinalResponse]  = None   # Agent 6

    # ── Métadonnées du pipeline ───────────────────────────────
    current_step : str        = Field(default="start")
    errors       : list[str]  = Field(default_factory=list)
    step_durations: dict[str, float] = Field(
        default_factory=dict,
        description="Durée en secondes de chaque étape"
    )