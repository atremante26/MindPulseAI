from pydantic import BaseModel
from typing import Dict, List

class ModelParams(BaseModel):
    """Model hyperparameters used for clustering."""
    min_cluster_size: int
    min_samples: int
    cluster_selection_epsilon: float

class ClusterInfo(BaseModel):
    """Information about a single cluster."""
    size: int
    percentage: float
    avg_age: float
    age_std: float
    gender_distribution: Dict[str, float]
    top_countries: Dict[str, int]
    in_treatment_pct: float
    family_history_pct: float
    remote_work_pct: float  
    work_interference_distribution: Dict[str, float] 
    top_work_interference: str
    benefits_distribution: Dict[str, float]
    mental_health_consequence_distribution: Dict[str, float]
    key_traits: List[str]

class ClustersResponse(BaseModel):
    """Complete clustering results response."""
    labels: List[int]
    silhouette_score: float
    cluster_persistence: List[float]
    n_clusters: int
    n_noise: int
    cluster_profiles: Dict[str, ClusterInfo]  
    timestamp: str
    model_params: ModelParams  
    generated_at: str