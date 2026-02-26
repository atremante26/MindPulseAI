import sys
from typing import List, Optional, Dict
from pathlib import Path
from pydantic import BaseModel

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.recommender.resource_schema import Concern, CostTier, ResourceType, AgeGroup

class RecommendationRequest(BaseModel):
    concerns: List[Concern]
    cost_preference: CostTier
    age: int
    resource_type_preferences: Optional[List[ResourceType]] = None
    online_only: bool = False
    crisis_need: bool = False

class Resource(BaseModel):
    id: str
    name: str
    type: ResourceType
    description: str
    url: Optional[str] = None
    phone: Optional[str] = None
    concerns: List[Concern]
    cost_tier: CostTier
    age_groups: List[AgeGroup]
    crisis_resource: bool
    rating: Optional[float] = None
    online_only: bool

class MatchBreakdown(BaseModel):
    concerns: float
    cost: float
    age: float
    overall: float


class RecommendationExplanation(BaseModel):
    summary: str
    reasons: List[str]
    match_breakdown: MatchBreakdown


class SingleRecommendation(BaseModel):
    resource: Resource
    match_score: float
    explanation: RecommendationExplanation

class RecommendationResponse(BaseModel):
    recommendations: List[SingleRecommendation]