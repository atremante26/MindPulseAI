import sys
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Schemas
from backend.schemas import RecommendationRequest, RecommendationResponse
from backend.services import get_recommendation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'recommendations_router.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define router
router = APIRouter()

@router.post("/recommend", response_model=RecommendationResponse)
def get_recommendations_endpoint(request: RecommendationRequest):
    """
    Post endpoint for user recommendations. 
    
    :param request: RecommendationRequest for user profile.
    :returns: RecommendationResponse with recommendations for user request.
    """
    try:
        return get_recommendation(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))