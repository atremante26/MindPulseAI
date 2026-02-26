import sys
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Schemas
from backend.schemas import RecommendationRequest
from analysis.recommender import MentalHealthRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'recommendations_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def get_recommendation(request: RecommendationRequest):
    """
    Gets recommendations for user profile request.
    
    :param request: RecommendationRequest for user profile.
    :returns: Dict with recommendations for user request.
    """
    try:
        recommender = MentalHealthRecommender()
        user_profile = {
            "concerns": [c.value for c in request.concerns],
            "cost_preference": request.cost_preference.value,
            "age": request.age,
            "resource_type_preferences": [t.value for t in request.resource_type_preferences] if request.resource_type_preferences else [],
            "online_only": request.online_only,
            "crisis_need": request.crisis_need
        }
        results = recommender.recommend(user_profile=user_profile)
        return {"recommendations": results}
    except Exception as e:
        logger.error(f"Error recommending resources: {e}", exc_info=True)
        raise RuntimeError(f"Error recommending resources: {e}") from e