import sys
import logging
from typing import List
from pathlib import Path
from fastapi import APIRouter, HTTPException, Depends
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas.clustering import UserInput, ClusterInfo, PredictionResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'clustering_router.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
router = APIRouter()

def get_clustering_service():
    from main import clustering_service
    if not clustering_service:
        raise HTTPException(status_code=503, detail="Service not initialized")
    return clustering_service

@router.get("/clusters", response_model=List[ClusterInfo])
def get_clusters(service = Depends(get_clustering_service)):
    try:
        return service.get_all_clusters()
    except Exception as e:
        logger.error(f"Error in get_clusters: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/clusters/{cluster_id}")
def get_cluster_detail(cluster_id: int, service = Depends(get_clustering_service)):
    try:
        return service.get_cluster_detail(cluster_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error in get_cluster_detail: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/clusters/predict", response_model=PredictionResponse) # this endpoint returns a PredictionResponse object
def predict_cluster(user_input: UserInput, service = Depends(get_clustering_service)): # Dependency Injection (call get_clustering_service() and pass result as 'service')
    try:
        return service.predict_cluster(user_input.model_dump())
    except Exception as e:
        logger.error(f"Error in predict_cluster: {e}")
        raise HTTPException(status_code=500, detail=str(e))