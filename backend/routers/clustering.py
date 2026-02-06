import sys
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas import ClusterResponse
from backend.services.clustering_service import load_data

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

# Define router
router = APIRouter()

@router.get("/clusters", response_model=ClusterResponse)
def get_clusters():
    """
    Get clustering analysis results.
    
    :returns: ClusterResponse with cluster profiles including:

        - Demographics (age, gender, country)

        - Mental health indicators (treatment, family history)

        - Work environment (remote work, interference)
        
        - Workplace awareness (benefits, consequences)
    """
    try:
        results = load_data()
        return results
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="CLustering results not found. Run clustering notebook first.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
