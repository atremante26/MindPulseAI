import sys
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas import ForecastsResponse
from backend.services.forecasting_service import load_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'forecasting_router.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define router
router = APIRouter()

@router.get("/forecasts", response_model=ForecastsResponse)
def get_forecasts():
    """
    Docstring for get_forecasts
    """
    try:
        forecasts = load_data()
        return forecasts
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Forecasts not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/historical")
def get_historical():
    pass