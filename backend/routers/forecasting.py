import sys
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas import ForecastsResponse, HistoricalResponse
from backend.services import load_clustering_data, get_historical

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
def get_forecasts_endpoint():
    """
    Get forecasting results.

    :returns: ForecastsResponse with forecast values and metadata.
    """
    try:
        return load_clustering_data()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Forecasts not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@router.get("/historical", response_model=HistoricalResponse)
def get_historical_endpoint():
    """
    Get historical data for Reddit volume, Reddit sentiment, News Volume, and News Sentiment.

    :returns: HistoricalResponse with historical data.
    """
    try:
        return get_historical()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))