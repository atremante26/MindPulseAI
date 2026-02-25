import sys
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas import WeeklyInsightsResponse, DatapointResponse, DatapointRequest
from backend.services import load_weekly_insights, get_datapoint_insight

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'insights_router.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Define router
router = APIRouter()

@router.get("/weekly", response_model=WeeklyInsightsResponse)
def get_weekly_insights_enpoint():
    """
    Get weekly insights summary.

    :returns: WeeklyInsightsResponse with weekly summary.
    """
    try:
        return load_weekly_insights()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Weekly Insights summary not found.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@router.post("/datapoint", response_model=DatapointResponse)
def get_datapoint_endpoint(request: DatapointRequest):
    """
    Get insights about single data point.

    :returns: DatapointResponse with insights about single data point.
    """
    try:
        return get_datapoint_insight(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))