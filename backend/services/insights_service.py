import sys
import json
import logging
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Schemas
from backend.schemas import WeeklyInsightsResponse, DatapointRequest, DatapointResponse
from analysis.insights import call_api_datapoint

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'insights_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load static context
static_context_file = PROJECT_ROOT / 'analysis' / 'outputs' / 'results' / 'insights' / 'static_context.txt'
with open(static_context_file, 'r') as f:
    STATIC_CONTEXT = f.read()

def load_weekly_insights():
    """
    Loads latest insights data.

    :returns: Dict with latest insights data.
    :raises: FileNotFoundError: If results not found.

    """
    # Path
    latest_insights_file = PROJECT_ROOT / 'analysis' / 'outputs' / 'results' / 'insights' / 'latest_insights.json'

    if not latest_insights_file.exists():
        logger.error(f"Latest insights not found at {latest_insights_file}")
        raise FileNotFoundError("Latest insights not found. Re-train insights model.")
    
    try:
        # Load data
        with open(latest_insights_file, 'r') as f:
            data = json.load(f)
            logger.info("Loaded latest insights successfully.")            
            return data
        
    except Exception as e:
        logger.error(f"Error loading latest insights: {e}", exc_info=True)
        raise

def get_datapoint_insight(request: DatapointRequest):
    """
    Gets insights for specific forecasting model datapoint.
    
    :param metric_name: "Reddit Volume", "Reddit Sentiment", etc.
    :param week_date: ISO date string "2026-02-10"
    :param value: Forecasted value
    :param baseline: Average baseline value
    :param confidence_lower: CI lower bound
    :param confidence_upper: CI upper bound
    :param surrounding_weeks: List of dicts with surrounding week data

    :returns: Dict with datapoint insights and metadata.
    """
    try:
        return call_api_datapoint(
            request.metric_name,
            request.week_date,
            request.value,
            request.baseline,
            request.confidence_lower,
            request.confidence_upper,
            [w.model_dump() for w in request.surrounding_weeks],
            STATIC_CONTEXT
        )
    except Exception as e:
        logger.error(f"Error calling API for datapoint insights: {e}", exc_info=True)
        raise RuntimeError(f"Datapoint API call failed with error: {e}")
