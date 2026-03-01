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
from backend.utils import load_from_s3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load static context
static_context_file = PROJECT_ROOT / 'analysis' / 'outputs' / 'results' / 'insights' / 'static_context.txt'
with open(static_context_file, 'r') as f:
    STATIC_CONTEXT = f.read()

def load_weekly_insights():
    return load_from_s3("artifacts/latest_insights.json")

def get_datapoint_insight(request: DatapointRequest):
    """
    Gets insights for specific forecasting model datapoint.
    
    :param DatapointRequest for specific forecasting model datapoint.
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
        raise RuntimeError(f"Datapoint API call failed with error: {e}") from e
