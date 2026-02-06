import sys
import json
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'forecasting_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data():
    """
    Loads latest forecasting data.

    :returns: Dict with latest forecasting data.
    :raises: FileNotFoundError: If results not found
    """
    # Path
    latest_forecasts_file = PROJECT_ROOT / 'analysis' / 'outputs' / 'results' / 'forecasting' / 'latest_forecasts.json'

    if not latest_forecasts_file.exists():
        logger.error(f"Latest forecasts not found at {latest_forecasts_file}")
        raise FileNotFoundError("Latest forecasts not found. Re-train forecasting models.")
    
    try:
        # Load data
        with open(latest_forecasts_file, 'r') as f:
            data = json.load(f)
            logger.info("Loaded latest forecasts successfully.")            
            return data
        
    except Exception as e:
        logger.error(f"Error loading latest forecasts: {e}", exc_info=True)
        raise