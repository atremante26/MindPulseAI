import sys
import json
import logging
import numpy as np
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
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'clustering_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_data():
    """
    Load pre-generated clustering metadata.

    :returns: Dict with cluster profiles and metadata
    :raises: FileNotFoundError: If results not found
    """
    # Path
    metadata_file = PROJECT_ROOT / 'analysis' / 'outputs' / 'results' / 'clustering' / 'cluster_metadata.json'
    
    if not metadata_file.exists():
        logger.error(f"Clustering results not found at {metadata_file}")
        raise FileNotFoundError("Clustering results not found. Run clustering notebook first.")
    
    try:
        # Load data
        with open(metadata_file, 'r') as f:
            data = json.load*(f) 

        logger.info(f"Loaded {data.get('n_clusters', 0)} clusters from clustering results")
        return data
    
    except Exception as e:
        logger.error(f"Error loading clustering results: {e}", exc_info=True)
        raise