import os
import sys
import pickle
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

class ClusteringService:
    def __init__(self):
        self.data = {}
        self.models_dir = PROJECT_ROOT / 'analysis' / 'outputs' / 'results' / 'clustering'
    
    def load_data(self):
        try:
            data_files = list(self.models_dir.glob("*.csv"))
            
            if not data_files:
                logger.warning("No model files found")
                return
            
            # Get latest file
            self.data = max(data_files, key=os.path.getmtime)

            logger.info("Clustering data loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading data: {e}", exc_info=True)
    
    def is_loaded(self) -> bool:
        return all(key in self.models for key in ['clusterer', 'preprocessing', 'results'])
    
    def get_all_clusters(self):
        if not self.is_loaded():
            raise ValueError("Models not loaded")
        
        results = self.models['results']
        labels = results['labels']
        cluster_profiles = results.get('cluster_profiles', {})
        unique_labels = [l for l in np.unique(labels) if l != -1]
        
        descriptions = {
            0: "Remote Workers with Severe Impact",
            1: "Mainstream Tech Workers",
            2: "Uninformed/Uncertain Group"
        }
        
        clusters = []
        for label in unique_labels:
            cluster_size = int(np.sum(labels == label))
            cluster_pct = float(cluster_size / len(labels) * 100)
            profile = cluster_profiles.get(label, {})
            if not isinstance(profile, dict):
                profile = {}
            
            # Convert numpy types to Python native types
            clean_profile = self._convert_numpy_types(profile)

            clusters.append({
                "cluster_id": int(label),
                "size": cluster_size,
                "percentage": round(cluster_pct, 1),
                "description": descriptions.get(label, f"Cluster {label}"),
                "characteristics": clean_profile
            })
        
        return clusters
    
    def get_cluster_detail(self, cluster_id: int):
        if not self.is_loaded():
            raise ValueError("Models not loaded")
        
        results = self.models['results']
        labels = results['labels']
        
        if cluster_id not in np.unique(labels):
            raise ValueError(f"Cluster {cluster_id} not found")
        
        cluster_profiles = results.get('cluster_profiles', {})
        profile = cluster_profiles.get(cluster_id, {})

        # Convert numpy types
        clean_profile = self._convert_numpy_types(profile)
        
        cluster_size = int(np.sum(labels == cluster_id))
        cluster_pct = float(cluster_size / len(labels) * 100)
        
        descriptions = {
            0: "Remote Workers with Severe Impact",
            1: "Mainstream Tech Workers",
            2: "Uninformed/Uncertain Group"
        }
        
        return {
            "cluster_id": cluster_id,
            "name": descriptions.get(cluster_id, f"Cluster {cluster_id}"),
            "size": cluster_size,
            "percentage": round(cluster_pct, 1),
            "profile": clean_profile,
            "description": descriptions.get(cluster_id, "")
        }
    
    def predict_cluster(self, user_data: dict):
        if not self.is_loaded():
            raise ValueError("Models not loaded")
        
        # Simple heuristic-based prediction
        # In production, implement proper distance-based prediction
        cluster_id = 1  # Default
        
        if user_data.get('remote_work', '').lower() == 'yes' and \
           user_data.get('work_interfere', '').lower() == 'often':
            cluster_id = 0
        elif user_data.get('benefits', '').lower() == "don't know":
            cluster_id = 2
        
        descriptions = {
            0: "Remote Workers with Severe Impact - You match the profile of remote workers experiencing frequent mental health interference at work",
            1: "Mainstream Tech Workers - You match the typical tech worker profile with moderate symptoms and workplace awareness",
            2: "Uninformed/Uncertain Group - You match the profile of employees who may lack awareness about workplace mental health resources"
        }
        
        cluster_names = {
            0: "Remote Workers with Severe Impact",
            1: "Mainstream Tech Workers",
            2: "Uninformed/Uncertain Group"
        }
        
        results = self.models.get('results', {})
        cluster_profiles = results.get('cluster_profiles', {})
        characteristics = cluster_profiles.get(cluster_id, {})
        
        return {
            "cluster_id": cluster_id,
            "cluster_name": cluster_names[cluster_id],
            "description": descriptions[cluster_id],
            "characteristics": characteristics,
            "confidence": 0.75
        }
    
    def _convert_numpy_types(self, obj):
        """
        Recursively convert numpy types to Python native types for JSON serialization.
        """
        import numpy as np
        
        if isinstance(obj, dict):
            return {key: self._convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_numpy_types(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj