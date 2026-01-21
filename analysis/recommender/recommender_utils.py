import sys
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.config.model_config import RECOMMENDER_CONFIG

# Configure logging
logger = logging.getLogger(__name__)

class MentalHealthRecommender:
    """
    Content-based recommender for mental health resources.

    Uses weighted feature matching to recommend resources based on:
        - Mental health concerns
        - Cost constraints
        - Age group
        - Resource type preferences
        - Crisis needs
    """

    def __init__(self, resources_path: Optional[Path] = None):
        """
        Initialize recommender with resource database.
        
        :param resources_path: Path to mental_health_resources.json
        """

        if resources_path is None:
            resources_path = PROJECT_ROOT / "data/resources/mental_health_resources.json"

        self.resources_path = resources_path
        self.resources = self._load_resources()
        self.config = RECOMMENDER_CONFIG

        # Initialize encoders
        self._init_encoder()

        logger.info(f"Initialized recommender with {len(self.resources)} resources.")

    def _load_resources(self) -> List[Dict]:
        """
        Load resource database from JSON.
        """
        try:
            with open(self.resources_path, 'r') as f:
                resources = json.load(f)
                logger.info(f"Loaded {len(resources)} resources from {self.resources_path.name}")
                return resources
        except Exception as e:
            logger.error(f"Failed to load resources: {e}")
            return self._get_fallback_resources()
    
    def _get_fallback_resources(self) -> List[Dict]:
        """Return essential crisis resources if database load fails."""
        return [
            {
                "id": "988_lifeline",
                "name": "988 Suicide & Crisis Lifeline",
                "type": "hotline",
                "description": "Free, 24/7 crisis support",
                "phone": "988",
                "concerns": ["suicidal_thoughts", "self_harm", "depression", "anxiety"],
                "cost_tier": "free",
                "age_groups": ["all"],
                "crisis_resource": True,
                "rating": 4.5,
                "tags": ["crisis", "hotline", "free", "24/7"]
            },
            {
                "id": "crisis_text_line",
                "name": "Crisis Text Line",
                "type": "hotline",
                "description": "Text HELLO to 741741",
                "phone": "741741",
                "concerns": ["suicidal_thoughts", "self_harm", "depression", "anxiety"],
                "cost_tier": "free",
                "age_groups": ["teen", "young_adult", "adult"],
                "crisis_resource": True,
                "rating": 4.7,
                "tags": ["crisis", "text", "free", "24/7"]
            }
        ]
    
    def _init_encoders(self) -> None:
        """Initialize multi-label binarizers for categorical features."""
        # Extract all unique values for each categorical feature
        all_concerns = set()
        all_types = set()
        all_ages = set()
        all_tags = set()

        # Extract categorical features
        for resource in self.resources:
            all_concerns.update(resource.get('concerns', []))
            all_types.update(resource.get('type', []))
            all_ages.update(resource.get('age_groups', []))
            all_tags.update(resource.get('tags', []))

        # Create encoders
        self.concern_encoder = MultiLabelBinarizer(classes=sorted(all_concerns))
        self.type_encoder = MultiLabelBinarizer(classes=sorted(all_types))
        self.age_encoder = MultiLabelBinarizer(classes=sorted(all_ages))
        self.tag_encoder = MultiLabelBinarizer(classes=sorted(all_tags))

        # Fit the encoders with dummy data 
        self.concern_encoder.fit([[]])
        self.type_encoder.fit([[]])
        self.age_encoder.fit([[]])
        self.tag_encoder.fit([[]])

        logger.info(f"Initialized encoders: {len(all_concerns)} concerns, {len(all_types)} types, {len(all_ages)} age groups, and {len(all_tags)} tags.")

    def _vectorize_resource(self, resource: Dict) -> np.ndarray:
        """
        Convert resource to feature vector.

        :param resource: Single mental health resource dictionary
        :return: NumPy array of features
        """

        features = []

        # Concerns (multi-hot encoding)
        concerns = self.concern_encoder.transform([resource.get('concerns', [])])[0]
        features.extend(concerns)

        # Cost tier (one-hot encoding)
        cost_tier = resource.get('cost_tier', 'medium')
        cost_vector = [
            1 if cost_tier == 'free' else 0,
            1 if cost_tier == 'low' else 0,
            1 if cost_tier == 'medium' else 0,
            1 if cost_tier == 'high' else 0
        ]
        features.extend(cost_vector)

        # Age Groups (multi-hot encoding)
        ages = self.age_encoder.transform([resource.get('age_groups', [])])[0]
        features.extend(ages)

        # Resource Type (one-hot encoding)
        types = self.type_encoder.transform([resource.get('type', '')])[0]
        features.extend(types)

        # Crisis Resource (binary)
        features.append(1 if resource.get('crisis_resource', False) else 0)

        # Online Only (binary)
        features.append(1 if resource.get('online_only', False) else 0)

        # Rating (normalized 0-1)
        rating = resource.get('rating', 3.0)
        features.append(rating / 5.0)   # Normalize 0-1

        return np.array(features)
    
