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
    
    def _vectorize_user_profile(self, user_profile: Dict) -> np.ndarray:
        """
        Convert user profile to feature vector (same dimensions as resources).

        User profile format:
        {
            "concerns": ["anxiety", "depression"],
            "cost_preference": "free",  # or "low", "medium", "high"
            "age": 23,  # Used to determine age_group
            "resource_type_preferences": ["app", "community"],  # Optional
            "crisis_need": False,  # True if user needs immediate help
            "online_only": False  # True if user wants only online resources
        }

        :param user_profile: Dict of user profile
        :return: NumPy array of features
        """

        features = []

        # Concerns (multi-hot encoding)
        concerns = self.concern_encoder.transform([user_profile.get('concerns', [])])[0]
        features.extend(concerns)

        # Cost tier (one-hot encoding)
        cost_tier = user_profile.get('cost_performance', 'low')
        cost_vector = [
            1 if cost_tier == 'free' else 0,
            1 if cost_tier == 'low' else 0,
            1 if cost_tier == 'medium' else 0,
            1 if cost_tier == 'high' else 0
        ]
        features.extend(cost_vector)

        # Age Groups (convert age to age_group)
        age = user_profile.get('age', 25)
        age_group = self._age_to_age_group(age)
        ages = self.age_encoder.transform([[age_group]])[0]
        features.extend(ages)

        # Resource Type Preferences (one-hot encoding)
        type_prefs = user_profile.get('resource_type_preferences', [])
        types = self.type_encoder.transform([type_prefs])[0] if type_prefs else np.zeros(len(self.type_encoder.classes_))
        features.extend(types)

        # Crisis Need (binary)
        features.append(1 if user_profile.get('crisis_need', False) else 0)

        # Online Only Preference (binary)
        features.append(1 if user_profile.get('online_only', False) else 0)

        # Rating preference (always want high rating, set to 1.0)
        features.append(1.0)

        return np.array(features)

    def _age_to_age_group(self, age: int) -> str:
        """
        Convert numeric age to string age group.
        
        :param age: Integer age of user.
        :return: String age group of user ("teen", "young_adult", "adult", "senior").
        """
        if age < 18:
            return "teen"
        elif age < 26:
            return "young_adult"
        elif age < 65:
            return "adult"
        else:
            return "senior"
    
    def _apply_feature_weights(self, similarity_scores: np.ndarray, user_vector: np.ndarray, resource_vectors: List[np.ndarray]) -> np.ndarray:
        """
        Apply feature-specific weights to similarity scores.

        This computes separate similarities for each feature group (concerns, cost, age, etc.)
        and combines them with configured weights for more nuanced recommendations.
        
        Feature groups:
        - Concerns: Most important (weight: 0.5)
        - Cost: Second most important (weight: 0.2)
        - Age: Moderate importance (weight: 0.15)
        - Type: Lower importance (weight: 0.1)
        - Crisis: Emergency override (weight: 0.05)
        
        Formula:
        weighted_score = (w1 * concerns_sim) + (w2 * cost_sim) + (w3 * age_sim) + ...
        
        :param similarity_scores: np.ndarray of similarity scores between user_vector and resource_vectors.
        :param user_vector: np.ndarray vector of user profile 
        :param resource_vectors: List of np.ndarray vectors of resources
        :return: np.ndarray of weighted scores
        """
        # Get feature weights from config
        weights = self.config['weights']
        
        # Calculate number of features in each group
        n_concerns = len(self.concern_encoder.classes_)
        n_cost = 4  # free, low, medium, high
        n_ages = len(self.age_encoder.classes_)
        n_types = len(self.type_encoder.classes_)
        n_crisis = 1  # binary
        n_online = 1  # binary
        n_rating = 1  # normalized 0-1
        
        # Define slice indices for each feature group
        idx = 0
        concerns_slice = slice(idx, idx + n_concerns)
        idx += n_concerns
        
        cost_slice = slice(idx, idx + n_cost)
        idx += n_cost
        
        age_slice = slice(idx, idx + n_ages)
        idx += n_ages
        
        type_slice = slice(idx, idx + n_types)
        idx += n_types
        
        crisis_slice = slice(idx, idx + n_crisis)
        idx += n_crisis
        
        online_slice = slice(idx, idx + n_online)
        idx += n_online
        
        rating_slice = slice(idx, idx + n_rating)
        
        # Calculate weighted scores for each resource
        weighted_scores = []
        
        for resource_vec in resource_vectors:
            # Extract feature subsets for user and resource
            user_concerns = user_vector[concerns_slice]
            resource_concerns = resource_vec[concerns_slice]
            
            user_cost = user_vector[cost_slice]
            resource_cost = resource_vec[cost_slice]
            
            user_age = user_vector[age_slice]
            resource_age = resource_vec[age_slice]
            
            user_type = user_vector[type_slice]
            resource_type = resource_vec[type_slice]
            
            user_crisis = user_vector[crisis_slice]
            resource_crisis = resource_vec[crisis_slice]
            
            # Compute similarity for each feature group
            # Using cosine similarity for each subset
            
            # Concerns similarity
            concerns_sim = self._cosine_similarity_1d(user_concerns, resource_concerns)
            
            # Cost similarity (exact match is better than cosine here)
            # If user wants "free" and resource is "free", perfect match
            cost_sim = self._exact_match_similarity(user_cost, resource_cost)
            
            # Age similarity
            age_sim = self._cosine_similarity_1d(user_age, resource_age)
            
            # Type similarity
            type_sim = self._cosine_similarity_1d(user_type, resource_type)
            
            # Crisis similarity (binary match)
            crisis_sim = 1.0 if (user_crisis[0] == 1 and resource_crisis[0] == 1) else 0.5
            
            # Combine with weights
            weighted_score = (
                weights['concerns'] * concerns_sim +
                weights['cost'] * cost_sim +
                weights['age'] * age_sim +
                weights['type'] * type_sim +
                weights['crisis'] * crisis_sim
            )
            
            weighted_scores.append(weighted_score)
        
        return np.array(weighted_scores)


    def _cosine_similarity_1d(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute cosine similarity between two 1D vectors.
        
        :
        - Float between 0 and 1 (1 = identical, 0 = completely different)
        """
        # Handle edge case: both vectors are all zeros
        if np.all(vec1 == 0) and np.all(vec2 == 0):
            return 1.0  # Both have no features = perfect match
        
        if np.all(vec1 == 0) or np.all(vec2 == 0):
            return 0.0  # One has features, other doesn't = no match
        
        # Compute cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_product = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        
        if norm_product == 0:
            return 0.0
        
        similarity = dot_product / norm_product
        
        # Ensure result is in [0, 1] range
        return max(0.0, min(1.0, similarity))


    def _exact_match_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        """
        Compute exact match similarity for one-hot encoded features.
        
        For cost tier: we want exact matches to score highest,
        but adjacent tiers should still score decently.
        
        Examples:
        - User wants "free", resource is "free" → 1.0
        - User wants "free", resource is "low" → 0.7
        - User wants "free", resource is "high" → 0.3
        """
        # Find which position is 1 in each vector
        user_idx = np.argmax(vec1) if np.any(vec1) else -1
        resource_idx = np.argmax(vec2) if np.any(vec2) else -1
        
        if user_idx == -1 or resource_idx == -1:
            return 0.5  # Neutral if either is unset
        
        # Perfect match
        if user_idx == resource_idx:
            return 1.0
        
        # Adjacent tiers (e.g., free vs low, or low vs medium)
        distance = abs(user_idx - resource_idx)
        if distance == 1:
            return 0.7
        elif distance == 2:
            return 0.4
        else:  # distance == 3 (free vs high)
            return 0.2
    
