import numpy as np
import pandas as pd
from hdbscan import HDBSCAN
from gower import gower_matrix
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def prepare_clustering_features(df, categorical_cols, numeric_cols):
    # Select only relevant columns
    feature_df = df[categorical_cols + numeric_cols].copy()
    
    # Handle any remaining nulls
    feature_df = feature_df.dropna()

    # Save filtered original
    filtered_original = feature_df.copy()
    
    # Encode categorical variables for Gower distance
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        feature_df[col] = le.fit_transform(feature_df[col].astype(str))
        le_dict[col] = le
    
    # Scale numeric variables
    scaler = StandardScaler()
    feature_df[numeric_cols] = scaler.fit_transform(feature_df[numeric_cols])
    
    return feature_df, le_dict, scaler, filtered_original

def compute_gower_distance(df, categorical_indices):
    # Create categorical mask
    cat_features = [i in categorical_indices for i in range(df.shape[1])]
    
    # Compute Gower distance
    distance_matrix = gower_matrix(df.values, cat_features=cat_features)
    
    return distance_matrix

def run_hdbscan_clustering(distance_matrix, min_cluster_size=15, min_samples=10, cluster_selection_epsilon=0.1):
    """Run HDBSCAN clustering on distance matrix"""
    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric='precomputed',
        cluster_selection_epsilon=cluster_selection_epsilon,
        prediction_data=True       # Enables cluster_persistence_ calculation
    )
    
    cluster_labels = clusterer.fit_predict(distance_matrix)
    
    return clusterer, cluster_labels

def evaluate_clustering(clusterer, distance_matrix, cluster_labels):
    """Evaluate cluster quality with silhouette score and cluster persistence."""
    # Filter out noise points
    non_noise_mask = cluster_labels != -1
    
    if sum(non_noise_mask) < 2:
        logger.warning("Less than 2 non-noise points, cannot compute silhouette score")
        return {
            'silhouette_score': None,
            'persistence_scores': None
        }
    
    n_clusters = len(set(cluster_labels[non_noise_mask]))
    if n_clusters < 2:
        logger.warning("Less than 2 clusters found, cannot compute silhouette score")
        return {
            'silhouette_score': None,
            'persistence_scores': None
        }
    
    # Extract distance matrix for non-noise points
    filtered_distance = distance_matrix[non_noise_mask][:, non_noise_mask]
    
    # Diagonal is exactly zero for precomputed distances
    np.fill_diagonal(filtered_distance, 0)
    
    silhouette_avg = silhouette_score(
        filtered_distance,
        cluster_labels[non_noise_mask], 
        metric='precomputed'
    )

    # Compute Cluster Persistence (Stability) Scores
    persistence_scores = []
    
    if hasattr(clusterer, 'cluster_persistence_'):
        persistence_scores = clusterer.cluster_persistence_.tolist()
        logger.info(f"Cluster persistence scores: {persistence_scores}")
    else:
            logger.warning("Cannot calculate persistence scores")
            persistence_scores = [0.0] * n_clusters

    return {
        'silhouette_score': float(silhouette_avg),
        'persistence_scores': persistence_scores
    }

def generate_cluster_profiles(data, labels):
    """Generate demographic and behavioral profiles for each cluster."""
    # Add cluster labels to data
    labeled_data = data.copy()
    labeled_data['cluster_id'] = labels

    # Create profiles
    profiles = {}

    for cluster_id in sorted(set(labels)):
        # Skip noise points
        if cluster_id == -1:
            continue
        
        # Filter data
        cluster_data = labeled_data[labeled_data['cluster_id'] == cluster_id]
        n_total = len(labeled_data)
        n_cluster = len(cluster_data)

        # Demographics - Age
        age_mean = float(cluster_data['Age'].mean())
        age_std = float(cluster_data['Age'].std())

        # Demographics - Gender
        gender_counts = cluster_data['Gender'].value_counts(normalize=True)
        gender_dist = {k: round(float(v) * 100, 1) for k, v in gender_counts.items()}

        # Demographics - Country
        country_counts = cluster_data['Country'].value_counts()
        top_countries = {k: int(v) for k, v in country_counts.head(3).items()}

        # Treatment and Family History
        treatment_pct = float((cluster_data['treatment'] == 'Yes').mean() * 100.0)
        family_history_pct = float((cluster_data['family_history'] == 'Yes').mean() * 100)
        
        # Work Factors
        remote_pct = float((cluster_data['remote_work'] == 'Yes').mean() * 100)
        
        work_interfere_counts = cluster_data['work_interfere'].value_counts(normalize=True)
        work_interfere_dist = {k: round(float(v) * 100, 1) for k, v in work_interfere_counts.items()}
        top_work_interfere = cluster_data['work_interfere'].mode()[0] if len(cluster_data) > 0 else 'Unknown'
        
        # Workplace Awareness
        benefits_counts = cluster_data['benefits'].value_counts(normalize=True)
        benefits_dist = {k: round(float(v) * 100, 1) for k, v in benefits_counts.items()}
        
        consequence_counts = cluster_data['mental_health_consequence'].value_counts(normalize=True)
        consequence_dist = {k: round(float(v) * 100, 1) for k, v in consequence_counts.items()}
        
        # Build profile dictionary
        profile = {
            # Cluster Size
            "size": int(n_cluster),
            "percentage": round(float(n_cluster / n_total * 100), 1),
            
            # Demographics
            "avg_age": round(age_mean, 1),
            "age_std": round(age_std, 1),
            "gender_distribution": gender_dist,
            "top_countries": top_countries,
            
            # Mental Health
            "in_treatment_pct": round(treatment_pct, 1),
            "family_history_pct": round(family_history_pct, 1),
            
            # Work Environment
            "remote_work_pct": round(remote_pct, 1),
            "work_interference_distribution": work_interfere_dist,
            "top_work_interference": top_work_interfere,
            
            # Workplace Awareness
            "benefits_distribution": benefits_dist,
            "mental_health_consequence_distribution": consequence_dist,
            
            # Summary
            "key_traits": []  # Populate with key statistics
        }
        
        # Identify key traits (characteristics that are >20% above/below average)
        avg_remote = float((labeled_data['remote_work'] == 'Yes').mean() * 100)
        avg_treatment = float((labeled_data['treatment'] == 'Yes').mean() * 100)
        
        if remote_pct > avg_remote + 20:
            profile["key_traits"].append(f"High remote work rate ({remote_pct:.0f}% vs {avg_remote:.0f}% avg)")
        elif remote_pct < avg_remote - 20:
            profile["key_traits"].append(f"Low remote work rate ({remote_pct:.0f}% vs {avg_remote:.0f}% avg)")
        
        if treatment_pct > avg_treatment + 20:
            profile["key_traits"].append(f"High treatment rate ({treatment_pct:.0f}% vs {avg_treatment:.0f}% avg)")
        elif treatment_pct < avg_treatment - 20:
            profile["key_traits"].append(f"Low treatment rate ({treatment_pct:.0f}% vs {avg_treatment:.0f}% avg)")
        
        if top_work_interfere == 'Often':
            profile["key_traits"].append("Frequent work interference")
        elif top_work_interfere == 'Never':
            profile["key_traits"].append("Minimal work interference")
        
        # Check for benefits awareness
        dont_know_pct = benefits_dist.get("Don't know", 0)
        if dont_know_pct > 30:
            profile["key_traits"].append(f"Low benefits awareness ({dont_know_pct:.0f}% don't know)")
        
        profiles[cluster_id] = profile

    return profiles