import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from gower import gower_matrix
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

def prepare_time_series(df, date_col, value_col):
    """Prepare time series data for forecasting"""
    # Select only the columns needed for time series analysis
    ts_df = df[[date_col, value_col]].copy()
    
    # Convert date column to datetime format
    ts_df[date_col] = pd.to_datetime(ts_df[date_col])
    
    # Sort by date to ensure chronological order
    ts_df = ts_df.sort_values(date_col)
    
    # Set date as index for time series operations
    ts_df = ts_df.set_index(date_col)
    
    return ts_df

def prepare_who_time_series(df, year_col='year', value_col='suicides_no'):
    """Convert WHO annual data to time series format"""
    # Aggregate by year (sum across countries/demographics)
    annual_totals = df.groupby(year_col)[value_col].sum().reset_index()
    
    # Convert year integer to datetime (January 1st of each year)
    annual_totals['date'] = pd.to_datetime(annual_totals[year_col], format='%Y')
    
    # Return in standard time series format
    return annual_totals[['date', value_col]].set_index('date')

def prepare_reddit_for_llm(df, max_posts=15, max_length=200):
    """Prepare Reddit posts sample for LLM analysis"""
    sample_df = df.sample(n=min(len(df), max_posts))
    
    posts_text = []
    for _, row in sample_df.iterrows():
        title = row['title'][:100]
        text = str(row['text'])[:max_length] if pd.notna(row['text']) else ""
        subreddit = row['subreddit']
        
        post_summary = f"Subreddit: {subreddit}\nTitle: {title}\nText: {text}\n---"
        posts_text.append(post_summary)
    
    return "\n".join(posts_text)

def create_statistical_summary_for_llm(cdc_df=None, who_df=None, care_df=None):
    """Generate statistical insights for LLM consumption"""
    summary = {}
    
    if cdc_df is not None and len(cdc_df) > 0:
        summary['CDC Mental Health Trends'] = {
            'Average anxiety prevalence': f"{cdc_df['anxiety'].mean():.1f}%",
            'Average depression prevalence': f"{cdc_df['depression'].mean():.1f}%",
            'Date range': f"{cdc_df['date'].min()} to {cdc_df['date'].max()}"
        }
    
    if who_df is not None and len(who_df) > 0:
        summary['WHO Global Suicide Data'] = {
            'Global suicide rate': f"{who_df['suicides/100k pop'].mean():.1f} per 100k",
            'Time period': f"{who_df['year'].min()}-{who_df['year'].max()}",
            'Countries': who_df['country'].nunique()
        }
    
    if care_df is not None and len(care_df) > 0:
        summary['Mental Health Care Access'] = {
            'Records analyzed': len(care_df)
        }
    
    return summary