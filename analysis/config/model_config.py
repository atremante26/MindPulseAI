# Clustering configuration
CLUSTERING_CONFIG = {
    'hdbscan': {
        'min_cluster_size': 25,
        'min_samples': 4,
        'cluster_selection_epsilon': 0.1
    },
    'preprocessing': {
        'categorical_columns': [
            'Gender', 'Country', 'treatment', 'work_interfere',
            'family_history', 'remote_work', 'mental_health_consequence', 'benefits'
        ],
        'numeric_columns': ['Age'],
        'scale_numeric': True
    }
}

# Forecasting Configuration
FORECASTING_CONFIG = {
    'prophet': {
        'changepoint_prior_scale': 0.05,  # Flexibility of trend changes (0.001-0.5)
        'seasonality_prior_scale': 10.0,   # Strength of seasonality (0.01-10)
        'seasonality_mode': 'additive',    # 'additive' or 'multiplicative'
        'weekly_seasonality': True,
        'yearly_seasonality': False,       # Not enough data yet
        'daily_seasonality': False,
        'confidence_interval': 0.95        # 95% confidence intervals
    },
    'forecast_horizon_weeks': 13,
    'forecast_horizon_days': 91,
    'sentiment': {
        'positive_threshold': 0.05,    # Scores > 0.05 are positive
        'negative_threshold': -0.05    # Scores < -0.05 are negative
    }
}

# LLM Insights Configuration
INSIGHTS_CONFIG = {
    'llm': {
        'model': 'claude-3-5-haiku-20241022',  
        'max_tokens': 2048,                     # Enough for detailed insights
        'temperature': 0.7,                     # Balance creativity and consistency
    },
    'sampling': {
        'reddit_posts_per_week': 30,           # Sample recent high-engagement posts
        'news_headlines_per_week': 15,         # Sample recent headlines
        'weeks_lookback': 1,                   # Look at last week's data
    },
    'analysis': {
        'min_weeks_for_insights': 4,           # Need 4+ weeks of forecasts for trends
        'include_static_context': True,        # Include WHO/mental health stats
        'parse_structured_output': True,       # Parse into sections
    }
}