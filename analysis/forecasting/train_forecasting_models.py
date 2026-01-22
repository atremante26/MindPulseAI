import sys
import logging
from pathlib import Path
from datetime import datetime
import pickle
import json
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.load_data import load_dataset
from analysis.config.model_config import FORECASTING_CONFIG
from analysis.forecasting import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'forecasting_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def prepare_reddit_data():
    """Load and prepare Reddit data for forecasting."""
    # Load data
    reddit_df = load_dataset('reddit')
    logger.info(f"Loaded {len(reddit_df)} Reddit posts")
    
    # Ensure datetime
    reddit_df['DATE'] = pd.to_datetime(reddit_df['DATE'])
    
    # Add sentiment
    reddit_df = add_sentiment_column(
        df=reddit_df,
        text_column='TEXT',
        sentiment_column='sentiment',
        is_aggregated=False # Each row is a single post
    )
    
    # Aggregate to weekly
    weekly_reddit = aggregate_to_weekly(
        df=reddit_df,
        date_col='DATE',
        agg_dict={
            'TEXT': 'count',        # Count total posts per week
            'sentiment': 'mean',    # Average sentiment per week
            'SCORE': 'mean',        # Average Reddit score
            'COMMENTS': 'mean'      # Average comments
        }
    )
    
    # Rename for consistency
    weekly_reddit.rename(columns={'TEXT': 'volume', 'DATE': 'date'}, inplace=True)
    
    # Fill missing weeks
    weekly_reddit = fill_missing_weeks(weekly_reddit, date_col='date')
    
    logger.info(f"Final Reddit data: {len(weekly_reddit)} weeks")
    logger.info(f"Date range: {weekly_reddit['date'].min()} to {weekly_reddit['date'].max()}")
    
    return weekly_reddit


def prepare_news_data():
    """Load and prepare News data for forecasting."""
    # Load data
    news_df = load_dataset('news')
    logger.info(f"Loaded {len(news_df)} News records")
    
    # Ensure datetime
    news_df['DATE'] = pd.to_datetime(news_df['DATE'])
    
    # Detect delimiter
    sample_headlines = str(news_df['SAMPLE_HEADLINES'].iloc[0])
    delimiters = {';': sample_headlines.count(';'),
                  '|': sample_headlines.count('|'),
                  '||': sample_headlines.count('||')}
    detected_delimiter = max(delimiters, key=delimiters.get)
    logger.info(f"Using delimiter: '{detected_delimiter}'")
    
    # Add sentiment
    news_df = add_sentiment_column(
        df=news_df,
        text_column='SAMPLE_HEADLINES',
        sentiment_column='sentiment',
        is_aggregated=True, # Each row has multiple headlines
        delimiter=detected_delimiter
    )
    
    # Aggregate to weekly
    weekly_news = aggregate_to_weekly(
        df=news_df,
        date_col='DATE',
        agg_dict={
            'ARTICLE_COUNT': 'sum', # Sum articles across any runs in the week
            'sentiment': 'mean'     # Average sentiment
        },
        verbose=True
    )
    
    # Rename for consistency
    weekly_news.rename(columns={'ARTICLE_COUNT': 'volume', 'DATE': 'date'}, inplace=True)
    
    # Remove first week if it's an outlier (data collection artifact)
    if len(weekly_news) > 2:
        first_week_vol = weekly_news['volume'].iloc[0]
        rest_mean = weekly_news['volume'].iloc[1:].mean()
        
        if first_week_vol > rest_mean * 1.5:
            logger.info(f"Removing first week outlier: {first_week_vol:.0f} (baseline: {rest_mean:.0f})")
            weekly_news = weekly_news.iloc[1:].reset_index(drop=True)
    
    # Fill missing weeks with linear interpolation
    date_diffs = weekly_news['date'].diff().dt.days.dropna()
    if not (date_diffs == 7).all():
        logger.info("Filling missing weeks in News data")
        weekly_news = fill_missing_weeks(weekly_news, date_col='date')
    else:
        logger.info("News data has perfect weekly frequency")
    
    logger.info(f"Final News data: {len(weekly_news)} weeks")
    logger.info(f"Date range: {weekly_news['date'].min()} to {weekly_news['date'].max()}")
    
    return weekly_news


def train_models(weekly_reddit, weekly_news):
    """Train all 4 Prophet forecasting models with optional hyperparameter tuning."""
    forecast_periods = FORECASTING_CONFIG['forecast_horizon_weeks']
    models = {}
    forecasts = {}
    metrics = {}
    
    # Hyperparameter search configuration
    TUNING_MIN_WEEKS = 20  # Minimum weeks needed for reliable tuning
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.05, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0]
    }
    
    # REDDIT VOLUME
    # Tune if enough data
    reddit_volume_config = FORECASTING_CONFIG['prophet'].copy()
    if len(weekly_reddit) >= TUNING_MIN_WEEKS:
        logger.info(f"Sufficient data ({len(weekly_reddit)} weeks) - running hyperparameter search")
        best_params = hyperparameter_search(
            df=weekly_reddit,
            value_col='volume',
            param_grid=param_grid,
            model_name='Reddit Volume'
        )
        reddit_volume_config['changepoint_prior_scale'] = best_params['changepoint_prior_scale']
        reddit_volume_config['seasonality_prior_scale'] = best_params['seasonality_prior_scale']
    else:
        logger.info(f"Limited data ({len(weekly_reddit)} weeks) - using default config")
    
    # Train
    models['reddit_volume'], data, freq = train_prophet(
        df=weekly_reddit,
        pred_col='volume',
        model_name='Reddit Volume',
        config=reddit_volume_config
    )
    forecasts['reddit_volume'] = predict_prophet(
        models['reddit_volume'], forecast_periods, 'Reddit Volume', freq
    )
    metrics['reddit_volume'] = evaluate_prophet(
        models['reddit_volume'], data, forecasts['reddit_volume'], use_cv=False
    )
    
    # REDDIT SENTIMENT
    # Tune if enough data
    reddit_sentiment_config = FORECASTING_CONFIG['prophet'].copy()
    if len(weekly_reddit) >= TUNING_MIN_WEEKS:
        logger.info(f"Sufficient data ({len(weekly_reddit)} weeks) - running hyperparameter search")
        best_params = hyperparameter_search(
            df=weekly_reddit,
            value_col='sentiment',
            param_grid=param_grid,
            model_name='Reddit Sentiment'
        )
        reddit_sentiment_config['changepoint_prior_scale'] = best_params['changepoint_prior_scale']
        reddit_sentiment_config['seasonality_prior_scale'] = best_params['seasonality_prior_scale']
    else:
        logger.info(f"Limited data ({len(weekly_reddit)} weeks) - using default config")
    
    # Train
    models['reddit_sentiment'], data, freq = train_prophet(
        df=weekly_reddit,
        pred_col='sentiment',
        model_name='Reddit Sentiment',
        config=reddit_sentiment_config
    )
    forecasts['reddit_sentiment'] = predict_prophet(
        models['reddit_sentiment'], forecast_periods, 'Reddit Sentiment', freq
    )
    metrics['reddit_sentiment'] = evaluate_prophet(
        models['reddit_sentiment'], data, forecasts['reddit_sentiment'], use_cv=False
    )
    
    # NEWS VOLUME
    # Special handling for limited News data
    news_volume_config = FORECASTING_CONFIG['prophet'].copy()
    
    if len(weekly_news) < 15:
        # Use flat growth for very limited data
        news_volume_config['growth'] = 'flat'
        news_volume_config['weekly_seasonality'] = False
        logger.info(f"Very limited data ({len(weekly_news)} weeks) - using 'flat growth' config")
    elif len(weekly_news) >= TUNING_MIN_WEEKS:
        # Tune if enough data
        logger.info(f"Sufficient data ({len(weekly_news)} weeks) - running hyperparameter search")
        best_params = hyperparameter_search(
            df=weekly_news,
            value_col='volume',
            param_grid=param_grid,
            model_name='News Volume'
        )
        news_volume_config['changepoint_prior_scale'] = best_params['changepoint_prior_scale']
        news_volume_config['seasonality_prior_scale'] = best_params['seasonality_prior_scale']
    else:
        logger.info(f"Limited data ({len(weekly_news)} weeks) - using default config")
    
    # Train
    models['news_volume'], data, freq = train_prophet(
        df=weekly_news,
        pred_col='volume',
        model_name='News Volume',
        config=news_volume_config
    )
    forecasts['news_volume'] = predict_prophet(
        models['news_volume'], forecast_periods, 'News Volume', freq
    )
    metrics['news_volume'] = evaluate_prophet(
        models['news_volume'], data, forecasts['news_volume'], use_cv=False
    )
    
    # NEWS SENTIMENT    
    # Tune if enough data
    news_sentiment_config = FORECASTING_CONFIG['prophet'].copy()
    if len(weekly_news) >= TUNING_MIN_WEEKS:
        logger.info(f"Sufficient data ({len(weekly_news)} weeks) - running hyperparameter search")
        best_params = hyperparameter_search(
            df=weekly_news,
            value_col='sentiment',
            param_grid=param_grid,
            model_name='News Sentiment'
        )
        news_sentiment_config['changepoint_prior_scale'] = best_params['changepoint_prior_scale']
        news_sentiment_config['seasonality_prior_scale'] = best_params['seasonality_prior_scale']
    else:
        logger.info(f"Limited data ({len(weekly_news)} weeks) - using default config")
    
    # Train
    models['news_sentiment'], data, freq = train_prophet(
        df=weekly_news,
        pred_col='sentiment',
        model_name='News Sentiment',
        config=news_sentiment_config
    )
    forecasts['news_sentiment'] = predict_prophet(
        models['news_sentiment'], forecast_periods, 'News Sentiment', freq
    )
    metrics['news_sentiment'] = evaluate_prophet(
        models['news_sentiment'], data, forecasts['news_sentiment'], use_cv=False
    )
    
    return models, forecasts, metrics


def save_models(models, forecasts, metrics, weekly_reddit, weekly_news):
    """Save trained models and forecasts."""
    # Get current timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Create directories
    models_dir = PROJECT_ROOT / 'analysis/models/saved_models/forecasting'
    outputs_dir = PROJECT_ROOT / 'analysis/outputs/results/forecasting'
    models_dir.mkdir(parents=True, exist_ok=True)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save Prophet models
    logger.info("Saving Prophet models (.pkl)")
    for model_name, model in models.items():
        model_path = models_dir / f'prophet_{model_name}_{timestamp}.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        logger.info(f"  ✓ {model_path.name}")
    
    # Extract future predictions
    logger.info("Extracting future predictions")
    forecast_periods = FORECASTING_CONFIG['forecast_horizon_weeks']
    predictions = {}
    
    for model_name, forecast in forecasts.items():
        # Get training data
        if 'reddit' in model_name:
            training_data = weekly_reddit
            pred_col = 'volume' if 'volume' in model_name else 'sentiment'
        else:
            training_data = weekly_news
            pred_col = 'volume' if 'volume' in model_name else 'sentiment'
        
        # Prepare prophet_df format
        prophet_df = training_data[['date', pred_col]].copy()
        prophet_df.columns = ['ds', 'y']
        prophet_df = prophet_df.dropna().sort_values('ds')
        
        # Extract future
        predictions[model_name] = get_future_predictions(
            forecast, prophet_df, forecast_periods
        )
    
    # Prepare JSON data
    def convert_timestamps(obj):
        if isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Type {type(obj)} not serializable")
    
    forecasts_data = {
        'reddit_volume': {
            'predictions': predictions['reddit_volume'].to_dict('records'),
            'metrics': metrics['reddit_volume'],
            'training_weeks': len(weekly_reddit),
            'model_type': 'Prophet',
            'frequency': 'weekly'
        },
        'reddit_sentiment': {
            'predictions': predictions['reddit_sentiment'].to_dict('records'),
            'metrics': metrics['reddit_sentiment'],
            'training_weeks': len(weekly_reddit),
            'model_type': 'Prophet',
            'frequency': 'weekly'
        },
        'news_volume': {
            'predictions': predictions['news_volume'].to_dict('records'),
            'metrics': metrics['news_volume'],
            'training_weeks': len(weekly_news),
            'model_type': 'Prophet (flat growth)' if len(weekly_news) < 15 else 'Prophet',
            'frequency': 'weekly'
        },
        'news_sentiment': {
            'predictions': predictions['news_sentiment'].to_dict('records'),
            'metrics': metrics['news_sentiment'],
            'training_weeks': len(weekly_news),
            'model_type': 'Prophet',
            'frequency': 'weekly'
        },
        'metadata': {
            'forecast_horizon_weeks': forecast_periods,
            'training_timestamp': timestamp,
            'prophet_config': FORECASTING_CONFIG['prophet'],
            'data_sources': {
                'reddit': {
                    'date_range': f"{weekly_reddit['date'].min().date()} to {weekly_reddit['date'].max().date()}",
                    'weeks': len(weekly_reddit)
                },
                'news': {
                    'date_range': f"{weekly_news['date'].min().date()} to {weekly_news['date'].max().date()}",
                    'weeks': len(weekly_news)
                }
            }
        }
    }
    
    # Save forecasts
    logger.info("Saving forecasts (JSON)")
    forecasts_path = outputs_dir / f'forecasts_{timestamp}.json'
    with open(forecasts_path, 'w') as f:
        json.dump(forecasts_data, f, indent=2, default=convert_timestamps)
    logger.info(f"  ✓ {forecasts_path.name}")
    
    # Save as latest
    latest_path = outputs_dir / 'latest_forecasts.json'
    with open(latest_path, 'w') as f:
        json.dump(forecasts_data, f, indent=2, default=convert_timestamps)
    logger.info(f"  ✓ latest_forecasts.json")
    
    # Save weekly data
    logger.info("Saving weekly aggregated data (CSV)")
    weekly_reddit.to_csv(outputs_dir / f'weekly_reddit_{timestamp}.csv', index=False)
    weekly_news.to_csv(outputs_dir / f'weekly_news_{timestamp}.csv', index=False)
    logger.info(f"  ✓ weekly_reddit_{timestamp}.csv")
    logger.info(f"  ✓ weekly_news_{timestamp}.csv")
    
    logger.info(f"\nAll artifacts saved with timestamp: {timestamp}")
    
    return timestamp


def main():
    try:
        # Define start time
        start_time = datetime.now()
        
        # Prepare data
        weekly_reddit = prepare_reddit_data()
        weekly_news = prepare_news_data()
        
        # Train models
        models, forecasts, metrics = train_models(weekly_reddit, weekly_news)
        
        # Save artifacts
        timestamp = save_models(models, forecasts, metrics, weekly_reddit, weekly_news)
        
        # Summary
        elapsed = (datetime.now() - start_time).total_seconds()
        
        # Log info
        logger.info("TRAINING COMPLETE!")
        logger.info(f"Timestamp: {timestamp}")
        logger.info(f"Duration: {elapsed:.1f} seconds")
        logger.info(f"Models trained: 4")
        logger.info(f"Reddit data: {len(weekly_reddit)} weeks")
        logger.info(f"News data: {len(weekly_news)} weeks")
        
        return 0
        
    except Exception as e:
        logger.error(f"\nTraining failed: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())