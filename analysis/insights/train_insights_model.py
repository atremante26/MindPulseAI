import sys
import json
import logging
import pandas as pd
from pathlib import Path
from datetime import datetime

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from analysis.load_data import load_dataset
from analysis.config.model_config import INSIGHTS_CONFIG
from insights import *

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(PROJECT_ROOT / 'logs' / 'insights_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_latest_forecasts():
    """Load forecasts from latest_forecasts.json"""
    forecasts_path = PROJECT_ROOT / 'analysis/outputs/results/forecasting/latest_forecasts.json'
    
    if not forecasts_path.exists():
        raise FileNotFoundError(f"Forecasts not found at {forecasts_path}. Run train_forecasting_models.py first.")
    
    with open(forecasts_path, 'r') as f:
        forecasts = json.load(f)
    
    logger.info(f"Loaded forecasts from {forecasts_path.name}")
    return forecasts


def save_insights(insights_data, timestamp):
    """Save insights to JSON files"""
    outputs_dir = PROJECT_ROOT / 'analysis/outputs/results/insights'
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Save with timestamp
    insights_path = outputs_dir / f'insights_{timestamp}.json'
    with open(insights_path, 'w') as f:
        json.dump(insights_data, f, indent=2)
    logger.info(f"  Saved {insights_path.name}")
    
    # Save as latest (for API access)
    latest_path = outputs_dir / 'latest_insights.json'
    with open(latest_path, 'w') as f:
        json.dump(insights_data, f, indent=2)
    logger.info(f"  Saved latest_insights.json")

def train():
    """
    Train insights model for weekly summary using Claude Haiku 3.5.

    Returns:
    - insights: Dict with insights text and metadata
    - timestamp: Start timestamp of training
    - elapsed: Total training time
    """
    # Training start time
    start_time = datetime.now()
    timestamp = start_time.strftime('%Y%m%d_%H%M%S')
    
    # Load datasets
    logger.info("\nLoading datasets...")
    reddit_df = load_dataset('reddit')
    news_df = load_dataset('news')
    forecasts = load_latest_forecasts()
    
    logger.info(f"  Reddit: {len(reddit_df)} posts")
    logger.info(f"  News: {len(news_df)} records")
    
    # Sample recent data
    logger.info("\nSampling recent data...")
    reddit_sample = sample_recent_reddit(
        reddit_df,
        n_posts=INSIGHTS_CONFIG['sampling']['reddit_posts_per_week'],
        weeks_back=INSIGHTS_CONFIG['sampling']['weeks_lookback']
    )
    news_sample = sample_recent_news(
        news_df,
        n_headlines=INSIGHTS_CONFIG['sampling']['news_headlines_per_week'],
        weeks_back=INSIGHTS_CONFIG['sampling']['weeks_lookback']
    )
    
    # Extract forecast summary
    logger.info("\nExtracting forecast trends...")
    forecast_summary = extract_forecast_summary(forecasts)
    
    logger.info(f"  Reddit volume trend: {forecast_summary['reddit_volume_trend']} ({forecast_summary['reddit_volume_change']:+.1f}%)")
    logger.info(f"  Reddit sentiment trend: {forecast_summary['reddit_sentiment_trend']} ({forecast_summary['reddit_sentiment_change']:+.1f}%)")
    logger.info(f"  Coverage ratio: {forecast_summary['coverage_ratio']:.1f}x")
    
    # Determine week range
    week_end = pd.to_datetime(reddit_df['DATE'].max())
    week_start = week_end - pd.Timedelta(weeks=INSIGHTS_CONFIG['sampling']['weeks_lookback'])
    
    logger.info(f"\nAnalyzing week: {week_start.date()} to {week_end.date()}")
    
    # Generate insights
    logger.info("\nGenerating insights with Claude 3.5 Haiku...")
    insights = call_api(
        reddit_sample=reddit_sample,
        news_sample=news_sample,
        forecast_summary=forecast_summary,
        week_start=week_start.strftime('%Y-%m-%d'),
        week_end=week_end.strftime('%Y-%m-%d')
    )
    
    # Parse structure
    logger.info("\nParsing insights structure...")
    structured = parse_insights_sections(insights['text'])
    
    # Prepare output
    insights_data = {
        'week_start': week_start.strftime('%Y-%m-%d'),
        'week_end': week_end.strftime('%Y-%m-%d'),
        'generated_at': timestamp,
        'full_text': insights['text'],
        'sections': structured,
        'metadata': insights['metadata'],
        'data_summary': {
            'reddit_posts_sampled': len(reddit_sample),
            'news_headlines_sampled': len(news_sample),
            'forecast_horizon_weeks': 13,
            'forecast_summary': forecast_summary
        }
    }
    
    # Save
    logger.info("\nSaving insights...")
    save_insights(insights_data, timestamp)

    # Total training time
    elapsed = (datetime.now() - start_time).total_seconds()

    return insights, timestamp, elapsed

def main():
    try:
        # Train Model
        logger.info("Training Insights Model...")
        insights, timestamp, elapsed = train()
        
        # Summary
        logger.info("SUMMARY:")
        logger.info(f"  Timestamp: {timestamp}")
        logger.info(f"  Duration: {elapsed:.1f} seconds")
        logger.info(f"  Cost: ${insights['metadata']['cost_estimate']:.4f}")
        logger.info(f"  Tokens: {insights['metadata']['total_tokens']:,}")
        
        return 0
        
    except Exception as e:
        logger.error(f"\nInsights generation failed: {str(e)}", exc_info=True)
        return 1

if __name__ == "__main__":
    sys.exit(main())