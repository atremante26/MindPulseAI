import os
import random
import logging
import pandas as pd
from datetime import timedelta
from anthropic import Anthropic
from dotenv import load_dotenv

from analysis.config.model_config import INSIGHTS_CONFIG
from analysis.insights import get_weekly_insights_prompt, get_datapoint_insight_prompt

# Load environment variables
load_dotenv()

# Configure logging
logger = logging.getLogger(__name__)

# Initialize Anthropic client
client = Anthropic(api_key=os.getenv('ANTHROPIC_API_KEY'))

def sample_recent_reddit(reddit_df, n_posts=30, weeks_back=1):
    """
    Sample recent high-engagement Reddit posts for insights generation.
    
    Parameters:
    - reddit_df: Full Reddit dataset with DATE, TEXT, SCORE, COMMENTS, sentiment
    - n_posts: Number of posts to sample
    - weeks_back: How many weeks back to look
    
    Returns:
    - List of dicts with post data
    """
    # Ensure DATE is datetime
    reddit_df['DATE'] = pd.to_datetime(reddit_df['DATE'])
    
    # Get recent posts
    cutoff_date = reddit_df['DATE'].max() - timedelta(weeks=weeks_back)
    recent = reddit_df[reddit_df['DATE'] > cutoff_date].copy()

    # Check if no posts found
    if len(recent) == 0:
        logger.warning("No recent Reddit posts found")
        return []
    
    logger.info(f"Found {len(recent)} Reddit posts from last {weeks_back} week(s)")
    
    # Calculate engagement score
    recent['engagement'] = recent['SCORE'] + (recent['COMMENTS'] * 2)  # Weight comments higher
    
    # Sample top posts by engagement
    if len(recent) > n_posts:
        sampled = recent.nlargest(n_posts, 'engagement')
    else:
        sampled = recent
    
    # Format for prompt
    posts = []
    for _, row in sampled.iterrows():
        posts.append({
            'title': row['TEXT'],
            'sentiment': row.get('sentiment', 0.0),
            'score': int(row['SCORE']),
            'comments': int(row['COMMENTS']),
            'date': row['DATE'].strftime('%Y-%m-%d')
        })
    
    logger.info(f"Sampled {len(posts)} high-engagement Reddit posts")
    return posts


def sample_recent_news(news_df, n_headlines=15, weeks_back=1):
    """
    Sample recent news headlines for insights generation.
    
    Parameters:
    - news_df: Full news dataset with DATE, SAMPLE_HEADLINES
    - n_headlines: Number of headlines to sample
    - weeks_back: How many weeks back to look
    
    Returns:
    - List of headline strings
    """
    # Ensure DATE is datetime
    news_df['DATE'] = pd.to_datetime(news_df['DATE'])
    
    # Get recent articles
    cutoff_date = news_df['DATE'].max() - timedelta(weeks=weeks_back)
    recent = news_df[news_df['DATE'] > cutoff_date]

    # Check if no headlines found
    if len(recent) == 0:
        logger.warning("No recent news found")
        return []
    
    logger.info(f"Found {len(recent)} news records from last {weeks_back} week(s)")
    
    # Extract all headlines
    headlines = []
    for _, row in recent.iterrows():
        sample_headlines = str(row['SAMPLE_HEADLINES'])
        
        # Parse delimited headlines
        if '||' in sample_headlines:
            parsed = sample_headlines.split('||')
        elif ';' in sample_headlines:
            parsed = sample_headlines.split(';')
        elif '|' in sample_headlines:
            parsed = sample_headlines.split('|')
        else:
            parsed = [sample_headlines]
        
        # Clean and add
        for h in parsed:
            h = h.strip()
            if h and len(h) > 10:  # Filter out empty or very short strings
                headlines.append(h)
    
    # Sample randomly to get diversity
    if len(headlines) > n_headlines:
        headlines = random.sample(headlines, n_headlines)
    
    logger.info(f"Sampled {len(headlines)} news headlines")
    return headlines


def extract_forecast_summary(forecasts_json):
    """
    Extract key metrics from forecasts for insights prompt.
    
    Parameters:
    - forecasts_json: Dict from latest_forecasts.json
    
    Returns:
    - Dict with summary metrics and trends
    """
    # Extract predictions
    reddit_vol = forecasts_json['reddit_volume']['predictions']
    reddit_sent = forecasts_json['reddit_sentiment']['predictions']
    news_vol = forecasts_json['news_volume']['predictions']
    news_sent = forecasts_json['news_sentiment']['predictions']
    
    # Calculate averages
    reddit_vol_avg = sum(p['yhat'] for p in reddit_vol) / len(reddit_vol)
    reddit_sent_avg = sum(p['yhat'] for p in reddit_sent) / len(reddit_sent)
    news_vol_avg = sum(p['yhat'] for p in news_vol) / len(news_vol)
    news_sent_avg = sum(p['yhat'] for p in news_sent) / len(news_sent)
    
    # Calculate trends (compare first vs last prediction)
    reddit_vol_change = ((reddit_vol[-1]['yhat'] - reddit_vol[0]['yhat']) / reddit_vol[0]['yhat']) * 100
    reddit_sent_change = ((reddit_sent[-1]['yhat'] - reddit_sent[0]['yhat']) / abs(reddit_sent[0]['yhat'])) * 100 if reddit_sent[0]['yhat'] != 0 else 0
    news_vol_change = ((news_vol[-1]['yhat'] - news_vol[0]['yhat']) / news_vol[0]['yhat']) * 100 if news_vol[0]['yhat'] > 0 else 0
    news_sent_change = ((news_sent[-1]['yhat'] - news_sent[0]['yhat']) / abs(news_sent[0]['yhat'])) * 100 if news_sent[0]['yhat'] != 0 else 0
    
    # Determine trends
    def get_trend(change):
        if abs(change) < 5:
            return "stable"
        return "increasing" if change > 0 else "declining"
    
    reddit_vol_trend = get_trend(reddit_vol_change)
    reddit_sent_trend = "improving" if reddit_sent_change > 5 else ("declining" if reddit_sent_change < -5 else "stable")
    news_vol_trend = get_trend(news_vol_change)
    news_sent_trend = "improving" if news_sent_change > 5 else ("declining" if news_sent_change < -5 else "stable")
    
    # Coverage ratio
    coverage_ratio = reddit_vol_avg / news_vol_avg if news_vol_avg > 0 else float('inf')
    
    return {
        'reddit_volume_avg': reddit_vol_avg,
        'reddit_volume_trend': reddit_vol_trend,
        'reddit_volume_change': reddit_vol_change,
        'reddit_sentiment_avg': reddit_sent_avg,
        'reddit_sentiment_trend': reddit_sent_trend,
        'reddit_sentiment_change': reddit_sent_change,
        'news_volume_avg': news_vol_avg,
        'news_volume_trend': news_vol_trend,
        'news_volume_change': news_vol_change,
        'news_sentiment_avg': news_sent_avg,
        'news_sentiment_trend': news_sent_trend,
        'news_sentiment_change': news_sent_change,
        'coverage_ratio': coverage_ratio
    }


def call_api(
    reddit_sample,
    news_sample,
    forecast_summary,
    week_start,
    week_end
):
    """
    Call Claude 3.5 Haiku to generate mental health insights.
    
    Returns:
    - Dict with insights text and metadata

    """
    logger.info("Generating insights with Claude 3.5 Haiku...")
    
    # Build prompt
    prompt = get_weekly_insights_prompt(
        week_start=week_start,
        week_end=week_end,
        reddit_sample=reddit_sample,
        news_sample=news_sample,
        forecast_summary=forecast_summary
    )
    
    # Call Claude API
    try:
        response = client.messages.create(
            model=INSIGHTS_CONFIG['llm']['model'],
            max_tokens=INSIGHTS_CONFIG['llm']['max_tokens'],
            temperature=INSIGHTS_CONFIG['llm']['temperature'],
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        insights_text = response.content[0].text
        
        # Calculate cost 
        input_tokens = response.usage.input_tokens # Input: $1.00 / 1M token
        output_tokens = response.usage.output_tokens # Output: $5.00 / 1M tokens
        cost = (input_tokens / 1_000_000 * 1.00) + (output_tokens / 1_000_000 * 5.00)
        
        logger.info(f"Generated insights ({len(insights_text)} characters)")
        logger.info(f"API usage: {input_tokens} input + {output_tokens} output tokens")
        logger.info(f"Estimated cost: ${cost:.4f}")
        
        return {
            'text': insights_text,
            'metadata': {
                'model': response.model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'total_tokens': input_tokens + output_tokens,
                'cost_estimate': cost
            }
        }
        
    except Exception as e:
        logger.error(f"Claude API call failed: {str(e)}")
        raise


def call_api_datapoint(
    metric_name,
    week_date,
    value,
    baseline,
    confidence_lower,
    confidence_upper,
    surrounding_weeks,
    static_context
):
    """
    Call Claude 3.5 Haiku to generate insight for a specific forecast datapoint.
    
    This is called when a user clicks on a chart point to understand it.
    
    Parameters:
    - metric_name: "Reddit Volume", "Reddit Sentiment", etc.
    - week_date: ISO date string "2026-02-10"
    - value: Forecasted value
    - baseline: Average baseline value
    - confidence_lower/upper: CI bounds
    - surrounding_weeks: List of dicts with surrounding week data
    - static_context: Mental health statistics
    
    Returns:
    - Dict with insight text and metadata
    """
    logger.info(f"Generating datapoint insight: {metric_name}, week {week_date}")
    
    # Calculate percent change
    pct_change = ((value - baseline) / abs(baseline)) * 100 if baseline != 0 else 0
    
    # Build prompt
    prompt = get_datapoint_insight_prompt(
        metric_name=metric_name,
        week_date=week_date,
        value=value,
        baseline=baseline,
        pct_change=pct_change,
        confidence_lower=confidence_lower,
        confidence_upper=confidence_upper,
        surrounding_weeks=surrounding_weeks,
        static_context=static_context
    )
    
    # Call Claude API
    try:
        response = client.messages.create(
            model=INSIGHTS_CONFIG['llm']['model'],
            max_tokens=1024,  # Shorter than weekly insights
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        
        insight_text = response.content[0].text
        
        # Calculate cost
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = (input_tokens / 1_000_000 * 1.00) + (output_tokens / 1_000_000 * 5.00)
        
        logger.info(f"Generated datapoint insight ({len(insight_text)} chars)")
        logger.info(f"Cost: ${cost:.4f}")
        
        return {
            'text': insight_text,
            'metadata': {
                'model': response.model,
                'input_tokens': input_tokens,
                'output_tokens': output_tokens,
                'cost_estimate': cost,
                'metric': metric_name,
                'week': week_date,
                'value': value
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to generate datapoint insight: {e}")
        raise


def prepare_surrounding_weeks_context(forecast_data, target_week_index, window=2):
    """
    Extract surrounding weeks for context.
    
    Parameters:
    - forecast_data: List of dicts with 'ds' and 'yhat' keys (Prophet output)
    - target_week_index: Index of the clicked week
    - window: Number of weeks before/after to include (default: 2)
    
    Returns:
    - List of dicts with week data
    """
    start_idx = max(0, target_week_index - window)
    end_idx = min(len(forecast_data), target_week_index + window + 1)
    
    surrounding = []
    for i in range(start_idx, end_idx):
        if i != target_week_index:  # Don't include the target week itself
            week_data = forecast_data[i]
            surrounding.append({
                'date': week_data['ds'] if isinstance(week_data['ds'], str) else week_data['ds'].strftime('%Y-%m-%d'),
                'value': week_data['yhat']
            })
    
    return surrounding


def parse_insights_sections(insights_text):
    """
    Parse Claude output into structured sections.
    
    Looks for section headers like:
    - **1. KEY THEMES**
    - **2. SENTIMENT ANALYSIS**
    
    Returns:
    - Dict with sections
    """
    sections = {
        'key_themes': '',
        'sentiment_analysis': '',
        'coverage_gap_insights': '',
        'concerning_patterns': '',
        'recommendations': ''
    }
    
    lines = insights_text.split('\n')
    current_section = None
    current_text = []
    
    for line in lines:
        line_upper = line.upper()
        
        # Detect section headers
        if 'KEY THEMES' in line_upper:
            if current_section:
                sections[current_section] = '\n'.join(current_text).strip()
            current_section = 'key_themes'
            current_text = []
        elif 'SENTIMENT ANALYSIS' in line_upper:
            if current_section:
                sections[current_section] = '\n'.join(current_text).strip()
            current_section = 'sentiment_analysis'
            current_text = []
        elif 'COVERAGE GAP' in line_upper:
            if current_section:
                sections[current_section] = '\n'.join(current_text).strip()
            current_section = 'coverage_gap_insights'
            current_text = []
        elif 'CONCERNING PATTERNS' in line_upper:
            if current_section:
                sections[current_section] = '\n'.join(current_text).strip()
            current_section = 'concerning_patterns'
            current_text = []
        elif 'RECOMMENDATIONS' in line_upper:
            if current_section:
                sections[current_section] = '\n'.join(current_text).strip()
            current_section = 'recommendations'
            current_text = []
        elif current_section and line.strip():
            # Add to current section (skip section header itself)
            if not any(marker in line for marker in ['**1.', '**2.', '**3.', '**4.', '**5.']):
                current_text.append(line.strip())
    
    # Last section
    if current_section:
        sections[current_section] = '\n'.join(current_text).strip()
    
    logger.info("Parsed insights into structured sections")
    return sections