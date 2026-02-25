from pydantic import BaseModel
from typing import Dict, List

class WeeklyInsightsSections(BaseModel):
    key_themes: str
    sentiment_analysis: str
    coverage_gap_insights: str
    concerning_patterns: str
    recommendations: str

class WeeklyInsightsMetadata(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cost_estimate: float

class WeeklyForecastSummary(BaseModel):
    reddit_volume_avg: float
    reddit_volume_trend: str
    reddit_volume_change: float
    reddit_sentiment_avg: float
    reddit_sentiment_trend: str
    reddit_sentiment_change: float
    news_volume_avg: float
    news_volume_trend: str
    news_volume_change: float
    news_sentiment_avg: float
    news_sentiment_trend: str
    news_sentiment_change: float
    coverage_ratio: float

class WeeklyDataSummary(BaseModel):
    reddit_posts_sampled: int
    news_headlines_sampled: int
    forecast_horizon_weeks: int
    forecast_summary: WeeklyForecastSummary

class WeeklyInsightsResponse(BaseModel):
    week_start: str
    week_end: str
    generated_at: str
    full_text: str
    sections: WeeklyInsightsSections
    metadata: WeeklyInsightsMetadata
    data_summary: WeeklyDataSummary

class DatapointSurroundingWeeks(BaseModel):
    date: str
    value: float

class DatapointRequest(BaseModel):
    metric_name: str
    week_date: str
    value: float
    baseline: float
    confidence_lower: float
    confidence_upper: float
    surrounding_weeks: List[DatapointSurroundingWeeks]

class DatapointResponseMetadata(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    cost_estimate: float
    metric: str
    week: str
    value: float

class DatapointResponse(BaseModel):
    text: str
    metadata: DatapointResponseMetadata