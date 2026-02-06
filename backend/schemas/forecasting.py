from pydantic import BaseModel
from typing import Dict, List

class ForecastPoint(BaseModel):
    """Single forecast prediction point."""
    ds: str 
    yhat: float
    yhat_lower: float
    yhat_upper: float

class Metrics(BaseModel):
    """Model performance metrics."""
    mae: float
    rmse: float
    mape: float
    mdape: float
    coverage: float

class TopicForecast(BaseModel):
    """Forecast for a single topic."""
    predictions: List[ForecastPoint] 
    metrics: Metrics  
    training_weeks: int
    model_type: str
    frequency: str

class ProphetConfig(BaseModel):
    """Prophet model configuration."""
    changepoint_prior_scale: float
    seasonality_prior_scale: float
    seasonality_mode: str
    weekly_seasonality: bool
    yearly_seasonality: bool
    daily_seasonality: bool
    confidence_interval: float

class DataSource(BaseModel):
    """Information about a single data source."""
    date_range: str
    weeks: int

class ForecastMetadata(BaseModel):
    """Metadata about the forecast generation."""
    forecast_horizon_weeks: int
    forecast_horizon_days: int
    training_timestamp: str
    prophet_config: ProphetConfig  
    data_sources: Dict[str, DataSource]  

class ForecastsResponse(BaseModel):
    """Complete forecasting results."""
    reddit_volume: TopicForecast
    reddit_sentiment: TopicForecast
    news_volume: TopicForecast
    news_sentiment: TopicForecast
    metadata: ForecastMetadata  