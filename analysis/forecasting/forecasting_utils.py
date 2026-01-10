import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize VADER analyzer
_sentiment_analyzer = SentimentIntensityAnalyzer()

def get_sentiment(text):
    """Calculate VADER compound sentiment score for a single text."""
    # Return 0.0 if null or empty text
    if pd.isna(text) or text == '':
        return 0.0
    
    try:
        # Return compound score (normalized, weighted composite score, from -1 to +1)
        scores = _sentiment_analyzer.polarity_scores(str(text))
        return scores['compound']
    except Exception as e:
        logger.warning(f"Error calculating sentiment: {e}")
        return 0.0


def get_aggregated_sentiment(texts, delimiter):
    """Calculate average sentiment for multiple texts in a single string."""
    # Return 0.0 if null or empty text
    if pd.isna(texts) or texts == '':
        return 0.0
    
    try:
        # Split texts by delimiter
        text_list = str(texts).split(delimiter)
        
        # Calculate sentiment for each text in text_list
        sentiments = []
        for text in text_list:
            text = text.strip()
            if text and len(text) > 10:
                score = get_sentiment(text)
                sentiments.append(score)
        
        # Return average (mean) sentiment or 0.0
        return np.mean(sentiments) if sentiments else 0.0 
    
    except Exception as e:
        logger.warning(f"Error calculating aggregated sentiment: {e}")
        return 0.0


def add_sentiment_column(df, text_column, sentiment_column='sentiment', is_aggregated=False, delimiter=';'):
    """Add sentiment analysis column to a DataFrame."""
    # If text is aggregated, use get_aggregated_sentiment
    if is_aggregated:
        logger.info(f"  Using aggregated mode with delimiter '{delimiter}'")
        df[sentiment_column] = df[text_column].apply(
            lambda x: get_aggregated_sentiment(x, delimiter)
        )
    else: # Otherwise, just get_sentiment
        logger.info(f"  Using single-text mode")
        df[sentiment_column] = df[text_column].apply(get_sentiment)
    
    logger.info(f"  Sentiment column '{sentiment_column}' added")
    logger.info(f"  Mean: {df[sentiment_column].mean():.3f}, Std: {df[sentiment_column].std():.3f}")
    
    return df

def aggregate_to_weekly(df, date_col='date', agg_dict=None):
    """
    Aggregate time series data to weekly frequency.
    Handles irregular ingestion patterns (daily testing + weekly production) by grouping all data within each week into a single observation.
    """
    df = df.copy()
    
    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Create week start date (Monday of each week)
    df['WEEK_START'] = df[date_col] - pd.to_timedelta(df[date_col].dt.dayofweek, unit='d')
    
    # Default aggregation functions if not provided
    if agg_dict is None:
        agg_dict = {}
        for col in df.columns:
            if col not in [date_col, 'WEEK_START']:
                # Intelligent defaults based on column name
                if 'volume' in col.lower() or 'count' in col.lower():
                    agg_dict[col] = 'sum'  # Sum counts/volumes
                elif 'sentiment' in col.lower() or 'score' in col.lower() or 'avg' in col.lower():
                    agg_dict[col] = 'mean'  # Average sentiments/scores
                else:
                    agg_dict[col] = 'mean'  # Default to mean

    # Aggregate by week
    weekly_df = df.groupby('WEEK_START').agg(agg_dict).reset_index()
    weekly_df.rename(columns={'WEEK_START': date_col}, inplace=True)
    
    return weekly_df

def fill_missing_weeks(df, date_col='date'):
    """
    Fill missing weeks in time series using linear interpolation.
    Linear interpolation assumes gradual changes between known points, which is more conservative than assuming sudden jumps or constant values.
    """
    df = df.copy()

    # Ensure date column is datetime
    df[date_col] = pd.to_datetime(df[date_col])
    
    # Get date range
    min_date = df[date_col].min()
    max_date = df[date_col].max()
    
    # Check for gaps
    date_diffs = df[date_col].diff().dt.days.dropna()
    max_gap = date_diffs.max()
    
    if max_gap <= 7:
        logger.info(f"No missing weeks detected (max gap: {max_gap:.0f} days)")
        return df
    
    # Create complete weekly range (every Monday)
    full_date_range = pd.date_range(
        start=min_date,
        end=max_date,
        freq='W-MON'
    )
    
    # Reindex to full range (creates NaN for missing weeks)
    df_filled = df.set_index(date_col).reindex(full_date_range)
    
    # Interpolate missing values
    df_filled = df_filled.interpolate(method='linear')
    
    # Reset index
    df_filled = df_filled.reset_index()
    df_filled.rename(columns={'index': date_col}, inplace=True)
    
    return df_filled

def train_prophet(df, pred_col, model_name, config):
    """
    Train Facebook's Prophet model on time-series data.
    Automatically detects weekly vs daily frequency.
    """
    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_df = df[['date', pred_col]].copy()
    prophet_df.columns = ['ds', 'y']

    # Drop NA values
    prophet_df = prophet_df.dropna()

    # Sort by date
    prophet_df = prophet_df.sort_values('ds')

    # Detect frequency (weekly vs daily)
    date_diffs = prophet_df['ds'].diff().dt.days.dropna()
    median_gap = date_diffs.median()
    
    if median_gap >= 6:  # Weekly data (7 days ± 1)
        freq = 'W'
        freq_name = 'weekly'
    else:  # Daily data
        freq = 'D'
        freq_name = 'daily'

    # Log info
    logger.info(f"\nTraining Prophet model: {model_name}")
    logger.info(f"  Training data: {len(prophet_df)} observations")
    logger.info(f"  Frequency: {freq_name} (median gap: {median_gap:.0f} days)")
    logger.info(f"  Date range: {prophet_df['ds'].min()} to {prophet_df['ds'].max()}")
    logger.info(f"  Value range: {prophet_df['y'].min():.2f} to {prophet_df['y'].max():.2f}")

    # Initialize model with config
    model = Prophet(
        changepoint_prior_scale=config.get('changepoint_prior_scale', 0.05),
        seasonality_prior_scale=config.get('seasonality_prior_scale', 10.0),
        seasonality_mode=config.get('seasonality_mode', 'additive'),
        weekly_seasonality=config.get('weekly_seasonality', True),
        yearly_seasonality=config.get('yearly_seasonality', False),
        daily_seasonality=config.get('daily_seasonality', False),
        interval_width=config.get('confidence_interval', 0.95),
        growth=config.get('growth', 'linear')
    )
    
    # Fit model
    model.fit(prophet_df)
    logger.info('  Model trained successfully!')

    return model, prophet_df, freq


def hyperparameter_search(df, value_col, param_grid, model_name):
    """
    Hyperparameter search using in-sample MAE.
    Use only when you have 20+ weeks of data.
    """
    logger.info(f"\nSearching hyperparameters for {model_name}...")
    
    results = []
    prophet_df = df[['date', value_col]].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df = prophet_df.dropna().sort_values('ds')
    
    for changepoint in param_grid.get('changepoint_prior_scale', [0.05]):
        for seasonality in param_grid.get('seasonality_prior_scale', [10.0]):
            model = Prophet(
                changepoint_prior_scale=changepoint,
                seasonality_prior_scale=seasonality,
                weekly_seasonality=True,
                yearly_seasonality=False,
                daily_seasonality=False
            )
            
            model.fit(prophet_df)
            forecast = model.predict(prophet_df)
            
            # In-sample error
            mae = np.mean(np.abs(prophet_df['y'] - forecast['yhat']))
            
            results.append({
                'changepoint_prior_scale': changepoint,
                'seasonality_prior_scale': seasonality,
                'in_sample_mae': mae
            })
    
    results_df = pd.DataFrame(results)
    best = results_df.loc[results_df['in_sample_mae'].idxmin()]
    
    logger.info(f"  Best parameters: changepoint={best['changepoint_prior_scale']}, seasonality={best['seasonality_prior_scale']}")
    logger.info(f"  Best MAE: {best['in_sample_mae']:.3f}")
    
    return dict(best)


def predict_prophet(model, periods, model_name, freq='W'):
    """Generate future predictions using Facebook's Prophet model."""
    # Define frequency name
    freq_name = 'weeks' if freq == 'W' else 'days'
    logger.info(f"\nGenerating {periods}-{freq_name} forecast for {model_name}...")
    
    # Create future dataframe with correct frequency
    future = model.make_future_dataframe(periods=periods, freq=freq)
    
    # Generate predictions
    forecast = model.predict(future)
    
    logger.info(f"  Forecast generated: {len(forecast)} total points")
    
    # Check for negative predictions (should not happen for counts)
    if 'volume' in model_name.lower():
        neg_count = (forecast['yhat'] < 0).sum()
        if neg_count > 0:
            logger.warning(f"  {neg_count} negative predictions detected - clamping to 0")
            forecast['yhat'] = forecast['yhat'].clip(lower=0)
            forecast['yhat_lower'] = forecast['yhat_lower'].clip(lower=0)
    
    return forecast


def plot_forecast(model, forecast, title, figsize=(14,6)):
    """Plot Prophet forecast with historical data and confidence intervals."""
    model.plot(forecast, figsize=figsize)
    fig = plt.gcf() 
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Value', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return fig


def plot_components(model, forecast, figsize=(14,8)):
    """Plot Prophet forecast components (trend, seasonality, etc.)"""
    model.plot_components(forecast, figsize=figsize)
    fig = plt.gcf()  
    plt.tight_layout()
    
    return fig


def evaluate_forecast(prophet_df, forecast):
    """
    Evaluate forecast performance on historical data without cross-validation.
    Use when data is too limited for Prophet's cross_validation.
    """
    # Merge actual and predicted values for historical period
    comparison = prophet_df.merge(
        forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], 
        on='ds', 
        how='inner'
    )
    
    # Calculate error metrics
    errors = comparison['y'] - comparison['yhat']
    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors**2))
    
    # MAPE (avoid division by zero)
    non_zero_mask = comparison['y'] != 0
    mape = np.mean(np.abs(errors[non_zero_mask] / comparison.loc[non_zero_mask, 'y'])) * 100
    
    # MDAPE (Median Absolute Percent Error)
    mdape = np.median(np.abs(errors[non_zero_mask] / comparison.loc[non_zero_mask, 'y'])) * 100
    
    # Coverage: % of actual values within prediction intervals
    within_interval = (
        (comparison['y'] >= comparison['yhat_lower']) & 
        (comparison['y'] <= comparison['yhat_upper'])
    )
    coverage = within_interval.mean()
    
    metrics = {
        'mae': float(mae),
        'rmse': float(rmse),
        'mape': float(mape),
        'mdape': float(mdape),
        'coverage': float(coverage)
    }
    
    logger.info(f"\nForecast Evaluation (In-Sample):")
    logger.info(f"  MAE:      {mae:.3f}")
    logger.info(f"  RMSE:     {rmse:.3f}")
    logger.info(f"  MAPE:     {mape:.2f}%")
    logger.info(f"  MDAPE:    {mdape:.2f}%")
    logger.info(f"  Coverage: {coverage:.1%} (target: 95%)")
    
    # Warning for high MAPE
    if mape > 100:
        logger.warning(f"      MAPE very high - likely due to small actual values")
        logger.warning(f"      Focus on MAE and MDAPE instead")
    
    return metrics


def evaluate_forecast_CV(model, initial_days, period_days, horizon_days):
    """
    Evaluate forecast using Prophet's cross-validation.
    Use when you have enough data (40+ weeks recommended).
    """
    logger.info(f"\nRunning cross-validation...")
    logger.info(f"  Initial: {initial_days} days")
    logger.info(f"  Period: {period_days} days")
    logger.info(f"  Horizon: {horizon_days} days")
    
    # Run cross-validation
    df_cv = cross_validation(
        model,
        initial=f'{initial_days} days',
        period=f'{period_days} days',
        horizon=f'{horizon_days} days'
    )
    
    logger.info(f"  Generated {len(df_cv)} prediction points")
    
    # Calculate performance metrics
    metrics_df = performance_metrics(df_cv, rolling_window=0.1)
    
    # Summary statistics
    summary = {
        'mae': float(metrics_df['mae'].mean()),
        'rmse': float(metrics_df['rmse'].mean()),
        'mape': float(metrics_df['mape'].mean()),
        'mdape': float(metrics_df['mdape'].mean()),
        'coverage': float(metrics_df['coverage'].mean())
    }
    
    logger.info(f"\nCross-Validation Results (Average):")
    logger.info(f"  MAE:      {summary['mae']:.3f}")
    logger.info(f"  RMSE:     {summary['rmse']:.3f}")
    logger.info(f"  MAPE:     {summary['mape']:.2f}%")
    logger.info(f"  MDAPE:    {summary['mdape']:.2f}%")
    logger.info(f"  Coverage: {summary['coverage']:.1%}")
    
    return metrics_df, summary


def evaluate_prophet(model, prophet_df, forecast, use_cv=False, cv_params=None):
    """
    Evaluate forecast performance using Prophet.
    Automatically chooses between in-sample and cross-validation.
    """
    # Check if we have enough data for CV
    data_days = (prophet_df['ds'].max() - prophet_df['ds'].min()).days
    
    if use_cv and cv_params and data_days >= cv_params.get('initial_days', 180) * 2:
        _, metrics = evaluate_forecast_CV(
            model=model,
            initial_days=cv_params['initial_days'],
            period_days=cv_params['period_days'],
            horizon_days=cv_params['horizon_days']
        )
    else:
        if use_cv:
            logger.warning(f"Insufficient data for CV ({data_days} days). Using in-sample evaluation.")
        metrics = evaluate_forecast(prophet_df, forecast)
    
    return metrics


def get_future_predictions(forecast, prophet_df, future_periods=None):
    """Extract only future predictions (not historical fit)."""
    # Get the last training date
    last_training_date = prophet_df['ds'].max()
    
    # Filter forecast to only dates AFTER training period
    future_mask = forecast['ds'] > last_training_date
    future_forecast = forecast[future_mask].copy()
    
    # Limit to requested number of periods if specified
    if future_periods:
        future_forecast = future_forecast.head(future_periods)
    
    logger.info(f"  Extracted {len(future_forecast)} future prediction points")
    logger.info(f"  Future period: {future_forecast['ds'].min()} to {future_forecast['ds'].max()}")
    
    return future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]