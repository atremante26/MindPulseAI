import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_prophet(df, pred_col, model_name, config):
    """Train Facebook's Prophet model on time-series data."""

    # Prepare data for Prophet (requires 'ds' and 'y' columns)
    prophet_df = df[['date', pred_col]].copy()
    prophet_df.columns = ['ds', 'y']

    # Drop NA values
    prophet_df = prophet_df.dropna()

    # Sort by date
    prophet_df = prophet_df.sort_values('ds')

    # Log info
    logger.info(f"\nTraining Prophet model: {model_name}")
    logger.info(f"  Training data: {len(prophet_df)} observations")
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
        interval_width=config.get('confidence_interval', 0.95)
    )
    
    # Fit model
    model.fit(prophet_df)
    logger.info('Model Trained Succesfully!')

    return model, prophet_df

def hyperparameter_search(df, value_col, param_grid, model_name):
    """
    Hyperparameter search using in-sample MAE.
    
    Note: Use only when you have 20+ weeks of data.
    For datasets <20 weeks, stick with default parameters.
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

def predict_prophet(model, periods_days, model_name):
    """Generate future predictions using Facebook's Prophet model."""
    logger.info(f"\nGenerating {periods_days}-day forecast for {model_name}...")
    
    # Create future dataframe
    future = model.make_future_dataframe(periods=periods_days, freq='D')
    
    # Generate predictions
    forecast = model.predict(future)
    
    logger.info(f"  Forecast generated: {len(forecast)} total points")
    
    return forecast

def plot_forecast(model, forecast, title, figsize=(14,6)):
    """Plot Prophet model with historical data and confidence intervals."""
    fig = model.plot(forecast, figsize=figsize)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Value')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_components(model, forecast, figsize=(14,6)):
    """Plot Prophet forecast components (seasonality, trends, etc.)"""
    fig = model.plot_components(forecast, figsize=figsize)
    plt.tight_layout()
    plt.show()

def evaluate_forecast(model, prophet_df, forecast):
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
    metrics_df = performance_metrics(df_cv, rolling_window=0.1)  # 10% window
    
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
    """ Evaluate forecast performance using Facebook's Prophet model."""
    # Check if we have enough data for CV
    data_days = (prophet_df['ds'].max() - prophet_df['ds'].min()).days
    
    if use_cv and cv_params and data_days >= cv_params.get('initial_days', 180) * 2:
        # Use cross-validation if requested and sufficient data
        _, metrics = evaluate_forecast_CV(
            model=model,
            initial_days=cv_params['initial_days'],
            period_days=cv_params['period_days'],
            horizon_days=cv_params['horizon_days']
        )
    else:
        # Use in-sample evaluation
        if use_cv:
            logger.warning(f"Insufficient data for CV ({data_days} days). Using in-sample evaluation.")
        metrics = evaluate_forecast(model, prophet_df, forecast)
    
    return metrics

def get_future_predictions(forecast, future_days=90):
    """Extract only future predictions (not historical fit)."""
    # Get last N rows (future predictions)
    future_forecast = forecast.tail(future_days).copy()
    
    return future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]