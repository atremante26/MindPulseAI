import os
import sys
import json
import boto3
import logging
import pandas as pd
from pathlib import Path
import snowflake.connector
from fastapi import HTTPException
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.backends import default_backend

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Schemas
from backend.schemas import HistoricalResponse, HistoricalPoint
from backend.utils import load_from_s3

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_forecasting_data():
    return load_from_s3("artifacts/latest_forecasts.json")

def snowflake_connection():
    """Establish connection to Snowflake using AWS Parameter store."""
    # Get private key from AWS Parameter Store
    ssm = boto3.client('ssm', region_name='us-east-1')
    response = ssm.get_parameter(
        Name='/mental-health-pipeline/snowflake/private-key',
        WithDecryption=True
    )
    private_key_pem = response['Parameter']['Value'].encode()
    
    # Load the private key
    private_key = serialization.load_pem_private_key(
        private_key_pem,
        password=None,
        backend=default_backend()
    )
    
    # Convert to bytes for Snowflake
    pkb = private_key.private_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    
    # Connect to Snowflake
    conn = snowflake.connector.connect(
        user=os.getenv("SNOWFLAKE_USER"),
        account=os.getenv("SNOWFLAKE_ACCOUNT"),
        warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
        database=os.getenv("SNOWFLAKE_DATABASE"),
        role=os.getenv("SNOWFLAKE_ROLE"),
        private_key=pkb
    )
    
    return conn

def load_from_sql_file(sql_path):
    """Load data from SQL file."""
    try:
        with open(sql_path, 'r') as f:
            query = f.read()

        conn = snowflake_connection()
        try:
            df = pd.read_sql(query, conn)
        finally:
            conn.close()
        
        logger.info(f"Loaded {len(df)} rows from {sql_path.name}")
        return df
    
    except Exception as e:
        logger.error(f"Error executing query from {sql_path}: {e}")
        raise RuntimeError(f"Failed to load data from {sql_path.name}: {e}") from e
    
def load_reddit():
    """Load Reddit historical data."""
    reddit_sql_path = PROJECT_ROOT / "pipeline/snowflake/reddit_sql/reddit_extract.sql"
    return load_from_sql_file(reddit_sql_path)

def load_news():
    """Load News historical data."""
    news_sql_path = PROJECT_ROOT / "pipeline/snowflake/news_sql/news_extract.sql"
    return load_from_sql_file(news_sql_path)

def get_historical():
    """
    Load historical data for both Reddit and News.
    
    :returns: HistoricalResponse with 4 topic time series.
    :raises: RuntimeError: If data loading or aggregation fails.
    """
    try:
        # Load and aggregate Reddit
        reddit_df = load_reddit()
        reddit_df.columns = reddit_df.columns.str.lower()
        reddit_df["date"] = pd.to_datetime(reddit_df["date"])

        reddit_df['week_start'] = reddit_df["date"] - pd.to_timedelta(reddit_df["date"].dt.dayofweek, unit='d')
        reddit_weekly = reddit_df.groupby("week_start").agg(
            volume=('title', 'count'),
            avg_sentiment=('sentiment', 'mean')
        ).reset_index()

        reddit_volume = [
            HistoricalPoint(ds=str(row.week_start), value=row.volume)
            for row in reddit_weekly.itertuples()
        ]
        reddit_sentiment = [
            HistoricalPoint(ds=str(row.week_start), value=row.avg_sentiment)
            for row in reddit_weekly.itertuples()
        ]

        # Load and shape News
        news_df = load_news()
        news_df.columns = news_df.columns.str.lower()
        news_df["date"] = pd.to_datetime(news_df["date"])

        news_volume = [
            HistoricalPoint(ds=str(row.date), value=row.article_count)
            for row in news_df.itertuples()
        ]
        news_sentiment = [
            HistoricalPoint(ds=str(row.date), value=row.sentiment)
            for row in news_df.itertuples()
        ]

        logger.info("Loaded historical data successfully.")
        return HistoricalResponse(
            reddit_volume=reddit_volume,
            reddit_sentiment=reddit_sentiment,
            news_volume=news_volume,
            news_sentiment=news_sentiment
        )

    except Exception as e:
        logger.error(f"Error loading historical data: {e}", exc_info=True)
        raise RuntimeError(f"Failed to load historical data: {e}") from e