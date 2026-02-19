from pipeline.ingestion import BaseIngestor, sentiment_analyzer
import logging
import pandas as pd
import requests
import os
from datetime import datetime, timedelta

# Configure logging
logger = logging.getLogger(__name__)

class NewsIngestor(BaseIngestor):
    """
    Ingestor for mental health news articles from NewsAPI.
    
    Fetches articles from the past 7 days matching mental health keywords.
    
    Attributes:
        api_key (str): NewsAPI authentication key from environment
        base_url (str): NewsAPI endpoint for article search
    """
    def __init__(self):
        super().__init__()
        self.api_key = os.getenv('NEWS_API_KEY')
        self.base_url = 'https://newsapi.org/v2/everything'
        
    def load_data(self) -> pd.DataFrame:
        """
        Fetch mental health news articles from NewsAPI.

        Queries NewsAPI for articles from the past 7 days matching mental health keywords. Uses free tier limit of 100 articles per request.
        
        :return: pd.DataFrame: Raw article data.
        """
        try:
            logger.info("Fetching mental health news from News API...")
            
            # Configure search parameters
            params = {
                'q': '("mental health" OR anxiety OR depression OR "suicide prevention")',
                'language': 'en',
                'sortBy': 'publishedAt',
                'from': (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d'),
                'to': datetime.now().strftime('%Y-%m-%d'),
                'pageSize': 100,  # Max allowed by free tier
                'apiKey': self.api_key
            }
            
            # Call API
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Check API status
            if data.get('status') != 'ok':
                raise ValueError(f"News API error: {data.get('message', 'Unknown error')}")
            
            articles = data.get('articles', [])
            logger.info(f"Fetched {len(articles)} articles from News API on {self.today}")
            
            # Convert to DataFrame
            df = pd.DataFrame(articles)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch News API data on {self.today}: {e}")
            raise

    def process_data(self, df) -> pd.DataFrame:
        """Process and aggregate raw news articles.
        
        :param pd.DataFrame df: Raw article data.
        :return: pd.Dataframe: Aggregated data with sentiment.
        """
        processed_df = df.copy()
        
        # Parse published date
        processed_df['publishedAt'] = pd.to_datetime(
            processed_df['publishedAt'], 
            errors='coerce'
        )
        processed_df['date'] = processed_df['publishedAt'].dt.date
        
        # Extract source name
        processed_df['source_name'] = processed_df['source'].apply(
            lambda x: x.get('name', 'Unknown') if isinstance(x, dict) else 'Unknown'
        )

        # Analyze sentiment
        logger.info(f"Calculating sentiment for {len(processed_df)} News articles...")
        sentiments = sentiment_analyzer.analyze_batch(processed_df["title"].fillna('').tolist())
        df["sentiment"] = sentiments
        
        # Group by date and aggregate
        daily_agg = processed_df.groupby('date').agg({
            'title': ['count', lambda x: ' | '.join(x.head(5))],  # Count + sample headlines
            'source_name': lambda x: ', '.join(set(x)),  # Unique sources
            'sentiment': 'mean'
        }).reset_index()
        
        # Flatten column names
        daily_agg.columns = ['date', 'article_count', 'sample_headlines', 'sources', 'sentiment']

        # Round sentiment
        daily_agg['sentiment'] = daily_agg['sentiment'].round(4)

        # Convert date to string for JSON serialization
        daily_agg['date'] = daily_agg['date'].astype(str)
        
        # Sort by date
        daily_agg = daily_agg.sort_values('date')
        
        logger.info(f"Processed {len(daily_agg)} days of news data")
        logger.info(f"Date range: {daily_agg['date'].min()} to {daily_agg['date'].max()}")
        logger.info(f"Total articles: {daily_agg['article_count'].sum()}")
        
        return daily_agg

    
if __name__ == "__main__":
    news_ingestor = NewsIngestor()
    news_ingestor.run("news", "news_suite", True, True)