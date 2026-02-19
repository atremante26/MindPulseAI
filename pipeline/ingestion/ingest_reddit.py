from pipeline.ingestion import BaseIngestor, sentiment_analyzer
import os
from datetime import datetime, timezone, timedelta
import logging
import praw
import pandas as pd

# Configure logging
logger = logging.getLogger(__name__)

class RedditIngestor(BaseIngestor):
    """
    Ingestor for mental health posts from Reddit.

    Fetches hot posts from mental health subreddits from the past 7 days.

    Attributes:
        reddit (praw.Reddit): Authenticated Reddit API client
        subreddits (list): Target subreddits for mental health content    
    """
    def __init__(self):
        super().__init__()
        self.reddit = praw.Reddit(
            client_id=os.getenv("REDDIT_CLIENT_ID"),
            client_secret=os.getenv("REDDIT_CLIENT_SECRET"),
            user_agent=os.getenv("REDDIT_USER_AGENT")
        )
        self.subreddits = ["mentalhealth", "depression", "anxiety"]
    
    def load_data(self) -> pd.DataFrame:
        """
        Fetch recent hot posts from mental health subreddits.

        Retrieves the top 50 hot posts from each configured subreddit, then filters them to only include posts from past 7 days.
        
        :return: pd.DataFrame: Reddit posts.
        """
        try:
            posts = []
            # Fetch hot posts from each target subreddit
            for sub in self.subreddits:
                for post in self.reddit.subreddit(sub).hot(limit=50):
                    posts.append({
                        "subreddit": sub,
                        "title": post.title,
                        "score": post.score, # Upvotes
                        "created_utc": post.created_utc,
                        "url": post.url,
                        "selftext": post.selftext[:500], # Truncate long posts
                        "num_comments": post.num_comments
                    })

            # Filter to posts from last 7 days
            last_week = datetime.now(timezone.utc) - timedelta(days=7)
            posts = [
                p for p in posts
                if datetime.fromtimestamp(p["created_utc"], tz=timezone.utc) > last_week
            ]
            return pd.DataFrame(posts)
        
        except Exception as e:
            logger.error(f"Reddit ingestion failed: {e}")
            raise 

    
    def process_data(self, df) -> pd.DataFrame:
        """
        Proccess Reddit posts.
        
        :param pd.DataFrame df: Raw Reddit posts.
        :return: pd.Dataframe: Processed Reddit posts.
        """
        # Date processing
        df["created_utc"] = pd.to_datetime(df["created_utc"], unit="s", utc=True)
        df["date"] = df["created_utc"].dt.strftime("%Y-%m-%d")

        # Analyze sentiment
        logger.info(f"Calculating sentiment for {len(df)} Reddit posts...")
        df["combined_text"] = df["title"] + ' ' + df["selftext"].fillna('')
        sentiments = sentiment_analyzer.analyze_batch(df['combined_text'].tolist())
        df["sentiment"] = sentiments

        return df[["subreddit", "date", "title", "selftext", "score", "num_comments", "sentiment"]].rename(
            columns={"selftext": "text", "num_comments": "comments"}
        )

if __name__ == "__main__":
    reddit_ingestor = RedditIngestor()
    reddit_ingestor.run("reddit", "reddit_suite", True, True)