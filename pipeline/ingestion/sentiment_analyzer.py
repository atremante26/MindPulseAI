
from transformers import pipeline
import logging
from typing import List

# Configure logging
logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """
    Sentiment Analyzer using DistilBERT transformer model.
    
    Model is loaded once when first used, then reused for all subsequent sentiment calculations.
    
    :returns: Scores between -1 (negative) and +1 (positive).
    """
    
    _instance = None
    _model = None
    
    def __new__(cls):
        """Ensure only one instance exists."""
        # Check if instance exists
        if cls._instance is None:
            # Create new instance
            cls._instance = super().__new__(cls)
            # Load model
            cls._instance._initialize_model()
        return cls._instance
    
    def _initialize_model(self):
        """Load sentiment model once at initialization."""
        logger.info("Loading sentiment analysis model...")

        try:
            self._model = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                device=-1  # CPU (ECS doesn't have GPU)
            )
            logger.info("Sentiment model loaded successfully!")

        except Exception as e:
            logger.error(f"Failed to load sentiment model: {e}")
            raise RuntimeError(f"Could not initialize sentiment analyzer: {e}")
    
    def analyze(self, text: str) -> float:
        """
        Calculate sentiment score for a single text.
        
        :param text: Input text (will be truncated to 512 chars)
            
        :returns: Float between -1 (very negative) and +1 (very positive). Returns 0.0 for empty text or errors
        """
        if not text or len(text.strip()) == 0:
            return 0.0
        
        # Truncate to model's maximum length
        truncated_text = text.strip()[:512]
        
        try:
            result = self._model(truncated_text)[0]
            
            # Convert to -1 to +1 scale
            score = result['score']
            if result['label'] == 'POSITIVE':
                return round(score, 4)
            else:
                return round(-score, 4)
                
        except Exception as e:
            logger.warning(f"Sentiment analysis failed (returning neutral): {e}")
            return 0.0
    
    def analyze_batch(self, texts: List[str], batch_size: int = 32) -> List[float]:
        """
        Batch process multiple texts (much faster than individual calls).
        
        :param texts: List of text strings to analyze
        :param batch_size: Number of texts to process at once (default: 32)
            
        :return: List of sentiment scores (same order as input)
        """
        if not texts:
            return []
        
        # Handle None/empty values
        processed_texts = []
        for text in texts:
            if not text or len(text.strip()) == 0:
                processed_texts.append("")
            else:
                processed_texts.append(text.strip()[:512])
        
        try:
            # Process in batches for efficiency
            all_scores = []
            
            for i in range(0, len(processed_texts), batch_size):
                batch = processed_texts[i:i + batch_size]
                
                # Replace empty strings with single space 
                batch = [text if text else " " for text in batch]
                
                results = self._model(batch)
                
                # Convert to -1 to +1 scale
                scores = []
                for j, result in enumerate(results):
                    # Return 0.0 for originally empty texts
                    if not processed_texts[i + j].strip():
                        scores.append(0.0)
                    elif result['label'] == 'POSITIVE':
                        scores.append(round(result['score'], 4))
                    else:
                        scores.append(round(-result['score'], 4))
                
                all_scores.extend(scores)
            
            return all_scores
            
        except Exception as e:
            logger.error(f"Batch sentiment analysis failed: {e}")
            # Return neutral scores for all on error
            return [0.0] * len(texts)


# Singleton instance
sentiment_analyzer = SentimentAnalyzer()