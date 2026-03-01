import json
import os 
import boto3
import logging
from dotenv import load_dotenv
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
    )

S3_BUCKET = "mental-health-project-pipeline"

def load_from_s3(key: str) -> dict:
    """Load and parse a JSON file from S3."""
    try:
        s3 = get_s3_client()
        file = s3.get_object(Bucket=S3_BUCKET, Key=key)
        return json.loads(file["Body"].read().decode("utf-8"))
    except Exception as e:
        logger.error(f"Error loading {key} from S3: {e}", exc_info=True)
        raise FileNotFoundError(f"{key} not found in S3 bucket: {S3_BUCKET}") from e