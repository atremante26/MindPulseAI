import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import clustering, forecasting, insights
from services.clustering_service import ClusteringService

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instances
clustering_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    global clustering_service
    logger.info("Starting up: Loading models...")
    clustering_service = ClusteringService()
    clustering_service.load_models()
    
    yield
    
    # Shutdown
    logger.info("Shutting down...")

app = FastAPI(
    title="MindPulse API",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(clustering.router, prefix="/api/clustering", tags=["clustering"])
app.include_router(forecasting.router, prefix="/api/forecasting", tags=["forecasting"])
app.include_router(insights.router, prefix="/api/insights", tags=["insights"])

@app.get("/")
def root():
    return {
        "message": "MindPulse API",
        "version": "1.0.0",
        "docs": "/docs"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "clustering_loaded": clustering_service.is_loaded() if clustering_service else False
    }