import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.routers import clustering_router, forecasting_router, insights_router, recommendations_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="MindPulse API",
    version="1.0.0",
    description="Mental health analytics API providing forecasting, clustering, LLM insights, and personalized resource recommendations."
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
app.include_router(clustering_router, prefix="/api/clustering", tags=["clustering"])
app.include_router(forecasting_router, prefix="/api/forecasting", tags=["forecasting"])
app.include_router(insights_router, prefix="/api/insights", tags=["insights"])
app.include_router(recommendations_router, prefix="/api/recommendations", tags=["recommendations"])

@app.get("/")
def root():
    return {
        "message": "MindPulseAI API",
        "version": "1.0.0",
        "docs": "/docs",
    }

@app.get("/health")
def health():
    return {"status": "healthy"}