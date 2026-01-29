"""
FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.config import settings
from app.database import init_db

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Football Analytics API",
    description="Hybrid AI-based football match prediction system",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize database and start scheduler on startup."""
    logger.info("Starting Football Analytics API...")
    init_db()
    logger.info("Database initialized")
    
    # Start fixtures scheduler (background thread)
    from app.scheduler import start_scheduler
    start_scheduler()
    logger.info("Fixtures scheduler started")


@app.on_event("shutdown")
async def shutdown_event():
    """Stop scheduler on shutdown."""
    from app.scheduler import stop_scheduler
    stop_scheduler()
    logger.info("Scheduler stopped")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Football Analytics API",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


# Import and include routers
from api.routes import predictions, matches, history, analytics

app.include_router(predictions.router, prefix="/api/predictions", tags=["Predictions"])
app.include_router(matches.router, prefix="/api/matches", tags=["Matches"])
app.include_router(history.router, prefix="/api/history", tags=["History"])
app.include_router(analytics.router, prefix="/api/analytics", tags=["Analytics"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=True
    )
