"""
PharmacoBench API - Main Application

REST API for drug sensitivity predictions and benchmark results.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

from api.routes import predictions, benchmarks, models
from api.schemas import HealthResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown."""
    logger.info("PharmacoBench API starting up...")
    # Startup: Load models, initialize caches, etc.
    yield
    # Shutdown: Cleanup resources
    logger.info("PharmacoBench API shutting down...")


# Create FastAPI application
app = FastAPI(
    title="PharmacoBench API",
    description="""
    REST API for drug sensitivity prediction and benchmark analysis.

    ## Features

    - **Predictions**: Predict drug sensitivity (IC50) for drug-cell line pairs
    - **Benchmarks**: Access benchmark results across models and split strategies
    - **Models**: List available models and their configurations

    ## Quick Start

    ```python
    import requests

    # Single prediction
    response = requests.post(
        "http://localhost:8000/predict",
        json={"drug_name": "Gefitinib", "cell_line_name": "A549", "model": "xgboost"}
    )
    print(response.json())
    ```

    ## Authentication

    Currently no authentication required (demo mode).
    """,
    version="0.2.0",
    contact={
        "name": "PharmacoBench Team",
        "url": "https://github.com/legomaheggoz-source/PharmacoBench",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
app.include_router(benchmarks.router, prefix="/api/v1", tags=["benchmarks"])
app.include_router(models.router, prefix="/api/v1", tags=["models"])


@app.get("/", response_model=HealthResponse, tags=["health"])
async def root():
    """Root endpoint returning API status."""
    return HealthResponse(
        status="healthy",
        version="0.2.0",
        message="PharmacoBench API is running"
    )


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health_check():
    """Health check endpoint for monitoring."""
    return HealthResponse(
        status="healthy",
        version="0.2.0",
        message="All systems operational"
    )


# Run with: uvicorn api.main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
