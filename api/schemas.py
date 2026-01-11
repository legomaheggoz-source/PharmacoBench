"""
API Schemas

Pydantic models for request/response validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum


class ModelType(str, Enum):
    """Available model types."""
    RIDGE = "ridge"
    ELASTICNET = "elasticnet"
    RANDOM_FOREST = "random_forest"
    XGBOOST = "xgboost"
    LIGHTGBM = "lightgbm"
    MLP = "mlp"
    GRAPHDRP = "graphdrp"
    DEEPCDR = "deepcdr"


class SplitStrategy(str, Enum):
    """Available split strategies."""
    RANDOM = "random"
    DRUG_BLIND = "drug_blind"
    CELL_BLIND = "cell_blind"
    DISJOINT = "disjoint"


# =============================================================================
# Health & Status
# =============================================================================

class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    message: str


# =============================================================================
# Predictions
# =============================================================================

class PredictionRequest(BaseModel):
    """Single prediction request."""
    drug_name: str = Field(..., description="Name of the drug")
    cell_line_name: str = Field(..., description="Name of the cell line")
    model: ModelType = Field(default=ModelType.XGBOOST, description="Model to use")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "drug_name": "Gefitinib",
                    "cell_line_name": "A549",
                    "model": "xgboost"
                }
            ]
        }
    }


class PredictionResponse(BaseModel):
    """Single prediction response."""
    drug_name: str
    cell_line_name: str
    model: str
    predicted_ln_ic50: float
    confidence_interval: Dict[str, float]
    sensitivity_class: str  # "Sensitive", "Intermediate", "Resistant"


class BatchPredictionRequest(BaseModel):
    """Batch prediction request."""
    pairs: List[Dict[str, str]] = Field(
        ...,
        description="List of drug-cell line pairs",
        min_length=1,
        max_length=1000
    )
    model: ModelType = Field(default=ModelType.XGBOOST)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "pairs": [
                        {"drug_name": "Gefitinib", "cell_line_name": "A549"},
                        {"drug_name": "Erlotinib", "cell_line_name": "MCF7"}
                    ],
                    "model": "xgboost"
                }
            ]
        }
    }


class BatchPredictionResponse(BaseModel):
    """Batch prediction response."""
    predictions: List[PredictionResponse]
    model: str
    count: int


# =============================================================================
# Benchmarks
# =============================================================================

class BenchmarkResult(BaseModel):
    """Single benchmark result."""
    model: str
    split_strategy: str
    rmse: float
    mae: float
    r_squared: float
    pearson: float
    spearman: float
    train_time_seconds: Optional[float] = None


class BenchmarkQuery(BaseModel):
    """Benchmark query parameters."""
    models: Optional[List[ModelType]] = None
    splits: Optional[List[SplitStrategy]] = None
    metric: Optional[str] = Field(default="rmse", description="Metric to sort by")


class BenchmarkResponse(BaseModel):
    """Benchmark results response."""
    results: List[BenchmarkResult]
    count: int
    best_model: Optional[str] = None
    best_rmse: Optional[float] = None


# =============================================================================
# Models
# =============================================================================

class ModelInfo(BaseModel):
    """Model information."""
    name: str
    type: str
    description: str
    parameters: Dict[str, Any]
    benchmark_rmse: Optional[float] = None
    is_available: bool = True


class ModelsResponse(BaseModel):
    """Available models response."""
    models: List[ModelInfo]
    count: int


# =============================================================================
# Data
# =============================================================================

class DrugInfo(BaseModel):
    """Drug information."""
    name: str
    smiles: Optional[str] = None
    target: Optional[str] = None
    phase: Optional[str] = None


class CellLineInfo(BaseModel):
    """Cell line information."""
    name: str
    tissue: Optional[str] = None
    subtype: Optional[str] = None
    mutations: Optional[List[str]] = None


class DatasetStats(BaseModel):
    """Dataset statistics."""
    n_drugs: int
    n_cell_lines: int
    n_records: int
    source: str
