"""
Models Router

Endpoints for listing available models.
"""

from fastapi import APIRouter
from typing import List

from api.schemas import ModelInfo, ModelsResponse

router = APIRouter()

# Model registry
MODEL_REGISTRY = {
    "ridge": ModelInfo(
        name="Ridge Regression",
        type="Linear",
        description="Linear regression with L2 regularization. Fast and interpretable baseline.",
        parameters={"alpha": [0.001, 0.01, 0.1, 1.0, 10.0]},
        benchmark_rmse=1.20,
        is_available=True,
    ),
    "elasticnet": ModelInfo(
        name="ElasticNet",
        type="Linear",
        description="Linear regression with L1+L2 regularization. Provides sparsity.",
        parameters={"alpha": [0.001, 0.01, 0.1, 1.0], "l1_ratio": [0.1, 0.5, 0.9]},
        benchmark_rmse=1.18,
        is_available=True,
    ),
    "random_forest": ModelInfo(
        name="Random Forest",
        type="Ensemble",
        description="Ensemble of decision trees. Handles non-linear relationships.",
        parameters={"n_estimators": [100, 200, 500], "max_depth": [10, 20, None]},
        benchmark_rmse=1.05,
        is_available=True,
    ),
    "xgboost": ModelInfo(
        name="XGBoost",
        type="Gradient Boosting",
        description="Gradient boosting with regularization. Strong tabular performance.",
        parameters={"n_estimators": [100, 200], "max_depth": [3, 6, 9], "learning_rate": [0.01, 0.1]},
        benchmark_rmse=0.98,
        is_available=True,
    ),
    "lightgbm": ModelInfo(
        name="LightGBM",
        type="Gradient Boosting",
        description="Fast gradient boosting. Memory efficient.",
        parameters={"n_estimators": [100, 200], "num_leaves": [31, 63], "learning_rate": [0.01, 0.1]},
        benchmark_rmse=0.99,
        is_available=True,
    ),
    "mlp": ModelInfo(
        name="MLP",
        type="Deep Learning",
        description="Multi-layer perceptron. Deep learning baseline.",
        parameters={"hidden_dims": [[256, 128], [512, 256, 128]], "dropout": [0.2, 0.3]},
        benchmark_rmse=1.00,
        is_available=False,  # Requires PyTorch
    ),
    "graphdrp": ModelInfo(
        name="GraphDRP",
        type="Graph Neural Network",
        description="Graph Neural Network. Encodes drug molecular structure.",
        parameters={"hidden_dim": [64, 128], "dropout": [0.2, 0.3]},
        benchmark_rmse=0.92,
        is_available=False,  # Requires PyTorch Geometric
    ),
    "deepcdr": ModelInfo(
        name="DeepCDR",
        type="Hybrid GNN",
        description="Hybrid GNN with multi-omics integration. State-of-the-art.",
        parameters={"hidden_dim": [64, 128], "dropout": [0.2, 0.3]},
        benchmark_rmse=0.88,
        is_available=False,  # Requires PyTorch Geometric
    ),
}


@router.get("/models", response_model=ModelsResponse)
async def list_models(available_only: bool = False):
    """
    List all available models.

    Set available_only=true to filter to models with loaded weights.
    """
    models = list(MODEL_REGISTRY.values())

    if available_only:
        models = [m for m in models if m.is_available]

    return ModelsResponse(
        models=models,
        count=len(models),
    )


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model(model_id: str):
    """
    Get information about a specific model.
    """
    if model_id not in MODEL_REGISTRY:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_id}' not found. Available: {list(MODEL_REGISTRY.keys())}"
        )

    return MODEL_REGISTRY[model_id]
