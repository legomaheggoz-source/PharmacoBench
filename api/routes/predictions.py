"""
Predictions Router

Endpoints for drug sensitivity predictions.
"""

from fastapi import APIRouter, HTTPException
import numpy as np
from typing import Dict

from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    BatchPredictionRequest,
    BatchPredictionResponse,
)

router = APIRouter()


def simulate_prediction(drug: str, cell_line: str, model: str) -> Dict:
    """
    Simulate a prediction.

    In production, this would load and run an actual trained model.
    """
    np.random.seed(hash(drug + cell_line + model) % 2**32)

    # Model-specific base performance
    model_rmse = {
        "ridge": 1.20,
        "elasticnet": 1.18,
        "random_forest": 1.05,
        "xgboost": 0.98,
        "lightgbm": 0.99,
        "mlp": 1.00,
        "graphdrp": 0.92,
        "deepcdr": 0.88,
    }.get(model, 1.0)

    # Generate prediction
    base_pred = -2.0 + np.random.normal(0, model_rmse * 0.5)
    drug_effect = (hash(drug) % 10) / 10 - 0.5
    cell_effect = (hash(cell_line) % 10) / 10 - 0.5

    prediction = base_pred + drug_effect + cell_effect
    ci_width = model_rmse * 0.3

    # Determine sensitivity class
    if prediction < -2:
        sensitivity = "Sensitive"
    elif prediction < -1:
        sensitivity = "Intermediate"
    else:
        sensitivity = "Resistant"

    return {
        "prediction": round(prediction, 4),
        "ci_lower": round(prediction - ci_width, 4),
        "ci_upper": round(prediction + ci_width, 4),
        "sensitivity": sensitivity,
    }


@router.post("/predict", response_model=PredictionResponse)
async def predict_single(request: PredictionRequest):
    """
    Predict drug sensitivity for a single drug-cell line pair.

    Returns predicted ln(IC50) value with confidence interval.
    """
    result = simulate_prediction(
        request.drug_name,
        request.cell_line_name,
        request.model.value
    )

    return PredictionResponse(
        drug_name=request.drug_name,
        cell_line_name=request.cell_line_name,
        model=request.model.value,
        predicted_ln_ic50=result["prediction"],
        confidence_interval={
            "lower": result["ci_lower"],
            "upper": result["ci_upper"],
        },
        sensitivity_class=result["sensitivity"],
    )


@router.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict drug sensitivity for multiple drug-cell line pairs.

    Maximum 1000 pairs per request.
    """
    if len(request.pairs) > 1000:
        raise HTTPException(
            status_code=400,
            detail="Maximum 1000 pairs per request"
        )

    predictions = []

    for pair in request.pairs:
        drug_name = pair.get("drug_name")
        cell_line_name = pair.get("cell_line_name")

        if not drug_name or not cell_line_name:
            raise HTTPException(
                status_code=400,
                detail="Each pair must have 'drug_name' and 'cell_line_name'"
            )

        result = simulate_prediction(drug_name, cell_line_name, request.model.value)

        predictions.append(PredictionResponse(
            drug_name=drug_name,
            cell_line_name=cell_line_name,
            model=request.model.value,
            predicted_ln_ic50=result["prediction"],
            confidence_interval={
                "lower": result["ci_lower"],
                "upper": result["ci_upper"],
            },
            sensitivity_class=result["sensitivity"],
        ))

    return BatchPredictionResponse(
        predictions=predictions,
        model=request.model.value,
        count=len(predictions),
    )
