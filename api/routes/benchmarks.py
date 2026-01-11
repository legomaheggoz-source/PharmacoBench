"""
Benchmarks Router

Endpoints for accessing benchmark results.
"""

from fastapi import APIRouter, Query
from typing import Optional, List
import numpy as np

from api.schemas import (
    BenchmarkResult,
    BenchmarkResponse,
    ModelType,
    SplitStrategy,
)

router = APIRouter()


def get_benchmark_data() -> List[dict]:
    """
    Get benchmark results.

    In production, this would load from a database or cache.
    """
    models = ["ridge", "elasticnet", "random_forest", "xgboost", "lightgbm", "mlp", "graphdrp", "deepcdr"]
    splits = ["random", "drug_blind", "cell_blind", "disjoint"]

    base_rmse = {
        "ridge": 1.2, "elasticnet": 1.18, "random_forest": 1.05,
        "xgboost": 0.98, "lightgbm": 0.99, "mlp": 1.0,
        "graphdrp": 0.92, "deepcdr": 0.88
    }

    split_penalty = {
        "random": 0, "drug_blind": 0.15, "cell_blind": 0.12, "disjoint": 0.25
    }

    results = []
    np.random.seed(42)

    for model in models:
        for split in splits:
            rmse = base_rmse[model] + split_penalty[split] + np.random.uniform(-0.05, 0.05)
            mae = rmse * 0.8 + np.random.uniform(-0.02, 0.02)
            spearman = 0.85 - (rmse - 0.8) * 0.3 + np.random.uniform(-0.02, 0.02)
            pearson = spearman + np.random.uniform(-0.02, 0.02)
            r2 = 1 - rmse**2 / 2 + np.random.uniform(-0.05, 0.05)

            results.append({
                "model": model,
                "split_strategy": split,
                "rmse": round(rmse, 4),
                "mae": round(mae, 4),
                "r_squared": round(max(0, min(1, r2)), 4),
                "pearson": round(min(1, max(0, pearson)), 4),
                "spearman": round(min(1, max(0, spearman)), 4),
                "train_time_seconds": round(np.random.uniform(10, 300), 1),
            })

    return results


@router.get("/benchmarks", response_model=BenchmarkResponse)
async def get_benchmarks(
    models: Optional[List[ModelType]] = Query(default=None),
    splits: Optional[List[SplitStrategy]] = Query(default=None),
    metric: str = Query(default="rmse", description="Metric to sort by"),
    limit: int = Query(default=100, ge=1, le=500),
):
    """
    Get benchmark results with optional filtering.

    Filter by models, split strategies, or sort by specific metrics.
    """
    all_results = get_benchmark_data()

    # Filter by models
    if models:
        model_values = [m.value for m in models]
        all_results = [r for r in all_results if r["model"] in model_values]

    # Filter by splits
    if splits:
        split_values = [s.value for s in splits]
        all_results = [r for r in all_results if r["split_strategy"] in split_values]

    # Sort by metric
    reverse = metric not in ["rmse", "mae"]  # Higher is better for correlations
    all_results.sort(key=lambda x: x.get(metric, 0), reverse=reverse)

    # Apply limit
    all_results = all_results[:limit]

    # Find best model
    best = min(all_results, key=lambda x: x["rmse"]) if all_results else None

    return BenchmarkResponse(
        results=[BenchmarkResult(**r) for r in all_results],
        count=len(all_results),
        best_model=best["model"] if best else None,
        best_rmse=best["rmse"] if best else None,
    )


@router.get("/benchmarks/summary")
async def get_benchmark_summary():
    """
    Get summary statistics across all benchmarks.
    """
    all_results = get_benchmark_data()

    # Calculate summary by model
    model_summary = {}
    for r in all_results:
        model = r["model"]
        if model not in model_summary:
            model_summary[model] = {"rmse": [], "spearman": []}
        model_summary[model]["rmse"].append(r["rmse"])
        model_summary[model]["spearman"].append(r["spearman"])

    summary = []
    for model, metrics in model_summary.items():
        summary.append({
            "model": model,
            "avg_rmse": round(np.mean(metrics["rmse"]), 4),
            "avg_spearman": round(np.mean(metrics["spearman"]), 4),
            "n_experiments": len(metrics["rmse"]),
        })

    # Sort by avg_rmse
    summary.sort(key=lambda x: x["avg_rmse"])

    return {
        "summary": summary,
        "best_model": summary[0]["model"] if summary else None,
        "total_experiments": len(all_results),
    }
