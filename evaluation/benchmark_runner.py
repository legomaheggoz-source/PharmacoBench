"""
Benchmark Runner

Orchestrates model training and evaluation across multiple models and split strategies.
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Type

import numpy as np
import pandas as pd

from models.base_model import BaseModel
from data.splitters import DataSplitter, SplitResult
from evaluation.metrics import compute_all_metrics, MetricsTracker

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Container for single benchmark result."""
    model_name: str
    split_strategy: str
    metrics: Dict[str, float]
    train_time_seconds: float
    predict_time_seconds: float
    hyperparameters: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    fold: Optional[int] = None

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            "model": self.model_name,
            "split": self.split_strategy,
            **self.metrics,
            "train_time": self.train_time_seconds,
            "predict_time": self.predict_time_seconds,
            "timestamp": self.timestamp,
            "fold": self.fold,
        }


@dataclass
class BenchmarkSummary:
    """Container for aggregated benchmark results."""
    results: List[BenchmarkResult]
    summary_df: pd.DataFrame
    best_model: str
    best_metrics: Dict[str, float]

    def get_model_ranking(self, metric: str = "rmse", ascending: bool = True) -> pd.DataFrame:
        """
        Rank models by a specific metric.

        Args:
            metric: Metric to rank by
            ascending: If True, lower is better (for RMSE, MAE)

        Returns:
            Ranked DataFrame
        """
        return self.summary_df.sort_values(metric, ascending=ascending)


class BenchmarkRunner:
    """
    Orchestrates benchmarking of multiple models across split strategies.

    Usage:
        runner = BenchmarkRunner()
        runner.add_model("ridge", RidgeModel())
        runner.add_model("rf", RandomForestModel())
        results = runner.run(X, y, strategies=["random", "drug_blind"])
    """

    def __init__(
        self,
        splitter: Optional[DataSplitter] = None,
        checkpoint_dir: Optional[str] = None,
    ):
        """
        Initialize benchmark runner.

        Args:
            splitter: DataSplitter instance (creates default if None)
            checkpoint_dir: Directory to save model checkpoints
        """
        self.splitter = splitter or DataSplitter()
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        self.models: Dict[str, BaseModel] = {}
        self.results: List[BenchmarkResult] = []

    def add_model(self, name: str, model: BaseModel) -> "BenchmarkRunner":
        """
        Add a model to benchmark.

        Args:
            name: Model identifier
            model: Model instance

        Returns:
            self (for chaining)
        """
        self.models[name] = model
        logger.info(f"Added model: {name} ({model.__class__.__name__})")
        return self

    def add_models(self, models: Dict[str, BaseModel]) -> "BenchmarkRunner":
        """Add multiple models."""
        for name, model in models.items():
            self.add_model(name, model)
        return self

    def run(
        self,
        df: pd.DataFrame,
        X_features: np.ndarray,
        y_target: np.ndarray,
        strategies: Optional[List[str]] = None,
        n_cv_folds: int = 0,
    ) -> BenchmarkSummary:
        """
        Run benchmarks across all models and strategies.

        Args:
            df: DataFrame with DRUG_NAME, CELL_LINE_NAME columns (for splitting)
            X_features: Feature matrix (n_samples, n_features)
            y_target: Target values (n_samples,)
            strategies: Split strategies to use (default: all 4)
            n_cv_folds: If > 0, run cross-validation with this many folds

        Returns:
            BenchmarkSummary with all results
        """
        if strategies is None:
            strategies = DataSplitter.VALID_STRATEGIES

        if len(self.models) == 0:
            raise ValueError("No models added. Call add_model() first.")

        logger.info(
            f"Starting benchmark: {len(self.models)} models × {len(strategies)} strategies"
        )

        self.results = []

        for strategy in strategies:
            logger.info(f"\n{'='*50}")
            logger.info(f"Split Strategy: {strategy}")
            logger.info(f"{'='*50}")

            if n_cv_folds > 0:
                # Cross-validation
                splits = self.splitter.cross_validation_split(df, strategy, n_cv_folds)
                self._run_cv(splits, X_features, y_target, df, strategy)
            else:
                # Single split
                split = self.splitter.split(df, strategy)
                self._run_single_split(split, X_features, y_target, df, strategy)

        # Create summary
        summary = self._create_summary()
        logger.info(f"\nBenchmark complete. Results: {len(self.results)}")

        return summary

    def _run_single_split(
        self,
        split: SplitResult,
        X: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame,
        strategy: str,
    ) -> None:
        """Run benchmark on single split."""
        # Get indices for train/val/test
        train_idx = df.index.isin(split.train_df.index)
        val_idx = df.index.isin(split.val_df.index)
        test_idx = df.index.isin(split.test_df.index)

        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        X_test, y_test = X[test_idx], y[test_idx]

        for model_name, model in self.models.items():
            result = self._train_and_evaluate(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                X_val=X_val,
                y_val=y_val,
                X_test=X_test,
                y_test=y_test,
                strategy=strategy,
            )
            self.results.append(result)

    def _run_cv(
        self,
        splits: List[SplitResult],
        X: np.ndarray,
        y: np.ndarray,
        df: pd.DataFrame,
        strategy: str,
    ) -> None:
        """Run cross-validation benchmark."""
        for model_name, model in self.models.items():
            tracker = MetricsTracker()

            for fold, split in enumerate(splits):
                # Get indices
                train_idx = df.index.isin(split.train_df.index)
                val_idx = df.index.isin(split.val_df.index)
                test_idx = df.index.isin(split.test_df.index)

                X_train, y_train = X[train_idx], y[train_idx]
                X_val, y_val = X[val_idx], y[val_idx]
                X_test, y_test = X[test_idx], y[test_idx]

                # Create fresh model instance for each fold
                model_instance = model.__class__(**model.hyperparameters)

                result = self._train_and_evaluate(
                    model_name=model_name,
                    model=model_instance,
                    X_train=X_train,
                    y_train=y_train,
                    X_val=X_val,
                    y_val=y_val,
                    X_test=X_test,
                    y_test=y_test,
                    strategy=strategy,
                    fold=fold,
                )
                self.results.append(result)
                tracker.add(result.metrics)

            # Log CV summary
            summary = tracker.get_summary()
            logger.info(
                f"{model_name} CV: RMSE={summary['mean']['rmse']:.4f}±{summary['std']['rmse']:.4f}, "
                f"Spearman={summary['mean']['spearman']:.4f}±{summary['std']['spearman']:.4f}"
            )

    def _train_and_evaluate(
        self,
        model_name: str,
        model: BaseModel,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
        strategy: str,
        fold: Optional[int] = None,
    ) -> BenchmarkResult:
        """Train model and evaluate on test set."""
        fold_str = f" (fold {fold})" if fold is not None else ""
        logger.info(f"Training {model_name}{fold_str}...")

        # Train
        start_time = time.time()
        model.fit(X_train, y_train, X_val, y_val)
        train_time = time.time() - start_time

        # Predict
        start_time = time.time()
        y_pred = model.predict(X_test)
        predict_time = time.time() - start_time

        # Compute metrics
        metrics = compute_all_metrics(y_test, y_pred)

        logger.info(
            f"  {model_name}: RMSE={metrics['rmse']:.4f}, "
            f"Spearman={metrics['spearman']:.4f}, "
            f"time={train_time:.1f}s"
        )

        # Save checkpoint if configured
        if self.checkpoint_dir:
            checkpoint_path = self.checkpoint_dir / f"{model_name}_{strategy}.pt"
            model.save(checkpoint_path)

        return BenchmarkResult(
            model_name=model_name,
            split_strategy=strategy,
            metrics=metrics,
            train_time_seconds=train_time,
            predict_time_seconds=predict_time,
            hyperparameters=model.get_hyperparameters(),
            fold=fold,
        )

    def _create_summary(self) -> BenchmarkSummary:
        """Create summary from results."""
        # Convert to DataFrame
        records = [r.to_dict() for r in self.results]
        df = pd.DataFrame(records)

        # Aggregate by model and strategy (mean across folds if CV)
        summary_df = df.groupby(["model", "split"]).agg({
            "rmse": ["mean", "std"],
            "mae": ["mean", "std"],
            "r2": ["mean", "std"],
            "pearson": ["mean", "std"],
            "spearman": ["mean", "std"],
            "train_time": "mean",
            "n_samples": "first",
        }).round(4)

        # Flatten column names
        summary_df.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0]
            for col in summary_df.columns
        ]
        summary_df = summary_df.reset_index()

        # Find best model (lowest average RMSE)
        model_avg_rmse = df.groupby("model")["rmse"].mean()
        best_model = model_avg_rmse.idxmin()
        best_metrics = df[df["model"] == best_model].iloc[0].to_dict()

        return BenchmarkSummary(
            results=self.results,
            summary_df=summary_df,
            best_model=best_model,
            best_metrics=best_metrics,
        )

    def get_results_dataframe(self) -> pd.DataFrame:
        """Get all results as DataFrame."""
        return pd.DataFrame([r.to_dict() for r in self.results])

    def save_results(self, path: str) -> None:
        """Save results to CSV."""
        df = self.get_results_dataframe()
        df.to_csv(path, index=False)
        logger.info(f"Results saved to {path}")


def main():
    """Test benchmark runner."""
    logging.basicConfig(level=logging.INFO)

    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 100

    df = pd.DataFrame({
        "DRUG_NAME": np.random.choice([f"Drug_{i}" for i in range(20)], n_samples),
        "CELL_LINE_NAME": np.random.choice([f"Cell_{i}" for i in range(50)], n_samples),
    })
    X = np.random.randn(n_samples, n_features)
    y = np.random.randn(n_samples)

    # Import models
    from models.traditional.ridge import RidgeModel
    from models.traditional.random_forest import RandomForestModel

    # Run benchmark
    runner = BenchmarkRunner()
    runner.add_model("ridge", RidgeModel(alpha=1.0))
    runner.add_model("rf", RandomForestModel(n_estimators=50, max_depth=10))

    summary = runner.run(
        df=df,
        X_features=X,
        y_target=y,
        strategies=["random", "drug_blind"],
    )

    print("\nSummary:")
    print(summary.summary_df.to_string())
    print(f"\nBest model: {summary.best_model}")


if __name__ == "__main__":
    main()
