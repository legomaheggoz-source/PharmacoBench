"""
Tests for benchmark runner.
"""

import pytest
import pandas as pd
import numpy as np

from evaluation.benchmark_runner import BenchmarkRunner


class TestBenchmarkRunner:
    """Test suite for BenchmarkRunner class."""

    @pytest.fixture
    def benchmark_runner(self):
        """Create a benchmark runner instance."""
        return BenchmarkRunner()

    @pytest.fixture
    def sample_benchmark_data(self):
        """Generate sample data for benchmarking."""
        np.random.seed(42)
        n_samples = 200

        drugs = [f"Drug_{i}" for i in range(20)]
        cell_lines = [f"Cell_{i}" for i in range(30)]

        data = pd.DataFrame({
            "DRUG_NAME": np.random.choice(drugs, n_samples),
            "CELL_LINE_NAME": np.random.choice(cell_lines, n_samples),
            "LN_IC50": np.random.normal(-2, 1.5, n_samples),
        })

        # Generate features (simplified)
        n_features = 50
        features = np.random.randn(n_samples, n_features)

        return data, features

    def test_initialization(self, benchmark_runner):
        """Test runner initialization."""
        assert benchmark_runner is not None
        assert hasattr(benchmark_runner, "models")

    def test_add_model(self, benchmark_runner):
        """Test adding a model to the runner."""
        from models.traditional.ridge import RidgeModel

        model = RidgeModel(alpha=1.0)
        benchmark_runner.add_model("ridge", model)

        assert "ridge" in benchmark_runner.models

    def test_remove_model(self, benchmark_runner):
        """Test removing a model from the runner."""
        from models.traditional.ridge import RidgeModel

        model = RidgeModel(alpha=1.0)
        benchmark_runner.add_model("ridge", model)
        benchmark_runner.remove_model("ridge")

        assert "ridge" not in benchmark_runner.models

    def test_run_single_model(self, benchmark_runner, sample_benchmark_data):
        """Test running a single model."""
        from models.traditional.ridge import RidgeModel

        data, features = sample_benchmark_data

        model = RidgeModel(alpha=1.0)
        benchmark_runner.add_model("ridge", model)

        results = benchmark_runner.run(
            df=data,
            X_features=features,
            y_target=data["LN_IC50"].values,
            strategies=["random"],
        )

        assert results is not None
        assert len(results.results) > 0

    def test_run_multiple_strategies(self, benchmark_runner, sample_benchmark_data):
        """Test running with multiple split strategies."""
        from models.traditional.ridge import RidgeModel

        data, features = sample_benchmark_data

        model = RidgeModel(alpha=1.0)
        benchmark_runner.add_model("ridge", model)

        results = benchmark_runner.run(
            df=data,
            X_features=features,
            y_target=data["LN_IC50"].values,
            strategies=["random", "drug_blind"],
        )

        # Should have results for each strategy
        strategies_in_results = set(r.split_strategy for r in results.results)
        assert "random" in strategies_in_results
        assert "drug_blind" in strategies_in_results

    def test_run_multiple_models(self, benchmark_runner, sample_benchmark_data):
        """Test running with multiple models."""
        from models.traditional.ridge import RidgeModel
        from models.traditional.random_forest import RandomForestModel

        data, features = sample_benchmark_data

        benchmark_runner.add_model("ridge", RidgeModel(alpha=1.0))
        benchmark_runner.add_model("rf", RandomForestModel(n_estimators=10, max_depth=5))

        results = benchmark_runner.run(
            df=data,
            X_features=features,
            y_target=data["LN_IC50"].values,
            strategies=["random"],
        )

        models_in_results = set(r.model_name for r in results.results)
        assert "ridge" in models_in_results
        assert "rf" in models_in_results

    def test_results_contain_metrics(self, benchmark_runner, sample_benchmark_data):
        """Test that results contain expected metrics."""
        from models.traditional.ridge import RidgeModel

        data, features = sample_benchmark_data

        model = RidgeModel(alpha=1.0)
        benchmark_runner.add_model("ridge", model)

        results = benchmark_runner.run(
            df=data,
            X_features=features,
            y_target=data["LN_IC50"].values,
            strategies=["random"],
        )

        result = results.results[0]
        assert hasattr(result, "rmse")
        assert hasattr(result, "mae")
        assert hasattr(result, "r2")
        assert hasattr(result, "pearson")
        assert hasattr(result, "spearman")

    def test_metrics_are_valid(self, benchmark_runner, sample_benchmark_data):
        """Test that metrics have valid values."""
        from models.traditional.ridge import RidgeModel

        data, features = sample_benchmark_data

        model = RidgeModel(alpha=1.0)
        benchmark_runner.add_model("ridge", model)

        results = benchmark_runner.run(
            df=data,
            X_features=features,
            y_target=data["LN_IC50"].values,
            strategies=["random"],
        )

        result = results.results[0]

        assert result.rmse >= 0
        assert result.mae >= 0
        assert result.mae <= result.rmse
        assert -1 <= result.pearson <= 1
        assert -1 <= result.spearman <= 1

    def test_summary_dataframe(self, benchmark_runner, sample_benchmark_data):
        """Test getting results as DataFrame."""
        from models.traditional.ridge import RidgeModel

        data, features = sample_benchmark_data

        model = RidgeModel(alpha=1.0)
        benchmark_runner.add_model("ridge", model)

        results = benchmark_runner.run(
            df=data,
            X_features=features,
            y_target=data["LN_IC50"].values,
            strategies=["random"],
        )

        df_results = results.to_dataframe()

        assert isinstance(df_results, pd.DataFrame)
        assert "model" in df_results.columns or "Model" in df_results.columns
        assert len(df_results) > 0

    def test_best_model(self, benchmark_runner, sample_benchmark_data):
        """Test finding the best model."""
        from models.traditional.ridge import RidgeModel
        from models.traditional.random_forest import RandomForestModel

        data, features = sample_benchmark_data

        benchmark_runner.add_model("ridge", RidgeModel(alpha=1.0))
        benchmark_runner.add_model("rf", RandomForestModel(n_estimators=10, max_depth=5))

        results = benchmark_runner.run(
            df=data,
            X_features=features,
            y_target=data["LN_IC50"].values,
            strategies=["random"],
        )

        best = results.get_best_model(metric="rmse")

        assert best is not None
        assert best.model_name in ["ridge", "rf"]

    def test_empty_models(self, benchmark_runner, sample_benchmark_data):
        """Test running with no models added."""
        data, features = sample_benchmark_data

        with pytest.raises((ValueError, RuntimeError)):
            benchmark_runner.run(
                df=data,
                X_features=features,
                y_target=data["LN_IC50"].values,
                strategies=["random"],
            )

    def test_invalid_strategy(self, benchmark_runner, sample_benchmark_data):
        """Test running with invalid strategy."""
        from models.traditional.ridge import RidgeModel

        data, features = sample_benchmark_data

        model = RidgeModel(alpha=1.0)
        benchmark_runner.add_model("ridge", model)

        with pytest.raises(ValueError):
            benchmark_runner.run(
                df=data,
                X_features=features,
                y_target=data["LN_IC50"].values,
                strategies=["invalid_strategy"],
            )
