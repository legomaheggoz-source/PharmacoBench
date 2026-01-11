"""
Tests for deep learning models.
"""

import pytest
import numpy as np

# Check if PyTorch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Check if PyTorch Geometric is available
try:
    import torch_geometric
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestMLPModel:
    """Test suite for MLP model."""

    def test_initialization(self):
        """Test model initialization."""
        from models.deep_learning.mlp import MLPModel

        model = MLPModel(input_dim=100, hidden_dims=[256, 128], dropout=0.2)
        assert model.input_dim == 100
        assert model.hidden_dims == [256, 128]

    def test_fit_predict(self, sample_train_test_data):
        """Test fit and predict methods."""
        from models.deep_learning.mlp import MLPModel

        X_train, X_test, y_train, y_test = sample_train_test_data

        model = MLPModel(
            input_dim=X_train.shape[1],
            hidden_dims=[32, 16],
            dropout=0.2,
            epochs=5,
            batch_size=16,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        assert predictions.shape == y_test.shape
        assert not np.any(np.isnan(predictions))

    def test_validation_loss(self, sample_train_test_data):
        """Test training with validation data."""
        from models.deep_learning.mlp import MLPModel

        X_train, X_test, y_train, y_test = sample_train_test_data

        model = MLPModel(
            input_dim=X_train.shape[1],
            hidden_dims=[32],
            epochs=3,
        )
        model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

        assert model.model is not None

    def test_save_load(self, sample_train_test_data, tmp_path):
        """Test model save and load."""
        from models.deep_learning.mlp import MLPModel

        X_train, X_test, y_train, y_test = sample_train_test_data

        model = MLPModel(
            input_dim=X_train.shape[1],
            hidden_dims=[32],
            epochs=3,
        )
        model.fit(X_train, y_train)
        original_predictions = model.predict(X_test)

        # Save
        save_path = tmp_path / "mlp_model.pt"
        model.save(str(save_path))
        assert save_path.exists()

        # Load
        new_model = MLPModel(input_dim=X_train.shape[1], hidden_dims=[32])
        new_model.load(str(save_path))
        loaded_predictions = new_model.predict(X_test)

        np.testing.assert_array_almost_equal(original_predictions, loaded_predictions, decimal=5)

    def test_hyperparameters(self):
        """Test hyperparameter retrieval."""
        from models.deep_learning.mlp import MLPModel

        model = MLPModel(
            input_dim=100,
            hidden_dims=[256, 128],
            dropout=0.3,
            learning_rate=0.001,
        )
        params = model.get_hyperparameters()

        assert params["input_dim"] == 100
        assert params["hidden_dims"] == [256, 128]
        assert params["dropout"] == 0.3


@pytest.mark.skipif(not PYG_AVAILABLE, reason="PyTorch Geometric not installed")
class TestGraphDRPModel:
    """Test suite for GraphDRP model."""

    @pytest.fixture
    def sample_graph_data(self):
        """Generate sample graph data for drug molecules."""
        from torch_geometric.data import Data, Batch
        import torch

        # Create simple molecular graphs
        graphs = []
        for _ in range(10):
            # Random graph with 5-10 nodes
            n_nodes = np.random.randint(5, 11)
            n_edges = np.random.randint(4, 15)

            x = torch.randn(n_nodes, 9)  # Node features
            edge_index = torch.randint(0, n_nodes, (2, n_edges))

            graphs.append(Data(x=x, edge_index=edge_index))

        return graphs

    def test_initialization(self):
        """Test model initialization."""
        from models.deep_learning.graphdrp import GraphDRPModel

        model = GraphDRPModel(
            drug_features=9,
            cell_features=100,
            hidden_dim=64,
        )
        assert model.hidden_dim == 64

    def test_fit_predict_with_fingerprints(self, sample_train_test_data):
        """Test fit and predict with fingerprint representations."""
        from models.deep_learning.graphdrp import GraphDRPModel

        X_train, X_test, y_train, y_test = sample_train_test_data

        # Simulate combined drug fingerprint + cell features
        model = GraphDRPModel(
            drug_features=100,  # Fingerprint size
            cell_features=X_train.shape[1] - 100,
            hidden_dim=32,
            epochs=2,
            use_graphs=False,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert predictions.shape == y_test.shape


@pytest.mark.skipif(not PYG_AVAILABLE, reason="PyTorch Geometric not installed")
class TestDeepCDRModel:
    """Test suite for DeepCDR model."""

    def test_initialization(self):
        """Test model initialization."""
        from models.deep_learning.deepcdr import DeepCDRModel

        model = DeepCDRModel(
            drug_features=9,
            mutation_features=100,
            expression_features=200,
            methylation_features=100,
            hidden_dim=64,
        )
        assert model.hidden_dim == 64

    def test_fit_predict_simplified(self, sample_train_test_data):
        """Test fit and predict with simplified input."""
        from models.deep_learning.deepcdr import DeepCDRModel

        X_train, X_test, y_train, y_test = sample_train_test_data

        model = DeepCDRModel(
            drug_features=10,
            mutation_features=20,
            expression_features=15,
            methylation_features=5,
            hidden_dim=16,
            epochs=2,
            use_graphs=False,
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        assert predictions.shape == y_test.shape


@pytest.mark.skipif(not TORCH_AVAILABLE, reason="PyTorch not installed")
class TestDeepLearningConsistency:
    """Test consistency across deep learning models."""

    def test_gpu_availability(self):
        """Test GPU detection."""
        import torch

        # Should not error regardless of GPU availability
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        assert device is not None

    def test_reproducibility(self, sample_train_test_data):
        """Test that models are reproducible with same seed."""
        from models.deep_learning.mlp import MLPModel
        import torch

        X_train, X_test, y_train, y_test = sample_train_test_data

        # First run
        torch.manual_seed(42)
        np.random.seed(42)
        model1 = MLPModel(input_dim=X_train.shape[1], hidden_dims=[16], epochs=2)
        model1.fit(X_train, y_train)
        pred1 = model1.predict(X_test)

        # Second run with same seed
        torch.manual_seed(42)
        np.random.seed(42)
        model2 = MLPModel(input_dim=X_train.shape[1], hidden_dims=[16], epochs=2)
        model2.fit(X_train, y_train)
        pred2 = model2.predict(X_test)

        np.testing.assert_array_almost_equal(pred1, pred2, decimal=4)
