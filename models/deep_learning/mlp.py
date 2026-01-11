"""
Multi-Layer Perceptron Model

Neural network baseline for drug sensitivity prediction.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


class MLPNetwork(nn.Module):
    """PyTorch MLP network."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.BatchNorm1d(dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = dim

        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


class MLPModel(BaseModel):
    """
    Multi-Layer Perceptron model for drug sensitivity prediction.

    Deep neural network with batch normalization and dropout.
    """

    def __init__(
        self,
        hidden_dims: List[int] = [512, 256, 128],
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 64,
        epochs: int = 100,
        early_stopping_patience: int = 10,
        device: Optional[str] = None,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize MLP model.

        Args:
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout probability
            learning_rate: Adam learning rate
            batch_size: Training batch size
            epochs: Maximum training epochs
            early_stopping_patience: Epochs without improvement before stopping
            device: 'cuda' or 'cpu' (auto-detect if None)
            random_state: Random seed
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not installed. Run: pip install torch")

        super().__init__(
            hidden_dims=hidden_dims,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
            **kwargs,
        )

        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.random_state = random_state

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        self.model_: Optional[MLPNetwork] = None
        self.input_dim_: Optional[int] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "MLPModel":
        """
        Train MLP model.

        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets

        Returns:
            self
        """
        torch.manual_seed(self.random_state)

        self.input_dim_ = X_train.shape[1]
        self.model_ = MLPNetwork(
            input_dim=self.input_dim_,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)

        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.FloatTensor(y_train),
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
        )

        val_loader = None
        if X_val is not None and y_val is not None:
            val_dataset = TensorDataset(
                torch.FloatTensor(X_val),
                torch.FloatTensor(y_val),
            )
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size)

        # Training setup
        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Training MLP on {X_train.shape[0]} samples, device={self.device}")

        for epoch in range(self.epochs):
            # Training
            self.model_.train()
            train_loss = 0.0

            for X_batch, y_batch in train_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                outputs = self.model_(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * X_batch.size(0)

            train_loss /= len(train_loader.dataset)

            # Validation
            if val_loader is not None:
                self.model_.eval()
                val_loss = 0.0

                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch = X_batch.to(self.device)
                        y_batch = y_batch.to(self.device)
                        outputs = self.model_(X_batch)
                        val_loss += criterion(outputs, y_batch).item() * X_batch.size(0)

                val_loss /= len(val_loader.dataset)

                # Early stopping
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 20 == 0:
                val_str = f", val_loss={val_loss:.4f}" if val_loader else ""
                logger.info(f"Epoch {epoch + 1}/{self.epochs}: train_loss={train_loss:.4f}{val_str}")

        self.is_fitted_ = True
        logger.info("MLP training complete")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        self.model_.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            predictions = self.model_(X_tensor).cpu().numpy()

        return predictions

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return {
            "hidden_dims": self.hidden_dims,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, list]:
        """Get default hyperparameter grid."""
        return {
            "hidden_dims": [[256, 128], [512, 256, 128], [256, 128, 64]],
            "dropout": [0.2, 0.3, 0.5],
            "learning_rate": [0.001, 0.0001],
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model_.state_dict(),
            "input_dim": self.input_dim_,
            "hyperparameters": self.hyperparameters,
        }, path)

    def load(self, path: Union[str, Path]) -> "MLPModel":
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)

        self.input_dim_ = checkpoint["input_dim"]
        self.model_ = MLPNetwork(
            input_dim=self.input_dim_,
            hidden_dims=self.hidden_dims,
            dropout=self.dropout,
        ).to(self.device)
        self.model_.load_state_dict(checkpoint["model_state_dict"])
        self.is_fitted_ = True

        return self
