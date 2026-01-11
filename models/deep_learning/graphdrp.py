"""
GraphDRP Model

Graph Drug Response Prediction using Graph Convolutional Networks.
Encodes drugs as molecular graphs and cell lines with 1D CNNs.

Based on: https://github.com/hauldhut/GraphDRP
"""

from __future__ import annotations  # Enable lazy type hint evaluation

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.data import Data, Batch
    from torch_geometric.utils import from_smiles
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


if TORCH_GEOMETRIC_AVAILABLE:

    class DrugEncoder(nn.Module):
        """GCN encoder for drug molecular graphs."""

        def __init__(
            self,
            num_features: int = 9,
            hidden_dim: int = 128,
            output_dim: int = 128,
        ):
            super().__init__()
            self.conv1 = GCNConv(num_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, output_dim)
            self.bn1 = nn.BatchNorm1d(hidden_dim)
            self.bn2 = nn.BatchNorm1d(hidden_dim)
            self.bn3 = nn.BatchNorm1d(output_dim)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch

            x = F.relu(self.bn1(self.conv1(x, edge_index)))
            x = F.relu(self.bn2(self.conv2(x, edge_index)))
            x = F.relu(self.bn3(self.conv3(x, edge_index)))

            # Global mean pooling
            x = global_mean_pool(x, batch)
            return x


    class CellEncoder(nn.Module):
        """1D CNN encoder for cell line genomic features."""

        def __init__(
            self,
            input_dim: int = 735,
            hidden_dims: List[int] = [32, 64, 128],
            output_dim: int = 128,
        ):
            super().__init__()

            self.conv1 = nn.Conv1d(1, hidden_dims[0], kernel_size=8, padding=4)
            self.conv2 = nn.Conv1d(hidden_dims[0], hidden_dims[1], kernel_size=8, padding=4)
            self.conv3 = nn.Conv1d(hidden_dims[1], hidden_dims[2], kernel_size=8, padding=4)

            self.pool = nn.AdaptiveMaxPool1d(1)
            self.fc = nn.Linear(hidden_dims[2], output_dim)

            self.bn1 = nn.BatchNorm1d(hidden_dims[0])
            self.bn2 = nn.BatchNorm1d(hidden_dims[1])
            self.bn3 = nn.BatchNorm1d(hidden_dims[2])

        def forward(self, x):
            # x: (batch, input_dim) -> (batch, 1, input_dim)
            x = x.unsqueeze(1)

            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))

            x = self.pool(x).squeeze(-1)
            x = self.fc(x)
            return x


    class GraphDRPNetwork(nn.Module):
        """GraphDRP: Drug encoder (GCN) + Cell encoder (CNN) + Predictor."""

        def __init__(
            self,
            drug_features: int = 9,
            cell_features: int = 735,
            hidden_dim: int = 128,
            dropout: float = 0.3,
        ):
            super().__init__()
            self.drug_encoder = DrugEncoder(
                num_features=drug_features,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
            )
            self.cell_encoder = CellEncoder(
                input_dim=cell_features,
                output_dim=hidden_dim,
            )

            # Predictor
            self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, 64)
            self.out = nn.Linear(64, 1)
            self.dropout = nn.Dropout(dropout)

        def forward(self, drug_data, cell_features):
            drug_emb = self.drug_encoder(drug_data)
            cell_emb = self.cell_encoder(cell_features)

            # Concatenate and predict
            combined = torch.cat([drug_emb, cell_emb], dim=-1)
            x = F.relu(self.fc1(combined))
            x = self.dropout(x)
            x = F.relu(self.fc2(x))
            x = self.dropout(x)
            x = self.out(x)
            return x.squeeze(-1)


class GraphDRPModel(BaseModel):
    """
    GraphDRP model for drug sensitivity prediction.

    Uses GCN to encode molecular structure and CNN for cell line features.
    """

    def __init__(
        self,
        drug_features: int = 9,
        cell_features: int = 735,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        learning_rate: float = 0.0001,
        batch_size: int = 128,
        epochs: int = 100,
        early_stopping_patience: int = 15,
        device: Optional[str] = None,
        random_state: int = 42,
        **kwargs,
    ):
        """
        Initialize GraphDRP model.

        Args:
            drug_features: Number of atom features
            cell_features: Dimension of cell line features
            hidden_dim: Hidden layer dimension
            dropout: Dropout probability
            learning_rate: Adam learning rate
            batch_size: Training batch size
            epochs: Maximum training epochs
            early_stopping_patience: Patience for early stopping
            device: 'cuda' or 'cpu'
            random_state: Random seed
        """
        if not TORCH_GEOMETRIC_AVAILABLE:
            raise ImportError(
                "PyTorch Geometric not installed. Run: pip install torch-geometric"
            )

        super().__init__(
            drug_features=drug_features,
            cell_features=cell_features,
            hidden_dim=hidden_dim,
            dropout=dropout,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epochs=epochs,
            early_stopping_patience=early_stopping_patience,
            random_state=random_state,
            **kwargs,
        )

        self.drug_features = drug_features
        self.cell_features = cell_features
        self.hidden_dim = hidden_dim
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

        self.model_: Optional[GraphDRPNetwork] = None

    @staticmethod
    def smiles_to_graph(smiles: str) -> Optional[Data]:
        """Convert SMILES to PyTorch Geometric graph."""
        try:
            return from_smiles(smiles)
        except Exception:
            return None

    def fit(
        self,
        X_train: Tuple[List[Data], np.ndarray],
        y_train: np.ndarray,
        X_val: Optional[Tuple[List[Data], np.ndarray]] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "GraphDRPModel":
        """
        Train GraphDRP model.

        Args:
            X_train: Tuple of (drug_graphs, cell_features)
            y_train: Training targets
            X_val: Validation data
            y_val: Validation targets

        Returns:
            self
        """
        torch.manual_seed(self.random_state)

        drug_graphs_train, cell_features_train = X_train

        # Infer feature dimensions
        if len(drug_graphs_train) > 0 and drug_graphs_train[0] is not None:
            self.drug_features = drug_graphs_train[0].x.shape[1]
        self.cell_features = cell_features_train.shape[1]

        self.model_ = GraphDRPNetwork(
            drug_features=self.drug_features,
            cell_features=self.cell_features,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        n_samples = len(drug_graphs_train)
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Training GraphDRP on {n_samples} samples, device={self.device}")

        for epoch in range(self.epochs):
            self.model_.train()
            train_loss = 0.0

            # Shuffle indices
            indices = np.random.permutation(n_samples)

            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]

                # Batch drug graphs
                batch_graphs = [drug_graphs_train[j] for j in batch_indices]
                batch_graphs = [g for g in batch_graphs if g is not None]

                if len(batch_graphs) == 0:
                    continue

                drug_batch = Batch.from_data_list(batch_graphs).to(self.device)
                cell_batch = torch.FloatTensor(
                    cell_features_train[batch_indices[:len(batch_graphs)]]
                ).to(self.device)
                y_batch = torch.FloatTensor(
                    y_train[batch_indices[:len(batch_graphs)]]
                ).to(self.device)

                optimizer.zero_grad()
                outputs = self.model_(drug_batch, cell_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_graphs)

            train_loss /= n_samples

            # Validation
            if X_val is not None and y_val is not None:
                drug_graphs_val, cell_features_val = X_val
                val_loss = self._evaluate(drug_graphs_val, cell_features_val, y_val)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{self.epochs}: train_loss={train_loss:.4f}")

        self.is_fitted_ = True
        logger.info("GraphDRP training complete")
        return self

    def _evaluate(
        self,
        drug_graphs: List[Data],
        cell_features: np.ndarray,
        y: np.ndarray,
    ) -> float:
        """Evaluate model on data."""
        self.model_.eval()
        total_loss = 0.0
        n_samples = len(drug_graphs)

        with torch.no_grad():
            for i in range(0, n_samples, self.batch_size):
                batch_graphs = drug_graphs[i:i + self.batch_size]
                batch_graphs = [g for g in batch_graphs if g is not None]

                if len(batch_graphs) == 0:
                    continue

                drug_batch = Batch.from_data_list(batch_graphs).to(self.device)
                cell_batch = torch.FloatTensor(
                    cell_features[i:i + len(batch_graphs)]
                ).to(self.device)
                y_batch = torch.FloatTensor(
                    y[i:i + len(batch_graphs)]
                ).to(self.device)

                outputs = self.model_(drug_batch, cell_batch)
                loss = F.mse_loss(outputs, y_batch)
                total_loss += loss.item() * len(batch_graphs)

        return total_loss / n_samples

    def predict(
        self,
        X: Tuple[List[Data], np.ndarray],
    ) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        drug_graphs, cell_features = X
        self.model_.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(drug_graphs), self.batch_size):
                batch_graphs = drug_graphs[i:i + self.batch_size]
                batch_graphs = [g for g in batch_graphs if g is not None]

                if len(batch_graphs) == 0:
                    continue

                drug_batch = Batch.from_data_list(batch_graphs).to(self.device)
                cell_batch = torch.FloatTensor(
                    cell_features[i:i + len(batch_graphs)]
                ).to(self.device)

                outputs = self.model_(drug_batch, cell_batch)
                predictions.extend(outputs.cpu().numpy().tolist())

        return np.array(predictions)

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return {
            "drug_features": self.drug_features,
            "cell_features": self.cell_features,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epochs": self.epochs,
        }

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, list]:
        """Get default hyperparameter grid."""
        return {
            "hidden_dim": [64, 128, 256],
            "dropout": [0.2, 0.3, 0.5],
            "learning_rate": [0.0001, 0.001],
        }

    def save(self, path: Union[str, Path]) -> None:
        """Save model."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state_dict": self.model_.state_dict(),
            "hyperparameters": self.hyperparameters,
        }, path)

    def load(self, path: Union[str, Path]) -> "GraphDRPModel":
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model_ = GraphDRPNetwork(
            drug_features=self.drug_features,
            cell_features=self.cell_features,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)
        self.model_.load_state_dict(checkpoint["model_state_dict"])
        self.is_fitted_ = True

        return self
