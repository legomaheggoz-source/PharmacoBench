"""
DeepCDR Model

Hybrid Graph Convolutional Network for Cancer Drug Response prediction.
Integrates multi-omics data (mutation, expression, methylation) with drug structure.

Based on: https://github.com/kimmo1019/DeepCDR
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
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    TORCH_GEOMETRIC_AVAILABLE = False

from models.base_model import BaseModel

logger = logging.getLogger(__name__)


if TORCH_GEOMETRIC_AVAILABLE:

    class DrugGCN(nn.Module):
        """GCN encoder for drug molecular graphs."""

        def __init__(
            self,
            num_features: int = 75,
            hidden_dim: int = 128,
            output_dim: int = 128,
            num_layers: int = 3,
        ):
            super().__init__()
            self.convs = nn.ModuleList()
            self.bns = nn.ModuleList()

            # First layer
            self.convs.append(GCNConv(num_features, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

            # Hidden layers
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_dim, hidden_dim))
                self.bns.append(nn.BatchNorm1d(hidden_dim))

            # Output layer
            self.convs.append(GCNConv(hidden_dim, output_dim))
            self.bns.append(nn.BatchNorm1d(output_dim))

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch

            for conv, bn in zip(self.convs, self.bns):
                x = F.relu(bn(conv(x, edge_index)))

            # Global pooling
            x = global_mean_pool(x, batch)
            return x


    class OmicsEncoder(nn.Module):
        """MLP encoder for omics data (mutation, expression, methylation)."""

        def __init__(
            self,
            input_dim: int,
            hidden_dims: List[int] = [256, 128],
            output_dim: int = 128,
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

            layers.append(nn.Linear(prev_dim, output_dim))
            self.encoder = nn.Sequential(*layers)

        def forward(self, x):
            return self.encoder(x)


    class DeepCDRNetwork(nn.Module):
        """
        DeepCDR: Integrates drug structure with multi-omics cell line data.

        Architecture:
        - Drug: GCN on molecular graph
        - Mutation: MLP encoder
        - Expression: MLP encoder
        - Methylation: MLP encoder (optional)
        - Fusion: Concatenate all embeddings
        - Prediction: MLP head
        """

        def __init__(
            self,
            drug_features: int = 75,
            mutation_dim: int = 735,
            expression_dim: int = 697,
            methylation_dim: Optional[int] = None,
            hidden_dim: int = 128,
            dropout: float = 0.3,
        ):
            super().__init__()

            # Drug encoder (GCN)
            self.drug_encoder = DrugGCN(
                num_features=drug_features,
                hidden_dim=hidden_dim,
                output_dim=hidden_dim,
            )

            # Mutation encoder
            self.mutation_encoder = OmicsEncoder(
                input_dim=mutation_dim,
                hidden_dims=[256, hidden_dim],
                output_dim=hidden_dim,
                dropout=dropout,
            )

            # Expression encoder
            self.expression_encoder = OmicsEncoder(
                input_dim=expression_dim,
                hidden_dims=[256, hidden_dim],
                output_dim=hidden_dim,
                dropout=dropout,
            )

            # Methylation encoder (optional)
            self.use_methylation = methylation_dim is not None
            if self.use_methylation:
                self.methylation_encoder = OmicsEncoder(
                    input_dim=methylation_dim,
                    hidden_dims=[256, hidden_dim],
                    output_dim=hidden_dim,
                    dropout=dropout,
                )

            # Fusion dimension
            fusion_dim = hidden_dim * (4 if self.use_methylation else 3)

            # Prediction head
            self.fusion = nn.Sequential(
                nn.Linear(fusion_dim, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        def forward(
            self,
            drug_data,
            mutation,
            expression,
            methylation=None,
        ):
            # Encode drug
            drug_emb = self.drug_encoder(drug_data)

            # Encode omics
            mut_emb = self.mutation_encoder(mutation)
            exp_emb = self.expression_encoder(expression)

            if self.use_methylation and methylation is not None:
                meth_emb = self.methylation_encoder(methylation)
                combined = torch.cat([drug_emb, mut_emb, exp_emb, meth_emb], dim=-1)
            else:
                combined = torch.cat([drug_emb, mut_emb, exp_emb], dim=-1)

            # Predict
            out = self.fusion(combined)
            return out.squeeze(-1)


class DeepCDRModel(BaseModel):
    """
    DeepCDR model for drug sensitivity prediction.

    Integrates molecular structure with multi-omics cell line data.
    """

    def __init__(
        self,
        drug_features: int = 75,
        mutation_dim: int = 735,
        expression_dim: int = 697,
        methylation_dim: Optional[int] = None,
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
        Initialize DeepCDR model.

        Args:
            drug_features: Number of atom features
            mutation_dim: Dimension of mutation features
            expression_dim: Dimension of expression features
            methylation_dim: Dimension of methylation features (None to disable)
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
            mutation_dim=mutation_dim,
            expression_dim=expression_dim,
            methylation_dim=methylation_dim,
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
        self.mutation_dim = mutation_dim
        self.expression_dim = expression_dim
        self.methylation_dim = methylation_dim
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

        self.model_: Optional[DeepCDRNetwork] = None

    def fit(
        self,
        X_train: Tuple[List[Data], np.ndarray, np.ndarray, Optional[np.ndarray]],
        y_train: np.ndarray,
        X_val: Optional[Tuple] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "DeepCDRModel":
        """
        Train DeepCDR model.

        Args:
            X_train: Tuple of (drug_graphs, mutation, expression, methylation)
            y_train: Training targets
            X_val: Validation data
            y_val: Validation targets

        Returns:
            self
        """
        torch.manual_seed(self.random_state)

        drug_graphs, mutation, expression, methylation = X_train

        # Infer dimensions
        if len(drug_graphs) > 0 and drug_graphs[0] is not None:
            self.drug_features = drug_graphs[0].x.shape[1]
        self.mutation_dim = mutation.shape[1]
        self.expression_dim = expression.shape[1]
        if methylation is not None:
            self.methylation_dim = methylation.shape[1]

        self.model_ = DeepCDRNetwork(
            drug_features=self.drug_features,
            mutation_dim=self.mutation_dim,
            expression_dim=self.expression_dim,
            methylation_dim=self.methylation_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)

        optimizer = optim.Adam(self.model_.parameters(), lr=self.learning_rate)
        criterion = nn.MSELoss()

        n_samples = len(drug_graphs)
        best_val_loss = float("inf")
        patience_counter = 0

        logger.info(f"Training DeepCDR on {n_samples} samples, device={self.device}")

        for epoch in range(self.epochs):
            self.model_.train()
            train_loss = 0.0

            indices = np.random.permutation(n_samples)

            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]

                # Prepare batch
                batch_graphs = [drug_graphs[j] for j in batch_indices]
                valid_mask = [g is not None for g in batch_graphs]
                batch_graphs = [g for g in batch_graphs if g is not None]

                if len(batch_graphs) == 0:
                    continue

                valid_indices = batch_indices[[i for i, v in enumerate(valid_mask) if v]]

                drug_batch = Batch.from_data_list(batch_graphs).to(self.device)
                mut_batch = torch.FloatTensor(mutation[valid_indices]).to(self.device)
                exp_batch = torch.FloatTensor(expression[valid_indices]).to(self.device)

                meth_batch = None
                if methylation is not None:
                    meth_batch = torch.FloatTensor(methylation[valid_indices]).to(self.device)

                y_batch = torch.FloatTensor(y_train[valid_indices]).to(self.device)

                optimizer.zero_grad()
                outputs = self.model_(drug_batch, mut_batch, exp_batch, meth_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                train_loss += loss.item() * len(batch_graphs)

            train_loss /= n_samples

            # Validation
            if X_val is not None and y_val is not None:
                val_loss = self._evaluate(*X_val, y_val)

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
        logger.info("DeepCDR training complete")
        return self

    def _evaluate(
        self,
        drug_graphs: List[Data],
        mutation: np.ndarray,
        expression: np.ndarray,
        methylation: Optional[np.ndarray],
        y: np.ndarray,
    ) -> float:
        """Evaluate model."""
        self.model_.eval()
        total_loss = 0.0
        n_samples = len(drug_graphs)

        with torch.no_grad():
            for i in range(0, n_samples, self.batch_size):
                batch_graphs = drug_graphs[i:i + self.batch_size]
                valid_mask = [g is not None for g in batch_graphs]
                batch_graphs = [g for g in batch_graphs if g is not None]

                if len(batch_graphs) == 0:
                    continue

                valid_count = sum(valid_mask)
                start_idx = i

                drug_batch = Batch.from_data_list(batch_graphs).to(self.device)
                mut_batch = torch.FloatTensor(mutation[start_idx:start_idx + valid_count]).to(self.device)
                exp_batch = torch.FloatTensor(expression[start_idx:start_idx + valid_count]).to(self.device)

                meth_batch = None
                if methylation is not None:
                    meth_batch = torch.FloatTensor(methylation[start_idx:start_idx + valid_count]).to(self.device)

                y_batch = torch.FloatTensor(y[start_idx:start_idx + valid_count]).to(self.device)

                outputs = self.model_(drug_batch, mut_batch, exp_batch, meth_batch)
                loss = F.mse_loss(outputs, y_batch)
                total_loss += loss.item() * valid_count

        return total_loss / n_samples

    def predict(
        self,
        X: Tuple[List[Data], np.ndarray, np.ndarray, Optional[np.ndarray]],
    ) -> np.ndarray:
        """Make predictions."""
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        drug_graphs, mutation, expression, methylation = X
        self.model_.eval()
        predictions = []

        with torch.no_grad():
            for i in range(0, len(drug_graphs), self.batch_size):
                batch_graphs = drug_graphs[i:i + self.batch_size]
                batch_graphs = [g for g in batch_graphs if g is not None]

                if len(batch_graphs) == 0:
                    continue

                valid_count = len(batch_graphs)

                drug_batch = Batch.from_data_list(batch_graphs).to(self.device)
                mut_batch = torch.FloatTensor(mutation[i:i + valid_count]).to(self.device)
                exp_batch = torch.FloatTensor(expression[i:i + valid_count]).to(self.device)

                meth_batch = None
                if methylation is not None:
                    meth_batch = torch.FloatTensor(methylation[i:i + valid_count]).to(self.device)

                outputs = self.model_(drug_batch, mut_batch, exp_batch, meth_batch)
                predictions.extend(outputs.cpu().numpy().tolist())

        return np.array(predictions)

    def get_hyperparameters(self) -> Dict[str, Any]:
        """Get current hyperparameters."""
        return {
            "drug_features": self.drug_features,
            "mutation_dim": self.mutation_dim,
            "expression_dim": self.expression_dim,
            "methylation_dim": self.methylation_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
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

    def load(self, path: Union[str, Path]) -> "DeepCDRModel":
        """Load model."""
        checkpoint = torch.load(path, map_location=self.device)

        self.model_ = DeepCDRNetwork(
            drug_features=self.drug_features,
            mutation_dim=self.mutation_dim,
            expression_dim=self.expression_dim,
            methylation_dim=self.methylation_dim,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
        ).to(self.device)
        self.model_.load_state_dict(checkpoint["model_state_dict"])
        self.is_fitted_ = True

        return self
