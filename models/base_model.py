"""
Base Model Interface

Abstract base class that all PharmacoBench models must implement.
"""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np

logger = logging.getLogger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all drug sensitivity prediction models.

    All models must implement:
    - fit(): Train the model
    - predict(): Make predictions
    - get_hyperparameters(): Return current hyperparameters
    - save(): Save model to disk
    - load(): Load model from disk

    Optional:
    - get_default_hyperparameters(): Return default hyperparameter grid
    - get_feature_importance(): Return feature importances
    """

    def __init__(self, **kwargs):
        """
        Initialize model with hyperparameters.

        Args:
            **kwargs: Model-specific hyperparameters
        """
        self.hyperparameters = kwargs
        self.is_fitted_ = False
        self.model_ = None

    @property
    def name(self) -> str:
        """Model name for display."""
        return self.__class__.__name__

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> "BaseModel":
        """
        Train the model.

        Args:
            X_train: Training features (n_samples, n_features)
            y_train: Training targets (n_samples,)
            X_val: Optional validation features
            y_val: Optional validation targets

        Returns:
            self
        """
        pass

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions.

        Args:
            X: Features (n_samples, n_features)

        Returns:
            Predictions (n_samples,)
        """
        pass

    @abstractmethod
    def get_hyperparameters(self) -> Dict[str, Any]:
        """
        Get current hyperparameters.

        Returns:
            Dict of hyperparameter names to values
        """
        pass

    @abstractmethod
    def save(self, path: Union[str, Path]) -> None:
        """
        Save model to disk.

        Args:
            path: File path to save model
        """
        pass

    @abstractmethod
    def load(self, path: Union[str, Path]) -> "BaseModel":
        """
        Load model from disk.

        Args:
            path: File path to load model from

        Returns:
            self
        """
        pass

    @classmethod
    def get_default_hyperparameters(cls) -> Dict[str, Any]:
        """
        Get default hyperparameter grid for tuning.

        Returns:
            Dict of hyperparameter names to lists of values to try
        """
        return {}

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importances if available.

        Returns:
            Feature importance array or None if not available
        """
        return None

    def __repr__(self) -> str:
        params = ", ".join(f"{k}={v}" for k, v in self.hyperparameters.items())
        return f"{self.name}({params})"


class ModelRegistry:
    """Registry for model classes."""

    _models: Dict[str, type] = {}

    @classmethod
    def register(cls, name: str, model_class: type) -> None:
        """Register a model class."""
        cls._models[name] = model_class

    @classmethod
    def get(cls, name: str) -> type:
        """Get a model class by name."""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available: {list(cls._models.keys())}")
        return cls._models[name]

    @classmethod
    def list_models(cls) -> list:
        """List all registered model names."""
        return list(cls._models.keys())

    @classmethod
    def create(cls, name: str, **kwargs) -> BaseModel:
        """Create a model instance by name."""
        model_class = cls.get(name)
        return model_class(**kwargs)
