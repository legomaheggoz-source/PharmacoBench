"""
PharmacoBench Models

Machine learning models for drug sensitivity prediction.
All models implement the BaseModel interface.
"""

from models.base_model import BaseModel
from models.traditional.ridge import RidgeModel
from models.traditional.elasticnet import ElasticNetModel
from models.traditional.random_forest import RandomForestModel
from models.traditional.xgboost_model import XGBoostModel
from models.traditional.lightgbm_model import LightGBMModel
from models.deep_learning.mlp import MLPModel

# Registry of all available models
MODEL_REGISTRY = {
    "ridge": RidgeModel,
    "elasticnet": ElasticNetModel,
    "random_forest": RandomForestModel,
    "xgboost": XGBoostModel,
    "lightgbm": LightGBMModel,
    "mlp": MLPModel,
}

# Models requiring PyTorch Geometric (optional)
try:
    from models.deep_learning.graphdrp import GraphDRPModel
    from models.deep_learning.deepcdr import DeepCDRModel
    MODEL_REGISTRY["graphdrp"] = GraphDRPModel
    MODEL_REGISTRY["deepcdr"] = DeepCDRModel
except ImportError:
    pass  # PyTorch Geometric not available

__all__ = [
    "BaseModel",
    "RidgeModel",
    "ElasticNetModel",
    "RandomForestModel",
    "XGBoostModel",
    "LightGBMModel",
    "MLPModel",
    "MODEL_REGISTRY",
]
