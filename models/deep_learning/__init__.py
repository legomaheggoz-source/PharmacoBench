"""Deep learning models (PyTorch-based)."""

from models.deep_learning.mlp import MLPModel

__all__ = ["MLPModel"]

# Optional GNN models
try:
    from models.deep_learning.graphdrp import GraphDRPModel
    from models.deep_learning.deepcdr import DeepCDRModel
    __all__.extend(["GraphDRPModel", "DeepCDRModel"])
except ImportError:
    pass  # PyTorch Geometric not available
