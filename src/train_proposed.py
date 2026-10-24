"""The proposed model shares the controlled training loop with graph ablations."""
from .train_baselines import RelationalNet, train_model

__all__ = ["RelationalNet", "train_model"]
