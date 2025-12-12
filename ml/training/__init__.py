"""
Training Package
================

Training loops and losses for multi-label classification (PyTorch).
"""

from .geographic_trainer import train_geographic_model, evaluate_geographic_model
from .losses import GeographicBCELoss

__all__ = ['train_geographic_model', 'evaluate_geographic_model', 'GeographicBCELoss']
