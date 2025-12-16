"""
PLA/PHB Predictive Modeling Package

This package provides tools for predicting mechanical and degradation
properties of PLA/PHB blends for biomedical applications.
"""

from .prediction import PLAPHBPredictor
from .optimization import ReverseDesignOptimizer

__version__ = "1.0.0"
__all__ = ["PLAPHBPredictor", "ReverseDesignOptimizer"]