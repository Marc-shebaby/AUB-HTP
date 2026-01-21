# heavy_tailed_ml/__init__.py
"""
Heavy-Tailed Machine Learning module for AUB-HTP.

This module provides robust machine learning methods for heavy-tailed data
where classical methods (that assume Gaussian-like distributions) fail.
"""

from .kmeans import KMeansHeavyTailed

__all__ = [
    "KMeansHeavyTailed",
]
