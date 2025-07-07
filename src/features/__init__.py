"""
Feature extraction and processing modules
"""

from .extractor import VesselFeatureExtractor, test_feature_extraction
from .kalman_filter import TrajectoryKalmanFilter, test_kalman_filter

__all__ = [
    "VesselFeatureExtractor",
    "TrajectoryKalmanFilter",
    "test_feature_extraction",
    "test_kalman_filter",
]
