"""Segmentação assistida de pólipos com superpixels e descritores de cor e textura."""

from polypseg.features import FeatureConfig, compute_features
from polypseg.segmenter import InteractiveSegmenter, SegmenterConfig
from polypseg.superpixels import SuperpixelConfig, compute_superpixels

__all__ = [
    "FeatureConfig",
    "InteractiveSegmenter",
    "SegmenterConfig",
    "SuperpixelConfig",
    "compute_features",
    "compute_superpixels",
]

__version__ = "0.1.0"
