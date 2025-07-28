from __future__ import annotations

import importlib
from typing import Dict, Type

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

FEATURE_EXTRACTORS: Dict[str, Type[BaseFeaturesExtractor]] = {}


def register(name: str):
    """
    Decorator that registers a custom feature extractor under the given key (name)
    """
    def _decor(cls: Type[BaseFeaturesExtractor]):
        FEATURE_EXTRACTORS[name] = cls
        return cls

    return _decor


def get_policy_kwargs(name: str = "tcn", **fx_kwargs) -> Dict:
    """
    Builds policy_kwargs for Stable-Baselines3 models.

    1.  Ensures that the requested feature extractor is imported and registered.
    2.  Returns a mapping with the correct SB3 keys:
        features_extractor_class
        features_extractor_kwargs
        share_features_extractor
    """
    if name not in FEATURE_EXTRACTORS:
        try:
            importlib.import_module(f"feature_extractors.{name}")
        except ImportError:
            available = list(FEATURE_EXTRACTORS.keys())
            raise ImportError(
                f"Feature extractor '{name}' not found. "
                f"Registered extractors: {available}"
            )

    if name not in FEATURE_EXTRACTORS:
        available = list(FEATURE_EXTRACTORS.keys())
        raise KeyError(
            f"Feature extractor '{name}' is not registered. "
            f"Available: {available}"
        )

    fx_cls = FEATURE_EXTRACTORS[name]
    return {
        "features_extractor_class": fx_cls,
        "features_extractor_kwargs": fx_kwargs,
        "share_features_extractor": True,
    }


__all__ = ["register", "get_policy_kwargs", "FEATURE_EXTRACTORS"]
