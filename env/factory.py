from __future__ import annotations

from typing import Dict, Callable, Any

from omegaconf.dictconfig import DictConfig

from .core import TradingEnv
from .masks import MASK_REGISTRY, BaseMask
from .metrics import METRIC_REGISTRY, MetricSet
from .observations import OBSERVATION_REGISTRY, ObservationBuilder
from .rewards import REWARD_REGISTRY, BaseReward


def _build(registry: Dict[str, Callable[..., Any]], name: str, kwargs: Dict[str, Any]):
    """
    Maps a string key and kwargs to an object using given registry.

    Parameters
    ----------
    registry: Dict[str, Callable[..., Any]]
        Mapping from names to classes/factory functions.
    name: str
        Key to look up inside the registry.
    kwargs: Dict[str, Any]
        Keyword arguments forwarded to the constructor.

    Returns
    -------
    Any
        Instantiated object (type depends on the registry).
    """
    try:
        cls = registry[name]
    except KeyError as exc:
        raise KeyError(f"Unknown key '{name}'. Available: {list(registry)}") from exc
    return cls(**kwargs)


def make_env(
        episodes_path: str,
        cfg: Dict | DictConfig,
        env_idx: int = 0,
        n_envs: int = 8,
):
    """
    Wraps environment creation inside a closure so it can be used with
    vectorised wrappers such as SubprocVecEnv or DummyVecEnv.

    Parameters
    ----------
    episodes_path: str
        path to the episodes.
    cfg : Dict | DictConfig
        Hydra / OmegaConf configuration node containing sub-configs for
        observations, reward, metrics, and mask.
    env_idx: int, default 0
        Which slice of data this worker should start at.
    n_envs : int, default 8
        Total number of parallel environments.

    Returns
    -------
    Callable[[], TradingEnv]
        Function that returns a fully-configured TradingEnv instance
    """
    def _init():
        obs_config = cfg.get("observations")
        obs_name = obs_config.get("name")
        obs_kwargs = obs_config.get("kwargs", {})

        reward_config = cfg.get("reward")
        reward_name = reward_config.get("name")
        reward_kwargs = reward_config.get("kwargs", {})

        metrics_config = cfg.get("metrics")
        metrics_name = metrics_config.get("name")
        metrics_kwargs = metrics_config.get("kwargs", {})

        mask_config = cfg.get("mask")
        mask_name = mask_config.get("name")
        mask_kwargs = mask_config.get("kwargs", {})

        obs_builder: ObservationBuilder = _build(
            OBSERVATION_REGISTRY, obs_name, obs_kwargs
        )
        reward_fn: BaseReward = _build(
            REWARD_REGISTRY, reward_name, reward_kwargs
        )
        metric_set: MetricSet = _build(
            METRIC_REGISTRY, metrics_name, metrics_kwargs
        )
        mask_fn: BaseMask = _build(
            MASK_REGISTRY, mask_name, mask_kwargs
        )

        # Create the environment
        return TradingEnv(
            episodes_path,
            cfg,
            start_ind=env_idx,
            stride=n_envs,
            observation_builder=obs_builder,
            reward_strategy=reward_fn,
            metric_set=metric_set,
            masking_strategy=mask_fn,
        )

    return _init


__all__ = ["make_env"]
