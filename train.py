from __future__ import annotations

from pathlib import Path
from typing import Iterable

import hydra
import hydra.utils as hu
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from sb3_contrib import MaskableRecurrentPPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from env.factory import make_env
from env.metrics import METRIC_REGISTRY
from feature_extractors.base import get_policy_kwargs


def get_device(cfg: DictConfig) -> torch.device:
    device = str(cfg.env.device).lower()
    if device != "auto":
        device = torch.device(device)
        if device.type == "cuda" and torch.cuda.is_available():
            return device
        if device.type == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return device
        return torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
        return torch.device("cuda:0")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class EpisodeInfoLogger(BaseCallback):
    """
    Custom callback that extracts per-episode stats from ``info["episode"]``
    and records them to TensorBoard under the ``rollout/`` namespace.
    """
    def __init__(self, keys: Iterable[str] | None = None, verbose: int = 0):
        super().__init__(verbose)
        self._keys = set(keys or [])

    def _on_step(self) -> bool:
        """
        Called every env step. Iterates through ``infos`` returned by
        the env and logs selected metrics.
        """
        for info in self.locals["infos"]:
            if "episode" not in info:
                continue
            for k, v in info["episode"].items():
                if k in {"r", "l"}:
                    continue
                if self._keys and k not in self._keys:
                    continue
                if np.isscalar(v):
                    self.logger.record(f"rollout/{k}", v)
        return True


@hydra.main(config_path="configs", config_name="default", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Training script that "glues" together:

    1. **TradingEnv** instances created via env.factory.make_env
    2. **MaskableRecurrentPPO** agent
    3. Custom feature extractor
    4. Callbacks for checkpointing and TensorBoard logging

    All component choices are defined in `configs/default.yaml` (Hydra).
    """
    print("============ Loaded config ============")
    print(OmegaConf.to_yaml(cfg, resolve=True))
    print("=======================================")

    # paths
    workdir = Path(hu.to_absolute_path(cfg.paths.model_dir)).resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    model_save_path = workdir / cfg.paths.model_name

    # environment
    device = get_device(cfg)
    n_envs = cfg.env.n_envs

    episodes_path = hu.to_absolute_path(cfg.env.episodes_path)

    # metrics
    metric_set_name = cfg.metrics.name
    metric_keys = METRIC_REGISTRY[metric_set_name].KEYS

    vec_env = SubprocVecEnv(
        [make_env(episodes_path, cfg, i, n_envs) for i in range(n_envs)],
        start_method="spawn",
    )
    vec_env = VecMonitor(
        vec_env,
        info_keywords=metric_keys,
    )

    # feature extractor
    policy_kwargs = get_policy_kwargs(cfg.feature_extractor.name, **cfg.feature_extractor.kwargs)

    # agent
    # @TODO: add more agents (A2C, DQN, and etc.)
    model = MaskableRecurrentPPO(
        policy="MultiInputLstmPolicy",
        env=vec_env,
        policy_kwargs=policy_kwargs,
        learning_rate=cfg.algo.learning_rate,
        n_steps=cfg.algo.n_steps,
        batch_size=cfg.algo.batch_size,
        ent_coef=cfg.algo.ent_coef,
        clip_range=cfg.algo.clip_range,
        gamma=cfg.algo.gamma,
        n_epochs=cfg.algo.n_epochs,
        device=device,
        verbose=1,
        tensorboard_log=hu.to_absolute_path(cfg.paths.logs_dir),
    )

    # callbacks
    callbacks = CallbackList(
        [
            EpisodeInfoLogger(keys=set(metric_keys)),
            CheckpointCallback(
                save_freq=cfg.algo.save_freq,
                save_path=hu.to_absolute_path(cfg.paths.checkpoint_dir),
                name_prefix=cfg.paths.model_name,
            ),
        ]
    )

    # training
    model.learn(
        total_timesteps=int(cfg.algo.total_timesteps),
        progress_bar=True,
        tb_log_name=cfg.paths.model_name,
        log_interval=1,
        callback=callbacks,
    )

    # save final model
    model.save(model_save_path)
    print(f"Model saved to {model_save_path}")


if __name__ == "__main__":
    main()
