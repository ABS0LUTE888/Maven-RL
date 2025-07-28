from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, TYPE_CHECKING

import numpy as np
import gymnasium as gym

if TYPE_CHECKING:
    from .core import TradingEnv


class ObservationBuilder(ABC):
    """
    Abstract base class for Observation Builders
    """
    def __init__(self, **kwargs):
        ...

    @abstractmethod
    def reset(self, env: "TradingEnv") -> None:
        """
        Called at the start of each episode to reset internal state
        """
        ...

    @abstractmethod
    def get_observation_space(self, env: "TradingEnv") -> gym.spaces.Box | gym.spaces.Dict:
        """
        Defines the structure of the observation returned at each step

        Returns gym.spaces.Box or gym.spaces.Dict
        """
        ...

    @abstractmethod
    def get_min_start_step(self, env: "TradingEnv") -> int:
        """
        Returns the minimum timestep index that doesn't cause any unexpected bugs
        """
        ...


    @abstractmethod
    def __call__(self, env: "TradingEnv") -> Dict[str, np.ndarray]:
        """
        Called at each step to generate the current observation
        """
        ...


class MultiTimeframeObservationBuilder(ObservationBuilder):
    """
    ObservationBuilder that combines multiple timeframes (1m and 5m) along with
    asset, prices (1m), private, and position information into a dictionary observation
    """
    def __init__(self, window_size_1m: int, window_size_5m: int, **kwargs):
        super().__init__(**kwargs)
        self.window_size_1m: int = window_size_1m
        self.window_size_5m: int = window_size_5m


    def reset(self, env: "TradingEnv") -> None:
        env.position_info[:] = 0.0
        env.slot_of_trade.clear()

    def get_observation_space(self, env: "TradingEnv") -> gym.spaces.Dict:
        return gym.spaces.Dict(
            {
                "asset": gym.spaces.Box(0, 1, shape=(len(env.asset_columns),), dtype=np.int32),
                "prices": gym.spaces.Box(-np.inf, np.inf, shape=(self.window_size_1m, len(env.price_columns)),
                                         dtype=np.float32),
                "indicators_1m": gym.spaces.Box(-np.inf, np.inf,
                                                shape=(self.window_size_1m, len(env.indicator_1m_columns)),
                                                dtype=np.float32),
                "indicators_5m": gym.spaces.Box(-np.inf, np.inf,
                                                shape=(self.window_size_5m, len(env.indicator_5m_columns)),
                                                dtype=np.float32),
                "private_info": gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
                "position_info": gym.spaces.Box(-np.inf, np.inf, shape=(env.max_num_trades, 4), dtype=np.float32),
            }
        )

    def get_min_start_step(self, env: "TradingEnv") -> int:
        return max(
            self.window_size_1m - 1,
            self.window_size_5m * 5 - 1
        )

    def __call__(self, env: "TradingEnv") -> Dict[str, np.ndarray]:
        step = env.current_step

        # 1min window
        win1m = env.episode.iloc[step - self.window_size_1m + 1: step + 1]
        prices_1m = win1m[env.price_columns].values.astype(np.float32)
        indicators_1m = win1m[env.indicator_1m_columns].values.astype(np.float32)
        asset_vec = win1m[env.asset_columns].iloc[-1].values.astype(np.float32)

        # 5min window
        offset = step % 5
        last_real = step - offset
        first_real = max(0, last_real - (self.window_size_5m - 1) * 5)
        idx_5m = range(first_real, last_real + 1, 5)
        ind_5m = env.episode.iloc[idx_5m][env.indicator_5m_columns].values.astype(np.float32)

        # private info
        norm_return = (env.equity / env.prev_equity) - 1.0 if env.prev_equity else 0.0
        n_active = len(env.trading_session.active_trades) if env.trading_session else 0
        private = np.array([norm_return, n_active], dtype=np.float32)

        # positions
        if env.trading_session:
            open_ids = {t.id for t in env.trading_session.active_trades}
        else:
            open_ids = set()
        for tid in list(env.slot_of_trade):
            if tid not in open_ids:
                env.position_info[env.slot_of_trade.pop(tid)] = 0.0

        price = env.get_price()
        empty_rows = np.where(~env.position_info.any(axis=1))[0]

        if env.trading_session:
            for trade in env.trading_session.active_trades:
                if trade.id in env.slot_of_trade:
                    slot = env.slot_of_trade[trade.id]
                elif empty_rows.size:
                    slot = int(empty_rows[0])
                    empty_rows = empty_rows[1:]
                    env.slot_of_trade[trade.id] = slot
                else:
                    continue

                rel_pnl = ((trade.margin + (
                        price - trade.entry_price) * trade.direction * trade.qty) / trade.margin) - 1.0

                env.position_info[slot] = (
                    trade.duration / env.max_position_timesteps,
                    trade.qty,
                    trade.direction,
                    rel_pnl,
                )

        return {
            "asset": asset_vec,
            "prices": prices_1m,
            "indicators_1m": indicators_1m,
            "indicators_5m": ind_5m,
            "private_info": private,
            "position_info": env.position_info.copy(),
        }


# Registry
OBSERVATION_REGISTRY: dict[str, type[ObservationBuilder]] = {
    "multi_timeframe_observation_builder": MultiTimeframeObservationBuilder,
}
