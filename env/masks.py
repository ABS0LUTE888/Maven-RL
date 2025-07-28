from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, Type, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from env.core import TradingEnv

__all__ = [
    "BaseMask",
    "MinTradeDurationMask",
    "NoMask",
    "MASK_REGISTRY",
]


class BaseMask(ABC):
    """
    Abstract base class for the action mask functions

    A mask is a binary NumPy array with the same length as the environment’s discrete action space.
    1 means the action is allowed, 0 means it is prohibited

    Action indices are:
    0 = Hold
    1 = Open Long
    2 = Open Short
    3 = Close All
    """
    def __init__(self, **kwargs):
        ...

    def reset(self, env: "TradingEnv") -> None:
        """
        Called by the environment at the start of each episode so the mask can
        clear internal state
        """
        ...

    @abstractmethod
    def __call__(self, env: "TradingEnv") -> np.ndarray:
        """
        Computes the action mask for the current step
        """
        ...


class MinTradeDurationMask(BaseMask):
    """
    Prohibits closing trades if at least one of them is open for less than min_trade_duration timesteps
    and prohibits opening new trades when the maximum number of trades is already reached
    """
    def __init__(self, min_trade_duration: int, **kwargs):
        super().__init__(**kwargs)
        self.min_trade_duration = min_trade_duration

    def __call__(self, env: "TradingEnv") -> np.ndarray:
        mask = np.ones(env.action_space.n, dtype=np.int8)

        can_open = (
                env.trading_session is None
                or len(env.trading_session.active_trades) < env.max_num_trades
        )

        if not can_open:
            mask[1] = 0  # open long
            mask[2] = 0  # open short

        can_close = (
                env.trading_session is not None
                and env.trading_session.active_trades
                and all(t.duration >= self.min_trade_duration for t in env.trading_session.active_trades)
        )
        if not can_close:
            mask[3] = 0  # close all

        return mask


class NoMask(BaseMask):
    """
    Always returns [1, ..., 1]; all actions are always allowed
    """
    def __call__(self, env: "TradingEnv") -> np.ndarray:
        return np.ones(env.action_space.n, dtype=np.int8)


#  Registry
MASK_REGISTRY: Dict[str, Type[BaseMask]] = {
    "min_trade_duration_mask": MinTradeDurationMask,
    "none": NoMask,
}
