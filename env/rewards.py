from __future__ import annotations

from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, Dict, Type, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .core import TradingEnv


class BaseReward(ABC):
    """
    Abstract base class for all reward functions
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
    def __call__(self, env: "TradingEnv", action: int) -> float:
        """
        Called at each step to compute and return the reward
        """
        ...


class OpenCloseReward(BaseReward):
    """
    Gives non-zero reward only on open/close actions,
    based on percentage change in equity since the last action
    """

    def reset(self, env):
        self._last_equity = env.equity

    def __call__(self, env, action):
        if action == 0:
            # no reward if action is "hold"
            return 0.0

        reward = float(env.equity / (self._last_equity + 1e-12)) - 1.0
        self._last_equity = env.equity
        return reward


class EquityDeltaReward(BaseReward):
    """
    Returns the percentage change in equity
    between the current and previous step, regardless of action
    """

    def reset(self, env):
        self._last_equity = env.equity

    def __call__(self, env, action):
        reward = float(env.equity / self._last_equity) - 1.0
        self._last_equity = env.equity
        return reward


class RiskAdjustedReward(BaseReward):
    """
    Reward function based on Sharpe or Sortino ratio over a rolling window of returns.
    It encourages risk-efficient strategies.
    """
    def __init__(
            self,
            window: int = 30,
            risk_free_rate: float = 0.0,
            use_sortino: bool = False,
            min_history: int = 5,
            eps: float = 1e-12,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.window = window
        self.risk_free_rate = risk_free_rate
        self.use_sortino = use_sortino
        self.min_history = min_history
        self.eps = eps

        self._rets: Deque[float] = deque(maxlen=window)
        self._prev_equity: float | None = None

    def reset(self, env):
        self._rets.clear()
        self._prev_equity = env.equity

    def __call__(self, env, action):
        if self._prev_equity is None:
            self._prev_equity = env.equity
            return 0.0

        r_t = (env.equity / self._prev_equity) - 1.0
        self._prev_equity = env.equity
        self._rets.append(r_t)

        if len(self._rets) < self.min_history:
            return 0.0

        excess = np.array(self._rets) - self.risk_free_rate

        if self.use_sortino:
            downside = excess[excess < 0.0]
            denom = np.std(downside) if downside.size else self.eps
        else:
            denom = np.std(excess)

        denom = max(denom, self.eps)
        ratio = np.mean(excess) / denom
        return float(ratio)


# Registry
REWARD_REGISTRY: Dict[str, Type[BaseReward]] = {
    "open_close": OpenCloseReward,
    "equity_delta": EquityDeltaReward,
    "risk_adjusted": RiskAdjustedReward,
}

__all__ = [
    "BaseReward",
    "OpenCloseReward",
    "EquityDeltaReward",
    "RiskAdjustedReward",
    "REWARD_REGISTRY",
]
