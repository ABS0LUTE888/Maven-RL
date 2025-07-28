from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import gymnasium as gym
import numpy as np


@dataclass
class Trade:
    entry_price: float
    qty: float
    margin: float
    direction: int  # +1 long, ‑1 short
    open_step: int
    close_step: Optional[int] = None
    duration: int = 0
    liquidated: bool = False
    exit_price: Optional[float] = None
    id: Optional[int] = None


@dataclass
class TradingSession:
    start_balance: float
    duration: int = 0
    active_trades: List[Trade] = field(default_factory=list)
    closed_trades: List[Trade] = field(default_factory=list)
    liquidated_trades: List[Trade] = field(default_factory=list)

    def active_pnl(self, price: float) -> float:
        return sum((price - t.entry_price) * t.direction * t.qty for t in self.active_trades)

    def active_qty(self) -> float:
        return sum(t.qty for t in self.active_trades)

    def active_margin(self) -> float:
        return sum(t.margin for t in self.active_trades)

    def add_duration(self) -> None:
        self.duration += 1
        for t in self.active_trades:
            t.duration += 1

    def is_active(self) -> bool:
        return bool(self.active_trades)


# Core environment
class TradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
            self,
            episodes_path: str,
            config: Dict,
            *,
            start_ind: int = 0,
            stride: int = 8,
            observation_builder,
            reward_strategy,
            metric_set,
            masking_strategy,
    ) -> None:
        super().__init__()

        # Factory stuff
        self.obs_builder = observation_builder
        self.reward_fn = reward_strategy
        self.metric_set = metric_set
        self.mask_fn = masking_strategy

        # Trading params
        self.leverage: float = config["env"]["leverage"]
        self.maintenance_margin_rate: float = config["env"]["maintenance_margin_rate"]
        self.open_fee: float = config["env"]["open_fee"]
        self.close_fee: float = config["env"]["close_fee"]

        self.min_trades_required: int = config["env"]["min_trades_required"]
        self.initial_balance: float = config["env"]["initial_balance"]
        self.max_position_timesteps: int = config["env"]["max_position_timesteps"]
        self.max_num_trades: int = config["env"]["max_num_trades"]

        # Episodes
        self.episodes_path = episodes_path
        self.next_episode = start_ind
        self.stride = stride
        with open(episodes_path, "rb") as f:
            _df0 = pickle.load(f)[0]

        # Column groups
        self.price_columns = ["1m_z_open", "1m_z_high", "1m_z_low", "1m_z_close"]
        self.asset_columns = [c for c in _df0.columns if c.startswith("asset_")]

        # @TODO: move indicator_1m_columns and indicator_1m_columns to observations.py
        self.indicator_1m_columns = [
            c for c in _df0.columns
            if c.startswith("1m_") and c not in self.price_columns + ["1m_close"]
        ]
        self.indicator_5m_columns = [c for c in _df0.columns if c.startswith("5m_") and c != "5m_close"]

        # Action Space + Observations
        # @TODO: move action_space to a separate file
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = self.obs_builder.get_observation_space(self)

        #  Internal state
        self.episodes: List = []  # All episodes
        self.lazy_loaded = False

        self.episode = None  # Current episode
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.prev_equity = self.initial_balance

        self.trading_session: Optional[TradingSession] = None
        self.current_step = 0

        # Bookkeeping
        self.done = False
        self.num_trades = 0
        self.liquidations = 0
        self.position_info = np.zeros((self.max_num_trades, 4), dtype=np.float32)
        self.slot_of_trade: Dict[int, int] = {}
        self.trade_history: List[Trade] = []
        self.equity_history: List[float] = [self.initial_balance]

        self.reset()

    # Helpers
    def _lazy_load(self) -> None:
        if self.lazy_loaded:
            return
        with open(self.episodes_path, "rb") as f:
            self.episodes = [df.reset_index(drop=True) for df in pickle.load(f)]
        self.lazy_loaded = True

    def get_price(self) -> float:
        return float(self.episode.loc[self.current_step, "1m_close"])

    # Actions
    def _open_position(self, margin: float, direction: int) -> None:
        if len(self.trading_session.active_trades) >= self.max_num_trades:
            return
        price = self.get_price()
        qty = margin * self.leverage / price
        open_cost = qty * price * self.open_fee

        if margin + open_cost > self.balance:
            return

        self.balance -= margin + open_cost
        self.trading_session.active_trades.append(
            Trade(
                entry_price=price,
                qty=qty,
                margin=margin,
                direction=direction,
                id=self.num_trades,
                open_step=self.current_step,
            )
        )
        self.num_trades += 1

    def _close_position(self, trade: Trade) -> None:
        price = self.get_price()
        trade.exit_price = price
        trade.close_step = self.current_step
        close_cost = trade.qty * price * self.close_fee
        pnl = (price - trade.entry_price) * trade.direction * trade.qty
        self.balance += trade.margin + pnl - close_cost
        self.trading_session.active_trades.remove(trade)
        self.trading_session.closed_trades.append(trade)
        self.trade_history.append(trade)

    def _close_all_positions(self) -> None:
        if not self.trading_session:
            return
        for trade in self.trading_session.active_trades[:]:
            self._close_position(trade)
        if not self.trading_session.active_trades:
            self.trading_session = None
            self.position_info.fill(0.0)
            self.slot_of_trade.clear()

    def _liquidate_position(self, trade: Trade, price: float) -> None:
        self.liquidations += 1
        trade.liquidated = True
        trade.exit_price = price
        trade.close_step = self.current_step
        close_cost = trade.qty * price * self.close_fee
        pnl = (price - trade.entry_price) * trade.direction * trade.qty
        self.balance += pnl - close_cost
        self.trading_session.active_trades.remove(trade)
        self.trading_session.liquidated_trades.append(trade)
        self.trade_history.append(trade)

    def _check_liquidation(self) -> None:
        if not self.trading_session:
            return
        price = self.get_price()
        for trade in self.trading_session.active_trades[:]:
            notional = trade.margin * self.leverage
            maintenance = notional * self.maintenance_margin_rate
            pnl = (price - trade.entry_price) * trade.direction * trade.qty
            if trade.margin + pnl < maintenance:
                self._liquidate_position(trade, price)

    # Gymnasium API
    def reset(self, *, seed: int | None = None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
        self._lazy_load()

        min_start_step = self.obs_builder.get_min_start_step(self)

        ep_idx = self.next_episode % len(self.episodes)
        self.next_episode += self.stride
        self.episode = self.episodes[ep_idx]

        if len(self.episode) <= min_start_step:
            raise ValueError(
                f"Episode too short: ({len(self.episode)} steps)"
            )

        # Initialize state
        self.balance = self.initial_balance
        self.equity = self.initial_balance
        self.prev_equity = self.initial_balance
        self.trading_session = None
        self.current_step = min_start_step
        self.done = False

        # Reset metrics
        self.num_trades = 0
        self.liquidations = 0
        self.position_info.fill(0.0)
        self.slot_of_trade.clear()
        self.trade_history.clear()
        self.equity_history = [self.initial_balance]

        if hasattr(self.reward_fn, "reset"):
            self.reward_fn.reset(self)
        if hasattr(self.obs_builder, "reset"):
            self.obs_builder.reset(self)
        if hasattr(self.mask_fn, "reset"):
            self.mask_fn.reset(self)

        observation = self.obs_builder(self)
        return observation, {}

    def action_masks(self):
        return self.mask_fn(self)

    def step(self, action: int):
        self._check_liquidation()
        if self.trading_session:
            self.trading_session.add_duration()

        if self.current_step >= len(self.episode) - 1:
            if self.trading_session:
                self._close_all_positions()
            self.done = True

        # Skip action processing if episode done
        if not self.done:
            if action == 3:  # Close all
                self._close_all_positions()
            elif action in (1, 2):  # Open long/short
                if self.trading_session is None:
                    self.trading_session = TradingSession(start_balance=self.balance)
                if len(self.trading_session.active_trades) < self.max_num_trades:
                    margin = 0.1 * self.balance
                    self._open_position(margin, 1 if action == 1 else -1)
            else:  # Hold (action=0)
                if self.trading_session:
                    for trade in self.trading_session.active_trades[:]:
                        if trade.duration >= self.max_position_timesteps:
                            self._close_position(trade)

        # Update equity
        price = self.get_price()
        if self.trading_session:
            self.equity = self.balance + self.trading_session.active_pnl(price) + self.trading_session.active_margin()
        else:
            self.equity = self.balance
        self.equity_history.append(self.equity)

        if self.equity <= 0:
            if self.trading_session:
                self._close_all_positions()
            self.done = True

        reward = self.reward_fn(self, action)

        if not self.done:
            self.current_step += 1
        else:
            self.current_step = min(self.current_step, len(self.episode) - 1)

        observation = self.obs_builder(self)

        info = {
            "balance": self.balance,
            "equity": self.equity,
            "num_trades": self.num_trades,
            "liquidations": self.liquidations,
            "action_mask": self.action_masks(),
        }
        if self.done:
            info.update(self.metric_set(self))

        return observation, reward, self.done, False, info
