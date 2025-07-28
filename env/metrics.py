from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, TYPE_CHECKING, ClassVar

import numpy as np

if TYPE_CHECKING:
    from .core import TradingEnv


class MetricSet(ABC):
    """
    Abstract base class for functions that give a collection of
    post-episode performance metrics
    """
    KEYS: ClassVar[Tuple[str, ...]]

    @abstractmethod
    def __call__(self, env: "TradingEnv") -> Dict[str, float]:
        """
        Compute all metrics for the finished episode
        """
        ...


class DefaultMetricSet(MetricSet):
    """
    Computes a versatile metric set suitable for most cases
    """
    KEYS = (
        "final_equity",
        "sharpe",
        "sortino",
        "calmar",
        "drawdown_max",
        "drawdown_avg",
        "time_to_recover_avg",
        "time_to_recover_max",
        "trade_duration_avg",
        "win_rate",
        "num_trades",
        "time_between_trades_avg",
        "long_short_ratio",
    )

    def __call__(self, env: "TradingEnv") -> Dict[str, float]:
        equity_history: np.ndarray = np.asarray(env.equity_history, dtype=np.float64)
        start_eq = float(equity_history[0]) if equity_history.size else float(env.initial_balance)
        end_eq = float(equity_history[-1]) if equity_history.size else start_eq

        # returns based stuff
        if equity_history.size > 1:
            returns = np.diff(np.log(equity_history))
            m, s = returns.mean(), returns.std()
            sharpe = float(m / (s + 1e-12) * np.sqrt(len(returns))) if s else 0.0

            neg = returns[returns < 0]
            neg_s = neg.std()
            sortino = float(m / (neg_s + 1e-12) * np.sqrt(len(returns))) if neg.size else 0.0
        else:
            sharpe = sortino = 0.0

        # drawdowns
        run_max = np.maximum.accumulate(equity_history) if equity_history.size else np.array([start_eq])
        dds = (run_max - equity_history) / run_max if equity_history.size else np.array([0.0])
        max_dd = float(dds.max()) if dds.size else 0.0
        avg_dd = float(dds.mean()) if dds.size else 0.0

        # time to recover
        ttr_list: List[int] = []
        in_dd, dd_start = False, 0
        for i in range(1, len(equity_history)):
            if equity_history[i] < run_max[i - 1]:
                if not in_dd:
                    in_dd, dd_start = True, i - 1
            elif in_dd and equity_history[i] >= run_max[i - 1]:
                ttr_list.append(i - dd_start)
                in_dd = False
        ttr_avg = float(np.mean(ttr_list)) if ttr_list else 0.0
        ttr_max = float(np.max(ttr_list)) if ttr_list else 0.0

        total_return = (end_eq / start_eq) - 1.0
        calmar = float(total_return / (max_dd + 1e-12))

        # wr + trade duration
        trades = env.trade_history
        trade_durs = [t.duration for t in trades]
        wins = sum(((t.exit_price - t.entry_price) * t.direction) > 0 for t in trades)
        num_trades = len(trades)

        longs = sum(t.direction == 1 for t in trades)
        shorts = sum(t.direction == -1 for t in trades)
        long_short_ratio = float(longs / (shorts if shorts else 1))

        intervals: List[tuple[int, int]] = []
        for t in trades:
            op = getattr(t, "open_step", None)
            if op is None:
                continue
            cl = getattr(t, "close_step", None)
            if cl is None:
                cl = op + t.duration
            intervals.append((op, cl))

        time_between_trades_avg = 0.0
        if len(intervals) >= 2:
            intervals.sort(key=lambda x: x[0])
            gaps: List[int] = []
            cur_start, cur_end = intervals[0]
            for op, cl in intervals[1:]:
                if op <= cur_end:
                    cur_end = max(cur_end, cl)
                else:
                    gap_len = op - cur_end - 1
                    if gap_len > 0:
                        gaps.append(gap_len)
                    cur_start, cur_end = op, cl
            if gaps:
                time_between_trades_avg = float(np.mean(gaps))

        if time_between_trades_avg == 0.0 and num_trades > 1:
            time_between_trades_avg = float(len(equity_history) / (num_trades - 1))

        return {
            "final_equity": end_eq,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "drawdown_max": max_dd,
            "drawdown_avg": avg_dd,
            "time_to_recover_avg": ttr_avg,
            "time_to_recover_max": ttr_max,
            "trade_duration_avg": float(np.mean(trade_durs)) if trade_durs else 0.0,
            "win_rate": (wins / num_trades * 100.0) if num_trades else 0.0,
            "num_trades": float(num_trades),
            "time_between_trades_avg": time_between_trades_avg,
            "long_short_ratio": long_short_ratio,
        }


#  Registry
METRIC_REGISTRY: dict[str, type[MetricSet]] = {
    "default": DefaultMetricSet,
}
