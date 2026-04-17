from dataclasses import dataclass
import math

import numpy as np

from .backtest_broker import BacktestBroker


@dataclass(frozen=True)
class MonteCarloReport:
    """Monte Carlo simulated equity paths from historical trade returns."""

    initial_balance: float
    horizon_trades: int
    paths: np.ndarray
    sampled_returns: np.ndarray
    final_balances: np.ndarray

    def quantile_path(self, q: float) -> np.ndarray:
        if q < 0.0 or q > 1.0:
            raise ValueError("q must be within [0, 1]")
        return np.quantile(self.paths, q, axis=0)


def _trade_returns_from_broker(broker: BacktestBroker) -> np.ndarray:
    if not broker.trade_log:
        raise ValueError("Broker has no closed trades for Monte Carlo simulation")

    returns: list[float] = []
    equity = float(broker._initial_balance)
    for trade in broker.trade_log:
        if equity <= 0.0 or not math.isfinite(equity):
            break
        pnl = float(trade.pnl)
        ret = pnl / equity
        returns.append(ret)
        equity += pnl

    if not returns:
        raise ValueError("Unable to derive trade returns from broker trade log")
    return np.asarray(returns, dtype=np.float64)


def simulate_monte_carlo_from_broker(
    broker: BacktestBroker,
    n_paths: int = 1_000,
    horizon_trades: int | None = None,
    random_state: int | None = None,
) -> MonteCarloReport:
    """Bootstrap trade-level returns to simulate alternative equity trajectories."""
    if n_paths <= 0:
        raise ValueError("n_paths must be > 0")

    historical_returns = _trade_returns_from_broker(broker)
    horizon = horizon_trades or int(historical_returns.size)
    if horizon <= 0:
        raise ValueError("horizon_trades must be > 0")

    rng = np.random.default_rng(random_state)
    sampled_returns = rng.choice(historical_returns, size=(n_paths, horizon), replace=True)

    paths = np.empty((n_paths, horizon + 1), dtype=np.float64)
    paths[:, 0] = float(broker._initial_balance)

    for i in range(horizon):
        next_values = paths[:, i] * (1.0 + sampled_returns[:, i])
        paths[:, i + 1] = np.maximum(next_values, 0.0)

    return MonteCarloReport(
        initial_balance=float(broker._initial_balance),
        horizon_trades=horizon,
        paths=paths,
        sampled_returns=sampled_returns,
        final_balances=paths[:, -1],
    )
