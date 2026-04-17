from pathlib import Path

import algotrading.live.mt5.mt5_session as mt5_session_module
from algotrading.backtest.backtest_broker import SymbolSpec
from algotrading.backtest.backtest_session import BacktestSession
from algotrading.core.strategy import Strategy, StrategyParams


class _NoSignalStrategy(Strategy[StrategyParams]):
    def next(self) -> None:
        return None


class _FakeMT5Broker:
    fetched_symbols: list[str] = []

    def __init__(self):
        pass

    def get_symbol_spec(
        self,
        symbol: str,
        slippage_pct: float = 0.0,
        commission_per_lot: float = 0.0,
    ) -> SymbolSpec:
        type(self).fetched_symbols.append(symbol)
        return SymbolSpec(
            value_per_point=1.5,
            spread_pct=0.2,
            slippage_pct=slippage_pct,
            contract_size=10.0,
            commission_per_lot=commission_per_lot,
        )

    def shutdown(self) -> None:
        return None


def _make_session(monkeypatch, strategies, cache_path, **kwargs):
    """Create an MT5Session with MT5Broker patched out."""
    monkeypatch.setattr(mt5_session_module, "MT5Broker", _FakeMT5Broker)
    # Patch MT5 data-fetch calls so backtest() can run without a real terminal
    monkeypatch.setattr(mt5_session_module.mt5, "copy_rates_from_pos", lambda *a, **kw: None)
    monkeypatch.setattr(mt5_session_module.mt5, "symbol_info", lambda *a, **kw: None)
    return mt5_session_module.MT5Session(
        strategies=strategies,
        primary_tf=15,
        symbol_specs_cache_path=cache_path,
        **kwargs,
    )


def _run_backtest(session, slippage_pct=0.0024, commission_per_lot=0.0):
    """Run a minimal backtest (no real data — just exercises spec resolution)."""
    return session.backtest(
        initial_balance=1_000.0,
        slippage_pct=slippage_pct,
        commission_per_lot=commission_per_lot,
        primary_count=5,
    )


def test_mt5_session_fetches_and_caches_symbol_specs(tmp_path: Path, monkeypatch) -> None:
    _FakeMT5Broker.fetched_symbols = []
    cache_path = tmp_path / "symbol_specs.json"
    strategy = _NoSignalStrategy("XAUUSD")

    session = _make_session(monkeypatch, [strategy], cache_path)
    bt = _run_backtest(session)

    assert isinstance(bt, BacktestSession)
    assert _FakeMT5Broker.fetched_symbols == ["XAUUSD"]
    assert bt.broker.value_per_point("XAUUSD") == 1.5
    assert cache_path.exists()
    cached = mt5_session_module.MT5Session._load_symbol_specs_cache(cache_path)
    assert cached["XAUUSD"].spread_pct == 0.2


def test_mt5_session_reuses_cached_symbol_specs(tmp_path: Path, monkeypatch) -> None:
    _FakeMT5Broker.fetched_symbols = []
    cache_path = tmp_path / "symbol_specs.json"
    mt5_session_module.MT5Session._save_symbol_specs_cache(
        cache_path,
        {
            "XAUUSD": SymbolSpec(
                value_per_point=2.5,
                spread_pct=0.15,
                slippage_pct=0.0,
                contract_size=100.0,
                commission_per_lot=0.0,
            )
        },
    )

    strategy = _NoSignalStrategy("XAUUSD")
    session = _make_session(monkeypatch, [strategy], cache_path)
    bt = _run_backtest(session, slippage_pct=0.0024, commission_per_lot=3.5)

    assert _FakeMT5Broker.fetched_symbols == []
    cached_spec = bt.broker._symbol_specs["XAUUSD"]
    assert cached_spec.value_per_point == 2.5
    assert cached_spec.spread_pct == 0.15
    assert cached_spec.slippage_pct == 0.0024
    assert cached_spec.commission_per_lot == 3.5


def test_mt5_session_multiple_strategies_same_symbol_fetches_spec_once(tmp_path: Path, monkeypatch) -> None:
    _FakeMT5Broker.fetched_symbols = []
    cache_path = tmp_path / "symbol_specs.json"
    strategies = [_NoSignalStrategy("XAUUSD"), _NoSignalStrategy("XAUUSD")]

    session = _make_session(monkeypatch, strategies, cache_path)
    bt = _run_backtest(session)

    assert _FakeMT5Broker.fetched_symbols == ["XAUUSD"]
    assert set(bt.broker._symbol_specs.keys()) == {"XAUUSD"}


def test_mt5_session_multiple_strategies_different_symbols_fetches_each_once(tmp_path: Path, monkeypatch) -> None:
    _FakeMT5Broker.fetched_symbols = []
    cache_path = tmp_path / "symbol_specs.json"
    strategies = [_NoSignalStrategy("XAUUSD"), _NoSignalStrategy("EURUSD")]

    session = _make_session(monkeypatch, strategies, cache_path)
    bt = _run_backtest(session)

    assert sorted(_FakeMT5Broker.fetched_symbols) == ["EURUSD", "XAUUSD"]
    assert set(bt.broker._symbol_specs.keys()) == {"XAUUSD", "EURUSD"}


def test_mt5_session_multiple_strategies_mixed_symbols_dedupes_and_covers_all(tmp_path: Path, monkeypatch) -> None:
    _FakeMT5Broker.fetched_symbols = []
    cache_path = tmp_path / "symbol_specs.json"
    strategies = [
        _NoSignalStrategy("XAUUSD"),
        _NoSignalStrategy("XAUUSD"),
        _NoSignalStrategy("EURUSD"),
    ]

    session = _make_session(monkeypatch, strategies, cache_path)
    bt = _run_backtest(session)

    assert sorted(_FakeMT5Broker.fetched_symbols) == ["EURUSD", "XAUUSD"]
    assert set(bt.broker._symbol_specs.keys()) == {"XAUUSD", "EURUSD"}
