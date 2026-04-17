from datetime import datetime, timezone
from typing import TypedDict
import numpy as np
import pytest

from algotrading.backtest.backtest_broker import BacktestBroker, SymbolSpec
from algotrading.core.bars import Bars
from algotrading.core.broker import BrokerView, WarmupBrokerView, SignalExecutionErrorReason
from algotrading.core.position import Position
from algotrading.core.signal import EntrySignal, SignalType
from algotrading.core.strategy import Strategy, StrategyParams
from algotrading.indicators.sma import SMA


# ---------------------------------------------------------------------------
# Broker-aware helpers
# ---------------------------------------------------------------------------

def make_broker(initial_balance: float = 10_000) -> BacktestBroker:
    return BacktestBroker(initial_balance=initial_balance)


def _ts(s: str = "2024-01-01T00:00:00") -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def feed_bar(strategy: "Strategy", symbol: str, open=1.0, high=2.0, low=0.5, close=1.5) -> None:
    """Feed one completed bar through the full broker+strategy pipeline.

    Mirrors the feed_data() pattern in main.py: broker.on_bar() first (pending
    fills + SL/TP), then strategy.on_bar() (indicators + next()).  MARKET orders
    queued in next() on bar N will fill at bar N+1's open via this helper.
    """
    import numpy as np
    strategy.broker.on_bar(symbol, open=open, high=high, low=low, close=close)
    strategy.on_bar(time=np.datetime64("2024-01-01"), open=open, high=high, low=low, close=close)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _Bar(TypedDict, total=False):
    time: np.datetime64
    open: float
    high: float
    low: float
    close: float


def make_bar(time="2024-01-01", open=1.0, high=2.0, low=0.5, close=1.5, **extra) -> _Bar:
    return _Bar(time=np.datetime64(time), open=open, high=high, low=low, close=close, **extra)  # type: ignore[misc]


class _NoSignalStrategy(Strategy[StrategyParams]):
    """Minimal concrete strategy that never signals."""
    def next(self) -> None:
        return None


class _CollectingStrategy(Strategy[StrategyParams]):
    """Records every value next() sees for assertions."""
    def __init__(self, symbol):
        super().__init__(symbol)
        self.next_calls = 0
        self.last_close: float = float("nan")

    def next(self) -> None:
        self.next_calls += 1
        self.last_close = float(self.bars.close[-1])
        return None


# ---------------------------------------------------------------------------
# Basic bar feeding
# ---------------------------------------------------------------------------

class TestOnBar:
    def test_bars_appended(self):
        s = _NoSignalStrategy("EURUSD")
        s.on_bar(**make_bar(close=1.1))
        s.on_bar(**make_bar(close=1.2))
        assert len(s.bars) == 2
        assert s.bars.close[-1] == pytest.approx(1.2)

    def test_next_called_with_no_indicators(self):
        s = _CollectingStrategy("EURUSD")
        s.on_bar(**make_bar(close=1.5))
        assert s.next_calls == 1
        assert s.last_close == pytest.approx(1.5)


# ---------------------------------------------------------------------------
# Dynamic extra fields via add_field
# ---------------------------------------------------------------------------

class TestDynamicFields:
    def test_add_field_and_feed_via_on_bar(self):
        s = _NoSignalStrategy("EURUSD")
        s.bars.add_field("noncomm_net", int)
        s.on_bar(**make_bar(), noncomm_net=500)
        assert s.bars["noncomm_net"][-1] == 500

    def test_add_multiple_fields(self):
        s = _NoSignalStrategy("EURUSD")
        s.bars.add_field("noncomm_net", int)
        s.bars.add_field("spread", float)
        s.on_bar(**make_bar(), noncomm_net=300, spread=0.0003)
        assert s.bars["noncomm_net"][-1] == 300
        assert s.bars["spread"][-1] == pytest.approx(0.0003)

    def test_add_field_after_data_raises(self):
        s = _NoSignalStrategy("EURUSD")
        s.on_bar(**make_bar())
        with pytest.raises(RuntimeError):
            s.bars.add_field("noncomm_net", int)

    def test_getitem_unknown_field_raises(self):
        s = _NoSignalStrategy("EURUSD")
        with pytest.raises(KeyError):
            _ = s.bars["nonexistent"]


# ---------------------------------------------------------------------------
# bars.append() — mutate last row in place
# ---------------------------------------------------------------------------

class TestBarsSet:
    def test_set_mutates_last_row(self):
        s = _NoSignalStrategy("EURUSD")
        s.bars.add_field("noncomm_net", int)
        s.on_bar(**make_bar())          # appends, noncomm_net defaults to 0
        s.bars.append(noncomm_net=999)
        assert s.bars["noncomm_net"][-1] == 999
        assert len(s.bars) == 1        # _size unchanged

    def test_set_before_any_data_raises(self):
        s = _NoSignalStrategy("EURUSD")
        s.bars.add_field("noncomm_net", int)
        with pytest.raises(RuntimeError):
            s.bars.append(noncomm_net=1)


# ---------------------------------------------------------------------------
# update_bars hook
# ---------------------------------------------------------------------------

class TestUpdateBars:
    def test_hook_called_before_next(self):
        order = []

        class HookStrategy(Strategy[StrategyParams]):
            def __init__(self, symbol):
                super().__init__(symbol)
                self.bars.add_field("flag", int)

            def update_bars(self):
                order.append("update_bars")
                self.bars.append(flag=42)

            def next(self):
                order.append("next")
                assert self.bars["flag"][-1] == 42
                return None

        s = HookStrategy("EURUSD")
        s.on_bar(**make_bar())
        assert order == ["update_bars", "next"]


# ---------------------------------------------------------------------------
# Indicators sourced from primary bars
# ---------------------------------------------------------------------------

class TestIndicators:
    def test_sma_on_close(self):
        class SMAStrategy(Strategy[StrategyParams]):
            def __init__(self, symbol):
                super().__init__(symbol)
                self.sma = self.I(SMA(3), source="close")

            def next(self):
                return None

        s = SMAStrategy("EURUSD")
        for close in [1.0, 2.0, 3.0]:
            s.on_bar(**make_bar(close=close))
        assert s.sma[-1] == pytest.approx(2.0)

    def test_sma_on_dynamic_field(self):
        class COTStrategy(Strategy[StrategyParams]):
            def __init__(self, symbol):
                super().__init__(symbol)
                self.bars.add_field("noncomm_net", float)
                self.sma_net = self.I(SMA(3), source="noncomm_net")

            def next(self):
                return None

        s = COTStrategy("EURUSD")
        for v in [100.0, 200.0, 300.0]:
            s.on_bar(**make_bar(), noncomm_net=v)
        assert s.sma_net[-1] == pytest.approx(200.0)

    def test_next_suppressed_during_warmup(self):
        class SMAStrategy(Strategy[StrategyParams]):
            def __init__(self, symbol):
                super().__init__(symbol)
                self.sma = self.I(SMA(3), source="close")
                self.next_calls = 0

            def next(self):
                self.next_calls += 1
                return None

        s = SMAStrategy("EURUSD")
        s.on_bar(**make_bar(close=1.0))
        s.on_bar(**make_bar(close=2.0))
        assert s.next_calls == 0       # SMA still warming up
        s.on_bar(**make_bar(close=3.0))
        assert s.next_calls == 1       # SMA ready

    def test_indicator_chaining(self):
        class ChainStrategy(Strategy[StrategyParams]):
            def __init__(self, symbol):
                super().__init__(symbol)
                self.sma = self.I(SMA(2), source="close")
                self.sma2 = self.I(SMA(2), source=self.sma)

            def next(self):
                return None

        s = ChainStrategy("EURUSD")
        for close in [1.0, 2.0, 3.0, 4.0, 5.0]:
            s.on_bar(**make_bar(close=close))
        # sma(period=2):  [nan, 1.5, 2.5, 3.5, 4.5]
        # sma2(period=2): [nan, nan, nan, mean(2.5,3.5)=3.0, mean(3.5,4.5)=4.0]
        assert s.sma2[-1] == pytest.approx(4.0)


# ---------------------------------------------------------------------------
# Multi-timeframe: new_bars()
# ---------------------------------------------------------------------------

class TestMultiTimeframe:
    def test_new_bars_registered(self):
        s = _NoSignalStrategy("EURUSD")
        d1 = s.new_bars("D1")
        assert "D1" in s.secondary_bars
        assert s.secondary_bars["D1"] is d1

    def test_indicator_sourced_from_secondary_bars(self):
        """Engine simulation: each D1 bar is fed before its corresponding primary bar.
        The D1 SMA(3) should advance once per D1 bar, not once per primary bar."""
        class MTFStrategy(Strategy[StrategyParams]):
            def __init__(self, symbol):
                super().__init__(symbol)
                self.bars_d1 = self.new_bars("D1")
                self.trend = self.I(SMA(3), source="close", bars=self.bars_d1)
                self.next_calls = 0

            def next(self):
                self.next_calls += 1

        s = MTFStrategy("EURUSD")

        # Simulate engine interleaving: feed one D1 bar then one primary bar each day.
        # SMA(3) warms up after 3 D1 values → next() first fires on the 3rd primary bar.
        for close in [10.0, 20.0, 30.0]:
            s.bars_d1.update(time=np.datetime64("2024-01-01"), open=close, high=close, low=close, close=close)
            s.on_bar(**make_bar(close=1.0))

        assert s.trend[-1] == pytest.approx(20.0)   # SMA(10, 20, 30)
        assert s.next_calls == 1                      # only fired when SMA ready (3rd bar)
        assert s.secondary_bars["D1"] is s.bars_d1

    def test_secondary_bars_batch_fed_before_primary(self):
        """If engine feeds N secondary bars before the next primary bar, all N must
        advance the indicator — not just the last one."""
        class MTFStrategy(Strategy[StrategyParams]):
            def __init__(self, symbol):
                super().__init__(symbol)
                self.bars_d1 = self.new_bars("D1")
                self.trend = self.I(SMA(3), source="close", bars=self.bars_d1)
                self.next_calls = 0

            def next(self):
                self.next_calls += 1

        s = MTFStrategy("EURUSD")

        # All 3 D1 bars arrive before any primary bar (e.g. catch-up on startup)
        for close in [10.0, 20.0, 30.0]:
            s.bars_d1.update(time=np.datetime64("2024-01-01"), open=close, high=close, low=close, close=close)

        # Single primary bar triggers replay of all 3 D1 bars into the SMA
        s.on_bar(**make_bar(close=1.0))

        assert s.trend[-1] == pytest.approx(20.0)   # SMA(10, 20, 30) — not SMA(30)
        assert s.next_calls == 1

    def test_next_suppressed_until_secondary_indicator_ready(self):
        """next() must not fire while the secondary indicator is still nan,
        even if the primary indicator is already valid."""
        class MTFStrategy(Strategy[StrategyParams]):
            def __init__(self, symbol):
                super().__init__(symbol)
                self.bars_d1 = self.new_bars("D1")
                self.fast = self.I(SMA(1), source="close")            # primary, ready bar 1
                self.trend = self.I(SMA(3), source="close", bars=self.bars_d1)  # secondary, needs 3 D1 bars
                self.next_calls = 0

            def next(self):
                self.next_calls += 1

        s = MTFStrategy("EURUSD")

        # Feed 2 primary bars with 1 D1 bar each — secondary SMA(3) still nan
        for close in [10.0, 20.0]:
            s.bars_d1.update(time=np.datetime64("2024-01-01"), open=close, high=close, low=close, close=close)
            s.on_bar(**make_bar(close=1.0))

        assert s.next_calls == 0   # secondary SMA not ready yet

        # 3rd D1 bar → SMA(3) becomes valid → next() fires
        s.bars_d1.update(time=np.datetime64("2024-01-01"), open=30.0, high=30.0, low=30.0, close=30.0)
        s.on_bar(**make_bar(close=1.0))

        assert s.next_calls == 1

    def test_next_suppressed_when_secondary_bars_empty(self):
        """next() must not fire at all if no secondary bars have been fed yet,
        even with a valid primary indicator (start-date mismatch scenario)."""
        class MTFStrategy(Strategy[StrategyParams]):
            def __init__(self, symbol):
                super().__init__(symbol)
                self.bars_d1 = self.new_bars("D1")
                self.fast = self.I(SMA(1), source="close")
                self.trend = self.I(SMA(1), source="close", bars=self.bars_d1)
                self.next_calls = 0

            def next(self):
                self.next_calls += 1

        s = MTFStrategy("EURUSD")

        # Multiple primary bars, but D1 never populated
        for _ in range(5):
            s.on_bar(**make_bar(close=1.0))

        assert s.next_calls == 0


# ---------------------------------------------------------------------------
# Broker attachment
# ---------------------------------------------------------------------------

class TestBrokerAttachment:
    def test_broker_raises_if_not_attached(self):
        s = _NoSignalStrategy("EURUSD")
        with pytest.raises(RuntimeError):
            _ = s.broker

    def test_warmup_broker_view_attaches(self):
        s = _NoSignalStrategy("EURUSD")
        WarmupBrokerView(s)
        assert s._broker is not None
        assert s.broker.balance == 0.0
        assert s.broker.positions == []

    def test_broker_view_attaches(self):
        s = _NoSignalStrategy("EURUSD")
        broker = make_broker()
        BrokerView(broker, s)
        assert s._broker is not None
        assert s.broker.balance == pytest.approx(10_000)

    def test_broker_view_replaces_warmup(self):
        s = _NoSignalStrategy("EURUSD")
        WarmupBrokerView(s)
        broker = make_broker()
        BrokerView(broker, s)
        assert s.broker.balance == pytest.approx(10_000)


# ---------------------------------------------------------------------------
# buy() / sell() → signal submission
# ---------------------------------------------------------------------------

class _BuyOnFirstBar(Strategy[StrategyParams]):
    def __init__(self, symbol, qty=1.0, sl=None, tp=None):
        super().__init__(symbol)
        self._qty = qty
        self._sl = sl
        self._tp = tp
        self.opened: list[Position] = []
        self.errors: list[SignalExecutionErrorReason] = []

    def next(self):
        if len(self.bars) == 1:
            self.buy(self._qty, sl_price=self._sl, tp_price=self._tp)

    def on_signal_execution_success(self, signal, position):
        self.opened.append(position)

    def on_signal_execution_error(self, signal, reason):
        self.errors.append(reason)


class TestBuySell:
    def test_buy_opens_position(self):
        s = _BuyOnFirstBar("EURUSD", qty=1.0)
        BrokerView(make_broker(), s)
        feed_bar(s, "EURUSD", close=1.1000)   # bar 1: next() queues MARKET
        feed_bar(s, "EURUSD", close=1.1050)   # bar 2: broker.on_bar fills at open
        assert len(s.opened) == 1
        assert s.opened[0].is_long
        assert s.opened[0].qty == pytest.approx(1.0)

    def test_sell_opens_short_position(self):
        class _SellFirst(Strategy[StrategyParams]):
            def __init__(self, symbol):
                super().__init__(symbol)
                self.opened: list[Position] = []
            def next(self):
                if len(self.bars) == 1:
                    self.sell(1.0)
            def on_signal_execution_success(self, signal, position):
                self.opened.append(position)

        s = _SellFirst("EURUSD")
        BrokerView(make_broker(), s)
        feed_bar(s, "EURUSD", close=1.1000)   # bar 1: next() queues MARKET
        feed_bar(s, "EURUSD", close=1.1050)   # bar 2: fills at open
        assert len(s.opened) == 1
        assert s.opened[0].is_short

    def test_buy_during_warmup_is_silent(self):
        s = _BuyOnFirstBar("EURUSD", qty=1.0)
        WarmupBrokerView(s)
        s.on_bar(**make_bar(close=1.1000))
        assert s.opened == []
        assert s.errors == []

    def test_insufficient_margin_fires_error(self):
        class _RawBuyOnFirstBar(Strategy[StrategyParams]):
            def __init__(self, symbol):
                super().__init__(symbol)
                self.opened: list[Position] = []
                self.errors: list[SignalExecutionErrorReason] = []

            def next(self):
                if len(self.bars) == 1:
                    self.broker.submit_signal(EntrySignal(
                        strategy_id=self.id,
                        symbol=self.symbol,
                        direction="LONG",
                        type=SignalType.MARKET,
                        exclusive=True,
                        qty=1_000_000.0,
                        price=float(self.bars.close[-1]),
                    ))

            def on_signal_execution_success(self, signal, position):
                self.opened.append(position)

            def on_signal_execution_error(self, signal, reason):
                self.errors.append(reason)

        s = _RawBuyOnFirstBar("EURUSD")
        broker = BacktestBroker(
            initial_balance=100,
            leverage=0.01,
            symbol_specs={"EURUSD": SymbolSpec(value_per_point=10.0)},
        )
        BrokerView(broker, s)
        feed_bar(s, "EURUSD", close=1.1000)   # bar 1: queues MARKET
        feed_bar(s, "EURUSD", close=1.1050)   # bar 2: fill attempt → insufficient margin
        assert s.opened == []
        assert len(s.errors) == 1
        assert s.errors[0] == SignalExecutionErrorReason.INSUFFICIENT_FUNDS

    def test_exclusive_buy_closes_existing_position(self):
        """exclusive=True (default) should close any open position at the next bar open.

        Timeline:
          bar 1 next() → MARKET queued (nothing to close yet)
          bar 2 broker.on_bar → MARKET 1 fills at open → position 1 opens
               next()       → buy(exclusive=True) queues MARKET 2 (position 1 stays open)
          bar 3 broker.on_bar → closes position 1 at open, then fills MARKET 2
        """
        positions_on_close: list[Position] = []

        class _FlipStrategy(Strategy[StrategyParams]):
            def __init__(self, symbol):
                super().__init__(symbol)

            def next(self):
                self.buy(1.0)  # exclusive=True by default

            def on_position_close_success(self, position):
                positions_on_close.append(position)

        s = _FlipStrategy("EURUSD")
        broker = make_broker()
        BrokerView(broker, s)

        feed_bar(s, "EURUSD", close=1.1000)
        feed_bar(s, "EURUSD", open=1.1000, high=1.1050, low=1.0950, close=1.1050)
        assert len(positions_on_close) == 0
        assert len(broker.get_positions("EURUSD", s.id)) == 1

        feed_bar(s, "EURUSD", open=1.1060, high=1.1100, low=1.1000, close=1.1080)
        assert len(positions_on_close) == 1

    def test_pending_market_fill_does_not_use_known_close_for_margin(self):
        errors: list[SignalExecutionErrorReason] = []

        class _LeakyMarginStrategy(Strategy[StrategyParams]):
            def next(self):
                if len(self.bars) == 1:
                    self.broker.submit_signal(EntrySignal(
                        strategy_id=self.id,
                        symbol=self.symbol,
                        direction="LONG",
                        type=SignalType.MARKET,
                        exclusive=False,
                        qty=1.0,
                        price=float(self.bars.close[-1]),
                    ))
                elif len(self.bars) == 2:
                    self.broker.submit_signal(EntrySignal(
                        strategy_id=self.id,
                        symbol=self.symbol,
                        direction="LONG",
                        type=SignalType.MARKET,
                        exclusive=False,
                        qty=6.0,
                        price=float(self.bars.close[-1]),
                    ))

            def on_signal_execution_error(self, signal, reason):
                errors.append(reason)

        broker = BacktestBroker(initial_balance=1_000.0, leverage=1.0)
        s = _LeakyMarginStrategy("EURUSD")
        BrokerView(broker, s)

        feed_bar(s, "EURUSD", open=100.0, high=100.0, low=100.0, close=100.0)
        feed_bar(s, "EURUSD", open=100.0, high=200.0, low=100.0, close=200.0)
        feed_bar(s, "EURUSD", open=200.0, high=400.0, low=200.0, close=400.0)

        assert errors == [SignalExecutionErrorReason.INSUFFICIENT_FUNDS]
        assert len(broker.get_positions("EURUSD", s.id)) == 1


# ---------------------------------------------------------------------------
# SL/TP callbacks via BrokerView.on_bar
# ---------------------------------------------------------------------------

class _SlTpStrategy(Strategy[StrategyParams]):
    def __init__(self, symbol):
        super().__init__(symbol)
        self.sl_hits: list[tuple[Position, float]] = []
        self.tp_hits: list[tuple[Position, float]] = []
        self.close_successes: list[Position] = []

    def next(self):
        if len(self.bars) == 1:
            self.buy(1.0, sl_price=1.0800, tp_price=1.1200)

    def on_sl_hit(self, position, exit_price):
        self.sl_hits.append((position, exit_price))

    def on_tp_hit(self, position, exit_price):
        self.tp_hits.append((position, exit_price))

    def on_position_close_success(self, position):
        self.close_successes.append(position)


def _setup_sl_tp_strategy():
    s = _SlTpStrategy("EURUSD")
    broker = make_broker(10_000)
    BrokerView(broker, s)
    # Bar 1: opens long at 1.1000, sl=1.08, tp=1.12
    broker.update_price("EURUSD", 1.1000, _ts("2024-01-01T00:00:00"))
    s.on_bar(**make_bar(close=1.1000))
    return s, broker


class TestSlTpCallbacks:
    def test_sl_hit_fires_callback(self):
        s, broker = _setup_sl_tp_strategy()
        # Bar 2: low drops below SL
        s.broker.on_bar("EURUSD", 1.0900, 1.0950, 1.0750, 1.0800, _ts("2024-01-01T00:01:00"))
        assert len(s.sl_hits) == 1
        assert len(s.tp_hits) == 0
        _, exit_price = s.sl_hits[0]
        assert exit_price <= 1.0800  # filled at SL or gapped worse

    def test_tp_hit_fires_callback(self):
        s, broker = _setup_sl_tp_strategy()
        # Bar 2: high crosses TP
        s.broker.on_bar("EURUSD", 1.1100, 1.1250, 1.1050, 1.1200, _ts("2024-01-01T00:01:00"))
        assert len(s.tp_hits) == 1
        assert len(s.sl_hits) == 0
        _, exit_price = s.tp_hits[0]
        assert exit_price >= 1.1200  # filled at TP or better

    def test_on_position_close_success_fires_after_sl(self):
        s, broker = _setup_sl_tp_strategy()
        s.broker.on_bar("EURUSD", 1.0900, 1.0950, 1.0750, 1.0800, _ts("2024-01-01T00:01:00"))
        assert len(s.close_successes) == 1

    def test_sl_takes_priority_over_tp_same_bar(self):
        """If both SL and TP are inside the bar range, SL wins (pessimistic)."""
        s, broker = _setup_sl_tp_strategy()
        # Bar 2: bar spans both SL (1.08) and TP (1.12) — low=1.07, high=1.13
        s.broker.on_bar("EURUSD", 1.1000, 1.1300, 1.0700, 1.1000, _ts("2024-01-01T00:01:00"))
        assert len(s.sl_hits) == 1
        assert len(s.tp_hits) == 0

    def test_no_sl_tp_hit_when_bar_within_range(self):
        s, broker = _setup_sl_tp_strategy()
        s.broker.on_bar("EURUSD", 1.1000, 1.1100, 1.0900, 1.1050, _ts("2024-01-01T00:01:00"))
        assert s.sl_hits == []
        assert s.tp_hits == []

    def test_sl_modification_in_callback_rechecked_same_bar(self):
        """SL moved into violated territory inside on_sl_hit should NOT re-trigger
        (the hit position is already closed). But a surviving position whose SL is
        moved into violated territory in a callback SHOULD trigger in the same bar."""
        triggered: list[str] = []

        class _TwoPositionStrategy(Strategy[StrategyParams]):
            def next(self):
                if len(self.bars) == 1:
                    # open two positions; only B has SL in the danger zone initially
                    self.buy(1.0, exclusive=False)  # position A — no SL
                    self.buy(1.0, exclusive=False)  # position B — no SL
            def on_signal_execution_success(self, signal, position):
                if len(self.broker.positions) == 2:
                    # set A's SL low enough to NOT be hit
                    self.broker.update_sl(self.broker.positions[0].id, 1.0500)
                    # set B's SL that WILL be hit by this bar's low
                    self.broker.update_sl(self.broker.positions[1].id, 1.0800)
            def on_sl_hit(self, position, exit_price):
                triggered.append(f"sl_{position.id}")

        s = _TwoPositionStrategy("EURUSD")
        broker = make_broker(100_000)
        BrokerView(broker, s)
        broker.update_price("EURUSD", 1.1000, _ts())
        s.on_bar(**make_bar(close=1.1000))

        # Bar where low=1.07 — hits B's SL (1.08), not A's (1.05)
        s.broker.on_bar("EURUSD", 1.1000, 1.1100, 1.0700, 1.0800, _ts("2024-01-01T00:01:00"))
        assert "sl_2" in triggered   # B's SL hit
        assert "sl_1" not in triggered  # A's SL not hit
