"""Tests for BacktestBroker: fills, spread/slippage/commission, PnL, SL/TP, equity."""
from datetime import datetime, timezone

import pytest

from algotrading.backtest.backtest_broker import BacktestBroker, ClosedTrade, SymbolSpec
from algotrading.core.broker import SignalExecutionError, SignalExecutionErrorReason
from algotrading.core.position import Position
from algotrading.core.signal import EntrySignal, ExitSignal, SignalType


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

STRATEGY_ID = 1
SYMBOL = "BTCUSD"


def _ts(s: str = "2024-01-01T00:00:00") -> datetime:
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


def _long(symbol=SYMBOL, qty=0.01, price=100.0, sl=None, tp=None) -> EntrySignal:
    return EntrySignal(
        strategy_id=STRATEGY_ID, symbol=symbol, direction="LONG",
        type=SignalType.MARKET, qty=qty, price=price, exclusive=False,
        sl=sl, tp=tp,
    )


def _short(symbol=SYMBOL, qty=0.01, price=100.0, sl=None, tp=None) -> EntrySignal:
    return EntrySignal(
        strategy_id=STRATEGY_ID, symbol=symbol, direction="SHORT",
        type=SignalType.MARKET, qty=qty, price=price, exclusive=False,
        sl=sl, tp=tp,
    )


def _exit(symbol=SYMBOL, position_ids=None) -> ExitSignal:
    return ExitSignal(strategy_id=STRATEGY_ID, symbol=symbol, position_ids=position_ids or [])


def make_broker(balance=10_000.0, spec: SymbolSpec | None = None, leverage=0.01) -> BacktestBroker:
    specs = {SYMBOL: spec} if spec is not None else {}
    return BacktestBroker(initial_balance=balance, leverage=leverage, symbol_specs=specs)


# ---------------------------------------------------------------------------
# Fill prices
# ---------------------------------------------------------------------------

class TestFillPrices:
    def test_long_fills_at_ask(self):
        """Long entry fills above mid by half-spread + slippage."""
        spec = SymbolSpec(spread_pct=1.0)          # 1% spread → 0.5% per side
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0)
        pos = broker.execute_signal(_long(price=100.0))
        assert pos.entry_price == pytest.approx(100.0 * (1 + 0.005))

    def test_short_fills_at_bid(self):
        """Short entry fills below mid by half-spread + slippage."""
        spec = SymbolSpec(spread_pct=1.0)
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0)
        pos = broker.execute_signal(_short(price=100.0))
        assert pos.entry_price == pytest.approx(100.0 * (1 - 0.005))

    def test_no_spread_fills_at_mid(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0)
        pos = broker.execute_signal(_long(price=100.0))
        assert pos.entry_price == pytest.approx(100.0)

    def test_slippage_adds_to_spread(self):
        spec = SymbolSpec(spread_pct=1.0, slippage_pct=0.5)  # total penalty = 1.0% long side
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0)
        pos = broker.execute_signal(_long(price=100.0))
        # penalty = 100 * (0.01/2 + 0.005) = 100 * 0.01 = 1.0
        assert pos.entry_price == pytest.approx(101.0)


# ---------------------------------------------------------------------------
# PnL calculation
# ---------------------------------------------------------------------------

class TestPnl:
    def test_long_profit(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, price=100.0))
        broker.update_price(SYMBOL, 110.0, _ts())
        pnl = broker.pnl(STRATEGY_ID)
        assert pnl == pytest.approx(10.0)

    def test_long_loss(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, price=100.0))
        broker.update_price(SYMBOL, 90.0, _ts())
        assert broker.pnl(STRATEGY_ID) == pytest.approx(-10.0)

    def test_short_profit(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, price=100.0))
        broker.update_price(SYMBOL, 90.0, _ts())
        assert broker.pnl(STRATEGY_ID) == pytest.approx(10.0)

    def test_value_per_point_scales_pnl(self):
        """vpp=0.5 halves the PnL vs vpp=1."""
        spec = SymbolSpec(value_per_point=0.5)
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, price=100.0))
        broker.update_price(SYMBOL, 110.0, _ts())
        assert broker.pnl(STRATEGY_ID) == pytest.approx(5.0)  # 10 * 0.5

    def test_pnl_pct_uses_account_currency(self):
        """pnl_pct denominator must be entry_price * qty * vpp, not raw price*qty."""
        spec = SymbolSpec(value_per_point=2.0)
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        pos = broker.execute_signal(_long(qty=1.0, price=100.0))
        broker.update_price(SYMBOL, 110.0, _ts())
        # pnl = 10 * 1.0 * 2.0 = 20.0
        # cost (account currency) = 100 * 1.0 * 2.0 = 200.0
        # pnl_pct = 20 / 200 * 100 = 10%
        assert broker.pnl_pct(STRATEGY_ID, pos.id) == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# Commission
# ---------------------------------------------------------------------------

class TestCommission:
    def test_commission_deducted_at_open(self):
        """Commission is deducted from balance immediately on open."""
        spec = SymbolSpec(contract_size=1.0, commission_per_lot=5.0)
        broker = make_broker(balance=1_000.0, spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0))   # 1 lot × €5 = €5
        assert broker.balance == pytest.approx(995.0)

    def test_commission_deducted_at_close(self):
        """Another commission leg is deducted on close."""
        spec = SymbolSpec(contract_size=1.0, commission_per_lot=5.0)
        broker = make_broker(balance=1_000.0, spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0))
        broker.update_price(SYMBOL, 100.0, _ts())  # flat — no price change
        broker.close_positions(_exit())
        # open commission 5 + close commission 5 = 10 total
        assert broker.balance == pytest.approx(990.0)

    def test_commission_cost_tracked_in_trade_log(self):
        spec = SymbolSpec(contract_size=1.0, commission_per_lot=5.0)
        broker = make_broker(balance=1_000.0, spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0))
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.close_positions(_exit())
        trade = broker.trade_log[0]
        assert trade.commission_cost == pytest.approx(10.0)

    def test_fractional_lots_scale_commission(self):
        spec = SymbolSpec(contract_size=1.0, commission_per_lot=5.0)
        broker = make_broker(balance=1_000.0, spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=0.5))   # 0.5 lots → €2.50 at open
        assert broker.balance == pytest.approx(997.5)


# ---------------------------------------------------------------------------
# Margin / funds check
# ---------------------------------------------------------------------------

class TestMarginCheck:
    def test_insufficient_funds_raises(self):
        broker = make_broker(balance=100.0, leverage=0.01)  # need 1% margin
        broker.update_price(SYMBOL, 100.0, _ts())
        # position value = 100 * 100 * 1.0 = 10_000; margin needed = 100; balance = 100 → ok
        # qty=1000 → margin needed = 100 * 1000 * 0.01 = 1000 > 100 → should fail
        from algotrading.core.broker import SignalExecutionError
        with pytest.raises(SignalExecutionError) as exc:
            broker.execute_signal(_long(qty=1_000.0, price=100.0))
        assert exc.value.reason == SignalExecutionErrorReason.INSUFFICIENT_FUNDS

    def test_sufficient_funds_opens_position(self):
        broker = make_broker(balance=10_000.0, leverage=0.01)
        broker.update_price(SYMBOL, 100.0, _ts())
        pos = broker.execute_signal(_long(qty=1.0, price=100.0))
        assert pos is not None

    def test_margin_used_grows_with_positions(self):
        broker = make_broker(balance=10_000.0, leverage=0.01)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0))
        margin_after_one = broker._margin_used()
        broker.execute_signal(_long(qty=1.0))
        assert broker._margin_used() == pytest.approx(margin_after_one * 2)


# ---------------------------------------------------------------------------
# Close positions
# ---------------------------------------------------------------------------

class TestClosePositions:
    def test_close_settles_pnl_to_balance(self):
        broker = make_broker(balance=1_000.0)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0))
        broker.update_price(SYMBOL, 110.0, _ts())
        broker.close_positions(_exit())
        assert broker.balance == pytest.approx(1_010.0)

    def test_close_records_trade(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0))
        broker.update_price(SYMBOL, 105.0, _ts())
        broker.close_positions(_exit())
        assert len(broker.trade_log) == 1
        trade = broker.trade_log[0]
        assert trade.pnl == pytest.approx(5.0)
        assert trade.is_win

    def test_partial_close_by_position_id(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        pos_a = broker.execute_signal(_long(qty=1.0))
        broker.execute_signal(_long(qty=1.0))
        broker.close_positions(_exit(position_ids=[pos_a.id]))
        remaining = broker.get_positions(SYMBOL, STRATEGY_ID)
        assert len(remaining) == 1
        assert remaining[0].id != pos_a.id

    def test_sum_of_trade_pnl_equals_balance_change(self):
        """sum(trade.pnl) must equal balance - initial_balance."""
        broker = make_broker(balance=1_000.0)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0))
        broker.update_price(SYMBOL, 95.0, _ts())
        broker.close_positions(_exit())
        broker.execute_signal(_short(qty=1.0))
        broker.update_price(SYMBOL, 90.0, _ts())
        broker.close_positions(_exit())
        total_pnl = sum(t.pnl for t in broker.trade_log)
        assert total_pnl == pytest.approx(broker.balance - 1_000.0)


# ---------------------------------------------------------------------------
# Cost breakdown in trade log
# ---------------------------------------------------------------------------

class TestCostBreakdown:
    def test_spread_cost_is_positive(self):
        spec = SymbolSpec(spread_pct=1.0)
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0))
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.close_positions(_exit())
        trade = broker.trade_log[0]
        assert trade.spread_cost > 0

    def test_slippage_cost_is_positive(self):
        spec = SymbolSpec(slippage_pct=0.5)
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0))
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.close_positions(_exit())
        trade = broker.trade_log[0]
        assert trade.slippage_cost > 0

    def test_costs_property_is_sum(self):
        spec = SymbolSpec(spread_pct=0.5, slippage_pct=0.2, contract_size=1.0, commission_per_lot=3.0)
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0))
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.close_positions(_exit())
        t = broker.trade_log[0]
        assert t.costs == pytest.approx(t.spread_cost + t.slippage_cost + t.commission_cost)

    def test_gross_pnl_equals_pnl_plus_costs(self):
        spec = SymbolSpec(spread_pct=1.0, contract_size=1.0, commission_per_lot=2.0)
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0))
        broker.update_price(SYMBOL, 110.0, _ts())
        broker.close_positions(_exit())
        t = broker.trade_log[0]
        assert t.gross_pnl == pytest.approx(t.pnl + t.costs)


# ---------------------------------------------------------------------------
# SL/TP detection
# ---------------------------------------------------------------------------

class TestSlTpDetection:
    def test_long_sl_hit_when_low_crosses(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=95.0))
        hits = broker.check_sl_tp(SYMBOL, open=98.0, high=99.0, low=94.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1
        _, _, is_sl = hits[0]
        assert is_sl is True

    def test_long_tp_hit_when_high_crosses(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, tp=110.0))
        hits = broker.check_sl_tp(SYMBOL, open=105.0, high=112.0, low=104.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1
        _, _, is_sl = hits[0]
        assert is_sl is False

    def test_short_sl_hit_when_high_crosses(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, sl=106.0))
        hits = broker.check_sl_tp(SYMBOL, open=103.0, high=107.0, low=102.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1
        _, _, is_sl = hits[0]
        assert is_sl is True

    def test_short_tp_hit_when_low_crosses(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, tp=90.0))
        hits = broker.check_sl_tp(SYMBOL, open=95.0, high=96.0, low=88.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1
        _, _, is_sl = hits[0]
        assert is_sl is False

    def test_sl_takes_priority_when_both_hit(self):
        """When both SL and TP are inside the bar, SL wins (pessimistic)."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=90.0, tp=110.0))
        hits = broker.check_sl_tp(SYMBOL, open=100.0, high=115.0, low=85.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1
        _, _, is_sl = hits[0]
        assert is_sl is True

    def test_no_hit_when_bar_within_range(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=90.0, tp=110.0))
        hits = broker.check_sl_tp(SYMBOL, open=100.0, high=105.0, low=95.0, strategy_id=STRATEGY_ID)
        assert hits == []

    def test_gap_through_sl_fills_at_open(self):
        """If bar opens beyond SL (gap), fill at open, not SL price."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=95.0))
        # open=92 < sl=95 → gapped below SL, should fill at 92 not 95
        hits = broker.check_sl_tp(SYMBOL, open=92.0, high=93.0, low=91.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1
        _, exit_price, _ = hits[0]
        assert exit_price < 95.0   # filled worse than SL due to gap

    def test_tp_gap_fills_at_open(self):
        """If bar gaps above TP, fill at open (better than TP)."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, tp=110.0))
        # open=115 > tp=110 → fill at open (better)
        hits = broker.check_sl_tp(SYMBOL, open=115.0, high=116.0, low=114.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1
        _, exit_price, _ = hits[0]
        assert exit_price > 110.0   # filled better due to gap

    def test_hit_closes_position(self):
        """After SL hit, position is removed from active positions."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=95.0))
        assert len(broker.get_positions(SYMBOL, STRATEGY_ID)) == 1
        broker.check_sl_tp(SYMBOL, open=96.0, high=97.0, low=94.0, strategy_id=STRATEGY_ID)
        assert broker.get_positions(SYMBOL, STRATEGY_ID) == []

    def test_hit_adds_to_trade_log(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=95.0))
        broker.check_sl_tp(SYMBOL, open=96.0, high=97.0, low=94.0, strategy_id=STRATEGY_ID)
        assert len(broker.trade_log) == 1

    def test_only_strategy_positions_affected(self):
        """check_sl_tp with strategy_id=1 must not touch strategy_id=2 positions."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        # strategy 2 position
        sig2 = EntrySignal(
            strategy_id=2, symbol=SYMBOL, direction="LONG",
            type=SignalType.MARKET, qty=1.0, price=100.0, exclusive=False, sl=95.0,
        )
        broker.execute_signal(sig2)
        hits = broker.check_sl_tp(SYMBOL, open=96.0, high=97.0, low=94.0, strategy_id=STRATEGY_ID)
        assert hits == []
        assert len(broker.get_positions(SYMBOL, 2)) == 1  # strategy 2 position untouched


# ---------------------------------------------------------------------------
# Same-bar SL and TP both triggered
# ---------------------------------------------------------------------------
#
# When a single bar's high/low violates both SL and TP simultaneously, SL must
# always win (pessimistic).  The fill price depends on which level the bar
# *opens* relative to — open may gap through SL, through TP, or sit between them.
#
# Notation: entry=100, sl=90 (below), tp=110 (above) for LONG.
#           entry=100, sl=110 (above), tp=90  (below) for SHORT.

class TestSameBarSlAndTp:

    # ---- LONG -------------------------------------------------------

    def test_long_both_inside_bar_sl_wins(self):
        """No gap: open between SL and TP, bar spans both. SL wins."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=90.0, tp=110.0))
        hits = broker.check_sl_tp(SYMBOL, open=100.0, high=115.0, low=85.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1                         # exactly one event
        _, exit_price, is_sl = hits[0]
        assert is_sl is True
        assert exit_price == pytest.approx(90.0)      # min(100, 90) = 90 — no gap
        assert broker.trade_log[0].pnl == pytest.approx((90.0 - 100.0) * 1.0)

    def test_long_sl_gapped_tp_inside_bar_sl_wins_at_open(self):
        """Bar opens below SL (gap down); TP is also inside bar but SL wins.
        Fill is at open because open < sl → trigger_mid = min(open, sl) = open."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=90.0, tp=110.0))
        # open=85 gaps below sl=90; high=115 also crosses tp=110
        hits = broker.check_sl_tp(SYMBOL, open=85.0, high=115.0, low=83.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1
        _, exit_price, is_sl = hits[0]
        assert is_sl is True
        assert exit_price == pytest.approx(85.0)      # min(85, 90) = 85 (gap, worse than SL)
        assert broker.trade_log[0].pnl == pytest.approx((85.0 - 100.0) * 1.0)

    def test_long_tp_gapped_sl_inside_bar_sl_wins_at_sl_price(self):
        """Bar opens above TP (gap up, favourable) BUT the bar also dips below SL.
        SL still wins and fills at SL price (not open), because open > sl so no gap
        through SL: trigger_mid = min(open, sl) = sl."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=90.0, tp=110.0))
        # open=115 > tp=110 (would be a great TP gap) but low=85 dips below sl=90
        hits = broker.check_sl_tp(SYMBOL, open=115.0, high=120.0, low=85.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1
        _, exit_price, is_sl = hits[0]
        assert is_sl is True
        # trigger_mid = min(115, 90) = 90 — SL is not gapped (open is above sl)
        assert exit_price == pytest.approx(90.0)
        assert broker.trade_log[0].pnl == pytest.approx((90.0 - 100.0) * 1.0)

    def test_long_only_one_hit_returned_never_two(self):
        """A single position can produce at most one hit per bar."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=90.0, tp=110.0))
        hits = broker.check_sl_tp(SYMBOL, open=100.0, high=120.0, low=80.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1

    # ---- SHORT -------------------------------------------------------

    def test_short_both_inside_bar_sl_wins(self):
        """No gap: open between TP and SL, bar spans both. SL wins."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, sl=110.0, tp=90.0))
        hits = broker.check_sl_tp(SYMBOL, open=100.0, high=115.0, low=85.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1
        _, exit_price, is_sl = hits[0]
        assert is_sl is True
        assert exit_price == pytest.approx(110.0)     # max(100, 110) = 110 — no gap
        assert broker.trade_log[0].pnl == pytest.approx((100.0 - 110.0) * 1.0)

    def test_short_sl_gapped_tp_inside_bar_sl_wins_at_open(self):
        """Bar opens above SHORT SL (gap up); TP also inside bar but SL wins.
        Fill is at open because open > sl → trigger_mid = max(open, sl) = open."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, sl=110.0, tp=90.0))
        # open=115 gaps above sl=110; low=85 also crosses tp=90
        hits = broker.check_sl_tp(SYMBOL, open=115.0, high=118.0, low=85.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1
        _, exit_price, is_sl = hits[0]
        assert is_sl is True
        assert exit_price == pytest.approx(115.0)     # max(115, 110) = 115 (gap, worse than SL)
        assert broker.trade_log[0].pnl == pytest.approx((100.0 - 115.0) * 1.0)

    def test_short_tp_gapped_sl_inside_bar_sl_wins_at_sl_price(self):
        """Bar opens below TP (gap down, favourable for short) BUT also crosses SL.
        SL still wins and fills at SL price: trigger_mid = max(open, sl) = sl."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, sl=110.0, tp=90.0))
        # open=85 < tp=90 (TP gap) but high=115 crosses sl=110
        hits = broker.check_sl_tp(SYMBOL, open=85.0, high=115.0, low=82.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1
        _, exit_price, is_sl = hits[0]
        assert is_sl is True
        # trigger_mid = max(85, 110) = 110 — SL is not gapped (open is below sl)
        assert exit_price == pytest.approx(110.0)
        assert broker.trade_log[0].pnl == pytest.approx((100.0 - 110.0) * 1.0)

    def test_short_only_one_hit_returned_never_two(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, sl=110.0, tp=90.0))
        hits = broker.check_sl_tp(SYMBOL, open=100.0, high=120.0, low=80.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1

    # ---- accounting invariant ----------------------------------------

    def test_balance_reflects_sl_fill_not_tp_fill(self):
        """After same-bar SL+TP, settled balance must use the SL fill price."""
        broker = make_broker(balance=1_000.0)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=90.0, tp=110.0))
        broker.check_sl_tp(SYMBOL, open=100.0, high=115.0, low=85.0, strategy_id=STRATEGY_ID)
        # SL fill at 90 → pnl = 90 - 100 = -10
        assert broker.balance == pytest.approx(1_000.0 - 10.0)
        # If TP had won instead, balance would be 1_010.0 — must NOT happen.
        assert broker.balance != pytest.approx(1_010.0)


# ---------------------------------------------------------------------------
# SL loop re-check (callback modifies SL into violated territory)
# ---------------------------------------------------------------------------

class TestSlTpLoop:
    def test_update_sl_after_hit_silently_returns_none(self):
        """update_sl on a closed position must return None, not raise."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        pos = broker.execute_signal(_long(qty=1.0, sl=95.0))
        broker.check_sl_tp(SYMBOL, open=96.0, high=97.0, low=94.0, strategy_id=STRATEGY_ID)
        result = broker.update_sl(pos.id, 80.0)
        assert result is None

    def test_sl_moved_into_bar_triggers_on_second_pass(self):
        """Surviving position with SL moved into current bar's violated range
        must be caught on the next loop iteration in BrokerView.on_bar."""
        from algotrading.core.strategy import Strategy, StrategyParams
        from algotrading.core.broker import BrokerView

        triggered_ids: list[int] = []

        class _MoveSlStrategy(Strategy[StrategyParams]):
            def next(self):
                if len(self.bars) == 1:
                    # open two positions; we'll manipulate B's SL in callback
                    self.buy(1.0, exclusive=False)
                    self.buy(1.0, exclusive=False)

            def on_signal_execution_success(self, signal, position):
                # Give both positions distinct SL levels after second is opened
                positions = self.broker.positions
                if len(positions) == 2:
                    self.broker.update_sl(positions[0].id, 92.0)  # will NOT be hit by low=91
                    self.broker.update_sl(positions[1].id, 80.0)  # safe initially

            def on_sl_hit(self, position, exit_price):
                triggered_ids.append(position.id)
                # When position[0] (SL=92) is hit, move position[1]'s SL into the bar
                remaining = self.broker.positions
                if remaining:
                    self.broker.update_sl(remaining[0].id, 89.0)  # low=91 doesn't hit 89

        s = _MoveSlStrategy("EURUSD")
        broker = BacktestBroker(initial_balance=100_000)
        BrokerView(broker, s)
        broker.update_price("EURUSD", 100.0, _ts())
        s.on_bar(time=_ts(), open=100.0, high=101.0, low=99.0, close=100.0)

        # Bar: low=91 → hits pos[0] SL at 92; callback moves pos[1] SL to 89 (safe)
        s.broker.on_bar("EURUSD", 95.0, 96.0, 91.0, 93.0, _ts("2024-01-01T00:01:00"))

        assert len(triggered_ids) == 1   # only pos[0] triggered


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

class TestEquityCurve:
    def test_equity_curve_appended_each_update(self):
        broker = make_broker(balance=1_000.0)
        for price in [100.0, 105.0, 110.0]:
            broker.update_price(SYMBOL, price, _ts())
        assert len(broker.equity_curve) == 3

    def test_equity_reflects_open_pnl(self):
        broker = make_broker(balance=1_000.0)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0))
        broker.update_price(SYMBOL, 110.0, _ts())
        eq = broker.equity_curve[-1]
        assert eq.equity == pytest.approx(1_010.0)
        assert eq.balance == pytest.approx(1_000.0)   # balance unchanged until close

    def test_balance_updates_after_close(self):
        broker = make_broker(balance=1_000.0)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0))
        broker.update_price(SYMBOL, 110.0, _ts())
        broker.close_positions(_exit())
        broker.update_price(SYMBOL, 110.0, _ts())
        eq = broker.equity_curve[-1]
        assert eq.balance == pytest.approx(1_010.0)
        assert eq.equity == pytest.approx(1_010.0)


# ---------------------------------------------------------------------------
# Gap-fill stress tests — entry, SL, and TP
# ---------------------------------------------------------------------------
#
# Naming convention:
#   "normal"   — bar moves into the trigger level but does not gap through it
#   "gap"      — bar opens beyond the trigger level
#   "boundary" — bar exactly touches the trigger level (open == trigger or edge == trigger)
#
# For SL: pessimistic fill → open price when gap, SL price otherwise.
# For TP: optimistic  fill → open price when gap, TP price otherwise.
# For LIMIT entry: optimistic fill (same direction as TP logic).
# For STOP  entry: pessimistic fill (same direction as SL logic).
#
# All PnL assertions use a no-spread broker so the math is transparent.

class TestGapFillStress:
    # ------------------------------------------------------------------ SL
    # LONG SL

    def test_long_sl_normal_fills_at_sl_price(self):
        """Bar moves to SL level without gapping; exit at SL price."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=90.0))
        hits = broker.check_sl_tp(SYMBOL, open=98.0, high=99.0, low=89.0, strategy_id=STRATEGY_ID)
        _, exit_price, is_sl = hits[0]
        assert is_sl
        assert exit_price == pytest.approx(90.0)          # min(98, 90) = 90
        trade = broker.trade_log[0]
        assert trade.pnl == pytest.approx((90.0 - 100.0) * 1.0)

    def test_long_sl_gap_fills_at_open(self):
        """Bar opens below SL (gap down); fill at open, not SL price."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=90.0))
        hits = broker.check_sl_tp(SYMBOL, open=85.0, high=87.0, low=83.0, strategy_id=STRATEGY_ID)
        _, exit_price, is_sl = hits[0]
        assert is_sl
        assert exit_price == pytest.approx(85.0)          # min(85, 90) = 85 < sl
        trade = broker.trade_log[0]
        assert trade.pnl == pytest.approx((85.0 - 100.0) * 1.0)

    def test_long_sl_gap_fills_at_open_with_spread(self):
        """Spread penalty is applied on top of the gap open price."""
        spec = SymbolSpec(spread_pct=1.0)                 # 0.5% per-side penalty
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=90.0))
        # entry fill = 100 * (1 + 0.005) = 100.5
        # gap: open=85, trigger_mid = min(85, 90) = 85
        # exit_price = _exit_price(sym, 85, long) = 85 * (1 - 0.005) = 84.575
        hits = broker.check_sl_tp(SYMBOL, open=85.0, high=87.0, low=83.0, strategy_id=STRATEGY_ID)
        _, exit_price, _ = hits[0]
        assert exit_price == pytest.approx(85.0 * (1 - 0.005))

    def test_long_sl_exact_open_equals_sl(self):
        """Bar opens exactly at SL price; trigger_mid = min(sl, sl) = sl."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=90.0))
        hits = broker.check_sl_tp(SYMBOL, open=90.0, high=91.0, low=89.0, strategy_id=STRATEGY_ID)
        _, exit_price, _ = hits[0]
        assert exit_price == pytest.approx(90.0)

    def test_long_sl_boundary_low_exactly_equals_sl(self):
        """Bar low == SL price exactly must trigger (≤ condition is inclusive)."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=90.0))
        hits = broker.check_sl_tp(SYMBOL, open=95.0, high=96.0, low=90.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1

    def test_long_sl_boundary_low_one_above_sl_no_trigger(self):
        """Bar low one unit above SL must not trigger."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, sl=90.0))
        hits = broker.check_sl_tp(SYMBOL, open=95.0, high=96.0, low=90.001, strategy_id=STRATEGY_ID)
        assert hits == []

    # SHORT SL

    def test_short_sl_normal_fills_at_sl_price(self):
        """Bar rises to SHORT SL without gapping; exit at SL price."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, sl=110.0))
        hits = broker.check_sl_tp(SYMBOL, open=102.0, high=111.0, low=101.0, strategy_id=STRATEGY_ID)
        _, exit_price, is_sl = hits[0]
        assert is_sl
        assert exit_price == pytest.approx(110.0)         # max(102, 110) = 110
        trade = broker.trade_log[0]
        assert trade.pnl == pytest.approx((100.0 - 110.0) * 1.0)

    def test_short_sl_gap_fills_at_open(self):
        """Bar opens above SHORT SL (gap up); fill at open, not SL price."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, sl=110.0))
        hits = broker.check_sl_tp(SYMBOL, open=115.0, high=117.0, low=113.0, strategy_id=STRATEGY_ID)
        _, exit_price, is_sl = hits[0]
        assert is_sl
        assert exit_price == pytest.approx(115.0)         # max(115, 110) = 115 > sl
        trade = broker.trade_log[0]
        assert trade.pnl == pytest.approx((100.0 - 115.0) * 1.0)

    def test_short_sl_gap_fills_at_open_with_spread(self):
        """Spread penalty applied on short SL gap fill (pays more on exit)."""
        spec = SymbolSpec(spread_pct=1.0)
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, sl=110.0))
        # gap: open=115, trigger_mid = max(115, 110) = 115
        # exit_price = _exit_price(sym, 115, long=False) = 115 * (1 + 0.005) = 115.575
        hits = broker.check_sl_tp(SYMBOL, open=115.0, high=117.0, low=113.0, strategy_id=STRATEGY_ID)
        _, exit_price, _ = hits[0]
        assert exit_price == pytest.approx(115.0 * (1 + 0.005))

    def test_short_sl_boundary_high_exactly_equals_sl(self):
        """Bar high == SHORT SL price exactly must trigger (≥ inclusive)."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, sl=110.0))
        hits = broker.check_sl_tp(SYMBOL, open=102.0, high=110.0, low=101.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1

    def test_short_sl_boundary_high_one_below_sl_no_trigger(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, sl=110.0))
        hits = broker.check_sl_tp(SYMBOL, open=102.0, high=109.999, low=101.0, strategy_id=STRATEGY_ID)
        assert hits == []

    # ------------------------------------------------------------------ TP
    # LONG TP

    def test_long_tp_normal_fills_at_tp_price(self):
        """Bar reaches LONG TP without gapping; exit at TP price."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, tp=110.0))
        hits = broker.check_sl_tp(SYMBOL, open=102.0, high=111.0, low=101.0, strategy_id=STRATEGY_ID)
        _, exit_price, is_sl = hits[0]
        assert not is_sl
        assert exit_price == pytest.approx(110.0)         # max(102, 110) = 110
        trade = broker.trade_log[0]
        assert trade.pnl == pytest.approx((110.0 - 100.0) * 1.0)

    def test_long_tp_gap_fills_at_open(self):
        """Bar opens above LONG TP (gap up); fill at open (optimistic)."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, tp=110.0))
        hits = broker.check_sl_tp(SYMBOL, open=115.0, high=116.0, low=114.0, strategy_id=STRATEGY_ID)
        _, exit_price, is_sl = hits[0]
        assert not is_sl
        assert exit_price == pytest.approx(115.0)         # max(115, 110) = 115 > tp
        trade = broker.trade_log[0]
        assert trade.pnl == pytest.approx((115.0 - 100.0) * 1.0)

    def test_long_tp_gap_fills_at_open_with_spread(self):
        """Spread applied on long TP gap (closes below mid)."""
        spec = SymbolSpec(spread_pct=1.0)
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, tp=110.0))
        # gap: open=115, trigger_mid = max(115, 110) = 115
        # exit_price = 115 * (1 - 0.005) = 114.425
        hits = broker.check_sl_tp(SYMBOL, open=115.0, high=116.0, low=114.0, strategy_id=STRATEGY_ID)
        _, exit_price, _ = hits[0]
        assert exit_price == pytest.approx(115.0 * (1 - 0.005))

    def test_long_tp_boundary_high_exactly_equals_tp(self):
        """Bar high == LONG TP exactly must trigger."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, tp=110.0))
        hits = broker.check_sl_tp(SYMBOL, open=102.0, high=110.0, low=101.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1

    def test_long_tp_boundary_high_one_below_tp_no_trigger(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_long(qty=1.0, tp=110.0))
        hits = broker.check_sl_tp(SYMBOL, open=102.0, high=109.999, low=101.0, strategy_id=STRATEGY_ID)
        assert hits == []

    # SHORT TP

    def test_short_tp_normal_fills_at_tp_price(self):
        """Bar drops to SHORT TP without gapping; exit at TP price."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, tp=90.0))
        hits = broker.check_sl_tp(SYMBOL, open=98.0, high=99.0, low=89.0, strategy_id=STRATEGY_ID)
        _, exit_price, is_sl = hits[0]
        assert not is_sl
        assert exit_price == pytest.approx(90.0)          # min(98, 90) = 90
        trade = broker.trade_log[0]
        assert trade.pnl == pytest.approx((100.0 - 90.0) * 1.0)

    def test_short_tp_gap_fills_at_open(self):
        """Bar opens below SHORT TP (gap down); fill at open (optimistic)."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, tp=90.0))
        hits = broker.check_sl_tp(SYMBOL, open=85.0, high=87.0, low=83.0, strategy_id=STRATEGY_ID)
        _, exit_price, is_sl = hits[0]
        assert not is_sl
        assert exit_price == pytest.approx(85.0)          # min(85, 90) = 85 < tp
        trade = broker.trade_log[0]
        assert trade.pnl == pytest.approx((100.0 - 85.0) * 1.0)

    def test_short_tp_gap_fills_at_open_with_spread(self):
        """Spread applied on short TP gap exit (pays more on buy-back)."""
        spec = SymbolSpec(spread_pct=1.0)
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, tp=90.0))
        # gap: open=85, trigger_mid = min(85, 90) = 85
        # exit_price = _exit_price(sym, 85, long=False) = 85 * (1 + 0.005) = 85.425
        hits = broker.check_sl_tp(SYMBOL, open=85.0, high=87.0, low=83.0, strategy_id=STRATEGY_ID)
        _, exit_price, _ = hits[0]
        assert exit_price == pytest.approx(85.0 * (1 + 0.005))

    def test_short_tp_boundary_low_exactly_equals_tp(self):
        """Bar low == SHORT TP exactly must trigger."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, tp=90.0))
        hits = broker.check_sl_tp(SYMBOL, open=98.0, high=99.0, low=90.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1

    def test_short_tp_boundary_low_one_above_tp_no_trigger(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.execute_signal(_short(qty=1.0, tp=90.0))
        hits = broker.check_sl_tp(SYMBOL, open=98.0, high=99.0, low=90.001, strategy_id=STRATEGY_ID)
        assert hits == []

    # ------------------------------------------------------------------ entry
    # LIMIT LONG entry gaps

    def test_limit_long_boundary_low_exactly_equals_price(self):
        """Bar low == limit price exactly must trigger."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_long(price=90.0))
        results = broker.fill_pending_signals(SYMBOL, open=95.0, high=96.0, low=90.0, strategy_id=STRATEGY_ID)
        assert len(results) == 1

    def test_limit_long_boundary_low_one_above_price_no_fill(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_long(price=90.0))
        results = broker.fill_pending_signals(SYMBOL, open=95.0, high=96.0, low=90.001, strategy_id=STRATEGY_ID)
        assert results == []

    def test_limit_long_gap_pnl_uses_open_not_limit(self):
        """PnL for a gapped limit fill must use open price (not limit) as cost basis."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_long(price=90.0))
        # Gap: open=80 < limit=90 → fill at 80
        broker.fill_pending_signals(SYMBOL, open=80.0, high=82.0, low=78.0, strategy_id=STRATEGY_ID)
        pos = broker.get_positions(SYMBOL, STRATEGY_ID)[0]
        assert pos.entry_price == pytest.approx(80.0)
        # Update price to 80 then close — gross_pnl should be ~0 (entry == exit, no spread)
        broker.update_price(SYMBOL, 80.0, _ts())
        broker.close_positions(_exit())
        assert broker.trade_log[0].gross_pnl == pytest.approx(0.0)

    # LIMIT SHORT entry gaps

    def test_limit_short_boundary_high_exactly_equals_price(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_short(price=110.0))
        results = broker.fill_pending_signals(SYMBOL, open=105.0, high=110.0, low=104.0, strategy_id=STRATEGY_ID)
        assert len(results) == 1

    def test_limit_short_boundary_high_one_below_price_no_fill(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_short(price=110.0))
        results = broker.fill_pending_signals(SYMBOL, open=105.0, high=109.999, low=104.0, strategy_id=STRATEGY_ID)
        assert results == []

    def test_limit_short_gap_pnl_uses_open_not_limit(self):
        """Gapped limit short fills at open (better for seller than limit price)."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_short(price=110.0))
        # Gap: open=120 > limit=110 → fill at 120
        broker.fill_pending_signals(SYMBOL, open=120.0, high=122.0, low=118.0, strategy_id=STRATEGY_ID)
        pos = broker.get_positions(SYMBOL, STRATEGY_ID)[0]
        assert pos.entry_price == pytest.approx(120.0)

    # STOP LONG entry gaps

    def test_stop_long_boundary_high_exactly_equals_price(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_stop_long(price=110.0))
        results = broker.fill_pending_signals(SYMBOL, open=105.0, high=110.0, low=104.0, strategy_id=STRATEGY_ID)
        assert len(results) == 1

    def test_stop_long_boundary_high_one_below_price_no_fill(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_stop_long(price=110.0))
        results = broker.fill_pending_signals(SYMBOL, open=105.0, high=109.999, low=104.0, strategy_id=STRATEGY_ID)
        assert results == []

    def test_stop_long_gap_pnl_uses_open_not_stop(self):
        """Gapped stop long fills at open (worse than stop price)."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_stop_long(price=110.0))
        # Gap: open=120 > stop=110 → fill at 120 (worse than intended)
        broker.fill_pending_signals(SYMBOL, open=120.0, high=122.0, low=118.0, strategy_id=STRATEGY_ID)
        pos = broker.get_positions(SYMBOL, STRATEGY_ID)[0]
        assert pos.entry_price == pytest.approx(120.0)

    # STOP SHORT entry gaps

    def test_stop_short_boundary_low_exactly_equals_price(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_stop_short(price=90.0))
        results = broker.fill_pending_signals(SYMBOL, open=95.0, high=96.0, low=90.0, strategy_id=STRATEGY_ID)
        assert len(results) == 1

    def test_stop_short_boundary_low_one_above_price_no_fill(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_stop_short(price=90.0))
        results = broker.fill_pending_signals(SYMBOL, open=95.0, high=96.0, low=90.001, strategy_id=STRATEGY_ID)
        assert results == []

    def test_stop_short_gap_pnl_uses_open_not_stop(self):
        """Gapped stop short fills at open (worse than stop price for seller)."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_stop_short(price=90.0))
        # Gap: open=80 < stop=90 → fill at 80 (worse than intended)
        broker.fill_pending_signals(SYMBOL, open=80.0, high=82.0, low=78.0, strategy_id=STRATEGY_ID)
        pos = broker.get_positions(SYMBOL, STRATEGY_ID)[0]
        assert pos.entry_price == pytest.approx(80.0)

    # ------------------------------------------------------------------ cross-bar
    # pending fill + SL hit in same bar

    def test_pending_fill_and_sl_both_gap_in_same_bar(self):
        """Bar gaps down through both limit entry and SL: fill then close via SL."""
        # Setup: limit long at 90, with SL at 75
        # Bar: open=70 < limit=90 < SL=75? No — open must gap through limit first,
        # then SL must also be inside the bar.
        # open=70, low=65 → limit triggers at open=70; SL at 75 not hit (75 > 70 = open,
        # but low=65 < 75) → SL fires after fill.
        from algotrading.core.strategy import Strategy, StrategyParams
        from algotrading.core.broker import BrokerView

        sl_hits: list = []

        class _S(Strategy[StrategyParams]):
            def next(self):
                if len(self.bars) == 1:
                    self.buy(qty=1.0, entry_price=90.0, sl_price=75.0, exclusive=False)

            def on_sl_hit(self, position, exit_price):
                sl_hits.append(exit_price)

        broker = make_broker()
        strategy = _S("BTCUSD")
        BrokerView(broker, strategy)

        strategy.broker.on_bar(SYMBOL, open=100.0, high=101.0, low=99.0, close=100.0, time=_ts("2024-01-01T00:00:00"))
        strategy.on_bar(time=_ts("2024-01-01T00:00:00"), open=100.0, high=101.0, low=99.0, close=100.0)

        # Bar 2 gaps below limit (fills at 70) and SL at 75 is hit (low=65 ≤ 75)
        strategy.broker.on_bar(SYMBOL, open=70.0, high=72.0, low=65.0, close=68.0, time=_ts("2024-01-01T01:00:00"))

        # Position was opened at 70 (gapped limit) then SL at 75 fires:
        # open=70 < sl=75 → trigger_mid = min(70, 75) = 70; exit at 70.
        # Entry and exit are both at 70, so gross_pnl ≈ 0.
        assert len(sl_hits) == 1
        assert len(broker.trade_log) == 1
        assert broker.trade_log[0].gross_pnl == pytest.approx(0.0)

    def test_multiple_positions_only_gapped_one_hits_sl(self):
        """Two LONG positions with different SLs; only the one the bar gaps through fires."""
        broker = make_broker(balance=100_000.0)
        broker.update_price(SYMBOL, 100.0, _ts())
        sig_a = EntrySignal(
            strategy_id=STRATEGY_ID, symbol=SYMBOL, direction="LONG",
            type=SignalType.MARKET, qty=1.0, price=100.0, exclusive=False, sl=85.0,
        )
        sig_b = EntrySignal(
            strategy_id=STRATEGY_ID, symbol=SYMBOL, direction="LONG",
            type=SignalType.MARKET, qty=1.0, price=100.0, exclusive=False, sl=70.0,
        )
        broker.execute_signal(sig_a)
        broker.execute_signal(sig_b)

        # Bar: open=80, low=78 → sl_a=85 is hit (low=78 ≤ 85); sl_b=70 is safe (low=78 > 70)
        hits = broker.check_sl_tp(SYMBOL, open=80.0, high=81.0, low=78.0, strategy_id=STRATEGY_ID)
        assert len(hits) == 1
        _, exit_price, is_sl = hits[0]
        assert is_sl
        # gap: min(80, 85) = 80 → exit at open
        assert exit_price == pytest.approx(80.0)
        assert len(broker.get_positions(SYMBOL, STRATEGY_ID)) == 1  # sig_b still open


# ---------------------------------------------------------------------------
# Pending signal queue (LIMIT / STOP fills)
# ---------------------------------------------------------------------------

def _limit_long(price: float, qty=1.0, sl=None, tp=None) -> EntrySignal:
    return EntrySignal(
        strategy_id=STRATEGY_ID, symbol=SYMBOL, direction="LONG",
        type=SignalType.LIMIT, qty=qty, price=price, exclusive=False,
        sl=sl, tp=tp,
    )


def _limit_short(price: float, qty=1.0, sl=None, tp=None) -> EntrySignal:
    return EntrySignal(
        strategy_id=STRATEGY_ID, symbol=SYMBOL, direction="SHORT",
        type=SignalType.LIMIT, qty=qty, price=price, exclusive=False,
        sl=sl, tp=tp,
    )


def _stop_long(price: float, qty=1.0, sl=None, tp=None) -> EntrySignal:
    return EntrySignal(
        strategy_id=STRATEGY_ID, symbol=SYMBOL, direction="LONG",
        type=SignalType.STOP, qty=qty, price=price, exclusive=False,
        sl=sl, tp=tp,
    )


def _stop_short(price: float, qty=1.0, sl=None, tp=None) -> EntrySignal:
    return EntrySignal(
        strategy_id=STRATEGY_ID, symbol=SYMBOL, direction="SHORT",
        type=SignalType.STOP, qty=qty, price=price, exclusive=False,
        sl=sl, tp=tp,
    )


class TestPendingSignals:
    # -- queue / cancel --------------------------------------------------

    def test_queue_signal_returns_pending_signal(self):
        broker = make_broker()
        ps = broker.queue_signal(_limit_long(90.0))
        assert ps.id == 1
        assert ps.signal.price == 90.0

    def test_queued_signals_appear_in_get_pending_signals(self):
        broker = make_broker()
        broker.queue_signal(_limit_long(90.0))
        broker.queue_signal(_limit_short(110.0))
        pending = broker.get_pending_signals(SYMBOL, STRATEGY_ID)
        assert len(pending) == 2

    def test_cancel_removes_signal(self):
        broker = make_broker()
        ps = broker.queue_signal(_limit_long(90.0))
        assert broker.cancel_signal(ps.id) is True
        assert broker.get_pending_signals(SYMBOL, STRATEGY_ID) == []

    def test_cancel_nonexistent_returns_false(self):
        broker = make_broker()
        assert broker.cancel_signal(999) is False

    def test_other_strategy_signals_not_visible(self):
        broker = make_broker()
        sig = EntrySignal(
            strategy_id=2, symbol=SYMBOL, direction="LONG",
            type=SignalType.LIMIT, qty=1.0, price=90.0, exclusive=False,
        )
        broker.queue_signal(sig)
        assert broker.get_pending_signals(SYMBOL, STRATEGY_ID) == []

    # -- LIMIT LONG trigger rules ----------------------------------------

    def test_limit_long_fills_when_low_touches_price(self):
        """LIMIT LONG triggers when bar's low <= limit price."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_long(price=95.0))
        results = broker.fill_pending_signals(SYMBOL, open=98.0, high=99.0, low=94.0, strategy_id=STRATEGY_ID)
        assert len(results) == 1
        _, pos = results[0]
        assert isinstance(pos, Position)

    def test_limit_long_does_not_fill_when_low_above_price(self):
        """LIMIT LONG must not trigger when bar's low stays above limit price."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_long(price=95.0))
        results = broker.fill_pending_signals(SYMBOL, open=98.0, high=99.0, low=96.0, strategy_id=STRATEGY_ID)
        assert results == []
        assert len(broker.get_pending_signals(SYMBOL, STRATEGY_ID)) == 1  # still queued

    def test_limit_long_fills_at_open_when_gap_below_limit(self):
        """If bar opens below the limit price the fill uses open (better price)."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_long(price=95.0))
        # open=90 gapped below 95 → fill mid = min(90, 95) = 90
        results = broker.fill_pending_signals(SYMBOL, open=90.0, high=93.0, low=89.0, strategy_id=STRATEGY_ID)
        assert len(results) == 1
        _, pos = results[0]
        assert isinstance(pos, Position)
        assert pos.entry_price == pytest.approx(90.0)  # no spread in default spec

    def test_limit_long_fills_at_limit_price_when_bar_dips_to_it(self):
        """Normal limit fill: open above limit, low touches it → fill at limit price."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_long(price=95.0))
        results = broker.fill_pending_signals(SYMBOL, open=98.0, high=99.0, low=94.0, strategy_id=STRATEGY_ID)
        _, pos = results[0]
        assert isinstance(pos, Position)
        assert pos.entry_price == pytest.approx(95.0)  # min(98, 95) = 95

    # -- LIMIT SHORT trigger rules ---------------------------------------

    def test_limit_short_fills_when_high_touches_price(self):
        """LIMIT SHORT triggers when bar's high >= limit price."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_short(price=105.0))
        results = broker.fill_pending_signals(SYMBOL, open=102.0, high=106.0, low=101.0, strategy_id=STRATEGY_ID)
        assert len(results) == 1

    def test_limit_short_does_not_fill_when_high_below_price(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_short(price=105.0))
        results = broker.fill_pending_signals(SYMBOL, open=102.0, high=104.0, low=101.0, strategy_id=STRATEGY_ID)
        assert results == []

    def test_limit_short_fills_at_open_when_gap_above_limit(self):
        """Gap above limit → fill at open (better for seller)."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_short(price=105.0))
        results = broker.fill_pending_signals(SYMBOL, open=110.0, high=112.0, low=108.0, strategy_id=STRATEGY_ID)
        _, pos = results[0]
        assert isinstance(pos, Position)
        assert pos.entry_price == pytest.approx(110.0)  # max(110, 105) = 110

    # -- STOP LONG trigger rules ----------------------------------------

    def test_stop_long_fills_when_high_touches_price(self):
        """STOP LONG triggers when bar's high >= stop price."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_stop_long(price=105.0))
        results = broker.fill_pending_signals(SYMBOL, open=102.0, high=106.0, low=101.0, strategy_id=STRATEGY_ID)
        assert len(results) == 1

    def test_stop_long_does_not_fill_when_high_below_price(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_stop_long(price=105.0))
        results = broker.fill_pending_signals(SYMBOL, open=102.0, high=104.0, low=101.0, strategy_id=STRATEGY_ID)
        assert results == []

    def test_stop_long_fills_at_open_when_gap_above_stop(self):
        """Gap through stop → pessimistic fill at open (worse than stop price)."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_stop_long(price=105.0))
        results = broker.fill_pending_signals(SYMBOL, open=110.0, high=112.0, low=108.0, strategy_id=STRATEGY_ID)
        _, pos = results[0]
        assert isinstance(pos, Position)
        assert pos.entry_price == pytest.approx(110.0)  # max(110, 105) = 110

    def test_stop_long_fills_at_stop_price_when_bar_breaks_it(self):
        """Normal stop fill: open below stop, high crosses it → fill at stop price."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_stop_long(price=105.0))
        results = broker.fill_pending_signals(SYMBOL, open=102.0, high=107.0, low=101.0, strategy_id=STRATEGY_ID)
        _, pos = results[0]
        assert isinstance(pos, Position)
        assert pos.entry_price == pytest.approx(105.0)  # max(102, 105) = 105

    # -- STOP SHORT trigger rules ---------------------------------------

    def test_stop_short_fills_when_low_touches_price(self):
        """STOP SHORT triggers when bar's low <= stop price."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_stop_short(price=95.0))
        results = broker.fill_pending_signals(SYMBOL, open=98.0, high=99.0, low=94.0, strategy_id=STRATEGY_ID)
        assert len(results) == 1

    def test_stop_short_fills_at_stop_price_when_bar_breaks_it(self):
        """Normal stop-short fill: open above stop, low crosses it → fill at stop price."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_stop_short(price=95.0))
        results = broker.fill_pending_signals(SYMBOL, open=98.0, high=99.0, low=93.0, strategy_id=STRATEGY_ID)
        _, pos = results[0]
        assert isinstance(pos, Position)
        assert pos.entry_price == pytest.approx(95.0)  # min(98, 95) = 95

    def test_stop_short_fills_at_open_when_gap_below_stop(self):
        """Gap through stop → pessimistic fill at open (worse than stop price)."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_stop_short(price=95.0))
        results = broker.fill_pending_signals(SYMBOL, open=90.0, high=92.0, low=88.0, strategy_id=STRATEGY_ID)
        _, pos = results[0]
        assert isinstance(pos, Position)
        assert pos.entry_price == pytest.approx(90.0)  # min(90, 95) = 90

    # -- fill outcomes ---------------------------------------------------

    def test_filled_signal_removed_from_queue(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_long(price=95.0))
        broker.fill_pending_signals(SYMBOL, open=94.0, high=96.0, low=93.0, strategy_id=STRATEGY_ID)
        assert broker.get_pending_signals(SYMBOL, STRATEGY_ID) == []

    def test_untriggered_signal_stays_in_queue(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_long(price=85.0))
        broker.fill_pending_signals(SYMBOL, open=94.0, high=96.0, low=93.0, strategy_id=STRATEGY_ID)
        assert len(broker.get_pending_signals(SYMBOL, STRATEGY_ID)) == 1

    def test_insufficient_margin_fires_error_result(self):
        """A triggered pending signal that can't be filled returns an error result."""
        # margin = fill_price * qty * vpp * leverage = 94 * 20 * 1 * 0.01 = 18.8 > balance 1.0
        broker = make_broker(balance=1.0)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_long(price=95.0, qty=20.0))
        results = broker.fill_pending_signals(SYMBOL, open=94.0, high=96.0, low=93.0, strategy_id=STRATEGY_ID)
        assert len(results) == 1
        _, result = results[0]
        assert isinstance(result, SignalExecutionError)
        assert result.reason == SignalExecutionErrorReason.INSUFFICIENT_FUNDS
        # Failed pending signals are removed from the queue (consumed, not retried)
        assert broker.get_pending_signals(SYMBOL, STRATEGY_ID) == []

    def test_fill_opens_position(self):
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_long(price=95.0))
        broker.fill_pending_signals(SYMBOL, open=94.0, high=96.0, low=93.0, strategy_id=STRATEGY_ID)
        positions = broker.get_positions(SYMBOL, STRATEGY_ID)
        assert len(positions) == 1
        assert positions[0].is_long

    def test_fill_with_spread_applied(self):
        """Spread/slippage applied on pending fills identically to market orders."""
        spec = SymbolSpec(spread_pct=1.0)
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_long(price=95.0))
        results = broker.fill_pending_signals(SYMBOL, open=94.0, high=96.0, low=93.0, strategy_id=STRATEGY_ID)
        _, pos = results[0]
        assert isinstance(pos, Position)
        # trigger_mid = min(94, 95) = 94; fill = 94 * (1 + 0.005) = 94.47
        assert pos.entry_price == pytest.approx(94.0 * (1 + 0.005))

    def test_fill_sl_tp_propagated(self):
        """SL and TP from the original signal are set on the opened position."""
        broker = make_broker()
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.queue_signal(_limit_long(price=95.0, sl=90.0, tp=110.0))
        broker.fill_pending_signals(SYMBOL, open=94.0, high=96.0, low=93.0, strategy_id=STRATEGY_ID)
        pos = broker.get_positions(SYMBOL, STRATEGY_ID)[0]
        assert pos.sl_price == 90.0
        assert pos.tp_price == 110.0

    # -- end-to-end via BrokerView --------------------------------------

    def test_brokerView_routes_limit_to_queue(self):
        """Strategy.buy with entry_price below current routes to pending queue."""
        from algotrading.core.strategy import Strategy, StrategyParams
        from algotrading.core.broker import BrokerView
        from algotrading.core.signal import PendingSignal

        queued: list[PendingSignal] = []

        class _LimitStrategy(Strategy[StrategyParams]):
            def next(self):
                if len(self.bars) == 1:
                    self.buy(qty=1.0, entry_price=90.0, exclusive=False)

            def on_signal_queued(self, pending):
                queued.append(pending)

        broker = make_broker()
        strategy = _LimitStrategy("BTCUSD")
        BrokerView(broker, strategy)

        strategy.broker.on_bar(SYMBOL, open=100.0, high=101.0, low=99.0, close=100.0, time=_ts())
        strategy.on_bar(time=_ts(), open=100.0, high=101.0, low=99.0, close=100.0)

        assert len(queued) == 1
        assert queued[0].signal.type == SignalType.LIMIT
        assert len(strategy.broker.pending_signals) == 1

    def test_brokerView_fills_limit_on_next_bar(self):
        """A pending limit fills via BrokerView.on_bar when price reaches it."""
        from algotrading.core.strategy import Strategy, StrategyParams
        from algotrading.core.broker import BrokerView

        filled: list = []

        class _LimitStrategy(Strategy[StrategyParams]):
            def next(self):
                if len(self.bars) == 1:
                    self.buy(qty=1.0, entry_price=90.0, exclusive=False)

            def on_signal_execution_success(self, signal, position):
                filled.append(position)

        broker = make_broker()
        strategy = _LimitStrategy("BTCUSD")
        BrokerView(broker, strategy)

        # Bar 1: strategy places limit buy at 90 (current price 100)
        strategy.broker.on_bar(SYMBOL, open=100.0, high=101.0, low=99.0, close=100.0, time=_ts("2024-01-01T00:00:00"))
        strategy.on_bar(time=_ts("2024-01-01T00:00:00"), open=100.0, high=101.0, low=99.0, close=100.0)

        assert filled == []  # not yet filled

        # Bar 2: price drops and touches 90
        strategy.broker.on_bar(SYMBOL, open=95.0, high=96.0, low=88.0, close=92.0, time=_ts("2024-01-01T01:00:00"))

        assert len(filled) == 1
        assert filled[0].is_long

    def test_brokerView_pending_signals_property(self):
        """BrokerView.pending_signals delegates to broker for this strategy."""
        from algotrading.core.strategy import Strategy, StrategyParams
        from algotrading.core.broker import BrokerView

        class _S(Strategy[StrategyParams]):
            def next(self):
                if len(self.bars) == 1:
                    self.buy(qty=1.0, entry_price=90.0, exclusive=False)

        broker = make_broker()
        strategy = _S("BTCUSD")
        BrokerView(broker, strategy)
        strategy.broker.on_bar(SYMBOL, open=100.0, high=101.0, low=99.0, close=100.0, time=_ts())
        strategy.on_bar(time=_ts(), open=100.0, high=101.0, low=99.0, close=100.0)

        assert len(strategy.broker.pending_signals) == 1

    def test_brokerView_cancel_pending_signals(self):
        """BrokerView.cancel_pending_signals removes all pending signals."""
        from algotrading.core.strategy import Strategy, StrategyParams
        from algotrading.core.broker import BrokerView

        class _S(Strategy[StrategyParams]):
            def next(self):
                if len(self.bars) == 1:
                    self.buy(qty=1.0, entry_price=90.0, exclusive=False)
                    self.buy(qty=1.0, entry_price=85.0, exclusive=False)

        broker = make_broker()
        strategy = _S("BTCUSD")
        BrokerView(broker, strategy)
        strategy.broker.on_bar(SYMBOL, open=100.0, high=101.0, low=99.0, close=100.0, time=_ts())
        strategy.on_bar(time=_ts(), open=100.0, high=101.0, low=99.0, close=100.0)

        assert strategy.broker.cancel_pending_signals() == 2
        assert strategy.broker.pending_signals == []

    def test_pending_fill_then_sl_hit_same_bar(self):
        """A position opened by a pending fill can have its SL hit in the same bar."""
        from algotrading.core.strategy import Strategy, StrategyParams
        from algotrading.core.broker import BrokerView

        sl_hits: list = []

        class _S(Strategy[StrategyParams]):
            def next(self):
                if len(self.bars) == 1:
                    # place limit long at 95 with SL at 80
                    self.buy(qty=1.0, entry_price=95.0, sl_price=80.0, exclusive=False)

            def on_sl_hit(self, position, exit_price):
                sl_hits.append((position, exit_price))

        broker = make_broker()
        strategy = _S("BTCUSD")
        BrokerView(broker, strategy)

        strategy.broker.on_bar(SYMBOL, open=100.0, high=101.0, low=99.0, close=100.0, time=_ts("2024-01-01T00:00:00"))
        strategy.on_bar(time=_ts("2024-01-01T00:00:00"), open=100.0, high=101.0, low=99.0, close=100.0)

        # Bar 2: price gaps down through limit (fills at 90) AND through SL at 80
        strategy.broker.on_bar(SYMBOL, open=90.0, high=92.0, low=75.0, close=80.0, time=_ts("2024-01-01T01:00:00"))

        assert len(sl_hits) == 1


class TestRealismParity:
    def test_qty_normalized_to_volume_step(self):
        spec = SymbolSpec(contract_size=100.0, volume_step=0.1)
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        pos = broker.execute_signal(_long(qty=123.0, price=100.0))
        # 123 / 100 = 1.23 lots -> rounded to 1.2 lots -> 120 qty
        assert pos.qty == pytest.approx(120.0)

    def test_qty_clamped_to_volume_max(self):
        spec = SymbolSpec(contract_size=100.0, volume_max=1.5)
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        pos = broker.execute_signal(_long(qty=500.0, price=100.0))
        assert pos.qty == pytest.approx(150.0)

    def test_qty_bumped_to_volume_min(self):
        spec = SymbolSpec(contract_size=100.0, volume_min=0.2)
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        pos = broker.execute_signal(_long(qty=5.0, price=100.0))
        assert pos.qty == pytest.approx(20.0)

    def test_historical_spread_override_applies_to_fill(self):
        spec = SymbolSpec(spread_pct=0.0)
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts())
        broker.set_bar_spread_pct(SYMBOL, 1.0)
        pos = broker.execute_signal(_long(qty=1.0, price=100.0))
        # 1.0% spread means 0.5% penalty per side on long entry
        assert pos.entry_price == pytest.approx(100.5)

    def test_close_uses_current_historical_spread_not_entry_spread(self):
        spec = SymbolSpec(spread_pct=0.0)
        broker = make_broker(spec=spec)
        broker.update_price(SYMBOL, 100.0, _ts("2024-01-01T00:00:00"))

        broker.set_bar_spread_pct(SYMBOL, 1.0)
        broker.execute_signal(_long(qty=1.0, price=100.0))

        broker.set_bar_spread_pct(SYMBOL, 3.0)
        broker.update_price(SYMBOL, 100.0, _ts("2024-01-01T01:00:00"))
        broker.close_positions(_exit())

        trade = broker.trade_log[0]
        assert trade.exit_price == pytest.approx(98.5)
        assert trade.spread_cost == pytest.approx(2.0)
