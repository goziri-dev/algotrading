from dataclasses import dataclass, replace
from datetime import datetime, timezone
import math

from algotrading.core.broker import (
    Broker, PositionCloseError,
    SignalExecutionError, SignalExecutionErrorReason,
)
from algotrading.core.position import Position
from algotrading.core.signal import EntrySignal, ExitSignal, PendingSignal, SignalDirection, SignalType


@dataclass
class SymbolSpec:
    """Per-symbol contract specification for realistic PnL calculation.

    `value_per_point` is the account-currency value of a 1-point price move
    for one engine qty unit. If omitted, PnL is computed as
    ``price_change * qty`` (raw, in quote currency).

    `spread_pct` is the bid/ask spread as a percentage of the fill price
    (e.g. 0.042 means 0.042%).  Longs are filled at `mid * (1 + spread_pct/200)`
    and closed at `mid * (1 - spread_pct/200)`; shorts the reverse.
    Using a percentage scales correctly across wide price ranges (e.g. multi-year
    crypto backtests where price may vary 10x).

    `slippage_pct` is additional execution slippage as a percentage of price,
    applied in the same direction as spread on every fill.

    `contract_size` is the number of units per lot (e.g. 100_000 for standard
    forex, 100.0 for XAUUSD, 1.0 for BTCUSD). Used to convert engine qty units
    into broker lots for commission.

    `commission_per_lot` is the broker commission charged per lot per side in
    account currency (so a $7 round-trip broker charging $3.50/lot/side on a
    EUR account would be €3.18 at 1.10 EURUSD).  Applied once at open and once
    at close.  Defaults to 0 (no commission).

    Example for BTCUSD on a EUR account with $7 round-trip commission::

        SymbolSpec(value_per_point=0.8528, spread_pct=0.042, contract_size=100_000.0,
               commission_per_lot=3.18)
    """
    value_per_point: float = 1.0
    spread_pct: float = 0.0
    slippage_pct: float = 0.0
    contract_size: float = 1.0
    commission_per_lot: float = 0.0
    volume_min: float = 0.0
    volume_max: float = 0.0
    volume_step: float = 0.0


@dataclass
class EquityPoint:
    time: datetime
    equity: float    # balance + unrealized PnL (all strategies)
    balance: float   # settled cash only


@dataclass
class ClosedTrade:
    position: Position
    exit_time: datetime
    exit_price: float # adjusted fill price (spread + slippage baked in)
    pnl: float # net PnL after ALL costs
    spread_cost: float # round-trip spread cost
    slippage_cost: float # round-trip slippage cost
    commission_cost: float # open + close commission

    @property
    def is_win(self) -> bool:
        return self.pnl > 0

    @property
    def costs(self) -> float:
        return self.spread_cost + self.slippage_cost + self.commission_cost

    @property
    def gross_pnl(self) -> float:
        """PnL computed at mid prices with no transaction costs."""
        return self.pnl + self.costs


class BacktestBroker(Broker):
    def __init__(
        self,
        initial_balance: float = 10_000,
        leverage: float = 1 / 100,
        symbol_specs: dict[str, SymbolSpec] | None = None,
    ):
        super().__init__()
        self._initial_balance = initial_balance
        self._balance = initial_balance
        self._leverage = leverage # margin ratio, e.g. 0.01 = 1% margin = 100:1 leverage
        self._symbol_specs: dict[str, SymbolSpec] = symbol_specs or {}
        self._positions: dict[int, Position] = {}
        self._symbol_prices: dict[str, float] = {}
        self._bar_spread_pct: dict[str, float | None] = {}
        self._entry_mids: dict[int, float] = {} # mid price at entry for cost tracking
        self._entry_spread_pct: dict[int, float] = {} # spread% at entry for historical spread cost tracking
        self._entry_commissions: dict[int, float] = {} # open commission paid, for cost tracking
        self._current_time: datetime = datetime.now(timezone.utc)
        self._next_id: int = 1
        self._pending_signals: dict[int, PendingSignal] = {}
        self._next_signal_id: int = 1
        self.trade_log: list[ClosedTrade] = []
        self.equity_curve: list[EquityPoint] = []

    @property
    def defers_market_orders(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Symbol helpers
    # ------------------------------------------------------------------

    def _vpp(self, symbol: str) -> float:
        spec = self._symbol_specs.get(symbol)
        return spec.value_per_point if spec is not None else 1.0

    def value_per_point(self, symbol: str) -> float:
        """Account-currency value of a 1-point move for one engine qty unit."""
        return self._vpp(symbol)

    def _penalty(self, symbol: str, mid: float) -> float:
        """Per-side price penalty (half-spread + slippage) scaled to current price."""
        spec = self._symbol_specs.get(symbol)
        if spec is None:
            return 0.0
        spread_pct = self._current_spread_pct(symbol)
        return mid * (spread_pct / 200 + spec.slippage_pct / 100)

    def _current_spread_pct(self, symbol: str) -> float:
        spec = self._symbol_specs.get(symbol)
        if spec is None:
            return 0.0
        override = self._bar_spread_pct.get(symbol)
        if override is None:
            return spec.spread_pct
        return max(0.0, override)

    def _normalize_qty(self, symbol: str, qty: float) -> float:
        """Apply broker lot constraints (min/max/step) to engine qty units."""
        spec = self._symbol_specs.get(symbol)
        if spec is None or spec.contract_size <= 0:
            return max(0.0, qty)
        lots = qty / spec.contract_size
        if spec.volume_max > 0:
            lots = min(lots, spec.volume_max)
        if spec.volume_min > 0 and lots > 0:
            lots = max(lots, spec.volume_min)
        if spec.volume_step > 0:
            lots = round(lots / spec.volume_step) * spec.volume_step
            decimals = max(0, -int(math.floor(math.log10(spec.volume_step)))) if spec.volume_step < 1 else 0
            lots = round(lots, min(8, decimals + 2))
        return max(0.0, lots * spec.contract_size)

    def _fill_price(self, symbol: str, mid: float, is_long: bool) -> float:
        """Entry fill price: longs pay more (ask+slippage), shorts receive less."""
        p = self._penalty(symbol, mid)
        return mid + p if is_long else mid - p

    def _exit_price(self, symbol: str, mid: float, is_long: bool) -> float:
        """Exit fill price: longs receive less (bid-slippage), shorts pay more."""
        p = self._penalty(symbol, mid)
        return mid - p if is_long else mid + p

    def _calc_pnl(self, symbol: str, price_change: float, qty: float, is_long: bool) -> float:
        return price_change * qty * self._vpp(symbol) * (1 if is_long else -1)

    def _commission(self, symbol: str, qty: float) -> float:
        """Commission for one side (open or close) in account currency."""
        spec = self._symbol_specs.get(symbol)
        if spec is None or spec.commission_per_lot == 0:
            return 0.0
        lots = qty / spec.contract_size
        return lots * spec.commission_per_lot

    # ------------------------------------------------------------------
    # Margin
    # ------------------------------------------------------------------

    def _position_margin(self, symbol: str, price: float, qty: float) -> float:
        """Margin required to hold this position in account currency."""
        return price * qty * self._vpp(symbol) * self._leverage

    def max_affordable_qty(self, symbol: str, price: float) -> float:
        """Maximum quantity currently affordable by margin at ``price``."""
        per_unit_margin = price * self._vpp(symbol) * self._leverage
        if per_unit_margin <= 0:
            return 0.0
        qty = max(0.0, self.margin_available / per_unit_margin)
        spec = self._symbol_specs.get(symbol)
        if spec is not None and spec.volume_max > 0 and spec.contract_size > 0:
            qty = min(qty, spec.volume_max * spec.contract_size)
        return qty

    def _margin_used(self) -> float:
        return sum(
            self._position_margin(pos.symbol, pos.entry_price, pos.qty)
            for pos in self._positions.values()
        )

    @property
    def margin_available(self) -> float:
        return max(0.0, self.equity - self._margin_used())

    # ------------------------------------------------------------------
    # Unrealized PnL / equity
    # ------------------------------------------------------------------

    def _unrealized_pnl(self) -> float:
        """Total unrealized PnL across ALL strategies on this broker (net of spread/slippage)."""
        return sum(
            self._calc_pnl(
                pos.symbol,
                self._exit_price(pos.symbol, self._symbol_prices.get(pos.symbol, pos.entry_price), pos.is_long) - pos.entry_price,
                pos.qty,
                pos.is_long,
            )
            for pos in self._positions.values()
        )

    @property
    def balance(self) -> float:
        return self._balance

    @property
    def equity(self) -> float:
        """Total account equity across all strategies (balance + all unrealized PnL)."""
        return self._balance + self._unrealized_pnl()

    # ------------------------------------------------------------------
    # Price feed
    # ------------------------------------------------------------------

    def update_price(self, symbol: str, price: float, time: datetime | None = None) -> None:
        """Push the current bar price (and optionally time) into the broker."""
        self._symbol_prices[symbol] = price
        if time is not None:
            self._current_time = time
        self.equity_curve.append(EquityPoint(
            time=self._current_time,
            equity=self.equity,
            balance=self.balance,
        ))

    def prepare_bar(self, symbol: str, open: float, time: datetime | None = None) -> None:
        """Mark the start-of-bar state without recording an equity-curve point."""
        self._symbol_prices[symbol] = open
        if time is not None:
            self._current_time = time

    def set_bar_spread_pct(self, symbol: str, spread_pct: float | None) -> None:
        """Override spread for the current bar using historical spread data."""
        self._bar_spread_pct[symbol] = spread_pct

    # ------------------------------------------------------------------
    # Order execution
    # ------------------------------------------------------------------

    def execute_signal(self, signal: EntrySignal) -> Position:
        exec_qty = self._normalize_qty(signal.symbol, signal.qty)
        if exec_qty <= 0:
            raise SignalExecutionError(
                "Invalid volume after broker lot constraints",
                SignalExecutionErrorReason.INVALID_PARAMS,
            )
        mid = self._symbol_prices.get(signal.symbol, signal.price)
        is_long = signal.direction == 'LONG'
        fill_price = self._fill_price(signal.symbol, mid, is_long)

        margin_required = self._position_margin(signal.symbol, fill_price, exec_qty)
        if margin_required > self.margin_available:
            raise SignalExecutionError(
                f"Insufficient margin: required={margin_required:.2f}, available={self.margin_available:.2f}",
                SignalExecutionErrorReason.INSUFFICIENT_FUNDS,
            )

        open_commission = self._commission(signal.symbol, exec_qty)
        self._balance -= open_commission

        pos = Position(
            id=self._next_id,
            symbol=signal.symbol,
            strategy_id=signal.strategy_id,
            direction=SignalDirection.LONG if is_long else SignalDirection.SHORT,
            qty=exec_qty,
            entry_time=self._current_time,
            entry_price=fill_price,
            sl_price=signal.sl,
            tp_price=signal.tp,
        )
        self._next_id += 1
        self._positions[pos.id] = pos
        self._entry_mids[pos.id] = mid
        self._entry_spread_pct[pos.id] = self._current_spread_pct(pos.symbol)
        self._entry_commissions[pos.id] = open_commission
        return pos

    def shutdown(self):
        pass

    def get_positions(self, symbol: str, strategy_id: int) -> list[Position]:
        return [pos for pos in self._positions.values() if pos.symbol == symbol and pos.strategy_id == strategy_id]

    def _do_close(
        self,
        pos: Position,
        exit_price: float,
        exit_mid: float | None = None,
        exit_spread_pct: float | None = None,
    ) -> None:
        """Close a position at a given price, settle balance, and record the trade."""
        if exit_mid is None:
            exit_mid = exit_price
        pnl_after_spread = self._calc_pnl(pos.symbol, exit_price - pos.entry_price, pos.qty, pos.is_long)
        close_commission = self._commission(pos.symbol, pos.qty)
        self._balance += pnl_after_spread - close_commission

        entry_mid = self._entry_mids.pop(pos.id, pos.entry_price)
        entry_spread_pct = self._entry_spread_pct.pop(pos.id, 0.0)
        open_commission = self._entry_commissions.pop(pos.id, 0.0)
        true_net_pnl = pnl_after_spread - close_commission - open_commission

        spec = self._symbol_specs.get(pos.symbol)
        if spec is not None:
            if exit_spread_pct is None:
                exit_spread_pct = self._current_spread_pct(pos.symbol)
            spread_cost = (
                entry_mid * entry_spread_pct / 200
                + exit_mid * exit_spread_pct / 200
            ) * pos.qty * spec.value_per_point
            mid_sum = entry_mid + exit_mid
            slippage_cost = mid_sum * spec.slippage_pct / 100 * pos.qty * spec.value_per_point
        else:
            spread_cost = slippage_cost = 0.0
        commission_cost = open_commission + close_commission

        self.trade_log.append(ClosedTrade(
            position=pos,
            exit_time=self._current_time,
            exit_price=exit_price,
            pnl=true_net_pnl,
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            commission_cost=commission_cost,
        ))
        del self._positions[pos.id]

    def close_positions(self, exit_signal: ExitSignal) -> list[tuple[Position, PositionCloseError]]:
        all_positions = self.get_positions(exit_signal.symbol, exit_signal.strategy_id)
        to_close = (
            all_positions if not exit_signal.position_ids
            else [p for p in all_positions if p.id in exit_signal.position_ids]
        )
        for pos in to_close:
            exit_mid = self._symbol_prices.get(pos.symbol, pos.entry_price)
            exit_spread_pct = self._current_spread_pct(pos.symbol)
            spec = self._symbol_specs.get(pos.symbol)
            slippage_pct = spec.slippage_pct if spec is not None else 0.0
            penalty = exit_mid * (exit_spread_pct / 200 + slippage_pct / 100)
            exit_price = exit_mid - penalty if pos.is_long else exit_mid + penalty
            self._do_close(pos, exit_price, exit_mid, exit_spread_pct)
        return []

    # ------------------------------------------------------------------
    # Pending signal queue
    # ------------------------------------------------------------------

    def queue_signal(self, signal: EntrySignal) -> PendingSignal:
        """Add a LIMIT or STOP signal to the pending queue."""
        ps = PendingSignal(
            id=self._next_signal_id,
            signal=signal,
            queued_time=self._current_time,
        )
        self._pending_signals[ps.id] = ps
        self._next_signal_id += 1
        return ps

    def cancel_signal(self, signal_id: int) -> bool:
        return self._pending_signals.pop(signal_id, None) is not None

    def get_pending_signals(self, symbol: str, strategy_id: int) -> list[PendingSignal]:
        return [
            ps for ps in self._pending_signals.values()
            if ps.signal.symbol == symbol and ps.signal.strategy_id == strategy_id
        ]

    def fill_pending_signals(
        self,
        symbol: str,
        open: float,
        high: float,
        low: float,
        strategy_id: int,
    ) -> list[tuple[PendingSignal, "Position | SignalExecutionError"]]:
        """Fill any pending signals that triggered within this bar.

        Trigger and fill rules:
        - LIMIT LONG  : triggers when ``low  <= signal.price``; fills at ``min(open, signal.price)``
          (optimistic: if price gapped below the limit we get the better open price)
        - LIMIT SHORT : triggers when ``high >= signal.price``; fills at ``max(open, signal.price)``
        - STOP  LONG  : triggers when ``high >= signal.price``; fills at ``max(open, signal.price)``
          (pessimistic: gaps through the stop fill at open, which may be worse)
        - STOP  SHORT : triggers when ``low  <= signal.price``; fills at ``min(open, signal.price)``

        Spread and slippage are applied on top of the trigger mid price via
        ``_fill_price``, exactly as for market orders.
        """
        results: list[tuple[PendingSignal, Position | SignalExecutionError]] = []
        to_check = [
            ps for ps in list(self._pending_signals.values())
            if ps.signal.symbol == symbol and ps.signal.strategy_id == strategy_id
        ]
        for ps in to_check:
            sig = ps.signal
            is_long = sig.direction == 'LONG'
            trigger_mid: float | None = None

            if sig.type == SignalType.MARKET:
                triggered = True
                trigger_mid = open  # always fill at the bar's open (next-bar execution)
            elif sig.type == SignalType.LIMIT:
                triggered = (is_long and low <= sig.price) or (not is_long and high >= sig.price)
                if triggered:
                    # Optimistic: fill at open if gap is better than limit
                    trigger_mid = min(open, sig.price) if is_long else max(open, sig.price)
            elif sig.type == SignalType.STOP:
                triggered = (is_long and high >= sig.price) or (not is_long and low <= sig.price)
                if triggered:
                    # Pessimistic: fill at open if gap is worse than stop
                    trigger_mid = max(open, sig.price) if is_long else min(open, sig.price)
            else:
                continue

            if not triggered:
                continue
            if trigger_mid is None:
                continue

            del self._pending_signals[ps.id]
            fill_price = self._fill_price(symbol, trigger_mid, is_long)

            exec_qty = self._normalize_qty(symbol, sig.qty)
            if exec_qty <= 0:
                results.append((ps, SignalExecutionError(
                    f"Invalid volume for pending signal {ps.id} after broker lot constraints",
                    SignalExecutionErrorReason.INVALID_PARAMS,
                )))
                continue

            margin_required = self._position_margin(symbol, fill_price, exec_qty)
            if margin_required > self.margin_available:
                results.append((ps, SignalExecutionError(
                    f"Insufficient margin for pending signal {ps.id}: "
                    f"required={margin_required:.2f}, available={self.margin_available:.2f}",
                    SignalExecutionErrorReason.INSUFFICIENT_FUNDS,
                )))
                continue

            open_commission = self._commission(symbol, exec_qty)
            self._balance -= open_commission

            pos = Position(
                id=self._next_id,
                symbol=symbol,
                strategy_id=sig.strategy_id,
                direction=SignalDirection.LONG if is_long else SignalDirection.SHORT,
                qty=exec_qty,
                entry_time=self._current_time,
                entry_price=fill_price,
                sl_price=sig.sl,
                tp_price=sig.tp,
            )
            self._next_id += 1
            self._positions[pos.id] = pos
            self._entry_mids[pos.id] = trigger_mid
            self._entry_spread_pct[pos.id] = self._current_spread_pct(symbol)
            self._entry_commissions[pos.id] = open_commission
            results.append((ps, pos))
        return results

    def check_sl_tp(
        self,
        symbol: str,
        open: float,
        high: float,
        low: float,
        strategy_id: int,
    ) -> list[tuple[Position, float, bool]]:
        hits: list[tuple[Position, float, bool]] = []
        positions = [
            p for p in list(self._positions.values())
            if p.symbol == symbol and p.strategy_id == strategy_id
        ]
        for pos in positions:
            sl_hit = (
                pos.sl_price is not None and
                (low <= pos.sl_price if pos.is_long else high >= pos.sl_price)
            )
            tp_hit = (
                not sl_hit and
                pos.tp_price is not None and
                (high >= pos.tp_price if pos.is_long else low <= pos.tp_price)
            )
            if not sl_hit and not tp_hit:
                continue

            if sl_hit:
                # Pessimistic: honour gaps through the stop
                trigger_mid = min(open, pos.sl_price) if pos.is_long else max(open, pos.sl_price)  # type: ignore[arg-type]
                exit_spread_pct = self._current_spread_pct(symbol)
                spec = self._symbol_specs.get(symbol)
                slippage_pct = spec.slippage_pct if spec is not None else 0.0
                penalty = trigger_mid * (exit_spread_pct / 200 + slippage_pct / 100)
                exit_price = trigger_mid - penalty if pos.is_long else trigger_mid + penalty
                self._do_close(pos, exit_price, trigger_mid, exit_spread_pct)
                hits.append((pos, exit_price, True))
            else:
                # Optimistic: fill at TP price or better if price gapped past it
                trigger_mid = max(open, pos.tp_price) if pos.is_long else min(open, pos.tp_price)  # type: ignore[arg-type]
                exit_spread_pct = self._current_spread_pct(symbol)
                spec = self._symbol_specs.get(symbol)
                slippage_pct = spec.slippage_pct if spec is not None else 0.0
                penalty = trigger_mid * (exit_spread_pct / 200 + slippage_pct / 100)
                exit_price = trigger_mid - penalty if pos.is_long else trigger_mid + penalty
                self._do_close(pos, exit_price, trigger_mid, exit_spread_pct)
                hits.append((pos, exit_price, False))
        return hits

    # ------------------------------------------------------------------
    # PnL queries
    # ------------------------------------------------------------------

    def pnl(self, strategy_id: int, position_id: int | None = None) -> float:
        if position_id is not None:
            pos = self._positions.get(position_id)
            if pos is None:
                return 0.0
            mid = self._symbol_prices.get(pos.symbol, pos.entry_price)
            exit_price = self._exit_price(pos.symbol, mid, pos.is_long)
            return self._calc_pnl(pos.symbol, exit_price - pos.entry_price, pos.qty, pos.is_long)
        return sum(
            self._calc_pnl(
                pos.symbol,
                self._exit_price(pos.symbol, self._symbol_prices.get(pos.symbol, pos.entry_price), pos.is_long) - pos.entry_price,
                pos.qty,
                pos.is_long,
            )
            for pos in self._positions.values()
            if pos.strategy_id == strategy_id
        )

    def pnl_pct(self, strategy_id: int, position_id: int | None = None) -> float:
        if position_id is not None:
            pos = self._positions.get(position_id)
            if pos is None or pos.entry_price == 0:
                return 0.0
            cost = pos.entry_price * pos.qty * self._vpp(pos.symbol)
            return self.pnl(strategy_id, position_id) / cost * 100 if cost else 0.0
        positions = [p for p in self._positions.values() if p.strategy_id == strategy_id]
        if not positions:
            return 0.0
        total_cost = sum(p.entry_price * p.qty * self._vpp(p.symbol) for p in positions)
        return self.pnl(strategy_id) / total_cost * 100 if total_cost else 0.0

    # ------------------------------------------------------------------
    # Position modification
    # ------------------------------------------------------------------

    def update_sl(self, position_id: int, new_sl_price: float) -> Position | None:
        pos = self._positions.get(position_id)
        if pos is None:
            return None  # already closed (e.g. hit SL/TP earlier this bar) — silently ignore
        updated = replace(pos, sl_price=new_sl_price)
        self._positions[position_id] = updated
        return updated

    def update_tp(self, position_id: int, new_tp_price: float) -> Position | None:
        pos = self._positions.get(position_id)
        if pos is None:
            return None  # already closed — silently ignore
        updated = replace(pos, tp_price=new_tp_price)
        self._positions[position_id] = updated
        return updated
