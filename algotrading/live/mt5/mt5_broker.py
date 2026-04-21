from datetime import datetime, timezone
from functools import cache
import math

import MetaTrader5 as mt5

from algotrading.backtest.backtest_broker import SymbolSpec
from algotrading.core.broker import (
    Broker, BrokerError,
    SignalExecutionError, SignalExecutionErrorReason,
    PositionCloseError, PositionCloseErrorReason,
    PositionModifyError, PositionModifyErrorReason,
)
from algotrading.core.position import Position
from algotrading.core.signal import (
    EntrySignal,
    ExitSignal,
    PendingSignal,
    SignalDirection,
    SignalType,
)


class MT5BrokerError(BrokerError):
    """Custom exception for MT5 broker-related errors."""
    pass


class MT5Broker(Broker):
    _DEVIATION = 4

    def __init__(self):
        super().__init__()
        self._tracked_positions: dict[tuple[str, int], dict[int, Position]] = {}
        self._pending_signals: dict[int, PendingSignal] = {}
        if not mt5.initialize(): # type: ignore
            raise MT5BrokerError(f"MT5 initialization failed: {mt5.last_error()}") # type: ignore

    _PENDING_ORDER_TYPE_MAP = {
        ("LONG", SignalType.LIMIT): mt5.ORDER_TYPE_BUY_LIMIT,
        ("SHORT", SignalType.LIMIT): mt5.ORDER_TYPE_SELL_LIMIT,
        ("LONG", SignalType.STOP): mt5.ORDER_TYPE_BUY_STOP,
        ("SHORT", SignalType.STOP): mt5.ORDER_TYPE_SELL_STOP,
    }

    def _position_from_raw(self, pos) -> Position:
        return Position(
            id=pos.ticket,
            symbol=pos.symbol,
            strategy_id=pos.magic,
            direction=SignalDirection.LONG if pos.type == mt5.ORDER_TYPE_BUY else SignalDirection.SHORT, # type: ignore
            qty=self._lots_to_qty(pos.symbol, pos.volume),
            entry_time=datetime.fromtimestamp(pos.time, tz=timezone.utc),
            entry_price=pos.price_open,
            sl_price=pos.sl or None,
            tp_price=pos.tp or None,
        )

    def _is_sl_or_tp_hit(self, position_id: int) -> tuple[float, bool] | None:
        """Return (exit_price, is_sl) if MT5 closed the position via SL/TP."""
        deals = mt5.history_deals_get(position=position_id) # type: ignore
        if not deals:
            return None

        deal_reason_sl = getattr(mt5, "DEAL_REASON_SL", -1)
        deal_reason_tp = getattr(mt5, "DEAL_REASON_TP", -1)
        deal_entry_out = getattr(mt5, "DEAL_ENTRY_OUT", None)

        for deal in sorted(deals, key=lambda d: (d.time, d.ticket), reverse=True):
            if deal_entry_out is not None and getattr(deal, "entry", None) != deal_entry_out:
                continue
            if deal.reason == deal_reason_sl:
                return float(deal.price), True
            if deal.reason == deal_reason_tp:
                return float(deal.price), False
        return None

    @staticmethod
    @cache
    def _build_signal_retcode_reasons() -> dict[int, SignalExecutionErrorReason]:
        R = SignalExecutionErrorReason
        return {
            mt5.TRADE_RETCODE_REQUOTE: R.REQUOTE,
            mt5.TRADE_RETCODE_PRICE_CHANGED: R.REQUOTE,
            mt5.TRADE_RETCODE_PRICE_OFF:  R.REQUOTE,
            mt5.TRADE_RETCODE_INVALID_VOLUME: R.INVALID_PARAMS,
            mt5.TRADE_RETCODE_INVALID_PRICE: R.INVALID_PARAMS,
            mt5.TRADE_RETCODE_INVALID_STOPS: R.INVALID_PARAMS,
            mt5.TRADE_RETCODE_INVALID_EXPIRATION: R.INVALID_PARAMS,
            mt5.TRADE_RETCODE_INVALID_FILL: R.INVALID_PARAMS,
            mt5.TRADE_RETCODE_INVALID: R.INVALID_PARAMS,
            mt5.TRADE_RETCODE_TRADE_DISABLED: R.BROKER_REJECTED,
            mt5.TRADE_RETCODE_SERVER_DISABLES_AT:R.BROKER_REJECTED,
            mt5.TRADE_RETCODE_CLIENT_DISABLES_AT:R.BROKER_REJECTED,
            mt5.TRADE_RETCODE_LONG_ONLY: R.BROKER_REJECTED,
            mt5.TRADE_RETCODE_SHORT_ONLY: R.BROKER_REJECTED,
            mt5.TRADE_RETCODE_CLOSE_ONLY: R.BROKER_REJECTED,
            mt5.TRADE_RETCODE_LIMIT_ORDERS: R.BROKER_REJECTED,
            mt5.TRADE_RETCODE_LIMIT_VOLUME: R.BROKER_REJECTED,
            mt5.TRADE_RETCODE_LIMIT_POSITIONS: R.BROKER_REJECTED,
            mt5.TRADE_RETCODE_MARKET_CLOSED: R.MARKET_CLOSED,
            mt5.TRADE_RETCODE_NO_MONEY: R.INSUFFICIENT_FUNDS,
            mt5.TRADE_RETCODE_CONNECTION: R.CONNECTION_ERROR,
            mt5.TRADE_RETCODE_TIMEOUT: R.CONNECTION_ERROR,
        }

    @staticmethod
    @cache
    def _build_close_retcode_reasons() -> dict[int, PositionCloseErrorReason]:
        R = PositionCloseErrorReason
        return {
            mt5.TRADE_RETCODE_REQUOTE: R.REQUOTE,
            mt5.TRADE_RETCODE_PRICE_CHANGED: R.REQUOTE,
            mt5.TRADE_RETCODE_PRICE_OFF: R.REQUOTE,
            mt5.TRADE_RETCODE_POSITION_CLOSED: R.POSITION_NOT_FOUND,
            mt5.TRADE_RETCODE_INVALID_ORDER: R.POSITION_NOT_FOUND,
            mt5.TRADE_RETCODE_INVALID_CLOSE_VOLUME: R.POSITION_NOT_FOUND,
            mt5.TRADE_RETCODE_CLOSE_ORDER_EXIST: R.POSITION_NOT_FOUND,
            mt5.TRADE_RETCODE_FIFO_CLOSE: R.BROKER_REJECTED,
            mt5.TRADE_RETCODE_CONNECTION: R.CONNECTION_ERROR,
            mt5.TRADE_RETCODE_TIMEOUT: R.CONNECTION_ERROR,
        }

    @property
    def balance(self) -> float:
        info = mt5.account_info() # type: ignore
        if info is None:
            raise MT5BrokerError(f"Failed to get account info: {mt5.last_error()}") # type: ignore
        return info.balance

    @property
    def equity(self) -> float:
        info = mt5.account_info() # type: ignore
        if info is None:
            raise MT5BrokerError(f"Failed to get account info: {mt5.last_error()}") # type: ignore
        return info.equity

    def execute_signal(self, signal: EntrySignal) -> Position:
        lots = self._qty_to_lots(signal.symbol, signal.qty)
        if lots <= 0:
            raise SignalExecutionError(
                "Invalid volume after broker lot constraints",
                SignalExecutionErrorReason.INVALID_PARAMS,
            )
        order = self._create_order(signal, lots)
        result = mt5.order_send(order) # type: ignore
        if result is None:
            raise SignalExecutionError(
                f"Order send returned None. MT5 error: {mt5.last_error()}", # type: ignore
                SignalExecutionErrorReason.UNKNOWN,
            )
        if result.retcode != mt5.TRADE_RETCODE_DONE: # type: ignore
            reason = self._build_signal_retcode_reasons().get(result.retcode, SignalExecutionErrorReason.BROKER_REJECTED)
            raise SignalExecutionError(
                f"Order execution failed: retcode={result.retcode}, comment={result.comment}",
                reason,
            )
        positions = mt5.positions_get(ticket=result.order) # type: ignore
        if not positions:
            raise SignalExecutionError(
                f"Order executed but position not found for ticket={result.order}",
                SignalExecutionErrorReason.UNKNOWN,
            )
        position = self._position_from_raw(positions[0])
        # Register so the next check_sl_tp can detect a same-bar SL/TP close.
        tracked = self._tracked_positions.setdefault((position.symbol, position.strategy_id), {})
        tracked[position.id] = position
        return position

    def _create_order(self, signal: EntrySignal, lots: float):
        return {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": signal.symbol,
            "volume": lots,
            "type": mt5.ORDER_TYPE_BUY if signal.direction == "LONG" else mt5.ORDER_TYPE_SELL,
            "price": signal.price,
            "sl": signal.sl or 0.0,
            "tp": signal.tp or 0.0,
            "deviation": signal.max_slippage if signal.max_slippage is not None else self._DEVIATION,
            "magic": signal.strategy_id,
        }

    def queue_signal(self, signal: EntrySignal) -> PendingSignal:
        lots = self._qty_to_lots(signal.symbol, signal.qty)
        if lots <= 0:
            raise SignalExecutionError(
                "Invalid volume after broker lot constraints",
                SignalExecutionErrorReason.INVALID_PARAMS,
            )
        order_type = self._PENDING_ORDER_TYPE_MAP.get((signal.direction, signal.type))
        if order_type is None:
            raise SignalExecutionError(
                f"Unsupported pending order: direction={signal.direction}, type={signal.type}",
                SignalExecutionErrorReason.INVALID_PARAMS,
            )
        request = {
            "action": mt5.TRADE_ACTION_PENDING, # type: ignore
            "symbol": signal.symbol,
            "volume": lots,
            "type": order_type,
            "price": signal.price,
            "sl": signal.sl or 0.0,
            "tp": signal.tp or 0.0,
            "magic": signal.strategy_id,
            "type_time": mt5.ORDER_TIME_GTC, # type: ignore
        }
        result = mt5.order_send(request) # type: ignore
        if result is None:
            raise SignalExecutionError(
                f"Pending order send returned None. MT5 error: {mt5.last_error()}", # type: ignore
                SignalExecutionErrorReason.UNKNOWN,
            )
        if result.retcode != mt5.TRADE_RETCODE_DONE: # type: ignore
            reason = self._build_signal_retcode_reasons().get(result.retcode, SignalExecutionErrorReason.BROKER_REJECTED)
            raise SignalExecutionError(
                f"Pending order failed: retcode={result.retcode}, comment={result.comment}",
                reason,
            )
        ticket = int(result.order)
        ps = PendingSignal(
            id=ticket,
            signal=signal,
            queued_time=datetime.now(timezone.utc),
        )
        self._pending_signals[ticket] = ps
        return ps

    def cancel_signal(self, signal_id: int) -> bool:
        ps = self._pending_signals.get(signal_id)
        if ps is None:
            return False
        request = {
            "action": mt5.TRADE_ACTION_REMOVE, # type: ignore
            "order": signal_id,
        }
        result = mt5.order_send(request) # type: ignore
        # Treat "order already gone" as success so the local queue stays consistent.
        already_gone = result is not None and result.retcode in (
            mt5.TRADE_RETCODE_INVALID_ORDER, # type: ignore
            mt5.TRADE_RETCODE_POSITION_CLOSED, # type: ignore
        )
        if result is not None and (result.retcode == mt5.TRADE_RETCODE_DONE or already_gone): # type: ignore
            del self._pending_signals[signal_id]
            return True
        return False

    def get_pending_signals(self, symbol: str, strategy_id: int) -> list[PendingSignal]:
        live_orders = mt5.orders_get(symbol=symbol) or () # type: ignore
        live_tickets = {int(o.ticket) for o in live_orders if o.magic == strategy_id}
        return [
            ps for ticket, ps in self._pending_signals.items()
            if ticket in live_tickets
            and ps.signal.symbol == symbol
            and ps.signal.strategy_id == strategy_id
        ]

    def fill_pending_signals(
        self,
        symbol: str,
        open: float,
        high: float,
        low: float,
        strategy_id: int,
    ) -> list[tuple[PendingSignal, "Position | SignalExecutionError"]]:
        """Detect server-side fills of tracked pending orders.

        MT5 fills pending orders server-side when price hits the trigger, so the
        bar OHL inputs are ignored. For each tracked pending that no longer
        appears in ``orders_get``, we consult ``history_orders_get`` to see if it
        filled (→ emit Position) or was cancelled externally (→ drop silently).
        """
        _ = open, high, low
        results: list[tuple[PendingSignal, Position | SignalExecutionError]] = []
        live_orders = mt5.orders_get(symbol=symbol) or () # type: ignore
        live_tickets = {int(o.ticket) for o in live_orders if o.magic == strategy_id}

        for ticket in list(self._pending_signals.keys()):
            ps = self._pending_signals[ticket]
            if ps.signal.symbol != symbol or ps.signal.strategy_id != strategy_id:
                continue
            if ticket in live_tickets:
                continue

            del self._pending_signals[ticket]
            historical = mt5.history_orders_get(ticket=ticket) # type: ignore
            if not historical:
                continue
            hist_order = historical[0]
            if hist_order.state != mt5.ORDER_STATE_FILLED: # type: ignore
                continue
            position_id = int(getattr(hist_order, "position_id", 0) or 0)
            if not position_id:
                continue
            positions = mt5.positions_get(ticket=position_id) # type: ignore
            if not positions:
                continue
            position = self._position_from_raw(positions[0])
            tracked = self._tracked_positions.setdefault((position.symbol, position.strategy_id), {})
            tracked[position.id] = position
            results.append((ps, position))
        return results

    def get_symbol_spec(
        self,
        symbol: str,
        slippage_pct: float = 0.0,
        commission_per_lot: float = 0.0,
    ) -> SymbolSpec:
        """Build a SymbolSpec for use in BacktestBroker by querying live MT5 data.

        `value_per_point` and `spread_pct` are read from live MT5 data.
        `contract_size` is read from the symbol's trade_contract_size.
        `slippage_pct` and `commission_per_lot` must be supplied manually.
        `swap_long_per_day` and `swap_short_per_day` are read from MT5 swap rates.

        Swap rates are fetched from MT5 (in account currency per lot per day),
        negated (since MT5 uses negative for costs), and divided by contract_size
        to convert from per-lot to per-unit. This gives daily swap costs/credits
        in account currency per engine qty unit.

        Returned `value_per_point` is normalized to one engine qty unit, not one
        broker lot, so backtest PnL/margin math matches live order sizing.

        `commission_per_lot` is per side in account currency. If your broker
        charges $7 round-trip per lot and your account is EUR at 1.10 EURUSDa,
        pass `commission_per_lot = 3.50 / 1.10` (~3.18).

        Call this once before constructing BacktestBroker::

            spec = live_broker.get_symbol_spec("EURUSD", commission_per_lot=3.18)
            backtest_broker = BacktestBroker(symbol_specs={"EURUSD": spec})
        """
        info = self._get_symbol_info(symbol)
        tick = mt5.symbol_info_tick(symbol) # type: ignore
        if tick is None:
            raise MT5BrokerError(f"Failed to get tick for '{symbol}': {mt5.last_error()}") # type: ignore
        mid = (tick.bid + tick.ask) / 2
        spread_pct = ((tick.ask - tick.bid) / mid) * 100
        contract_size = float(info.trade_contract_size)
        if info.trade_tick_size == 0 or contract_size == 0:
            raise MT5BrokerError(f"Invalid tick size or contract size for '{symbol}'")
        value_per_point = (info.trade_tick_value / info.trade_tick_size) / contract_size
        
        # Fetch swap rates from MT5. These are in account currency per lot per day.
        # Negative values in MT5 mean costs to the trader (what they pay).
        # We want positive values to represent costs, so negate MT5's values
        # and divide by contract_size to convert from per-lot to per-unit.
        swap_long_per_lot = float(info.swap_long) if info.swap_long else 0.0
        swap_short_per_lot = float(info.swap_short) if info.swap_short else 0.0
        # Negate (flip sign) and divide by contract_size to get per-unit cost
        swap_long_per_day = -swap_long_per_lot / contract_size if contract_size > 0 else 0.0
        swap_short_per_day = -swap_short_per_lot / contract_size if contract_size > 0 else 0.0
        
        return SymbolSpec(
            value_per_point=value_per_point,
            spread_pct=spread_pct,
            slippage_pct=slippage_pct,
            contract_size=contract_size,
            commission_per_lot=commission_per_lot,
            volume_min=float(info.volume_min),
            volume_max=float(info.volume_max),
            volume_step=float(info.volume_step),
            swap_long_per_day=swap_long_per_day,
            swap_short_per_day=swap_short_per_day,
        )

    def value_per_point(self, symbol: str) -> float:
        """Account-currency value of a 1-point move for one engine qty unit."""
        return self.get_symbol_spec(symbol).value_per_point

    def max_affordable_qty(self, symbol: str, price: float) -> float:
        """Maximum engine-qty currently affordable by free margin at ``price``.

        Uses MT5's live margin calculation (``order_calc_margin``) against the
        account's current free margin so ``buy()`` and ``sell()`` can clamp
        oversized orders before they hit the server — matching the backtest
        broker's behavior. Falls back to ``inf`` (no clamp) if MT5 can't
        compute the margin; the server will still reject with NO_MONEY.
        """
        info = mt5.account_info()  # type: ignore
        if info is None:
            return float("inf")
        margin_free = float(info.margin_free)
        if margin_free <= 0:
            return 0.0
        per_lot_margin = mt5.order_calc_margin(  # type: ignore
            mt5.ORDER_TYPE_BUY, symbol, 1.0, price  # type: ignore
        )
        if per_lot_margin is None or per_lot_margin <= 0:
            return float("inf")
        symbol_info = self._get_symbol_info(symbol)
        max_lots = margin_free / float(per_lot_margin)
        if symbol_info.volume_max > 0:
            max_lots = min(max_lots, float(symbol_info.volume_max))
        return max(0.0, max_lots * float(symbol_info.trade_contract_size))

    def _get_symbol_info(self, symbol: str):
        info = mt5.symbol_info(symbol) # type: ignore
        if info is None:
            raise MT5BrokerError(f"Symbol '{symbol}' not found in MT5")
        return info

    def _qty_to_lots(self, symbol: str, qty: float) -> float:
        symbol_info = self._get_symbol_info(symbol)
        if symbol_info.trade_contract_size <= 0:
            return 0.0
        lots = qty / symbol_info.trade_contract_size
        if symbol_info.volume_max > 0:
            lots = min(lots, symbol_info.volume_max)
        if symbol_info.volume_min > 0 and lots > 0:
            lots = max(lots, symbol_info.volume_min)
        step = symbol_info.volume_step
        if step > 0:
            lots = round(lots / step) * step
            decimals = max(0, -int(math.floor(math.log10(step)))) if step < 1 else 0
            lots = round(lots, min(8, decimals + 2))
        return max(0.0, lots)

    def _lots_to_qty(self, symbol: str, lots: float) -> float:
        return lots * self._get_symbol_info(symbol).trade_contract_size

    def close_positions(self, exit_signal: ExitSignal) -> list[tuple[Position, PositionCloseError]]:
        all_positions = self.get_positions(exit_signal.symbol, exit_signal.strategy_id)
        positions = (
            all_positions
            if not exit_signal.position_ids
            else [pos for pos in all_positions if pos.id in exit_signal.position_ids]
        )
        failures: list[tuple[Position, PositionCloseError]] = []
        for pos in positions:
            try:
                close_type = mt5.ORDER_TYPE_SELL if pos.direction == SignalDirection.LONG else mt5.ORDER_TYPE_BUY # type: ignore
                tick = mt5.symbol_info_tick(exit_signal.symbol) # type: ignore
                contract_size = self._get_symbol_info(exit_signal.symbol).trade_contract_size
                order = {
                    "action": mt5.TRADE_ACTION_DEAL, # type: ignore
                    "symbol": exit_signal.symbol,
                    "volume": pos.qty / contract_size if contract_size > 0 else pos.qty,
                    "type": close_type,
                    "price": tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask, # type: ignore
                    "deviation": self._DEVIATION,
                    "position": pos.id,
                }
                result = mt5.order_send(order) # type: ignore
                if result is None:
                    raise PositionCloseError(
                        f"Close position returned None. MT5 error: {mt5.last_error()}", # type: ignore
                        PositionCloseErrorReason.UNKNOWN,
                    )
                if result.retcode != mt5.TRADE_RETCODE_DONE: # type: ignore
                    reason = self._build_close_retcode_reasons().get(result.retcode, PositionCloseErrorReason.BROKER_REJECTED)
                    raise PositionCloseError(
                        f"Close position failed: retcode={result.retcode}, comment={result.comment}",
                        reason,
                    )
            except PositionCloseError as e:
                failures.append((pos, e))
        return failures

    def _get_live_positions(self, strategy_id: int, position_id: int | None):
        if position_id is not None:
            return mt5.positions_get(ticket=position_id) or () # type: ignore
        positions = mt5.positions_get() or () # type: ignore
        return [p for p in positions if p.magic == strategy_id]

    def pnl(self, strategy_id: int, position_id: int | None = None) -> float:
        positions = self._get_live_positions(strategy_id, position_id)
        return sum(p.profit for p in positions)

    def pnl_pct(self, strategy_id: int, position_id: int | None = None) -> float:
        positions = self._get_live_positions(strategy_id, position_id)
        if not positions:
            return 0.0
        total_pnl = sum(p.profit for p in positions)
        total_cost = sum(
            p.price_open * p.volume * self._get_symbol_info(p.symbol).trade_contract_size
            for p in positions
        )
        if total_cost == 0:
            return 0.0
        return (total_pnl / total_cost) * 100

    def _modify_position(self, position_id: int, sl: float, tp: float) -> Position:
        positions = mt5.positions_get(ticket=position_id) # type: ignore
        if not positions:
            raise PositionModifyError(
                f"Position {position_id} not found",
                PositionModifyErrorReason.POSITION_NOT_FOUND,
            )
        pos = positions[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP, # type: ignore
            "symbol": pos.symbol,
            "position": position_id,
            "sl": sl,
            "tp": tp,
        }
        result = mt5.order_send(request) # type: ignore
        if result is None:
            raise PositionModifyError(
                f"Modify position returned None. MT5 error: {mt5.last_error()}", # type: ignore
                PositionModifyErrorReason.UNKNOWN,
            )
        if result.retcode != mt5.TRADE_RETCODE_DONE: # type: ignore
            reason = PositionModifyErrorReason.BROKER_REJECTED
            if result.retcode in (mt5.TRADE_RETCODE_INVALID_STOPS, mt5.TRADE_RETCODE_INVALID): # type: ignore
                reason = PositionModifyErrorReason.INVALID_PARAMS
            elif result.retcode in (mt5.TRADE_RETCODE_CONNECTION, mt5.TRADE_RETCODE_TIMEOUT): # type: ignore
                reason = PositionModifyErrorReason.CONNECTION_ERROR
            raise PositionModifyError(
                f"Modify position failed: retcode={result.retcode}, comment={result.comment}",
                reason,
            )
        updated = mt5.positions_get(ticket=position_id) # type: ignore
        if not updated:
            raise PositionModifyError(
                f"Position {position_id} not found after modification",
                PositionModifyErrorReason.UNKNOWN,
            )
        return self._position_from_raw(updated[0])

    def update_sl(self, position_id: int, new_sl_price: float) -> Position | None:
        positions = mt5.positions_get(ticket=position_id) # type: ignore
        if not positions:
            raise PositionModifyError(
                f"Position {position_id} not found",
                PositionModifyErrorReason.POSITION_NOT_FOUND,
            )
        return self._modify_position(position_id, sl=new_sl_price, tp=positions[0].tp)

    def update_tp(self, position_id: int, new_tp_price: float) -> Position | None:
        positions = mt5.positions_get(ticket=position_id) # type: ignore
        if not positions:
            raise PositionModifyError(
                f"Position {position_id} not found",
                PositionModifyErrorReason.POSITION_NOT_FOUND,
            )
        return self._modify_position(position_id, sl=positions[0].sl, tp=new_tp_price)

    def _get_raw_positions(self, symbol: str, strategy_id: int):
        positions = mt5.positions_get(symbol=symbol) # type: ignore
        if positions is None:
            raise MT5BrokerError(f"Failed to get positions for symbol '{symbol}': {mt5.last_error()}") # type: ignore
        return [pos for pos in positions if pos.magic == strategy_id]

    def get_positions(self, symbol: str, strategy_id: int) -> list[Position]:
        return [
            self._position_from_raw(pos)
            for pos in self._get_raw_positions(symbol, strategy_id)
        ]

    def check_sl_tp(
        self,
        symbol: str,
        open: float,
        high: float,
        low: float,
        strategy_id: int,
    ) -> list[tuple[Position, float, bool]]:
        """Detect server-side SL/TP closes and report them like BacktestBroker.

        Live MT5 positions are closed by the trade server, so we detect transitions
        from "open" to "closed" and inspect deal reasons for SL/TP.
        """
        _ = open, high, low
        key = (symbol, strategy_id)
        previous = self._tracked_positions.get(key)
        current_positions = self.get_positions(symbol, strategy_id)
        current_map = {pos.id: pos for pos in current_positions}
        self._tracked_positions[key] = current_map

        if previous is None:
            return []

        hits: list[tuple[Position, float, bool]] = []
        for position_id, prev_pos in previous.items():
            if position_id in current_map:
                continue
            hit = self._is_sl_or_tp_hit(position_id)
            if hit is None:
                continue
            exit_price, is_sl = hit
            hits.append((prev_pos, exit_price, is_sl))
        return hits

    def shutdown(self):
        mt5.shutdown() # type: ignore
