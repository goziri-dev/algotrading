# Writing Strategies

This guide builds from the smallest possible strategy up to the full feature set, covering the entire `Strategy` API.

---

## 1. The Minimal Strategy

Every strategy subclasses `Strategy` and implements one abstract method: `next()`. The engine calls `next()` once per completed bar, after all indicators are updated.

```python
from algotrading import Strategy, StrategyParams
from algotrading.utils import fraction_to_qty

class MyStrategy(Strategy[StrategyParams]):
    def next(self) -> None:
        vpp = self.broker.value_per_point(self.symbol)
        qty = fraction_to_qty(self.equity, 10.0, self.price, vpp)
        self.buy(qty=qty)
```

**Key properties available in `next()`:**

| Property | Type | Description |
|---|---|---|
| `self.price` | `float` | Current bar's close price |
| `self.equity` | `float` | Account equity (balance + unrealized PnL) |
| `self.positions` | `list[Position]` | Open positions for this strategy |
| `self.symbol` | `str` | The symbol this strategy trades |
| `self.bars` | `Bars` | Full primary OHLC history |
| `self.broker` | `BrokerView` | Order and account access |

---

## 2. Strategy Parameters

Subclass `StrategyParams` to define typed, named parameters. Parameters are passed at construction and drive the strategy's unique ID — two instances with different params are treated as distinct strategies.

```python
from dataclasses import dataclass
from algotrading import Strategy, StrategyParams

@dataclass  # required so the new fields are picked up by the generated __init__
class MyParams(StrategyParams):
    fast_period: int = 10
    slow_period: int = 30
    risk_pct: float = 1.0

class MyStrategy(Strategy[MyParams]):
    def __init__(self, symbol: str, params: MyParams = MyParams()):
        super().__init__(symbol=symbol, params=params)
        # access params via self.params
        print(self.params.fast_period)

    def next(self) -> None:
        ...
```

`self.id` is an `int` derived from an MD5 hash of `(class name, symbol, params)`. Change any param value and you get a different ID — which matters for live trading where the broker uses the ID to route orders to the right strategy.

---

## 3. Indicators

Register indicators in `__init__` with `self.I()`. The engine feeds each indicator automatically before `next()` is called. `next()` is **not called** until every registered indicator has a finite (non-NaN) value.

```python
from algotrading.indicators import SMA, EMA, ATR

class MyStrategy(Strategy[MyParams]):
    def __init__(self, symbol: str, params: MyParams = MyParams()):
        super().__init__(symbol=symbol, params=params)
        self.fast = self.I(SMA(period=params.fast_period), source='close')
        self.slow = self.I(SMA(period=params.slow_period), source='close')

    def next(self) -> None:
        print(self.fast[-1])   # current bar value
        print(self.fast[-2])   # one bar ago
```

### Indicator indexing

Indicators use negative indexing like Python lists:

```python
self.sma[-1]   # current bar
self.sma[-2]   # previous bar
self.sma[-10]  # 10 bars ago — returns nan if not enough history
```

### Source types

`source` can be a bar field name, another indicator, or a tuple for multi-input indicators:

```python
# Single bar field
self.sma   = self.I(SMA(20),  source='close')
self.ema   = self.I(EMA(20),  source='open')

# Multi-input (ATR needs high, low, close)
self.atr   = self.I(ATR(14),  source=('high', 'low', 'close'))

# Chained: feed one indicator's output into another
self.sma_of_ema = self.I(SMA(10), source=self.ema)
```

### Multi-output indicators (BBANDS, ADX)

`BBANDS` and `ADX` return named tuples each call but also expose individual arrays:

```python
from algotrading.indicators import BBANDS, ADX

class MyStrategy(Strategy[StrategyParams]):
    def __init__(self, symbol, params=StrategyParams()):
        super().__init__(symbol=symbol, params=params)
        self.bb  = self.I(BBANDS(period=20, mult=2.0), source='close')
        self.adx = self.I(ADX(di_length=14, adx_smoothing=14), source=('high', 'low', 'close'))

    def next(self) -> None:
        # BBANDS — access individual band arrays
        mid   = self.bb.mid[-1]
        upper = self.bb.upper[-1]
        lower = self.bb.lower[-1]

        # ADX — same pattern
        adx_val   = self.adx.adx[-1]
        plus_di   = self.adx.plus_di[-1]
        minus_di  = self.adx.minus_di[-1]

        price_near_lower = self.price < lower * 1.001
        strong_trend     = adx_val > 25
```

### Crossover utilities

```python
from algotrading.utils import crossover, crossunder, bars_since

def next(self) -> None:
    # Did fast SMA cross above slow SMA on this bar?
    if crossover(self.fast, self.slow):
        ...

    # Did fast SMA cross below slow SMA on this bar?
    if crossunder(self.fast, self.slow):
        ...

    # How many bars since fast was above slow?
    n = bars_since(self.fast.value > self.slow.value)
    # returns inf if it has never been true
```

`crossover` / `crossunder` accept any mix of `Indicator`, numpy array, pandas Series, or scalar.

---

## 4. Position Sizing

Always size by equity, not by balance. Use the provided helpers — they account for `value_per_point` so sizing is correct across different symbols.

### `fraction_to_qty` — allocate a % of equity as notional

```python
from algotrading.utils import fraction_to_qty

def next(self) -> None:
    vpp = self.broker.value_per_point(self.symbol)
    # deploy 50% of current equity
    qty = fraction_to_qty(self.equity, 50.0, self.price, vpp)
    self.buy(qty=qty)
```

### `risk_to_qty` — size by how much you're willing to lose

```python
from algotrading.utils import risk_to_qty

def next(self) -> None:
    vpp = self.broker.value_per_point(self.symbol)
    sl  = self.price - 2 * self.atr[-1]       # 2× ATR stop
    # risk exactly 1% of equity on this trade
    qty = risk_to_qty(self.equity, 1.0, entry=self.price, stop_loss=sl, value_per_point=vpp)
    self.buy(qty=qty, sl_price=sl)
```

| Helper | Signature | Use when |
|---|---|---|
| `fraction_to_qty` | `(equity, fraction_pct, entry, value_per_point)` | Allocate X% of equity as notional value |
| `risk_to_qty` | `(equity, risk_pct, entry, stop_loss, value_per_point)` | Lose at most X% of equity if SL is hit |

`self.broker.max_affordable_qty(symbol, price)` returns the maximum quantity the account margin can support. `buy()` and `sell()` automatically cap qty at 99% of that value, so you don't need to check manually.

---

## 5. Entering Trades

### Market orders (immediate)

```python
self.buy(qty=qty)    # long at current close
self.sell(qty=qty)   # short at current close
```

### With stop-loss and take-profit

```python
atr = self.atr[-1]
sl  = self.price - 2 * atr
tp  = self.price + 4 * atr   # 2:1 RR

self.buy(qty=qty, sl_price=sl, tp_price=tp)
```

SL and TP are optional and independent — you can set one without the other.

### Limit orders

Pass an `entry_price` **below** the current price for a long limit, **above** for a short limit. The order waits in the pending queue until the bar's low (long) or high (short) reaches the level.

```python
# Buy limit 1% below current price
limit_price = self.price * 0.99
self.buy(qty=qty, entry_price=limit_price, sl_price=limit_price - atr)
```

### Stop orders

Pass an `entry_price` **above** the current price for a long stop-entry (breakout), **below** for a short stop-entry.

```python
# Buy on breakout above yesterday's high
breakout = self.bars.high[-2]
self.buy(qty=qty, entry_price=breakout)
```

The engine infers order type automatically from the relationship between `entry_price` and the current close. You never set `SignalType` directly.

### The `exclusive` flag

By default `exclusive=True`: when `buy()` or `sell()` is called, any open position in the opposite direction (or in the same direction if re-entering) is **closed first**. Set `exclusive=False` to stack positions:

```python
# Add to an existing long without closing it first
self.buy(qty=additional_qty, exclusive=False)
```

---

## 6. Managing Open Positions

### Inspecting positions

`self.positions` returns all open positions for this strategy on this symbol.

```python
def next(self) -> None:
    for pos in self.positions:
        print(pos.id)           # int — unique position ID
        print(pos.symbol)       # str
        print(pos.direction)    # SignalDirection.LONG / SHORT
        print(pos.qty)          # float
        print(pos.entry_price)  # float
        print(pos.entry_time)   # datetime (UTC)
        print(pos.sl_price)     # float | None
        print(pos.tp_price)     # float | None
        print(pos.is_long)      # bool convenience
        print(pos.is_short)     # bool convenience
        print(pos.sl_distance)  # abs(entry - sl) or None
```

### Checking unrealized PnL

```python
pnl     = self.pnl()          # total across all open positions
pnl_pos = self.pnl(pos.id)    # single position

pnl_pct     = self.pnl_pct()
pnl_pct_pos = self.pnl_pct(pos.id)
```

### Updating SL and TP (trailing stops)

```python
def next(self) -> None:
    for pos in self.positions:
        if pos.is_long:
            new_sl = self.price - 2 * self.atr[-1]
            if pos.sl_price is None or new_sl > pos.sl_price:
                self.update_sl(pos.id, new_sl)   # only ratchet up

        if pos.is_short:
            new_tp = self.price - 3 * self.atr[-1]
            self.update_tp(pos.id, new_tp)
```

`update_sl` and `update_tp` are also available as `self.broker.update_sl(pos_id, price)` / `self.broker.update_tp(pos_id, price)`.

### Closing positions manually

```python
self.close_positions()                    # close all
self.close_positions(position_ids=[pos.id])  # close specific
```

---

## 7. Pending Signals

Limit and stop-entry orders sit in the broker's pending queue until triggered. You can inspect and cancel them.

```python
def next(self) -> None:
    for ps in self.pending_signals:
        print(ps.id)             # unique signal ID
        print(ps.signal.price)   # trigger price
        print(ps.signal.type)    # "limit" or "stop"
        print(ps.queued_time)    # datetime it was submitted

    self.cancel_signal(ps.id)       # cancel one
    self.cancel_pending_signals()   # cancel all
```

A common pattern — cancel any stale limit if price has moved too far:

```python
def next(self) -> None:
    for ps in self.pending_signals:
        if abs(self.price - ps.signal.price) > 2 * self.atr[-1]:
            self.cancel_signal(ps.id)
```

---

## 8. Execution Callbacks

Override any of these hooks to react to order lifecycle events. They all have empty default implementations — only override what you need.

### `on_signal_execution_success`

Called when an order fills and a position opens.

```python
from algotrading.core.signal import EntrySignal
from algotrading.core.position import Position

def on_signal_execution_success(self, signal: EntrySignal, position: Position) -> None:
    print(f"Opened {position.direction} @ {position.entry_price}, id={position.id}")
```

### `on_signal_execution_error`

Called when an order fails. The `reason` tells you whether to retry.

```python
from algotrading.core.broker import SignalExecutionErrorReason

def on_signal_execution_error(self, signal: EntrySignal, reason: SignalExecutionErrorReason) -> None:
    if reason == SignalExecutionErrorReason.REQUOTE:
        # Price moved — safe to retry immediately
        self.buy(qty=signal.qty)
    elif reason == SignalExecutionErrorReason.INSUFFICIENT_FUNDS:
        pass  # don't retry
    else:
        print(f"Order failed: {reason}")
```

**`SignalExecutionErrorReason` values:**

| Value | Meaning |
|---|---|
| `REQUOTE` | Price moved — retry immediately |
| `INVALID_PARAMS` | Bad volume/price/stops — fix signal before retrying |
| `MARKET_CLOSED` | Retry when market reopens |
| `INSUFFICIENT_FUNDS` | Don't retry |
| `CONNECTION_ERROR` | Retry after reconnect |
| `BROKER_REJECTED` | Broker declined — unspecified |
| `UNKNOWN` | Unrecognised error |

### `on_signal_queued`

Called when a LIMIT or STOP signal is accepted into the pending queue.

```python
from algotrading.core.signal import PendingSignal

def on_signal_queued(self, pending: PendingSignal) -> None:
    print(f"Queued {pending.signal.type} @ {pending.signal.price}, id={pending.id}")
```

### `on_sl_hit` and `on_tp_hit`

Fire **before** `on_position_close_success`. Use them to log, record stats, or immediately re-enter.

```python
def on_sl_hit(self, position: Position, exit_price: float) -> None:
    loss = abs(position.entry_price - exit_price) * position.qty
    print(f"SL hit — lost {loss:.2f}")

def on_tp_hit(self, position: Position, exit_price: float) -> None:
    profit = abs(position.entry_price - exit_price) * position.qty
    print(f"TP hit — gained {profit:.2f}")
```

### `on_position_close_success`

Called after every successful close — whether from `close_positions()`, SL, or TP.

```python
def on_position_close_success(self, position: Position) -> None:
    print(f"Position {position.id} closed")
```

### `on_position_close_error`

Called when a close attempt fails (live only — backtests don't produce close errors).

```python
from algotrading.core.broker import PositionCloseErrorReason

def on_position_close_error(self, position: Position, reason: PositionCloseErrorReason) -> None:
    if reason == PositionCloseErrorReason.REQUOTE:
        self.close_positions([position.id])  # retry
```

---

## 9. Bars Object

`self.bars` is a `Bars` instance that grows one row per completed bar. Use standard negative indexing.

```python
def next(self) -> None:
    close_now  = self.bars.close[-1]
    close_prev = self.bars.close[-2]
    high_now   = self.bars.high[-1]
    low_now    = self.bars.low[-1]
    open_now   = self.bars.open[-1]
    time_now   = self.bars.time[-1]    # numpy datetime64

    # Full history as numpy arrays (no copy)
    all_closes = self.bars.close       # NDArray[float64]
    all_times  = self.bars.time        # NDArray[datetime64]
```

---

## 10. Multi-Timeframe Strategies

Call `self.new_bars(timeframe)` in `__init__` to register a secondary timeframe. The engine will feed that timeframe's bars automatically. Attach indicators to the secondary bars with the `bars=` argument to `self.I()`.

```python
import MetaTrader5 as mt5
from algotrading.indicators import SMA, ATR

class MTFStrategy(Strategy[StrategyParams]):
    def __init__(self, symbol: str, params: StrategyParams = StrategyParams()):
        super().__init__(symbol=symbol, params=params)

        # Primary timeframe indicators (M15)
        self.fast = self.I(SMA(10),  source='close')
        self.slow = self.I(SMA(30),  source='close')
        self.atr  = self.I(ATR(14),  source=('high', 'low', 'close'))

        # Secondary timeframe (H4) for trend filter
        self.bars_h4  = self.new_bars(timeframe=mt5.TIMEFRAME_H4)
        self.trend_h4 = self.I(SMA(50), source='close', bars=self.bars_h4)

        # Another secondary timeframe (D1)
        self.bars_d1  = self.new_bars(timeframe=mt5.TIMEFRAME_D1)
        self.trend_d1 = self.I(SMA(200), source='close', bars=self.bars_d1)

    def next(self) -> None:
        trend_up = self.price > self.trend_h4[-1] and self.price > self.trend_d1[-1]

        if crossover(self.fast, self.slow) and trend_up:
            vpp = self.broker.value_per_point(self.symbol)
            qty = fraction_to_qty(self.equity, 50.0, self.price, vpp)
            self.buy(qty=qty)
```

**Rules for secondary bars:**
- `new_bars()` must be called in `__init__`, before any data is fed.
- Each timeframe can only be registered once per strategy instance.
- Indicators registered on secondary bars are updated every time a secondary bar closes — not just on primary bar closes.
- `next()` waits until **all** indicators (primary and secondary) are finite before being called.

**Accessing secondary bar data directly:**

```python
def next(self) -> None:
    # Raw OHLC from H4 bars
    h4_close = self.bars_h4.close[-1]
    h4_high  = self.bars_h4.high[-1]
```

---

## 11. Dynamic Bar Fields (`update_bars`)

`update_bars()` is called after OHLC is appended but before indicators run. Override it to attach custom data (e.g. COT reports, sentiment scores, custom signals from an external feed) to the bar.

First, add the field in `__init__` using `self.bars.add_field()`:

```python
from datetime import datetime

from algotrading.data import COTClient
from algotrading.indicators import COTIndex

class COTStrategy(Strategy[StrategyParams]):
    def __init__(self, symbol: str, params: StrategyParams = StrategyParams()):
        super().__init__(symbol=symbol, params=params)
        self.bars.add_field("noncomm_net", int)
        self.bars.add_field("cot_spread", int)
        self.cot = self.I(COTIndex(26), source='cot_spread')

        # 088691 = COMEX Gold futures COT contract code
        self.cot_client = COTClient(contract_code="088691")
        self.cot_client.fetch_historical(start_date=datetime(2018, 1, 1))

    def update_bars(self) -> None:
        data = self.cot_client.strategy_data_at(self.bars.time[-1], auto_live_refresh=True)
        if data is not None:
            self.bars.append(
                noncomm_net=data["noncomm_net"],
                cot_spread=data["cot_spread"],
            )

    def next(self) -> None:
        cot_val = self.cot[-1]
        ...
```

`self.bars.append(**kwargs)` mutates the **last** appended row in place — it does not add a new row.

---

## 12. Shared State Between Strategies

`Strategy.shared_state` is a class-level dict accessible by all strategy instances. Use `write_state` / `read_state` to share a value computed by one strategy with others in the same session.

```python
class RegimeStrategy(Strategy[StrategyParams]):
    def __init__(self, symbol, params=StrategyParams()):
        super().__init__(symbol=symbol, params=params)
        self.trend = self.I(SMA(200), source='close')

    def next(self) -> None:
        self.write_state("bull_market", self.price > self.trend[-1])


class TradingStrategy(Strategy[StrategyParams]):
    def __init__(self, symbol, regime: RegimeStrategy, params=StrategyParams()):
        super().__init__(symbol=symbol, params=params)
        self._regime = regime

    def next(self) -> None:
        bull = self.read_state(self._regime, "bull_market")
        if bull is None or not bull:
            return  # treat uninitialized state as the conservative default
        ...
```

**Ordering caveat:** execution order across strategies matches the order of the `strategies` list passed to `MT5Session`. If `TradingStrategy.next()` runs before `RegimeStrategy.next()` on the first bar, `read_state` returns `None`. Always handle `None` as the safe/conservative case.

For portfolio-level guards like drawdown limits, computing inline is simpler and avoids the ordering problem entirely:

```python
def next(self) -> None:
    drawdown_pct = (1 - self.equity / self.params.initial_balance) * 100
    if drawdown_pct > 10.0:
        return
```

| Method | Signature | Description |
|---|---|---|
| `write_state(key, value)` | `(str, Any)` | Store a value under `key` for this strategy's ID |
| `read_state(strategy, key)` | `(Strategy, str)` | Read a value written by `strategy`. Returns `None` if missing. |

---

## 13. Putting It All Together — Advanced Example

This example combines most of the above: params, multi-timeframe, risk sizing, SL/TP, trailing stop, callbacks, and shared state guard.

```python
from dataclasses import dataclass
from datetime import datetime

import MetaTrader5 as mt5

from algotrading import Strategy, StrategyParams
from algotrading.core.broker import SignalExecutionErrorReason, PositionCloseErrorReason
from algotrading.core.position import Position
from algotrading.core.signal import EntrySignal, PendingSignal
from algotrading.indicators import SMA, ATR, RSI
from algotrading.utils import crossover, crossunder, risk_to_qty, bars_since


@dataclass
class AdvancedParams(StrategyParams):
    fast_period: int  = 10
    slow_period: int  = 30
    trend_period: int = 50
    atr_period: int   = 14
    rsi_period: int   = 14
    risk_pct: float   = 1.0
    sl_atr_mult: float = 2.0
    tp_atr_mult: float = 4.0


class AdvancedStrategy(Strategy[AdvancedParams]):

    def __init__(self, symbol: str, params: AdvancedParams = AdvancedParams()):
        super().__init__(symbol=symbol, params=params)
        p = params

        # Primary (M15) indicators
        self.fast = self.I(SMA(p.fast_period),  source='close')
        self.slow = self.I(SMA(p.slow_period),  source='close')
        self.atr  = self.I(ATR(p.atr_period),   source=('high', 'low', 'close'))
        self.rsi  = self.I(RSI(p.rsi_period),   source='close')

        # H4 trend filter
        self.bars_h4  = self.new_bars(timeframe=mt5.TIMEFRAME_H4)
        self.trend_h4 = self.I(SMA(p.trend_period), source='close', bars=self.bars_h4)

    # ------------------------------------------------------------------
    # Core logic
    # ------------------------------------------------------------------

    def next(self) -> None:
        # Portfolio-level risk guard (written by a separate RiskManager strategy)
        if self.read_state(self, "risk_off"):
            return

        p = self.params
        atr = self.atr[-1]
        vpp = self.broker.value_per_point(self.symbol)
        trend_up = self.price > self.trend_h4[-1]

        # Entry: SMA crossover aligned with H4 trend, RSI not overextended
        if crossover(self.fast, self.slow) and trend_up and self.rsi[-1] < 70:
            if not self.positions:
                sl = self.price - p.sl_atr_mult * atr
                tp = self.price + p.tp_atr_mult * atr
                qty = risk_to_qty(self.equity, p.risk_pct, self.price, sl, vpp)
                self.buy(qty=qty, sl_price=sl, tp_price=tp)

        elif crossunder(self.fast, self.slow) and not trend_up and self.rsi[-1] > 30:
            if not self.positions:
                sl = self.price + p.sl_atr_mult * atr
                tp = self.price - p.tp_atr_mult * atr
                qty = risk_to_qty(self.equity, p.risk_pct, self.price, sl, vpp)
                self.sell(qty=qty, sl_price=sl, tp_price=tp)

        # Trailing stop: ratchet SL toward price as trade runs in favour
        for pos in self.positions:
            atr = self.atr[-1]
            if pos.is_long:
                new_sl = self.price - p.sl_atr_mult * atr
                if pos.sl_price is None or new_sl > pos.sl_price:
                    self.update_sl(pos.id, new_sl)
            else:
                new_sl = self.price + p.sl_atr_mult * atr
                if pos.sl_price is None or new_sl < pos.sl_price:
                    self.update_sl(pos.id, new_sl)

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def on_signal_execution_success(self, signal: EntrySignal, position: Position) -> None:
        print(f"[{self.symbol}] Opened {position.direction} {position.qty:.4f} @ {position.entry_price}")

    def on_signal_execution_error(self, signal: EntrySignal, reason: SignalExecutionErrorReason) -> None:
        if reason == SignalExecutionErrorReason.REQUOTE:
            # Retry once at market
            vpp = self.broker.value_per_point(self.symbol)
            qty = risk_to_qty(
                self.equity, self.params.risk_pct,
                self.price, self.price - self.params.sl_atr_mult * self.atr[-1], vpp
            )
            self.buy(qty=qty)

    def on_sl_hit(self, position: Position, exit_price: float) -> None:
        bars_held = bars_since(self.bars.time == position.entry_time)
        print(f"SL hit after {bars_held} bars — {position.direction} {exit_price:.2f}")

    def on_tp_hit(self, position: Position, exit_price: float) -> None:
        print(f"TP hit — {position.direction} closed @ {exit_price:.2f}")

    def on_position_close_error(self, position: Position, reason: PositionCloseErrorReason) -> None:
        if reason == PositionCloseErrorReason.REQUOTE:
            self.close_positions([position.id])
```

---

## Quick Reference

### `Strategy` properties

| Property | Type | Description |
|---|---|---|
| `self.price` | `float` | Latest close price |
| `self.equity` | `float` | Balance + unrealized PnL |
| `self.positions` | `list[Position]` | Open positions |
| `self.symbol` | `str` | Trading symbol |
| `self.params` | `T` | Strategy parameters |
| `self.bars` | `Bars` | Primary OHLC history |
| `self.id` | `int` | Unique strategy ID |
| `self.broker` | `BrokerView` | Order and account access |

### `Strategy` methods

| Method | Description |
|---|---|
| `self.I(ind, source, bars=None)` | Register an indicator |
| `self.new_bars(timeframe)` | Register a secondary timeframe |
| `self.buy(qty, entry_price, sl_price, tp_price, exclusive)` | Long entry |
| `self.sell(qty, entry_price, sl_price, tp_price, exclusive)` | Short entry |
| `self.update_sl(position_id, sl_price)` | Move stop-loss |
| `self.update_tp(position_id, tp_price)` | Move take-profit |
| `self.close_positions(position_ids=None)` | Close all or specific positions |
| `self.pnl(position_id=None)` | Unrealized PnL in account currency |
| `self.pnl_pct(position_id=None)` | Unrealized PnL as % of cost basis |
| `self.pending_signals` | Queued LIMIT/STOP signals |
| `self.cancel_signal(signal_id)` | Cancel one pending signal |
| `self.cancel_pending_signals()` | Cancel all pending signals |
| `self.write_state(key, value)` | Write to shared state |
| `self.read_state(strategy, key)` | Read from shared state |

### `BrokerView` methods (via `self.broker`)

| Method | Description |
|---|---|
| `pnl(position_id=None)` | Unrealized PnL in account currency |
| `pnl_pct(position_id=None)` | Unrealized PnL as % of cost basis |
| `value_per_point(symbol)` | Account-currency value of 1-point move |
| `close_positions(position_ids=None)` | Close all or specific positions |
| `pending_signals` | List of queued LIMIT/STOP signals |
| `cancel_signal(signal_id)` | Cancel one pending signal |
| `cancel_pending_signals()` | Cancel all pending signals |
| `max_affordable_qty(symbol, price)` | Max qty the current margin can support |

> **Note:** `pnl()` and `pnl_pct()` are also available directly on the strategy as `self.pnl()` / `self.pnl_pct()`, consistent with `self.equity`, `self.positions`, `self.update_sl()`, and `self.update_tp()`.

### Overridable callbacks

| Method | Fires when |
|---|---|
| `next()` | Every completed bar (all indicators finite) |
| `update_bars()` | After OHLC appended, before indicators |
| `on_signal_execution_success(signal, position)` | Order fills |
| `on_signal_execution_error(signal, reason)` | Order fails |
| `on_signal_queued(pending)` | LIMIT/STOP accepted into queue |
| `on_sl_hit(position, exit_price)` | Stop-loss triggered |
| `on_tp_hit(position, exit_price)` | Take-profit triggered |
| `on_position_close_success(position)` | Any successful close |
| `on_position_close_error(position, reason)` | Close attempt fails |

### Available indicators

| Class | Constructor | Source |
|---|---|---|
| `SMA(period)` | `SMA(20)` | single field |
| `EMA(period)` | `EMA(20)` | single field |
| `RSI(period)` | `RSI(14)` | single field |
| `ATR(period)` | `ATR(14)` | `('high', 'low', 'close')` |
| `BBANDS(period, mult)` | `BBANDS(20, 2.0)` | single field |
| `ADX(di_length, adx_smoothing)` | `ADX(14, 14)` | `('high', 'low', 'close')` |
| `COTIndex(period)` | `COTIndex(26)` | custom field via `update_bars` |
