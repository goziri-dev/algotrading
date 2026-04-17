from datetime import datetime, timezone
from collections.abc import Sequence
from typing import Any, Callable

from tqdm.auto import tqdm

from .strategy import Strategy


def mt5_timeframe_duration(timeframe: Any) -> int:
    """Convert an MT5 timeframe identifier to duration in seconds.

    MT5 minute constants use their minute count as value (e.g. ``M15=15``),
    while hourly and above are enum values (e.g. ``H4=16388``), not minutes.
    """
    tf = timeframe
    if isinstance(tf, str):
        key = tf.strip().upper()
        string_map = {
            "M1": 60,
            "M2": 120,
            "M3": 180,
            "M4": 240,
            "M5": 300,
            "M6": 360,
            "M10": 600,
            "M12": 720,
            "M15": 900,
            "M20": 1200,
            "M30": 1800,
            "H1": 3600,
            "H2": 7200,
            "H3": 10800,
            "H4": 14400,
            "H6": 21600,
            "H8": 28800,
            "H12": 43200,
            "D1": 86400,
            "W1": 604800,
            "MN1": 2592000,
        }
        if key in string_map:
            return string_map[key]

    tf_int = int(tf)

    # Minute-based constants (M1..M30 family)
    if 1 <= tf_int <= 30:
        return tf_int * 60

    # MT5 enum constants for hourly+ frames.
    enum_map = {
        16385: 3600,     # H1
        16386: 7200,     # H2
        16387: 10800,    # H3
        16388: 14400,    # H4
        16390: 21600,    # H6
        16392: 28800,    # H8
        16396: 43200,    # H12
        16408: 86400,    # D1
        32769: 604800,   # W1
        49153: 2592000,  # MN1 (fixed 30d approximation)
    }
    if tf_int in enum_map:
        return enum_map[tf_int]

    # Backward-compatible fallback for custom numeric conventions.
    return tf_int * 60


def feed_bars(
    symbol: str,
    strategies: Sequence[Strategy],
    primary_rates: Any,
    secondary_rates: dict[Any, Any] | None = None,
    timeframe_duration: Callable[[Any], int] = mt5_timeframe_duration,
    primary_duration: int | None = None,
    show_progress: bool = False,
) -> int | None:
    """Feed primary + look-ahead-safe secondary bars into ``strategies``.

    For each primary bar, secondary bars whose close time falls at or before the
    primary bar's decision time are fed first. The decision time is modeled as
    the primary bar close (open + primary duration), enforcing the invariant
    that a higher-timeframe bar is only visible after it has fully closed.

    Args:
        symbol:             The trading symbol (used for ``broker.on_bar``).
        strategies:         Strategies to receive the bars (must all trade ``symbol``).
        primary_rates:      Iterable of bar dicts with ``time`` (unix seconds),
                            ``open``, ``high``, ``low``, ``close``.
        secondary_rates:    Optional mapping of timeframe → bar iterable.
        timeframe_duration: Callable that converts a timeframe key to its bar
                            duration in seconds.  Defaults to MT5 minute constants.
        primary_duration:   Optional primary bar duration in seconds. If omitted,
                    inferred from primary timestamps.
        show_progress:      When true, render a progress bar while feeding bars.

    Returns:
        Unix timestamp of the last primary bar fed, or ``None`` if empty.
    """
    primary_list = list(primary_rates)
    primary_duration_sec = _resolve_primary_duration(primary_list, strategies, primary_duration)

    sec_iters: dict[Any, Any] = {}
    sec_pending: dict[Any, Any] = {}
    if secondary_rates:
        for tf, rates in secondary_rates.items():
            it = iter(rates)
            sec_iters[tf] = it
            sec_pending[tf] = next(it, None)

    last_bar_time: int | None = None
    bar_iter = tqdm(
        primary_list,
        total=len(primary_list),
        desc=f"Backtest {symbol}",
        unit="bar",
        disable=not show_progress,
    )
    for bar in bar_iter:
        primary_open = bar["time"]
        primary_decision_time = primary_open + primary_duration_sec
        spread_pct = _bar_spread_pct(bar)

        # Drain secondary bars whose close_time ≤ this primary bar's close time.
        for tf, it in sec_iters.items():
            tf_duration = timeframe_duration(tf)
            while sec_pending[tf] is not None:
                sec_bar = sec_pending[tf]
                if sec_bar["time"] + tf_duration <= primary_decision_time:
                    _feed_secondary_bar(tf, sec_bar, strategies)
                    sec_pending[tf] = next(it, None)
                else:
                    break

        bar_time = datetime.fromtimestamp(primary_open, tz=timezone.utc)
        for s in strategies:
            s.broker.on_bar(
                symbol=symbol,
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                time=bar_time,
                spread_pct=spread_pct,
            )
            s.on_bar(
                time=bar_time,
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
            )
        last_bar_time = primary_open

    return last_bar_time


def feed_bars_aligned(
    symbols: list[str],
    by_symbol: dict[str, list[Strategy]],
    primary_rates: dict[str, Any],
    secondary_rates: dict[str, dict[Any, Any]],
    primary_timeframes: dict[str, Any],
    timeframe_duration: Callable[[Any], int] = mt5_timeframe_duration,
    show_progress: bool = False,
) -> None:
    """Feed primary bars from multiple symbols interleaved in timestamp order.

    Strategies that share state across symbols (via ``write_state`` /
    ``read_state``) can only read the previous bar's value from another symbol
    — there is no look-ahead bias.  Ties at the same timestamp are broken by
    the order of ``symbols``.

    Args:
        symbols:            Symbols to process, in priority order for tie-breaking.
        by_symbol:          Mapping of symbol → list of strategies for that symbol.
        primary_rates:      Mapping of symbol → bar iterable.
        secondary_rates:    Mapping of symbol → {timeframe → bar iterable}.
        primary_timeframes: Mapping of symbol → primary timeframe key.
        timeframe_duration: Callable converting a timeframe key to seconds.
        show_progress:      When true, render a single portfolio-level progress bar.
    """
    # Build per-symbol state: secondary iterators, pending secondary bar, primary duration.
    sym_states: dict[str, dict[str, Any]] = {}
    for symbol in symbols:
        p_rates = list(primary_rates.get(symbol) or [])
        if not p_rates:
            continue
        strategies = by_symbol.get(symbol, [])
        explicit_duration = timeframe_duration(primary_timeframes[symbol]) if symbol in primary_timeframes else None
        primary_duration_sec = _resolve_primary_duration(p_rates, strategies, explicit_duration)

        sec_iters: dict[Any, Any] = {}
        sec_pending: dict[Any, Any] = {}
        for tf, rates in (secondary_rates.get(symbol) or {}).items():
            it = iter(rates)
            sec_iters[tf] = it
            sec_pending[tf] = next(it, None)

        sym_states[symbol] = {
            "strategies": strategies,
            "primary_duration_sec": primary_duration_sec,
            "sec_iters": sec_iters,
            "sec_pending": sec_pending,
        }

    if not sym_states:
        return

    # Merge all primary bars tagged with symbol. Stable sort by (time, symbol_index)
    # so ties resolve in the caller-specified symbol order.
    symbol_index = {sym: i for i, sym in enumerate(symbols)}
    all_bars: list[tuple[int, int, str, Any]] = [
        (int(bar["time"]), symbol_index.get(symbol, 0), symbol, bar)
        for symbol in sym_states
        for bar in (primary_rates.get(symbol) or [])
    ]
    all_bars.sort(key=lambda x: (x[0], x[1]))

    bar_iter = tqdm(
        all_bars,
        total=len(all_bars),
        desc="Backtest",
        unit="bar",
        disable=not show_progress,
    )
    for timestamp, _, symbol, bar in bar_iter:
        state = sym_states[symbol]
        strategies = state["strategies"]
        primary_decision_time = timestamp + state["primary_duration_sec"]
        spread_pct = _bar_spread_pct(bar)

        # Drain secondary bars whose close falls at or before this primary bar's close.
        for tf, it in state["sec_iters"].items():
            tf_duration = timeframe_duration(tf)
            while state["sec_pending"].get(tf) is not None:
                sec_bar = state["sec_pending"][tf]
                if sec_bar["time"] + tf_duration <= primary_decision_time:
                    _feed_secondary_bar(tf, sec_bar, strategies)
                    state["sec_pending"][tf] = next(it, None)
                else:
                    break

        bar_time = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        for s in strategies:
            s.broker.on_bar(
                symbol=symbol,
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
                time=bar_time,
                spread_pct=spread_pct,
            )
            s.on_bar(
                time=bar_time,
                open=bar["open"],
                high=bar["high"],
                low=bar["low"],
                close=bar["close"],
            )


def _resolve_primary_duration(
    primary_rates: list[Any],
    strategies: Sequence[Strategy],
    explicit_duration: int | None,
) -> int:
    if explicit_duration is not None and explicit_duration > 0:
        return explicit_duration

    deltas: list[int] = []
    if len(primary_rates) >= 2:
        deltas.extend(
            int(primary_rates[i]["time"] - primary_rates[i - 1]["time"])
            for i in range(1, len(primary_rates))
            if int(primary_rates[i]["time"] - primary_rates[i - 1]["time"]) > 0
        )

    if primary_rates:
        first_time = int(primary_rates[0]["time"])
        previous_primary_times = [
            int(s.bars.time[-1].astype("datetime64[s]").astype("int64"))
            for s in strategies
            if len(s.bars) > 0
        ]
        if previous_primary_times:
            delta = first_time - max(previous_primary_times)
            if delta > 0:
                deltas.append(delta)

    return min(deltas) if deltas else 0


def _feed_secondary_bar(timeframe: Any, bar: Any, strategies: Sequence[Strategy]) -> None:
    bar_time = datetime.fromtimestamp(bar["time"], tz=timezone.utc)
    for s in strategies:
        bars_obj = s.secondary_bars.get(timeframe)
        if bars_obj is None:
            continue
        bars_obj.update(
            time=bar_time,
            open=bar["open"],
            high=bar["high"],
            low=bar["low"],
            close=bar["close"],
        )


def _bar_spread_pct(bar: Any) -> float | None:
    """Extract per-bar spread percentage when available."""
    try:
        raw = bar["spread_pct"]
    except Exception:
        return None
    if raw is None:
        return None
    value = float(raw)
    if value < 0:
        return None
    return value
