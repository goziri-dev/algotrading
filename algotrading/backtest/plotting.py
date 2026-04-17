from __future__ import annotations

import calendar
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence
import webbrowser

import numpy as np
import pandas as pd

from .backtest_broker import BacktestBroker, ClosedTrade
from .monte_carlo import MonteCarloReport


def _require_matplotlib():
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for plotting. Install it with 'uv add matplotlib'."
        ) from exc
    return plt, mdates


def _require_plotly():
    try:
        import plotly.graph_objects as go
    except ImportError as exc:
        raise ImportError(
            "plotly is required for interactive plotting. Install it with 'uv add plotly'."
        ) from exc
    return go


def _require_lightweight_charts_static():
    try:
        from lightweight_charts.widgets import StaticLWC
    except ImportError as exc:
        raise ImportError(
            "lightweight-charts is required for interactive trade plotting. Install it with 'uv add lightweight-charts'."
        ) from exc
    return StaticLWC


class LightweightChartFigure:
    def __init__(self, html: str, *, metadata: dict[str, Any] | None = None) -> None:
        self.html = html
        self.metadata: dict[str, Any] = metadata or {}

    def write_html(
        self,
        file: str | Path,
        *,
        full_html: bool = True,
        auto_open: bool = False,
        **_: Any,
    ) -> None:
        target = Path(file)
        target.parent.mkdir(parents=True, exist_ok=True)
        if full_html:
            content = self.html
        else:
            lower = self.html.lower()
            start = lower.find("<body")
            if start >= 0:
                body_start = self.html.find(">", start)
                end = lower.rfind("</body>")
                if body_start >= 0 and end > body_start:
                    content = self.html[body_start + 1:end]
                else:
                    content = self.html
            else:
                content = self.html
        target.write_text(content, encoding="utf-8")
        if auto_open:
            webbrowser.open(target.resolve().as_uri())


def _to_python_datetimes(times: Sequence[datetime] | np.ndarray) -> list[datetime]:
    if isinstance(times, np.ndarray):
        if np.issubdtype(times.dtype, np.datetime64):
            # Convert numpy datetime64[ns] into timezone-naive Python datetimes.
            return [
                datetime.utcfromtimestamp(float(value.astype("datetime64[ns]").astype(np.int64)) / 1e9)
                for value in times
            ]
        return [value for value in times.tolist()]
    return [value for value in times]


def _equity_series(broker: BacktestBroker) -> tuple[list[datetime], list[float]]:
    if not broker.equity_curve:
        return [], []
    return (
        [point.time for point in broker.equity_curve],
        [point.equity for point in broker.equity_curve],
    )


def _drawdown_from_equity(equity: Sequence[float]) -> np.ndarray:
    if not equity:
        return np.array([], dtype=np.float64)
    curve = np.asarray(equity, dtype=np.float64)
    running_peak = np.maximum.accumulate(curve)
    safe_peak = np.where(running_peak <= 0.0, np.nan, running_peak)
    return (curve / safe_peak - 1.0) * 100.0


def _monthly_return_grids(broker: BacktestBroker) -> tuple[list[int], np.ndarray, np.ndarray]:
    times, equity = _equity_series(broker)
    if not times:
        raise ValueError("Broker equity curve is empty.")

    month_keys: list[tuple[int, int]] = []
    month_end_equity: list[float] = []
    for time, value in zip(times, equity):
        key = (time.year, time.month)
        if month_keys and key == month_keys[-1]:
            month_end_equity[-1] = float(value)
            continue
        month_keys.append(key)
        month_end_equity.append(float(value))

    years = list(dict.fromkeys(year for year, _ in month_keys))
    value_grid = np.full((len(years), 12), np.nan, dtype=np.float64)
    pct_grid = np.full((len(years), 12), np.nan, dtype=np.float64)
    year_to_row = {year: i for i, year in enumerate(years)}

    prev_equity = float(broker._initial_balance)
    for (year, month), end_equity in zip(month_keys, month_end_equity):
        monthly_value = end_equity - prev_equity
        monthly_pct = (monthly_value / prev_equity * 100.0) if prev_equity > 0.0 else np.nan
        row = year_to_row[year]
        value_grid[row, month - 1] = monthly_value
        pct_grid[row, month - 1] = monthly_pct
        prev_equity = end_equity

    return years, value_grid, pct_grid


def _nearest_prices_for_times(
    target_times: Sequence[datetime],
    price_times: Sequence[datetime],
    prices: Sequence[float] | np.ndarray,
) -> list[float]:
    if not target_times or not price_times:
        return []

    def to_ns(values: Sequence[datetime]) -> np.ndarray:
        ns_values: list[int] = []
        for value in values:
            if value.tzinfo is None:
                value_utc = value.replace(tzinfo=timezone.utc)
            else:
                value_utc = value.astimezone(timezone.utc)
            ns_values.append(int(value_utc.timestamp() * 1_000_000_000))
        return np.asarray(ns_values, dtype=np.int64)

    target_num = to_ns(target_times)
    price_num = to_ns(price_times)
    price_values = np.asarray(prices, dtype=np.float64)

    idx = np.searchsorted(price_num, target_num)
    idx = np.clip(idx, 0, len(price_values) - 1)

    for i in range(len(idx)):
        right = idx[i]
        if right == 0:
            continue
        if right >= len(price_num):
            idx[i] = len(price_num) - 1
            continue
        left = right - 1
        if abs(target_num[i] - price_num[left]) <= abs(price_num[right] - target_num[i]):
            idx[i] = left

    return price_values[idx].tolist()


def _downsample_indices(indices: list[int], max_points: int) -> list[int]:
    if max_points <= 0 or len(indices) <= max_points:
        return indices
    sampled = np.linspace(0, len(indices) - 1, num=max_points, dtype=int)
    return [indices[i] for i in sampled.tolist()]


def _downsample_xy(
    times: Sequence[datetime],
    values: Sequence[float] | np.ndarray,
    max_points: int,
) -> tuple[list[datetime], np.ndarray]:
    arr = np.asarray(values, dtype=np.float64)
    if max_points <= 0 or len(arr) <= max_points:
        return list(times), arr
    sample_idx = np.linspace(0, len(arr) - 1, num=max_points, dtype=int)
    return [times[i] for i in sample_idx.tolist()], arr[sample_idx]


def _to_ns_array(times: Sequence[datetime]) -> np.ndarray:
    ns_values: list[int] = []
    for value in times:
        if value.tzinfo is None:
            value_utc = value.replace(tzinfo=timezone.utc)
        else:
            value_utc = value.astimezone(timezone.utc)
        ns_values.append(int(value_utc.timestamp() * 1_000_000_000))
    return np.asarray(ns_values, dtype=np.int64)


def _infer_rangebreaks(times: Sequence[datetime]) -> list[dict[str, object]]:
    if len(times) < 4:
        return []

    ns = _to_ns_array(times)
    deltas = np.diff(ns)
    deltas = deltas[deltas > 0]
    if deltas.size == 0:
        return []

    bar_step_ns = int(np.median(deltas))
    if bar_step_ns <= 0:
        return []

    expected_count = int((ns[-1] - ns[0]) // bar_step_ns) + 1
    # Protect against pathological spans that could generate huge temporary arrays.
    if expected_count <= 0 or expected_count > 50_000:
        return []

    expected_ns = ns[0] + np.arange(expected_count, dtype=np.int64) * bar_step_ns
    observed_set = set(int(v) for v in ns.tolist())
    missing_ns = [int(v) for v in expected_ns.tolist() if int(v) not in observed_set]
    if not missing_ns:
        return []

    missing_times = [datetime.fromtimestamp(v / 1e9, tz=timezone.utc) for v in missing_ns]
    bar_step_ms = max(1, bar_step_ns // 1_000_000)
    return [{"values": missing_times, "dvalue": bar_step_ms}]


def _nearest_indices_for_times(
    target_times: Sequence[datetime],
    reference_times: Sequence[datetime],
) -> np.ndarray:
    if not target_times or not reference_times:
        return np.array([], dtype=np.int64)

    target_num = _to_ns_array(target_times)
    reference_num = _to_ns_array(reference_times)

    idx = np.searchsorted(reference_num, target_num)
    idx = np.clip(idx, 0, len(reference_num) - 1)

    for i in range(len(idx)):
        right = idx[i]
        if right == 0:
            continue
        if right >= len(reference_num):
            idx[i] = len(reference_num) - 1
            continue
        left = right - 1
        if abs(target_num[i] - reference_num[left]) <= abs(reference_num[right] - target_num[i]):
            idx[i] = left
    return idx.astype(np.int64)


def _stack_same_timestamp_markers(
    marker_times: Sequence[datetime],
    marker_values: Sequence[float] | np.ndarray,
    *,
    step: float,
    direction: float,
) -> list[float]:
    if not marker_times:
        return []
    counts: dict[int, int] = {}
    values = np.asarray(marker_values, dtype=np.float64)
    ts_ns = _to_ns_array(marker_times)
    stacked: list[float] = []
    for i, t_ns in enumerate(ts_ns.tolist()):
        lane = counts.get(t_ns, 0)
        counts[t_ns] = lane + 1
        stacked.append(float(values[i] + direction * lane * step))
    return stacked


def _marker_lane_step(
    prices: Sequence[float] | np.ndarray,
    highs: Sequence[float] | np.ndarray | None,
    lows: Sequence[float] | np.ndarray | None,
) -> float:
    prices_arr = np.asarray(prices, dtype=np.float64)
    if highs is not None and lows is not None and len(highs) == len(lows) and len(highs) > 0:
        high_arr = np.asarray(highs, dtype=np.float64)
        low_arr = np.asarray(lows, dtype=np.float64)
        candle_span = high_arr - low_arr
        median_span = float(np.nanmedian(candle_span)) if candle_span.size else 0.0
        if np.isfinite(median_span) and median_span > 0:
            return median_span * 0.22

    span = float(np.nanmax(prices_arr) - np.nanmin(prices_arr)) if prices_arr.size else 0.0
    return max(span * 0.004, 1e-6)


def _trade_hover_label(trade: ClosedTrade, *, event: str) -> str:
    return (
        f"Trade #{trade.position.id}"
        f"<br>{event}: {trade.position.direction.value}"
        f"<br>Entry: {trade.position.entry_time:%Y-%m-%d %H:%M} @ {float(trade.position.entry_price):,.5f}"
        f"<br>Exit: {trade.exit_time:%Y-%m-%d %H:%M} @ {float(trade.exit_price):,.5f}"
        f"<br>PnL: {float(trade.pnl):,.2f}"
    )


def plot_equity_curve(
    broker: BacktestBroker,
    *,
    title: str = "Equity Curve",
    ax=None,
):
    plt, mdates = _require_matplotlib()
    times, equity = _equity_series(broker)
    if not times:
        raise ValueError("Broker equity curve is empty.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure

    ax.plot(times, equity, color="#1f77b4", linewidth=1.8, label="Equity")
    ax.axhline(broker._initial_balance, color="#8c8c8c", linestyle="--", linewidth=1.0, label="Initial")
    ax.set_title(title)
    ax.set_ylabel("Account Value")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    return fig, ax


def plot_drawdown(
    broker: BacktestBroker,
    *,
    title: str = "Drawdown",
    ax=None,
):
    plt, mdates = _require_matplotlib()
    times, equity = _equity_series(broker)
    if not times:
        raise ValueError("Broker equity curve is empty.")

    drawdown_pct = _drawdown_from_equity(equity)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 3))
    else:
        fig = ax.figure

    ax.fill_between(times, drawdown_pct, 0.0, color="#d62728", alpha=0.25, label="Drawdown %")
    ax.plot(times, drawdown_pct, color="#d62728", linewidth=1.2)
    ax.set_title(title)
    ax.set_ylabel("Drawdown %")
    ax.set_ylim(min(drawdown_pct.min(), -1.0) * 1.05, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower left")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    return fig, ax


def plot_trade_pnl_distribution(
    broker: BacktestBroker,
    *,
    title: str = "Trade PnL Distribution",
    bins: int | Sequence[float] = 25,
    ax=None,
):
    plt, _ = _require_matplotlib()

    if not broker.trade_log:
        raise ValueError("Broker trade log is empty.")

    pnl = np.asarray([float(trade.pnl) for trade in broker.trade_log], dtype=np.float64)
    pnl_pct = np.asarray(
        [
            (float(trade.pnl) / cost_basis * 100.0) if cost_basis else 0.0
            for trade in broker.trade_log
            for cost_basis in [
                float(trade.position.entry_price)
                * float(trade.position.qty)
                * float(broker.value_per_point(trade.position.symbol))
            ]
        ],
        dtype=np.float64,
    )
    if pnl.size == 0 or pnl_pct.size == 0:
        raise ValueError("Broker trade log is empty.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.figure

    ax.hist(
        pnl_pct,
        bins=bins,
        color="#4e79a7",
        edgecolor="#f6f6f6",
        linewidth=0.7,
        alpha=0.9,
    )
    ax.axvline(0.0, color="#555555", linestyle="--", linewidth=1.0, label="Break-even")

    mean_pnl_pct = float(np.mean(pnl_pct))
    median_pnl_pct = float(np.median(pnl_pct))
    mean_pnl = float(np.mean(pnl))
    median_pnl = float(np.median(pnl))
    ax.axvline(
        mean_pnl_pct,
        color="#d62728",
        linestyle="-",
        linewidth=1.1,
        label=f"Mean {mean_pnl_pct:+.2f}% ({mean_pnl:+.2f})",
    )
    ax.axvline(
        median_pnl_pct,
        color="#2ca02c",
        linestyle=":",
        linewidth=1.4,
        label=f"Median {median_pnl_pct:+.2f}% ({median_pnl:+.2f})",
    )

    winners = int(np.count_nonzero(pnl_pct >= 0.0))
    losers = int(np.count_nonzero(pnl_pct < 0.0))
    ax.set_title(title)
    ax.set_xlabel("Trade PnL (%)")
    ax.set_ylabel("Number of trades")
    ax.grid(alpha=0.22, axis="y")
    ax.legend(loc="best")
    ax.text(
        0.99,
        0.99,
        f"n={pnl.size} | wins={winners} | losses={losers}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#555555",
    )
    fig.tight_layout()
    return fig, ax


def plot_equity_and_drawdown(
    broker: BacktestBroker,
    *,
    title: str = "Equity and Drawdown",
):
    plt, _ = _require_matplotlib()
    fig, (ax1, ax2) = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(12, 7),
        sharex=True,
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    plot_equity_curve(broker, title=title, ax=ax1)
    plot_drawdown(broker, title="Drawdown", ax=ax2)
    fig.tight_layout()
    return fig, (ax1, ax2)


def plot_equity_vs_benchmark(
    broker: BacktestBroker,
    benchmark_times: Sequence[datetime] | np.ndarray,
    benchmark_prices: Sequence[float] | np.ndarray,
    *,
    benchmark_name: str = "SPX",
    title: str | None = None,
    normalize_to: float = 100.0,
    ax=None,
):
    plt, mdates = _require_matplotlib()
    eq_times, equity = _equity_series(broker)
    if not eq_times:
        raise ValueError("Broker equity curve is empty.")

    b_times = _to_python_datetimes(benchmark_times)
    b_prices = np.asarray(benchmark_prices, dtype=np.float64)
    if len(b_times) == 0 or len(b_prices) == 0:
        raise ValueError("Benchmark series is empty.")

    eq_base = equity[0]
    bench_base = float(b_prices[0])
    if eq_base <= 0 or bench_base <= 0:
        raise ValueError("Normalization requires positive starting values.")

    eq_norm = np.asarray(equity, dtype=np.float64) / eq_base * normalize_to
    bench_norm = b_prices / bench_base * normalize_to

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure

    ax.plot(eq_times, eq_norm, color="#1f77b4", linewidth=1.8, label="Strategy")
    ax.plot(b_times, bench_norm, color="#2ca02c", linewidth=1.4, alpha=0.9, label=benchmark_name)
    ax.set_title(title or f"Normalized Equity vs {benchmark_name}")
    ax.set_ylabel(f"Normalized ({normalize_to:.0f}=start)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    return fig, ax


def plot_equity_vs_symbols(
    broker: BacktestBroker,
    symbol_series: dict[str, tuple[Sequence[datetime] | np.ndarray, Sequence[float] | np.ndarray]],
    *,
    title: str = "Normalized Equity vs Symbols",
    normalize_to: float = 100.0,
    ax=None,
):
    plt, mdates = _require_matplotlib()
    eq_times, equity = _equity_series(broker)
    if not eq_times:
        raise ValueError("Broker equity curve is empty.")
    if not symbol_series:
        raise ValueError("At least one symbol series is required.")

    eq_base = float(equity[0])
    if eq_base <= 0.0:
        raise ValueError("Normalization requires positive equity start.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure

    eq_norm = np.asarray(equity, dtype=np.float64) / eq_base * normalize_to
    ax.plot(eq_times, eq_norm, color="#1f77b4", linewidth=2.0, label="Strategy Equity")

    palette = [
        "#2ca02c",
        "#ff7f0e",
        "#9467bd",
        "#17becf",
        "#bcbd22",
        "#8c564b",
    ]
    plotted = 0
    for i, symbol in enumerate(sorted(symbol_series)):
        times_raw, prices_raw = symbol_series[symbol]
        times = _to_python_datetimes(times_raw)
        prices = np.asarray(prices_raw, dtype=np.float64)
        if len(times) == 0 or prices.size == 0 or len(times) != prices.size:
            continue
        base = float(prices[0])
        if base <= 0.0:
            continue
        norm = prices / base * normalize_to
        ax.plot(
            times,
            norm,
            color=palette[i % len(palette)],
            linewidth=1.3,
            alpha=0.95,
            label=f"{symbol} B&H",
        )
        plotted += 1

    if plotted == 0:
        raise ValueError("No valid symbol price series were available for plotting.")

    ax.set_title(title)
    ax.set_ylabel(f"Normalized ({normalize_to:.0f}=start)")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    return fig, ax


def plot_symbol_return_correlation(
    symbol_series: dict[str, tuple[Sequence[datetime] | np.ndarray, Sequence[float] | np.ndarray]],
    *,
    title: str = "Symbol Return Correlation",
    ax=None,
):
    plt, _ = _require_matplotlib()
    if len(symbol_series) < 2:
        raise ValueError("At least two symbol series are required for correlation.")

    returns_by_symbol: dict[str, pd.Series] = {}
    for symbol, (times_raw, prices_raw) in symbol_series.items():
        times = _to_python_datetimes(times_raw)
        prices = np.asarray(prices_raw, dtype=np.float64)
        if len(times) < 3 or prices.size < 3 or len(times) != prices.size:
            continue
        series = pd.Series(
            prices,
            index=pd.to_datetime(times, utc=True),
            dtype=np.float64,
        ).sort_index()
        returns = series.pct_change().replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) >= 2:
            returns_by_symbol[symbol] = returns

    if len(returns_by_symbol) < 2:
        raise ValueError("Not enough valid symbol return series for correlation.")

    returns_df = pd.concat(returns_by_symbol, axis=1, join="inner").dropna(how="any")
    if returns_df.shape[1] < 2 or returns_df.shape[0] < 2:
        raise ValueError("Insufficient overlapping return observations for correlation.")

    corr = returns_df.corr().to_numpy(dtype=np.float64)
    labels = list(returns_df.columns)

    if ax is None:
        fig_size = max(4.8, 1.5 + len(labels) * 0.85)
        fig, ax = plt.subplots(figsize=(fig_size, fig_size))
    else:
        fig = ax.figure

    image = ax.imshow(corr, cmap="RdYlGn", vmin=-1.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticklabels(labels)

    for row in range(len(labels)):
        for col in range(len(labels)):
            value = float(corr[row, col])
            text_color = "#ffffff" if abs(value) >= 0.55 else "#222222"
            ax.text(col, row, f"{value:+.2f}", ha="center", va="center", fontsize=9, color=text_color)

    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="Correlation")
    fig.tight_layout()
    return fig, ax


def plot_monthly_returns_heatmap(
    broker: BacktestBroker,
    *,
    title: str = "Monthly Returns Heatmap",
    ax=None,
):
    plt, _ = _require_matplotlib()
    years, value_grid, pct_grid = _monthly_return_grids(broker)

    if ax is None:
        fig_height = max(3.2, 1.9 + 0.68 * len(years))
        fig, ax = plt.subplots(figsize=(13, fig_height))
    else:
        fig = ax.figure

    masked_pct = np.ma.masked_invalid(pct_grid)
    finite_pct = pct_grid[np.isfinite(pct_grid)]
    max_abs_pct = float(np.max(np.abs(finite_pct))) if finite_pct.size else 0.0
    color_limit = max(max_abs_pct, 1.0)
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad(color="#f2f2f2")

    image = ax.imshow(
        masked_pct,
        aspect="auto",
        cmap=cmap,
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="nearest",
    )

    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Year")
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(list(calendar.month_abbr[1:]))
    ax.set_yticks(np.arange(len(years)))
    ax.set_yticklabels([str(year) for year in years])

    ax.set_xticks(np.arange(-0.5, 12, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(years), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)

    for row in range(len(years)):
        for col in range(12):
            pct_value = pct_grid[row, col]
            value_value = value_grid[row, col]
            if not np.isfinite(pct_value) or not np.isfinite(value_value):
                continue
            text_color = "white" if abs(pct_value) >= color_limit * 0.55 else "#222222"
            ax.text(
                col,
                row,
                f"{value_value:+,.2f}\n{pct_value:+.1f}%",
                ha="center",
                va="center",
                fontsize=8.5,
                color=text_color,
            )

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Monthly return %")
    fig.tight_layout()
    return fig, ax


def plot_monte_carlo_paths(
    report: MonteCarloReport,
    *,
    title: str = "Monte Carlo Equity Paths",
    max_paths: int = 120,
    band_quantiles: tuple[float, float] = (0.05, 0.95),
    median_quantile: float = 0.5,
    ax=None,
):
    plt, _ = _require_matplotlib()

    if report.paths.ndim != 2 or report.paths.shape[1] < 2:
        raise ValueError("Monte Carlo report has no path data to plot")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4.5))
    else:
        fig = ax.figure

    n_paths = report.paths.shape[0]
    max_show = max(1, min(max_paths, n_paths))
    sample_idx = np.linspace(0, n_paths - 1, num=max_show, dtype=int)
    x = np.arange(report.paths.shape[1], dtype=np.int64)

    for i in sample_idx.tolist():
        ax.plot(x, report.paths[i], color="#9aa0a6", alpha=0.16, linewidth=0.8)

    q_low, q_high = band_quantiles
    low_path = report.quantile_path(q_low)
    high_path = report.quantile_path(q_high)
    median_path = report.quantile_path(median_quantile)

    ax.fill_between(x, low_path, high_path, color="#1f77b4", alpha=0.22, label=f"{q_low:.0%}-{q_high:.0%} band")
    ax.plot(x, median_path, color="#1f77b4", linewidth=2.0, label=f"Median ({median_quantile:.0%})")
    ax.axhline(report.initial_balance, color="#555555", linestyle="--", linewidth=1.0, label="Initial balance")

    ax.set_title(title)
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Equity")
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    return fig, ax


def plot_price_with_trades(
    price_times: Sequence[datetime] | np.ndarray,
    prices: Sequence[float] | np.ndarray,
    trades: Sequence[ClosedTrade],
    *,
    symbol: str | None = None,
    title: str | None = None,
    max_markers_per_group: int = 90,
    use_trade_fill_prices: bool = True,
    ax=None,
):
    plt, mdates = _require_matplotlib()
    times = _to_python_datetimes(price_times)
    price_values = np.asarray(prices, dtype=np.float64)

    if len(times) == 0 or len(price_values) == 0:
        raise ValueError("Price series is empty.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    else:
        fig = ax.figure

    ax.plot(times, price_values, color="#222222", linewidth=1.35, alpha=0.9, label=(symbol or "Price"))

    if trades:
        entry_times = [trade.position.entry_time for trade in trades]
        exit_times = [trade.exit_time for trade in trades]
        if use_trade_fill_prices:
            entry_prices = [float(trade.position.entry_price) for trade in trades]
            exit_prices = [float(trade.exit_price) for trade in trades]
        else:
            entry_prices = _nearest_prices_for_times(entry_times, times, price_values)
            exit_prices = _nearest_prices_for_times(exit_times, times, price_values)

        long_idx = [i for i, trade in enumerate(trades) if trade.position.is_long]
        short_idx = [i for i, trade in enumerate(trades) if trade.position.is_short]
        win_idx = [i for i, trade in enumerate(trades) if trade.pnl >= 0]
        loss_idx = [i for i, trade in enumerate(trades) if trade.pnl < 0]

        long_idx = _downsample_indices(long_idx, max_markers_per_group)
        short_idx = _downsample_indices(short_idx, max_markers_per_group)
        win_idx = _downsample_indices(win_idx, max_markers_per_group)
        loss_idx = _downsample_indices(loss_idx, max_markers_per_group)

        if long_idx:
            ax.scatter(
                [entry_times[i] for i in long_idx],
                [entry_prices[i] for i in long_idx],
                marker="^",
                s=30,
                facecolors="none",
                edgecolors="#2ca02c",
                linewidths=1.25,
                alpha=0.85,
                label="Long entry",
                zorder=3,
            )
        if short_idx:
            ax.scatter(
                [entry_times[i] for i in short_idx],
                [entry_prices[i] for i in short_idx],
                marker="v",
                s=30,
                facecolors="none",
                edgecolors="#d62728",
                linewidths=1.25,
                alpha=0.85,
                label="Short entry",
                zorder=3,
            )
        if win_idx:
            ax.scatter(
                [exit_times[i] for i in win_idx],
                [exit_prices[i] for i in win_idx],
                marker="o",
                s=24,
                color="#1f77b4",
                alpha=0.65,
                label="Winning exit",
                zorder=3,
            )
        if loss_idx:
            ax.scatter(
                [exit_times[i] for i in loss_idx],
                [exit_prices[i] for i in loss_idx],
                marker="x",
                s=30,
                color="#ff7f0e",
                alpha=0.75,
                linewidths=1.2,
                label="Losing exit",
                zorder=3,
            )

    ax.set_title(title or (f"{symbol} Price with Trade Markers" if symbol else "Price with Trade Markers"))
    ax.set_ylabel("Price")
    ax.grid(alpha=0.25)
    if len(trades) > 0:
        plotted = min(len(trades), max_markers_per_group) if max_markers_per_group > 0 else len(trades)
        ax.text(
            0.99,
            0.01,
            f"trades={len(trades)} | markers/group≤{max_markers_per_group}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color="#555555",
        )
    ax.legend(loc="best")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    fig.autofmt_xdate()
    return fig, ax


def plot_price_with_trades_interactive(
    price_times: Sequence[datetime] | np.ndarray,
    prices: Sequence[float] | np.ndarray,
    trades: Sequence[ClosedTrade],
    *,
    symbol: str | None = None,
    title: str | None = None,
    max_markers_per_group: int = 300,
    use_trade_fill_prices: bool = True,
    price_open: Sequence[float] | np.ndarray | None = None,
    price_high: Sequence[float] | np.ndarray | None = None,
    price_low: Sequence[float] | np.ndarray | None = None,
    price_close: Sequence[float] | np.ndarray | None = None,
    render_mode: str = "auto",
    max_candles: int = 1_500,
    max_line_points: int = 12_000,
    marker_layout: str = "staggered",
    indicator_overlays: Sequence[dict[str, Any]] | None = None,
    indicator_panels: Sequence[dict[str, Any]] | None = None,
):
    StaticLWC = _require_lightweight_charts_static()
    times = _to_python_datetimes(price_times)
    price_values = np.asarray(prices, dtype=np.float64)

    if len(times) == 0 or len(price_values) == 0:
        raise ValueError("Price series is empty.")

    has_ohlc = all(v is not None for v in (price_open, price_high, price_low, price_close))
    if render_mode not in {"auto", "candlestick", "line"}:
        raise ValueError("render_mode must be one of: 'auto', 'candlestick', 'line'.")
    if marker_layout not in {"staggered", "fill"}:
        raise ValueError("marker_layout must be one of: 'staggered', 'fill'.")

    use_candles = False
    if render_mode == "candlestick":
        use_candles = has_ohlc
    elif render_mode == "line":
        use_candles = False
    else:
        use_candles = has_ohlc and len(times) <= max_candles

    open_values: np.ndarray | None = None
    high_values: np.ndarray | None = None
    low_values: np.ndarray | None = None
    close_values: np.ndarray | None = None

    line_sample_idx: np.ndarray | None = None
    display_times = times

    if use_candles:
        open_values = np.asarray(price_open, dtype=np.float64)  # type: ignore[arg-type]
        high_values = np.asarray(price_high, dtype=np.float64)  # type: ignore[arg-type]
        low_values = np.asarray(price_low, dtype=np.float64)  # type: ignore[arg-type]
        close_values = np.asarray(price_close, dtype=np.float64)  # type: ignore[arg-type]
        same_length = (
            len(open_values) == len(times)
            and len(high_values) == len(times)
            and len(low_values) == len(times)
            and len(close_values) == len(times)
        )
        if not same_length:
            raise ValueError("OHLC arrays must have the same length as price_times.")
        time_values = pd.to_datetime(times, utc=True).tz_localize(None).to_numpy(dtype="datetime64[ns]")
        price_df = pd.DataFrame(
            {
                "time": time_values,
                "open": open_values,
                "high": high_values,
                "low": low_values,
                "close": close_values,
            }
        )
    else:
        if max_line_points <= 0 or len(price_values) <= max_line_points:
            line_sample_idx = np.arange(len(price_values), dtype=int)
            line_times = list(times)
            line_prices = np.asarray(price_values, dtype=np.float64)
        else:
            line_sample_idx = np.linspace(0, len(price_values) - 1, num=max_line_points, dtype=int)
            line_times = [times[i] for i in line_sample_idx.tolist()]
            line_prices = price_values[line_sample_idx]

        display_times = line_times
        line_name = symbol or "Price"
        line_time_values = pd.to_datetime(line_times, utc=True).tz_localize(None).to_numpy(dtype="datetime64[ns]")
        price_df = pd.DataFrame({"time": line_time_values, line_name: np.asarray(line_prices, dtype=np.float64)})

    panel_specs = list(indicator_panels or [])
    panel_count = len(panel_specs)
    main_height = 1.0 if panel_count == 0 else max(0.45, 1.0 - 0.22 * panel_count)
    panel_height = 0.22 if panel_count == 0 else (1.0 - main_height) / panel_count
    right_price_scale_width = 72

    chart = StaticLWC(width=1280, height=720, inner_height=main_height) # type: ignore[no-untyped-call]
    chart.layout(background_color="#ffffff", text_color="#1f2937", font_size=12, font_family="Segoe UI")
    chart.grid(vert_enabled=False, horz_enabled=False)
    chart.time_scale(right_offset=8, min_bar_spacing=0.5, time_visible=True, seconds_visible=False)
    chart.legend(visible=True, ohlc=True, percent=True, lines=True, color="#0f172a", font_size=11)
    chart.run_script(
        f"{chart.id}.chart.applyOptions({{rightPriceScale: {{minimumWidth: {right_price_scale_width}}}}});"
    )
    chart.candle_style(
        up_color="rgba(34, 197, 94, 1.0)",
        down_color="rgba(220, 38, 38, 1.0)",
        border_up_color="rgba(34, 197, 94, 1.0)",
        border_down_color="rgba(220, 38, 38, 1.0)",
        wick_up_color="rgba(34, 197, 94, 1.0)",
        wick_down_color="rgba(220, 38, 38, 1.0)",
    )
    chart.watermark(symbol or "Price", font_size=24, color="rgba(15, 23, 42, 0.22)")

    if use_candles:
        chart.set(price_df)
        marker_series = chart
    else:
        line_name = symbol or "Price"
        line = chart.create_line(name=line_name, color="rgba(31, 41, 55, 0.95)", width=2)
        line.set(price_df)
        marker_series = line

    marker_series.run_script(f"try {{ {marker_series.id}.series.applyOptions({{crosshairMarkerVisible: false}}); }} catch(e) {{ console.warn('[plot] marker_series applyOptions failed:', e); }}")

    def _values_for_display(values: Sequence[float] | np.ndarray) -> np.ndarray | None:
        arr = np.asarray(values, dtype=np.float64)
        if len(arr) == len(display_times):
            return arr
        if line_sample_idx is not None and len(arr) == len(times):
            return arr[line_sample_idx]
        if len(arr) == len(times):
            return arr
        return None

    overlay_names: list[str] = []
    for overlay in indicator_overlays or []:
        name = str(overlay.get("name", "Indicator"))
        values_obj = overlay.get("values")
        if values_obj is None:
            continue
        values = _values_for_display(values_obj)
        if values is None:
            continue
        overlay_df = pd.DataFrame(
            {
                "time": pd.to_datetime(display_times, utc=True).tz_localize(None).to_numpy(dtype="datetime64[ns]"),
                name: values,
            }
        )
        overlay_line = chart.create_line(
            name=name,
            color=str(overlay.get("color", "#4f46e5")),
            width=int(overlay.get("width", 2)),
            price_line=False,
            price_label=False,
        )
        overlay_line.set(overlay_df)
        overlay_line.run_script(f"try {{ {overlay_line.id}.series.applyOptions({{crosshairMarkerVisible: false}}); }} catch(e) {{ console.warn('[plot] overlay applyOptions failed:', e); }}")
        overlay_names.append(name)

    panel_titles: list[str] = []
    panel_chart_ids: list[str] = []
    panel_first_trace_ids: list[str] = []
    for panel in panel_specs:
        panel_title = str(panel.get("title", "Indicator"))
        traces = list(panel.get("traces", []))
        if not traces:
            continue

        subchart = chart.create_subchart(
            width=1.0,
            height=panel_height,
            sync=None,
        )
        subchart.layout(background_color="#ffffff", text_color="#1f2937", font_size=11, font_family="Segoe UI")
        subchart.grid(vert_enabled=False, horz_enabled=False)
        subchart.time_scale(right_offset=8, min_bar_spacing=0.5, time_visible=True, seconds_visible=False)
        subchart.legend(visible=True, ohlc=False, percent=False, lines=True, color="#1f2937", font_size=10)
        subchart.run_script(
            f"{subchart.id}.chart.applyOptions({{rightPriceScale: {{minimumWidth: {right_price_scale_width}}}}});"
        )
        panel_chart_ids.append(subchart.id)

        # Ensure every panel has the same full timeline domain as the main chart,
        # even when indicator traces are sparse.
        anchor_line = subchart.create_line(
            name="",
            color="rgba(0, 0, 0, 0)",
            width=1,
            price_line=False,
            price_label=False,
        )
        anchor_df = pd.DataFrame(
            {
                "time": pd.to_datetime(display_times, utc=True).tz_localize(None).to_numpy(dtype="datetime64[ns]"),
                "value": np.zeros(len(display_times), dtype=np.float64),
            }
        )
        anchor_line.set(anchor_df)
        anchor_line.run_script(
            f"try {{ {anchor_line.id}.series.applyOptions({{visible: false, lastValueVisible: false, priceLineVisible: false, crosshairMarkerVisible: false}}); }} catch(e) {{ console.warn('[plot] anchor applyOptions failed:', e); }}"
        )

        first_trace_id: str | None = None
        for trace in traces:
            trace_name = str(trace.get("name", panel_title))
            trace_values_obj = trace.get("values")
            if trace_values_obj is None:
                continue
            trace_values = _values_for_display(trace_values_obj)
            if trace_values is None:
                continue
            trace_df = pd.DataFrame(
                {
                    "time": pd.to_datetime(display_times, utc=True).tz_localize(None).to_numpy(dtype="datetime64[ns]"),
                    trace_name: trace_values,
                }
            )
            panel_line = subchart.create_line(
                name=trace_name,
                color=str(trace.get("color", "#4f46e5")),
                width=int(trace.get("width", 2)),
                price_line=False,
                price_label=False,
            )
            panel_line.set(trace_df)
            panel_line.run_script(f"try {{ {panel_line.id}.series.applyOptions({{crosshairMarkerVisible: false}}); }} catch(e) {{ console.warn('[plot] panel_line applyOptions failed:', e); }}")
            if first_trace_id is None:
                first_trace_id = panel_line.id

        panel_first_trace_ids.append(first_trace_id or "")
        panel_titles.append(panel_title)

    if panel_chart_ids:
        chart_var = chart.id.removeprefix('window.')
        sub_pairs = list(zip(panel_chart_ids, panel_first_trace_ids))
        sub_entries = ', '.join(
            f'{{ ch: {c}.chart, indicSer: {t}.series }}' if t else f'{{ ch: {c}.chart, indicSer: null }}'
            for c, t in sub_pairs
        )
        chart.run_script(f'''
            (function() {{
                var mainChart = {chart.id}.chart;
                var _subs = [{sub_entries}];
                if (!mainChart || _subs.length === 0) return;

                var mainScale = mainChart.timeScale();

                // ── RANGE SYNC ─────────────────────────────────────────────────────
                // Uses getVisibleRange() (real timestamps) → setVisibleRange() on each sub.
                // Intentionally TIME-BASED — do NOT switch to setVisibleLogicalRange.
                // setVisibleRange clips to the subchart's series extent — that is why each
                // subchart has a timeline anchor covering ALL candle timestamps, so the
                // extent matches the main chart.
                // Initial sync runs in requestAnimationFrame so all series data has been
                // committed to the time scale before the call executes.
                console.log('[rangeSync] init, _subs.length=', _subs.length, '_subs=', _subs);
                window.rangeSyncHandler_{chart_var} = function(source) {{
                    var tr = mainScale.getVisibleRange();
                    console.log('[rangeSync] handler fired, source=' + source + ', tr=', tr, '_subs=', _subs);
                    if (!tr) return;
                    _subs.forEach(function(sub, i) {{
                        console.log('[rangeSync] sub[' + i + '].ch=', sub.ch, 'timeScale=', sub.ch ? sub.ch.timeScale() : null);
                        sub.ch.timeScale().setVisibleRange({{ from: tr.from, to: tr.to }});
                    }});
                }};
                mainScale.subscribeVisibleLogicalRangeChange(function(range) {{
                    if (range) {{ window.rangeSyncHandler_{chart_var}('subscription'); }}
                }});
                window.rangeSyncInterval_{chart_var} = setInterval(function() {{
                    window.rangeSyncHandler_{chart_var}('interval');
                }}, 2000);
                requestAnimationFrame(function() {{ window.rangeSyncHandler_{chart_var}('raf'); }});

                // ── CROSSHAIR SYNC ─────────────────────────────────────────────────
                // param.time (exact candle timestamp) → timeToCoordinate on the sub →
                // coordinateToLogical → dataByIndex(NearestLeft=-1) → setCrosshairPosition
                // with the real indicator value so the legend updates correctly.
                // NearestLeft handles sparse indicators: if no data at this exact bar,
                // snaps to the last known value to the left.
                window.crosshairSyncHandler_{chart_var} = function(param) {{
                    _subs.forEach(function(sub) {{
                        if (!param || !param.time) {{
                            try {{ sub.ch.clearCrosshairPosition(); }} catch (_) {{}}
                            return;
                        }}
                        try {{
                            var coord = sub.ch.timeScale().timeToCoordinate(param.time);
                            if (coord === null || coord === undefined) {{
                                sub.ch.clearCrosshairPosition();
                                return;
                            }}
                            var logical = sub.ch.timeScale().coordinateToLogical(coord);
                            if (logical === null || logical === undefined) {{
                                sub.ch.clearCrosshairPosition();
                                return;
                            }}
                            if (!sub.indicSer) {{
                                sub.ch.clearCrosshairPosition();
                                return;
                            }}
                            var dp = sub.indicSer.dataByIndex(Math.round(logical), -1);
                            if (dp) {{
                                sub.ch.setCrosshairPosition(dp.value, dp.time, sub.indicSer);
                            }} else {{
                                sub.ch.clearCrosshairPosition();
                            }}
                        }} catch (_) {{
                            try {{ sub.ch.clearCrosshairPosition(); }} catch (_2) {{}}
                        }}
                    }});
                }};
                mainChart.subscribeCrosshairMove(window.crosshairSyncHandler_{chart_var});
            }})();
        ''')

    marker_times: dict[str, list[datetime]] = {
        "long_entry": [],
        "short_entry": [],
        "winning_exit": [],
        "losing_exit": [],
    }
    marker_positions: dict[str, str] = {}
    if trades:
        entry_times = [trade.position.entry_time for trade in trades]
        exit_times = [trade.exit_time for trade in trades]
        long_idx = _downsample_indices(
            [i for i, trade in enumerate(trades) if trade.position.is_long],
            max_markers_per_group,
        )
        short_idx = _downsample_indices(
            [i for i, trade in enumerate(trades) if trade.position.is_short],
            max_markers_per_group,
        )
        win_idx = _downsample_indices(
            [i for i, trade in enumerate(trades) if trade.pnl >= 0],
            max_markers_per_group,
        )
        loss_idx = _downsample_indices(
            [i for i, trade in enumerate(trades) if trade.pnl < 0],
            max_markers_per_group,
        )

        if marker_layout == "staggered":
            entry_position = "above"
            long_entry_position = "above"
            short_entry_position = "above"
            exit_position = "below"
        else:
            # For candles: position longs above, shorts below to avoid rendering issues with "inside".
            # Exit markers go below. This prevents arrow markers from disappearing when inside candles.
            long_entry_position = "above"
            short_entry_position = "below"
            exit_position = "below"
        marker_positions = {
            "entry": "mixed",  # mixed positioning for entries
            "exit": exit_position,
        }

        entry_markers: list[dict[str, str | datetime]] = []
        exit_markers: list[dict[str, str | datetime]] = []

        def _marker_text(trade: ClosedTrade, *, event: str) -> str:
            direction = trade.position.direction.value
            pnl = float(trade.pnl)
            return f"#{trade.position.id} {event} {direction} PnL {pnl:+.2f}"

        for i in long_idx:
            entry_markers.append(
                {
                    "time": entry_times[i],
                    "position": long_entry_position,
                    "shape": "arrow_up",
                    "color": "#2563eb",
                    "text": _marker_text(trades[i], event="Entry"),
                }
            )
            marker_times["long_entry"].append(entry_times[i])
        for i in short_idx:
            entry_markers.append(
                {
                    "time": entry_times[i],
                    "position": short_entry_position,
                    "shape": "arrow_down",
                    "color": "#2563eb",
                    "text": _marker_text(trades[i], event="Entry"),
                }
            )
            marker_times["short_entry"].append(entry_times[i])
        for i in win_idx:
            exit_markers.append(
                {
                    "time": exit_times[i],
                    "position": exit_position,
                    "shape": "circle",
                    "color": "#16a34a",
                    "text": _marker_text(trades[i], event="Exit"),
                }
            )
            marker_times["winning_exit"].append(exit_times[i])
        for i in loss_idx:
            exit_markers.append(
                {
                    "time": exit_times[i],
                    "position": exit_position,
                    "shape": "circle",
                    "color": "#dc2626",
                    "text": _marker_text(trades[i], event="Exit"),
                }
            )
            marker_times["losing_exit"].append(exit_times[i])

        if use_candles and entry_markers and exit_markers:
            entry_marker_series = chart.create_line(
                name="Entries",
                color="rgba(0, 0, 0, 0)",
                width=1,
            )
            entry_marker_price_df = pd.DataFrame(
                {
                    "time": price_df["time"],
                    "Entries": price_df["close"],
                }
            )
            entry_marker_series.set(entry_marker_price_df)
            entry_marker_series.run_script(
                f"try {{ {entry_marker_series.id}.series.applyOptions({{crosshairMarkerVisible: false}}); }} catch(e) {{ console.warn('[plot] entry_marker applyOptions failed:', e); }}"
            )
            entry_marker_series.marker_list(entry_markers)
            marker_series.marker_list(exit_markers)
        elif entry_markers or exit_markers:
            marker_series.marker_list([*entry_markers, *exit_markers])

    headline = title or (f"{symbol} Price with Trade Markers" if symbol else "Price with Trade Markers")
    chart.run_script(
        f"""
        (function() {{
            if (document.getElementById('plot-heading')) return;
            const heading = document.createElement('div');
            heading.id = 'plot-heading';
            heading.style.position = 'absolute';
            heading.style.top = '8px';
            heading.style.left = '12px';
            heading.style.zIndex = '20';
            heading.style.fontFamily = 'Segoe UI, sans-serif';
            heading.style.fontSize = '14px';
            heading.style.fontWeight = '600';
            heading.style.color = '#0f172a';
            heading.style.background = 'rgba(255,255,255,0.88)';
            heading.style.padding = '4px 8px';
            heading.style.borderRadius = '6px';
            heading.textContent = {headline!r};
            document.getElementById('container').appendChild(heading);

            const applyLegendContrast = () => {{
                let styleTag = document.getElementById('legend-contrast-style');
                if (!styleTag) {{
                    styleTag = document.createElement('style');
                    styleTag.id = 'legend-contrast-style';
                    styleTag.textContent = `
                        .legend-toggle-switch {{
                            opacity: 1 !important;
                            display: inline-flex !important;
                            align-items: center;
                            justify-content: center;
                            min-width: 28px !important;
                            min-height: 22px !important;
                            background: #e2e8f0 !important;
                            border: 1px solid #64748b !important;
                            border-radius: 5px !important;
                            padding: 2px 4px !important;
                            box-shadow: inset 0 0 0 1px rgba(255, 255, 255, 0.65) !important;
                        }}
                        .legend-toggle-switch:hover {{
                            background: #cbd5e1 !important;
                            border-color: #334155 !important;
                        }}
                        .legend-toggle-switch svg {{
                            opacity: 1 !important;
                            overflow: visible;
                            width: 22px !important;
                            height: 16px !important;
                        }}
                        .legend-toggle-switch svg g {{
                            fill: #0f172a !important;
                            stroke: #0f172a !important;
                        }}
                        .legend-toggle-switch svg path {{
                            stroke: #0f172a !important;
                            stroke-opacity: 1 !important;
                            fill-opacity: 1 !important;
                            stroke-width: 2.4 !important;
                        }}
                    `;
                    document.head.appendChild(styleTag);
                }}
                document.querySelectorAll('.legend-toggle-switch').forEach((el) => {{
                    el.style.opacity = '1';
                    el.style.filter = 'none';
                    el.style.background = '#e2e8f0';
                    el.style.borderColor = '#64748b';
                }});
                document.querySelectorAll('.legend-toggle-switch svg g').forEach((g) => {{
                    g.setAttribute('fill', '#0f172a');
                    g.setAttribute('stroke', '#0f172a');
                }});
                document.querySelectorAll('.legend-toggle-switch svg path').forEach((path) => {{
                    path.style.stroke = '#0f172a';
                    path.style.strokeWidth = '2.4';
                    path.style.opacity = '1';
                    if (path.style.fill && path.style.fill !== 'none') {{
                        path.style.fill = '#0f172a';
                    }}
                }});
                document.querySelectorAll('.legend *').forEach((el) => {{
                    if (el instanceof HTMLElement) {{
                        el.style.opacity = '1';
                    }}
                }});
            }};
            applyLegendContrast();
            requestAnimationFrame(applyLegendContrast);
        }})();
        """
    )
    chart.fit()
    chart.load()

    html_doc = f"{chart._html}</script></body></html>"
    metadata = {
        "engine": "lightweight-charts",
        "title": headline,
        "mode": "candlestick" if use_candles else "line",
        "marker_layout": marker_layout,
        "marker_positions": marker_positions,
        "marker_times": marker_times,
        "rangebreak_count": len(_infer_rangebreaks(times)),
        "use_trade_fill_prices": use_trade_fill_prices,
        "overlay_indicators": overlay_names,
        "panel_indicators": panel_titles,
    }
    return LightweightChartFigure(html_doc, metadata=metadata)


def plot_trade_history_table_interactive(
    trades: Sequence[ClosedTrade],
    *,
    title: str = "Trade History",
):
    go = _require_plotly()

    if not trades:
        raise ValueError("Trade history is empty.")

    ids = [trade.position.id for trade in trades]
    symbols = [trade.position.symbol for trade in trades]
    directions = [trade.position.direction.value for trade in trades]
    qty = [f"{float(trade.position.qty):,.4f}" for trade in trades]
    entry_time = [trade.position.entry_time.strftime("%Y-%m-%d %H:%M") for trade in trades]
    entry_price = [f"{float(trade.position.entry_price):,.5f}" for trade in trades]
    exit_time = [trade.exit_time.strftime("%Y-%m-%d %H:%M") for trade in trades]
    exit_price = [f"{float(trade.exit_price):,.5f}" for trade in trades]
    pnl = [float(trade.pnl) for trade in trades]
    pnl_text = [f"{value:+,.2f}" for value in pnl]
    costs = [float(trade.costs) for trade in trades]
    costs_text = [f"{value:,.2f}" for value in costs]
    duration_hours = [
        f"{(trade.exit_time - trade.position.entry_time).total_seconds() / 3600.0:,.1f}"
        for trade in trades
    ]

    neutral_bg = ["#ffffff" for _ in trades]
    pnl_bg = ["#dff4df" if value >= 0.0 else "#fde2e2" for value in pnl]
    cell_fill = [
        neutral_bg,   # Trade #
        neutral_bg,   # Symbol
        neutral_bg,   # Direction
        neutral_bg,   # Qty
        neutral_bg,   # Entry Time
        neutral_bg,   # Entry Price
        neutral_bg,   # Exit Time
        neutral_bg,   # Exit Price
        pnl_bg,       # PnL
        neutral_bg,   # Costs
        neutral_bg,   # Hours
    ]

    fig = go.Figure(
        data=[
            go.Table(
                columnwidth=[0.7, 0.9, 0.8, 0.8, 1.3, 1.0, 1.3, 1.0, 0.9, 0.9, 0.9],
                header={
                    "values": [
                        "Trade #",
                        "Symbol",
                        "Direction",
                        "Qty",
                        "Entry Time",
                        "Entry Price",
                        "Exit Time",
                        "Exit Price",
                        "PnL",
                        "Costs",
                        "Hours",
                    ],
                    "fill_color": "#2b2d42",
                    "font": {"color": "#ffffff", "size": 12},
                    "align": "left",
                    "height": 28,
                },
                cells={
                    "values": [
                        ids,
                        symbols,
                        directions,
                        qty,
                        entry_time,
                        entry_price,
                        exit_time,
                        exit_price,
                        pnl_text,
                        costs_text,
                        duration_hours,
                    ],
                    "fill_color": cell_fill,
                    "align": "left",
                    "height": 24,
                    "font": {
                        "size": 11,
                        "color": [
                            ["#222222"] * len(trades),
                            ["#222222"] * len(trades),
                            ["#222222"] * len(trades),
                            ["#222222"] * len(trades),
                            ["#222222"] * len(trades),
                            ["#222222"] * len(trades),
                            ["#222222"] * len(trades),
                            ["#222222"] * len(trades),
                            ["#222222"] * len(trades),
                            ["#222222"] * len(trades),
                            ["#222222"] * len(trades),
                        ],
                    },
                },
            )
        ]
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        margin={"l": 16, "r": 16, "t": 60, "b": 16},
        annotations=[
            {
                "x": 1.0,
                "y": 1.12,
                "xref": "paper",
                "yref": "paper",
                "text": f"trades={len(trades)}",
                "showarrow": False,
                "xanchor": "right",
                "yanchor": "top",
                "font": {"size": 11, "color": "#666666"},
            }
        ],
    )
    return fig


def save_interactive_figure_html(fig, output_path: str | Path, *, auto_open: bool = False) -> None:
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(fig, LightweightChartFigure):
        fig.write_html(str(target), full_html=True, auto_open=auto_open)
        return

    fig.write_html(
        str(target),
        include_plotlyjs="cdn",
        full_html=True,
        auto_open=auto_open,
        config={
            "scrollZoom": True,
            "doubleClick": "reset+autosize",
            "displaylogo": False,
        },
    )


def save_trade_history_interactive_html(
    *,
    trades: Sequence[ClosedTrade],
    output_path: str | Path,
    title: str = "Trade History",
    auto_open: bool = False,
) -> Path:
    fig = plot_trade_history_table_interactive(trades, title=title)
    target = Path(output_path)
    save_interactive_figure_html(fig, target, auto_open=auto_open)
    return target


def show() -> None:
    plt, _ = _require_matplotlib()
    plt.show()
