from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from algotrading.indicators.indicator import IndicatorPlotSpec

from .backtest_broker import ClosedTrade
from .monte_carlo import simulate_monte_carlo_from_broker
from .plotting import (
    plot_equity_and_drawdown,
    plot_equity_vs_benchmark,
    plot_monthly_returns_heatmap,
    plot_monte_carlo_paths,
    plot_price_with_trades_interactive,
    plot_trade_pnl_distribution,
    save_interactive_figure_html,
    save_swap_history_interactive_html,
    save_trade_history_interactive_html,
)
from .report import BacktestReport


class BacktestPlotter:
    """Central plotting facade for backtest outputs.

    This class wraps the lower-level plotting functions so callers can pass a
    single report object instead of manually wiring broker/strategy arrays into
    many function calls.
    """

    def __init__(self, report: BacktestReport, trades: list[ClosedTrade] | None = None):
        self.report = report
        self._trades = list(trades) if trades is not None else list(report.broker.trade_log)

    def _save_plot(self, fig: Any, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=140, bbox_inches="tight")

    def _indicator_specs(
        self,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        overlays: list[dict[str, Any]] = []
        panels_by_title: dict[str, list[dict[str, Any]]] = {}
        marker_groups: list[dict[str, Any]] = []
        primary_times = [t for t in self.report.price_times]
        primary_ns = pd.to_datetime(primary_times, utc=True).astype("int64").to_numpy(dtype=np.int64)
        bar_count = len(primary_times)
        primary_times_py = pd.to_datetime(primary_times, utc=True).to_pydatetime().tolist()

        def _align_values(
            values: np.ndarray,
            source_times: list[Any],
            *,
            forward_fill: bool = True,
        ) -> np.ndarray | None:
            if values.ndim != 1:
                return None
            if len(values) == bar_count:
                return values
            # Indicators sourced from another indicator skip bars during the
            # upstream's warmup, so their length can be shorter than the primary
            # timeline. Right-align by prepending nans — values map to the tail.
            if len(source_times) == bar_count and 0 < len(values) < bar_count:
                out = np.full(bar_count, np.nan, dtype=np.float64)
                out[bar_count - len(values):] = values
                return out
            if len(values) != len(source_times) or bar_count == 0:
                return None

            source_ns = pd.to_datetime(source_times, utc=True).astype("int64").to_numpy(dtype=np.int64)
            out = np.full(bar_count, np.nan, dtype=np.float64)
            idx = np.searchsorted(primary_ns, source_ns)
            valid = (idx >= 0) & (idx < bar_count) & (primary_ns[np.clip(idx, 0, bar_count - 1)] == source_ns)
            if not np.any(valid):
                return None

            out[idx[valid]] = values[valid]

            if forward_fill:
                # HTF values persist across primary bars until next HTF close.
                for i in range(1, len(out)):
                    if np.isnan(out[i]):
                        out[i] = out[i - 1]
            return out

        default_colors = [
            "#1d4ed8",
            "#7c3aed",
            "#0f766e",
            "#b45309",
            "#2563eb",
            "#dc2626",
        ]
        color_idx = 0

        for indicator, source_bars in self.report.strategy.plotted_indicator_bindings:
            spec: IndicatorPlotSpec = indicator.plot_spec()
            custom_label = self.report.strategy.get_indicator_plot_label(indicator)
            for trace in spec.traces:
                values = getattr(indicator, trace.attr, None)
                if values is None:
                    continue
                raw_arr = np.asarray(values, dtype=np.float64)
                source_times = [t for t in source_bars.time]
                arr = _align_values(raw_arr, source_times)
                if arr is None:
                    continue
                if custom_label:
                    if len(spec.traces) == 1:
                        label = custom_label
                    else:
                        suffix = trace.label or trace.attr
                        label = f"{custom_label} {suffix}"
                else:
                    label = trace.label or f"{indicator.__class__.__name__}:{trace.attr}"
                color = trace.color or default_colors[color_idx % len(default_colors)]
                color_idx += 1
                trace_spec = {
                    "name": label,
                    "values": arr,
                    "color": color,
                    "width": trace.width,
                }
                if spec.kind == "overlay":
                    overlays.append(trace_spec)
                else:
                    title = spec.panel_title or indicator.__class__.__name__
                    panels_by_title.setdefault(title, []).append(trace_spec)

            for marker in spec.markers:
                values = getattr(indicator, marker.attr, None)
                if values is None:
                    continue
                raw_arr = np.asarray(values, dtype=np.float64)
                source_times = [t for t in source_bars.time]
                arr = _align_values(raw_arr, source_times, forward_fill=False)
                if arr is None:
                    continue
                entries: list[dict[str, Any]] = []
                for i, price in enumerate(arr):
                    if np.isnan(price):
                        continue
                    entries.append({
                        "time": primary_times_py[i],
                        "position": marker.position,
                        "shape": marker.shape,
                        "color": marker.color,
                        "text": marker.label,
                    })
                if not entries:
                    continue
                label = (
                    f"{custom_label} {marker.label}".strip()
                    if custom_label
                    else f"{indicator.__class__.__name__}:{marker.attr}"
                )
                marker_groups.append({"name": label, "markers": entries})

        panels = [{"title": title, "traces": traces} for title, traces in panels_by_title.items()]
        return overlays, panels, marker_groups

    def save_standard_bundle(
        self,
        *,
        output_dir: Path,
        max_markers_per_group: int = 0,
        max_line_points: int = 0,
        include_monthly_heatmap: bool = True,
    ) -> dict[str, Any]:
        """Save standard report plots and return output paths."""
        broker = self.report.broker
        trades = self._trades

        broker_for_trades = broker
        if len(trades) != len(broker.trade_log):
            broker_for_trades = copy.copy(broker)
            broker_for_trades.trade_log = list(trades)

        output_dir.mkdir(parents=True, exist_ok=True)

        outputs: dict[str, Any] = {}

        fig_eq, _ = plot_equity_and_drawdown(broker)
        equity_path = output_dir / "equity_drawdown.png"
        self._save_plot(fig_eq, equity_path)
        outputs["equity_drawdown"] = equity_path

        if include_monthly_heatmap and trades:
            fig_monthly, _ = plot_monthly_returns_heatmap(
                broker,
                trades=trades,
                title=f"{self.report.symbol} Monthly Returns Heatmap",
            )
            monthly_path = output_dir / "monthly_returns_heatmap.png"
            self._save_plot(fig_monthly, monthly_path)
            outputs["monthly_returns"] = monthly_path

        overlays, panels, indicator_markers = self._indicator_specs()
        fig_history = plot_price_with_trades_interactive(
            price_times=self.report.price_times,
            prices=self.report.price_close,
            trades=trades,
            symbol=self.report.symbol,
            title=f"{self.report.symbol} Trade History",
            max_markers_per_group=max_markers_per_group,
            max_line_points=max_line_points,
            price_open=self.report.price_open,
            price_high=self.report.price_high,
            price_low=self.report.price_low,
            price_close=self.report.price_close,
            render_mode="candlestick",
            marker_layout="fill",
            indicator_overlays=overlays,
            indicator_panels=panels,
            indicator_markers=indicator_markers,
        )
        trade_history_path = output_dir / "trade_history.html"
        save_interactive_figure_html(fig_history, trade_history_path)
        outputs["trade_history"] = trade_history_path

        if trades:
            mc = simulate_monte_carlo_from_broker(broker_for_trades, n_paths=1_000, random_state=42)
            fig_mc, _ = plot_monte_carlo_paths(mc)
            mc_path = output_dir / "monte_carlo.png"
            self._save_plot(fig_mc, mc_path)
            outputs["monte_carlo"] = mc_path

            fig_pnl, _ = plot_trade_pnl_distribution(broker_for_trades)
            pnl_path = output_dir / "trade_pnl_distribution.png"
            self._save_plot(fig_pnl, pnl_path)
            outputs["trade_pnl_distribution"] = pnl_path

            trade_history_table_path = output_dir / "trade_history_table.html"
            save_trade_history_interactive_html(
                trades=trades,
                output_path=trade_history_table_path,
                title=f"{self.report.symbol} Trade History Table",
            )
            outputs["trade_history_table"] = trade_history_table_path

            swap_history_table_path = output_dir / "swap_history_table.html"
            save_swap_history_interactive_html(
                trades=trades,
                output_path=swap_history_table_path,
                title=f"{self.report.symbol} Swap History Table",
            )
            outputs["swap_history_table"] = swap_history_table_path

        if self.report.benchmark_times and self.report.benchmark_prices:
            fig_vs, _ = plot_equity_vs_benchmark(
                broker,
                benchmark_times=self.report.benchmark_times,
                benchmark_prices=self.report.benchmark_prices,
                benchmark_name=self.report.benchmark_name,
            )
            vs_path = output_dir / "equity_vs_benchmark.png"
            self._save_plot(fig_vs, vs_path)
            outputs["equity_vs_benchmark"] = vs_path

        return outputs
