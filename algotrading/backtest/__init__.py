from .backtest_broker import BacktestBroker, SymbolSpec
from .backtester import Backtester
from .backtest_session import BacktestSession
from .report import BacktestReport
from .plotter import BacktestPlotter
from .workflow import save_backtest_plots
from .plotting import (
	plot_drawdown,
	plot_equity_and_drawdown,
	plot_equity_curve,
	plot_trade_pnl_distribution,
	plot_equity_vs_benchmark,
	plot_equity_vs_symbols,
	plot_symbol_return_correlation,
	plot_monthly_returns_heatmap,
	plot_monte_carlo_paths,
	plot_price_with_trades,
	plot_price_with_trades_interactive,
	plot_trade_history_table_interactive,
	save_trade_history_interactive_html,
	save_interactive_figure_html,
	show,
)
from .summary import BacktestStats, calculate_backtest_stats, print_backtest_summary
from .monte_carlo import MonteCarloReport, simulate_monte_carlo_from_broker
from .optimization import (
	OptimizationReport,
	OptimizationResult,
	TimeSeriesCVEvaluation,
	WalkForwardReport,
	WalkForwardStepResult,
	WalkForwardWindow,
	optimize_parameters,
	optimize_parameters_random,
	optimize_parameters_random_cv,
	parameter_grid,
	random_parameter_samples,
	run_walk_forward,
	time_series_cv_windows,
	walk_forward_windows,
)
