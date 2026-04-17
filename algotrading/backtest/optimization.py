from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
import math
import random
from statistics import mean


type Params = dict[str, object]
type ParamConstraint = Callable[[Params], bool]


@dataclass(frozen=True)
class OptimizationResult[T]:
    """Single evaluated parameter set."""

    params: dict[str, object]
    evaluation: T
    score: float


@dataclass(frozen=True)
class OptimizationReport[T]:
    """Ranked optimization results."""

    results: list[OptimizationResult[T]]
    objective_name: str
    maximize: bool

    @property
    def best(self) -> OptimizationResult[T]:
        if not self.results:
            raise ValueError("No optimization results available")
        return self.results[0]


@dataclass(frozen=True)
class WalkForwardWindow:
    """Inclusive-exclusive index ranges for one walk-forward step."""

    train_start: int
    train_end: int
    test_start: int
    test_end: int


@dataclass(frozen=True)
class WalkForwardStepResult[T]:
    """Optimization + out-of-sample evaluation for one window."""

    window: WalkForwardWindow
    optimization: OptimizationReport[T]
    in_sample_best: OptimizationResult[T]
    out_of_sample_evaluation: T
    out_of_sample_score: float


@dataclass(frozen=True)
class WalkForwardReport[T]:
    """Collection of walk-forward steps with a convenience aggregate."""

    steps: list[WalkForwardStepResult[T]]
    objective_name: str
    maximize: bool

    @property
    def mean_out_of_sample_score(self) -> float | None:
        finite_scores = [step.out_of_sample_score for step in self.steps if math.isfinite(step.out_of_sample_score)]
        if not finite_scores:
            return None
        return mean(finite_scores)


@dataclass(frozen=True)
class TimeSeriesCVEvaluation[T]:
    """Per-parameter evaluation across multiple chronological validation folds."""

    fold_windows: list[WalkForwardWindow]
    fold_evaluations: list[T]
    fold_scores: list[float]
    aggregated_score: float


def parameter_grid(
    param_space: Mapping[str, Sequence[object] | Iterable[object]],
    constraint: ParamConstraint | None = None,
) -> list[Params]:
    """Build cartesian-product parameter combinations from a param space mapping."""
    if not param_space:
        raise ValueError("param_space must not be empty")

    keys = list(param_space.keys())
    values_per_key: list[list[object]] = []
    for key in keys:
        values = list(param_space[key])
        if not values:
            raise ValueError(f"Parameter '{key}' has no candidate values")
        values_per_key.append(values)

    grid: list[Params] = []

    def _build(index: int, current: Params) -> None:
        if index == len(keys):
            params = current.copy()
            if constraint is None or constraint(params):
                grid.append(params)
            return
        key = keys[index]
        for value in values_per_key[index]:
            current[key] = value
            _build(index + 1, current)

    _build(0, {})
    return grid


def _objective_getter[T](objective: str | Callable[[T], float]) -> tuple[Callable[[T], float], str]:
    if isinstance(objective, str):
        name = objective

        def _get_metric(evaluation: T) -> float:
            value = getattr(evaluation, objective)
            if value is None:
                raise ValueError(f"Objective '{objective}' is None for evaluation {evaluation}")
            return float(value)

        return _get_metric, name

    return objective, getattr(objective, "__name__", "objective")


def _normalize_score(score: float, maximize: bool) -> float:
    if math.isnan(score):
        return -math.inf if maximize else math.inf
    return score


def optimize_parameters[T](
    param_space: Mapping[str, Sequence[object] | Iterable[object]],
    evaluate: Callable[[Params], T],
    objective: str | Callable[[T], float],
    maximize: bool = True,
    constraint: ParamConstraint | None = None,
) -> OptimizationReport[T]:
    """Evaluate all parameter combinations and rank by objective score."""
    score_fn, objective_name = _objective_getter(objective)
    results: list[OptimizationResult[T]] = []

    for params in parameter_grid(param_space, constraint=constraint):
        evaluation = evaluate(params)
        raw_score = score_fn(evaluation)
        score = _normalize_score(float(raw_score), maximize)
        results.append(
            OptimizationResult(
                params=params,
                evaluation=evaluation,
                score=score,
            )
        )

    if not results:
        raise ValueError("No valid parameter combinations to evaluate")

    results.sort(key=lambda result: result.score, reverse=maximize)
    return OptimizationReport(results=results, objective_name=objective_name, maximize=maximize)


def random_parameter_samples(
    param_space: Mapping[str, Sequence[object] | Iterable[object]],
    n_iter: int,
    random_state: int | None = None,
    replace: bool = False,
    constraint: ParamConstraint | None = None,
) -> list[Params]:
    """Randomly sample parameter combinations from a discrete search space."""
    if n_iter <= 0:
        raise ValueError("n_iter must be > 0")

    grid = parameter_grid(param_space, constraint=constraint)
    if not grid:
        raise ValueError("No valid parameter combinations available after applying constraint")
    rng = random.Random(random_state)

    if replace:
        return [grid[rng.randrange(len(grid))].copy() for _ in range(n_iter)]

    if n_iter > len(grid):
        raise ValueError(
            f"n_iter={n_iter} exceeds total combinations={len(grid)} when replace=False"
        )
    picks = rng.sample(grid, k=n_iter)
    return [params.copy() for params in picks]


def optimize_parameters_random[T](
    param_space: Mapping[str, Sequence[object] | Iterable[object]],
    evaluate: Callable[[Params], T],
    objective: str | Callable[[T], float],
    n_iter: int,
    maximize: bool = True,
    random_state: int | None = None,
    replace: bool = False,
    constraint: ParamConstraint | None = None,
) -> OptimizationReport[T]:
    """Evaluate a random subset of parameter combinations and rank by score."""
    score_fn, objective_name = _objective_getter(objective)
    results: list[OptimizationResult[T]] = []

    for params in random_parameter_samples(
        param_space=param_space,
        n_iter=n_iter,
        random_state=random_state,
        replace=replace,
        constraint=constraint,
    ):
        evaluation = evaluate(params)
        raw_score = score_fn(evaluation)
        score = _normalize_score(float(raw_score), maximize)
        results.append(
            OptimizationResult(
                params=params,
                evaluation=evaluation,
                score=score,
            )
        )

    results.sort(key=lambda result: result.score, reverse=maximize)
    return OptimizationReport(results=results, objective_name=objective_name, maximize=maximize)


def walk_forward_windows(
    total_size: int,
    train_size: int,
    test_size: int,
    step_size: int | None = None,
    anchored: bool = False,
) -> list[WalkForwardWindow]:
    """Create rolling or anchored walk-forward windows over index-based data."""
    if total_size <= 0:
        raise ValueError("total_size must be > 0")
    if train_size <= 0:
        raise ValueError("train_size must be > 0")
    if test_size <= 0:
        raise ValueError("test_size must be > 0")

    step = step_size or test_size
    if step <= 0:
        raise ValueError("step_size must be > 0")

    windows: list[WalkForwardWindow] = []
    cursor = train_size

    while cursor + test_size <= total_size:
        train_start = 0 if anchored else cursor - train_size
        windows.append(
            WalkForwardWindow(
                train_start=train_start,
                train_end=cursor,
                test_start=cursor,
                test_end=cursor + test_size,
            )
        )
        cursor += step

    return windows


def time_series_cv_windows(
    total_size: int,
    min_train_size: int,
    validation_size: int,
    step_size: int | None = None,
    anchored: bool = True,
) -> list[WalkForwardWindow]:
    """Build chronological folds for time-series cross-validation.

    Fold train windows grow when anchored=True (default), which is a common
    expanding-window CV setup for financial time series.
    """
    return walk_forward_windows(
        total_size=total_size,
        train_size=min_train_size,
        test_size=validation_size,
        step_size=step_size,
        anchored=anchored,
    )


def _default_fold_aggregator(scores: Sequence[float]) -> float:
    finite = [s for s in scores if math.isfinite(s)]
    if not finite:
        return math.nan
    return mean(finite)


def run_walk_forward[T](
    param_space: Mapping[str, Sequence[object] | Iterable[object]],
    total_size: int,
    train_size: int,
    test_size: int,
    evaluate_window: Callable[[Params, int, int], T],
    objective: str | Callable[[T], float],
    maximize: bool = True,
    step_size: int | None = None,
    anchored: bool = False,
    optimizer: Callable[
        [
            Mapping[str, Sequence[object] | Iterable[object]],
            Callable[[Params], T],
            str | Callable[[T], float],
            bool,
            ParamConstraint | None,
        ],
        OptimizationReport[T],
    ] = optimize_parameters,
    constraint: ParamConstraint | None = None,
) -> WalkForwardReport[T]:
    """Run walk-forward optimization over indexed data slices.

    evaluate_window receives (params, start, end) where start/end are
    inclusive-exclusive indices into your dataset.
    """
    score_fn, objective_name = _objective_getter(objective)
    steps: list[WalkForwardStepResult[T]] = []

    for window in walk_forward_windows(
        total_size=total_size,
        train_size=train_size,
        test_size=test_size,
        step_size=step_size,
        anchored=anchored,
    ):
        optimization = optimizer(
            param_space=param_space,
            evaluate=lambda p, w=window: evaluate_window(p, w.train_start, w.train_end),
            objective=objective,
            maximize=maximize,
            constraint=constraint,
        )
        best = optimization.best

        out_eval = evaluate_window(best.params, window.test_start, window.test_end)
        out_score = _normalize_score(float(score_fn(out_eval)), maximize)
        steps.append(
            WalkForwardStepResult(
                window=window,
                optimization=optimization,
                in_sample_best=best,
                out_of_sample_evaluation=out_eval,
                out_of_sample_score=out_score,
            )
        )

    if not steps:
        raise ValueError("No walk-forward windows produced. Check total_size/train_size/test_size.")

    return WalkForwardReport(steps=steps, objective_name=objective_name, maximize=maximize)


def optimize_parameters_random_cv[T](
    param_space: Mapping[str, Sequence[object] | Iterable[object]],
    evaluate_window: Callable[[Params, int, int], T],
    objective: str | Callable[[T], float],
    total_size: int,
    min_train_size: int,
    validation_size: int,
    n_iter: int,
    maximize: bool = True,
    random_state: int | None = None,
    replace: bool = False,
    constraint: ParamConstraint | None = None,
    step_size: int | None = None,
    anchored: bool = True,
    fold_aggregator: Callable[[Sequence[float]], float] = _default_fold_aggregator,
) -> OptimizationReport[TimeSeriesCVEvaluation[T]]:
    """Random-search optimizer scored by aggregated time-series CV fold results."""
    windows = time_series_cv_windows(
        total_size=total_size,
        min_train_size=min_train_size,
        validation_size=validation_size,
        step_size=step_size,
        anchored=anchored,
    )
    if not windows:
        raise ValueError("No time-series CV windows produced. Check size arguments.")

    score_fn, _ = _objective_getter(objective)

    def _evaluate_params_cv(params: Params) -> TimeSeriesCVEvaluation[T]:
        fold_evaluations: list[T] = []
        fold_scores: list[float] = []
        for window in windows:
            evaluation = evaluate_window(params, window.test_start, window.test_end)
            raw_score = float(score_fn(evaluation))
            fold_evaluations.append(evaluation)
            fold_scores.append(_normalize_score(raw_score, maximize))

        aggregated_raw = float(fold_aggregator(fold_scores))
        aggregated_score = _normalize_score(aggregated_raw, maximize)
        return TimeSeriesCVEvaluation(
            fold_windows=windows,
            fold_evaluations=fold_evaluations,
            fold_scores=fold_scores,
            aggregated_score=aggregated_score,
        )

    return optimize_parameters_random(
        param_space=param_space,
        evaluate=_evaluate_params_cv,
        objective=lambda evaluation: evaluation.aggregated_score,
        n_iter=n_iter,
        maximize=maximize,
        random_state=random_state,
        replace=replace,
        constraint=constraint,
    )
