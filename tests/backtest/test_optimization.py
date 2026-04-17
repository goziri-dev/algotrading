from dataclasses import dataclass

import pytest

from algotrading.backtest.optimization import (
    optimize_parameters,
    optimize_parameters_random,
    optimize_parameters_random_cv,
    parameter_grid,
    random_parameter_samples,
    run_walk_forward,
    time_series_cv_windows,
    walk_forward_windows,
)


@dataclass(frozen=True)
class Eval:
    score: float
    loss: float


def test_parameter_grid_builds_cartesian_product() -> None:
    grid = parameter_grid({"fast": [5, 10], "slow": [20, 30]})

    assert len(grid) == 4
    assert {tuple(sorted(params.items())) for params in grid} == {
        (("fast", 5), ("slow", 20)),
        (("fast", 5), ("slow", 30)),
        (("fast", 10), ("slow", 20)),
        (("fast", 10), ("slow", 30)),
    }


def test_parameter_grid_applies_constraint() -> None:
    grid = parameter_grid(
        {"fast": [5, 10, 20], "slow": [10, 20, 30]},
        constraint=lambda p: int(p["fast"]) < int(p["slow"]),
    )

    assert all(int(p["fast"]) < int(p["slow"]) for p in grid)


def test_optimize_parameters_with_string_objective() -> None:
    report = optimize_parameters(
        param_space={"x": [1, 2, 3]},
        evaluate=lambda params: Eval(score=float(params["x"]), loss=10 - float(params["x"])),
        objective="score",
        maximize=True,
    )

    assert report.best.params["x"] == 3
    assert report.best.score == pytest.approx(3.0)


def test_optimize_parameters_with_callable_objective_minimize() -> None:
    report = optimize_parameters(
        param_space={"x": [1, 2, 3]},
        evaluate=lambda params: Eval(score=float(params["x"]), loss=abs(2 - float(params["x"]))),
        objective=lambda result: result.loss,
        maximize=False,
    )

    assert report.best.params["x"] == 2
    assert report.best.score == pytest.approx(0.0)


def test_random_parameter_samples_is_seeded_and_unique_without_replacement() -> None:
    samples_a = random_parameter_samples(
        param_space={"a": [1, 2, 3], "b": [10, 20]},
        n_iter=4,
        random_state=7,
    )
    samples_b = random_parameter_samples(
        param_space={"a": [1, 2, 3], "b": [10, 20]},
        n_iter=4,
        random_state=7,
    )

    assert samples_a == samples_b
    assert len({tuple(sorted(s.items())) for s in samples_a}) == 4


def test_optimize_parameters_random_evaluates_subset() -> None:
    report = optimize_parameters_random(
        param_space={"x": [1, 2, 3, 4, 5]},
        evaluate=lambda params: Eval(score=float(params["x"]), loss=0.0),
        objective="score",
        n_iter=2,
        maximize=True,
        random_state=11,
    )

    assert len(report.results) == 2


def test_optimize_parameters_random_respects_constraint() -> None:
    report = optimize_parameters_random(
        param_space={"fast": [5, 10, 20], "slow": [10, 20]},
        evaluate=lambda params: Eval(score=float(params["slow"]) - float(params["fast"]), loss=0.0),
        objective="score",
        n_iter=2,
        maximize=True,
        random_state=3,
        constraint=lambda p: int(p["fast"]) < int(p["slow"]),
    )

    assert all(int(r.params["fast"]) < int(r.params["slow"]) for r in report.results)


def test_walk_forward_windows_support_rolling_and_anchored() -> None:
    rolling = walk_forward_windows(total_size=60, train_size=20, test_size=10)
    anchored = walk_forward_windows(total_size=60, train_size=20, test_size=10, anchored=True)

    assert [(w.train_start, w.train_end, w.test_start, w.test_end) for w in rolling] == [
        (0, 20, 20, 30),
        (10, 30, 30, 40),
        (20, 40, 40, 50),
        (30, 50, 50, 60),
    ]
    assert [(w.train_start, w.train_end, w.test_start, w.test_end) for w in anchored] == [
        (0, 20, 20, 30),
        (0, 30, 30, 40),
        (0, 40, 40, 50),
        (0, 50, 50, 60),
    ]


def test_time_series_cv_windows_expanding_by_default() -> None:
    folds = time_series_cv_windows(
        total_size=50,
        min_train_size=20,
        validation_size=10,
    )

    assert [(f.train_start, f.train_end, f.test_start, f.test_end) for f in folds] == [
        (0, 20, 20, 30),
        (0, 30, 30, 40),
        (0, 40, 40, 50),
    ]


def test_run_walk_forward_reoptimizes_each_window() -> None:
    def evaluate_window(params: dict[str, object], start: int, end: int) -> Eval:
        # First regime prefers p=1; later regime prefers p=2.
        target = 1 if end <= 20 else 2
        p = int(params["p"])
        score = 10.0 - abs(p - target)
        return Eval(score=score, loss=10.0 - score)

    report = run_walk_forward(
        param_space={"p": [1, 2]},
        total_size=40,
        train_size=20,
        test_size=10,
        evaluate_window=evaluate_window,
        objective="score",
        maximize=True,
        step_size=10,
    )

    assert len(report.steps) == 2
    assert report.steps[0].in_sample_best.params["p"] == 1
    assert report.steps[1].in_sample_best.params["p"] == 2
    assert report.steps[0].out_of_sample_score == pytest.approx(9.0)
    assert report.steps[1].out_of_sample_score == pytest.approx(10.0)
    assert report.mean_out_of_sample_score == pytest.approx(9.5)


def test_optimize_parameters_random_cv_aggregates_fold_scores() -> None:
    def evaluate_window(params: dict[str, object], start: int, end: int) -> Eval:
        # Parameter p=2 should dominate all folds; score also changes by fold.
        p = int(params["p"])
        fold_bonus = (end - start) / 10.0
        score = 10.0 - abs(p - 2) + fold_bonus
        return Eval(score=score, loss=10.0 - score)

    report = optimize_parameters_random_cv(
        param_space={"p": [1, 2, 3, 4]},
        evaluate_window=evaluate_window,
        objective="score",
        total_size=40,
        min_train_size=20,
        validation_size=10,
        n_iter=3,
        random_state=5,
        constraint=lambda p: int(p["p"]) <= 3,
    )

    assert len(report.results) == 3
    assert report.best.params["p"] == 2
    assert len(report.best.evaluation.fold_windows) == 2
    assert len(report.best.evaluation.fold_scores) == 2


def test_run_walk_forward_accepts_random_optimizer() -> None:
    def evaluate_window(params: dict[str, object], start: int, end: int) -> Eval:
        p = int(params["p"])
        target = 1 if end <= 20 else 2
        score = 10.0 - abs(p - target)
        return Eval(score=score, loss=10.0 - score)

    report = run_walk_forward(
        param_space={"p": [1, 2, 3, 4]},
        total_size=40,
        train_size=20,
        test_size=10,
        evaluate_window=evaluate_window,
        objective="score",
        maximize=True,
        step_size=10,
        optimizer=lambda param_space, evaluate, objective, maximize, constraint: optimize_parameters_random(
            param_space=param_space,
            evaluate=evaluate,
            objective=objective,
            maximize=maximize,
            n_iter=2,
            random_state=19,
            constraint=constraint,
        ),
        constraint=lambda p: int(p["p"]) <= 3,
    )

    assert len(report.steps) == 2
