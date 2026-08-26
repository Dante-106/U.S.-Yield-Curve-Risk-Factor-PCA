import numpy as np
import pandas as pd
import pytest

from yield_curve_pca.config import CORE_TENORS, MATURITY_YEARS, ForecastConfig
from yield_curve_pca.forecast import (
    required_forecast_holdout,
    required_forecast_observations,
    walk_forward_forecast,
)
from yield_curve_pca.pca import economic_shape_templates


def _synthetic_ar_curve(seed=21, observations=420):
    rng = np.random.default_rng(seed)
    templates = economic_shape_templates([MATURITY_YEARS[t] for t in CORE_TENORS])
    factors = np.zeros((observations, 3))
    shocks = rng.normal(size=(observations, 3)) * np.array([3.0, 1.5, 0.75])
    for index in range(1, observations):
        factors[index] = np.array([0.85, 0.70, 0.55]) * factors[index - 1] + shocks[index]
    changes = factors @ templates + rng.normal(scale=0.05, size=(observations, len(CORE_TENORS)))
    dates = pd.date_range("2015-01-02", periods=observations, freq="W-FRI")
    change_frame = pd.DataFrame(changes, index=dates, columns=CORE_TENORS)
    initial_date = dates[0] - pd.offsets.Week(weekday=4)
    level_dates = pd.DatetimeIndex([initial_date, *dates])
    level_values = np.vstack(
        (
            np.full((1, len(CORE_TENORS)), 3.0),
            3.0 + np.cumsum(changes, axis=0) / 100.0,
        )
    )
    levels = pd.DataFrame(level_values, index=level_dates, columns=CORE_TENORS)
    return change_frame, levels


def test_factor_forecast_beats_no_change_on_predictable_synthetic_data():
    changes, levels = _synthetic_ar_curve()
    config = ForecastConfig(
        minimum_training_weeks=200,
        interval_history_weeks=104,
        minimum_interval_observations=52,
        interval_evaluation_observations=52,
        interval_validation_alpha=0.40,
        interval_bootstrap_replications=2_200,
    )
    result = walk_forward_forecast(changes, levels, config)
    benchmark = result.model_comparison.loc["No-change benchmark", "Selection curve RMSE (bp)"]
    ar_rmse = result.model_comparison.loc["PCA factor AR(1)", "Selection curve RMSE (bp)"]
    assert ar_rmse < benchmark * 0.90
    assert result.selected_model in {"PCA factor AR(1)", "PCA factor VAR(1)"}


def test_future_mutation_does_not_change_earlier_forecasts_or_adoption():
    changes, levels = _synthetic_ar_curve(observations=336)
    config = ForecastConfig(
        minimum_training_weeks=180,
        interval_history_weeks=104,
        minimum_interval_observations=52,
        interval_evaluation_observations=52,
        interval_validation_alpha=0.40,
        interval_bootstrap_replications=2_200,
    )
    baseline = walk_forward_forecast(changes, levels, config)
    mutated = changes.copy()
    mutated.iloc[-20:] += 500.0
    mutated_levels = levels.copy()
    mutated_levels.iloc[1:] = 3.0 + mutated.cumsum().to_numpy() / 100.0
    challenged = walk_forward_forecast(mutated, mutated_levels, config)
    cutoff = changes.index[-21]
    for model in baseline.predictions_bp:
        left = baseline.predictions_bp[model].loc[:cutoff]
        right = challenged.predictions_bp[model].loc[:cutoff]
        assert np.allclose(left, right, atol=1e-12, rtol=0)
    assert baseline.selected_model == challenged.selected_model
    pd.testing.assert_frame_equal(
        baseline.model_comparison,
        challenged.model_comparison,
    )

    partition_counts = baseline.audit_trail["Evaluation partition"].value_counts()
    assert partition_counts.to_dict() == {
        "Selection": 52,
        "Confirmation": 52,
        "Interval evaluation": 52,
    }


def test_real_history_has_no_accepted_directional_edge(curve_bundle):
    config = ForecastConfig()
    result = walk_forward_forecast(
        curve_bundle.weekly_changes_bp,
        curve_bundle.weekly_yields_pct,
        config,
    )
    assert result.selected_model == "No-change benchmark"
    assert (result.audit_trail["Training end"] < result.audit_trail.index).all()
    assert set(result.interval_coverage["Status"]) == {"INCONCLUSIVE"}
    assert set(result.full_history_interval_diagnostic["Status"]) == {"INCONCLUSIVE"}
    assert result.full_history_interval_diagnostic["Coverage gap"].min() < -0.03
    assert np.isnan(result.metrics.loc[("No-change benchmark", "Curve average"), "Direction accuracy"])
    assert result.forecast_as_of == pd.Timestamp("2025-12-26")
    assert result.target_period_end == pd.Timestamp("2026-01-02")
    assert result.simultaneous_interval_diagnostic.loc[0.80, "Joint hit rate of marginal tenor bands"] < 0.60


def test_forecast_rejects_misaligned_or_non_finite_level_history():
    changes, levels = _synthetic_ar_curve(observations=336)
    config = ForecastConfig(
        minimum_training_weeks=180,
        interval_history_weeks=104,
        minimum_interval_observations=52,
        interval_evaluation_observations=52,
        interval_validation_alpha=0.40,
        interval_bootstrap_replications=2_200,
    )
    bad = levels.iloc[1:].copy()
    bad.iloc[-1, 0] = np.nan
    try:
        walk_forward_forecast(changes, bad, config)
    except ValueError as exc:
        assert "missing or non-finite" in str(exc) or "pre-change curve" in str(exc)
    else:
        raise AssertionError("Misaligned/non-finite level history must be rejected.")

    inconsistent = levels.copy()
    inconsistent.iloc[-1] += 1.2345
    try:
        walk_forward_forecast(changes, inconsistent, config)
    except ValueError as exc:
        assert "do not reconcile" in str(exc)
    else:
        raise AssertionError("Level/change mismatch must be rejected.")


def test_forecast_sample_boundary_is_shared_and_fail_fast():
    config = ForecastConfig(
        minimum_training_weeks=180,
        interval_history_weeks=104,
        minimum_interval_observations=104,
        minimum_model_selection_observations=20,
        confirmation_observations=20,
        interval_evaluation_observations=52,
        interval_validation_alpha=0.40,
        interval_bootstrap_replications=2_200,
    )
    assert required_forecast_holdout(config) == 156
    assert required_forecast_observations(config) == 336

    changes, levels = _synthetic_ar_curve(observations=335)
    with pytest.raises(ValueError, match="observed=155, required=156"):
        walk_forward_forecast(changes, levels, config)

    exact_changes, exact_levels = _synthetic_ar_curve(observations=336)
    exact = walk_forward_forecast(exact_changes, exact_levels, config)
    assert len(exact.audit_trail) == 156
