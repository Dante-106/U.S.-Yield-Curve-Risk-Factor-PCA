import numpy as np
import pandas as pd
import pytest

from yield_curve_pca.config import RiskConfig
from yield_curve_pca.risk import (
    _finite_sample_predictive_var,
    _historical_var_es,
    _tail_summary,
    exact_historical_scenarios,
    map_linear_curve_risk,
    optimize_key_rate_hedge,
    rolling_historical_var_backtest,
)


def _illustrative_dv01(tenors):
    return pd.Series(
        [0.0, 0.0, 25_000.0, 100_000.0, 150_000.0, 200_000.0, 150_000.0, 100_000.0, 50_000.0],
        index=tenors,
    )


def test_factor_and_residual_pnl_reconcile(curve_bundle, structural_pca):
    dv01 = _illustrative_dv01(structural_pca.tenors)
    result = map_linear_curve_risk(curve_bundle.weekly_changes_bp, dv01, structural_pca, RiskConfig(), 3)
    explained = result.historical_factor_pnl_usd.sum(axis=1) + result.historical_residual_pnl_usd
    assert np.max(np.abs(result.historical_full_pnl_usd - explained)) < 1e-6
    assert result.variance_reconciliation.loc["Total", "Variance share"] == pytest.approx(1.0)
    assert (result.historical_scenarios["Calendar horizon (days)"] > 0).all()
    assert result.historical_scenarios.index.min() >= curve_bundle.weekly_changes_bp.index[-520]
    assert result.full_history_scenarios.index.min() < result.historical_scenarios.index.min()
    assert result.var_backtest_summary.loc["Rolling historical VaR", "Exceptions"] == 8
    assert result.var_backtest_summary.loc["Rolling historical VaR", "Kupiec p-value"] > 0.05
    assert len(result.var_backtest_detail) == 835


def test_positive_yield_shock_loses_money_for_positive_dv01(structural_pca):
    dv01 = pd.Series(1_000.0, index=structural_pca.tenors)
    parallel_up = pd.Series(1.0, index=structural_pca.tenors)
    pnl = float(-(dv01 @ parallel_up))
    assert pnl == -9_000.0


def test_full_span_hedge_removes_linear_covariance_risk(structural_pca):
    dv01 = _illustrative_dv01(structural_pca.tenors)
    hedge_matrix = pd.DataFrame(
        np.eye(len(structural_pca.tenors)),
        index=[f"Hedge {tenor}" for tenor in structural_pca.tenors],
        columns=structural_pca.tenors,
    )
    result = optimize_key_rate_hedge(
        dv01,
        hedge_matrix,
        structural_pca,
        ridge_penalty=0.0,
        quantity_bounds=(-1_000_000.0, 1_000_000.0),
        maximum_absolute_quantity=1_000_000.0,
    )
    assert np.max(np.abs(result.post_hedge_dv01)) < 1e-5
    assert result.post_variance_usd2 < result.pre_variance_usd2 * 1e-12


def test_ill_conditioned_hedge_set_fails_before_returning_extreme_quantities(structural_pca):
    dv01 = _illustrative_dv01(structural_pca.tenors)
    first = np.ones(len(structural_pca.tenors))
    second = first.copy()
    second[-1] += 1.0e-10
    hedge_matrix = pd.DataFrame(
        [first, second],
        index=["Nearly duplicate A", "Nearly duplicate B"],
        columns=structural_pca.tenors,
    )
    with pytest.raises(ValueError, match="ill-conditioned"):
        optimize_key_rate_hedge(
            dv01,
            hedge_matrix,
            structural_pca,
            quantity_bounds=(-1_000_000.0, 1_000_000.0),
            maximum_absolute_quantity=1_000_000.0,
        )


def test_empirical_expected_shortfall_uses_exact_fractional_tail_mass():
    pnl = pd.Series([-100.0, *([0.0] * 103)])
    value_at_risk, expected_shortfall, tail_mass = _historical_var_es(pnl, 0.975)
    assert value_at_risk == 0.0
    assert tail_mass == pytest.approx(2.6)
    assert expected_shortfall == pytest.approx(100.0 / 2.6)


def test_historical_var_uses_finite_sample_predictive_order_statistic():
    losses = pd.Series(np.arange(520, dtype=float))
    value_at_risk, rank, exception_bound = _finite_sample_predictive_var(losses, 0.99)
    assert rank == 516
    assert value_at_risk == 515.0
    assert exception_bound == pytest.approx(5.0 / 521.0)


@pytest.mark.parametrize(("sample_size", "confidence"), [(52, 0.99), (520, 0.999)])
def test_historical_var_reports_unattainable_finite_sample_target(sample_size, confidence):
    _, _, exception_bound = _finite_sample_predictive_var(np.arange(sample_size, dtype=float), confidence)
    assert exception_bound > 1.0 - confidence


@pytest.mark.parametrize(
    ("review_level", "expected"),
    [
        (19.999, "REVIEW LEVEL MET"),
        (20.0, "REVIEW LEVEL MET"),
        (20.001, "BELOW REVIEW LEVEL"),
    ],
)
def test_tail_mass_review_boundary_is_centralized(review_level, expected):
    dates = pd.date_range("2000-01-07", periods=800, freq="W-FRI")
    pnl = pd.Series(np.linspace(-100.0, 100.0, len(dates)), index=dates)
    summary = _tail_summary(
        pnl,
        RiskConfig(
            historical_window_weeks=800,
            minimum_effective_tail_mass_review=review_level,
        ),
        history_window=800,
    )
    assert summary.loc["Historical ES", "Tail-mass review status"] == expected


def test_full_history_tail_summary_is_not_silently_truncated(curve_bundle, structural_pca):
    result = map_linear_curve_risk(
        curve_bundle.weekly_changes_bp,
        _illustrative_dv01(structural_pca.tenors),
        structural_pca,
        RiskConfig(),
        3,
    )
    assert result.tail_risk_summary["History weeks"].eq(520).all()
    assert (
        result.full_history_tail_risk_summary["History weeks"].eq(len(curve_bundle.weekly_changes_bp)).all()
    )


def test_variance_reconciliation_is_stable_at_full_rank(curve_bundle, structural_pca):
    result = map_linear_curve_risk(
        curve_bundle.weekly_changes_bp,
        _illustrative_dv01(structural_pca.tenors) * 1.0e6,
        structural_pca,
        RiskConfig(),
        len(structural_pca.tenors),
    )
    assert result.variance_reconciliation.loc["Residual key-rate risk", "Variance (USD²)"] == pytest.approx(
        0.0, abs=1.0e-6
    )


@pytest.mark.parametrize("bad_ridge", [np.nan, np.inf, True])
def test_hedge_optimizer_rejects_non_finite_or_boolean_ridge(structural_pca, bad_ridge):
    dv01 = _illustrative_dv01(structural_pca.tenors)
    hedge_matrix = pd.DataFrame(
        [np.ones(len(structural_pca.tenors))],
        index=["Hedge"],
        columns=structural_pca.tenors,
    )
    with pytest.raises(ValueError, match="ridge_penalty"):
        optimize_key_rate_hedge(
            dv01,
            hedge_matrix,
            structural_pca,
            ridge_penalty=bad_ridge,
            quantity_bounds=(-1_000_000.0, 1_000_000.0),
            maximum_absolute_quantity=1_000_000.0,
        )


def test_hedge_optimizer_requires_finite_operational_quantity_controls(structural_pca):
    dv01 = _illustrative_dv01(structural_pca.tenors)
    hedge_matrix = pd.DataFrame(
        [np.ones(len(structural_pca.tenors))],
        index=["Hedge"],
        columns=structural_pca.tenors,
    )
    with pytest.raises(TypeError, match="required keyword-only"):
        optimize_key_rate_hedge(dv01, hedge_matrix, structural_pca)


def test_hedge_optimizer_rejects_redundant_and_tiny_hedges(structural_pca):
    dv01 = _illustrative_dv01(structural_pca.tenors)
    duplicate = pd.DataFrame(
        [np.ones(len(structural_pca.tenors))] * 2,
        index=["A", "B"],
        columns=structural_pca.tenors,
    )
    controls = {
        "quantity_bounds": (-1_000_000.0, 1_000_000.0),
        "maximum_absolute_quantity": 1_000_000.0,
    }
    with pytest.raises(ValueError, match="redundant|underdetermined"):
        optimize_key_rate_hedge(dv01, duplicate, structural_pca, **controls)

    tiny = duplicate.iloc[[0]] * 1.0e-12
    with pytest.raises(ValueError, match="risk-scale floor"):
        optimize_key_rate_hedge(dv01, tiny, structural_pca, **controls)


def test_hedge_optimizer_rejects_zero_pre_hedge_risk_with_finite_result_contract(
    structural_pca,
):
    zero_dv01 = pd.Series(0.0, index=structural_pca.tenors)
    hedge_matrix = pd.DataFrame(
        np.eye(len(structural_pca.tenors)),
        index=[f"Hedge {tenor}" for tenor in structural_pca.tenors],
        columns=structural_pca.tenors,
    )
    with pytest.raises(ValueError, match="Pre-hedge portfolio.*risk-scale floor"):
        optimize_key_rate_hedge(
            zero_dv01,
            hedge_matrix,
            structural_pca,
            quantity_bounds=(-1_000_000.0, 1_000_000.0),
            maximum_absolute_quantity=1_000_000.0,
            minimum_variance_reduction=0.5,
        )


def test_var_backtest_cannot_pass_with_an_undersized_exception_sample():
    dates = pd.date_range("2020-01-03", periods=53, freq="W-FRI")
    pnl = pd.Series(np.linspace(-10.0, 10.0, len(dates)), index=dates)
    summary, detail = rolling_historical_var_backtest(
        pnl,
        history_window=52,
        confidence=0.975,
        minimum_backtest_observations=100,
    )
    assert len(detail) == 1
    assert summary.loc["Rolling historical VaR", "Status"] == "NOT EVALUATED"
    assert summary.loc["Rolling historical VaR", "Initial estimation observations"] == 52
    assert np.isnan(summary.loc["Rolling historical VaR", "Kupiec p-value"])


def test_var_backtest_cannot_evaluate_unattainable_confidence_target():
    dates = pd.date_range("2020-01-03", periods=352, freq="W-FRI")
    pnl = pd.Series(np.linspace(-10.0, 10.0, len(dates)), index=dates)
    summary, _ = rolling_historical_var_backtest(
        pnl,
        history_window=52,
        confidence=0.99,
        minimum_backtest_observations=250,
    )
    assert summary.loc["Rolling historical VaR", "Status"] == "NOT EVALUATED"
    assert not bool(summary.loc["Rolling historical VaR", "Coverage target achievable"])


@pytest.mark.parametrize("future_losses", [np.zeros(300), np.arange(104.0, 404.0)])
def test_var_backtest_handles_zero_and_all_exception_boundaries(future_losses):
    losses = np.concatenate([np.arange(104.0), future_losses])
    dates = pd.date_range("2000-01-07", periods=len(losses), freq="W-FRI")
    summary, detail = rolling_historical_var_backtest(
        pd.Series(-losses, index=dates),
        history_window=104,
        confidence=0.95,
        minimum_backtest_observations=250,
    )
    row = summary.loc["Rolling historical VaR"]
    assert row["Status"] == "WARN"
    assert np.isfinite(row["Kupiec p-value"])
    assert np.isfinite(row["Conditional coverage p-value"])
    assert detail["Exception"].sum() in {0, 300}


@pytest.mark.parametrize("bad_kind", ["not-frame", "duplicate", "nan", "string"])
def test_exact_historical_scenarios_rejects_invalid_direct_inputs(bad_kind):
    dates = pd.date_range("2020-01-03", periods=60, freq="W-FRI")
    changes = pd.DataFrame({"2Y": np.arange(60.0), "10Y": np.arange(60.0)}, index=dates)
    dv01 = pd.Series({"2Y": 1_000.0, "10Y": 2_000.0})
    if bad_kind == "not-frame":
        bad = changes.to_numpy()
    elif bad_kind == "duplicate":
        bad = changes.copy()
        bad.columns = ["2Y", "2Y"]
    elif bad_kind == "nan":
        bad = changes.copy()
        bad.iloc[0, 0] = np.nan
    else:
        bad = changes.astype(object)
        bad.iloc[0, 0] = "bad"
    with pytest.raises((TypeError, ValueError)):
        exact_historical_scenarios(bad, dv01)


def test_var_backtest_detects_clustered_exceptions_at_plausible_frequency():
    history_window = 104
    test_observations = 300
    losses = np.zeros(history_window + test_observations)
    for start in (history_window, history_window + 110, history_window + 220):
        losses[start : start + 5] = np.arange(100.0, 105.0)
    dates = pd.date_range("2000-01-07", periods=len(losses), freq="W-FRI")
    summary, detail = rolling_historical_var_backtest(
        pd.Series(-losses, index=dates),
        history_window=history_window,
        confidence=0.95,
        minimum_backtest_observations=250,
    )
    row = summary.loc["Rolling historical VaR"]
    assert detail["Exception"].sum() == 15
    assert row["Kupiec p-value"] > 0.05
    assert row["Christoffersen independence p-value"] < 0.05
    assert row["Conditional coverage p-value"] < 0.05
    assert row["Status"] == "WARN"
