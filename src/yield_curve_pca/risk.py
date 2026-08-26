"""Linear key-rate risk mapping, historical tails, stress, and hedge diagnostics.

The sign convention is explicit: input DV01 is a positive price gain for a
one-basis-point fall in yield.  Therefore linear P&L is ``-DV01 @ shock_bp``.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np
import pandas as pd
from scipy.optimize import lsq_linear
from scipy.special import xlogy
from scipy.stats import chi2

from .config import RiskConfig
from .pca import PCAFit


@dataclass(frozen=True)
class RiskMappingResult:
    key_rate_dv01_usd_per_bp: pd.Series
    factor_exposure_usd_per_score: pd.Series
    historical_full_pnl_usd: pd.Series
    historical_factor_pnl_usd: pd.DataFrame
    historical_residual_pnl_usd: pd.Series
    tail_risk_summary: pd.DataFrame
    full_history_tail_risk_summary: pd.DataFrame
    variance_reconciliation: pd.DataFrame
    pure_factor_scenarios: pd.DataFrame
    historical_scenarios: pd.DataFrame
    full_history_scenarios: pd.DataFrame
    var_backtest_summary: pd.DataFrame
    var_backtest_detail: pd.DataFrame


@dataclass(frozen=True)
class HedgeResult:
    quantities: pd.Series
    pre_hedge_dv01: pd.Series
    post_hedge_dv01: pd.Series
    pre_factor_exposure: pd.Series
    post_factor_exposure: pd.Series
    pre_variance_usd2: float
    post_variance_usd2: float
    variance_reduction: float
    design_condition_number: float
    regularization_usd2_per_quantity2: float
    solver_status: str


def _validate_dv01(dv01: pd.Series, tenors: tuple[str, ...]) -> pd.Series:
    if not isinstance(dv01, pd.Series):
        raise TypeError("key-rate DV01 must be a pandas Series indexed by tenor.")
    if dv01.index.duplicated().any() or set(dv01.index) != set(tenors):
        raise ValueError(f"key-rate DV01 must contain exactly these tenors: {tenors}.")
    ordered = dv01.reindex(tenors).astype(float)
    if not np.isfinite(ordered.to_numpy()).all():
        raise ValueError("key-rate DV01 contains missing or non-finite values.")
    return ordered


def _validate_retained_factors(retained_factors: int, fit: PCAFit) -> None:
    if (
        not isinstance(retained_factors, int)
        or isinstance(retained_factors, bool)
        or not 1 <= retained_factors <= len(fit.tenors)
    ):
        raise ValueError("retained_factors must be an integer within the fitted dimension.")


def _finite_real(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and bool(np.isfinite(value))


def factor_exposure_from_dv01(
    key_rate_dv01_usd_per_bp: pd.Series,
    fit: PCAFit,
    retained_factors: int = 3,
) -> pd.Series:
    dv01 = _validate_dv01(key_rate_dv01_usd_per_bp, fit.tenors)
    _validate_retained_factors(retained_factors, fit)
    exposure = -(fit.shock_basis_bp_per_score[:retained_factors] @ dv01.to_numpy())
    return pd.Series(exposure, index=fit.component_names[:retained_factors], name="USD per factor-score unit")


def _historical_var_es(pnl: pd.Series, confidence: float) -> tuple[float, float, float]:
    clean = pnl.dropna().astype(float)
    if len(clean) < 52:
        raise ValueError("At least 52 P&L observations are required for historical tail metrics.")
    losses = -clean
    value_at_risk, _, _ = _finite_sample_predictive_var(losses, confidence)
    ordered = np.sort(losses.to_numpy())[::-1]
    tail_mass = len(ordered) * (1.0 - confidence)
    whole_observations = int(np.floor(tail_mass))
    fractional_observation = tail_mass - whole_observations
    numerator = float(ordered[:whole_observations].sum())
    if fractional_observation > np.finfo(float).eps:
        numerator += fractional_observation * float(ordered[whole_observations])
    expected_shortfall = numerator / tail_mass
    return value_at_risk, float(expected_shortfall), float(tail_mass)


def _finite_sample_predictive_var(
    losses: pd.Series | np.ndarray,
    confidence: float,
) -> tuple[float, int, float]:
    """Return a conservative next-observation historical VaR order statistic.

    The one-based ascending rank is ``ceil((n + 1) * confidence)``, capped at
    ``n``.  Under an exchangeable continuous distribution, the strict
    exceedance probability is ``(n + 1 - rank) / (n + 1)``; ties can only make
    the strict-exceedance rate smaller.  The reported probability is therefore
    also the finite-sample upper bound used by the diagnostic coverage test.
    """

    values = np.asarray(losses, dtype=float)
    if values.ndim != 1 or values.size < 1 or not np.isfinite(values).all():
        raise ValueError("Historical losses must be a non-empty, finite one-dimensional sample.")
    if not _finite_real(confidence) or not 0.5 < confidence < 1.0:
        raise ValueError("Historical VaR confidence must lie in (0.5, 1).")
    rank = min(values.size, int(np.ceil((values.size + 1) * float(confidence))))
    value_at_risk = float(np.partition(values, rank - 1)[rank - 1])
    exception_bound = float((values.size + 1 - rank) / (values.size + 1))
    return value_at_risk, rank, exception_bound


def _tail_summary(
    pnl: pd.Series,
    config: RiskConfig,
    *,
    history_window: int | None,
) -> pd.DataFrame:
    window = pnl if history_window is None else pnl.iloc[-history_window:]
    window_scope = "Full available history" if history_window is None else f"Latest {history_window} weeks"
    rows: list[dict[str, object]] = []
    for label, confidence in (
        ("Historical VaR", config.var_confidence),
        ("Historical ES", config.es_confidence),
    ):
        value_at_risk, expected_shortfall, effective_tail_mass = _historical_var_es(window, confidence)
        _, order_statistic_rank, exception_bound = _finite_sample_predictive_var(-window, confidence)
        target_achievable = exception_bound <= (1.0 - confidence) + np.finfo(float).eps
        tail_status = (
            "REVIEW LEVEL MET"
            if effective_tail_mass >= config.minimum_effective_tail_mass_review
            else "BELOW REVIEW LEVEL"
        )
        rows.append(
            {
                "Measure": label,
                "Confidence": confidence,
                "Loss (USD)": value_at_risk if label == "Historical VaR" else expected_shortfall,
                "Effective tail mass": effective_tail_mass,
                "Tail-mass review threshold": config.minimum_effective_tail_mass_review,
                "Tail-mass review status": tail_status,
                "Order-statistic rank": (order_statistic_rank if label == "Historical VaR" else pd.NA),
                "Finite-sample exception bound": (exception_bound if label == "Historical VaR" else np.nan),
                "Coverage target achievable": (target_achievable if label == "Historical VaR" else pd.NA),
                "Estimator": (
                    "finite-sample predictive order statistic"
                    if label == "Historical VaR"
                    else "fractional empirical upper-tail average"
                ),
                "History weeks": len(window),
                "Window scope": window_scope,
                "Window start": window.index.min(),
                "Window end": window.index.max(),
            }
        )
    return pd.DataFrame(rows).set_index("Measure")


def _model_covariance_bp2(fit: PCAFit) -> np.ndarray:
    basis = fit.shock_basis_bp_per_score
    return basis.T @ np.diag(fit.eigenvalues) @ basis


def _variance_reconciliation(dv01: pd.Series, fit: PCAFit, retained_factors: int) -> pd.DataFrame:
    vector = dv01.to_numpy()
    covariance = _model_covariance_bp2(fit)
    full_variance = float(vector @ covariance @ vector)
    all_exposures = -(fit.shock_basis_bp_per_score @ vector)
    factor_variances = all_exposures[:retained_factors] ** 2 * fit.eigenvalues[:retained_factors]
    residual_variance = float(
        np.sum(all_exposures[retained_factors:] ** 2 * fit.eigenvalues[retained_factors:])
    )
    reconciled_variance = float(factor_variances.sum()) + residual_variance
    tolerance = 1.0e-6 + 1.0e-10 * abs(full_variance)
    if abs(full_variance - reconciled_variance) > tolerance:
        raise RuntimeError("PCA factor/residual variance does not reconcile to the physical covariance.")
    rows: list[dict[str, object]] = []
    for index, variance in enumerate(factor_variances):
        rows.append(
            {
                "Risk source": fit.component_names[index],
                "Variance (USD²)": float(variance),
                "Variance share": float(variance / full_variance) if full_variance > 0 else np.nan,
            }
        )
    rows.append(
        {
            "Risk source": "Residual key-rate risk",
            "Variance (USD²)": residual_variance,
            "Variance share": float(residual_variance / full_variance) if full_variance > 0 else np.nan,
        }
    )
    rows.append(
        {
            "Risk source": "Total",
            "Variance (USD²)": full_variance,
            "Variance share": 1.0 if full_variance > 0 else np.nan,
        }
    )
    return pd.DataFrame(rows).set_index("Risk source")


def pure_factor_scenarios(
    key_rate_dv01_usd_per_bp: pd.Series,
    fit: PCAFit,
    config: RiskConfig,
    retained_factors: int = 3,
) -> pd.DataFrame:
    dv01 = _validate_dv01(key_rate_dv01_usd_per_bp, fit.tenors)
    _validate_retained_factors(retained_factors, fit)
    rows: list[dict[str, object]] = []
    for factor_index in range(retained_factors):
        sigma_shock = fit.one_sigma_shocks_bp[factor_index]
        for multiple in config.factor_sigma_multiples:
            for sign in (-1.0, 1.0):
                shock = sign * multiple * sigma_shock
                row: dict[str, object] = {
                    "Scenario": f"{fit.component_names[factor_index]} {sign * multiple:+.0f}σ",
                    "Factor": fit.component_names[factor_index],
                    "Sigma multiple": sign * multiple,
                    "Linear P&L (USD)": float(-(dv01.to_numpy() @ shock)),
                }
                row.update(
                    {
                        f"{tenor} shock (bp)": float(value)
                        for tenor, value in zip(fit.tenors, shock, strict=True)
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows).set_index("Scenario")


def exact_historical_scenarios(
    changes_bp: pd.DataFrame,
    key_rate_dv01_usd_per_bp: pd.Series,
    *,
    worst_count: int = 10,
    history_window: int | None = None,
) -> pd.DataFrame:
    if not isinstance(changes_bp, pd.DataFrame) or changes_bp.empty:
        raise TypeError("Historical changes must be a non-empty pandas DataFrame.")
    if changes_bp.columns.duplicated().any():
        raise ValueError("Historical changes must have unique tenor columns.")
    try:
        changes = changes_bp.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Historical changes must be numeric.") from exc
    if not np.isfinite(changes.to_numpy()).all():
        raise ValueError("Historical changes contain missing or non-finite values.")
    dv01 = _validate_dv01(key_rate_dv01_usd_per_bp, tuple(changes_bp.columns))
    if not isinstance(changes_bp.index, pd.DatetimeIndex):
        raise ValueError("Historical changes must use a DatetimeIndex.")
    if changes_bp.index.duplicated().any() or not changes_bp.index.is_monotonic_increasing:
        raise ValueError("Historical changes must have a unique, ascending DatetimeIndex.")
    if not isinstance(worst_count, int) or isinstance(worst_count, bool) or worst_count < 1:
        raise ValueError("worst_count must be a positive integer.")
    if history_window is not None and (
        not isinstance(history_window, int) or isinstance(history_window, bool) or history_window < 52
    ):
        raise ValueError("history_window must be an integer of at least 52 when supplied.")
    pnl = -(changes @ dv01)
    eligible_pnl = pnl if history_window is None else pnl.iloc[-history_window:]
    selected_dates = eligible_pnl.nsmallest(min(worst_count, len(eligible_pnl))).index
    rows: list[dict[str, object]] = []
    for date in selected_dates:
        position = changes.index.get_loc(date)
        shock = changes.loc[date]
        row: dict[str, object] = {
            "Observation date": date,
            "Previous observation date": (changes.index[position - 1] if position > 0 else pd.NaT),
            "Calendar horizon (days)": (
                (date - changes.index[position - 1]).days if position > 0 else np.nan
            ),
            "Linear P&L (USD)": float(pnl.loc[date]),
        }
        row.update({f"{tenor} shock (bp)": float(shock[tenor]) for tenor in changes.columns})
        rows.append(row)
    return pd.DataFrame(rows).set_index("Observation date")


def rolling_historical_var_backtest(
    pnl_usd: pd.Series,
    *,
    history_window: int,
    confidence: float,
    test_alpha: float = 0.05,
    minimum_backtest_observations: int = 250,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Diagnose one-step fixed-KRD historical VaR with coverage controls."""

    if not isinstance(pnl_usd.index, pd.DatetimeIndex):
        raise ValueError("VaR backtest P&L must use a DatetimeIndex.")
    if pnl_usd.index.duplicated().any() or not pnl_usd.index.is_monotonic_increasing:
        raise ValueError("VaR backtest P&L must have a unique, ascending DatetimeIndex.")
    pnl = pnl_usd.astype(float)
    if not np.isfinite(pnl.to_numpy()).all():
        raise ValueError("VaR backtest P&L contains missing or non-finite values.")
    if len(pnl) < 52:
        raise ValueError("VaR backtest requires at least 52 finite P&L observations.")
    if not isinstance(history_window, int) or isinstance(history_window, bool) or history_window < 52:
        raise ValueError("VaR backtest history_window must be an integer of at least 52.")
    if not _finite_real(confidence) or not 0.5 < confidence < 1.0:
        raise ValueError("VaR backtest confidence must lie in (0.5, 1).")
    if not _finite_real(test_alpha) or not 0.0 < test_alpha < 0.5:
        raise ValueError("VaR backtest test_alpha must lie in (0, 0.5).")
    if (
        not isinstance(minimum_backtest_observations, int)
        or isinstance(minimum_backtest_observations, bool)
        or minimum_backtest_observations < 100
    ):
        raise ValueError("minimum_backtest_observations must be an integer of at least 100.")

    losses = -pnl
    initial_estimation_observations = min(len(losses), history_window)
    _, order_statistic_rank, exception_bound = _finite_sample_predictive_var(
        losses.iloc[:initial_estimation_observations], confidence
    )
    nominal_exception_probability = 1.0 - confidence
    rows: list[dict[str, object]] = []
    for position in range(history_window, len(losses)):
        history = losses.iloc[position - history_window : position]
        var, rank, window_exception_bound = _finite_sample_predictive_var(history, confidence)
        if rank != order_statistic_rank or not np.isclose(window_exception_bound, exception_bound):
            raise RuntimeError("Fixed-window historical VaR order-statistic controls drifted.")
        realized = float(losses.iloc[position])
        rows.append(
            {
                "Observation date": losses.index[position],
                "Estimation start": history.index.min(),
                "Estimation end": history.index.max(),
                "Historical VaR (USD)": var,
                "Realized loss (USD)": realized,
                "Exception": realized > var,
                "VaR order-statistic rank": rank,
                "Finite-sample exception bound": window_exception_bound,
                "Nominal exception probability": nominal_exception_probability,
                "Quantile convention": "ceil((n+1)*confidence), capped at n",
            }
        )
    detail = pd.DataFrame(
        rows,
        columns=(
            "Observation date",
            "Estimation start",
            "Estimation end",
            "Historical VaR (USD)",
            "Realized loss (USD)",
            "Exception",
            "VaR order-statistic rank",
            "Finite-sample exception bound",
            "Nominal exception probability",
            "Quantile convention",
        ),
    ).set_index("Observation date")
    exceptions = detail["Exception"].astype(int).to_numpy()
    observations = len(exceptions)
    exception_count = int(exceptions.sum())
    expected_probability = exception_bound
    target_achievable = expected_probability <= (nominal_exception_probability + np.finfo(float).eps)
    observed_probability = exception_count / observations if observations else float("nan")

    if observations < minimum_backtest_observations or not target_achievable:
        summary = pd.DataFrame(
            {
                "History window": [history_window],
                "Initial estimation observations": [initial_estimation_observations],
                "Confidence": [confidence],
                "Backtest observations": [observations],
                "Minimum backtest observations": [minimum_backtest_observations],
                "Exceptions": [exception_count],
                "Expected exceptions": [observations * expected_probability],
                "Nominal expected exceptions": [observations * nominal_exception_probability],
                "Finite-sample exception bound": [expected_probability],
                "Nominal exception probability": [nominal_exception_probability],
                "VaR order-statistic rank": [order_statistic_rank],
                "Coverage target achievable": [target_achievable],
                "Quantile convention": ["ceil((n+1)*confidence), capped at n"],
                "Observed exception rate": [observed_probability],
                "Kupiec LR": [np.nan],
                "Kupiec p-value": [np.nan],
                "Christoffersen independence LR": [np.nan],
                "Christoffersen independence p-value": [np.nan],
                "Conditional coverage LR": [np.nan],
                "Conditional coverage p-value": [np.nan],
                "Test alpha": [test_alpha],
                "Evaluation note": [
                    "Insufficient backtest observations"
                    if observations < minimum_backtest_observations
                    else "Configured confidence is unattainable at this history-window size"
                ],
                "Status": ["NOT EVALUATED"],
            },
            index=pd.Index(["Rolling historical VaR"], name="Backtest"),
        )
        return summary, detail

    log_null = float(
        xlogy(exception_count, expected_probability)
        + xlogy(observations - exception_count, 1.0 - expected_probability)
    )
    log_unrestricted = float(
        xlogy(exception_count, observed_probability)
        + xlogy(observations - exception_count, 1.0 - observed_probability)
    )
    kupiec_lr = max(0.0, -2.0 * (log_null - log_unrestricted))
    kupiec_p = float(chi2.sf(kupiec_lr, 1))

    previous = exceptions[:-1]
    current = exceptions[1:]
    n00 = int(np.sum((previous == 0) & (current == 0)))
    n01 = int(np.sum((previous == 0) & (current == 1)))
    n10 = int(np.sum((previous == 1) & (current == 0)))
    n11 = int(np.sum((previous == 1) & (current == 1)))
    pi01 = n01 / (n00 + n01) if n00 + n01 else 0.0
    pi11 = n11 / (n10 + n11) if n10 + n11 else 0.0
    transition_probability = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    independent_log_likelihood = float(
        xlogy(n00 + n10, 1.0 - transition_probability) + xlogy(n01 + n11, transition_probability)
    )
    markov_log_likelihood = float(
        xlogy(n00, 1.0 - pi01) + xlogy(n01, pi01) + xlogy(n10, 1.0 - pi11) + xlogy(n11, pi11)
    )
    independence_lr = max(0.0, -2.0 * (independent_log_likelihood - markov_log_likelihood))
    independence_p = float(chi2.sf(independence_lr, 1))
    conditional_coverage_lr = kupiec_lr + independence_lr
    conditional_coverage_p = float(chi2.sf(conditional_coverage_lr, 2))
    summary = pd.DataFrame(
        {
            "History window": [history_window],
            "Initial estimation observations": [initial_estimation_observations],
            "Confidence": [confidence],
            "Backtest observations": [observations],
            "Minimum backtest observations": [minimum_backtest_observations],
            "Exceptions": [exception_count],
            "Expected exceptions": [observations * expected_probability],
            "Nominal expected exceptions": [observations * nominal_exception_probability],
            "Finite-sample exception bound": [expected_probability],
            "Nominal exception probability": [nominal_exception_probability],
            "VaR order-statistic rank": [order_statistic_rank],
            "Coverage target achievable": [target_achievable],
            "Quantile convention": ["ceil((n+1)*confidence), capped at n"],
            "Observed exception rate": [observed_probability],
            "Kupiec LR": [kupiec_lr],
            "Kupiec p-value": [kupiec_p],
            "Christoffersen independence LR": [independence_lr],
            "Christoffersen independence p-value": [independence_p],
            "Conditional coverage LR": [conditional_coverage_lr],
            "Conditional coverage p-value": [conditional_coverage_p],
            "Test alpha": [test_alpha],
            "Evaluation note": [
                "Kupiec null uses the finite-sample continuous-case exception bound; "
                "serial dependence is challenged separately"
            ],
            "Status": [
                "PASS" if min(kupiec_p, independence_p, conditional_coverage_p) >= test_alpha else "WARN"
            ],
        },
        index=pd.Index(["Rolling historical VaR"], name="Backtest"),
    )
    return summary, detail


def map_linear_curve_risk(
    changes_bp: pd.DataFrame,
    key_rate_dv01_usd_per_bp: pd.Series,
    fit: PCAFit,
    config: RiskConfig,
    retained_factors: int = 3,
) -> RiskMappingResult:
    """Map a key-rate DV01 profile to PCA factors and residual curve risk."""

    if not isinstance(changes_bp.index, pd.DatetimeIndex):
        raise ValueError("Risk history must use a DatetimeIndex.")
    if changes_bp.index.duplicated().any() or not changes_bp.index.is_monotonic_increasing:
        raise ValueError("Risk history must have a unique, ascending DatetimeIndex.")
    if tuple(changes_bp.columns) != fit.tenors:
        raise ValueError("Risk-history tenor order must match the PCA calibration.")
    _validate_retained_factors(retained_factors, fit)
    dv01 = _validate_dv01(key_rate_dv01_usd_per_bp, fit.tenors)
    score_frame = fit.transform(changes_bp).iloc[:, :retained_factors]
    exposure = factor_exposure_from_dv01(dv01, fit, retained_factors)
    factor_pnl = score_frame.mul(exposure, axis=1)
    mean_pnl = float(-(fit.center_bp @ dv01.to_numpy()))
    factor_pnl.insert(0, "Mean change", mean_pnl)
    full_pnl = -(changes_bp @ dv01)
    full_pnl.name = "Full key-rate linear P&L (USD)"
    approximated = factor_pnl.sum(axis=1)
    residual_pnl = full_pnl - approximated
    residual_pnl.name = "Residual key-rate P&L (USD)"
    var_backtest_summary, var_backtest_detail = rolling_historical_var_backtest(
        full_pnl,
        history_window=config.historical_window_weeks,
        confidence=config.var_confidence,
        test_alpha=config.backtest_alpha,
        minimum_backtest_observations=config.minimum_var_backtest_observations,
    )

    return RiskMappingResult(
        key_rate_dv01_usd_per_bp=dv01,
        factor_exposure_usd_per_score=exposure,
        historical_full_pnl_usd=full_pnl,
        historical_factor_pnl_usd=factor_pnl,
        historical_residual_pnl_usd=residual_pnl,
        tail_risk_summary=_tail_summary(full_pnl, config, history_window=config.historical_window_weeks),
        full_history_tail_risk_summary=_tail_summary(full_pnl, config, history_window=None),
        variance_reconciliation=_variance_reconciliation(dv01, fit, retained_factors),
        pure_factor_scenarios=pure_factor_scenarios(dv01, fit, config, retained_factors),
        historical_scenarios=exact_historical_scenarios(
            changes_bp,
            dv01,
            history_window=config.historical_window_weeks,
        ),
        full_history_scenarios=exact_historical_scenarios(changes_bp, dv01),
        var_backtest_summary=var_backtest_summary,
        var_backtest_detail=var_backtest_detail,
    )


def optimize_key_rate_hedge(
    key_rate_dv01_usd_per_bp: pd.Series,
    hedge_key_rate_dv01: pd.DataFrame,
    fit: PCAFit,
    *,
    quantity_bounds: tuple[float | np.ndarray, float | np.ndarray],
    maximum_absolute_quantity: float,
    ridge_penalty: float = 1.0e-8,
    retained_factors: int = 3,
    maximum_condition_number: float = 1.0e8,
    minimum_variance_reduction: float = 0.0,
    minimum_standalone_volatility_usd: float = 1.0,
    minimum_pre_hedge_volatility_usd: float = 1.0,
) -> HedgeResult:
    """Minimize total PCA covariance risk, including omitted-factor residual risk.

    ``ridge_penalty`` is dimensionless and scales the median standalone modeled
    variance of the candidate hedges. Quantity, portfolio/candidate risk-scale,
    improvement, and conditioning gates prevent a numerically optimal but
    operationally meaningless hedge from being returned.
    """

    dv01 = _validate_dv01(key_rate_dv01_usd_per_bp, fit.tenors)
    if not isinstance(hedge_key_rate_dv01, pd.DataFrame) or hedge_key_rate_dv01.empty:
        raise ValueError("hedge_key_rate_dv01 must be a non-empty instrument-by-tenor DataFrame.")
    if (
        hedge_key_rate_dv01.index.duplicated().any()
        or hedge_key_rate_dv01.columns.duplicated().any()
        or set(hedge_key_rate_dv01.columns) != set(fit.tenors)
    ):
        raise ValueError("Hedge rows must be unique and columns must match the fitted tenors.")
    try:
        hedge_matrix = hedge_key_rate_dv01.reindex(columns=fit.tenors).astype(float).to_numpy()
    except (TypeError, ValueError) as exc:
        raise ValueError("Hedge key-rate DV01 matrix must be numeric.") from exc
    if not np.isfinite(hedge_matrix).all():
        raise ValueError("Hedge key-rate DV01 matrix contains missing or non-finite values.")
    _validate_retained_factors(retained_factors, fit)
    if not _finite_real(ridge_penalty) or ridge_penalty < 0:
        raise ValueError("ridge_penalty must be finite, non-boolean, and non-negative.")
    if not _finite_real(maximum_condition_number) or maximum_condition_number <= 1.0:
        raise ValueError("maximum_condition_number must be finite and greater than one.")
    if not _finite_real(maximum_absolute_quantity) or maximum_absolute_quantity <= 0:
        raise ValueError("maximum_absolute_quantity is required and must be positive and finite.")
    if not _finite_real(minimum_variance_reduction) or not 0.0 <= minimum_variance_reduction <= 1.0:
        raise ValueError("minimum_variance_reduction must lie in [0, 1].")
    if not _finite_real(minimum_standalone_volatility_usd) or minimum_standalone_volatility_usd <= 0:
        raise ValueError("minimum_standalone_volatility_usd must be positive and finite.")
    if not _finite_real(minimum_pre_hedge_volatility_usd) or minimum_pre_hedge_volatility_usd <= 0:
        raise ValueError("minimum_pre_hedge_volatility_usd must be positive and finite.")
    if quantity_bounds is None or not isinstance(quantity_bounds, tuple) or len(quantity_bounds) != 2:
        raise ValueError("Explicit finite lower and upper quantity_bounds are required.")
    try:
        lower = np.broadcast_to(np.asarray(quantity_bounds[0], dtype=float), len(hedge_matrix)).copy()
        upper = np.broadcast_to(np.asarray(quantity_bounds[1], dtype=float), len(hedge_matrix)).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("quantity_bounds must broadcast to one bound per hedge instrument.") from exc
    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower >= upper):
        raise ValueError("quantity_bounds must be finite and strictly ordered.")
    lower = np.maximum(lower, -float(maximum_absolute_quantity))
    upper = np.minimum(upper, float(maximum_absolute_quantity))
    if np.any(lower >= upper):
        raise ValueError("maximum_absolute_quantity is incompatible with quantity_bounds.")

    covariance = _model_covariance_bp2(fit)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    covariance_spectral_norm = float(np.max(np.abs(eigenvalues)))
    eigenvalues = np.clip(eigenvalues, 0.0, None)
    dv01_vector = dv01.to_numpy()
    pre_variance_raw = float(dv01_vector @ covariance @ dv01_vector)
    pre_variance_tolerance = 1.0e-12 * max(
        1.0,
        float(covariance_spectral_norm * (dv01_vector @ dv01_vector)),
    )
    if pre_variance_raw < -pre_variance_tolerance:
        raise RuntimeError("Pre-hedge modeled variance is materially negative.")
    pre_variance = max(0.0, pre_variance_raw)
    pre_volatility = float(np.sqrt(pre_variance))
    if pre_volatility < minimum_pre_hedge_volatility_usd:
        raise ValueError(
            "Pre-hedge portfolio falls below the modeled risk-scale floor: "
            f"volatility={pre_volatility:.6g} USD, "
            f"floor={minimum_pre_hedge_volatility_usd:.6g} USD."
        )
    square_root = np.diag(np.sqrt(eigenvalues)) @ eigenvectors.T
    raw_design = square_root @ hedge_matrix.T
    standalone_volatility = np.linalg.norm(raw_design, axis=0)
    if np.any(standalone_volatility < minimum_standalone_volatility_usd):
        bad = hedge_key_rate_dv01.index[standalone_volatility < minimum_standalone_volatility_usd].tolist()
        raise ValueError(f"Hedge instruments fall below the standalone risk-scale floor: {bad}.")
    if np.linalg.matrix_rank(raw_design) < raw_design.shape[1]:
        raise ValueError("Hedge design has redundant or underdetermined instrument columns.")
    condition_number = float(np.linalg.cond(raw_design))
    if not np.isfinite(condition_number) or condition_number > maximum_condition_number:
        raise ValueError(
            "Hedge design is rank deficient or ill-conditioned: "
            f"condition_number={condition_number:.3g}, limit={maximum_condition_number:.3g}."
        )
    design = raw_design
    target = -(square_root @ dv01.to_numpy())
    if ridge_penalty > 0:
        standalone_variance = np.sum(raw_design**2, axis=0)
        positive_variance = standalone_variance[standalone_variance > np.finfo(float).eps]
        regularization_scale = float(np.median(positive_variance)) if len(positive_variance) else 0.0
        absolute_regularization = ridge_penalty * regularization_scale
        design = np.vstack((design, np.sqrt(absolute_regularization) * np.eye(len(hedge_matrix))))
        target = np.concatenate((target, np.zeros(len(hedge_matrix))))
    else:
        absolute_regularization = 0.0
    solution = lsq_linear(design, target, bounds=(lower, upper), method="trf", lsmr_tol="auto")
    if not solution.success or not np.isfinite(solution.x).all():
        raise RuntimeError(f"Key-rate hedge optimization failed: {solution.message}")
    if np.max(np.abs(solution.x)) > maximum_absolute_quantity * (1.0 + 1.0e-12):
        raise ValueError(
            "Hedge solution exceeds maximum_absolute_quantity; tighten bounds or change the hedge set."
        )
    quantities = pd.Series(solution.x, index=hedge_key_rate_dv01.index, name="Hedge quantity")
    post_dv01 = dv01 + hedge_key_rate_dv01.reindex(columns=fit.tenors).T @ quantities
    post_dv01.name = "Post-hedge DV01 (USD/bp)"

    post_vector = post_dv01.to_numpy()
    post_variance_raw = float(post_vector @ covariance @ post_vector)
    post_variance_tolerance = 1.0e-12 * max(
        1.0,
        float(covariance_spectral_norm * (post_vector @ post_vector)),
    )
    if post_variance_raw < -post_variance_tolerance:
        raise RuntimeError("Post-hedge modeled variance is materially negative.")
    post_variance = max(0.0, post_variance_raw)
    variance_reduction = 1.0 - post_variance / pre_variance
    if post_variance > pre_variance * (1.0 + 1.0e-12):
        raise RuntimeError("Hedge optimization increased modeled variance.")
    if not np.isfinite(variance_reduction):
        raise RuntimeError("Hedge variance reduction is non-finite.")
    if variance_reduction < minimum_variance_reduction:
        raise ValueError(
            f"Hedge variance reduction {variance_reduction:.2%} is below the configured minimum."
        )
    return HedgeResult(
        quantities=quantities,
        pre_hedge_dv01=dv01,
        post_hedge_dv01=post_dv01,
        pre_factor_exposure=factor_exposure_from_dv01(dv01, fit, retained_factors),
        post_factor_exposure=factor_exposure_from_dv01(post_dv01, fit, retained_factors),
        pre_variance_usd2=pre_variance,
        post_variance_usd2=post_variance,
        variance_reduction=float(variance_reduction),
        design_condition_number=condition_number,
        regularization_usd2_per_quantity2=absolute_regularization,
        solver_status=solution.message,
    )
