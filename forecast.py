"""Strictly walk-forward PCA-factor forecasting and benchmark validation."""

from __future__ import annotations

from dataclasses import dataclass
from math import erfc, sqrt

import numpy as np
import pandas as pd
from scipy.stats import binomtest, norm

from .config import ForecastConfig
from .pca import PCAFit, fit_curve_pca

FORECAST_MODELS: tuple[str, ...] = (
    "No-change benchmark",
    "Historical-mean change",
    "PCA factor AR(1)",
    "PCA factor VAR(1)",
)


@dataclass(frozen=True)
class ForecastResult:
    actual_changes_bp: pd.DataFrame
    predictions_bp: dict[str, pd.DataFrame]
    metrics: pd.DataFrame
    model_comparison: pd.DataFrame
    selected_model: str
    audit_trail: pd.DataFrame
    latest_change_forecast_bp: pd.Series
    latest_level_forecast_pct: pd.Series
    prediction_intervals_pct: pd.DataFrame
    interval_coverage: pd.DataFrame
    simultaneous_interval_diagnostic: pd.DataFrame
    full_history_interval_diagnostic: pd.DataFrame
    full_history_simultaneous_diagnostic: pd.DataFrame
    forecast_as_of: pd.Timestamp
    target_period_end: pd.Timestamp


def required_forecast_holdout(config: ForecastConfig) -> int:
    """Return the minimum holdout needed for disjoint governed evaluations."""

    adoption_history = max(
        config.minimum_model_selection_observations + config.confirmation_observations,
        config.minimum_interval_observations,
    )
    return adoption_history + config.interval_evaluation_observations


def required_forecast_observations(config: ForecastConfig) -> int:
    """Return the minimum total change observations needed to run forecasting."""

    return config.minimum_training_weeks + required_forecast_holdout(config)


def _fit_ar1(scores: np.ndarray, coefficient_bound: float) -> np.ndarray:
    predictions = np.empty(scores.shape[1], dtype=float)
    for factor in range(scores.shape[1]):
        lagged = scores[:-1, factor]
        current = scores[1:, factor]
        design = np.column_stack((np.ones(len(lagged)), lagged))
        intercept, coefficient = np.linalg.lstsq(design, current, rcond=None)[0]
        coefficient = float(np.clip(coefficient, -coefficient_bound, coefficient_bound))
        intercept = float(current.mean() - coefficient * lagged.mean())
        predictions[factor] = intercept + coefficient * scores[-1, factor]
    return predictions


def _fit_var1(scores: np.ndarray, ridge: float, coefficient_bound: float) -> np.ndarray:
    score_scale = scores.std(axis=0, ddof=1)
    if np.any(score_scale <= np.finfo(float).eps):
        return np.zeros(scores.shape[1])
    standardized = scores / score_scale
    lagged = standardized[:-1]
    current = standardized[1:]
    design = np.column_stack((np.ones(len(lagged)), lagged))
    penalty = np.diag([0.0, *([ridge] * lagged.shape[1])])
    system = design.T @ design + penalty
    target = design.T @ current
    try:
        coefficients = np.linalg.solve(system, target)
    except np.linalg.LinAlgError:
        coefficients = np.linalg.lstsq(system, target, rcond=None)[0]
    transition = coefficients[1:]
    spectral_radius = float(np.max(np.abs(np.linalg.eigvals(transition))))
    if spectral_radius > coefficient_bound:
        transition = transition * (coefficient_bound / spectral_radius)
    intercept = current.mean(axis=0) - lagged.mean(axis=0) @ transition
    forecast_standardized = intercept + standardized[-1] @ transition
    return forecast_standardized * score_scale


def _forecast_from_fit(fit: PCAFit, model: str, config: ForecastConfig) -> np.ndarray:
    if model == "No-change benchmark":
        return np.zeros(len(fit.tenors))
    if model == "Historical-mean change":
        return fit.center_bp.copy()
    scores = fit.scores.iloc[:, : config.retained_factors].to_numpy()
    if model == "PCA factor AR(1)":
        factor_forecast = _fit_ar1(scores, config.ar_coefficient_bound)
    elif model == "PCA factor VAR(1)":
        factor_forecast = _fit_var1(scores, config.var_ridge, config.ar_coefficient_bound)
    else:
        raise ValueError(f"Unknown forecast model: {model}.")
    return fit.inverse_transform(factor_forecast).iloc[0].to_numpy()


def _direction_accuracy(actual: np.ndarray, prediction: np.ndarray) -> float:
    directional_observations = (actual != 0.0) & (prediction != 0.0)
    if not directional_observations.any():
        return float("nan")
    return float(
        np.mean(np.sign(actual[directional_observations]) == np.sign(prediction[directional_observations]))
    )


def _metric_rows(actual: pd.DataFrame, predictions: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, prediction in predictions.items():
        error = prediction - actual
        for tenor in actual.columns:
            rows.append(
                {
                    "Model": model,
                    "Tenor": tenor,
                    "Observations": len(actual),
                    "RMSE (bp)": float(np.sqrt(np.mean(error[tenor] ** 2))),
                    "MAE (bp)": float(np.mean(np.abs(error[tenor]))),
                    "Bias (bp)": float(np.mean(error[tenor])),
                    "Direction accuracy": _direction_accuracy(
                        actual[tenor].to_numpy(), prediction[tenor].to_numpy()
                    ),
                }
            )
        rows.append(
            {
                "Model": model,
                "Tenor": "Curve average",
                "Observations": int(actual.size),
                "RMSE (bp)": float(np.sqrt(np.mean(error.to_numpy() ** 2))),
                "MAE (bp)": float(np.mean(np.abs(error.to_numpy()))),
                "Bias (bp)": float(np.mean(error.to_numpy())),
                "Direction accuracy": _direction_accuracy(actual.to_numpy(), prediction.to_numpy()),
            }
        )
    return pd.DataFrame(rows).set_index(["Model", "Tenor"])


def newey_west_loss_test(loss_advantage: pd.Series, lags: int = 4) -> tuple[float, float, float]:
    """Test whether mean benchmark loss minus candidate loss is positive."""

    values = loss_advantage.dropna().to_numpy(dtype=float)
    if len(values) < max(20, 2 * lags + 2):
        return float("nan"), float("nan"), float("nan")
    centered = values - values.mean()
    n_obs = len(values)
    long_run_variance = float(centered @ centered / n_obs)
    for lag in range(1, min(lags, n_obs - 2) + 1):
        autocovariance = float(centered[lag:] @ centered[:-lag] / n_obs)
        bartlett_weight = 1.0 - lag / (lags + 1.0)
        long_run_variance += 2.0 * bartlett_weight * autocovariance
    if long_run_variance <= np.finfo(float).eps:
        return float(values.mean()), float("nan"), float("nan")
    statistic = float(values.mean() / sqrt(long_run_variance / n_obs))
    two_sided_p_value = float(erfc(abs(statistic) / sqrt(2.0)))
    return float(values.mean()), statistic, two_sided_p_value


def _loss_advantages(
    actual: pd.DataFrame,
    benchmark: pd.DataFrame,
    candidate: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    """Return raw and Clark-West adjusted curve-average squared-loss advantages."""

    benchmark_error = actual - benchmark
    candidate_error = actual - candidate
    raw = (benchmark_error**2 - candidate_error**2).mean(axis=1)
    forecast_divergence = benchmark - candidate
    clark_west = (benchmark_error**2 - (candidate_error**2 - forecast_divergence**2)).mean(axis=1)
    return raw, clark_west


def _one_sided_positive_p_value(statistic: float, two_sided_p_value: float) -> float:
    if not np.isfinite(statistic) or not np.isfinite(two_sided_p_value):
        return float("nan")
    return float(two_sided_p_value / 2.0 if statistic >= 0.0 else 1.0 - two_sided_p_value / 2.0)


def _compare_models(
    actual: pd.DataFrame,
    predictions: dict[str, pd.DataFrame],
    config: ForecastConfig,
) -> tuple[pd.DataFrame, str]:
    selection_end = len(actual) - config.confirmation_observations
    if selection_end < config.minimum_model_selection_observations:
        raise ValueError(
            "Forecast holdout does not contain the configured independent selection and confirmation samples."
        )
    selection_actual = actual.iloc[:selection_end]
    confirmation_actual = actual.iloc[selection_end:]
    selection_predictions = {
        model: prediction.loc[selection_actual.index] for model, prediction in predictions.items()
    }
    confirmation_predictions = {
        model: prediction.loc[confirmation_actual.index] for model, prediction in predictions.items()
    }
    benchmark = selection_predictions["No-change benchmark"]
    confirmation_benchmark = confirmation_predictions["No-change benchmark"]
    benchmark_rmse = float(np.sqrt(np.mean((benchmark.to_numpy() - selection_actual.to_numpy()) ** 2)))
    confirmation_benchmark_rmse = float(
        np.sqrt(np.mean((confirmation_benchmark.to_numpy() - confirmation_actual.to_numpy()) ** 2))
    )
    rows: list[dict[str, object]] = []
    eligible_models: list[tuple[str, float]] = []
    adjusted_alpha = config.adoption_alpha / (len(FORECAST_MODELS) - 1)
    for model in FORECAST_MODELS:
        prediction = selection_predictions[model]
        rmse = float(np.sqrt(np.mean((prediction.to_numpy() - selection_actual.to_numpy()) ** 2)))
        raw_advantage, adjusted_advantage = _loss_advantages(selection_actual, benchmark, prediction)
        advantage, statistic, p_value = newey_west_loss_test(raw_advantage, config.hac_lags)
        cw_advantage, cw_statistic, cw_two_sided = newey_west_loss_test(adjusted_advantage, config.hac_lags)
        cw_one_sided = _one_sided_positive_p_value(cw_statistic, cw_two_sided)
        improvement = 1.0 - rmse / benchmark_rmse if benchmark_rmse > 0 else float("nan")
        passes_selection = (
            model != "No-change benchmark"
            and improvement > config.minimum_rmse_improvement
            and np.isfinite(cw_one_sided)
            and cw_statistic > 0
            and cw_one_sided < adjusted_alpha
        )
        if passes_selection:
            eligible_models.append((model, rmse))

        confirmation_prediction = confirmation_predictions[model]
        confirmation_rmse = float(
            np.sqrt(np.mean((confirmation_prediction.to_numpy() - confirmation_actual.to_numpy()) ** 2))
        )
        confirmation_improvement = (
            1.0 - confirmation_rmse / confirmation_benchmark_rmse
            if confirmation_benchmark_rmse > 0
            else float("nan")
        )
        _, confirmation_cw = _loss_advantages(
            confirmation_actual,
            confirmation_benchmark,
            confirmation_prediction,
        )
        (
            confirmation_cw_advantage,
            confirmation_cw_statistic,
            confirmation_cw_two_sided,
        ) = newey_west_loss_test(confirmation_cw, config.hac_lags)
        confirmation_cw_one_sided = _one_sided_positive_p_value(
            confirmation_cw_statistic, confirmation_cw_two_sided
        )
        passes_confirmation = (
            model != "No-change benchmark"
            and confirmation_improvement > config.minimum_rmse_improvement
            and np.isfinite(confirmation_cw_one_sided)
            and confirmation_cw_statistic > 0
            and confirmation_cw_one_sided < config.adoption_alpha
        )
        rows.append(
            {
                "Model": model,
                "Selection start": selection_actual.index.min(),
                "Selection end": selection_actual.index.max(),
                "Selection observations": len(selection_actual),
                "Selection curve RMSE (bp)": rmse,
                "Selection RMSE improvement vs no-change": improvement,
                "Mean weekly squared-loss advantage": advantage,
                "HAC statistic": statistic,
                "Two-sided p-value": p_value,
                "Clark-West adjusted advantage": cw_advantage,
                "Clark-West HAC statistic": cw_statistic,
                "Clark-West one-sided p-value": cw_one_sided,
                "Multiplicity-adjusted alpha": adjusted_alpha,
                "Passes selection gate": passes_selection,
                "Confirmation start": confirmation_actual.index.min(),
                "Confirmation end": confirmation_actual.index.max(),
                "Confirmation observations": len(confirmation_actual),
                "Confirmation curve RMSE (bp)": confirmation_rmse,
                "Confirmation RMSE improvement": confirmation_improvement,
                "Confirmation Clark-West advantage": confirmation_cw_advantage,
                "Confirmation Clark-West HAC statistic": confirmation_cw_statistic,
                "Confirmation Clark-West one-sided p-value": confirmation_cw_one_sided,
                "Confirmation alpha": config.adoption_alpha,
                "Passes confirmation gate": passes_confirmation,
            }
        )
    preliminary = min(eligible_models, key=lambda item: item[1])[0] if eligible_models else None
    comparison = pd.DataFrame(rows).set_index("Model")
    comparison["Passes adoption gate"] = False
    if preliminary is not None and bool(comparison.loc[preliminary, "Passes confirmation gate"]):
        comparison.loc[preliminary, "Passes adoption gate"] = True
        selected = preliminary
    else:
        selected = "No-change benchmark"
    comparison["Selected"] = comparison.index == selected
    return comparison, selected


def _circular_block_coverage_bounds(
    hit_values: np.ndarray,
    config: ForecastConfig,
    *,
    family_size: int,
    seed_offset: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return conservative, dependence-aware simultaneous coverage bounds.

    Weekly rows are resampled as circular moving blocks, preserving cross-tenor
    dependence.  Each configured block length receives a Bonferroni-adjusted
    percentile interval; the returned envelope is the most conservative bound
    across block-length sensitivities.
    """

    observations, tenors = hit_values.shape
    adjusted_tail = config.interval_validation_alpha / (2.0 * family_size)
    lower_bounds: list[np.ndarray] = []
    upper_bounds: list[np.ndarray] = []
    for block_position, configured_length in enumerate(config.interval_bootstrap_block_lengths):
        block_length = min(configured_length, observations)
        blocks_needed = int(np.ceil(observations / block_length))
        rng = np.random.default_rng(
            config.interval_bootstrap_random_seed + seed_offset + 10_007 * block_position
        )
        bootstrap_means = np.empty((config.interval_bootstrap_replications, tenors), dtype=float)
        chunk_size = 250
        offsets = np.arange(block_length, dtype=int)
        for chunk_start in range(0, config.interval_bootstrap_replications, chunk_size):
            chunk_end = min(chunk_start + chunk_size, config.interval_bootstrap_replications)
            starts = rng.integers(
                0,
                observations,
                size=(chunk_end - chunk_start, blocks_needed),
            )
            indices = (starts[..., None] + offsets) % observations
            indices = indices.reshape(chunk_end - chunk_start, -1)[:, :observations]
            bootstrap_means[chunk_start:chunk_end] = hit_values[indices].mean(axis=1)
        lower_bounds.append(np.quantile(bootstrap_means, adjusted_tail, axis=0, method="linear"))
        upper_bounds.append(np.quantile(bootstrap_means, 1.0 - adjusted_tail, axis=0, method="linear"))
    return (
        np.min(np.vstack(lower_bounds), axis=0),
        np.max(np.vstack(upper_bounds), axis=0),
    )


def _prequential_interval_coverage(
    residuals_bp: pd.DataFrame,
    config: ForecastConfig,
    coverages: tuple[float, ...] = (0.80, 0.95),
    evaluation_start_position: int | None = None,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for coverage in coverages:
        tail = (1.0 - coverage) / 2.0
        hits: list[pd.Series] = []
        first_evaluation = max(
            config.minimum_interval_observations,
            0 if evaluation_start_position is None else evaluation_start_position,
        )
        for position in range(first_evaluation, len(residuals_bp)):
            history = residuals_bp.iloc[max(0, position - config.interval_history_weeks) : position]
            lower = history.quantile(tail)
            upper = history.quantile(1.0 - tail)
            observed = residuals_bp.iloc[position]
            hits.append((observed >= lower) & (observed <= upper))
        if not hits:
            for tenor in residuals_bp.columns:
                records.append(
                    {
                        "Nominal coverage": coverage,
                        "Tenor": tenor,
                        "Hits": 0,
                        "Observed coverage": np.nan,
                        "Evaluated forecasts": 0,
                        "Acceptable coverage floor": coverage - config.interval_coverage_tolerance,
                        "Wilson lower bound": np.nan,
                        "Wilson upper bound": np.nan,
                        "IID-binomial undercoverage p-value": np.nan,
                        "Hit lag-1 autocorrelation": np.nan,
                        "Familywise block-bootstrap coverage lower bound": np.nan,
                        "Familywise block-bootstrap coverage upper bound": np.nan,
                        "Familywise gate alpha": config.interval_validation_alpha,
                        "Bootstrap replications": config.interval_bootstrap_replications,
                        "Bootstrap tail order statistics": np.nan,
                        "Bootstrap block lengths": ",".join(
                            str(value) for value in config.interval_bootstrap_block_lengths
                        ),
                        "Confidence level": 1.0 - config.interval_validation_alpha,
                        "Status": "NOT EVALUATED",
                    }
                )
            continue
        hit_frame = pd.DataFrame(hits)
        hit_values = hit_frame.loc[:, residuals_bp.columns].to_numpy(dtype=float)
        family_size = len(coverages) * len(residuals_bp.columns)
        block_lower, block_upper = _circular_block_coverage_bounds(
            hit_values,
            config,
            family_size=family_size,
            seed_offset=int(round(coverage * 10_000)),
        )
        for tenor_position, tenor in enumerate(residuals_bp.columns):
            hit_count = int(hit_frame[tenor].sum())
            observations = len(hit_frame)
            observed_coverage = hit_count / observations
            acceptable_floor = coverage - config.interval_coverage_tolerance
            alpha = config.interval_validation_alpha
            two_sided_z = float(norm.ppf(1.0 - alpha / 2.0))
            denominator = 1.0 + two_sided_z**2 / observations
            center = (observed_coverage + two_sided_z**2 / (2.0 * observations)) / denominator
            half_width = (
                two_sided_z
                * np.sqrt(
                    observed_coverage * (1.0 - observed_coverage) / observations
                    + two_sided_z**2 / (4.0 * observations**2)
                )
                / denominator
            )
            undercoverage_p_value = float(
                binomtest(
                    hit_count,
                    observations,
                    p=acceptable_floor,
                    alternative="less",
                ).pvalue
            )
            lagged_hits = hit_values[:-1, tenor_position]
            current_hits = hit_values[1:, tenor_position]
            if (
                observations > 1
                and np.std(lagged_hits) > np.finfo(float).eps
                and np.std(current_hits) > np.finfo(float).eps
            ):
                lag_one = float(np.corrcoef(current_hits, lagged_hits)[0, 1])
            else:
                lag_one = float("nan")
            dependence_robust_lower = float(block_lower[tenor_position])
            dependence_robust_upper = float(block_upper[tenor_position])
            if dependence_robust_upper < acceptable_floor:
                status = "WARN"
            elif dependence_robust_lower >= acceptable_floor:
                status = "PASS"
            else:
                status = "INCONCLUSIVE"
            records.append(
                {
                    "Nominal coverage": coverage,
                    "Tenor": tenor,
                    "Hits": hit_count,
                    "Observed coverage": observed_coverage,
                    "Evaluated forecasts": observations,
                    "Acceptable coverage floor": acceptable_floor,
                    "Wilson lower bound": center - half_width,
                    "Wilson upper bound": center + half_width,
                    "IID-binomial undercoverage p-value": undercoverage_p_value,
                    "Hit lag-1 autocorrelation": lag_one,
                    "Familywise block-bootstrap coverage lower bound": dependence_robust_lower,
                    "Familywise block-bootstrap coverage upper bound": dependence_robust_upper,
                    "Familywise gate alpha": alpha,
                    "Bootstrap replications": config.interval_bootstrap_replications,
                    "Bootstrap tail order statistics": (
                        config.interval_bootstrap_replications * alpha / (2.0 * family_size)
                    ),
                    "Bootstrap block lengths": ",".join(
                        str(value) for value in config.interval_bootstrap_block_lengths
                    ),
                    "Confidence level": 1.0 - alpha,
                    "Status": status,
                }
            )
    result = pd.DataFrame(records).set_index(["Nominal coverage", "Tenor"])
    nominal = result.index.get_level_values("Nominal coverage").to_numpy(dtype=float)
    result["Coverage gap"] = result["Observed coverage"].to_numpy() - nominal
    return result


def _prequential_simultaneous_diagnostic(
    residuals_bp: pd.DataFrame,
    config: ForecastConfig,
    coverages: tuple[float, ...] = (0.80, 0.95),
    evaluation_start_position: int | None = None,
) -> pd.DataFrame:
    """Report joint hits of marginal tenor bands without calling them joint intervals."""

    rows: list[dict[str, object]] = []
    for coverage in coverages:
        tail = (1.0 - coverage) / 2.0
        joint_hits: list[bool] = []
        first_evaluation = max(
            config.minimum_interval_observations,
            0 if evaluation_start_position is None else evaluation_start_position,
        )
        for position in range(first_evaluation, len(residuals_bp)):
            history = residuals_bp.iloc[max(0, position - config.interval_history_weeks) : position]
            observed = residuals_bp.iloc[position]
            within_marginal_bands = (observed >= history.quantile(tail)) & (
                observed <= history.quantile(1.0 - tail)
            )
            joint_hits.append(bool(within_marginal_bands.all()))
        rows.append(
            {
                "Marginal tenor-band nominal coverage": coverage,
                "Joint hit rate of marginal tenor bands": (
                    float(np.mean(joint_hits)) if joint_hits else np.nan
                ),
                "Evaluated forecasts": len(joint_hits),
                "Status": "DIAGNOSTIC",
                "Interpretation": (
                    "The marginal bands are not calibrated as a simultaneous curve region; "
                    "this hit rate must not be compared directly with the marginal nominal level."
                ),
            }
        )
    return pd.DataFrame(rows).set_index("Marginal tenor-band nominal coverage")


def _latest_intervals(
    current_curve_pct: pd.Series,
    change_forecast_bp: pd.Series,
    residuals_bp: pd.DataFrame,
    config: ForecastConfig,
    coverages: tuple[float, ...] = (0.80, 0.95),
) -> pd.DataFrame:
    history = residuals_bp.iloc[-config.interval_history_weeks :]
    rows: list[pd.Series] = []
    labels: list[str] = []
    point_level = current_curve_pct + change_forecast_bp / 100.0
    rows.append(point_level)
    labels.append("Point forecast")
    for coverage in coverages:
        tail = (1.0 - coverage) / 2.0
        rows.append(current_curve_pct + (change_forecast_bp + history.quantile(tail)) / 100.0)
        labels.append(f"{coverage:.0%} lower")
        rows.append(current_curve_pct + (change_forecast_bp + history.quantile(1.0 - tail)) / 100.0)
        labels.append(f"{coverage:.0%} upper")
    result = pd.DataFrame(rows, index=labels)
    result.index.name = "Forecast statistic"
    return result


def walk_forward_forecast(
    changes_bp: pd.DataFrame,
    weekly_yields_pct: pd.DataFrame,
    config: ForecastConfig,
    *,
    weekly_rule: str = "W-FRI",
) -> ForecastResult:
    """Run expanding-origin forecasts with every calibration restricted to history."""

    if not isinstance(changes_bp, pd.DataFrame) or not isinstance(weekly_yields_pct, pd.DataFrame):
        raise TypeError("changes_bp and weekly_yields_pct must be pandas DataFrames.")
    if not isinstance(changes_bp.index, pd.DatetimeIndex) or not isinstance(
        weekly_yields_pct.index, pd.DatetimeIndex
    ):
        raise ValueError("Forecast inputs must use DatetimeIndex labels.")
    for name, index in (("changes_bp", changes_bp.index), ("weekly_yields_pct", weekly_yields_pct.index)):
        if not index.is_unique or not index.is_monotonic_increasing:
            raise ValueError(f"{name} index must be unique and strictly ordered.")
    frame = changes_bp.astype(float)
    if frame.isna().any().any() or not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Forecast input contains missing or non-finite values.")
    if tuple(frame.columns) != tuple(weekly_yields_pct.columns):
        raise ValueError("Yield levels and yield changes must have identical tenor order.")
    try:
        levels = weekly_yields_pct.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("Yield levels must be numeric.") from exc
    if levels.isna().any().any() or not np.isfinite(levels.to_numpy()).all():
        raise ValueError("Yield levels contain missing or non-finite values.")
    if len(levels) != len(frame) + 1 or not levels.index[1:].equals(frame.index):
        raise ValueError(
            "Yield levels must contain the pre-change curve followed by one level for every change date."
        )
    implied_changes = levels.diff().iloc[1:] * 100.0
    maximum_reconciliation_error = float(np.max(np.abs(implied_changes.to_numpy() - frame.to_numpy())))
    if maximum_reconciliation_error > config.level_change_reconciliation_tolerance_bp:
        raise ValueError(
            "Yield levels do not reconcile to supplied changes: "
            f"maximum error={maximum_reconciliation_error:.6g} bp."
        )
    holdout_observations = len(frame) - config.minimum_training_weeks
    minimum_holdout = required_forecast_holdout(config)
    if holdout_observations < minimum_holdout:
        raise ValueError(
            "Forecast holdout is too short for the configured disjoint selection, "
            "confirmation, interval-history, and interval-evaluation samples: "
            f"observed={holdout_observations}, required={minimum_holdout}."
        )

    prediction_rows: dict[str, list[pd.Series]] = {model: [] for model in FORECAST_MODELS}
    audit_rows: list[dict[str, object]] = []
    for target_position in range(config.minimum_training_weeks, len(frame)):
        training = frame.iloc[:target_position]
        fit = fit_curve_pca(training, standardize=False)
        target_date = frame.index[target_position]
        for model in FORECAST_MODELS:
            forecast = pd.Series(
                _forecast_from_fit(fit, model, config),
                index=frame.columns,
                name=target_date,
            )
            prediction_rows[model].append(forecast)
        audit_rows.append(
            {
                "Target date": target_date,
                "Training start": training.index.min(),
                "Training end": training.index.max(),
                "Training observations": len(training),
                "Maximum source position": target_position - 1,
                "Target position": target_position,
            }
        )

    predictions = {model: pd.DataFrame(rows) for model, rows in prediction_rows.items()}
    first_prediction_index = next(iter(predictions.values())).index
    actual = frame.loc[first_prediction_index]
    metrics = _metric_rows(actual, predictions)
    interval_evaluation_start = len(actual) - config.interval_evaluation_observations
    adoption_actual = actual.iloc[:interval_evaluation_start]
    adoption_predictions = {
        model: prediction.loc[adoption_actual.index] for model, prediction in predictions.items()
    }
    comparison, selected = _compare_models(adoption_actual, adoption_predictions, config)
    interval_actual = actual.iloc[interval_evaluation_start:]
    comparison["Interval evaluation start"] = interval_actual.index.min()
    comparison["Interval evaluation end"] = interval_actual.index.max()
    comparison["Interval evaluation observations"] = len(interval_actual)

    final_fit = fit_curve_pca(frame, standardize=False)
    forecast_as_of = pd.Timestamp(levels.index[-1])
    target_period_end = (forecast_as_of.to_period(weekly_rule) + 1).end_time.normalize()
    latest_change = pd.Series(
        _forecast_from_fit(final_fit, selected, config),
        index=frame.columns,
        name=target_period_end,
    )
    current_curve = levels.iloc[-1].reindex(frame.columns)
    latest_level = current_curve + latest_change / 100.0
    latest_level.name = target_period_end
    residuals = actual - predictions[selected]
    intervals = _latest_intervals(current_curve, latest_change, residuals, config)
    coverage = _prequential_interval_coverage(
        residuals,
        config,
        evaluation_start_position=interval_evaluation_start,
    )
    simultaneous_diagnostic = _prequential_simultaneous_diagnostic(
        residuals,
        config,
        evaluation_start_position=interval_evaluation_start,
    )
    full_history_interval_diagnostic = _prequential_interval_coverage(
        residuals,
        config,
    )
    full_history_simultaneous_diagnostic = _prequential_simultaneous_diagnostic(
        residuals,
        config,
    )
    audit_trail = pd.DataFrame(audit_rows).set_index("Target date")
    selection_end = len(adoption_actual) - config.confirmation_observations
    audit_trail["Evaluation partition"] = "Selection"
    partition_column = audit_trail.columns.get_loc("Evaluation partition")
    audit_trail.iloc[selection_end:interval_evaluation_start, partition_column] = "Confirmation"
    audit_trail.iloc[interval_evaluation_start:, partition_column] = "Interval evaluation"

    if not (audit_trail["Training end"] < audit_trail.index).all():
        raise RuntimeError("Forecast audit trail detected look-ahead contamination.")

    return ForecastResult(
        actual_changes_bp=actual,
        predictions_bp=predictions,
        metrics=metrics,
        model_comparison=comparison,
        selected_model=selected,
        audit_trail=audit_trail,
        latest_change_forecast_bp=latest_change,
        latest_level_forecast_pct=latest_level,
        prediction_intervals_pct=intervals,
        interval_coverage=coverage,
        simultaneous_interval_diagnostic=simultaneous_diagnostic,
        full_history_interval_diagnostic=full_history_interval_diagnostic,
        full_history_simultaneous_diagnostic=full_history_simultaneous_diagnostic,
        forecast_as_of=forecast_as_of,
        target_period_end=target_period_end,
    )
