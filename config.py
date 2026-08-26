"""Validated configuration and market-data conventions.

The package deliberately keeps market-data, statistical-model, forecast, and
portfolio-risk settings separate.  This prevents a harmless presentation
change from silently changing a calibrated model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from math import isfinite
from numbers import Real
from pathlib import Path
from typing import Literal

FRED_SERIES: tuple[tuple[str, str], ...] = (
    ("DGS1MO", "1M"),
    ("DGS3MO", "3M"),
    ("DGS6MO", "6M"),
    ("DGS1", "1Y"),
    ("DGS2", "2Y"),
    ("DGS3", "3Y"),
    ("DGS5", "5Y"),
    ("DGS7", "7Y"),
    ("DGS10", "10Y"),
    ("DGS20", "20Y"),
    ("DGS30", "30Y"),
)

MATURITY_YEARS: dict[str, float] = {
    "1M": 1.0 / 12.0,
    "3M": 0.25,
    "6M": 0.50,
    "1Y": 1.0,
    "2Y": 2.0,
    "3Y": 3.0,
    "5Y": 5.0,
    "7Y": 7.0,
    "10Y": 10.0,
    "20Y": 20.0,
    "30Y": 30.0,
}

CORE_TENORS: tuple[str, ...] = (
    "3M",
    "6M",
    "1Y",
    "2Y",
    "3Y",
    "5Y",
    "7Y",
    "10Y",
    "20Y",
)
ALL_TENORS: tuple[str, ...] = tuple(tenor for _, tenor in FRED_SERIES)


def _validate_iso_date(value: str, field_name: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an ISO date (YYYY-MM-DD); received {value!r}.") from exc


def _is_integer(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _is_finite_number(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and isfinite(value)


@dataclass(frozen=True)
class DataConfig:
    """Controls source selection and weekly-curve construction.

    ``snapshot`` is the deterministic peer-review mode. ``live`` requires a
    successful FRED response. ``live_then_snapshot`` is operationally robust,
    but its output can differ when FRED revises history.
    """

    start_date: str = "2000-01-01"
    end_date: str = "2025-12-31"
    source_mode: Literal["snapshot", "live", "live_then_snapshot"] = "snapshot"
    weekly_rule: str = "W-FRI"
    boundary_week_policy: Literal["include_and_flag", "drop"] = "drop"
    core_tenors: tuple[str, ...] = CORE_TENORS
    snapshot_path: Path = Path("data/h15_treasury_cmt_2000_2025.csv.gz")
    manifest_path: Path = Path("data/source_manifest.json")
    cache_dir: Path = Path(".cache/yield_curve_pca")
    use_cache: bool = True
    cache_max_age_hours: float = 24.0
    request_timeout_seconds: float = 10.0
    retries: int = 2
    maximum_response_bytes: int = 25_000_000
    minimum_complete_weeks: int = 260
    maximum_start_coverage_gap_days: int = 7
    maximum_end_coverage_gap_days: int = 7
    maximum_within_week_observation_lag_days: int = 1
    maximum_calendar_gap_days: int = 12
    maximum_staleness_days: int | None = None

    def __post_init__(self) -> None:
        start = _validate_iso_date(self.start_date, "start_date")
        end = _validate_iso_date(self.end_date, "end_date")
        if start >= end:
            raise ValueError("start_date must be earlier than end_date.")
        if self.source_mode not in {"snapshot", "live", "live_then_snapshot"}:
            raise ValueError("source_mode must be snapshot, live, or live_then_snapshot.")
        if self.weekly_rule not in {f"W-{day}" for day in ("MON", "TUE", "WED", "THU", "FRI", "SAT", "SUN")}:
            raise ValueError("weekly_rule must be one of W-MON through W-SUN.")
        if self.boundary_week_policy not in {"include_and_flag", "drop"}:
            raise ValueError("boundary_week_policy must be 'include_and_flag' or 'drop'.")
        if not isinstance(self.core_tenors, tuple):
            raise TypeError("core_tenors must be an immutable tuple.")
        if len(self.core_tenors) < 3 or len(set(self.core_tenors)) != len(self.core_tenors):
            raise ValueError("core_tenors must contain at least three unique tenors.")
        unknown = set(self.core_tenors) - set(ALL_TENORS)
        if unknown:
            raise ValueError(f"Unknown core tenors: {sorted(unknown)}.")
        if not all(
            isinstance(path, Path) for path in (self.snapshot_path, self.manifest_path, self.cache_dir)
        ):
            raise TypeError("snapshot_path, manifest_path, and cache_dir must be pathlib.Path values.")
        if not isinstance(self.use_cache, bool):
            raise TypeError("use_cache must be boolean.")
        if not _is_finite_number(self.request_timeout_seconds) or not 0 < self.request_timeout_seconds <= 120:
            raise ValueError("request_timeout_seconds must lie in (0, 120].")
        if not _is_integer(self.retries) or not 1 <= self.retries <= 10:
            raise ValueError("retries must be an integer between one and ten.")
        if not _is_finite_number(self.cache_max_age_hours) or not 0 < self.cache_max_age_hours <= 8_760:
            raise ValueError("cache_max_age_hours must lie in (0, 8760].")
        if (
            not _is_integer(self.maximum_response_bytes)
            or not 1_000 <= self.maximum_response_bytes <= 250_000_000
        ):
            raise ValueError("maximum_response_bytes must lie in [1000, 250000000].")
        if not _is_integer(self.minimum_complete_weeks) or not 52 <= self.minimum_complete_weeks <= 10_000:
            raise ValueError("minimum_complete_weeks must lie in [52, 10000].")
        for name, value in (
            ("maximum_start_coverage_gap_days", self.maximum_start_coverage_gap_days),
            ("maximum_end_coverage_gap_days", self.maximum_end_coverage_gap_days),
            (
                "maximum_within_week_observation_lag_days",
                self.maximum_within_week_observation_lag_days,
            ),
        ):
            if not _is_integer(value) or not 0 <= value <= 366:
                raise ValueError(f"{name} must be an integer in [0, 366].")
        if not _is_integer(self.maximum_calendar_gap_days) or not 7 <= self.maximum_calendar_gap_days <= 366:
            raise ValueError("maximum_calendar_gap_days must lie in [7, 366].")
        if self.maximum_staleness_days is not None and (
            not _is_integer(self.maximum_staleness_days) or not 0 <= self.maximum_staleness_days <= 36_600
        ):
            raise ValueError("maximum_staleness_days must lie in [0, 36600].")


@dataclass(frozen=True)
class PCAConfig:
    """Controls structural and recency-weighted PCA calibrations."""

    retained_factors: int = 3
    minimum_template_similarity: float = 0.70
    minimum_template_dominance_margin: float = 0.10
    current_halflife_weeks: float = 52.0
    minimum_ewma_effective_observations: float = 100.0
    rolling_window_years: int = 5
    minimum_rolling_observations: int = 200
    maximum_principal_angle_warning_degrees: float = 30.0
    minimum_loading_cosine_warning: float = 0.80
    maximum_oos_rmse_warning_bp: float = 5.0
    minimum_methodology_regime_observations: int = 104
    maximum_methodology_subspace_angle_warning_degrees: float = 15.0
    maximum_methodology_sigma_ratio_warning: float = 1.25
    maximum_current_factor_sigma_ratio_warning: float = 1.25
    bootstrap_replications: int = 2_000
    bootstrap_block_weeks: int = 13
    bootstrap_sensitivity_replications: int = 2_000
    bootstrap_sensitivity_blocks: tuple[int, ...] = (4, 13, 26)
    oos_minimum_training_observations: int = 520
    oos_minimum_holdout_observations: int = 104
    oos_refit_every_observations: int = 13
    random_seed: int = 20260825

    def __post_init__(self) -> None:
        if not _is_integer(self.retained_factors) or not 1 <= self.retained_factors <= 3:
            raise ValueError("retained_factors must be between one and three for the economic factor map.")
        if (
            not _is_finite_number(self.minimum_template_similarity)
            or not 0.0 < self.minimum_template_similarity <= 1.0
        ):
            raise ValueError("minimum_template_similarity must lie in (0, 1].")
        if (
            not _is_finite_number(self.minimum_template_dominance_margin)
            or not 0.0 <= self.minimum_template_dominance_margin < 1.0
        ):
            raise ValueError("minimum_template_dominance_margin must lie in [0, 1).")
        if not _is_finite_number(self.current_halflife_weeks) or self.current_halflife_weeks <= 1.0:
            raise ValueError("current_halflife_weeks must exceed one week.")
        if (
            not _is_finite_number(self.minimum_ewma_effective_observations)
            or self.minimum_ewma_effective_observations < 20.0
        ):
            raise ValueError("minimum_ewma_effective_observations must be at least 20.")
        if not _is_integer(self.rolling_window_years) or not 2 <= self.rolling_window_years <= 30:
            raise ValueError("rolling_window_years must lie in [2, 30].")
        if (
            not _is_integer(self.minimum_rolling_observations)
            or not 52 <= self.minimum_rolling_observations <= 1_560
        ):
            raise ValueError("minimum_rolling_observations must lie in [52, 1560].")
        if self.minimum_rolling_observations > 52 * self.rolling_window_years:
            raise ValueError("minimum_rolling_observations cannot exceed the configured rolling window.")
        if (
            not _is_finite_number(self.maximum_principal_angle_warning_degrees)
            or not 0.0 < self.maximum_principal_angle_warning_degrees <= 90.0
        ):
            raise ValueError("maximum_principal_angle_warning_degrees must lie in (0, 90].")
        if (
            not _is_finite_number(self.minimum_loading_cosine_warning)
            or not 0.0 <= self.minimum_loading_cosine_warning <= 1.0
        ):
            raise ValueError("minimum_loading_cosine_warning must lie in [0, 1].")
        if not _is_finite_number(self.maximum_oos_rmse_warning_bp) or self.maximum_oos_rmse_warning_bp <= 0.0:
            raise ValueError("maximum_oos_rmse_warning_bp must be positive and finite.")
        if (
            not _is_integer(self.minimum_methodology_regime_observations)
            or self.minimum_methodology_regime_observations < 52
        ):
            raise ValueError("minimum_methodology_regime_observations must be at least 52.")
        if (
            not _is_finite_number(self.maximum_methodology_subspace_angle_warning_degrees)
            or not 0.0 < self.maximum_methodology_subspace_angle_warning_degrees <= 90.0
        ):
            raise ValueError("maximum_methodology_subspace_angle_warning_degrees must lie in (0, 90].")
        if (
            not _is_finite_number(self.maximum_methodology_sigma_ratio_warning)
            or self.maximum_methodology_sigma_ratio_warning <= 1.0
        ):
            raise ValueError("maximum_methodology_sigma_ratio_warning must exceed one.")
        if (
            not _is_finite_number(self.maximum_current_factor_sigma_ratio_warning)
            or self.maximum_current_factor_sigma_ratio_warning <= 1.0
        ):
            raise ValueError("maximum_current_factor_sigma_ratio_warning must exceed one.")
        if (
            not _is_integer(self.bootstrap_replications)
            or not _is_integer(self.bootstrap_block_weeks)
            or not 800 <= self.bootstrap_replications <= 100_000
            or not 2 <= self.bootstrap_block_weeks <= 5_200
        ):
            raise ValueError(
                "bootstrap_replications must lie in [800, 100000] so each 2.5% "
                "quantile tail has at least 20 expected order statistics; block length "
                "must lie in [2, 5200]."
            )
        if (
            not _is_integer(self.bootstrap_sensitivity_replications)
            or not 800 <= self.bootstrap_sensitivity_replications <= 20_000
        ):
            raise ValueError(
                "bootstrap_sensitivity_replications must lie in [800, 20000] so each "
                "2.5% quantile tail has at least 20 expected order statistics."
            )
        if not isinstance(self.bootstrap_sensitivity_blocks, tuple):
            raise TypeError("bootstrap_sensitivity_blocks must be an immutable tuple.")
        if (
            not self.bootstrap_sensitivity_blocks
            or len(set(self.bootstrap_sensitivity_blocks)) != len(self.bootstrap_sensitivity_blocks)
            or len(self.bootstrap_sensitivity_blocks) > 20
            or any(
                not _is_integer(block) or not 2 <= block <= 5_200
                for block in self.bootstrap_sensitivity_blocks
            )
        ):
            raise ValueError("bootstrap_sensitivity_blocks must contain unique values of at least two weeks.")
        if (
            not _is_integer(self.oos_minimum_training_observations)
            or not _is_integer(self.oos_minimum_holdout_observations)
            or not _is_integer(self.oos_refit_every_observations)
            or self.oos_minimum_training_observations < 104
            or self.oos_minimum_holdout_observations < 52
            or not 104 <= self.oos_minimum_training_observations <= 10_000
            or not 52 <= self.oos_minimum_holdout_observations <= 5_200
            or not 1 <= self.oos_refit_every_observations <= 520
        ):
            raise ValueError("OOS reconstruction settings are invalid.")
        if not _is_integer(self.random_seed):
            raise ValueError("random_seed must be an integer.")


@dataclass(frozen=True)
class ForecastConfig:
    """Controls strictly out-of-sample, one-week-ahead forecast validation."""

    enabled: bool = True
    minimum_training_weeks: int = 520
    retained_factors: int = 3
    var_ridge: float = 1.0e-3
    ar_coefficient_bound: float = 0.995
    hac_lags: int = 4
    interval_history_weeks: int = 260
    minimum_interval_observations: int = 104
    minimum_model_selection_observations: int = 52
    confirmation_observations: int = 52
    interval_evaluation_observations: int = 260
    interval_coverage_tolerance: float = 0.03
    interval_validation_alpha: float = 0.05
    interval_bootstrap_replications: int = 20_000
    interval_bootstrap_block_lengths: tuple[int, ...] = (4, 13, 26)
    interval_bootstrap_random_seed: int = 17_729
    level_change_reconciliation_tolerance_bp: float = 1.0e-8
    adoption_alpha: float = 0.05
    minimum_rmse_improvement: float = 0.01

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled must be boolean.")
        if not _is_integer(self.minimum_training_weeks) or not 104 <= self.minimum_training_weeks <= 10_000:
            raise ValueError("minimum_training_weeks must lie in [104, 10000].")
        if not _is_integer(self.retained_factors) or not 1 <= self.retained_factors <= 3:
            raise ValueError("retained_factors must be between one and three.")
        if not _is_finite_number(self.var_ridge) or self.var_ridge < 0:
            raise ValueError("var_ridge cannot be negative.")
        if not _is_finite_number(self.ar_coefficient_bound) or not 0 < self.ar_coefficient_bound < 1:
            raise ValueError("ar_coefficient_bound must lie in (0, 1).")
        if not _is_integer(self.hac_lags) or not 0 <= self.hac_lags <= 52:
            raise ValueError("hac_lags must lie in [0, 52].")
        if (
            not _is_integer(self.interval_history_weeks)
            or not _is_integer(self.minimum_interval_observations)
            or self.minimum_interval_observations < 20
            or self.interval_history_weeks < self.minimum_interval_observations
            or self.interval_history_weeks > 5_200
        ):
            raise ValueError("interval_history_weeks must cover minimum_interval_observations.")
        if (
            not _is_integer(self.minimum_model_selection_observations)
            or not _is_integer(self.confirmation_observations)
            or not _is_integer(self.interval_evaluation_observations)
            or self.minimum_model_selection_observations < 20
            or self.confirmation_observations < 20
            or self.interval_evaluation_observations < 20
            or self.minimum_model_selection_observations > 5_200
            or self.confirmation_observations > 5_200
            or self.interval_evaluation_observations > 5_200
        ):
            raise ValueError(
                "Forecast selection, confirmation, and interval-evaluation samples must each "
                "contain at least 20 observations."
            )
        if (
            not _is_finite_number(self.interval_coverage_tolerance)
            or not 0.0 <= self.interval_coverage_tolerance < 0.20
        ):
            raise ValueError("interval_coverage_tolerance must lie in [0, 0.20).")
        if (
            not _is_finite_number(self.interval_validation_alpha)
            or not 0.0 < self.interval_validation_alpha < 0.50
        ):
            raise ValueError("interval_validation_alpha must lie in (0, 0.50).")
        if (
            not _is_integer(self.interval_bootstrap_replications)
            or not 500 <= self.interval_bootstrap_replications <= 200_000
        ):
            raise ValueError("interval_bootstrap_replications must be at least 500.")
        maximum_coverage_family = 2 * len(ALL_TENORS)
        conservative_tail_order_statistics = (
            self.interval_bootstrap_replications
            * self.interval_validation_alpha
            / (2.0 * maximum_coverage_family)
        )
        if conservative_tail_order_statistics < 20.0:
            raise ValueError(
                "interval_bootstrap_replications and interval_validation_alpha must provide "
                "at least 20 expected order statistics in each Bonferroni tail for the "
                f"maximum {maximum_coverage_family}-metric family."
            )
        if not isinstance(self.interval_bootstrap_block_lengths, tuple):
            raise TypeError("interval_bootstrap_block_lengths must be an immutable tuple.")
        if (
            not self.interval_bootstrap_block_lengths
            or any(
                not _is_integer(block_length) or block_length < 2
                for block_length in self.interval_bootstrap_block_lengths
            )
            or len(set(self.interval_bootstrap_block_lengths)) != len(self.interval_bootstrap_block_lengths)
            or len(self.interval_bootstrap_block_lengths) > 20
            or any(block_length > 520 for block_length in self.interval_bootstrap_block_lengths)
        ):
            raise ValueError("interval_bootstrap_block_lengths must contain unique integers of at least two.")
        if not _is_integer(self.interval_bootstrap_random_seed):
            raise ValueError("interval_bootstrap_random_seed must be an integer.")
        if self.interval_evaluation_observations < 2 * max(self.interval_bootstrap_block_lengths):
            raise ValueError(
                "interval_evaluation_observations must be at least twice the longest "
                "interval bootstrap block."
            )
        if 2 * self.hac_lags + 2 > min(
            self.minimum_model_selection_observations,
            self.confirmation_observations,
        ):
            raise ValueError(
                "Each forecast test partition must contain at least 2*hac_lags+2 "
                "observations for the HAC loss-differential test."
            )
        if (
            not _is_finite_number(self.level_change_reconciliation_tolerance_bp)
            or not 0.0 < self.level_change_reconciliation_tolerance_bp <= 1.0e-3
        ):
            raise ValueError("level_change_reconciliation_tolerance_bp must lie in (0, 1e-3].")
        if not _is_finite_number(self.adoption_alpha) or not 0 < self.adoption_alpha < 1:
            raise ValueError("adoption_alpha must lie in (0, 1).")
        if (
            not _is_finite_number(self.minimum_rmse_improvement)
            or not 0.0 <= self.minimum_rmse_improvement < 1.0
        ):
            raise ValueError("minimum_rmse_improvement must lie in [0, 1).")


@dataclass(frozen=True)
class RiskConfig:
    """Controls linear key-rate risk, tail metrics, and scenario reporting."""

    historical_window_weeks: int = 520
    var_confidence: float = 0.99
    es_confidence: float = 0.975
    backtest_alpha: float = 0.05
    maximum_krd_asof_gap_days: int = 7
    minimum_var_backtest_observations: int = 250
    minimum_effective_tail_mass_review: float = 20.0
    factor_sigma_multiples: tuple[float, ...] = (1.0, 2.0, 3.0)

    def __post_init__(self) -> None:
        if not _is_integer(self.historical_window_weeks) or not 104 <= self.historical_window_weeks <= 10_000:
            raise ValueError("historical_window_weeks must lie in [104, 10000].")
        for name, value in (("var_confidence", self.var_confidence), ("es_confidence", self.es_confidence)):
            if not _is_finite_number(value) or not 0.5 < value < 1.0:
                raise ValueError(f"{name} must lie in (0.5, 1).")
        maximum_predictive_var_confidence = self.historical_window_weeks / (
            self.historical_window_weeks + 1.0
        )
        if self.var_confidence > maximum_predictive_var_confidence:
            raise ValueError(
                "var_confidence is unattainable with historical_window_weeks under the "
                "finite-sample predictive order-statistic convention."
            )
        if not _is_finite_number(self.backtest_alpha) or not 0.0 < self.backtest_alpha < 0.5:
            raise ValueError("backtest_alpha must lie in (0, 0.5).")
        if not _is_integer(self.maximum_krd_asof_gap_days) or not 0 <= self.maximum_krd_asof_gap_days <= 366:
            raise ValueError("maximum_krd_asof_gap_days must be a non-negative integer.")
        if (
            not _is_integer(self.minimum_var_backtest_observations)
            or not 100 <= self.minimum_var_backtest_observations <= 10_000
        ):
            raise ValueError("minimum_var_backtest_observations must be at least 100.")
        if (
            not _is_finite_number(self.minimum_effective_tail_mass_review)
            or not 1.0 <= self.minimum_effective_tail_mass_review <= 10_000.0
        ):
            raise ValueError("minimum_effective_tail_mass_review must lie in [1, 10000].")
        if not isinstance(self.factor_sigma_multiples, tuple):
            raise TypeError("factor_sigma_multiples must be an immutable tuple.")
        if (
            not self.factor_sigma_multiples
            or len(self.factor_sigma_multiples) > 20
            or len(set(self.factor_sigma_multiples)) != len(self.factor_sigma_multiples)
            or any(not _is_finite_number(x) or not 0 < x <= 10 for x in self.factor_sigma_multiples)
        ):
            raise ValueError("factor_sigma_multiples must contain positive values.")


@dataclass(frozen=True)
class PipelineConfig:
    """Complete, validated notebook configuration."""

    data: DataConfig = field(default_factory=DataConfig)
    pca: PCAConfig = field(default_factory=PCAConfig)
    forecast: ForecastConfig = field(default_factory=ForecastConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)

    def __post_init__(self) -> None:
        for name, value, expected in (
            ("data", self.data, DataConfig),
            ("pca", self.pca, PCAConfig),
            ("forecast", self.forecast, ForecastConfig),
            ("risk", self.risk, RiskConfig),
        ):
            if not isinstance(value, expected):
                raise TypeError(f"PipelineConfig.{name} must be {expected.__name__}.")
