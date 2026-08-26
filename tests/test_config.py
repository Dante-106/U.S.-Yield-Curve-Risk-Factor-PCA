import re

import pytest

import yield_curve_pca
from yield_curve_pca.config import DataConfig, ForecastConfig, PCAConfig, RiskConfig


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DataConfig(start_date="2025-01-01", end_date="2024-01-01"),
        lambda: DataConfig(core_tenors=("3M", "3M", "2Y")),
        lambda: DataConfig(weekly_rule="W-FOO"),
        lambda: DataConfig(maximum_staleness_days=-1),
        lambda: DataConfig(maximum_start_coverage_gap_days=-1),
        lambda: DataConfig(maximum_within_week_observation_lag_days=-1),
        lambda: PCAConfig(current_halflife_weeks=1),
        lambda: PCAConfig(current_halflife_weeks=float("nan")),
        lambda: PCAConfig(minimum_ewma_effective_observations=10),
        lambda: PCAConfig(bootstrap_replications=0),
        lambda: PCAConfig(maximum_current_factor_sigma_ratio_warning=1.0),
        lambda: PCAConfig(bootstrap_replications=799),
        lambda: PCAConfig(bootstrap_sensitivity_replications=799),
        lambda: PCAConfig(rolling_window_years=2, minimum_rolling_observations=200),
        lambda: ForecastConfig(minimum_training_weeks=50),
        lambda: ForecastConfig(interval_coverage_tolerance=0.20),
        lambda: ForecastConfig(level_change_reconciliation_tolerance_bp=float("nan")),
        lambda: ForecastConfig(hac_lags=2.5),
        lambda: ForecastConfig(
            minimum_model_selection_observations=52,
            confirmation_observations=52,
            hac_lags=26,
        ),
        lambda: ForecastConfig(var_ridge=float("nan")),
        lambda: RiskConfig(var_confidence=1.0),
        lambda: RiskConfig(var_confidence=0.999),
        lambda: RiskConfig(factor_sigma_multiples=(1.0, float("nan"))),
        lambda: DataConfig(source_mode="bogus"),
        lambda: DataConfig(request_timeout_seconds=float("nan")),
        lambda: DataConfig(request_timeout_seconds=121),
        lambda: DataConfig(retries=11),
        lambda: DataConfig(maximum_response_bytes=250_000_001),
        lambda: PCAConfig(bootstrap_replications=100_001),
        lambda: PCAConfig(bootstrap_sensitivity_replications=20_001),
        lambda: ForecastConfig(interval_bootstrap_replications=200_001),
        lambda: ForecastConfig(
            interval_evaluation_observations=51,
            interval_bootstrap_block_lengths=(4, 13, 26),
        ),
        lambda: RiskConfig(factor_sigma_multiples=(11.0,)),
        lambda: RiskConfig(minimum_effective_tail_mass_review=0.999),
    ],
)
def test_invalid_configuration_fails_before_work(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DataConfig(core_tenors=["3M", "2Y", "10Y"]),
        lambda: PCAConfig(bootstrap_sensitivity_blocks=[4, 13, 26]),
        lambda: RiskConfig(factor_sigma_multiples=[1.0, 2.0]),
        lambda: ForecastConfig(interval_bootstrap_block_lengths=[4, 13, 26]),
    ],
)
def test_mutable_sequence_configuration_is_rejected(factory):
    with pytest.raises(TypeError, match="immutable tuple"):
        factory()


def test_package_and_project_versions_are_consistent(project_root):
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, flags=re.MULTILINE)
    assert match is not None
    assert match.group(1) == yield_curve_pca.__version__
