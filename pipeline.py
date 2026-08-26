"""Orchestration layer for the auditable notebook and command-line use."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .config import ALL_TENORS, PipelineConfig
from .data import CurveDataBundle, load_curve_data
from .forecast import ForecastResult, required_forecast_observations, walk_forward_forecast
from .pca import (
    PCAFit,
    bootstrap_block_length_sensitivity,
    compare_pca_regimes,
    ewma_weights,
    expanding_oos_reconstruction,
    fit_curve_pca,
    moving_block_bootstrap_stability,
    reconstruction_diagnostics,
    rolling_stability,
    sequential_window_stability,
)
from .validation import (
    ValidationSummary,
    assemble_validation_summary,
    factor_distribution_diagnostics,
    model_health_table,
    validate_pca_algebra,
)

_THIRTY_YEAR_CMT_CONTINUOUS_START = "2006-02-09"


@dataclass(frozen=True)
class PipelineResult:
    data: CurveDataBundle
    structural_pca: PCAFit
    current_ewma_pca: PCAFit
    correlation_pca: PCAFit
    structural_current_ewma_comparison: pd.DataFrame
    specification_challenge: pd.DataFrame
    methodology_sensitivity: pd.DataFrame
    reconstruction_summary: pd.DataFrame
    reconstruction_by_tenor: pd.DataFrame
    rolling_stability: pd.DataFrame
    sequential_stability: pd.DataFrame
    bootstrap_stability: pd.DataFrame
    bootstrap_block_sensitivity: pd.DataFrame
    oos_reconstruction: pd.DataFrame
    oos_reconstruction_metrics: pd.DataFrame
    oos_reconstruction_audit: pd.DataFrame
    forecast: ForecastResult | None
    forecast_note: str
    validation: ValidationSummary


def _specification_challenge(
    bundle: CurveDataBundle,
    structural: PCAFit,
    correlation: PCAFit,
    current: PCAFit,
    retained_factors: int,
    weekly_rule: str,
) -> pd.DataFrame:
    rows = []
    for label, fit, note in (
        ("Structural covariance PCA", structural, "Full core history; equal time weights"),
        ("Structural correlation PCA", correlation, "Full core history; tenor volatility standardized"),
        ("Current EWMA covariance PCA", current, "Full core history; exponentially weighted"),
    ):
        rows.append(
            {
                "Specification": label,
                "Start": fit.training_start.date(),
                "End": fit.training_end.date(),
                "Tenors": len(fit.tenors),
                "Observations": len(fit.scores),
                "Effective observations": fit.effective_observations,
                "PC1": fit.explained_ratio[0],
                "PC2": fit.explained_ratio[1],
                "PC3": fit.explained_ratio[2],
                f"Top-{retained_factors} cumulative": fit.explained_ratio[:retained_factors].sum(),
                "Minimum template similarity": fit.template_similarity.min(),
                "Minimum template dominance margin": fit.template_dominance_margin.min(),
                "Control note": note,
            }
        )
    daily_extended = bundle.daily_yields_pct.loc[_THIRTY_YEAR_CMT_CONTINUOUS_START:, list(ALL_TENORS)].dropna(
        how="any"
    )
    if len(daily_extended) >= 60:
        extended_weekly = daily_extended.groupby(daily_extended.index.to_period(weekly_rule), sort=True).tail(
            1
        )
        extended_weekly = extended_weekly.loc[: bundle.weekly_yields_pct.index.max()]
        extended_changes = extended_weekly.diff().iloc[1:] * 100.0
        if len(extended_changes) >= 52:
            common_core_fit = fit_curve_pca(
                extended_changes.loc[:, list(structural.tenors)],
                minimum_template_similarity=structural.minimum_template_similarity,
                minimum_template_dominance_margin=structural.minimum_template_dominance_margin,
            )
            extended_fit = fit_curve_pca(
                extended_changes,
                minimum_template_similarity=structural.minimum_template_similarity,
                minimum_template_dominance_margin=structural.minimum_template_dominance_margin,
            )
            for label, fit, note in (
                (
                    f"Common-period core {len(structural.tenors)}-tenor PCA",
                    common_core_fit,
                    f"Same rows as the {len(extended_fit.tenors)}-tenor challenge",
                ),
                ("Common-period extended 11-tenor PCA", extended_fit, "Adds 1M and 30Y on identical rows"),
            ):
                rows.append(
                    {
                        "Specification": label,
                        "Start": fit.training_start.date(),
                        "End": fit.training_end.date(),
                        "Tenors": len(fit.tenors),
                        "Observations": len(fit.scores),
                        "Effective observations": fit.effective_observations,
                        "PC1": fit.explained_ratio[0],
                        "PC2": fit.explained_ratio[1],
                        "PC3": fit.explained_ratio[2],
                        f"Top-{retained_factors} cumulative": fit.explained_ratio[:retained_factors].sum(),
                        "Minimum template similarity": fit.template_similarity.min(),
                        "Minimum template dominance margin": fit.template_dominance_margin.min(),
                        "Control note": note,
                    }
                )
    return pd.DataFrame(rows).set_index("Specification")


def _methodology_sensitivity(changes: pd.DataFrame, config: PipelineConfig) -> pd.DataFrame:
    """Compare covariance PCA before and after Treasury's 2021 methodology change."""

    cutoff = pd.Timestamp("2021-12-06")
    pre = changes.loc[changes.index < cutoff]
    post_candidates = changes.index[changes.index >= cutoff]
    if len(post_candidates):
        post = changes.loc[changes.index > post_candidates[0]]
    else:
        post = changes.iloc[0:0]
    minimum = config.pca.minimum_methodology_regime_observations
    if len(pre) < minimum or len(post) < minimum:
        return pd.DataFrame(
            {
                "Status": ["NOT EVALUATED"],
                "Reference observations": [len(pre)],
                "Candidate observations": [len(post)],
                "Minimum required per regime": [minimum],
                "Interpretation": ["Sample does not provide two sufficiently large methodology regimes."],
            },
            index=pd.Index(["Treasury methodology regime"], name="Control"),
        )
    pre_fit = fit_curve_pca(
        pre,
        standardize=False,
        minimum_template_similarity=config.pca.minimum_template_similarity,
        minimum_template_dominance_margin=config.pca.minimum_template_dominance_margin,
    )
    post_fit = fit_curve_pca(
        post,
        standardize=False,
        minimum_template_similarity=config.pca.minimum_template_similarity,
        minimum_template_dominance_margin=config.pca.minimum_template_dominance_margin,
    )
    comparison = compare_pca_regimes(pre_fit, post_fit, config.pca.retained_factors)
    comparison["Pre top-3 variance"] = pre_fit.explained_ratio[:3].sum()
    comparison["Post top-3 variance"] = post_fit.explained_ratio[:3].sum()
    comparison["Status"] = "EVALUATED"
    return comparison


def run_pipeline(config: PipelineConfig, project_root: Path | None = None) -> PipelineResult:
    """Execute the deterministic, end-to-end research and validation workflow."""

    bundle = load_curve_data(config.data, project_root=project_root)
    changes = bundle.weekly_changes_bp
    retained = config.pca.retained_factors
    structural = fit_curve_pca(
        changes,
        standardize=False,
        minimum_template_similarity=config.pca.minimum_template_similarity,
        minimum_template_dominance_margin=config.pca.minimum_template_dominance_margin,
    )
    correlation = fit_curve_pca(
        changes,
        standardize=True,
        minimum_template_similarity=config.pca.minimum_template_similarity,
        minimum_template_dominance_margin=config.pca.minimum_template_dominance_margin,
    )
    current_weights = ewma_weights(len(changes), config.pca.current_halflife_weeks)
    current_effective_observations = 1.0 / float(current_weights @ current_weights)
    if current_effective_observations < config.pca.minimum_ewma_effective_observations:
        raise ValueError(
            "EWMA effective observations fall below the configured minimum: "
            f"{current_effective_observations:.1f} < "
            f"{config.pca.minimum_ewma_effective_observations:.1f}."
        )
    current = fit_curve_pca(
        changes,
        standardize=False,
        weights=current_weights,
        minimum_template_similarity=config.pca.minimum_template_similarity,
        minimum_template_dominance_margin=config.pca.minimum_template_dominance_margin,
    )
    structural_current_comparison = compare_pca_regimes(structural, current, retained)

    reconstruction_summary, reconstruction_by_tenor = reconstruction_diagnostics(
        changes, structural, retained
    )
    rolling_window = min(52 * config.pca.rolling_window_years, len(changes))
    if rolling_window < config.pca.minimum_rolling_observations:
        raise ValueError("Available history is shorter than PCAConfig.minimum_rolling_observations.")
    rolling = rolling_stability(
        changes,
        structural,
        window_observations=rolling_window,
        step_observations=13,
        factors=retained,
        standardize=False,
    )
    if len(changes) >= 2 * rolling_window:
        sequential = sequential_window_stability(
            changes,
            window_observations=rolling_window,
            step_observations=13,
            factors=retained,
            standardize=False,
            minimum_template_similarity=config.pca.minimum_template_similarity,
            minimum_template_dominance_margin=(config.pca.minimum_template_dominance_margin),
        )
    else:
        sequential = pd.DataFrame(
            columns=(
                "Reference start",
                "Reference end",
                "Monitoring start",
                "Observations per window",
                "Reference mode",
                f"Monitoring top-{retained} variance",
                "Maximum principal angle (deg)",
                "Minimum aligned loading cosine",
            )
        )
    bootstrap = moving_block_bootstrap_stability(
        changes,
        structural,
        replications=config.pca.bootstrap_replications,
        block_length=config.pca.bootstrap_block_weeks,
        factors=retained,
        random_seed=config.pca.random_seed,
    )
    bootstrap_sensitivity = bootstrap_block_length_sensitivity(
        changes,
        structural,
        block_lengths=config.pca.bootstrap_sensitivity_blocks,
        replications=config.pca.bootstrap_sensitivity_replications,
        factors=retained,
        random_seed=config.pca.random_seed,
    )
    required_oos_sample = (
        config.pca.oos_minimum_training_observations + config.pca.oos_minimum_holdout_observations
    )
    if len(changes) >= required_oos_sample:
        oos_reconstruction, oos_metrics, oos_audit = expanding_oos_reconstruction(
            changes,
            minimum_training_observations=config.pca.oos_minimum_training_observations,
            retained_factors=retained,
            refit_every=config.pca.oos_refit_every_observations,
            standardize=False,
        )
    else:
        oos_reconstruction = pd.DataFrame(columns=changes.columns, dtype=float)
        oos_metrics = pd.DataFrame(
            {
                "MAE (bp)": float("nan"),
                "RMSE (bp)": float("nan"),
                "P95 absolute error (bp)": float("nan"),
                "Maximum absolute error (bp)": float("nan"),
                "Holdout observations": 0,
            },
            index=changes.columns,
        )
        oos_audit = pd.DataFrame(
            columns=(
                "Target position",
                "Basis refit sequence",
                "Basis refit trigger position",
                "Maximum source position",
                "Training start",
                "Training end",
                "Training observations",
            ),
            index=pd.DatetimeIndex([], name="Target date"),
        )
    minimum_forecast_sample = required_forecast_observations(config.forecast)
    if config.forecast.enabled and len(changes) >= minimum_forecast_sample:
        forecast = walk_forward_forecast(
            changes,
            bundle.weekly_yields_pct,
            config.forecast,
            weekly_rule=config.data.weekly_rule,
        )
        forecast_note = "Executed strict expanding-origin one-week-ahead validation."
    elif config.forecast.enabled:
        forecast = None
        forecast_note = (
            f"Skipped: {len(changes)} observations do not exceed the configured "
            f"training-plus-interval-validation minimum of {minimum_forecast_sample}."
        )
    else:
        forecast = None
        forecast_note = "Disabled by configuration."

    algebra = pd.concat(
        {
            "Structural": validate_pca_algebra(
                changes,
                structural,
                retained_factors=retained,
                minimum_template_similarity=config.pca.minimum_template_similarity,
                minimum_template_dominance_margin=config.pca.minimum_template_dominance_margin,
            ),
            "Correlation challenge": validate_pca_algebra(
                changes,
                correlation,
                retained_factors=retained,
                minimum_template_similarity=config.pca.minimum_template_similarity,
                minimum_template_dominance_margin=config.pca.minimum_template_dominance_margin,
            ),
            "Current EWMA challenge": validate_pca_algebra(
                changes,
                current,
                retained_factors=retained,
                minimum_template_similarity=config.pca.minimum_template_similarity,
                minimum_template_dominance_margin=config.pca.minimum_template_dominance_margin,
            ),
        },
        names=("Calibration", "Control"),
    )
    distribution = factor_distribution_diagnostics(structural, retained)
    methodology = _methodology_sensitivity(changes, config)
    health = model_health_table(
        rolling,
        oos_metrics,
        maximum_principal_angle_warning=config.pca.maximum_principal_angle_warning_degrees,
        minimum_loading_cosine_warning=config.pca.minimum_loading_cosine_warning,
        maximum_oos_rmse_warning_bp=config.pca.maximum_oos_rmse_warning_bp,
    )
    if sequential.empty:
        health.loc["Sequential prior-window subspace angle"] = {
            "Result": float("nan"),
            "Threshold": config.pca.maximum_principal_angle_warning_degrees,
            "Status": "WARN",
            "Action": "Obtain two full monitoring windows before interpreting sequential drift.",
        }
    else:
        sequential_max_angle = float(sequential["Maximum principal angle (deg)"].max())
        cosine_columns = [column for column in sequential.columns if column.endswith(" cosine")]
        sequential_minimum_cosine = float(sequential[cosine_columns].min().min())
        latest_sequential_angle = float(sequential["Maximum principal angle (deg)"].iloc[-1])
        latest_sequential_cosine = float(sequential.loc[sequential.index[-1], cosine_columns].min())
        health.loc["Sequential prior-window subspace angle"] = {
            "Result": sequential_max_angle,
            "Threshold": config.pca.maximum_principal_angle_warning_degrees,
            "Status": (
                "PASS"
                if sequential_max_angle <= config.pca.maximum_principal_angle_warning_degrees
                else "WARN"
            ),
            "Action": (
                "Escalate chronologically separated adjacent-window drift; this is not a "
                "historical-vintage decision replay."
            ),
        }
        health.loc["Sequential prior-window minimum loading cosine"] = {
            "Result": sequential_minimum_cosine,
            "Threshold": config.pca.minimum_loading_cosine_warning,
            "Status": (
                "PASS" if sequential_minimum_cosine >= config.pca.minimum_loading_cosine_warning else "WARN"
            ),
            "Action": "Do not assume a universally stable PC2/PC3 factor taxonomy.",
        }
        health.loc["Latest sequential subspace angle"] = {
            "Result": latest_sequential_angle,
            "Threshold": config.pca.maximum_principal_angle_warning_degrees,
            "Status": (
                "PASS"
                if latest_sequential_angle <= config.pca.maximum_principal_angle_warning_degrees
                else "WARN"
            ),
            "Action": "Read with the historical-worst adjacent-window result, not in isolation.",
        }
        health.loc["Latest sequential minimum loading cosine"] = {
            "Result": latest_sequential_cosine,
            "Threshold": config.pca.minimum_loading_cosine_warning,
            "Status": (
                "PASS" if latest_sequential_cosine >= config.pca.minimum_loading_cosine_warning else "WARN"
            ),
            "Action": "Read with the historical-worst adjacent-window result, not in isolation.",
        }
    if set(methodology["Status"]) == {"EVALUATED"}:
        maximum_methodology_angle = float(methodology["Ordered subspace principal angle (deg)"].max())
        sigma_ratios = methodology["Candidate/reference sigma ratio"].astype(float)
        maximum_symmetric_sigma_ratio = float(pd.concat((sigma_ratios, 1.0 / sigma_ratios)).max())
        health.loc["Treasury methodology subspace sensitivity"] = {
            "Result": maximum_methodology_angle,
            "Threshold": config.pca.maximum_methodology_subspace_angle_warning_degrees,
            "Status": (
                "PASS"
                if maximum_methodology_angle <= config.pca.maximum_methodology_subspace_angle_warning_degrees
                else "WARN"
            ),
            "Action": "Do not attribute pre/post differences solely to market regimes.",
        }
        health.loc["Treasury methodology factor-volatility sensitivity"] = {
            "Result": maximum_symmetric_sigma_ratio,
            "Threshold": config.pca.maximum_methodology_sigma_ratio_warning,
            "Status": (
                "PASS"
                if maximum_symmetric_sigma_ratio <= config.pca.maximum_methodology_sigma_ratio_warning
                else "WARN"
            ),
            "Action": "Review limit calibration across the source-methodology boundary.",
        }
    else:
        health.loc["Treasury methodology sensitivity"] = {
            "Result": float("nan"),
            "Threshold": config.pca.minimum_methodology_regime_observations,
            "Status": "WARN",
            "Action": "Obtain enough pre/post observations or explicitly waive this challenge.",
        }
    maximum_current_angle = float(
        structural_current_comparison["Ordered subspace principal angle (deg)"].max()
    )
    current_sigma_ratios = structural_current_comparison["Candidate/reference sigma ratio"].astype(float)
    maximum_current_symmetric_sigma_ratio = float(
        pd.concat((current_sigma_ratios, 1.0 / current_sigma_ratios)).max()
    )
    health.loc["Current EWMA subspace sensitivity"] = {
        "Result": maximum_current_angle,
        "Threshold": config.pca.maximum_principal_angle_warning_degrees,
        "Status": (
            "PASS" if maximum_current_angle <= config.pca.maximum_principal_angle_warning_degrees else "WARN"
        ),
        "Action": "Review recency-weighted factor rotation against the structural reference.",
    }
    health.loc["Current EWMA factor-volatility sensitivity"] = {
        "Result": maximum_current_symmetric_sigma_ratio,
        "Threshold": config.pca.maximum_current_factor_sigma_ratio_warning,
        "Status": (
            "PASS"
            if maximum_current_symmetric_sigma_ratio <= config.pca.maximum_current_factor_sigma_ratio_warning
            else "WARN"
        ),
        "Action": "Review absolute factor shock calibration, not explained shares alone.",
    }
    if forecast is not None:
        independent_statuses = set(forecast.interval_coverage["Status"])
        independent_status = "PASS" if independent_statuses == {"PASS"} else "WARN"
        evidence_margin = (
            forecast.interval_coverage["Familywise block-bootstrap coverage lower bound"]
            - forecast.interval_coverage["Acceptable coverage floor"]
        )
        health.loc["Independent interval calibration evidence margin"] = {
            "Result": float(evidence_margin.min()),
            "Threshold": 0.0,
            "Status": independent_status,
            "Action": (
                "PASS requires every tenor's dependence-aware, familywise block-bootstrap "
                "lower bound to meet the review floor; INCONCLUSIVE is conservatively escalated."
            ),
        }
        long_statuses = set(forecast.full_history_interval_diagnostic["Status"])
        long_status = "PASS" if long_statuses == {"PASS"} else "WARN"
        health.loc["Long-history interval calibration diagnostic"] = {
            "Result": float(forecast.full_history_interval_diagnostic["Coverage gap"].min()),
            "Threshold": -config.forecast.interval_coverage_tolerance,
            "Status": long_status,
            "Action": (
                "Treat as a retrospective selected-model diagnostic; review regime-conditional "
                "or validated conformal intervals before any limit use."
            ),
        }
    validation = assemble_validation_summary(
        algebra,
        distribution,
        health,
        additional_statuses=tuple(bundle.quality_table["status"]),
    )

    return PipelineResult(
        data=bundle,
        structural_pca=structural,
        current_ewma_pca=current,
        correlation_pca=correlation,
        structural_current_ewma_comparison=structural_current_comparison,
        specification_challenge=_specification_challenge(
            bundle,
            structural,
            correlation,
            current,
            retained,
            config.data.weekly_rule,
        ),
        methodology_sensitivity=methodology,
        reconstruction_summary=reconstruction_summary,
        reconstruction_by_tenor=reconstruction_by_tenor,
        rolling_stability=rolling,
        sequential_stability=sequential,
        bootstrap_stability=bootstrap,
        bootstrap_block_sensitivity=bootstrap_sensitivity,
        oos_reconstruction=oos_reconstruction,
        oos_reconstruction_metrics=oos_metrics,
        oos_reconstruction_audit=oos_audit,
        forecast=forecast,
        forecast_note=forecast_note,
        validation=validation,
    )
