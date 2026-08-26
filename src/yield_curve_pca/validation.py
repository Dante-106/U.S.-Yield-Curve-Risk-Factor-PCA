"""Independent numerical challenges, distribution tests, and model health gates."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import chi2, jarque_bera, kurtosis, skew

from .pca import PCAFit


@dataclass(frozen=True)
class ValidationSummary:
    overall_assessment: str
    algebra_checks: pd.DataFrame
    distribution_checks: pd.DataFrame
    model_health: pd.DataFrame


def _model_values(changes_bp: pd.DataFrame, fit: PCAFit) -> np.ndarray:
    if tuple(changes_bp.columns) != fit.tenors:
        raise ValueError("Validation data tenor order differs from the fitted model.")
    values = changes_bp.astype(float).to_numpy()
    if not np.isfinite(values).all():
        raise ValueError("Validation data contain missing or non-finite values.")
    return (values - fit.center_bp) / fit.scale_bp


def _weighted_covariance(values: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    if weights is None:
        centered = values - values.mean(axis=0)
        return centered.T @ centered / (len(values) - 1)
    normalized = weights / weights.sum()
    centered = values - normalized @ values
    correction = 1.0 - float(normalized @ normalized)
    return (centered * normalized[:, None]).T @ centered / correction


def _status(value: float, threshold: float, lower_is_better: bool = True) -> str:
    passed = value <= threshold if lower_is_better else value >= threshold
    return "PASS" if passed else "FAIL"


def validate_pca_algebra(
    changes_bp: pd.DataFrame,
    fit: PCAFit,
    *,
    retained_factors: int = 3,
    minimum_template_similarity: float = 0.70,
    minimum_template_dominance_margin: float = 0.10,
    tolerance: float = 1.0e-10,
) -> pd.DataFrame:
    """Challenge the oriented API with an independent weighted SVD route."""

    model_values = _model_values(changes_bp, fit)
    covariance = _weighted_covariance(model_values, fit.weights)
    transformed = fit.transform(changes_bp)
    reconstructed = fit.inverse_transform(transformed)

    if fit.weights is None:
        challenger_matrix = model_values - model_values.mean(axis=0)
        denominator = len(model_values) - 1
    else:
        weights = fit.weights / fit.weights.sum()
        challenger_centered = model_values - weights @ model_values
        correction = 1.0 - float(weights @ weights)
        challenger_matrix = challenger_centered * np.sqrt(weights[:, None] / correction)
        denominator = 1.0
    _, singular_values, challenger_components = np.linalg.svd(challenger_matrix, full_matrices=False)
    challenger_eigenvalues = singular_values**2 / denominator

    fitted_projector = fit.components[:retained_factors].T @ fit.components[:retained_factors]
    challenger_projector = (
        challenger_components[:retained_factors].T @ challenger_components[:retained_factors]
    )
    projector_error = float(np.linalg.norm(fitted_projector - challenger_projector, ord="fro"))

    score_values = transformed.to_numpy()
    score_covariance = _weighted_covariance(score_values, fit.weights)
    off_diagonal = score_covariance - np.diag(np.diag(score_covariance))
    covariance_reconstruction = fit.components.T @ np.diag(fit.eigenvalues) @ fit.components

    rows = [
        {
            "Control": "Oriented score API consistency",
            "Result": float(np.max(np.abs(transformed.to_numpy() - fit.scores.to_numpy()))),
            "Threshold": tolerance,
            "Status": _status(
                float(np.max(np.abs(transformed.to_numpy() - fit.scores.to_numpy()))), tolerance
            ),
            "Interpretation": "transform(training data) must equal the exported, sign-oriented scores.",
        },
        {
            "Control": "Full-rank round-trip error (bp)",
            "Result": float(np.max(np.abs(reconstructed.to_numpy() - changes_bp.to_numpy()))),
            "Threshold": tolerance,
            "Status": _status(
                float(np.max(np.abs(reconstructed.to_numpy() - changes_bp.to_numpy()))), tolerance
            ),
            "Interpretation": "transform and inverse_transform must use one coherent basis.",
        },
        {
            "Control": "Eigenvector orthonormality",
            "Result": float(np.max(np.abs(fit.components @ fit.components.T - np.eye(len(fit.tenors))))),
            "Threshold": tolerance,
            "Status": _status(
                float(np.max(np.abs(fit.components @ fit.components.T - np.eye(len(fit.tenors))))),
                tolerance,
            ),
            "Interpretation": "The fitted basis must remain orthonormal after economic sign orientation.",
        },
        {
            "Control": "Covariance eigendecomposition error",
            "Result": float(np.max(np.abs(covariance - covariance_reconstruction))),
            "Threshold": tolerance,
            "Status": _status(float(np.max(np.abs(covariance - covariance_reconstruction))), tolerance),
            "Interpretation": "V'ΛV must independently reconstruct the modeled covariance.",
        },
        {
            "Control": "Independent SVD eigenvalue error",
            "Result": float(np.max(np.abs(challenger_eigenvalues - fit.eigenvalues))),
            "Threshold": tolerance,
            "Status": _status(float(np.max(np.abs(challenger_eigenvalues - fit.eigenvalues))), tolerance),
            "Interpretation": "A separate SVD computation must reconcile the eigenvalues.",
        },
        {
            "Control": f"Independent top-{retained_factors} projector error",
            "Result": projector_error,
            "Threshold": 1.0e-8,
            "Status": _status(projector_error, 1.0e-8),
            "Interpretation": "Subspace comparison is sign- and permutation-invariant.",
        },
        {
            "Control": "Maximum factor-score covariance off-diagonal",
            "Result": float(np.max(np.abs(off_diagonal))),
            "Threshold": 1.0e-8,
            "Status": _status(float(np.max(np.abs(off_diagonal))), 1.0e-8),
            "Interpretation": "Scores must diagonalize the calibration covariance under the configured weights.",
        },
        {
            "Control": "Minimum economic-template similarity",
            "Result": float(fit.template_similarity.min()),
            "Threshold": minimum_template_similarity,
            "Status": "PASS" if fit.template_similarity.min() >= minimum_template_similarity else "WARN",
            "Interpretation": "Weak matches must be reported as rotated/unidentified, not silently forced.",
        },
        {
            "Control": "Minimum economic-template dominance margin",
            "Result": float(fit.template_dominance_margin.min()),
            "Threshold": minimum_template_dominance_margin,
            "Status": (
                "PASS" if fit.template_dominance_margin.min() >= minimum_template_dominance_margin else "WARN"
            ),
            "Interpretation": (
                "A component must dominate its next-best template; near-tied economic labels "
                "remain unidentified."
            ),
        },
    ]
    if fit.standardize:
        diagonal_error = float(np.max(np.abs(np.diag(covariance) - 1.0)))
        rows.append(
            {
                "Control": "Correlation-PCA unit diagonal",
                "Result": diagonal_error,
                "Threshold": 1.0e-10,
                "Status": _status(diagonal_error, 1.0e-10),
                "Interpretation": "Standardized tenor changes must have unit modeled variance.",
            }
        )
    return pd.DataFrame(rows).set_index("Control")


def _ljung_box(values: np.ndarray, lags: int = 10) -> tuple[float, float]:
    series = np.asarray(values, dtype=float)
    series = series[np.isfinite(series)]
    n_obs = len(series)
    if n_obs <= lags + 2 or np.std(series, ddof=1) <= np.finfo(float).eps:
        return float("nan"), float("nan")
    centered = series - series.mean()
    denominator = float(centered @ centered)
    autocorrelations = [float(centered[lag:] @ centered[:-lag] / denominator) for lag in range(1, lags + 1)]
    q_statistic = (
        n_obs
        * (n_obs + 2.0)
        * sum(rho**2 / (n_obs - lag) for lag, rho in enumerate(autocorrelations, start=1))
    )
    return float(q_statistic), float(chi2.sf(q_statistic, lags))


def factor_distribution_diagnostics(fit: PCAFit, factors: int = 3, lags: int = 10) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for column in fit.scores.columns[:factors]:
        values = fit.scores[column].to_numpy()
        jb = jarque_bera(values)
        lb_level, lb_level_p = _ljung_box(values, lags)
        lb_square, lb_square_p = _ljung_box(values**2, lags)
        rows.append(
            {
                "Factor": column,
                "Skewness": float(skew(values, bias=False)),
                "Excess kurtosis": float(kurtosis(values, fisher=True, bias=False)),
                "Jarque–Bera statistic": float(jb.statistic),
                "Jarque–Bera p-value": float(jb.pvalue),
                f"Ljung–Box({lags}) statistic": lb_level,
                f"Ljung–Box({lags}) p-value": lb_level_p,
                f"Squared-score LB({lags}) statistic": lb_square,
                f"Squared-score LB({lags}) p-value": lb_square_p,
                "Status": "WARN" if jb.pvalue < 0.05 or lb_square_p < 0.05 else "PASS",
                "Interpretation": (
                    "Gaussian/iid tail scaling is not supported; retain empirical/stressed diagnostics."
                    if jb.pvalue < 0.05 or lb_square_p < 0.05
                    else "No distribution warning at the configured tests."
                ),
            }
        )
    return pd.DataFrame(rows).set_index("Factor")


def model_health_table(
    rolling_stability: pd.DataFrame,
    oos_reconstruction_metrics: pd.DataFrame,
    *,
    maximum_principal_angle_warning: float = 30.0,
    minimum_loading_cosine_warning: float = 0.80,
    maximum_oos_rmse_warning_bp: float = 5.0,
) -> pd.DataFrame:
    cosine_columns = [column for column in rolling_stability if column.endswith(" cosine")]
    maximum_angle = float(rolling_stability["Maximum principal angle (deg)"].max())
    minimum_cosine = float(rolling_stability[cosine_columns].min().min())
    maximum_oos_rmse = float(oos_reconstruction_metrics["RMSE (bp)"].max())
    holdout_observations = int(oos_reconstruction_metrics["Holdout observations"].min())
    rows = [
        {
            "Monitor": "Maximum rolling subspace angle",
            "Result": maximum_angle,
            "Threshold": maximum_principal_angle_warning,
            "Status": "PASS" if maximum_angle <= maximum_principal_angle_warning else "WARN",
            "Action": "Review regime change, eigenvalue separation, and calibration window.",
        },
        {
            "Monitor": "Minimum aligned loading cosine",
            "Result": minimum_cosine,
            "Threshold": minimum_loading_cosine_warning,
            "Status": "PASS" if minimum_cosine >= minimum_loading_cosine_warning else "WARN",
            "Action": "Do not treat rotated PC2/PC3 identities as stable limit factors.",
        },
        {
            "Monitor": "Maximum tenor OOS reconstruction RMSE (bp)",
            "Result": maximum_oos_rmse,
            "Threshold": maximum_oos_rmse_warning_bp,
            "Status": (
                "PASS"
                if holdout_observations > 0 and maximum_oos_rmse <= maximum_oos_rmse_warning_bp
                else "WARN"
            ),
            "Action": (
                "Not evaluated: preserve full key-rate risk and obtain the configured holdout."
                if holdout_observations == 0
                else "Retain key-rate and localized residual stresses where top-3 reconstruction is weak."
            ),
        },
    ]
    return pd.DataFrame(rows).set_index("Monitor")


def assemble_validation_summary(
    algebra_checks: pd.DataFrame,
    distribution_checks: pd.DataFrame,
    health_checks: pd.DataFrame,
    additional_statuses: tuple[str, ...] = (),
) -> ValidationSummary:
    statuses = (
        list(algebra_checks["Status"])
        + list(distribution_checks["Status"])
        + list(health_checks["Status"])
        + list(additional_statuses)
    )
    if "FAIL" in statuses:
        assessment = "NEEDS REVISION"
    elif "WARN" in statuses:
        assessment = "SHARE WITH CAVEATS"
    else:
        assessment = "READY TO SHARE"
    return ValidationSummary(assessment, algebra_checks, distribution_checks, health_checks)
