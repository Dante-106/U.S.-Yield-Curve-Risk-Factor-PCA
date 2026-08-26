"""Economically identified PCA and stability diagnostics for yield changes."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from itertools import permutations

import numpy as np
import pandas as pd

from .config import MATURITY_YEARS

ECONOMIC_SHAPES: tuple[str, ...] = ("Level", "Slope", "Curvature")


def _validate_changes(changes_bp: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(changes_bp, pd.DataFrame):
        raise TypeError("changes_bp must be a pandas DataFrame.")
    if changes_bp.shape[0] < 3 or changes_bp.shape[1] < 3:
        raise ValueError("PCA requires at least three observations and three tenors.")
    if changes_bp.columns.duplicated().any() or changes_bp.index.duplicated().any():
        raise ValueError("PCA input labels must be unique.")
    unknown = set(changes_bp.columns) - set(MATURITY_YEARS)
    if unknown:
        raise ValueError(f"Unknown tenor labels: {sorted(unknown)}.")
    try:
        frame = changes_bp.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("PCA input must be numeric.") from exc
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("PCA input contains missing or non-finite values.")
    if (frame.std(ddof=1) <= np.finfo(float).eps).any():
        bad = frame.columns[frame.std(ddof=1) <= np.finfo(float).eps].tolist()
        raise ValueError(f"PCA input has zero-variance tenors: {bad}.")
    return frame


def economic_shape_templates(maturities_years: Iterable[float]) -> np.ndarray:
    """Return orthonormal level/slope/curvature templates by log maturity."""

    maturities = np.asarray(tuple(maturities_years), dtype=float)
    if (
        maturities.ndim != 1
        or len(maturities) < 3
        or not np.isfinite(maturities).all()
        or np.any(maturities <= 0)
        or len(np.unique(maturities)) != len(maturities)
    ):
        raise ValueError("maturities_years must contain at least three distinct positive values.")
    log_maturity = np.log(maturities)
    centered = log_maturity - log_maturity.mean()
    design = np.column_stack((np.ones_like(centered), centered, centered**2))
    q, _ = np.linalg.qr(design)
    templates = q.T
    if templates[0].mean() < 0:
        templates[0] *= -1
    shortest = int(np.argmin(maturities))
    longest = int(np.argmax(maturities))
    if templates[1, longest] - templates[1, shortest] < 0:
        templates[1] *= -1
    curvature_target = -(centered**2)
    if float(templates[2] @ curvature_target) < 0:
        templates[2] *= -1
    return templates


def _best_assignment(similarity: np.ndarray) -> tuple[int, ...]:
    if similarity.shape != (3, 3):
        raise ValueError("Economic assignment requires a 3x3 similarity matrix.")
    return max(permutations(range(3)), key=lambda order: sum(similarity[i, order[i]] for i in range(3)))


@dataclass(frozen=True)
class PCAFit:
    """One internally consistent, economically oriented PCA calibration.

    ``components`` are unit-norm eigenvectors in model space.  The physical
    curve shock for one unit of factor score is ``components * scale_bp``.
    This distinction is essential for correlation PCA.
    """

    tenors: tuple[str, ...]
    center_bp: np.ndarray
    scale_bp: np.ndarray
    components: np.ndarray
    eigenvalues: np.ndarray
    explained_ratio: np.ndarray
    scores: pd.DataFrame
    suggested_shapes: tuple[str, ...]
    assigned_shapes: tuple[str, ...]
    identification_status: tuple[str, ...]
    template_similarity: np.ndarray
    template_dominance_margin: np.ndarray
    templates: np.ndarray
    standardize: bool
    weights: np.ndarray | None
    effective_observations: float
    minimum_template_similarity: float
    minimum_template_dominance_margin: float
    training_start: pd.Timestamp
    training_end: pd.Timestamp

    @property
    def component_names(self) -> tuple[str, ...]:
        names: list[str] = []
        for index in range(len(self.tenors)):
            label = self.assigned_shapes[index] if index < len(self.assigned_shapes) else "Residual"
            names.append(f"PC{index + 1} — {label}")
        return tuple(names)

    @property
    def shock_basis_bp_per_score(self) -> np.ndarray:
        return self.components * self.scale_bp[None, :]

    @property
    def one_sigma_shocks_bp(self) -> np.ndarray:
        return self.shock_basis_bp_per_score * np.sqrt(self.eigenvalues)[:, None]

    @property
    def score_unit(self) -> str:
        return "standardized-score units" if self.standardize else "bp projection units"

    def transform(self, changes_bp: pd.DataFrame) -> pd.DataFrame:
        if tuple(changes_bp.columns) != self.tenors:
            raise ValueError(f"Expected tenor order {self.tenors}; received {tuple(changes_bp.columns)}.")
        values = changes_bp.astype(float).to_numpy()
        if not np.isfinite(values).all():
            raise ValueError("transform input contains missing or non-finite values.")
        model_values = (values - self.center_bp) / self.scale_bp
        scores = model_values @ self.components.T
        return pd.DataFrame(scores, index=changes_bp.index, columns=self.component_names)

    def inverse_transform(self, factor_scores: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        index = None
        if isinstance(factor_scores, pd.DataFrame):
            index = factor_scores.index
            expected = self.component_names[: factor_scores.shape[1]]
            if tuple(factor_scores.columns) != expected:
                raise ValueError(
                    f"Factor-score columns must be exactly {expected}; "
                    f"received {tuple(factor_scores.columns)}."
                )
            values = factor_scores.to_numpy(dtype=float)
        else:
            values = np.asarray(factor_scores, dtype=float)
        if values.ndim == 1:
            values = values.reshape(1, -1)
        if values.ndim != 2 or not 1 <= values.shape[1] <= len(self.tenors):
            raise ValueError("factor_scores must have between one and the full number of components.")
        if not np.isfinite(values).all():
            raise ValueError("factor_scores contain missing or non-finite values.")
        reconstructed_model = values @ self.components[: values.shape[1]]
        reconstructed_bp = reconstructed_model * self.scale_bp + self.center_bp
        return pd.DataFrame(reconstructed_bp, index=index, columns=self.tenors)

    def loading_table(self, factors: int = 3) -> pd.DataFrame:
        count = min(factors, len(self.tenors))
        return pd.DataFrame(
            self.components[:count].T,
            index=self.tenors,
            columns=self.component_names[:count],
        )

    def physical_basis_table(self, factors: int = 3) -> pd.DataFrame:
        count = min(factors, len(self.tenors))
        return pd.DataFrame(
            self.shock_basis_bp_per_score[:count].T,
            index=self.tenors,
            columns=self.component_names[:count],
        )

    def sigma_shock_table(self, factors: int = 3) -> pd.DataFrame:
        count = min(factors, len(self.tenors))
        return pd.DataFrame(
            self.one_sigma_shocks_bp[:count].T,
            index=self.tenors,
            columns=self.component_names[:count],
        )

    def tenor_factor_correlation(self, factors: int = 3) -> pd.DataFrame:
        count = min(factors, len(self.tenors))
        physical_covariance = (
            self.shock_basis_bp_per_score.T @ np.diag(self.eigenvalues) @ self.shock_basis_bp_per_score
        )
        tenor_vol = np.sqrt(np.clip(np.diag(physical_covariance), 0.0, None))
        numerator = self.shock_basis_bp_per_score[:count].T * np.sqrt(self.eigenvalues[:count])
        correlations = np.divide(
            numerator,
            tenor_vol[:, None],
            out=np.zeros_like(numerator),
            where=tenor_vol[:, None] > 0,
        )
        return pd.DataFrame(correlations, index=self.tenors, columns=self.component_names[:count])

    def component_index(self, shape: str) -> int:
        if shape not in ECONOMIC_SHAPES:
            raise ValueError(f"shape must be one of {ECONOMIC_SHAPES}.")
        try:
            return self.assigned_shapes.index(shape)
        except ValueError as exc:
            raise KeyError(f"No component was assigned to {shape}.") from exc


def ewma_weights(observations: int, halflife_weeks: float) -> np.ndarray:
    if (
        not isinstance(observations, int)
        or isinstance(observations, bool)
        or observations < 3
        or not np.isfinite(halflife_weeks)
        or halflife_weeks <= 1
    ):
        raise ValueError("EWMA requires at least three observations and a half-life above one.")
    age = np.arange(observations - 1, -1, -1, dtype=float)
    weights = np.exp(np.log(0.5) * age / halflife_weeks)
    return weights / weights.sum()


def fit_curve_pca(
    changes_bp: pd.DataFrame,
    *,
    standardize: bool = False,
    weights: np.ndarray | None = None,
    minimum_template_similarity: float = 0.70,
    minimum_template_dominance_margin: float = 0.10,
) -> PCAFit:
    """Fit a full-rank covariance or correlation PCA with consistent signs."""

    frame = _validate_changes(changes_bp)
    if not isinstance(standardize, bool):
        raise TypeError("standardize must be boolean.")
    if not 0.0 < minimum_template_similarity <= 1.0:
        raise ValueError("minimum_template_similarity must lie in (0, 1].")
    if not 0.0 <= minimum_template_dominance_margin < 1.0:
        raise ValueError("minimum_template_dominance_margin must lie in [0, 1).")
    values = frame.to_numpy()
    observations, dimensions = values.shape

    normalized_weights: np.ndarray | None
    if weights is None:
        normalized_weights = None
        center = values.mean(axis=0)
        centered = values - center
        raw_covariance = centered.T @ centered / (observations - 1)
        effective_observations = float(observations)
    else:
        if (
            not isinstance(frame.index, pd.DatetimeIndex)
            or not frame.index.is_unique
            or not frame.index.is_monotonic_increasing
        ):
            raise ValueError(
                "Position-weighted PCA requires a unique, ascending DatetimeIndex so "
                "recency weights cannot be assigned to the wrong dates."
            )
        normalized_weights = np.asarray(weights, dtype=float)
        if normalized_weights.shape != (observations,) or not np.isfinite(normalized_weights).all():
            raise ValueError("weights must be a finite vector with one value per observation.")
        if (normalized_weights < 0).any() or normalized_weights.sum() <= 0:
            raise ValueError("weights must be non-negative with a positive sum.")
        normalized_weights = normalized_weights / normalized_weights.sum()
        center = normalized_weights @ values
        centered = values - center
        correction = 1.0 - float(normalized_weights @ normalized_weights)
        if correction <= np.finfo(float).eps:
            raise ValueError("weights have insufficient effective observations.")
        raw_covariance = (centered * normalized_weights[:, None]).T @ centered / correction
        effective_observations = 1.0 / float(normalized_weights @ normalized_weights)

    raw_variance = np.diag(raw_covariance)
    if np.any(raw_variance <= np.finfo(float).eps):
        bad = frame.columns[raw_variance <= np.finfo(float).eps].tolist()
        raise ValueError(f"PCA covariance is degenerate for tenors: {bad}.")
    scale = np.sqrt(raw_variance) if standardize else np.ones(dimensions)
    model_values = centered / scale

    if normalized_weights is None:
        covariance = model_values.T @ model_values / (observations - 1)
    else:
        correction = 1.0 - float(normalized_weights @ normalized_weights)
        covariance = (model_values * normalized_weights[:, None]).T @ model_values / correction

    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    components = eigenvectors[:, order].T
    scores = model_values @ components.T

    maturities = [MATURITY_YEARS[tenor] for tenor in frame.columns]
    templates = economic_shape_templates(maturities)
    physical_identification_basis = components[:3] * scale[None, :]
    physical_identification_basis /= np.linalg.norm(
        physical_identification_basis,
        axis=1,
        keepdims=True,
    )
    similarity = np.abs(physical_identification_basis @ templates.T)
    assignment = _best_assignment(similarity)
    suggested_shapes = tuple(ECONOMIC_SHAPES[index] for index in assignment)
    matched_similarity = np.array([similarity[i, assignment[i]] for i in range(3)])
    dominance_margin = np.array(
        [
            matched_similarity[i]
            - max(similarity[i, template] for template in range(3) if template != assignment[i])
            for i in range(3)
        ]
    )
    identification_status = tuple(
        (
            "IDENTIFIED"
            if similarity_value >= minimum_template_similarity and margin >= minimum_template_dominance_margin
            else "UNIDENTIFIED"
        )
        for similarity_value, margin in zip(matched_similarity, dominance_margin, strict=True)
    )
    assigned_shapes = tuple(
        shape if status == "IDENTIFIED" else "Unidentified"
        for shape, status in zip(suggested_shapes, identification_status, strict=True)
    )

    for component_index, template_index in enumerate(assignment):
        if float(physical_identification_basis[component_index] @ templates[template_index]) < 0:
            components[component_index] *= -1.0
            scores[:, component_index] *= -1.0
    for component_index in range(3, dimensions):
        anchor = int(np.argmax(np.abs(components[component_index])))
        if components[component_index, anchor] < 0:
            components[component_index] *= -1.0
            scores[:, component_index] *= -1.0

    explained_ratio = eigenvalues / eigenvalues.sum()
    component_names = tuple(
        f"PC{i + 1} — {assigned_shapes[i] if i < 3 else 'Residual'}" for i in range(dimensions)
    )
    score_frame = pd.DataFrame(scores, index=frame.index, columns=component_names)
    return PCAFit(
        tenors=tuple(frame.columns),
        center_bp=center,
        scale_bp=scale,
        components=components,
        eigenvalues=eigenvalues,
        explained_ratio=explained_ratio,
        scores=score_frame,
        suggested_shapes=suggested_shapes,
        assigned_shapes=assigned_shapes,
        identification_status=identification_status,
        template_similarity=matched_similarity,
        template_dominance_margin=dominance_margin,
        templates=templates,
        standardize=standardize,
        weights=normalized_weights,
        effective_observations=effective_observations,
        minimum_template_similarity=minimum_template_similarity,
        minimum_template_dominance_margin=minimum_template_dominance_margin,
        training_start=pd.Timestamp(frame.index.min()),
        training_end=pd.Timestamp(frame.index.max()),
    )


def reconstruction_diagnostics(
    changes_bp: pd.DataFrame,
    fit: PCAFit,
    retained_factors: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return aggregate and tenor-level in-sample reconstruction diagnostics."""

    if not 1 <= retained_factors <= len(fit.tenors):
        raise ValueError("retained_factors is outside the fitted dimension.")
    scores = fit.transform(changes_bp)
    full = fit.inverse_transform(scores)
    reduced = fit.inverse_transform(scores.iloc[:, :retained_factors])
    residual = changes_bp.astype(float) - reduced
    aggregate = pd.DataFrame(
        {
            "Metric": [
                "Full-rank maximum absolute error",
                f"Top-{retained_factors} mean absolute error",
                f"Top-{retained_factors} root mean squared error",
                f"Top-{retained_factors} 95th percentile absolute error",
                f"Top-{retained_factors} maximum absolute error",
            ],
            "Value (bp)": [
                float((changes_bp - full).abs().to_numpy().max()),
                float(residual.abs().to_numpy().mean()),
                float(np.sqrt(np.mean(residual.to_numpy() ** 2))),
                float(np.quantile(np.abs(residual.to_numpy()), 0.95)),
                float(residual.abs().to_numpy().max()),
            ],
        }
    )
    tenor = pd.DataFrame(
        {
            "MAE (bp)": residual.abs().mean(),
            "RMSE (bp)": np.sqrt((residual**2).mean()),
            "P95 absolute error (bp)": residual.abs().quantile(0.95),
            "Maximum absolute error (bp)": residual.abs().max(),
        }
    )
    return aggregate, tenor


def _align_to_reference(candidate: PCAFit, reference: PCAFit, factors: int) -> tuple[np.ndarray, np.ndarray]:
    if candidate.tenors != reference.tenors:
        raise ValueError("Candidate and reference tenor universes must match.")
    candidate_basis = candidate.shock_basis_bp_per_score[:factors]
    reference_basis = reference.shock_basis_bp_per_score[:factors]
    candidate_norm = candidate_basis / np.linalg.norm(candidate_basis, axis=1, keepdims=True)
    reference_norm = reference_basis / np.linalg.norm(reference_basis, axis=1, keepdims=True)
    similarity = np.abs(candidate_norm @ reference_norm.T)
    assignment = max(
        permutations(range(factors)),
        key=lambda order: sum(similarity[order[j], j] for j in range(factors)),
    )
    aligned = np.stack([candidate_norm[assignment[j]] for j in range(factors)])
    for index in range(factors):
        if aligned[index] @ reference_norm[index] < 0:
            aligned[index] *= -1.0
    cosines = np.array([aligned[i] @ reference_norm[i] for i in range(factors)])
    return aligned, cosines


def compare_pca_regimes(
    reference: PCAFit,
    candidate: PCAFit,
    factors: int = 3,
) -> pd.DataFrame:
    """Compare physical PCA bases, subspaces, and one-sigma shock magnitudes."""

    if reference.tenors != candidate.tenors:
        raise ValueError("Reference and candidate tenor universes must match.")
    if not isinstance(factors, int) or isinstance(factors, bool) or not 1 <= factors <= 3:
        raise ValueError("factors must be an integer between one and three.")
    reference_basis = reference.shock_basis_bp_per_score[:factors]
    candidate_basis = candidate.shock_basis_bp_per_score[:factors]
    reference_norm = reference_basis / np.linalg.norm(reference_basis, axis=1, keepdims=True)
    candidate_norm = candidate_basis / np.linalg.norm(candidate_basis, axis=1, keepdims=True)
    similarity = np.abs(candidate_norm @ reference_norm.T)
    assignment = max(
        permutations(range(factors)),
        key=lambda order: sum(similarity[order[j], j] for j in range(factors)),
    )
    aligned_cosines = np.array([similarity[assignment[index], index] for index in range(factors)])
    reference_subspace = _orthonormal_row_space(reference_basis)
    candidate_subspace = _orthonormal_row_space(candidate_basis)
    singular_values = np.clip(
        np.linalg.svd(candidate_subspace @ reference_subspace.T, compute_uv=False),
        -1.0,
        1.0,
    )
    angles = np.degrees(np.arccos(singular_values))
    reference_sigma = np.linalg.norm(reference.one_sigma_shocks_bp[:factors], axis=1)
    candidate_sigma = np.array(
        [np.linalg.norm(candidate.one_sigma_shocks_bp[candidate_index]) for candidate_index in assignment]
    )
    return pd.DataFrame(
        {
            "Reference factor": reference.component_names[:factors],
            "Candidate matched factor": [
                candidate.component_names[candidate_index] for candidate_index in assignment
            ],
            "Physical loading cosine": aligned_cosines,
            "Ordered subspace principal angle (deg)": angles,
            "Reference one-sigma norm (bp)": reference_sigma,
            "Candidate one-sigma norm (bp)": candidate_sigma,
            "Candidate/reference sigma ratio": candidate_sigma / reference_sigma,
            "Reference observations": len(reference.scores),
            "Candidate observations": len(candidate.scores),
            "Reference start": reference.training_start,
            "Reference end": reference.training_end,
            "Candidate start": candidate.training_start,
            "Candidate end": candidate.training_end,
        },
        index=pd.Index(range(1, factors + 1), name="Factor rank"),
    )


def _orthonormal_row_space(basis: np.ndarray) -> np.ndarray:
    """Return an orthonormal row basis without assuming physical loadings are orthogonal."""

    array = np.asarray(basis, dtype=float)
    if array.ndim != 2 or array.shape[0] > array.shape[1]:
        raise ValueError("basis must be a two-dimensional, full-row-rank matrix.")
    q, r = np.linalg.qr(array.T, mode="reduced")
    tolerance = np.finfo(float).eps * max(array.shape) * np.linalg.norm(r, ord=2)
    if np.any(np.abs(np.diag(r)) <= tolerance):
        raise ValueError("basis is rank deficient and has no stable subspace representation.")
    return q.T


def _require_ordered_time_index(frame: pd.DataFrame, operation: str) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError(f"{operation} requires a DatetimeIndex.")
    if not frame.index.is_monotonic_increasing:
        raise ValueError(f"{operation} requires observations in ascending time order.")


def rolling_stability(
    changes_bp: pd.DataFrame,
    reference: PCAFit,
    *,
    window_observations: int,
    step_observations: int = 13,
    factors: int = 3,
    standardize: bool | None = None,
) -> pd.DataFrame:
    """Monitor like-for-like loading alignment and retained subspace stability."""

    frame = _validate_changes(changes_bp)
    _require_ordered_time_index(frame, "rolling_stability")
    if not 1 <= factors <= min(3, len(frame.columns)):
        raise ValueError("factors must be between one and three and within the fitted dimension.")
    if window_observations < 52 or window_observations > len(frame):
        raise ValueError("window_observations must be between 52 and the sample length.")
    if step_observations < 1:
        raise ValueError("step_observations must be positive.")
    if standardize is None:
        standardize = reference.standardize
    elif not isinstance(standardize, bool):
        raise TypeError("standardize must be boolean or None.")
    elif standardize != reference.standardize:
        raise ValueError(
            "rolling_stability requires the candidate and reference to use the same "
            "covariance/correlation specification. Use a methodology challenge for "
            "cross-specification comparison."
        )
    rows: list[dict[str, object]] = []
    end_positions = list(range(window_observations, len(frame) + 1, step_observations))
    if end_positions[-1] != len(frame):
        end_positions.append(len(frame))
    reference_subspace = _orthonormal_row_space(reference.shock_basis_bp_per_score[:factors])

    for end_position in end_positions:
        sample = frame.iloc[end_position - window_observations : end_position]
        candidate = fit_curve_pca(
            sample,
            standardize=standardize,
            minimum_template_similarity=reference.minimum_template_similarity,
            minimum_template_dominance_margin=reference.minimum_template_dominance_margin,
        )
        candidate_subspace = _orthonormal_row_space(candidate.shock_basis_bp_per_score[:factors])
        singular_values = np.linalg.svd(candidate_subspace @ reference_subspace.T, compute_uv=False)
        singular_values = np.clip(singular_values, -1.0, 1.0)
        principal_angles = np.degrees(np.arccos(singular_values))
        _, factor_cosines = _align_to_reference(candidate, reference, factors)
        projection_distance = np.linalg.norm(
            candidate_subspace.T @ candidate_subspace - reference_subspace.T @ reference_subspace,
            ord="fro",
        ) / np.sqrt(2.0 * factors)
        cutoff_gap = (
            (candidate.eigenvalues[factors - 1] - candidate.eigenvalues[factors])
            / candidate.eigenvalues[factors - 1]
            if factors < len(candidate.eigenvalues) and candidate.eigenvalues[factors - 1] > 0
            else np.nan
        )
        row: dict[str, object] = {
            "Window start": sample.index.min(),
            "Window end": sample.index.max(),
            "Reference start": reference.training_start,
            "Reference end": reference.training_end,
            "Reference mode": "RETROSPECTIVE_FULL_SAMPLE",
            "Observations": len(sample),
            f"Top-{factors} variance": float(candidate.explained_ratio[:factors].sum()),
            "Maximum principal angle (deg)": float(principal_angles.max()),
            "Projection distance": float(projection_distance),
            "Cutoff eigenvalue gap": float(cutoff_gap),
            "Minimum template similarity": float(candidate.template_similarity.min()),
        }
        for factor_index in range(factors):
            row[f"PC{factor_index + 1} loading cosine"] = float(factor_cosines[factor_index])
            row[f"PC{factor_index + 1} reference identification"] = reference.identification_status[
                factor_index
            ]
            row[f"PC{factor_index + 1} suggested template"] = reference.suggested_shapes[factor_index]
        rows.append(row)
    return pd.DataFrame(rows).set_index("Window end")


def sequential_window_stability(
    changes_bp: pd.DataFrame,
    *,
    window_observations: int,
    step_observations: int = 13,
    factors: int = 3,
    standardize: bool = False,
    minimum_template_similarity: float = 0.70,
    minimum_template_dominance_margin: float = 0.10,
) -> pd.DataFrame:
    """Compare adjacent non-overlapping windows using no observations after each alert date."""

    frame = _validate_changes(changes_bp)
    _require_ordered_time_index(frame, "sequential_window_stability")
    if not 1 <= factors <= min(3, len(frame.columns)):
        raise ValueError("factors must be between one and three and within the fitted dimension.")
    if window_observations < 52 or 2 * window_observations > len(frame):
        raise ValueError("Two complete stability windows are required.")
    if step_observations < 1:
        raise ValueError("step_observations must be positive.")
    end_positions = list(range(2 * window_observations, len(frame) + 1, step_observations))
    if end_positions[-1] != len(frame):
        end_positions.append(len(frame))
    rows: list[dict[str, object]] = []
    for end_position in end_positions:
        reference_sample = frame.iloc[
            end_position - 2 * window_observations : end_position - window_observations
        ]
        candidate_sample = frame.iloc[end_position - window_observations : end_position]
        reference = fit_curve_pca(
            reference_sample,
            standardize=standardize,
            minimum_template_similarity=minimum_template_similarity,
            minimum_template_dominance_margin=minimum_template_dominance_margin,
        )
        candidate = fit_curve_pca(
            candidate_sample,
            standardize=standardize,
            minimum_template_similarity=minimum_template_similarity,
            minimum_template_dominance_margin=minimum_template_dominance_margin,
        )
        comparison = compare_pca_regimes(reference, candidate, factors)
        row: dict[str, object] = {
            "Reference start": reference_sample.index.min(),
            "Reference end": reference_sample.index.max(),
            "Monitoring start": candidate_sample.index.min(),
            "Monitoring end": candidate_sample.index.max(),
            "Observations per window": window_observations,
            "Reference mode": "SEQUENTIAL_PRIOR_NON_OVERLAPPING_WINDOW",
            f"Monitoring top-{factors} variance": float(candidate.explained_ratio[:factors].sum()),
            "Maximum principal angle (deg)": float(
                comparison["Ordered subspace principal angle (deg)"].max()
            ),
            "Minimum aligned loading cosine": float(comparison["Physical loading cosine"].min()),
        }
        for rank, value in comparison["Physical loading cosine"].items():
            row[f"Factor {rank} cosine"] = float(value)
        rows.append(row)
    result = pd.DataFrame(rows).set_index("Monitoring end")
    if not (result["Reference end"] < result["Monitoring start"]).all():
        raise RuntimeError("Sequential stability control detected overlapping reference data.")
    return result


def moving_block_bootstrap_stability(
    changes_bp: pd.DataFrame,
    reference: PCAFit,
    *,
    replications: int = 250,
    block_length: int = 13,
    factors: int = 3,
    random_seed: int = 20260825,
) -> pd.DataFrame:
    """Estimate loading and subspace uncertainty with a circular block bootstrap."""

    frame = _validate_changes(changes_bp)
    _require_ordered_time_index(frame, "moving_block_bootstrap_stability")
    if reference.weights is not None:
        raise ValueError(
            "moving_block_bootstrap_stability currently supports equal-weight references only; "
            "an EWMA reference requires a separately specified weighted resampling design."
        )
    if not 1 <= factors <= min(3, len(frame.columns)):
        raise ValueError("factors must be between one and three and within the fitted dimension.")
    if replications < 1:
        raise ValueError("replications must be positive.")
    if not 2 <= block_length <= len(frame):
        raise ValueError("block_length must be between two and the sample length.")
    rng = np.random.default_rng(random_seed)
    observations = len(frame)
    blocks_needed = int(np.ceil(observations / block_length))
    rows: list[dict[str, float]] = []
    reference_subspace = _orthonormal_row_space(reference.shock_basis_bp_per_score[:factors])

    for _ in range(replications):
        starts = rng.integers(0, observations, size=blocks_needed)
        indices = np.concatenate([(start + np.arange(block_length)) % observations for start in starts])[
            :observations
        ]
        sample = frame.iloc[indices].copy()
        sample.index = pd.RangeIndex(observations)
        candidate = fit_curve_pca(
            sample,
            standardize=reference.standardize,
            minimum_template_similarity=reference.minimum_template_similarity,
            minimum_template_dominance_margin=reference.minimum_template_dominance_margin,
        )
        candidate_subspace = _orthonormal_row_space(candidate.shock_basis_bp_per_score[:factors])
        singular_values = np.clip(
            np.linalg.svd(candidate_subspace @ reference_subspace.T, compute_uv=False), -1.0, 1.0
        )
        _, cosines = _align_to_reference(candidate, reference, factors)
        row = {
            f"Top-{factors} variance": float(candidate.explained_ratio[:factors].sum()),
            "Maximum principal angle (deg)": float(np.degrees(np.arccos(singular_values)).max()),
        }
        for factor_index in range(factors):
            row[f"PC{factor_index + 1} loading cosine"] = float(cosines[factor_index])
        rows.append(row)

    results = pd.DataFrame(rows)
    summary = results.quantile([0.025, 0.50, 0.975]).T
    summary.columns = ["2.5%", "Median", "97.5%"]
    summary["Replications"] = replications
    summary["Block length (weeks)"] = block_length
    summary["Random seed"] = random_seed
    summary["Expected order statistics per 2.5% tail"] = replications * 0.025
    summary["Reference identification status"] = ""
    summary["Reference suggested template"] = ""
    for factor_index in range(factors):
        row_name = f"PC{factor_index + 1} loading cosine"
        summary.loc[row_name, "Reference identification status"] = reference.identification_status[
            factor_index
        ]
        summary.loc[row_name, "Reference suggested template"] = reference.suggested_shapes[factor_index]
    return summary


def bootstrap_block_length_sensitivity(
    changes_bp: pd.DataFrame,
    reference: PCAFit,
    *,
    block_lengths: tuple[int, ...],
    replications: int,
    factors: int = 3,
    random_seed: int = 20260825,
) -> pd.DataFrame:
    """Challenge conditional bootstrap uncertainty across dependence block lengths."""

    rows: list[dict[str, float]] = []
    for offset, block_length in enumerate(block_lengths):
        summary = moving_block_bootstrap_stability(
            changes_bp,
            reference,
            replications=replications,
            block_length=block_length,
            factors=factors,
            random_seed=random_seed + 10_007 * offset,
        )
        rows.append(
            {
                "Block length (weeks)": block_length,
                "Replications": replications,
                "Random seed": random_seed + 10_007 * offset,
                "Expected order statistics per 2.5% tail": replications * 0.025,
                "Angle median (deg)": summary.loc["Maximum principal angle (deg)", "Median"],
                "Angle 97.5% (deg)": summary.loc["Maximum principal angle (deg)", "97.5%"],
                f"Top-{factors} variance median": summary.loc[f"Top-{factors} variance", "Median"],
                f"Top-{factors} variance 2.5%": summary.loc[f"Top-{factors} variance", "2.5%"],
                "Minimum median loading cosine": min(
                    summary.loc[f"PC{factor_index + 1} loading cosine", "Median"]
                    for factor_index in range(factors)
                ),
            }
        )
    return pd.DataFrame(rows).set_index("Block length (weeks)")


def expanding_oos_reconstruction(
    changes_bp: pd.DataFrame,
    *,
    minimum_training_observations: int,
    retained_factors: int = 3,
    refit_every: int = 13,
    standardize: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Project each holdout observation on a basis estimated strictly before it."""

    frame = _validate_changes(changes_bp)
    _require_ordered_time_index(frame, "expanding_oos_reconstruction")
    if not 1 <= retained_factors <= min(3, len(frame.columns)):
        raise ValueError("retained_factors must be between one and three and within the fitted dimension.")
    if minimum_training_observations < 52 or minimum_training_observations >= len(frame):
        raise ValueError("minimum_training_observations must leave a non-empty holdout sample.")
    if refit_every < 1:
        raise ValueError("refit_every must be positive.")
    predictions: list[pd.Series] = []
    audit_rows: list[dict[str, object]] = []
    current_fit: PCAFit | None = None
    refit_sequence = 0
    basis_refit_position = -1
    for position in range(minimum_training_observations, len(frame)):
        if current_fit is None or (position - minimum_training_observations) % refit_every == 0:
            current_fit = fit_curve_pca(frame.iloc[:position], standardize=standardize)
            refit_sequence += 1
            basis_refit_position = position
        observed = frame.iloc[[position]]
        factor_scores = current_fit.transform(observed).iloc[:, :retained_factors]
        reconstructed = current_fit.inverse_transform(factor_scores).iloc[0]
        reconstructed.name = frame.index[position]
        predictions.append(reconstructed)
        audit_rows.append(
            {
                "Target date": frame.index[position],
                "Target position": position,
                "Basis refit sequence": refit_sequence,
                "Basis refit trigger position": basis_refit_position,
                "Maximum source position": basis_refit_position - 1,
                "Training start": current_fit.training_start,
                "Training end": current_fit.training_end,
                "Training observations": len(current_fit.scores),
            }
        )
    reconstruction = pd.DataFrame(predictions)
    actual = frame.loc[reconstruction.index]
    residual = actual - reconstruction
    metrics = pd.DataFrame(
        {
            "MAE (bp)": residual.abs().mean(),
            "RMSE (bp)": np.sqrt((residual**2).mean()),
            "P95 absolute error (bp)": residual.abs().quantile(0.95),
            "Maximum absolute error (bp)": residual.abs().max(),
            "Holdout observations": len(residual),
        }
    )
    audit = pd.DataFrame(audit_rows).set_index("Target date")
    if not (audit["Training end"] < audit.index).all():
        raise RuntimeError("OOS reconstruction audit detected look-ahead contamination.")
    return reconstruction, metrics, audit
