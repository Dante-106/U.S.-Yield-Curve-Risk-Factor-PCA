import numpy as np
import pandas as pd
import pytest

from yield_curve_pca.config import CORE_TENORS, MATURITY_YEARS
from yield_curve_pca.pca import (
    ECONOMIC_SHAPES,
    compare_pca_regimes,
    economic_shape_templates,
    ewma_weights,
    expanding_oos_reconstruction,
    fit_curve_pca,
    moving_block_bootstrap_stability,
    reconstruction_diagnostics,
    rolling_stability,
    sequential_window_stability,
)


def test_real_sample_golden_results(curve_bundle, structural_pca):
    assert structural_pca.explained_ratio[:3].sum() == pytest.approx(0.9686557544, abs=1e-10)
    assert structural_pca.assigned_shapes == ("Level", "Slope", "Curvature")
    assert structural_pca.template_similarity == pytest.approx([0.9477033, 0.9487832, 0.9320335], abs=1e-6)
    assert structural_pca.transform(curve_bundle.weekly_changes_bp).to_numpy() == pytest.approx(
        structural_pca.scores.to_numpy(), abs=1e-12
    )


def test_oriented_transform_inverse_transform_round_trip(curve_bundle, structural_pca):
    scores = structural_pca.transform(curve_bundle.weekly_changes_bp)
    reconstructed = structural_pca.inverse_transform(scores)
    assert np.max(np.abs(reconstructed - curve_bundle.weekly_changes_bp).to_numpy()) < 1e-10

    reversed_scores = scores.iloc[:, [2, 1, 0]]
    with pytest.raises(ValueError, match="Factor-score columns"):
        structural_pca.inverse_transform(reversed_scores)


def test_correlation_pca_physical_round_trip(curve_bundle):
    fit = fit_curve_pca(curve_bundle.weekly_changes_bp, standardize=True)
    scores = fit.transform(curve_bundle.weekly_changes_bp)
    reconstructed = fit.inverse_transform(scores)
    assert fit.score_unit == "standardized-score units"
    assert np.max(np.abs(reconstructed - curve_bundle.weekly_changes_bp).to_numpy()) < 1e-10
    physical = fit.shock_basis_bp_per_score[:3]
    physical /= np.linalg.norm(physical, axis=1, keepdims=True)
    expected_similarity = np.array(
        [
            abs(physical[index] @ fit.templates[ECONOMIC_SHAPES.index(shape)])
            for index, shape in enumerate(fit.suggested_shapes)
        ]
    )
    assert fit.template_similarity == pytest.approx(expected_similarity, abs=1e-12)


def test_synthetic_factor_subspace_recovery():
    rng = np.random.default_rng(17)
    tenors = list(CORE_TENORS)
    templates = economic_shape_templates([MATURITY_YEARS[t] for t in tenors])
    factors = rng.normal(size=(1500, 3)) * np.array([20.0, 8.0, 3.0])
    noise = rng.normal(scale=0.15, size=(1500, len(tenors)))
    data = pd.DataFrame(factors @ templates + noise, columns=tenors)
    fit = fit_curve_pca(data)
    fitted_projector = fit.components[:3].T @ fit.components[:3]
    true_projector = templates.T @ templates
    assert np.linalg.norm(fitted_projector - true_projector, ord="fro") < 0.02


@pytest.mark.parametrize("bad_kind", ["nan", "inf", "constant", "duplicate"])
def test_invalid_pca_inputs_fail_fast(curve_bundle, bad_kind):
    bad = curve_bundle.weekly_changes_bp.iloc[:100].copy()
    if bad_kind == "nan":
        bad.iloc[0, 0] = np.nan
    elif bad_kind == "inf":
        bad.iloc[0, 0] = np.inf
    elif bad_kind == "constant":
        bad.iloc[:, 0] = 1.0
    else:
        bad.columns = [*bad.columns[:-1], bad.columns[-2]]
    with pytest.raises(ValueError):
        fit_curve_pca(bad)


def test_three_factor_reconstruction_matches_golden(curve_bundle, structural_pca):
    aggregate, _ = reconstruction_diagnostics(curve_bundle.weekly_changes_bp, structural_pca, 3)
    mae = aggregate.loc[aggregate["Metric"] == "Top-3 mean absolute error", "Value (bp)"].iloc[0]
    assert mae == pytest.approx(1.3419, abs=1e-3)


def test_correlation_pca_rolling_subspace_is_finite_and_bounded(curve_bundle):
    changes = curve_bundle.weekly_changes_bp.iloc[:400]
    reference = fit_curve_pca(changes, standardize=True)
    rolling = rolling_stability(
        changes,
        reference,
        window_observations=260,
        step_observations=52,
    )
    assert np.isfinite(rolling["Maximum principal angle (deg)"]).all()
    assert rolling["Maximum principal angle (deg)"].between(0.0, 90.0).all()


def test_rolling_stability_rejects_cross_specification_comparison(curve_bundle):
    changes = curve_bundle.weekly_changes_bp.iloc[:400]
    reference = fit_curve_pca(changes, standardize=True)
    with pytest.raises(ValueError, match="same covariance/correlation specification"):
        rolling_stability(
            changes,
            reference,
            window_observations=260,
            standardize=False,
        )


def test_economic_orientation_is_invariant_to_tenor_column_order(curve_bundle):
    original = fit_curve_pca(curve_bundle.weekly_changes_bp)
    reversed_changes = curve_bundle.weekly_changes_bp.loc[:, list(reversed(original.tenors))]
    reversed_fit = fit_curve_pca(reversed_changes)
    reordered = pd.DataFrame(
        reversed_fit.components[:3],
        columns=reversed_fit.tenors,
    ).loc[:, original.tenors]
    cosines = np.sum(original.components[:3] * reordered.to_numpy(), axis=1)
    assert cosines == pytest.approx([1.0, 1.0, 1.0], abs=1e-12)


def test_position_weighted_pca_rejects_shuffled_dates(curve_bundle):
    changes = curve_bundle.weekly_changes_bp.iloc[:200]
    shuffled = changes.sample(frac=1.0, random_state=5)
    with pytest.raises(ValueError, match="unique, ascending DatetimeIndex"):
        fit_curve_pca(shuffled, weights=ewma_weights(len(shuffled), 52.0))


def test_equal_weight_bootstrap_rejects_ewma_reference(curve_bundle):
    changes = curve_bundle.weekly_changes_bp.iloc[:300]
    reference = fit_curve_pca(
        changes,
        weights=ewma_weights(len(changes), 52.0),
    )
    with pytest.raises(ValueError, match="equal-weight references only"):
        moving_block_bootstrap_stability(
            changes,
            reference,
            replications=5,
            block_length=13,
        )


def test_weak_economic_matches_are_not_force_labelled():
    rng = np.random.default_rng(91)
    templates = economic_shape_templates([MATURITY_YEARS[tenor] for tenor in CORE_TENORS])
    _, _, full_basis = np.linalg.svd(templates, full_matrices=True)
    complement = full_basis[3:6]
    factors = rng.normal(size=(800, 3)) * np.array([20.0, 10.0, 5.0])
    changes = pd.DataFrame(factors @ complement, columns=CORE_TENORS)
    fit = fit_curve_pca(changes)
    assert fit.assigned_shapes == ("Unidentified", "Unidentified", "Unidentified")
    assert fit.identification_status == ("UNIDENTIFIED",) * 3
    assert (fit.template_similarity < 1.0e-10).all()


def test_stability_outputs_do_not_propagate_unidentified_economic_labels(curve_bundle):
    changes = curve_bundle.weekly_changes_bp.iloc[:300]
    reference = fit_curve_pca(changes, minimum_template_dominance_margin=0.99)
    assert reference.identification_status == ("UNIDENTIFIED",) * 3
    rolling = rolling_stability(
        changes,
        reference,
        window_observations=260,
        step_observations=40,
    )
    assert {f"PC{i} loading cosine" for i in range(1, 4)} <= set(rolling.columns)
    assert not {"Level cosine", "Slope cosine", "Curvature cosine"} & set(rolling.columns)
    assert set(rolling["PC1 reference identification"]) == {"UNIDENTIFIED"}
    bootstrap = moving_block_bootstrap_stability(
        changes,
        reference,
        replications=3,
        block_length=13,
    )
    assert "PC1 loading cosine" in bootstrap.index
    assert bootstrap.loc["PC1 loading cosine", "Reference identification status"] == "UNIDENTIFIED"


def test_near_tied_economic_templates_remain_unidentified():
    templates = economic_shape_templates([MATURITY_YEARS[tenor] for tenor in CORE_TENORS])
    rotated_basis = np.vstack(
        (
            (templates[0] + templates[1]) / np.sqrt(2.0),
            (templates[0] - templates[1]) / np.sqrt(2.0),
            templates[2],
        )
    )
    rng = np.random.default_rng(404)
    raw_scores = rng.normal(size=(800, 3))
    raw_scores -= raw_scores.mean(axis=0)
    orthogonal_scores, _ = np.linalg.qr(raw_scores)
    scores = orthogonal_scores * np.sqrt(799.0 * np.array([400.0, 100.0, 25.0]))
    changes = pd.DataFrame(scores @ rotated_basis, columns=CORE_TENORS)

    fit = fit_curve_pca(changes)

    assert fit.template_similarity[:2] == pytest.approx([2**-0.5, 2**-0.5], abs=1e-10)
    assert np.abs(fit.template_dominance_margin[:2]).max() < 1.0e-10
    assert fit.assigned_shapes[:2] == ("Unidentified", "Unidentified")
    assert fit.identification_status[:2] == ("UNIDENTIFIED", "UNIDENTIFIED")


def test_three_tenor_curvature_orientation_has_no_empty_bucket_dependency():
    maturities = [MATURITY_YEARS[tenor] for tenor in ("1Y", "2Y", "3Y")]
    templates = economic_shape_templates(maturities)
    centered = np.log(maturities) - np.mean(np.log(maturities))
    assert np.isfinite(templates).all()
    assert templates[2] @ (-(centered**2)) > 0


def test_regime_comparison_is_identity_for_same_fit(structural_pca):
    comparison = compare_pca_regimes(structural_pca, structural_pca)
    assert comparison["Physical loading cosine"].to_numpy() == pytest.approx([1.0] * 3)
    assert comparison["Ordered subspace principal angle (deg)"].max() < 1.0e-5
    assert comparison["Candidate/reference sigma ratio"].to_numpy() == pytest.approx([1.0] * 3)


def test_sequential_stability_has_no_future_reference_contamination(curve_bundle):
    changes = curve_bundle.weekly_changes_bp.iloc[:650].copy()
    baseline = sequential_window_stability(
        changes,
        window_observations=260,
        step_observations=13,
    )
    mutated = changes.copy()
    mutated.iloc[-20:] += 500.0
    challenged = sequential_window_stability(
        mutated,
        window_observations=260,
        step_observations=13,
    )
    assert baseline.iloc[0].equals(challenged.iloc[0])
    assert (baseline["Reference end"] < baseline["Monitoring start"]).all()


def test_oos_reconstruction_exports_basis_vintage_audit(curve_bundle):
    changes = curve_bundle.weekly_changes_bp.iloc[:650]
    reconstruction, metrics, audit = expanding_oos_reconstruction(
        changes,
        minimum_training_observations=520,
        retained_factors=3,
        refit_every=13,
    )
    assert len(reconstruction) == len(audit) == 130
    assert (audit["Training end"] < audit.index).all()
    assert (audit["Maximum source position"] < audit["Target position"]).all()
    assert (metrics["Holdout observations"] == 130).all()
