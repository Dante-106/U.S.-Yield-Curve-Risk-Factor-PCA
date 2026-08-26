from yield_curve_pca.validation import factor_distribution_diagnostics, validate_pca_algebra


def test_independent_pca_validation_passes(curve_bundle, structural_pca):
    checks = validate_pca_algebra(curve_bundle.weekly_changes_bp, structural_pca)
    assert "FAIL" not in set(checks["Status"])
    assert checks.loc["Oriented score API consistency", "Result"] == 0.0


def test_tail_claim_is_backed_by_formal_diagnostics(structural_pca):
    diagnostics = factor_distribution_diagnostics(structural_pca)
    assert (diagnostics["Excess kurtosis"] > 2.0).all()
    assert (diagnostics["Jarque–Bera p-value"] < 1e-10).all()
