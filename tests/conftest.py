from pathlib import Path

import pytest

from yield_curve_pca.config import DataConfig
from yield_curve_pca.data import load_curve_data
from yield_curve_pca.pca import fit_curve_pca


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def curve_bundle(project_root):
    return load_curve_data(DataConfig(), project_root)


@pytest.fixture(scope="session")
def structural_pca(curve_bundle):
    return fit_curve_pca(curve_bundle.weekly_changes_bp)
