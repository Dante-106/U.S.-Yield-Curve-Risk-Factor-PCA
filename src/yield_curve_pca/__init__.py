"""U.S. Treasury yield-curve PCA, forecast validation, and linear risk tools."""

from .config import (
    ALL_TENORS,
    CORE_TENORS,
    FRED_SERIES,
    MATURITY_YEARS,
    DataConfig,
    ForecastConfig,
    PCAConfig,
    PipelineConfig,
    RiskConfig,
)
from .data import CurveDataBundle, DataProvenance, load_curve_data
from .forecast import ForecastResult, walk_forward_forecast
from .pca import PCAFit, fit_curve_pca
from .pipeline import PipelineResult, run_pipeline
from .risk import HedgeResult, RiskMappingResult, map_linear_curve_risk, optimize_key_rate_hedge
from .validation import ValidationSummary

__all__ = [
    "ALL_TENORS",
    "CORE_TENORS",
    "FRED_SERIES",
    "MATURITY_YEARS",
    "DataConfig",
    "ForecastConfig",
    "PCAConfig",
    "PipelineConfig",
    "RiskConfig",
    "CurveDataBundle",
    "DataProvenance",
    "ForecastResult",
    "PCAFit",
    "PipelineResult",
    "RiskMappingResult",
    "HedgeResult",
    "ValidationSummary",
    "fit_curve_pca",
    "load_curve_data",
    "map_linear_curve_risk",
    "optimize_key_rate_hedge",
    "run_pipeline",
    "walk_forward_forecast",
]

__version__ = "3.3.0"
