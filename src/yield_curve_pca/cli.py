"""Small, dependency-light CLI for repeatable batch runs."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import platform
import re
import shutil
import sys
import tempfile
from dataclasses import asdict, fields, replace
from datetime import date, datetime, timezone
from hashlib import sha256
from importlib import metadata
from io import BytesIO
from math import isclose, isfinite
from pathlib import Path
from uuid import uuid4
from zoneinfo import ZoneInfo

import pandas as pd

from . import __version__
from .config import DataConfig, ForecastConfig, PCAConfig, PipelineConfig, RiskConfig
from .pipeline import run_pipeline
from .risk import map_linear_curve_risk

LOGGER = logging.getLogger("yield_curve_pca")

_INVOCATION_ONLY_PROVENANCE_FIELDS = (
    "loaded_at_utc",
    "network_error",
    "cache_persistence_error",
)

try:
    import fcntl
except ImportError:
    fcntl = None


class _ControlledArgumentParser(argparse.ArgumentParser):
    """Turn malformed CLI input into a ledgerable exception instead of SystemExit."""

    def error(self, message: str) -> None:
        raise ValueError(f"argument parsing failed: {message}")


def _parser() -> argparse.ArgumentParser:
    parser = _ControlledArgumentParser(description="Run U.S. Treasury yield-curve PCA controls.")
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--config-json", type=Path, help="Strict JSON PipelineConfig input.")
    parser.add_argument(
        "--dump-effective-config",
        type=Path,
        help="Write the validated effective configuration as JSON.",
    )
    parser.add_argument("--start-date", default=None)
    parser.add_argument(
        "--end-date",
        default=None,
        help=(
            "Inclusive ISO end date. Defaults to 2025-12-31 for the frozen snapshot and "
            "the current New York date for live modes."
        ),
    )
    parser.add_argument(
        "--source-mode",
        choices=("snapshot", "live", "live_then_snapshot"),
        default=None,
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument(
        "--maximum-staleness-days",
        type=int,
        default=None,
        help=(
            "Fail when the final weekly curve is older than this calendar-day SLA. "
            "Current live runs default to seven days; explicit historical end dates do not."
        ),
    )
    parser.add_argument(
        "--krd-csv",
        type=Path,
        help="Optional CSV with columns tenor,dv01_usd_per_bp for linear portfolio mapping.",
    )
    parser.add_argument("--krd-as-of", help="Required ISO valuation date when --krd-csv is used.")
    parser.add_argument(
        "--krd-curve-id",
        help="Required sensitivity-curve identifier when --krd-csv is used.",
    )
    parser.add_argument("--krd-currency", choices=("USD",), default="USD")
    parser.add_argument("--krd-bump-bp", type=float, default=1.0)
    parser.add_argument(
        "--portfolio-id",
        help=(
            "Required caller-supplied portfolio identifier with KRD; syntax is checked, "
            "but upstream lineage is not verified."
        ),
    )
    parser.add_argument(
        "--position-snapshot-id",
        help=(
            "Required caller-supplied position-snapshot identifier/hash with KRD; "
            "syntax is checked, but upstream lineage is not verified."
        ),
    )
    parser.add_argument(
        "--sensitivity-engine-id",
        help=(
            "Required caller-supplied sensitivity-engine/version identifier with KRD; "
            "syntax is checked, but upstream lineage is not verified."
        ),
    )
    parser.add_argument("--skip-forecast", action="store_true")
    parser.add_argument(
        "--accept-warnings",
        action="store_true",
        help="Return success when validation is SHARE WITH CAVEATS; warnings remain in outputs.",
    )
    parser.add_argument(
        "--warning-approval-id",
        help="Required review/ticket identifier when --accept-warnings is used.",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING", "ERROR"), default="INFO")
    return parser


def _read_krd(path: Path) -> tuple[pd.Series, bytes, str]:
    if not path.is_file():
        raise FileNotFoundError(f"KRD file does not exist: {path}.")
    with path.open("rb") as handle:
        raw_bytes = handle.read(1_000_001)
    if len(raw_bytes) > 1_000_000:
        raise ValueError("KRD file exceeds the 1 MB control limit.")
    frame = pd.read_csv(BytesIO(raw_bytes))
    required = {"tenor", "dv01_usd_per_bp"}
    if set(frame.columns) != required:
        raise ValueError(f"KRD file must contain exactly {sorted(required)}.")
    if frame["tenor"].duplicated().any():
        raise ValueError("KRD file contains duplicate tenors.")
    return (
        frame.set_index("tenor")["dv01_usd_per_bp"],
        raw_bytes,
        sha256(raw_bytes).hexdigest(),
    )


def _krd_metadata_from_args(args: argparse.Namespace) -> dict[str, object] | None:
    if args.krd_csv is None:
        if any(
            value is not None
            for value in (
                args.krd_as_of,
                args.krd_curve_id,
                args.portfolio_id,
                args.position_snapshot_id,
                args.sensitivity_engine_id,
            )
        ):
            raise ValueError("KRD metadata cannot be supplied without --krd-csv.")
        return None
    if not args.krd_as_of or not args.krd_curve_id or not args.krd_curve_id.strip():
        raise ValueError("--krd-as-of and --krd-curve-id are required with --krd-csv.")
    for field_name in ("portfolio_id", "position_snapshot_id", "sensitivity_engine_id"):
        value = getattr(args, field_name)
        if value is None or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", value.strip()):
            raise ValueError(
                f"--{field_name.replace('_', '-')} must be a syntactically valid 1-128 character identifier."
            )
    try:
        valuation_date = date.fromisoformat(args.krd_as_of)
    except ValueError as exc:
        raise ValueError("--krd-as-of must be an ISO date (YYYY-MM-DD).") from exc
    if valuation_date > datetime.now(ZoneInfo("America/New_York")).date():
        raise ValueError("--krd-as-of cannot be a future New York date.")
    if isinstance(args.krd_bump_bp, bool) or not isfinite(args.krd_bump_bp) or args.krd_bump_bp <= 0:
        raise ValueError("--krd-bump-bp must be positive and finite.")
    if not isclose(float(args.krd_bump_bp), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("--krd-bump-bp must equal the controlled 1 bp sensitivity bump.")
    return {
        "valuation_date": valuation_date.isoformat(),
        "currency": args.krd_currency,
        "sensitivity_curve_id": args.krd_curve_id.strip(),
        "parallel_bump_bp": float(args.krd_bump_bp),
        "dv01_sign_convention": "positive price gain for a 1 bp yield decline",
        "portfolio_id": args.portfolio_id.strip(),
        "position_snapshot_id": args.position_snapshot_id.strip(),
        "sensitivity_engine_id": args.sensitivity_engine_id.strip(),
    }


def _validate_krd_metadata(metadata: dict[str, object]) -> dict[str, object]:
    required = {
        "valuation_date",
        "currency",
        "sensitivity_curve_id",
        "parallel_bump_bp",
        "dv01_sign_convention",
        "portfolio_id",
        "position_snapshot_id",
        "sensitivity_engine_id",
    }
    if not isinstance(metadata, dict) or set(metadata) != required:
        raise ValueError(f"KRD metadata must contain exactly {sorted(required)}.")
    try:
        valuation_date = date.fromisoformat(str(metadata["valuation_date"]))
    except ValueError as exc:
        raise ValueError("KRD valuation_date must be an ISO date.") from exc
    if valuation_date > datetime.now(ZoneInfo("America/New_York")).date():
        raise ValueError("KRD valuation_date cannot be a future New York date.")
    if metadata["currency"] != "USD":
        raise ValueError("KRD currency must be USD.")
    bump = metadata["parallel_bump_bp"]
    if isinstance(bump, bool) or not isinstance(bump, (int, float)) or not isfinite(bump):
        raise ValueError("KRD parallel_bump_bp must be finite numeric 1.0.")
    if not isclose(float(bump), 1.0, rel_tol=0.0, abs_tol=1.0e-12):
        raise ValueError("KRD parallel_bump_bp must equal 1.0.")
    if metadata["dv01_sign_convention"] != "positive price gain for a 1 bp yield decline":
        raise ValueError("KRD DV01 sign convention is unsupported.")
    if not isinstance(metadata["sensitivity_curve_id"], str):
        raise ValueError("KRD sensitivity_curve_id must be a string.")
    curve_id = metadata["sensitivity_curve_id"].strip()
    if not curve_id or len(curve_id) > 128:
        raise ValueError("KRD sensitivity_curve_id must contain 1-128 characters.")
    normalized = dict(metadata)
    normalized["valuation_date"] = valuation_date.isoformat()
    normalized["sensitivity_curve_id"] = curve_id
    normalized["parallel_bump_bp"] = 1.0
    for field_name in ("portfolio_id", "position_snapshot_id", "sensitivity_engine_id"):
        if not isinstance(metadata[field_name], str):
            raise ValueError(f"KRD {field_name} must be a string identifier.")
        value = metadata[field_name].strip()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", value):
            raise ValueError(f"KRD {field_name} is not a controlled identifier.")
        normalized[field_name] = value
    return normalized


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _csv_scalar_kind(value: str) -> str:
    if value == "":
        return "missing"
    if value in {"True", "False"}:
        return "boolean"
    if re.fullmatch(r"[+-]?\d+", value):
        return "integer"
    try:
        float(value)
    except ValueError:
        if re.fullmatch(r"\d{4}-\d{2}-\d{2}(?:[ T].*)?", value):
            return "datetime"
        return "string"
    return "number"


def _artifact_contract(path: Path) -> dict[str, object]:
    """Return a lightweight structural contract in addition to the content hash."""

    if path.suffix == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            try:
                columns = next(reader)
            except StopIteration as exc:
                raise RuntimeError(f"CSV artifact is empty: {path}.") from exc
            if len(columns) != len(set(columns)):
                raise RuntimeError(f"CSV artifact has duplicate header fields: {path}.")
            observed_types = [set() for _ in columns]
            row_count = 0
            for row in reader:
                if len(row) != len(columns):
                    raise RuntimeError(f"CSV artifact has a ragged row: {path}.")
                row_count += 1
                for position, value in enumerate(row):
                    observed_types[position].add(_csv_scalar_kind(value))
        return {
            "media_type": "text/csv",
            "bytes": path.stat().st_size,
            "rows": row_count,
            "columns": columns,
            "observed_scalar_types": [sorted(values) for values in observed_types],
        }
    if path.suffix == ".json":
        value = json.loads(path.read_text(encoding="utf-8"))
        return {
            "media_type": "application/json",
            "bytes": path.stat().st_size,
            "top_level_type": type(value).__name__,
            "top_level_keys": sorted(value) if isinstance(value, dict) else None,
        }
    raise RuntimeError(f"No artifact-contract rule exists for {path.name!r}.")


def _implementation_sha256() -> str:
    package_root = Path(__file__).resolve().parent
    digest = sha256()
    for source in sorted(package_root.rglob("*.py")):
        relative = source.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(source.read_bytes())
    return digest.hexdigest()


def _environment_identity() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": metadata.version("numpy"),
        "pandas": metadata.version("pandas"),
        "scipy": metadata.version("scipy"),
        "matplotlib": metadata.version("matplotlib"),
    }


def _data_config_from_args(
    args: argparse.Namespace,
    *,
    new_york_today: str | None = None,
) -> DataConfig:
    """Resolve deterministic-research versus current-risk CLI defaults explicitly."""

    source_mode = args.source_mode or "snapshot"
    start_date = args.start_date or "2000-01-01"
    end_was_defaulted = args.end_date is None
    if args.end_date is not None:
        end_date = args.end_date
    elif source_mode == "snapshot":
        end_date = "2025-12-31"
    else:
        end_date = new_york_today or datetime.now(ZoneInfo("America/New_York")).date().isoformat()
    maximum_staleness_days = args.maximum_staleness_days
    if end_was_defaulted and source_mode in {"live", "live_then_snapshot"}:
        maximum_staleness_days = 7 if maximum_staleness_days is None else maximum_staleness_days
    return DataConfig(
        start_date=start_date,
        end_date=end_date,
        source_mode=source_mode,
        maximum_staleness_days=maximum_staleness_days,
    )


def _load_pipeline_config(path: Path) -> PipelineConfig:
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file does not exist: {path}.")
    with path.open("rb") as handle:
        payload = handle.read(1_000_001)
    if len(payload) > 1_000_000:
        raise ValueError("Configuration file exceeds the 1 MB control limit.")
    try:

        def reject_duplicate_keys(pairs):
            document: dict[str, object] = {}
            for key, value in pairs:
                if key in document:
                    raise ValueError(f"Duplicate JSON configuration key: {key!r}.")
                document[key] = value
            return document

        document = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Configuration file must be valid UTF-8 JSON.") from exc
    if not isinstance(document, dict):
        raise ValueError("Configuration root must be a JSON object.")
    sections = {
        "data": DataConfig,
        "pca": PCAConfig,
        "forecast": ForecastConfig,
        "risk": RiskConfig,
    }
    unknown_sections = set(document) - set(sections)
    if unknown_sections:
        raise ValueError(f"Unknown configuration sections: {sorted(unknown_sections)}.")

    instances: dict[str, object] = {}
    for section, config_class in sections.items():
        values = document.get(section, {})
        if not isinstance(values, dict):
            raise ValueError(f"Configuration section {section!r} must be a JSON object.")
        allowed = {field.name for field in fields(config_class)}
        unknown_fields = set(values) - allowed
        if unknown_fields:
            raise ValueError(
                f"Unknown fields in configuration section {section!r}: {sorted(unknown_fields)}."
            )
        normalized = dict(values)
        if section == "data":
            for name in ("snapshot_path", "manifest_path", "cache_dir"):
                if name in normalized:
                    normalized[name] = Path(normalized[name])
            if "core_tenors" in normalized:
                normalized["core_tenors"] = tuple(normalized["core_tenors"])
            if (
                normalized.get("source_mode") in {"live", "live_then_snapshot"}
                and "end_date" not in normalized
            ):
                normalized["end_date"] = datetime.now(ZoneInfo("America/New_York")).date().isoformat()
                normalized.setdefault("maximum_staleness_days", 7)
        elif section == "pca" and "bootstrap_sensitivity_blocks" in normalized:
            normalized["bootstrap_sensitivity_blocks"] = tuple(normalized["bootstrap_sensitivity_blocks"])
        elif section == "forecast" and "interval_bootstrap_block_lengths" in normalized:
            normalized["interval_bootstrap_block_lengths"] = tuple(
                normalized["interval_bootstrap_block_lengths"]
            )
        elif section == "risk" and "factor_sigma_multiples" in normalized:
            normalized["factor_sigma_multiples"] = tuple(normalized["factor_sigma_multiples"])
        instances[section] = config_class(**normalized)
    return PipelineConfig(**instances)


def _write_effective_config(config: PipelineConfig, path: Path) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(_json_normalized(asdict(config)), indent=2, sort_keys=True) + "\n").encode("utf-8")
    temp_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            temp_path = Path(handle.name)
            handle.write(payload)
            handle.flush()
        temp_path.replace(path)
        temp_path = None
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def _decision_exit_code(
    validation_assessment: str,
    risk_assessment: str,
    *,
    warnings_accepted: bool,
) -> int:
    if validation_assessment not in {
        "READY TO SHARE",
        "SHARE WITH CAVEATS",
        "NEEDS REVISION",
    }:
        raise ValueError(f"Unknown validation assessment: {validation_assessment!r}.")
    if risk_assessment not in {
        "NOT REQUESTED",
        "SHARE WITH CAVEATS",
        "NEEDS REVISION",
    }:
        raise ValueError(f"Unknown risk assessment: {risk_assessment!r}.")
    assessments = {validation_assessment, risk_assessment}
    if "NEEDS REVISION" in assessments:
        return 4
    if "SHARE WITH CAVEATS" in assessments and not warnings_accepted:
        return 3
    return 0


def _normalize_warning_approval_id(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,127}", normalized):
        raise ValueError(
            "--warning-approval-id must be 1-128 characters using letters, numbers, "
            "period, underscore, colon, slash, or hyphen."
        )
    return normalized


def _write_pca_artifacts(stage: Path, prefix: str, fit) -> None:
    dimensions = len(fit.tenors)
    summary = pd.DataFrame(
        {
            "Eigenvalue": fit.eigenvalues,
            "Variance share": fit.explained_ratio,
            "Cumulative variance share": fit.explained_ratio.cumsum(),
            "Assigned shape": [*fit.assigned_shapes, *(["Residual"] * (dimensions - 3))],
            "Suggested shape": [*fit.suggested_shapes, *([""] * (dimensions - 3))],
            "Identification status": [*fit.identification_status, *([""] * (dimensions - 3))],
            "Template similarity": [*fit.template_similarity, *([float("nan")] * (dimensions - 3))],
            "Template dominance margin": [
                *fit.template_dominance_margin,
                *([float("nan")] * (dimensions - 3)),
            ],
        },
        index=fit.component_names,
    )
    summary.to_csv(stage / f"{prefix}_summary.csv")
    pd.DataFrame(
        {"Center (bp)": fit.center_bp, "Scale (bp)": fit.scale_bp},
        index=fit.tenors,
    ).to_csv(stage / f"{prefix}_center_scale.csv")
    fit.loading_table(dimensions).to_csv(stage / f"{prefix}_model_space_loadings.csv")
    fit.physical_basis_table(dimensions).to_csv(stage / f"{prefix}_physical_basis.csv")
    fit.sigma_shock_table(dimensions).to_csv(stage / f"{prefix}_one_sigma_shocks_bp.csv")
    fit.scores.to_csv(stage / f"{prefix}_scores.csv")


def _prediction_artifact_name(model: str) -> str:
    slug = "_".join(model.lower().replace("(", "").replace(")", "").split())
    return f"forecast_predictions_{slug}.csv"


def _expected_artifact_names(result, *, risk_requested: bool) -> set[str]:
    names = {
        "source_payload.csv",
        "weekly_yields_pct.csv",
        "weekly_changes_bp.csv",
        "data_quality.csv",
        "specification_challenge.csv",
        "methodology_sensitivity.csv",
        "structural_current_ewma_comparison.csv",
        "reconstruction_summary.csv",
        "reconstruction_by_tenor.csv",
        "rolling_stability.csv",
        "sequential_stability.csv",
        "bootstrap_stability.csv",
        "bootstrap_block_sensitivity.csv",
        "oos_reconstruction.csv",
        "oos_reconstruction_metrics.csv",
        "oos_reconstruction_audit.csv",
        "validation_algebra.csv",
        "validation_distribution.csv",
        "validation_model_health.csv",
    }
    for prefix in ("structural_pca", "correlation_pca", "current_ewma_pca"):
        names.update(
            {
                f"{prefix}_summary.csv",
                f"{prefix}_center_scale.csv",
                f"{prefix}_model_space_loadings.csv",
                f"{prefix}_physical_basis.csv",
                f"{prefix}_one_sigma_shocks_bp.csv",
                f"{prefix}_scores.csv",
            }
        )
    if result.forecast is not None:
        names.update(
            {
                "forecast_model_comparison.csv",
                "forecast_metrics.csv",
                "forecast_audit_trail.csv",
                "forecast_intervals_pct.csv",
                "forecast_interval_coverage.csv",
                "forecast_simultaneous_interval_diagnostic.csv",
                "forecast_full_history_interval_diagnostic.csv",
                "forecast_full_history_simultaneous_diagnostic.csv",
                "forecast_metadata.json",
                "forecast_actual_changes_bp.csv",
                "forecast_latest_change_bp.csv",
                "forecast_latest_level_pct.csv",
                *(_prediction_artifact_name(model) for model in result.forecast.predictions_bp),
            }
        )
    if risk_requested:
        names.update(
            {
                "key_rate_dv01_input.csv",
                "key_rate_dv01_normalized.csv",
                "key_rate_dv01_metadata.json",
                "structural_factor_exposure.csv",
                "structural_variance_reconciliation.csv",
                "structural_pure_factor_scenarios.csv",
                "structural_historical_factor_pnl.csv",
                "structural_historical_residual_pnl.csv",
                "current_ewma_factor_exposure.csv",
                "current_ewma_variance_reconciliation.csv",
                "current_ewma_pure_factor_scenarios.csv",
                "current_ewma_historical_factor_pnl.csv",
                "current_ewma_historical_residual_pnl.csv",
                "risk_calibration_comparison.csv",
                "historical_tail_risk.csv",
                "historical_tail_risk_full_history.csv",
                "historical_scenarios_recent.csv",
                "historical_scenarios_full_history.csv",
                "historical_full_pnl.csv",
                "historical_var_backtest_summary.csv",
                "historical_var_backtest_detail.csv",
            }
        )
    return names


def _json_normalized(value):
    return json.loads(json.dumps(value, sort_keys=True, default=str))


def _stable_provenance_identity(value) -> dict[str, object]:
    document = value.to_dict() if hasattr(value, "to_dict") else dict(value)
    for field_name in _INVOCATION_ONLY_PROVENANCE_FIELDS:
        document.pop(field_name, None)
    return _json_normalized(document)


def _analytic_identity(
    config: PipelineConfig,
    provenance,
    *,
    krd_sha256: str | None,
    krd_metadata: dict[str, object] | None,
) -> dict[str, object]:
    return {
        "package_version": __version__,
        "implementation_sha256": _implementation_sha256(),
        "environment": _environment_identity(),
        "config": _json_normalized(asdict(config)),
        "data_provenance": _stable_provenance_identity(provenance),
        "krd_sha256": krd_sha256,
        "risk_input_metadata": krd_metadata,
    }


def _forecast_metadata(forecast) -> dict[str, object] | None:
    if forecast is None:
        return None
    partitions: dict[str, dict[str, object]] = {}
    for label in ("Selection", "Confirmation", "Interval evaluation"):
        subset = forecast.audit_trail.loc[forecast.audit_trail["Evaluation partition"] == label]
        if subset.empty:
            raise RuntimeError(f"Forecast audit trail is missing partition {label!r}.")
        partitions[label] = {
            "start": pd.Timestamp(subset.index.min()).date().isoformat(),
            "end": pd.Timestamp(subset.index.max()).date().isoformat(),
            "observations": len(subset),
        }
    return {
        "selected_model": forecast.selected_model,
        "forecast_as_of": forecast.forecast_as_of.date().isoformat(),
        "target_period_end": forecast.target_period_end.date().isoformat(),
        "holdout_start": pd.Timestamp(forecast.audit_trail.index.min()).date().isoformat(),
        "holdout_end": pd.Timestamp(forecast.audit_trail.index.max()).date().isoformat(),
        "holdout_observations": len(forecast.audit_trail),
        "evaluation_partitions": partitions,
        "interval_gate": (
            "familywise circular moving-block bootstrap coverage bounds across tenors and nominal coverages"
        ),
        "full_history_interval_diagnostic_status": "conditional/post-selection diagnostic",
    }


def _validate_existing_run(
    run_dir: Path,
    run_id: str,
    *,
    expected_identity: dict[str, object],
    expected_artifacts: set[str],
    expected_controls: dict[str, object],
) -> Path:
    manifest_path = run_dir / "run_manifest.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Existing run directory is incomplete: {run_dir}.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Existing run manifest is unreadable: {manifest_path}.") from exc
    if manifest.get("run_id") != run_id:
        raise RuntimeError(f"Existing run manifest has the wrong identity: {run_dir}.")
    if manifest.get("artifact_schema_version") != 3:
        raise RuntimeError(f"Existing run manifest has an unsupported artifact schema: {run_dir}.")
    for name, expected in expected_identity.items():
        observed = manifest.get(name)
        if name == "data_provenance":
            observed = _stable_provenance_identity(observed or {})
        if _json_normalized(observed) != _json_normalized(expected):
            raise RuntimeError(f"Existing run manifest identity mismatch for {name}: {run_dir}.")
    for name, expected in expected_controls.items():
        if _json_normalized(manifest.get(name)) != _json_normalized(expected):
            raise RuntimeError(f"Existing run manifest control mismatch for {name}: {run_dir}.")
    artifact_hashes = manifest.get("artifact_sha256")
    if not isinstance(artifact_hashes, dict) or set(artifact_hashes) != expected_artifacts:
        raise RuntimeError(f"Existing run manifest has an incomplete artifact schema: {run_dir}.")
    artifact_contracts = manifest.get("artifact_contracts")
    if not isinstance(artifact_contracts, dict) or set(artifact_contracts) != expected_artifacts:
        raise RuntimeError(f"Existing run manifest has incomplete artifact contracts: {run_dir}.")
    observed_files = {path.name for path in run_dir.iterdir() if path.is_file()}
    if observed_files != expected_artifacts | {"run_manifest.json"}:
        raise RuntimeError(f"Existing run directory has missing or untracked artifacts: {run_dir}.")
    for name, expected_hash in artifact_hashes.items():
        relative = Path(name)
        if relative.is_absolute() or len(relative.parts) != 1:
            raise RuntimeError(f"Unsafe artifact name in existing run manifest: {name!r}.")
        artifact = run_dir / relative
        if not artifact.is_file() or _file_sha256(artifact) != expected_hash:
            raise RuntimeError(f"Existing run artifact failed SHA-256 validation: {artifact}.")
        if _json_normalized(_artifact_contract(artifact)) != _json_normalized(artifact_contracts[name]):
            raise RuntimeError(f"Existing run artifact failed structural validation: {artifact}.")
    return run_dir


def _append_execution_event(
    output_dir: Path,
    run_id: str,
    *,
    reused: bool,
    validation_assessment: str,
    risk_assessment: str,
    warning_approval_id: str | None,
    source_mode: str,
    source_acquired_at_utc: str | None,
    source_payload_sha256: str,
    network_error: str | None,
    cache_persistence_error: str | None,
    data_quality_non_pass: list[dict[str, object]],
) -> None:
    event = {
        "execution_id": str(uuid4()),
        "execution_status": "SUCCEEDED",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analytic_result_id": run_id,
        "analytic_result_reused": reused,
        "validation_assessment": validation_assessment,
        "risk_assessment": risk_assessment,
        "warning_approval_id": warning_approval_id,
        "source_mode": source_mode,
        "source_acquired_at_utc": source_acquired_at_utc,
        "source_payload_sha256": source_payload_sha256,
        "network_error": network_error,
        "cache_persistence_error": cache_persistence_error,
        "data_quality_non_pass": data_quality_non_pass,
    }
    _append_ledger_payload(output_dir, event)


def _execution_data_context(value) -> dict[str, object]:
    data = value.data if hasattr(value, "data") else value
    non_pass = data.quality_table.loc[data.quality_table["status"] != "PASS"]
    return {
        "source_mode": data.provenance.mode,
        "source_acquired_at_utc": data.provenance.source_acquired_at_utc,
        "source_payload_sha256": data.provenance.source_payload_sha256,
        "network_error": data.provenance.network_error,
        "cache_persistence_error": data.provenance.cache_persistence_error,
        "data_quality_non_pass": _json_normalized(non_pass.to_dict(orient="records")),
    }


def _append_ledger_payload(output_dir: Path, event: dict[str, object]) -> None:
    """Append exactly one durable JSONL record; serialize concurrent POSIX writers."""

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = (json.dumps(event, sort_keys=True) + "\n").encode("utf-8")
    ledger_path = output_dir / "execution_ledger.jsonl"
    descriptor = os.open(ledger_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
        remaining = memoryview(payload)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError("Execution-ledger write made no forward progress.")
            remaining = remaining[written:]
        os.fsync(descriptor)
    finally:
        if fcntl is not None:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _append_failure_event(
    output_dir: Path,
    *,
    failure_stage: str,
    error: Exception,
    warning_approval_id: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    event = {
        "execution_id": str(uuid4()),
        "execution_status": "FAILED",
        "executed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "analytic_result_id": None,
        "failure_stage": failure_stage,
        "error_type": type(error).__name__,
        "error_message": str(error)[:1_000],
        "warning_approval_id": warning_approval_id,
    }
    _append_ledger_payload(output_dir, event)


def _write_outputs(
    result,
    config: PipelineConfig,
    output_dir: Path,
    krd_path: Path | None,
    krd_metadata: dict[str, object] | None = None,
    warning_approval_id: str | None = None,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    krd: pd.Series | None = None
    krd_bytes: bytes | None = None
    krd_sha256: str | None = None
    if krd_path is not None:
        if krd_metadata is None:
            raise ValueError("Controlled KRD metadata are required when a KRD input is supplied.")
        krd_metadata = _validate_krd_metadata(krd_metadata)
        krd, krd_bytes, krd_sha256 = _read_krd(krd_path)
    elif krd_metadata is not None:
        raise ValueError("KRD metadata were supplied without a KRD input.")

    structural_risk = None
    current_risk = None
    risk_assessment = "NOT REQUESTED"
    risk_controls: dict[str, object] | None = None
    if krd is not None:
        structural_risk = map_linear_curve_risk(
            result.data.weekly_changes_bp,
            krd,
            result.structural_pca,
            config.risk,
            config.pca.retained_factors,
        )
        current_risk = map_linear_curve_risk(
            result.data.weekly_changes_bp,
            krd,
            result.current_ewma_pca,
            config.risk,
            config.pca.retained_factors,
        )
        pnl_difference = float(
            (structural_risk.historical_full_pnl_usd - current_risk.historical_full_pnl_usd).abs().max()
        )
        if pnl_difference > 1.0e-6:
            raise RuntimeError("Structural and recency-weighted full key-rate P&L do not reconcile.")
        backtest_status = str(structural_risk.var_backtest_summary.loc["Rolling historical VaR", "Status"])
        minimum_tail_mass = float(structural_risk.tail_risk_summary["Effective tail mass"].min())
        market_as_of = result.data.weekly_yields_pct.index.max().date()
        krd_as_of = date.fromisoformat(str(krd_metadata["valuation_date"]))
        signed_as_of_gap_days = (krd_as_of - market_as_of).days
        as_of_gap_days = abs(signed_as_of_gap_days)
        same_curve_basis_self_attested = str(krd_metadata["sensitivity_curve_id"]) == "UST Treasury CMT H.15"
        tail_mass_statuses = set(structural_risk.tail_risk_summary["Tail-mass review status"])
        statistical_adequacy = backtest_status == "PASS" and tail_mass_statuses == {"REVIEW LEVEL MET"}
        risk_assessment = (
            "NEEDS REVISION"
            if (
                not same_curve_basis_self_attested
                or signed_as_of_gap_days > 0
                or as_of_gap_days > config.risk.maximum_krd_asof_gap_days
                or not statistical_adequacy
            )
            else "SHARE WITH CAVEATS"
        )
        risk_controls = {
            "linear_pnl_reconciliation_max_abs_usd": pnl_difference,
            "rolling_var_backtest_status": backtest_status,
            "minimum_effective_tail_mass": minimum_tail_mass,
            "tail_mass_review_threshold": config.risk.minimum_effective_tail_mass_review,
            "tail_mass_review_status": structural_risk.tail_risk_summary["Tail-mass review status"].to_dict(),
            "market_data_as_of": market_as_of.isoformat(),
            "krd_valuation_date": krd_as_of.isoformat(),
            "absolute_as_of_gap_days": as_of_gap_days,
            "signed_krd_minus_market_as_of_days": signed_as_of_gap_days,
            "as_of_direction": (
                "KRD AFTER MARKET DATA"
                if signed_as_of_gap_days > 0
                else "KRD BEFORE MARKET DATA"
                if signed_as_of_gap_days < 0
                else "SAME DATE"
            ),
            "maximum_krd_asof_gap_days": config.risk.maximum_krd_asof_gap_days,
            "same_curve_basis_self_attested": same_curve_basis_self_attested,
            "krd_metadata_lineage_status": "CALLER ATTESTED; NOT VERIFIED",
            "statistical_adequacy_for_risk_use": statistical_adequacy,
            "statistical_adequacy_note": (
                "Requires rolling VaR PASS and every recent-window tail measure to meet "
                "the configured effective-tail-mass review level; this remains necessary "
                "but not sufficient for organizational approval."
            ),
            "conditionally_permitted_technical_scope": (
                "descriptive linear KRD scenario risk only, subject to external governance"
            ),
            "prohibited_uses": [
                "current or EOD risk",
                "official limits",
                "official VaR/ES",
                "actual or hypothetical P&L backtesting",
                "PLA",
                "regulatory capital",
                "FRTB",
                "pricing",
                "management forecasting",
                "autonomous hedging",
                "trading",
            ],
        }
    expected_identity = _analytic_identity(
        config,
        result.data.provenance,
        krd_sha256=krd_sha256,
        krd_metadata=krd_metadata,
    )
    implementation_sha256 = str(expected_identity["implementation_sha256"])
    environment = expected_identity["environment"]
    normalized_config = expected_identity["config"]
    forecast_metadata = _forecast_metadata(result.forecast)
    expected_controls: dict[str, object] = {
        "validation_assessment": result.validation.overall_assessment,
        "forecast_status": result.forecast_note,
        "selected_forecast_model": (None if result.forecast is None else result.forecast.selected_model),
        "forecast_metadata": forecast_metadata,
        "risk_mapping": "completed" if krd is not None else "not requested",
        "risk_assessment": risk_assessment,
        "risk_controls": risk_controls,
    }
    expected_artifacts = _expected_artifact_names(result, risk_requested=krd is not None)
    identity_payload = json.dumps(
        expected_identity,
        sort_keys=True,
        default=str,
    ).encode()
    run_id = sha256(identity_payload).hexdigest()
    run_dir = output_dir / run_id
    if run_dir.exists():
        validated = _validate_existing_run(
            run_dir,
            run_id,
            expected_identity=expected_identity,
            expected_artifacts=expected_artifacts,
            expected_controls=expected_controls,
        )
        _append_execution_event(
            output_dir,
            run_id,
            reused=True,
            validation_assessment=result.validation.overall_assessment,
            risk_assessment=risk_assessment,
            warning_approval_id=warning_approval_id,
            **_execution_data_context(result),
        )
        return validated

    stage = Path(tempfile.mkdtemp(prefix=f".{run_id}.", dir=output_dir))
    try:
        (stage / "source_payload.csv").write_bytes(result.data.source_payload)
        result.data.weekly_yields_pct.to_csv(stage / "weekly_yields_pct.csv")
        result.data.weekly_changes_bp.to_csv(stage / "weekly_changes_bp.csv")
        result.data.quality_table.to_csv(stage / "data_quality.csv", index=False)
        result.specification_challenge.to_csv(stage / "specification_challenge.csv")
        result.methodology_sensitivity.to_csv(stage / "methodology_sensitivity.csv")
        result.structural_current_ewma_comparison.to_csv(stage / "structural_current_ewma_comparison.csv")
        result.reconstruction_summary.to_csv(stage / "reconstruction_summary.csv", index=False)
        result.reconstruction_by_tenor.to_csv(stage / "reconstruction_by_tenor.csv")
        result.rolling_stability.to_csv(stage / "rolling_stability.csv")
        result.sequential_stability.to_csv(stage / "sequential_stability.csv")
        result.bootstrap_stability.to_csv(stage / "bootstrap_stability.csv")
        result.bootstrap_block_sensitivity.to_csv(stage / "bootstrap_block_sensitivity.csv")
        result.oos_reconstruction.to_csv(stage / "oos_reconstruction.csv")
        result.oos_reconstruction_metrics.to_csv(stage / "oos_reconstruction_metrics.csv")
        result.oos_reconstruction_audit.to_csv(stage / "oos_reconstruction_audit.csv")
        _write_pca_artifacts(stage, "structural_pca", result.structural_pca)
        _write_pca_artifacts(stage, "correlation_pca", result.correlation_pca)
        _write_pca_artifacts(stage, "current_ewma_pca", result.current_ewma_pca)
        result.validation.algebra_checks.to_csv(stage / "validation_algebra.csv")
        result.validation.distribution_checks.to_csv(stage / "validation_distribution.csv")
        result.validation.model_health.to_csv(stage / "validation_model_health.csv")
        if result.forecast is not None:
            result.forecast.model_comparison.to_csv(stage / "forecast_model_comparison.csv")
            result.forecast.metrics.to_csv(stage / "forecast_metrics.csv")
            result.forecast.audit_trail.to_csv(stage / "forecast_audit_trail.csv")
            result.forecast.prediction_intervals_pct.to_csv(stage / "forecast_intervals_pct.csv")
            result.forecast.interval_coverage.to_csv(stage / "forecast_interval_coverage.csv")
            result.forecast.simultaneous_interval_diagnostic.to_csv(
                stage / "forecast_simultaneous_interval_diagnostic.csv"
            )
            result.forecast.full_history_interval_diagnostic.to_csv(
                stage / "forecast_full_history_interval_diagnostic.csv"
            )
            result.forecast.full_history_simultaneous_diagnostic.to_csv(
                stage / "forecast_full_history_simultaneous_diagnostic.csv"
            )
            (stage / "forecast_metadata.json").write_text(
                json.dumps(forecast_metadata, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            result.forecast.actual_changes_bp.to_csv(stage / "forecast_actual_changes_bp.csv")
            result.forecast.latest_change_forecast_bp.to_csv(
                stage / "forecast_latest_change_bp.csv", header=True
            )
            result.forecast.latest_level_forecast_pct.to_csv(
                stage / "forecast_latest_level_pct.csv", header=True
            )
            for model, predictions in result.forecast.predictions_bp.items():
                predictions.to_csv(stage / _prediction_artifact_name(model))

        risk_status = "not requested"
        if structural_risk is not None and current_risk is not None:
            assert krd is not None and krd_bytes is not None and krd_metadata is not None
            (stage / "key_rate_dv01_input.csv").write_bytes(krd_bytes)
            structural_risk.key_rate_dv01_usd_per_bp.to_csv(
                stage / "key_rate_dv01_normalized.csv", header=True
            )
            (stage / "key_rate_dv01_metadata.json").write_text(
                json.dumps(krd_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            for prefix, risk in (
                ("structural", structural_risk),
                ("current_ewma", current_risk),
            ):
                risk.factor_exposure_usd_per_score.to_csv(
                    stage / f"{prefix}_factor_exposure.csv", header=True
                )
                risk.variance_reconciliation.to_csv(stage / f"{prefix}_variance_reconciliation.csv")
                risk.pure_factor_scenarios.to_csv(stage / f"{prefix}_pure_factor_scenarios.csv")
                risk.historical_factor_pnl_usd.to_csv(stage / f"{prefix}_historical_factor_pnl.csv")
                risk.historical_residual_pnl_usd.to_csv(
                    stage / f"{prefix}_historical_residual_pnl.csv", header=True
                )
            calibration_comparison = pd.DataFrame(
                {
                    "Weekly modeled volatility (USD)": {
                        "Structural covariance": structural_risk.variance_reconciliation.loc[
                            "Total", "Variance (USD²)"
                        ]
                        ** 0.5,
                        "Current EWMA covariance": current_risk.variance_reconciliation.loc[
                            "Total", "Variance (USD²)"
                        ]
                        ** 0.5,
                    }
                }
            )
            calibration_comparison["Ratio to structural"] = (
                calibration_comparison["Weekly modeled volatility (USD)"]
                / calibration_comparison.loc["Structural covariance", "Weekly modeled volatility (USD)"]
            )
            calibration_comparison.to_csv(stage / "risk_calibration_comparison.csv")
            structural_risk.tail_risk_summary.to_csv(stage / "historical_tail_risk.csv")
            structural_risk.full_history_tail_risk_summary.to_csv(
                stage / "historical_tail_risk_full_history.csv"
            )
            structural_risk.historical_scenarios.to_csv(stage / "historical_scenarios_recent.csv")
            structural_risk.full_history_scenarios.to_csv(stage / "historical_scenarios_full_history.csv")
            structural_risk.historical_full_pnl_usd.to_csv(stage / "historical_full_pnl.csv", header=True)
            structural_risk.var_backtest_summary.to_csv(stage / "historical_var_backtest_summary.csv")
            structural_risk.var_backtest_detail.to_csv(stage / "historical_var_backtest_detail.csv")
            risk_status = "completed"

        observed_artifacts = {artifact.name for artifact in stage.iterdir() if artifact.is_file()}
        if observed_artifacts != expected_artifacts:
            missing = sorted(expected_artifacts - observed_artifacts)
            unexpected = sorted(observed_artifacts - expected_artifacts)
            raise RuntimeError(
                f"Output artifact schema mismatch; missing={missing}, unexpected={unexpected}."
            )

        artifact_hashes = {
            artifact.name: _file_sha256(artifact)
            for artifact in sorted(stage.iterdir())
            if artifact.is_file()
        }
        artifact_contracts = {
            artifact.name: _artifact_contract(artifact)
            for artifact in sorted(stage.iterdir())
            if artifact.is_file()
        }
        manifest = {
            "run_id": run_id,
            "artifact_schema_version": 3,
            "package_version": __version__,
            "implementation_sha256": implementation_sha256,
            "environment": environment,
            "completed_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": normalized_config,
            "data_provenance": result.data.provenance.to_dict(),
            "validation_assessment": result.validation.overall_assessment,
            "forecast_status": result.forecast_note,
            "selected_forecast_model": None if result.forecast is None else result.forecast.selected_model,
            "forecast_metadata": forecast_metadata,
            "risk_mapping": risk_status,
            "krd_sha256": krd_sha256,
            "risk_input_metadata": krd_metadata,
            "risk_assessment": risk_assessment,
            "risk_controls": risk_controls,
            "artifact_sha256": artifact_hashes,
            "artifact_contracts": artifact_contracts,
        }
        (stage / "run_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
        )
        try:
            stage.replace(run_dir)
        except OSError:
            if run_dir.exists():
                validated = _validate_existing_run(
                    run_dir,
                    run_id,
                    expected_identity=expected_identity,
                    expected_artifacts=expected_artifacts,
                    expected_controls=expected_controls,
                )
                _append_execution_event(
                    output_dir,
                    run_id,
                    reused=True,
                    validation_assessment=result.validation.overall_assessment,
                    risk_assessment=risk_assessment,
                    warning_approval_id=warning_approval_id,
                    **_execution_data_context(result),
                )
                return validated
            raise
        stage = None
        _append_execution_event(
            output_dir,
            run_id,
            reused=False,
            validation_assessment=result.validation.overall_assessment,
            risk_assessment=risk_assessment,
            warning_approval_id=warning_approval_id,
            **_execution_data_context(result),
        )
        return run_dir
    finally:
        if stage is not None and stage.exists():
            shutil.rmtree(stage)


def _provisional_output_dir(argv: list[str]) -> Path:
    """Find a ledger destination without accepting or executing other arguments."""

    output_dir = Path("outputs")
    for position, token in enumerate(argv):
        if token.startswith("--output-dir=") and token.partition("=")[2]:
            output_dir = Path(token.partition("=")[2])
        elif token == "--output-dir" and position + 1 < len(argv) and not argv[position + 1].startswith("-"):
            output_dir = Path(argv[position + 1])
    return output_dir


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    ledger_output_dir = _provisional_output_dir(raw_argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    failure_stage = "argument parsing"
    warning_approval_id: str | None = None
    try:
        args = _parser().parse_args(raw_argv)
        ledger_output_dir = args.output_dir
        LOGGER.setLevel(getattr(logging, args.log_level))
        failure_stage = "argument validation"
        warning_approval_id = _normalize_warning_approval_id(args.warning_approval_id)
        if args.accept_warnings and warning_approval_id is None:
            raise ValueError("--warning-approval-id is required with --accept-warnings.")
        if warning_approval_id is not None and not args.accept_warnings:
            raise ValueError("--warning-approval-id requires --accept-warnings.")
        krd_metadata = _krd_metadata_from_args(args)
        failure_stage = "configuration"
        if args.config_json is not None:
            explicit_data_overrides = {
                "--start-date": args.start_date,
                "--end-date": args.end_date,
                "--source-mode": args.source_mode,
                "--maximum-staleness-days": args.maximum_staleness_days,
            }
            conflicts = [name for name, value in explicit_data_overrides.items() if value is not None]
            if conflicts:
                raise ValueError(f"--config-json cannot be combined with data overrides: {conflicts}.")
            config = _load_pipeline_config(args.config_json)
        else:
            data_config = _data_config_from_args(args)
            config = PipelineConfig(data=data_config)
        if args.skip_forecast:
            config = replace(config, forecast=replace(config.forecast, enabled=False))
        if args.dump_effective_config is not None:
            _write_effective_config(config, args.dump_effective_config)
        LOGGER.info("Running yield-curve PCA %s", __version__)
        failure_stage = "pipeline execution"
        result = run_pipeline(config, project_root=args.project_root.resolve())
        warning_count = int((result.data.quality_table["status"] == "WARN").sum())
        LOGGER.info(
            "Data mode=%s | weekly_changes=%d | data_warnings=%d",
            result.data.provenance.mode,
            len(result.data.weekly_changes_bp),
            warning_count,
        )
        failure_stage = "controlled output commit"
        run_dir = _write_outputs(
            result,
            config,
            args.output_dir.resolve(),
            args.krd_csv,
            krd_metadata,
            warning_approval_id,
        )
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        risk_assessment = manifest["risk_assessment"]
        LOGGER.info(
            "Completed: %s | validation=%s | forecast=%s",
            run_dir,
            result.validation.overall_assessment,
            None if result.forecast is None else result.forecast.selected_model,
        )
        exit_code = _decision_exit_code(
            result.validation.overall_assessment,
            risk_assessment,
            warnings_accepted=args.accept_warnings,
        )
        if exit_code == 4:
            LOGGER.error("Validation failed; outputs are diagnostic only.")
            return 4
        if exit_code == 3:
            LOGGER.error("Validation has warnings; rerun with --accept-warnings only after review.")
            return 3
        return exit_code
    except Exception as exc:
        try:
            _append_failure_event(
                ledger_output_dir.resolve(),
                failure_stage=failure_stage,
                error=exc,
                warning_approval_id=warning_approval_id,
            )
        except Exception as ledger_error:
            LOGGER.error(
                "Failure ledger write also failed: %s: %s",
                type(ledger_error).__name__,
                ledger_error,
            )
        LOGGER.error("Run failed: %s: %s", type(exc).__name__, exc)
        return 2


if __name__ == "__main__":
    sys.exit(main())
