"""Official H.15 ingestion, deterministic snapshot handling, and data QA."""

from __future__ import annotations

import gzip
import json
import os
import tempfile
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from hashlib import sha256
from importlib.resources import as_file, files
from io import BytesIO, StringIO
from pathlib import Path
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from .config import ALL_TENORS, FRED_SERIES, DataConfig

try:
    import fcntl
except ImportError:
    fcntl = None

try:
    import msvcrt
except ImportError:
    msvcrt = None

FRED_CSV_ENDPOINT = "https://fred.stlouisfed.org/graph/fredgraph.csv"


class LiveSourceUnavailableError(RuntimeError):
    """The live endpoint could not be reached after bounded retries."""


class LivePayloadValidationError(ValueError):
    """The live endpoint responded, but its payload failed integrity controls."""


@dataclass(frozen=True)
class DataProvenance:
    mode: str
    requested_start: str
    requested_end: str
    observed_start: str
    observed_end: str
    loaded_at_utc: str
    source_acquired_at_utc: str | None
    source_as_of_date: str
    content_sha256: str
    source_payload_sha256: str
    rows: int
    columns: int
    units: str = "percent_per_annum"
    source: str = "Federal Reserve H.15 through FRED"
    historical_vintage_status: str = "single frozen/latest-revised history; not a point-in-time vintage panel"
    network_error: str | None = None
    cache_persistence_error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class QualityCheck:
    control: str
    status: Literal["PASS", "WARN", "FAIL"]
    result: str
    threshold: str
    risk: str


@dataclass(frozen=True)
class CurveDataBundle:
    daily_yields_pct: pd.DataFrame
    weekly_yields_pct: pd.DataFrame
    weekly_changes_bp: pd.DataFrame
    observation_gap_days: pd.Series
    provenance: DataProvenance
    quality_checks: tuple[QualityCheck, ...]
    source_payload: bytes

    @property
    def quality_table(self) -> pd.DataFrame:
        return pd.DataFrame(asdict(check) for check in self.quality_checks)

    def raise_for_failures(self) -> None:
        failed = [check for check in self.quality_checks if check.status == "FAIL"]
        if failed:
            details = "; ".join(f"{item.control}: {item.result}" for item in failed)
            raise ValueError(f"Curve data failed production controls: {details}")


def _resolve(path: Path, project_root: Path | None) -> Path:
    if path.is_absolute():
        return path
    root = Path.cwd() if project_root is None else Path(project_root)
    return (root / path).resolve()


def _canonical_csv_bytes(frame: pd.DataFrame) -> bytes:
    return frame.to_csv(date_format="%Y-%m-%d", float_format="%.8f").encode("utf-8")


def normalize_h15_frame(raw: pd.DataFrame, config: DataConfig) -> pd.DataFrame:
    """Normalize FRED IDs or tenor-labelled input without filling observations."""

    if not isinstance(raw, pd.DataFrame) or raw.empty:
        raise ValueError("H.15 payload is empty or is not a DataFrame.")
    frame = raw.copy()
    if frame.columns.duplicated().any():
        raise ValueError("H.15 payload contains duplicate column labels.")
    date_candidates = [name for name in ("observation_date", "DATE", "Date") if name in frame.columns]
    if len(date_candidates) != 1:
        raise ValueError(f"Expected exactly one date column; found {date_candidates!r}.")
    date_column = date_candidates[0]

    series_to_tenor = dict(FRED_SERIES)
    if set(series_to_tenor).issubset(frame.columns):
        frame = frame.rename(columns=series_to_tenor)
    if frame.columns.duplicated().any():
        raise ValueError("H.15 payload mixes FRED series IDs with duplicate tenor labels.")
    missing = sorted(set(ALL_TENORS) - set(frame.columns))
    if missing:
        raise ValueError(f"H.15 payload is missing required tenors: {missing}.")

    frame = frame[[date_column, *ALL_TENORS]].rename(columns={date_column: "Date"})
    frame["Date"] = pd.to_datetime(frame["Date"], errors="raise")
    if getattr(frame["Date"].dt, "tz", None) is not None:
        frame["Date"] = frame["Date"].dt.tz_localize(None)
    for tenor in ALL_TENORS:
        values = frame[tenor]
        text = values.astype("string").str.strip()
        approved_missing = values.isna() | text.isin(("", "."))
        cleaned = values.mask(approved_missing)
        try:
            frame[tenor] = pd.to_numeric(cleaned, errors="raise")
        except (TypeError, ValueError) as exc:
            coerced = pd.to_numeric(cleaned, errors="coerce")
            bad = (~approved_missing) & coerced.isna()
            examples = [f"{frame.loc[index, 'Date']}={values.loc[index]!r}" for index in frame.index[bad][:3]]
            raise ValueError(
                f"H.15 payload contains unapproved non-numeric tokens for {tenor}: {examples}."
            ) from exc

    frame = frame.set_index("Date").sort_index()
    duplicate_count = int(frame.index.duplicated(keep=False).sum())
    if duplicate_count:
        raise ValueError(f"H.15 payload contains {duplicate_count} rows with duplicate dates.")
    frame = frame.loc[config.start_date : config.end_date]
    frame.index.name = "Date"

    if frame.empty:
        raise ValueError("No H.15 observations fall inside the requested date range.")
    if frame[list(config.core_tenors)].dropna(how="any").empty:
        raise ValueError("No jointly observed core yield curves fall inside the requested date range.")
    return frame


def _read_snapshot_paths(
    snapshot_path: Path,
    manifest_path: Path,
    config: DataConfig,
) -> tuple[pd.DataFrame, bytes, str | None]:
    if not snapshot_path.is_file() or not manifest_path.is_file():
        raise FileNotFoundError(
            f"Deterministic snapshot or manifest is missing: {snapshot_path}, {manifest_path}."
        )
    if manifest_path.stat().st_size > 1_000_000:
        raise ValueError("H.15 snapshot manifest exceeds the 1 MB control limit.")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        expected_compressed_bytes = int(manifest["compressed_bytes"])
        expected_uncompressed_bytes = int(manifest["uncompressed_bytes"])
        expected_rows = int(manifest["rows_excluding_header"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise ValueError("H.15 snapshot manifest is malformed or incomplete.") from exc
    if manifest.get("units") != "percent_per_annum":
        raise ValueError("H.15 snapshot manifest has an unsupported unit convention.")
    if manifest.get("series_ids") != [series_id for series_id, _ in FRED_SERIES]:
        raise ValueError("H.15 snapshot manifest series universe does not match the model contract.")
    if manifest.get("columns") != len(FRED_SERIES) + 1:
        raise ValueError("H.15 snapshot manifest column count does not match the model contract.")
    compressed_bytes = snapshot_path.stat().st_size
    if compressed_bytes != expected_compressed_bytes:
        raise ValueError("Compressed H.15 snapshot byte count does not match its manifest.")
    if compressed_bytes > config.maximum_response_bytes:
        raise ValueError("Compressed H.15 snapshot exceeds the configured size limit.")
    compressed_digest = sha256()
    with snapshot_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            compressed_digest.update(chunk)
    observed_compressed_hash = compressed_digest.hexdigest()
    if observed_compressed_hash != manifest["compressed_sha256"]:
        raise ValueError("Compressed H.15 snapshot failed SHA-256 validation.")
    try:
        with gzip.open(snapshot_path, "rb") as handle:
            raw_bytes = handle.read(config.maximum_response_bytes + 1)
    except (gzip.BadGzipFile, EOFError, OSError) as exc:
        raise ValueError("H.15 snapshot is not a valid gzip stream.") from exc
    if len(raw_bytes) > config.maximum_response_bytes:
        raise ValueError("Uncompressed H.15 snapshot exceeds the configured size limit.")
    if len(raw_bytes) != expected_uncompressed_bytes:
        raise ValueError("Uncompressed H.15 snapshot byte count does not match its manifest.")
    if sha256(raw_bytes).hexdigest() != manifest["uncompressed_sha256"]:
        raise ValueError("Uncompressed H.15 snapshot failed SHA-256 validation.")
    raw = pd.read_csv(BytesIO(raw_bytes))
    if len(raw) != expected_rows:
        raise ValueError("H.15 snapshot row count does not match its manifest.")
    date_column = next(
        (name for name in ("observation_date", "DATE", "Date") if name in raw.columns),
        None,
    )
    if date_column is None:
        raise ValueError("H.15 snapshot has no recognized date column.")
    raw_dates = pd.to_datetime(raw[date_column], errors="raise")
    if str(raw_dates.min().date()) != manifest.get("first_date") or str(
        raw_dates.max().date()
    ) != manifest.get("last_date"):
        raise ValueError("H.15 snapshot date bounds do not match its manifest.")
    acquired_at = manifest.get("snapshot_created_at_utc")
    if acquired_at is not None:
        try:
            parsed_acquired_at = datetime.fromisoformat(acquired_at)
        except (TypeError, ValueError) as exc:
            raise ValueError("H.15 snapshot acquisition timestamp is invalid.") from exc
        if parsed_acquired_at.tzinfo is None:
            raise ValueError("H.15 snapshot acquisition timestamp must include a timezone.")
    return normalize_h15_frame(raw, config), raw_bytes, acquired_at


def _load_snapshot(
    config: DataConfig,
    project_root: Path | None,
) -> tuple[pd.DataFrame, bytes, str | None]:
    snapshot_path = _resolve(config.snapshot_path, project_root)
    manifest_path = _resolve(config.manifest_path, project_root)
    if snapshot_path.is_file() and manifest_path.is_file():
        return _read_snapshot_paths(snapshot_path, manifest_path, config)

    default_snapshot = DataConfig.__dataclass_fields__["snapshot_path"].default
    default_manifest = DataConfig.__dataclass_fields__["manifest_path"].default
    if config.snapshot_path == default_snapshot and config.manifest_path == default_manifest:
        resource_root = files("yield_curve_pca.resources")
        with (
            as_file(resource_root.joinpath("h15_treasury_cmt_2000_2025.csv.gz")) as packaged_snapshot,
            as_file(resource_root.joinpath("source_manifest.json")) as packaged_manifest,
        ):
            return _read_snapshot_paths(packaged_snapshot, packaged_manifest, config)
    raise FileNotFoundError(
        f"Deterministic snapshot or manifest is missing: {snapshot_path}, {manifest_path}."
    )


def _cache_paths(config: DataConfig, project_root: Path | None) -> tuple[Path, Path]:
    cache_dir = _resolve(config.cache_dir, project_root)
    safe_key = sha256(
        (config.start_date + "|" + config.end_date + "|" + ",".join(x for x, _ in FRED_SERIES)).encode()
    ).hexdigest()[:16]
    return cache_dir / f"h15_{safe_key}.csv", cache_dir / f"h15_{safe_key}.metadata.json"


def _cache_query_identity(config: DataConfig) -> dict[str, object]:
    return {
        "start_date": config.start_date,
        "end_date": config.end_date,
        "series_ids": [series_id for series_id, _ in FRED_SERIES],
    }


def _cache_source_path(csv_path: Path) -> Path:
    return csv_path.with_suffix(".source.csv")


def _end_coverage_metrics(weekly: pd.DataFrame, config: DataConfig) -> tuple[int, int]:
    observed_end = weekly.index.max()
    requested_end = pd.Timestamp(config.end_date)
    calendar_lag = (requested_end - observed_end).days
    period_end = observed_end.to_period(config.weekly_rule).end_time.normalize()
    eligible_period_cutoff = min(requested_end, period_end)
    within_week_lag = (eligible_period_cutoff - observed_end).days
    return int(calendar_lag), int(within_week_lag)


def _expected_boundary_periods(
    config: DataConfig,
    complete_core: pd.DataFrame,
) -> tuple[pd.Period, pd.Period]:
    if complete_core.empty:
        raise ValueError("No complete core curves are available for boundary-period controls.")
    requested_start = pd.Timestamp(config.start_date)
    requested_end = pd.Timestamp(config.end_date)
    start_period = complete_core.index.min().to_period(config.weekly_rule)
    end_period = complete_core.index.max().to_period(config.weekly_rule)
    if config.boundary_week_policy == "drop" and requested_start > start_period.start_time.normalize():
        start_period += 1
    if config.boundary_week_policy == "drop" and requested_end < end_period.end_time.normalize():
        end_period -= 1
    return start_period, end_period


def _read_fresh_cache(
    config: DataConfig,
    project_root: Path | None,
) -> tuple[pd.DataFrame, bytes, str] | None:
    if not config.use_cache:
        return None
    csv_path, metadata_path = _cache_paths(config, project_root)
    source_path = _cache_source_path(csv_path)
    if not csv_path.is_file() or not source_path.is_file() or not metadata_path.is_file():
        return None
    try:
        if csv_path.stat().st_size > config.maximum_response_bytes:
            return None
        if metadata_path.stat().st_size > 100_000:
            return None
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        if metadata.get("cache_schema_version") != 2:
            return None
        if metadata.get("query") != _cache_query_identity(config):
            return None
        retrieved_at = datetime.fromisoformat(metadata["retrieved_at_utc"])
        if retrieved_at.tzinfo is None:
            return None
        age_hours = (
            datetime.now(timezone.utc) - retrieved_at.astimezone(timezone.utc)
        ).total_seconds() / 3600.0
        if age_hours < 0 or age_hours > config.cache_max_age_hours:
            return None
        if source_path.stat().st_size > config.maximum_response_bytes:
            return None
        raw_bytes = csv_path.read_bytes()
        if len(raw_bytes) != metadata.get("canonical_bytes"):
            return None
        if sha256(raw_bytes).hexdigest() != metadata.get("canonical_sha256"):
            return None
        source_bytes = source_path.read_bytes()
        if len(source_bytes) != metadata.get("source_payload_bytes"):
            return None
        if sha256(source_bytes).hexdigest() != metadata.get("source_payload_sha256"):
            return None
        frame = normalize_h15_frame(pd.read_csv(BytesIO(raw_bytes)), config)
    except (
        KeyError,
        TypeError,
        UnicodeError,
        ValueError,
        OSError,
        json.JSONDecodeError,
        pd.errors.ParserError,
    ):
        return None
    complete_core = frame.loc[:, list(config.core_tenors)].dropna(how="any")
    if complete_core.empty:
        return None
    start_lag_days = (complete_core.index.min() - pd.Timestamp(config.start_date)).days
    try:
        weekly, changes, gaps = synchronize_weekly_curve(frame, config)
    except ValueError:
        return None
    end_lag_days, within_week_lag_days = _end_coverage_metrics(weekly, config)
    expected_start_period, expected_end_period = _expected_boundary_periods(config, complete_core)
    if (
        start_lag_days < 0
        or start_lag_days > config.maximum_start_coverage_gap_days
        or end_lag_days < 0
        or end_lag_days > config.maximum_end_coverage_gap_days
        or within_week_lag_days < 0
        or within_week_lag_days > config.maximum_within_week_observation_lag_days
        or weekly.index.min().to_period(config.weekly_rule) != expected_start_period
        or weekly.index.max().to_period(config.weekly_rule) != expected_end_period
        or len(changes) < config.minimum_complete_weeks
    ):
        return None
    quality = _quality_checks(frame, weekly, changes, gaps, config, "LOCAL_CACHE")
    if any(check.status == "FAIL" for check in quality):
        return None
    return frame, source_bytes, metadata["retrieved_at_utc"]


@contextmanager
def _cache_write_lock(lock_path: Path, timeout_seconds: float = 5.0):
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    acquired = False
    exclusive_fallback = False
    windows_lock = False
    try:
        if fcntl is not None:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            while not acquired:
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except BlockingIOError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out waiting for cache lock {lock_path}.") from None
                    time.sleep(0.05)
        elif msvcrt is not None:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            while not acquired:
                try:
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                    acquired = True
                    windows_lock = True
                except OSError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out waiting for cache lock {lock_path}.") from None
                    time.sleep(0.05)
        else:
            while descriptor is None:
                try:
                    descriptor = os.open(
                        lock_path,
                        os.O_CREAT | os.O_EXCL | os.O_RDWR,
                        0o600,
                    )
                    acquired = True
                    exclusive_fallback = True
                except FileExistsError:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Timed out waiting for cache lock {lock_path}.") from None
                    time.sleep(0.05)
        assert descriptor is not None
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        if descriptor is not None:
            if acquired and fcntl is not None:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            elif acquired and windows_lock:
                os.lseek(descriptor, 0, os.SEEK_SET)
                msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
            os.close(descriptor)
        if exclusive_fallback:
            lock_path.unlink(missing_ok=True)


def _atomic_cache_write(
    canonical_bytes: bytes,
    config: DataConfig,
    project_root: Path | None,
    *,
    source_payload: bytes | None = None,
) -> None:
    if not config.use_cache:
        return
    csv_path, metadata_path = _cache_paths(config, project_root)
    source_path = _cache_source_path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with _cache_write_lock(csv_path.with_suffix(".lock")):
        source_bytes = canonical_bytes if source_payload is None else source_payload
        if (
            len(canonical_bytes) > config.maximum_response_bytes
            or len(source_bytes) > config.maximum_response_bytes
        ):
            raise ValueError("Cache payload exceeds the configured size limit.")
        metadata_bytes = (
            json.dumps(
                {
                    "cache_schema_version": 2,
                    "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "query": _cache_query_identity(config),
                    "canonical_bytes": len(canonical_bytes),
                    "canonical_sha256": sha256(canonical_bytes).hexdigest(),
                    "source_payload_bytes": len(source_bytes),
                    "source_payload_sha256": sha256(source_bytes).hexdigest(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("ascii")
        for target, payload in (
            (csv_path, canonical_bytes),
            (source_path, source_bytes),
            (metadata_path, metadata_bytes),
        ):
            temp_path: Path | None = None
            try:
                with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as handle:
                    temp_path = Path(handle.name)
                    handle.write(payload)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, target)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)


def _download_live(config: DataConfig) -> tuple[pd.DataFrame, bytes]:
    query = urlencode(
        {
            "id": ",".join(series_id for series_id, _ in FRED_SERIES),
            "cosd": config.start_date,
            "coed": config.end_date,
        }
    )
    url = f"{FRED_CSV_ENDPOINT}?{query}"
    last_error: Exception | None = None
    for attempt in range(config.retries):
        try:
            request = Request(
                url,
                headers={
                    "User-Agent": "USYieldCurvePCA/3.0 (+research-and-risk-control)",
                    "Accept": "text/csv,*/*;q=0.8",
                    "Connection": "close",
                },
            )
            with urlopen(request, timeout=config.request_timeout_seconds) as response:
                declared_length = response.headers.get("Content-Length")
                try:
                    exceeds_declared_limit = (
                        bool(declared_length) and int(declared_length) > config.maximum_response_bytes
                    )
                except (TypeError, ValueError) as exc:
                    raise LivePayloadValidationError(
                        "FRED response has an invalid Content-Length header."
                    ) from exc
                if exceeds_declared_limit:
                    raise LivePayloadValidationError("FRED response exceeds the configured size limit.")
                payload = response.read(config.maximum_response_bytes + 1)
            if len(payload) > config.maximum_response_bytes:
                raise LivePayloadValidationError("FRED response exceeds the configured size limit.")
            try:
                frame = normalize_h15_frame(pd.read_csv(StringIO(payload.decode("utf-8"))), config)
            except (UnicodeDecodeError, ValueError, pd.errors.ParserError) as exc:
                raise LivePayloadValidationError(
                    f"FRED payload failed schema or value controls: {exc}"
                ) from exc
            return frame, payload
        except LivePayloadValidationError:
            raise
        except (HTTPError, URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt + 1 < config.retries:
                time.sleep(min(2.0**attempt, 4.0))
    assert last_error is not None
    raise LiveSourceUnavailableError(
        f"Live FRED download failed: {type(last_error).__name__}: {last_error}"
    ) from last_error


def _load_source(
    config: DataConfig,
    project_root: Path | None,
) -> tuple[pd.DataFrame, bytes, str, Exception | None, str | None]:
    if config.source_mode == "snapshot":
        frame, payload, acquired_at = _load_snapshot(config, project_root)
        return frame, payload, "SNAPSHOT", None, acquired_at

    if config.source_mode == "live_then_snapshot":
        cached = _read_fresh_cache(config, project_root)
        if cached is not None:
            frame, payload, acquired_at = cached
            return frame, payload, "LOCAL_CACHE", None, acquired_at

    try:
        frame, payload = _download_live(config)
        acquired_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
        canonical = _canonical_csv_bytes(frame)
        weekly, changes, gaps = synchronize_weekly_curve(frame, config)
        live_quality = _quality_checks(frame, weekly, changes, gaps, config, "LIVE_FRED")
        failed = [check for check in live_quality if check.status == "FAIL"]
        if failed:
            details = "; ".join(f"{check.control}: {check.result}" for check in failed)
            raise ValueError(f"Live FRED response failed cache-admission controls: {details}")
        try:
            _atomic_cache_write(
                canonical,
                config,
                project_root,
                source_payload=payload,
            )
            return frame, payload, "LIVE_FRED", None, acquired_at
        except (OSError, TimeoutError) as cache_error:
            return frame, payload, "LIVE_FRED_NO_CACHE", cache_error, acquired_at
    except LiveSourceUnavailableError as network_error:
        if config.source_mode == "live":
            raise
        frame, payload, acquired_at = _load_snapshot(config, project_root)
        return frame, payload, "SNAPSHOT_FALLBACK", network_error, acquired_at


def synchronize_weekly_curve(
    daily_yields_pct: pd.DataFrame,
    config: DataConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Select the last same-day complete curve in each Friday-ending week."""

    core = daily_yields_pct.loc[:, list(config.core_tenors)].dropna(how="any")
    weekly = core.groupby(core.index.to_period(config.weekly_rule), sort=True).tail(1).copy()
    if config.boundary_week_policy == "drop" and not weekly.empty:
        requested_start = pd.Timestamp(config.start_date)
        requested_end = pd.Timestamp(config.end_date)
        first_period = weekly.index[0].to_period(config.weekly_rule)
        last_period = weekly.index[-1].to_period(config.weekly_rule)
        if requested_start > first_period.start_time.normalize():
            weekly = weekly.iloc[1:]
        if not weekly.empty and requested_end < last_period.end_time.normalize():
            weekly = weekly.iloc[:-1]
    weekly.index.name = "Observation date"
    if len(weekly) < 2:
        raise ValueError("At least two complete weekly curves are required.")
    changes_bp = weekly.diff().iloc[1:] * 100.0
    changes_bp.index.name = "Observation date"
    gaps = weekly.index.to_series().diff().dt.days.iloc[1:]
    gaps.name = "Calendar gap (days)"
    return weekly, changes_bp, gaps


def _quality_checks(
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    changes: pd.DataFrame,
    gaps: pd.Series,
    config: DataConfig,
    mode: str,
) -> tuple[QualityCheck, ...]:
    checks: list[QualityCheck] = []

    def add(
        control: str, condition: bool, result: str, threshold: str, risk: str, warn: bool = False
    ) -> None:
        status: Literal["PASS", "WARN", "FAIL"] = "PASS" if condition else ("WARN" if warn else "FAIL")
        checks.append(QualityCheck(control, status, result, threshold, risk))

    add(
        "Unique, ordered daily dates",
        daily.index.is_unique and daily.index.is_monotonic_increasing,
        f"duplicates={int(daily.index.duplicated().sum())}",
        "zero duplicates; ascending dates",
        "Duplicate or unordered dates corrupt returns and time-series splits.",
    )
    add(
        "Complete synchronized weekly panel",
        not weekly.isna().any().any() and not changes.isna().any().any(),
        f"weekly_nulls={int(weekly.isna().sum().sum())}; change_nulls={int(changes.isna().sum().sum())}",
        "zero",
        "Incomplete curves change the covariance universe through time.",
    )
    add(
        "Minimum estimation history",
        len(changes) >= config.minimum_complete_weeks,
        f"{len(changes):,} weekly changes",
        f">={config.minimum_complete_weeks:,}",
        "An undersized history makes eigenvectors and tail metrics unstable.",
    )
    complete_core = daily.loc[:, list(config.core_tenors)].dropna(how="any")
    start_lag = (complete_core.index.min() - pd.Timestamp(config.start_date)).days
    add(
        "Requested start-date coverage",
        0 <= start_lag <= config.maximum_start_coverage_gap_days,
        f"first_complete_daily_curve={complete_core.index.min().date()}; lag={start_lag} calendar days",
        f"0 to {config.maximum_start_coverage_gap_days} days",
        "A late-starting source can silently shorten the requested calibration history.",
    )
    end_lag, within_week_lag = _end_coverage_metrics(weekly, config)
    add(
        "Requested end-date coverage",
        0 <= end_lag <= config.maximum_end_coverage_gap_days,
        f"last={weekly.index.max().date()}; lag={end_lag} calendar days",
        f"0 to {config.maximum_end_coverage_gap_days} days",
        "A truncated cache can masquerade as a complete requested sample.",
    )
    add(
        "Final weekly observation coverage",
        0 <= within_week_lag <= config.maximum_within_week_observation_lag_days,
        f"last={weekly.index.max().date()}; within-week lag={within_week_lag} days",
        f"0 to {config.maximum_within_week_observation_lag_days} days",
        "A cache can reach the requested week yet omit its latest eligible observation.",
    )
    expected_start_period, expected_end_period = _expected_boundary_periods(config, complete_core)
    observed_start_period = weekly.index.min().to_period(config.weekly_rule)
    observed_end_period = weekly.index.max().to_period(config.weekly_rule)
    add(
        "Eligible weekly-period coverage",
        observed_start_period == expected_start_period and observed_end_period == expected_end_period,
        (
            f"observed={observed_start_period} to {observed_end_period}; "
            f"expected={expected_start_period} to {expected_end_period}"
        ),
        "first and final eligible weekly periods present",
        "A missing boundary period silently changes the calibration window.",
    )
    excessive_gaps = int((gaps > config.maximum_calendar_gap_days).sum())
    add(
        "Weekly observation gaps",
        excessive_gaps == 0,
        f"max={int(gaps.max())} days; excessive={excessive_gaps}",
        f"no gap > {config.maximum_calendar_gap_days} days",
        "Long gaps mix risk horizons and inflate observed changes.",
    )
    non_seven_day_gaps = int((gaps != 7).sum())
    add(
        "Nominal weekly-horizon consistency",
        non_seven_day_gaps == 0,
        f"non-7-day changes={non_seven_day_gaps} of {len(gaps)}; range={int(gaps.min())}-{int(gaps.max())} days",
        "review every non-7-day interval",
        "Holiday rolls and partial periods create slightly different risk horizons.",
        warn=True,
    )
    first_period = weekly.index.min().to_period(config.weekly_rule)
    last_period = weekly.index.max().to_period(config.weekly_rule)
    partial_start = pd.Timestamp(config.start_date) > first_period.start_time.normalize()
    partial_end = pd.Timestamp(config.end_date) < last_period.end_time.normalize()
    partial_boundaries = [
        label for label, partial in (("start", partial_start), ("end", partial_end)) if partial
    ]
    add(
        "Boundary-week completeness",
        not partial_boundaries,
        f"included partial boundaries={partial_boundaries or 'none'}; policy={config.boundary_week_policy}",
        "no partial boundary week in the estimation sample",
        "Partial boundary weeks have a shorter market-information horizon.",
        warn=True,
    )
    minimum_yield = float(np.nanmin(daily[list(ALL_TENORS)].to_numpy()))
    maximum_yield = float(np.nanmax(daily[list(ALL_TENORS)].to_numpy()))
    add(
        "Yield-domain reasonableness",
        minimum_yield > -5.0 and maximum_yield < 25.0,
        f"{minimum_yield:.2f}% to {maximum_yield:.2f}%",
        "(-5%, 25%)",
        "Wrong units or parsing errors can dominate covariance estimates.",
    )
    add(
        "Yield-unit scale",
        maximum_yield >= 0.25,
        f"maximum observed yield={maximum_yield:.4f}%",
        ">=0.25% for the requested U.S. Treasury sample",
        "Decimal yields misread as percentages shrink changes and risk by a factor of 100.",
    )
    zero_variance = changes.std(ddof=1).loc[lambda x: x <= np.finfo(float).eps].index.tolist()
    add(
        "Non-degenerate tenor changes",
        not zero_variance,
        f"zero-variance tenors={zero_variance or 'none'}",
        "none",
        "A constant tenor makes correlation PCA undefined.",
    )
    max_jump = float(changes.abs().max().max())
    add(
        "Weekly jump review",
        max_jump <= 250.0,
        f"maximum absolute move={max_jump:.1f} bp",
        "<=250 bp or independently reviewed",
        "Extreme values may be real stress events or source errors and require review.",
        warn=True,
    )
    exact_zero_count = int((daily[list(ALL_TENORS)] == 0.0).sum().sum())
    add(
        "Zero-bound observations",
        exact_zero_count == 0,
        f"exact zero yields={exact_zero_count}",
        "review rather than impute or delete",
        "A binding publication floor can alter covariance and forecast dynamics.",
        warn=True,
    )
    methodology_change = pd.Timestamp("2021-12-06")
    spans_methodology_change = daily.index.min() < methodology_change <= daily.index.max()
    if spans_methodology_change:
        checks.append(
            QualityCheck(
                "Treasury curve-methodology regime",
                "WARN",
                "sample spans the 2021-12-06 change to monotone-convex curve construction",
                "run a pre/post methodology sensitivity",
                "A source-methodology change can look like a model regime shift.",
            )
        )
    checks.append(
        QualityCheck(
            "Historical point-in-time vintage control",
            "WARN",
            (
                "single hash-verified frozen history; reproducible but not a historical vintage panel"
                if mode.startswith("SNAPSHOT")
                else "latest-revised source/cache; not a historical vintage panel"
            ),
            "as-of vintage archive at every historical decision date",
            "A single latest-revised/frozen history can introduce revision look-ahead in backtests.",
        )
    )
    new_york_today = pd.Timestamp(datetime.now(ZoneInfo("America/New_York")).date())
    staleness = (new_york_today - weekly.index.max()).days
    if config.maximum_staleness_days is not None:
        add(
            "Operational freshness",
            staleness <= config.maximum_staleness_days,
            f"{staleness} calendar days",
            f"<={config.maximum_staleness_days}",
            "Stale data make current-risk factors and forecasts decision-irrelevant.",
        )
    else:
        checks.append(
            QualityCheck(
                "Operational freshness",
                "WARN",
                f"{staleness} calendar days; no current-risk SLA configured",
                "configure an approved EOD staleness SLA for current use",
                "Without a freshness gate, results are historical research rather than current risk.",
            )
        )
    if mode == "SNAPSHOT_FALLBACK":
        checks.append(
            QualityCheck(
                "Live-source availability",
                "WARN",
                "live source failed; hash-verified snapshot used",
                "live source preferred for current production runs",
                "Results are reproducible but may not contain the latest market observations.",
            )
        )
    if mode == "LIVE_FRED_NO_CACHE":
        checks.append(
            QualityCheck(
                "Live cache persistence",
                "WARN",
                "validated live payload used, but controlled cache persistence failed",
                "persist canonical and exact upstream payloads with hashes",
                "The analytic result is usable with caveats, but operational replay is impaired.",
            )
        )
    return tuple(checks)


def load_curve_data(config: DataConfig, project_root: Path | None = None) -> CurveDataBundle:
    """Load, validate, synchronize, and hash the selected H.15 curve sample."""

    daily, source_bytes, mode, network_error, source_acquired_at = _load_source(config, project_root)
    weekly, changes, gaps = synchronize_weekly_curve(daily, config)
    quality = _quality_checks(daily, weekly, changes, gaps, config, mode)
    canonical_hash = sha256(_canonical_csv_bytes(daily)).hexdigest()
    provenance = DataProvenance(
        mode=mode,
        requested_start=config.start_date,
        requested_end=config.end_date,
        observed_start=str(daily.index.min().date()),
        observed_end=str(daily.index.max().date()),
        loaded_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        source_acquired_at_utc=source_acquired_at,
        source_as_of_date=str(daily.index.max().date()),
        content_sha256=canonical_hash,
        source_payload_sha256=sha256(source_bytes).hexdigest(),
        rows=len(daily),
        columns=daily.shape[1],
        network_error=(
            f"{type(network_error).__name__}: {network_error}"
            if network_error is not None and mode == "SNAPSHOT_FALLBACK"
            else None
        ),
        cache_persistence_error=(
            f"{type(network_error).__name__}: {network_error}"
            if network_error is not None and mode == "LIVE_FRED_NO_CACHE"
            else None
        ),
    )
    bundle = CurveDataBundle(daily, weekly, changes, gaps, provenance, quality, source_bytes)
    bundle.raise_for_failures()
    return bundle
