import gzip
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd
import pytest

from yield_curve_pca.config import ALL_TENORS, CORE_TENORS, FRED_SERIES, DataConfig
from yield_curve_pca.data import (
    LivePayloadValidationError,
    LiveSourceUnavailableError,
    _atomic_cache_write,
    _cache_paths,
    _cache_write_lock,
    _download_live,
    _end_coverage_metrics,
    _load_source,
    _read_fresh_cache,
    load_curve_data,
    synchronize_weekly_curve,
)


def test_approved_snapshot_and_weekly_sample(curve_bundle):
    assert curve_bundle.provenance.mode == "SNAPSHOT"
    assert curve_bundle.daily_yields_pct.shape == (6783, 11)
    assert curve_bundle.weekly_changes_bp.shape == (1355, 9)
    assert curve_bundle.weekly_changes_bp.isna().sum().sum() == 0
    assert set(curve_bundle.quality_table["status"]) <= {"PASS", "WARN"}


def test_percent_change_to_basis_points_is_exact():
    dates = pd.to_datetime(["2024-01-05", "2024-01-12"])
    daily = pd.DataFrame(5.00, index=dates, columns=CORE_TENORS)
    daily.iloc[1] = 5.01
    config = DataConfig(
        start_date="2024-01-01",
        end_date="2024-01-31",
        minimum_complete_weeks=52,
        boundary_week_policy="include_and_flag",
    )
    _, changes, gaps = synchronize_weekly_curve(daily, config)
    assert changes.iloc[0].to_numpy() == pytest.approx([1.0] * len(CORE_TENORS))
    assert gaps.iloc[0] == 7


def test_snapshot_subrange_is_configurable(project_root):
    config = DataConfig(
        start_date="2010-01-01",
        end_date="2020-12-31",
        minimum_complete_weeks=260,
        boundary_week_policy="include_and_flag",
    )
    bundle = load_curve_data(config, project_root)
    assert bundle.weekly_changes_bp.index.min() >= pd.Timestamp("2010-01-01")
    assert bundle.weekly_yields_pct.index.max() == pd.Timestamp("2020-12-31")
    assert len(bundle.weekly_changes_bp) > 500


def test_request_beyond_snapshot_fails_loudly(project_root):
    config = DataConfig(end_date="2026-03-31", minimum_complete_weeks=260)
    with pytest.raises(ValueError, match="Requested end-date coverage"):
        load_curve_data(config, project_root)


def test_request_before_snapshot_fails_loudly(project_root):
    config = DataConfig(start_date="1990-01-01", minimum_complete_weeks=260)
    with pytest.raises(ValueError, match="Requested start-date coverage"):
        load_curve_data(config, project_root)


def test_corrupt_snapshot_hash_is_rejected(project_root, tmp_path):
    snapshot = tmp_path / "snapshot.csv.gz"
    manifest = tmp_path / "manifest.json"
    shutil.copyfile(project_root / "data/h15_treasury_cmt_2000_2025.csv.gz", snapshot)
    shutil.copyfile(project_root / "data/source_manifest.json", manifest)
    snapshot.write_bytes(snapshot.read_bytes()[:-32])
    config = DataConfig(snapshot_path=Path(snapshot), manifest_path=Path(manifest))
    with pytest.raises(ValueError, match="byte count|SHA-256"):
        load_curve_data(config)


def test_packaged_snapshot_supports_runs_outside_repository(tmp_path):
    bundle = load_curve_data(DataConfig(), tmp_path)
    assert bundle.provenance.mode == "SNAPSHOT"
    assert bundle.weekly_changes_bp.shape == (1355, 9)


def test_repository_and_packaged_snapshot_resources_are_identical(project_root):
    resource_root = project_root / "src/yield_curve_pca/resources"
    for name in ("h15_treasury_cmt_2000_2025.csv.gz", "source_manifest.json"):
        assert (project_root / "data" / name).read_bytes() == (resource_root / name).read_bytes()


def test_non_ascii_cache_hash_is_rejected_as_cache_miss(curve_bundle, tmp_path):
    config = DataConfig(cache_dir=tmp_path)
    csv_path, hash_path = _cache_paths(config, None)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    curve_bundle.daily_yields_pct.reset_index().to_csv(csv_path, index=False)
    hash_path.write_bytes(b"\xff\xfe\x00")
    assert _read_fresh_cache(config, None) is None


def test_oversized_cache_is_rejected_before_read(curve_bundle, tmp_path):
    config = DataConfig(cache_dir=tmp_path, maximum_response_bytes=1_000)
    csv_path, metadata_path = _cache_paths(config, None)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.write_bytes(b"x" * 1_001)
    metadata_path.write_text("{}\n", encoding="ascii")
    assert _read_fresh_cache(config, None) is None


def test_cache_round_trip_has_query_and_acquisition_metadata(curve_bundle, tmp_path):
    config = DataConfig(cache_dir=tmp_path)
    raw_bytes = curve_bundle.daily_yields_pct.to_csv(date_format="%Y-%m-%d").encode()
    source_payload = b"exact-raw-upstream-bytes"
    _atomic_cache_write(raw_bytes, config, None, source_payload=source_payload)
    cached = _read_fresh_cache(config, None)
    assert cached is not None
    cached_frame, cached_bytes, acquired_at = cached
    assert cached_bytes == source_payload
    assert cached_frame.equals(curve_bundle.daily_yields_pct)
    assert acquired_at.endswith("+00:00")
    _, metadata_path = _cache_paths(config, None)
    metadata = json.loads(metadata_path.read_text(encoding="ascii"))
    assert metadata["query"]["end_date"] == config.end_date
    assert metadata["canonical_sha256"] == hashlib.sha256(raw_bytes).hexdigest()
    assert metadata["source_payload_sha256"] == hashlib.sha256(source_payload).hexdigest()


def test_cache_lock_releases_after_owner_write_failure(tmp_path, monkeypatch):
    from yield_curve_pca import data as data_module

    real_write = data_module.os.write
    calls = 0

    def fail_once(descriptor, payload):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("simulated lock write failure")
        return real_write(descriptor, payload)

    monkeypatch.setattr(data_module.os, "write", fail_once)
    lock_path = tmp_path / "cache.lock"
    with pytest.raises(OSError, match="simulated"):
        with _cache_write_lock(lock_path):
            pass
    with _cache_write_lock(lock_path):
        pass


def test_live_mode_bypasses_cache_and_rejects_bad_payload_before_commit(curve_bundle, tmp_path, monkeypatch):
    from yield_curve_pca import data as data_module

    config = DataConfig(source_mode="live", cache_dir=tmp_path)

    def cache_must_not_be_read(*_args, **_kwargs):
        raise AssertionError("live mode must not read cache")

    monkeypatch.setattr(data_module, "_read_fresh_cache", cache_must_not_be_read)
    monkeypatch.setattr(
        data_module,
        "_download_live",
        lambda _config: (curve_bundle.daily_yields_pct, b"raw-upstream-payload"),
    )
    frame, payload, mode, error, acquired_at = _load_source(config, None)
    assert frame.equals(curve_bundle.daily_yields_pct)
    assert payload == b"raw-upstream-payload"
    assert mode == "LIVE_FRED"
    assert error is None
    assert acquired_at is not None

    bad_frame = curve_bundle.daily_yields_pct.copy()
    bad_frame.iloc[0, 0] = 99.0
    committed = False

    def record_commit(*_args, **_kwargs):
        nonlocal committed
        committed = True

    monkeypatch.setattr(data_module, "_download_live", lambda _config: (bad_frame, b"bad"))
    monkeypatch.setattr(data_module, "_atomic_cache_write", record_commit)
    with pytest.raises(ValueError, match="cache-admission"):
        _load_source(config, None)
    assert not committed

    decimal_frame = curve_bundle.daily_yields_pct / 100.0
    monkeypatch.setattr(
        data_module,
        "_download_live",
        lambda _config: (decimal_frame, b"decimal-scaled"),
    )
    with pytest.raises(ValueError, match="Yield-unit scale"):
        _load_source(config, None)


def test_live_cache_persistence_failure_is_an_explicit_warning(curve_bundle, tmp_path, monkeypatch):
    from yield_curve_pca import data as data_module

    config = DataConfig(source_mode="live", cache_dir=tmp_path)
    monkeypatch.setattr(
        data_module,
        "_download_live",
        lambda _config: (curve_bundle.daily_yields_pct, b"controlled-live-payload"),
    )
    monkeypatch.setattr(
        data_module,
        "_atomic_cache_write",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("disk unavailable")),
    )
    bundle = load_curve_data(config)
    assert bundle.provenance.mode == "LIVE_FRED_NO_CACHE"
    assert bundle.provenance.network_error is None
    assert "disk unavailable" in (bundle.provenance.cache_persistence_error or "")
    cache_control = bundle.quality_table.set_index("control").loc["Live cache persistence"]
    assert cache_control["status"] == "WARN"


def test_live_then_snapshot_only_falls_back_for_source_unavailability(project_root, monkeypatch):
    from yield_curve_pca import data as data_module

    config = DataConfig(source_mode="live_then_snapshot", use_cache=False)
    monkeypatch.setattr(
        data_module,
        "_download_live",
        lambda _config: (_ for _ in ()).throw(LiveSourceUnavailableError("offline")),
    )
    _, _, mode, network_error, _ = _load_source(config, project_root)
    assert mode == "SNAPSHOT_FALLBACK"
    assert isinstance(network_error, LiveSourceUnavailableError)

    monkeypatch.setattr(
        data_module,
        "_download_live",
        lambda _config: (_ for _ in ()).throw(LivePayloadValidationError("bad schema")),
    )
    with pytest.raises(LivePayloadValidationError, match="bad schema"):
        _load_source(config, project_root)


@pytest.mark.parametrize("declared_length", ["invalid", "999999999"])
def test_live_download_rejects_invalid_or_oversized_content_length(monkeypatch, declared_length):
    from yield_curve_pca import data as data_module

    class FakeResponse:
        headers = {"Content-Length": declared_length}

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self, _size):
            return b""

    monkeypatch.setattr(data_module, "urlopen", lambda *_args, **_kwargs: FakeResponse())
    with pytest.raises(LivePayloadValidationError):
        _download_live(DataConfig(source_mode="live", retries=1))


@pytest.mark.parametrize("date_column", ["Date", "observation_date", "DATE"])
def test_unapproved_non_numeric_yield_token_fails_loudly(project_root, date_column):
    raw = pd.read_csv(project_root / "data/h15_treasury_cmt_2000_2025.csv.gz")
    if date_column != "Date":
        raw = raw.rename(
            columns={
                "Date": date_column,
                **{tenor: series_id for series_id, tenor in FRED_SERIES},
            }
        )
        yield_column = "DGS3MO"
    else:
        yield_column = "3M"
    raw[yield_column] = raw[yield_column].astype(object)
    raw.loc[10, yield_column] = "oops"
    config = DataConfig(start_date="2000-01-01", end_date="2001-12-31", minimum_complete_weeks=52)
    from yield_curve_pca.data import normalize_h15_frame

    with pytest.raises(ValueError, match="unapproved non-numeric"):
        normalize_h15_frame(raw, config)


def test_mixed_fred_and_tenor_column_names_fail_loudly(project_root):
    raw = pd.read_csv(project_root / "data/h15_treasury_cmt_2000_2025.csv.gz")
    raw = raw.rename(columns={tenor: series_id for series_id, tenor in FRED_SERIES})
    raw["3M"] = raw["DGS3MO"]
    from yield_curve_pca.data import normalize_h15_frame

    with pytest.raises(ValueError, match="mixes FRED series IDs"):
        normalize_h15_frame(raw, DataConfig())


def test_joint_core_start_coverage_uses_requested_universe(project_root):
    config = DataConfig(core_tenors=ALL_TENORS)
    with pytest.raises(ValueError, match="Requested start-date coverage"):
        load_curve_data(config, project_root)


def test_drop_policy_accepts_first_full_period_after_midweek_start(project_root):
    config = DataConfig(
        start_date="2010-01-06",
        end_date="2012-12-31",
        minimum_complete_weeks=52,
        boundary_week_policy="drop",
    )
    bundle = load_curve_data(config, project_root)
    assert bundle.weekly_yields_pct.index.min() == pd.Timestamp("2010-01-15")


def test_within_week_end_coverage_detects_truncated_final_week():
    config = DataConfig()
    weekly = pd.DataFrame(index=pd.DatetimeIndex(["2025-12-24"]), columns=CORE_TENORS)
    end_lag, within_week_lag = _end_coverage_metrics(weekly, config)
    assert end_lag == 7
    assert within_week_lag == 2


def test_snapshot_decompression_is_bounded_before_csv_parse(tmp_path):
    snapshot = tmp_path / "bomb.csv.gz"
    manifest = tmp_path / "manifest.json"
    payload = b"x" * 2_000
    snapshot.write_bytes(gzip.compress(payload))
    manifest.write_text(
        json.dumps(
            {
                "compressed_bytes": snapshot.stat().st_size,
                "uncompressed_bytes": len(payload),
                "compressed_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
                "uncompressed_sha256": hashlib.sha256(payload).hexdigest(),
                "rows_excluding_header": 1,
                "units": "percent_per_annum",
                "series_ids": [series_id for series_id, _ in FRED_SERIES],
                "columns": 12,
            }
        ),
        encoding="utf-8",
    )
    config = DataConfig(
        snapshot_path=snapshot,
        manifest_path=manifest,
        maximum_response_bytes=1_000,
    )
    with pytest.raises(ValueError, match="size limit"):
        load_curve_data(config)
