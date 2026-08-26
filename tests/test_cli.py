import json
from dataclasses import replace

import pandas as pd
import pytest

from yield_curve_pca.cli import (
    _data_config_from_args,
    _decision_exit_code,
    _krd_metadata_from_args,
    _load_pipeline_config,
    _normalize_warning_approval_id,
    _parser,
    _validate_krd_metadata,
    _write_effective_config,
    _write_outputs,
    main,
)
from yield_curve_pca.config import PipelineConfig
from yield_curve_pca.pipeline import run_pipeline


def test_cli_outputs_are_atomic_and_krd_content_addressed(project_root, tmp_path):
    config = PipelineConfig()
    config = replace(
        config,
        pca=replace(
            config.pca,
            bootstrap_replications=800,
            bootstrap_sensitivity_replications=800,
        ),
        forecast=replace(config.forecast, enabled=False),
    )
    result = run_pipeline(config, project_root)
    first_krd = tmp_path / "first.csv"
    second_krd = tmp_path / "second.csv"
    base = pd.DataFrame(
        {
            "tenor": result.structural_pca.tenors,
            "dv01_usd_per_bp": range(1, len(result.structural_pca.tenors) + 1),
        }
    )
    base.to_csv(first_krd, index=False)
    changed = base.copy()
    changed.loc[0, "dv01_usd_per_bp"] = 999
    changed.to_csv(second_krd, index=False)
    krd_metadata = {
        "valuation_date": "2025-12-26",
        "currency": "USD",
        "sensitivity_curve_id": "UST Treasury CMT H.15",
        "parallel_bump_bp": 1.0,
        "dv01_sign_convention": "positive price gain for a 1 bp yield decline",
        "portfolio_id": "TEST-PORTFOLIO",
        "position_snapshot_id": "POSITION-SHA256-TEST",
        "sensitivity_engine_id": "RISK-ENGINE/V1",
    }

    first_run = _write_outputs(
        result,
        config,
        tmp_path / "outputs",
        first_krd,
        krd_metadata,
        "RISK-REVIEW-1",
    )
    repeated_run = _write_outputs(
        result,
        config,
        tmp_path / "outputs",
        first_krd,
        krd_metadata,
        "RISK-REVIEW-2",
    )
    second_run = _write_outputs(result, config, tmp_path / "outputs", second_krd, krd_metadata)

    assert repeated_run == first_run
    assert second_run != first_run
    ledger_rows = [
        json.loads(line)
        for line in (tmp_path / "outputs" / "execution_ledger.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(ledger_rows) == 3
    assert ledger_rows[1]["analytic_result_reused"] is True
    assert ledger_rows[0]["warning_approval_id"] == "RISK-REVIEW-1"
    assert ledger_rows[1]["warning_approval_id"] == "RISK-REVIEW-2"
    for run_dir in (first_run, second_run):
        manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
        assert manifest["risk_mapping"] == "completed"
        assert manifest["implementation_sha256"]
        assert manifest["environment"]["python"]
        assert manifest["artifact_sha256"]
        assert manifest["artifact_schema_version"] == 3
        assert set(manifest["artifact_contracts"]) == set(manifest["artifact_sha256"])
        assert manifest["risk_assessment"] == "NEEDS REVISION"
        assert not manifest["risk_controls"]["statistical_adequacy_for_risk_use"]
        assert set(manifest["risk_controls"]["prohibited_uses"]) == {
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
        }
        assert manifest["risk_controls"]["krd_metadata_lineage_status"] == "CALLER ATTESTED; NOT VERIFIED"
        assert "warning_approval_id" not in manifest
        assert (run_dir / "structural_factor_exposure.csv").is_file()
        assert (run_dir / "current_ewma_factor_exposure.csv").is_file()
        assert (run_dir / "source_payload.csv").is_file()
        assert (run_dir / "bootstrap_block_sensitivity.csv").is_file()
        assert (run_dir / "structural_current_ewma_comparison.csv").is_file()
        assert (run_dir / "key_rate_dv01_input.csv").read_bytes() in {
            first_krd.read_bytes(),
            second_krd.read_bytes(),
        }
        assert (run_dir / "historical_var_backtest_summary.csv").is_file()
        assert (run_dir / "historical_tail_risk_full_history.csv").is_file()
        assert (run_dir / "historical_scenarios_full_history.csv").is_file()

    first_manifest_path = first_run / "run_manifest.json"
    first_manifest = json.loads(first_manifest_path.read_text(encoding="utf-8"))
    first_manifest["data_provenance"]["observed_end"] = "1900-01-01"
    first_manifest_path.write_text(
        json.dumps(first_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="data_provenance"):
        _write_outputs(result, config, tmp_path / "outputs", first_krd, krd_metadata)
    first_manifest["data_provenance"]["observed_end"] = result.data.provenance.observed_end
    first_manifest_path.write_text(
        json.dumps(first_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    incomplete_manifest_path = second_run / "run_manifest.json"
    incomplete_manifest = json.loads(incomplete_manifest_path.read_text(encoding="utf-8"))
    incomplete_manifest["artifact_sha256"].pop("data_quality.csv")
    incomplete_manifest_path.write_text(
        json.dumps(incomplete_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="incomplete artifact schema"):
        _write_outputs(result, config, tmp_path / "outputs", second_krd, krd_metadata)

    (first_run / "data_quality.csv").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="SHA-256"):
        _write_outputs(result, config, tmp_path / "outputs", first_krd, krd_metadata)


def test_cli_resolves_current_live_and_historical_snapshot_defaults():
    live_args = _parser().parse_args(["--source-mode", "live"])
    live = _data_config_from_args(live_args, new_york_today="2026-08-25")
    assert live.end_date == "2026-08-25"
    assert live.maximum_staleness_days == 7

    snapshot_args = _parser().parse_args([])
    snapshot = _data_config_from_args(snapshot_args, new_york_today="2099-01-01")
    assert snapshot.end_date == "2025-12-31"
    assert snapshot.maximum_staleness_days is None

    historical_args = _parser().parse_args(["--source-mode", "live", "--end-date", "2020-12-31"])
    historical = _data_config_from_args(historical_args, new_york_today="2099-01-01")
    assert historical.end_date == "2020-12-31"
    assert historical.maximum_staleness_days is None


def test_krd_metadata_requires_the_controlled_one_bp_bump():
    args = _parser().parse_args(
        [
            "--krd-csv",
            "krd.csv",
            "--krd-as-of",
            "2025-12-26",
            "--krd-curve-id",
            "UST Treasury CMT H.15",
            "--krd-bump-bp",
            "2",
            "--portfolio-id",
            "TEST-PORTFOLIO",
            "--position-snapshot-id",
            "POSITION-SHA256-TEST",
            "--sensitivity-engine-id",
            "RISK-ENGINE/V1",
        ]
    )
    with pytest.raises(ValueError, match="controlled 1 bp"):
        _krd_metadata_from_args(args)


def test_krd_metadata_requires_and_preserves_all_lineage_identifiers():
    args = _parser().parse_args(
        [
            "--krd-csv",
            "krd.csv",
            "--krd-as-of",
            "2025-12-26",
            "--krd-curve-id",
            "UST Treasury CMT H.15",
            "--portfolio-id",
            "US-RATES-BOOK-001",
            "--position-snapshot-id",
            "POSITIONS:SHA256:ABC123",
            "--sensitivity-engine-id",
            "RISK-ENGINE/V3.1.0",
        ]
    )
    metadata = _krd_metadata_from_args(args)
    assert metadata is not None
    assert metadata["portfolio_id"] == "US-RATES-BOOK-001"
    assert metadata["position_snapshot_id"] == "POSITIONS:SHA256:ABC123"
    assert metadata["sensitivity_engine_id"] == "RISK-ENGINE/V3.1.0"
    assert _validate_krd_metadata(metadata) == metadata


@pytest.mark.parametrize(
    "omitted_flag",
    ("--portfolio-id", "--position-snapshot-id", "--sensitivity-engine-id"),
)
def test_krd_metadata_fails_closed_when_a_lineage_identifier_is_missing(omitted_flag):
    argv = [
        "--krd-csv",
        "krd.csv",
        "--krd-as-of",
        "2025-12-26",
        "--krd-curve-id",
        "UST Treasury CMT H.15",
        "--portfolio-id",
        "US-RATES-BOOK-001",
        "--position-snapshot-id",
        "POSITIONS-SHA256-ABC123",
        "--sensitivity-engine-id",
        "RISK-ENGINE/V3.1.0",
    ]
    flag_position = argv.index(omitted_flag)
    del argv[flag_position : flag_position + 2]
    with pytest.raises(ValueError, match=omitted_flag):
        _krd_metadata_from_args(_parser().parse_args(argv))


def test_direct_krd_metadata_validation_rejects_uncontrolled_lineage_identifier():
    metadata = {
        "valuation_date": "2025-12-26",
        "currency": "USD",
        "sensitivity_curve_id": "UST Treasury CMT H.15",
        "parallel_bump_bp": 1.0,
        "dv01_sign_convention": "positive price gain for a 1 bp yield decline",
        "portfolio_id": "US RATES BOOK",
        "position_snapshot_id": "POSITIONS-SHA256-ABC123",
        "sensitivity_engine_id": "RISK-ENGINE/V3.1.0",
    }
    with pytest.raises(ValueError, match="portfolio_id"):
        _validate_krd_metadata(metadata)


@pytest.mark.parametrize("non_string", (None, True, 1))
def test_direct_krd_metadata_validation_rejects_non_string_identifiers(non_string):
    metadata = {
        "valuation_date": "2025-12-26",
        "currency": "USD",
        "sensitivity_curve_id": "UST Treasury CMT H.15",
        "parallel_bump_bp": 1.0,
        "dv01_sign_convention": "positive price gain for a 1 bp yield decline",
        "portfolio_id": non_string,
        "position_snapshot_id": "POSITIONS-SHA256-ABC123",
        "sensitivity_engine_id": "RISK-ENGINE/V3.1.0",
    }
    with pytest.raises(ValueError, match="portfolio_id must be a string"):
        _validate_krd_metadata(metadata)


def test_direct_krd_metadata_validation_rejects_non_string_curve_identifier():
    metadata = {
        "valuation_date": "2025-12-26",
        "currency": "USD",
        "sensitivity_curve_id": 123,
        "parallel_bump_bp": 1.0,
        "dv01_sign_convention": "positive price gain for a 1 bp yield decline",
        "portfolio_id": "US-RATES-BOOK-001",
        "position_snapshot_id": "POSITIONS-SHA256-ABC123",
        "sensitivity_engine_id": "RISK-ENGINE/V3.1.0",
    }
    with pytest.raises(ValueError, match="sensitivity_curve_id must be a string"):
        _validate_krd_metadata(metadata)


def test_strict_json_config_loads_tuples_and_rejects_unknown_fields(tmp_path):
    valid = tmp_path / "valid.json"
    valid.write_text(
        json.dumps(
            {
                "data": {"core_tenors": ["3M", "2Y", "10Y"]},
                "pca": {"bootstrap_sensitivity_blocks": [4, 13, 26]},
                "forecast": {"interval_bootstrap_block_lengths": [4, 13, 26]},
                "risk": {"factor_sigma_multiples": [1.0, 2.0]},
            }
        ),
        encoding="utf-8",
    )
    config = _load_pipeline_config(valid)
    assert config.data.core_tenors == ("3M", "2Y", "10Y")
    assert config.pca.bootstrap_sensitivity_blocks == (4, 13, 26)
    assert config.forecast.interval_bootstrap_block_lengths == (4, 13, 26)
    assert config.risk.factor_sigma_multiples == (1.0, 2.0)

    invalid = tmp_path / "invalid.json"
    invalid.write_text('{"pca": {"mystery_knob": 1}}', encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown fields"):
        _load_pipeline_config(invalid)


def test_effective_config_round_trips_and_duplicate_keys_fail(tmp_path):
    effective = tmp_path / "effective.json"
    original = PipelineConfig()
    _write_effective_config(original, effective)
    assert _load_pipeline_config(effective) == original

    for payload in (
        '{"data": {}, "data": {}}',
        '{"data": {"source_mode": "snapshot", "source_mode": "live"}}',
    ):
        duplicate = tmp_path / "duplicate.json"
        duplicate.write_text(payload, encoding="utf-8")
        with pytest.raises(ValueError, match="Duplicate JSON"):
            _load_pipeline_config(duplicate)


def test_warning_approval_identifier_is_normalized_and_bounded():
    assert _normalize_warning_approval_id("  RISK-123  ") == "RISK-123"
    for invalid in ("   ", "bad approval", "x" * 129):
        with pytest.raises(ValueError, match="warning-approval-id"):
            _normalize_warning_approval_id(invalid)


def test_cli_records_failed_execution_attempt(tmp_path):
    output_dir = tmp_path / "failed-output"
    assert (
        main(
            [
                "--accept-warnings",
                "--warning-approval-id",
                "   ",
                "--output-dir",
                str(output_dir),
            ]
        )
        == 2
    )
    event = json.loads((output_dir / "execution_ledger.jsonl").read_text(encoding="utf-8").strip())
    assert event["execution_status"] == "FAILED"
    assert event["failure_stage"] == "argument validation"


def test_cli_records_argument_parsing_failure(tmp_path):
    output_dir = tmp_path / "parse-failure"
    assert main(["--output-dir", str(output_dir), "--bogus-option"]) == 2
    event = json.loads((output_dir / "execution_ledger.jsonl").read_text(encoding="utf-8").strip())
    assert event["execution_status"] == "FAILED"
    assert event["failure_stage"] == "argument parsing"
    assert event["error_type"] == "ValueError"


@pytest.mark.parametrize(
    ("validation", "risk", "accepted", "expected"),
    [
        ("READY TO SHARE", "NOT REQUESTED", False, 0),
        ("SHARE WITH CAVEATS", "NOT REQUESTED", False, 3),
        ("READY TO SHARE", "SHARE WITH CAVEATS", False, 3),
        ("SHARE WITH CAVEATS", "SHARE WITH CAVEATS", True, 0),
        ("NEEDS REVISION", "NOT REQUESTED", True, 4),
        ("READY TO SHARE", "NEEDS REVISION", True, 4),
    ],
)
def test_cli_decision_exit_contract(validation, risk, accepted, expected):
    assert _decision_exit_code(validation, risk, warnings_accepted=accepted) == expected


@pytest.mark.parametrize(
    ("validation", "risk"),
    [("TYPO", "NOT REQUESTED"), ("READY TO SHARE", "TYPO"), ("", "")],
)
def test_cli_decision_exit_contract_fails_closed_on_unknown_status(validation, risk):
    with pytest.raises(ValueError, match="Unknown"):
        _decision_exit_code(validation, risk, warnings_accepted=True)
