# Operating Runbook

## Scope

This runbook supports controlled research and diagnostic execution. It is not an authorization for official risk, limits, P&L, VaR/ES, FRTB, capital, forecasting, hedging, or trading. The default frozen output is stale and the default illustrative KRD evidence is statistically inadequate for risk use.

## Pre-run controls

Record or verify before every scheduled invocation:

1. model inventory ID, approved use, environment, operator, reviewer, ticket, and expiry;
2. New York business date, source requested end, maximum-staleness SLA, and expected publication calendar;
3. package/release signature, Python 3.10–3.13, dependency/SBOM policy, and immutable configuration;
4. available disk capacity, output quota, retention window, backup/restore status, and monitoring destination;
5. for KRD: portfolio, legal entity, desk, valuation timestamp/timezone, official position and sensitivity-engine lineage, curve ID, 1 bp bump, USD units, sign, and P&L reconciliation status.

Do not place credentials, positions, client data, limits, or proprietary curves in the repository.

## Install and verify

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,notebook]"
python -m pip check
ruff check src tests scripts
pytest -q
```

For a release candidate, also execute the notebook, validate its committed outputs, build wheel and sdist, run tests from an extracted sdist, and smoke-test both artifacts in new virtual environments. Do not distribute a pre-existing `dist/` artifact without rebuilding it from the frozen source tree.

For a notebook-only Colab run, upload `notebooks/us_treasury_yield_curve_pca.ipynb` and select **Runtime → Run all**. Do not clone an arbitrary branch or add an unreviewed download cell. Confirm that the first outputs state `standalone embedded`, show `validated Python runtime` for Python 3.10–3.13, list the exact dependency versions, and show matching implementation and compressed-snapshot hashes. A repository run must instead state `repository`. Python below 3.10, or any hash, member-contract, extraction-boundary, dependency, calculation, or output failure is a hard stop. A future Python version labeled `newer-than-validated Python runtime; guarded execution` is not formally certified and requires the full release suite before distribution, although the notebook may continue through its remaining controls.

## Deterministic research execution

```bash
yield-curve-pca \
  --project-root . \
  --source-mode snapshot \
  --output-dir outputs
```

Expected repository technical disposition is `SHARE WITH CAVEATS`; without accepted caveats the exit code is `3`. If an authorized reviewer has approved the **restricted diagnostic execution**, record the local ticket label:

```bash
yield-curve-pca \
  --project-root . \
  --source-mode snapshot \
  --output-dir outputs \
  --accept-warnings \
  --warning-approval-id MR-REVIEW-123
```

The identifier is format-checked only. It does not verify the reviewer, entitlement, ticket state, scope, due date, or expiry and does not approve model or risk use.

## Live operation

Review `config/pipeline.example.json`, then run:

```bash
yield-curve-pca \
  --project-root . \
  --config-json config/pipeline.example.json \
  --output-dir outputs
```

The example uses `live_then_snapshot` and a seven-calendar-day freshness SLA. A fresh validated cache may be used. Otherwise the process calls FRED; it falls back to the frozen snapshot only for bounded source unavailability. An HTTP response with bad schema, units, values, coverage, or integrity fails closed. If fallback data is stale, the freshness control fails and the run returns failure rather than presenting stale data as current.

For a current scheduled run, archive the exact upstream bytes and manifest, pin the requested end date/configuration, reconcile published market-data status, and route FAIL/WARN to the named owner. Do not silently retry with another source or impute a curve.

## Controlled KRD execution

The CSV must contain exactly `tenor,dv01_usd_per_bp`, one row for every model tenor, and no duplicates/non-finite values. DV01 is positive price gain for a 1 bp yield decline.

```bash
yield-curve-pca \
  --project-root . \
  --source-mode snapshot \
  --krd-csv controlled_krd.csv \
  --krd-as-of 2025-12-26 \
  --krd-curve-id "UST Treasury CMT H.15" \
  --krd-currency USD \
  --krd-bump-bp 1 \
  --portfolio-id US-RATES-BOOK-001 \
  --position-snapshot-id POSITIONS-SHA256-EXAMPLE \
  --sensitivity-engine-id RISK-ENGINE/VERSION-EXAMPLE \
  --output-dir outputs
```

All KRD metadata, including the curve, portfolio, position-snapshot, and sensitivity-engine identifiers, are caller attestations. The program checks required presence, syntax, date, declared currency/bump/sign, and the configured curve string; it does not authenticate an upstream system or prove sensitivity lineage. A KRD dated after market data, more than seven days before it, on another declared curve, with a non-1 bp bump, or with wrong declared units/sign fails scope controls. Even with matched metadata, risk assessment requires rolling VaR PASS and every recent-window VaR/ES effective tail mass to meet 20. The bundled illustrative KRD fails the tail-mass condition and is `NEEDS REVISION`, so warning acceptance cannot produce exit 0.

Required external evidence for any official P&L explanation includes position-system extract/hash, sensitivity-engine/version/hash, valuation curve/vintage, market and position cutoffs, official and hypothetical P&L, reconciliation tolerance, unresolved breaks, approver, and retention location.

## Exit-code contract

| Code | Meaning | Required action |
|---:|---|---|
| 0 | Repository controls accepted for the declared restricted scope | Still enforce organizational use approval |
| 2 | Argument, configuration, data, calculation, integrity, or output failure | Stop; investigate failure event and logs |
| 3 | SHARE WITH CAVEATS without accepted review label | Stop automated downstream consumption; review caveats |
| 4 | NEEDS REVISION | Hard stop for decision use; outputs are diagnostic only |

Unknown assessment strings fail closed.

## Output and lineage contract

Each analytic result is addressed by SHA-256 over package version, implementation source, Python/dependency environment, validated configuration, stable data provenance, and KRD bytes/metadata. The committed directory is atomic and contains a manifest with a fixed artifact-name set, SHA-256 for every artifact, and schema-v3 structural contracts (bytes, rows, columns, and observed scalar types for CSV; top-level structure for JSON).

If the same run ID already exists, the pipeline is deliberately recalculated. Newly derived controls and expected artifact names are compared with the existing manifest; every file hash and structural contract is read back. Only then is the existing directory reported as reused. This costs calculation time but avoids trusting a mutable local manifest as its own validation authority.

Each invocation appends a success or failure JSONL event containing the analytic ID, reuse flag, review label, source mode/acquisition/hash, live fallback error, cache-persistence error, and current non-PASS data-quality snapshot. POSIX writers use an advisory lock, exact write loop, and `fsync`. The ledger is local and mutable—not signed, hash-chained, WORM, or the enterprise audit record. Forward events to approved immutable centralized storage.

## Required review sequence

1. Confirm market-data as-of, staleness, vintage status, exact hashes, row counts, missingness, horizon warnings, zeros and jump checks.
2. Confirm factor labels are `IDENTIFIED`; review both similarity and dominance margin.
3. Review top-three compression with tenor residuals, in-sample maximum error, expanding-OOS errors, and complete KRD limits.
4. Review historical-worst and latest rolling/sequential stability separately. Never let a recent PASS erase a historical breach.
5. Review 4/13/26-week bootstrap sensitivity and 2021 methodology sensitivity. Do not give sigma shocks a probability meaning.
6. Review forecast selection, confirmation and independent interval periods. `No-change benchmark` means no demonstrated edge; 18 INCONCLUSIVE bands mean no interval authorization.
7. For KRD, review signed as-of gap, self-attested curve field, official lineage, structural/EWMA comparison, residual variance, recent/full tails, tail mass, exception dates, and full-history/hypothetical stresses.
8. Record hard stop, action owner, issue ID, due date, expiry, and approval authority.

## Monitoring and escalation

Default review levels are examples until calibrated to institutional risk appetite:

- data FAIL or stale current run: stop use and notify data/operations owner;
- factor label not identified: prohibit economic-factor naming and factor limits;
- rolling/sequential maximum angle above 30° or cosine below 0.80: suspend stable-factor claims and review recalibration/limits;
- worst tenor OOS reconstruction RMSE above 5 bp: expand factor/residual/key-rate controls;
- methodology sigma ratio above 1.25: review source-method transition and calibration;
- forecast challenger gate failure: retain no-change and take no forecast action;
- interval WARN or INCONCLUSIVE: prohibit production bands; investigate regime/dependence and power;
- VaR coverage WARN/NOT EVALUATED or any recent tail mass below 20: risk assessment NEEDS REVISION;
- official-P&L/KRD/curve/as-of break: stop risk use and reconcile upstream lineage.

## Cache, concurrency, and recovery

POSIX uses crash-released `flock`; Windows uses crash-released byte-range locking. An exotic-platform exclusive-file fallback is fail-closed and may require manual removal after a crashed owner. Before deleting such a lock, verify no writer process is active, preserve the lock and cache metadata for incident evidence, and obtain operations approval. Never delete a broad cache/output path to clear one lock.

A validated live payload whose cache commit fails is used only as `LIVE_FRED_NO_CACHE` with an explicit warning and invocation event. Investigate disk, permissions, quota, and atomic-rename support before the next run.

On interruption, a hidden staging directory may remain. Confirm no active process owns it; compare its prefix/run ID, timestamps, and files with the ledger; preserve incident evidence; then remove only that exact staging directory under the retention policy. Never use recursive deletion against the repository, workspace root, home directory, or unresolved variable.

## Storage and retention

The repository does not auto-delete committed results or ledger events. The deployment owner must monitor free space and inode consumption, set quotas and alerts, define retention by result class, archive source/config/code/environment/manifest/artifacts/ledger together, and test restore. Retention deletion must be ticketed, approved, exact-targeted, and consistent with legal/regulatory holds. Keep immutable copies before local cleanup.

## Release checklist

- source, tests, config example, model card, validation report, runbook, notebook builder, executed notebook, sdist, and wheel describe the same version and defaults;
- all tests and lint pass on Python 3.10–3.13;
- notebook generator sync passes; repository and isolated single-file executions both pass; every code cell has an execution count/output; no error output or open figure remains; embedded archive, source, and data hashes reconcile;
- extracted sdist tests pass; isolated wheel and sdist installs pass `pip check` and snapshot smoke;
- no stale `dist/`, build, cache, output, credential, position, or proprietary-data artifact is included;
- release hash, SBOM/dependency report, signature/provenance attestation, approvers, issue exceptions, and rollback plan are recorded externally.
