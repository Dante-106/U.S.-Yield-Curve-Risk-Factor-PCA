# Technical Validation and Effective-Challenge Report (Non-Organizational)

## Executive conclusion

Version 3.3.0 is numerically coherent and reproducible for its declared frozen-history PCA research and descriptive first-order risk scope. The repository-scope technical disposition is **SHARE WITH CAVEATS**. Proposed KRD use on the default evidence is **NEEDS REVISION** because recent VaR and ES effective tail masses are below the configured review level. Current risk, official limits, official VaR/ES, PLA/FRTB, capital, autonomous forecasting, hedging, and trading are **NOT ELIGIBLE**.

The most decision-relevant result is negative: the code now works as specified, but the evidence does not support production action. Historical factor instability is large, no forecast challenger passes adoption, interval calibration evidence is inconclusive, and the frozen data is stale.

## Validation scope and independence

The review challenged source integrity, sample construction, PCA algebra and units, label identification, temporal leakage, stability, resampling uncertainty, forecast selection, interval evidence, KRD mapping, variance reconciliation, historical tails, VaR coverage, hedge controls, CLI fail-closed behavior, artifacts, packaging, and notebook reproducibility.

Three separate role-based lenses were applied:

- Fixed Income Risk Manager: decision relevance, key-rate residuals, scenarios, limits, P&L and hedge boundaries;
- Model Validation Manager: conceptual soundness, implementation verification, outcomes analysis, assumptions, governance and change control;
- CEO/CRO: currentness, materiality, stop-use criteria, accountability and whether an output can safely drive a decision.

The repository's alternate algebra/SVD route is not organizationally independent validation. Institutional validation remains external.

## Data challenge

| Control | Observed | Conclusion |
|---|---:|---|
| Daily rows / source tenors | 6,783 / 11 | Exact manifest and resource hashes reconcile |
| Weekly levels / changes | 1,356 / 1,355 | Baseline 2000-01-07; changes 2000-01-14–2025-12-26 |
| Complete PCA tenors | 9 | Exact synchronized same-day curve rows |
| Non-seven-day changes | 76; range 6–8 days | WARN; weekly horizon is not perfectly constant |
| Exact-zero yields | 121 | WARN; publication-floor dynamics may affect covariance |
| Treasury methodology change | 2021-12-06 | WARN; pre/post source method and market regime are confounded |
| Historical vintages | one frozen/latest-revised history | WARN; no PIT decision replay or revision-free backtest |
| Freshness on 2026-08-26 NY | 243 days | WARN; not current risk or forecast data |

The snapshot mode verifies exact compressed and uncompressed hashes, source IDs, schema, units, rows, columns, and dates. Gzip and live responses are bounded. Cache admission repeats data QA and stores exact upstream plus canonical bytes. Live mode bypasses cache; `live_then_snapshot` falls back only for genuine source unavailability, while a malformed payload fails closed.

## PCA implementation verification

The original notebook's orientation bug is closed. A single coherent symmetric eigendecomposition owns components, scores, transforms, inverse transforms, and physical shock mappings. Full-rank reconstruction error is `7.1e-14` bp rather than the original 143.33 bp failure; original displayed/estimator transform mismatch reached 211.99 bp.

| Result | Value | Review |
|---|---:|---|
| PC1 variance | 77.6249% | Level, conditionally identified |
| PC2 variance | 14.0276% | Slope, conditionally identified |
| PC3 variance | 5.2131% | Curvature, conditionally identified |
| Top-three cumulative | 96.8656% | Strong in-sample compression |
| Template cosine | 0.9477 / 0.9488 / 0.9320 | Above 0.70 |
| Dominance margin | 0.6791 / 0.6749 / 0.7781 | Above 0.10 |
| Top-three MAE / RMSE | 1.3421 / 1.9882 bp | Residual/local shocks remain |
| Top-three p95 / maximum absolute error | 3.9039 / 22.7134 bp | Explicit residual stress required |
| Worst tenor expanding-OOS RMSE | 2.3783 bp | PASS versus 5 bp review level |

Correlation PCA is a specification challenge with explicit standardized-score versus physical-bp units. Rolling stability now inherits the reference covariance/correlation specification and rejects cross-specification comparison. EWMA weights require a unique ascending time index and at least 100 effective observations.

## Stability and bootstrap challenge

| Monitor | Worst | Latest | Review level | Status |
|---|---:|---:|---:|---|
| Retrospective rolling maximum subspace angle | 71.173° | n/a | ≤30° | WARN |
| Retrospective rolling minimum aligned cosine | 0.3935 | n/a | ≥0.80 | WARN |
| Sequential adjacent-window maximum angle | 73.903° | 9.271° | ≤30° | WARN historically |
| Sequential adjacent-window minimum cosine | 0.2833 | 0.9339 | ≥0.80 | WARN historically |

Rolling results compare with the terminal full-sample reference and are retrospective, not historical alerts. Sequential windows are chronologically separated with no future rows but use one latest-revised vintage, not PIT archives.

The primary 13-week circular block bootstrap uses 2,000 replications, deterministic seed 20260825, and 50 expected order statistics in each 2.5% tail. Its median/97.5% maximum angle is 3.0202°/7.9689°, and top-three variance 2.5%/median/97.5% is 96.3121%/96.9231%/97.3778%. Loading-cosine lower tails are 0.9958/0.9750/0.9717 for Level/Slope/Curvature.

| Block weeks | Replications | Angle median | Angle 97.5% | Top-three variance 2.5% |
|---:|---:|---:|---:|---:|
| 4 | 2,000 | 2.6453° | 6.8342° | 96.4427% |
| 13 | 2,000 | 3.0705° | 7.9460° | 96.3125% |
| 26 | 2,000 | 3.3258° | 9.0982° | 96.1728% |

The bootstrap demonstrates conditional sampling precision around the full sample; it does not overcome the much larger historical regime instability.

Current-EWMA versus structural physical comparison gives factor one-sigma ratios of 1.0442/0.8152/0.8391 and ordered top-three angles of 1.1167°/4.5377°/5.5977°. The maximum symmetric sigma ratio is 1.2267, below the 1.25 current-factor review level. These absolute shock measures are more decision-relevant than explained shares alone.

## Source-methodology sensitivity

The pre/post 2021-12-06 comparison uses 1,143 pre and 211 post observations. Maximum ordered top-three subspace angle is 7.8752°, below the 15° review level. PC1 one-sigma norm rises from 28.3215 bp to 36.2093 bp, a ratio of 1.2785, above the 1.25 review level. Because source methodology and market regime change together, the result is a sensitivity warning, not causal attribution.

## Forecast validation

All models are refitted at every origin and satisfy `training_end < target_date`. The holdout is separated into selection, confirmation, and independent interval evaluation.

| Model | Selection RMSE | Improvement vs no-change | Selection CW one-sided p | Confirmation RMSE | Confirmation improvement | Confirmation CW p | Adopted |
|---|---:|---:|---:|---:|---:|---:|---:|
| No-change benchmark | 8.0363 bp | 0.0000% | n/a | 11.1074 bp | 0.0000% | n/a | Fallback |
| Historical mean | 8.0631 bp | -0.3337% | 0.4802 | 11.0212 bp | 0.7762% | 0.0912 | No |
| PCA AR(1) | 8.0558 bp | -0.2422% | 0.2330 | 11.0349 bp | 0.6524% | 0.0696 | No |
| PCA VAR(1) | 8.0533 bp | -0.2118% | 0.1321 | 10.8612 bp | 2.2165% | 0.0679 | No |

Selection uses 523 observations through 2020-01-03 and a one-sided Clark-West/HAC alpha of `0.05/3`. Confirmation uses 52 new observations through 2020-12-31 and alpha 0.05. PCA VAR's later RMSE improvement does not repair its failed selection and does not pass confirmation significance. No-change is a benchmark fallback, not model approval.

The independent 260-week interval evaluation runs from 2021-01-08 through 2025-12-26. Minimum observed coverage is 68.4615% for the 80% bands and 88.8462% for the 95% bands; worst gap is -11.5385 percentage points. All 18 row-wise dependence-aware, familywise block-bootstrap statuses are INCONCLUSIVE; worst lower-bound margin to the acceptable floor is -28.1538 percentage points. The 20,000-replication, 4/13/26-week block envelope has 27.78 expected observations in each adjusted tail. That is auditable but still tail-Monte-Carlo limited.

The 260-week window has low certification power. Under the more optimistic IID one-sided `alpha/18` approximation, true nominal coverage has roughly 4.9% power for the 80% band and 15.9% for the 95% band to clear a 3 percentage-point floor; roughly 2,469 and 917 IID observations would be needed for 80% power. Under the formal two-tail `alpha/(2*18)` convention, power is only about 2.34%/9.40% and the corresponding 80%-power sample sizes are roughly 2,788/1,030. Dependence reduces effective sample further. The window is therefore a severe-miscalibration challenge, not proof of normal calibration.

Marginal-band joint-hit rates are 41.1538% and 76.5385%. These are diagnostic only and are not simultaneous curve bands. Full-history residual coverage uses 731 prequential observations but is conditional/post-selection retrospective contra-evidence; its worst marginal gap is -6.6758 percentage points and it also remains INCONCLUSIVE.

## Illustrative KRD risk challenge

The notebook KRD is fixed and illustrative, not a position. Positive DV01 is gain for a 1 bp yield fall. Linear P&L, factor P&L plus mean plus residual P&L, and full covariance variance reconcile within numerical tolerance. Residual variance is computed directly from omitted full-rank contributions, avoiding subtractive cancellation.

| Tail diagnostic | Recent 520 weeks | Full 1,355 weeks | Adequacy |
|---|---:|---:|---|
| 99% VaR | $23.400m | $24.225m | Tail mass 5.2 / 13.55, both below 20 |
| VaR order-statistic rank | 516 | 1,343 | Finite-sample predictive convention |
| Strict exception bound | 0.9597% | 0.9587% | Target achievable |
| 97.5% fractional ES | $22.781m | $25.276m | Tail mass 13 below; 33.875 meets review level |

The rolling 520-week VaR diagnostic has 835 observations and eight exceptions versus 8.013 finite-sample expected. Kupiec, independence, and conditional-coverage p-values are 0.9962, 0.6938, and 0.9254. Seven exceptions fall in 2022–2025, so aggregate PASS is regime-concentrated and low power. Zero-exception, all-exception, clustered-exception, undersized-sample, and unattainable-confidence paths are independently tested.

The recent-window tail-mass condition fails. Accordingly, the CLI's KRD `risk_assessment` is `NEEDS REVISION`; `--accept-warnings` cannot convert it to exit 0. Curve, portfolio, position-snapshot, and sensitivity-engine identifiers remain caller-attested metadata. Their presence and syntax are controlled, but they do not prove upstream lineage.

For the illustrative KRD, structural full-covariance variance/weekly volatility is about USD 82.2535tn / USD 9.069m; current EWMA is USD 90.1697tn / USD 9.496m, a 4.71% volatility increase. The notebook and CLI report this absolute comparison in addition to factor exposures.

The full-history worst illustrative linear loss is $47.625m on 2001-11-16. Pure-factor sigma shocks are historical-covariance-scaled and have no Gaussian probability, return-period, or joint-plausibility interpretation.

## Version 3.1.0 refactor challenge

The humanization pass removed every Python comment token from package source, release scripts, tests, and committed notebook code cells. The implementation now conveys intent through descriptive constants, domain-specific variable names, smaller explicit expressions, docstrings, typed contracts, tests, and durable documentation. Tokenization tests enforce the exact scope and avoid unreliable text-pattern checks.

The refactor was challenged against version 3.0.0 with two independent full default pipeline executions. A recursively normalized serialization covered data provenance and QA, structural and EWMA PCA state, stability and bootstrap diagnostics, methodology and specification challenges, reconstruction, expanding-origin forecasts, interval evidence, validation, and structural/current illustrative KRD risk. Invocation-only timestamps and transient error fields were excluded. Both releases produced byte-identical serialized evidence with SHA-256 `8838bcbc07a60919726cffd668be589069630d07aa35e849799e95321e3e45d3`. The version and code-style changes therefore introduce no mathematical or financial output drift.

## Version 3.2.0 portability challenge

The original v3.1.0 Colab claim was invalid because its smoke test retained the repository `src/`, `data/`, and configured import path. A true notebook-only reproduction failed in the first code cell with `FileNotFoundError`. Version 3.2.0 embeds only the required package files and frozen resources in a deterministic ZIP payload. Notebook startup checks Python and required libraries, validates the payload SHA-256, exact ordered member contract, compressed and expanded size limits, and extraction containment, then verifies the extracted implementation and snapshot hashes before calculation. No remote code or data is downloaded.

Independent execution from an empty directory in a newly created Python environment with no installed `us-yield-curve-pca` package exercises the same 17 code cells and output controls. A second compatibility challenge deliberately omits Jinja2; table rendering falls back to DataFrame display while the full analytical run completes. Repository mode, standalone mode, notebook-source synchronization, and payload-tamper failure are permanent regression gates.

## Version 3.3.0 runtime-compatibility challenge

Version 3.2.0 imposed a bootstrap condition of Python 3.10–3.12 and raised `RuntimeError` before dependency, integrity, or analytical checks on Python 3.13. That static upper bound was unrelated to the model's actual capabilities and reproduced the reported Colab first-cell failure. Version 3.3.0 formally supports Python 3.10–3.13 in package metadata and CI. Its notebook bootstrap rejects only Python below 3.10; a later runtime beyond the validated range is visibly classified as guarded execution rather than silently certified or pre-emptively blocked.

The exact reported path was challenged with CPython 3.13.14 in a clean environment containing the scientific notebook dependencies but neither this project nor Jinja2. The notebook was launched from an isolated directory, selected its embedded runtime, executed 17/17 code cells, produced 59 outputs including five PNG figures, and emitted zero error outputs. The complete 152-test suite and Ruff checks independently passed on CPython 3.10.20, 3.11.15, 3.12.13, and 3.13.14. A regression test inspects the generated bootstrap and prohibits restoration of an upper-version hard rejection.

The v3.2.0 implementation on Python 3.12 and v3.3.0 implementation on Python 3.13 were separately run through the full default pipeline plus structural/current illustrative KRD risk mapping. After excluding invocation-only timestamps and transient acquisition errors, both produced the same 3,718,192-byte normalized serialization with SHA-256 `d74a201c74e1e93e4b95ffeafb985d3b2974cc1c31adce688e8c73a3f071204b`. The compatibility correction introduces no mathematical or financial output drift.

## Engineering and operational challenge

- strict immutable configuration with finite bounds and cross-field sample/power constraints;
- exact snapshot/resource equality and bounded decompression/network/cache reads;
- POSIX and Windows crash-released cache locks, atomic cache writes, and explicit no-cache warning;
- strict KRD and JSON schemas, duplicate-key rejection, metadata/as-of/sign/curve controls;
- deterministic implementation/environment/config/data/KRD identity;
- atomic run directories; fixed artifact-name set; SHA-256 and schema-v3 structural contracts;
- existing outputs are reused only after full recalculation derives controls independently and all files/hashes/contracts reconcile; the mutable manifest is not trusted as an external anchor;
- success/failure invocation ledger with fallback/cache/QA context, POSIX serialized exact writes and fsync;
- generated notebook binds embedded implementation and data hashes and must execute in both complete-repository and isolated single-file modes without errors or open figures;
- comment-token regression gates cover Python files and committed notebook code cells;
- CI covers Python 3.10–3.13, lint, tests, notebook sync, repository and standalone notebook execution, output validation, wheel/sdist build, extracted-sdist tests, and isolated install smoke under a 45-minute job timeout.

The local ledger is not signed, hash-chained, WORM, or entitlement-aware. Dependency bounds are not a lockfile or SBOM. Output retention, disk capacity, stale staging cleanup, backup/restore, observability, and disaster recovery remain deployment responsibilities.

## Role-based conclusions

**Fixed Income Risk Manager:** retain complete KRD and residual risk; do not use PC2/PC3 as stable limits; do not put current frozen output in an EOD pack; no trade or hedge action.

**Model Validation Manager:** mathematical implementation and no-leakage controls are strong for declared scope. PIT data, use-specific thresholds, independent institutional validation, outcome monitoring, official lineage, and governance are missing. Technical caveats are not approval.

**CEO/CRO:** the decision is `NO CURRENT RISK OR FORECAST ACTION`. Any management page must lead with market/KRD/position/P&L as-of, freshness, stop-use status, official P&L reconciliation, limit headroom, full-revaluation stress, tail adequacy, exception concentration, named owner, issue due date, and approval authority.

## Final disposition

The code has passed the repository-level notebook/package release checks and is suitable for controlled peer review, reproducible historical research, and diagnostic challenge. It is not eligible for current or official risk, limits, VaR/ES, capital, pricing, P&L, forecast action, hedge sizing, or trading. Any scope expansion requires fresh/PIT data, official lineage, nonlinear benchmarks, use-specific thresholds, institutional independent validation, formal approval, and operating controls. Exact release evidence is recorded in `docs/RELEASE_VERIFICATION.md`.
