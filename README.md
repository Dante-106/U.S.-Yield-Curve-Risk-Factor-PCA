# U.S. Treasury Yield-Curve PCA

An auditable Python implementation of U.S. Treasury constant-maturity yield-curve PCA, walk-forward forecast challenge, and illustrative first-order key-rate risk diagnostics.

The repository-scope technical disposition for the bundled frozen sample is **SHARE WITH CAVEATS**. That is not organizational model validation, model approval, production authorization, or management acceptance of risk. The code fails closed around data integrity, time ordering, factor identification, forecast adoption, finite-sample VaR, and controlled output commit.

## Decision boundary

| Use | Repository disposition | Reason |
|---|---:|---|
| Reproducible PCA research and implementation challenge | Conditionally permitted | Use the frozen, hash-verified sample and retain every warning |
| Descriptive factor decomposition and reconstruction | Conditionally permitted | Factor identity is conditional; residual/key-rate risk remains visible |
| Illustrative fixed-KRD linear scenarios | Diagnostic only | Requires controlled metadata; statistical adequacy currently fails |
| Current desk risk, official limits, VaR/ES, P&L attribution, PLA/FRTB, capital | **Not eligible** | No current approved feed, official positions/P&L, nonlinear revaluation, or institutional controls |
| Autonomous forecast, hedge, trade, or management decision | **Not eligible** | No challenger clears adoption; interval evidence is inconclusive |

## Frozen-sample result

The deterministic snapshot was acquired on 2026-08-25, has a source-as-of date of 2025-12-31, and supports weekly levels from 2000-01-07 through 2025-12-26. It produces 1,355 synchronized weekly changes from 2000-01-14 through 2025-12-26. On the 2026-08-26 New York release-verification date it is 243 calendar days stale and is therefore historical research, not current risk.

| Control/result | Value | Interpretation |
|---|---:|---|
| PC1 / PC2 / PC3 variance share | 77.6249% / 14.0276% / 5.2131% | Top three explain 96.8656% in sample |
| Level / Slope / Curvature template cosine | 0.9477 / 0.9488 / 0.9320 | All clear the 0.70 similarity gate |
| Template dominance margin | 0.6791 / 0.6749 / 0.7781 | All clear the 0.10 ambiguity gate |
| Top-3 in-sample MAE / RMSE | 1.3421 bp / 1.9882 bp | Local residual shocks are not eliminated |
| Worst tenor expanding-OOS RMSE | 2.3783 bp | Clears the 5 bp review level |
| Worst retrospective rolling angle / cosine | 71.173° / 0.3935 | Material historical instability; no stable limit-factor claim |
| Worst sequential adjacent-window angle / cosine | 73.903° / 0.2833 | Chronologically separated regime instability is material |
| Latest sequential angle / cosine | 9.271° / 0.9339 | Recent comparison is better, but does not erase the historical worst |
| Current-EWMA vs structural max angle / sigma ratio | 5.598° / 1.2267 symmetric max | Absolute shock scale monitored; both clear current 30°/1.25 levels |
| Forecast selected | No-change benchmark | Fallback benchmark; no challenger passes both adoption stages |
| Independent marginal interval statuses | 18/18 INCONCLUSIVE | Five-year challenge window cannot certify calibration |
| Overall technical disposition | SHARE WITH CAVEATS | Restricted diagnostic scope only |

The data QA layer reports five warnings: 76 non-seven-day weekly horizons (six to eight days), 121 exact-zero yield observations, the 2021-12-06 Treasury methodology change, lack of a point-in-time vintage panel, and operational staleness.

## What changed from the supplied notebook

The original notebook had a severe factor-sign inconsistency: presentation copies of components and scores were reoriented, while the fitted estimator used by `transform` and `inverse_transform` retained different signs. The mismatch reached 211.99 bp and a supposedly full-rank round trip missed by 143.33 bp. Version 3.0.0 uses one coherent eigendecomposition and applies economic orientation to the actual fitted state. Full-rank reconstruction now agrees to about `7.1e-14` bp.

The rewrite also adds:

- exact source-byte and canonical-content hashes, bounded gzip/HTTP/cache reads, strict schemas, synchronized complete curves, holiday-horizon flags, and freshness controls;
- covariance, correlation, and EWMA specifications with physical-unit mappings and effective-observation controls;
- similarity **and dominance** gates for Level/Slope/Curvature labels;
- retrospective rolling, chronologically separated adjacent-window, 2,000-replication block-bootstrap, 4/13/26-week block sensitivity, methodology-regime, and expanding-OOS reconstruction challenges;
- expanding-origin no-change, historical-mean, PCA AR(1), and PCA VAR(1) forecasts with disjoint selection, confirmation, and interval-evaluation periods;
- a one-sided Clark-West/HAC statistical gate, Bonferroni multiplicity control, and a 1% RMSE materiality hurdle;
- dependence-aware row-wise circular block-bootstrap interval evidence across all tenors and both nominal coverages; Wilson and exact-binomial outputs are descriptive IID diagnostics only;
- explicit positive-DV01 sign convention, direct factor/residual variance reconciliation, recent and full-history empirical tails, exact contemporaneous shocks, rolling fixed-KRD VaR diagnostics, and bounded hedge optimization;
- strict JSON configuration, atomic content-addressed outputs, SHA-256 plus structural artifact contracts, invocation-level lineage, deterministic notebook generation, and wheel/sdist verification.

Version 3.1.0 removes comments from source, scripts, tests, and committed notebook code cells. Explanatory intent now lives in descriptive names, explicit constants, focused control flow, docstrings, tests, and the model-governance documents. A tokenizer-based regression test prevents comments from re-entering executable code while preserving auditable API and use-limit documentation.

Version 3.2.0 closes a standalone-notebook portability defect. The prior Colab check ran beside the complete repository and therefore did not test a notebook-only upload. The release notebook now contains a deterministic minimal runtime with the exact package source, frozen snapshot, and source manifest. In a notebook-only environment it verifies the embedded archive hash, exact member contract, extraction boundaries, implementation hash, and snapshot hash before calculation. In a complete repository it binds to the local source and data instead. Missing Jinja2 now degrades table presentation to plain DataFrames without interrupting calculations.

Version 3.3.0 closes a second Colab portability defect. Version 3.2.0 rejected Python 3.13 before dependency or model checks because its notebook bootstrap hard-coded an upper runtime bound. The package and CI matrix now validate Python 3.10–3.13. The standalone notebook fails only below Python 3.10; a future runtime newer than the formally validated range is identified as guarded execution and must still pass dependency, payload, source, data, calculation, and output controls. This prevents a routine Colab runtime upgrade from creating an artificial first-cell failure while preserving an explicit validation boundary.

## Data and methodology

The source is the Federal Reserve H.15 Treasury constant-maturity series obtained through FRED and distributed as a frozen snapshot with a manifest. These are Treasury CMT par-yield proxies, not SOFR/OIS discount factors, zero-coupon yields, executable prices, or a desk curve. The [Federal Reserve H.15 release](https://www.federalreserve.gov/releases/h15/), [Treasury interest-rate statistics](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics), [Treasury curve methodology](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/treasury-yield-curve-methodology), and [FRED H.15 release page](https://fred.stlouisfed.org/release?rid=18) are the governing public references.

The run uses one frozen/latest-revised history. It is not a point-in-time vintage panel, cannot replay each historical decision information set, and cannot rule out revision look-ahead. Rolling stability uses the terminal full-sample basis and is retrospective. Sequential adjacent windows contain no observations after each monitoring-window end, but still share the same latest-revised vintage.

## Forecast controls

The default 835-observation holdout is partitioned without overlap:

- selection: 523 weeks, 2009-12-31 through 2020-01-03;
- confirmation: 52 weeks, 2020-01-10 through 2020-12-31;
- independent interval evaluation: 260 weeks, 2021-01-08 through 2025-12-26.

Selection RMSE is 8.0363 bp for no-change versus 8.0631, 8.0558, and 8.0533 bp for mean, PCA AR, and PCA VAR. PCA VAR improves confirmation RMSE by 2.2165%, but its one-sided Clark-West/HAC p-value is 0.06793 and it had already failed selection. It is not adopted. The next reported target period ends 2026-01-02, but the forecast is historical and not decision-current.

The independent interval window is a severe-miscalibration challenge, not calibration certification. The minimum marginal coverage is 68.4615% for an 80% band and 88.8462% for a 95% band; the worst gap is -11.5385 percentage points. All 18 dependence-aware familywise results are INCONCLUSIVE. Marginal-band joint-hit diagnostics are 41.1538% and 76.5385%; those values are not comparable to the marginal nominal levels and do not create simultaneous curve regions.

## Illustrative risk controls

Input DV01 means positive USD price gain for a 1 bp fall in yield, so linear scenario P&L is `-DV01 @ yield_shock_bp`. A KRD CLI input must include exact tenor coverage plus valuation date, USD, a 1 bp bump, sign convention, the self-attested `UST Treasury CMT H.15` curve identifier, and caller-supplied portfolio, position-snapshot, and sensitivity-engine identifiers. The program validates only identifier syntax and declared metadata; it cannot prove upstream position, curve, valuation, or sensitivity lineage. A KRD dated after the market data fails matched-scope use; an older KRD may differ by at most the configured seven calendar days.

For the notebook's fixed illustrative KRD only:

| Diagnostic | Recent 520 weeks | Full 1,355 weeks |
|---|---:|---:|
| 99% historical VaR | $23.400m | $24.225m |
| VaR finite-sample rank / exception bound | 516/520; 0.9597% | 1,343/1,355; 0.9587% |
| 97.5% fractional empirical ES | $22.781m | $25.276m |
| Effective ES tail mass | 13.0 (below 20) | 33.875 (review level met) |

The 520-week rolling VaR diagnostic has 835 observations, eight exceptions, a finite-sample expected count of 8.013, and Kupiec/independence/conditional-coverage p-values of 0.9962/0.6938/0.9254. Seven of eight exceptions occur in 2022–2025, so an aggregate PASS remains a low-power, regime-concentrated diagnostic. The full-history worst linear loss is $47.625m on 2001-11-16. Because both recent-window tail measures have fewer than 20 effective tail observations, CLI risk assessment is **NEEDS REVISION** and warning acceptance cannot return success for a KRD run.

For that illustrative KRD, structural full-covariance weekly modeled volatility is $9.069m and current-EWMA volatility is $9.496m, a 4.71% increase. This absolute comparison is shown alongside factor exposure; explained shares alone are not a current-risk measure.

None of these outputs is official VaR/ES, actual/hypothetical P&L backtesting, PLA, FRTB, capital, or a hedge recommendation. `±1/2/3σ` factor scenarios are historical-covariance-scaled shocks. Non-Gaussian and non-IID diagnostics reject probability or return-period interpretations, and a pure-factor shock is not asserted to be jointly plausible.

## Installation and execution

Supported and tested Python versions are 3.10–3.13.

For Colab, upload `notebooks/us_treasury_yield_curve_pca.ipynb` as a single file and select **Runtime → Run all**. No repository clone, project installation, or network access is required for the frozen default analysis. The notebook reports `standalone embedded`, its runtime-validation status, exact dependency versions, and both binding hashes before running the model. Python below 3.10 or missing required libraries fails with an explicit message. Python above the formally validated range is not silently certified: it is labeled `newer-than-validated Python runtime; guarded execution`, and every remaining integrity and analytical control must still pass.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,notebook]"
ruff check src tests scripts
pytest -q
```

Deterministic research run:

```bash
yield-curve-pca \
  --project-root . \
  --source-mode snapshot \
  --output-dir outputs
```

Because the frozen research result has caveats, a non-KRD run exits `3` unless an independently reviewed local ticket label is supplied:

```bash
yield-curve-pca \
  --project-root . \
  --source-mode snapshot \
  --output-dir outputs \
  --accept-warnings \
  --warning-approval-id MR-REVIEW-123
```

The label records an invocation; it does not prove authority or approve use. Exit codes are `0` accepted within repository technical controls, `2` execution/configuration failure, `3` unaccepted caveats, and `4` needs revision. A matching content-addressed directory is reused only **after the full calculation is rerun** and independently derived controls, artifact names, hashes, and structural contracts are reconciled. This intentionally avoids treating a mutable local manifest as a trust anchor.

For live operation, use `config/pipeline.example.json`, a current end date, the seven-day freshness gate, archived source bytes, and an approved scheduler/environment. `live_then_snapshot` falls back only when the source is unavailable; a malformed live payload fails closed.

## Repository map

- `src/yield_curve_pca/`: validated data, PCA, forecast, risk, reporting, pipeline, and CLI modules
- `notebooks/us_treasury_yield_curve_pca.ipynb`: generated, fully executed, hash-bound standalone/Colab review notebook
- `data/` and packaged `resources/`: byte-identical frozen snapshot and source manifest
- `config/pipeline.example.json`: controlled live-then-snapshot operating example
- `docs/MODEL_CARD.md`: assumptions, limits, monitoring, and governance boundary
- `docs/VALIDATION_REPORT.md`: non-organizational technical validation and effective challenge
- `docs/RELEASE_VERIFICATION.md`: final test, notebook, package, and isolated-install evidence
- `docs/OPERATING_RUNBOOK.md`: execution, escalation, retention, and recovery
- `docs/ORIGINAL_CODE_REVIEW.md`: supplied-notebook defect analysis and remediation map
- `tests/`: normal, boundary, failure, golden, and integration coverage

The release gate tokenizes every Python file and every notebook code cell, so the comment-free implementation contract is checked in local tests, GitHub Actions, and source-distribution verification. GitHub Actions also executes the notebook twice on every supported Python version: once against the complete repository and once from an isolated directory through the embedded runtime.

## Governance references

Model-risk governance should be mapped to the adopting institution's policies and applicable supervisory expectations, including [Federal Reserve SR 11-7 / OCC 2011-12](https://www.federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm). The 2026 agencies' rescission of that guidance does not turn this repository into an approved model; the organization must determine its current governing framework and retain independent effective challenge. See the [Federal Reserve notice](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm) and [OCC Bulletin 2026-13](https://www.occ.treas.gov/news-issuances/bulletins/2026/bulletin-2026-13.html). Regulatory market-risk use would additionally require controls beyond this project, consistent with the applicable [Basel market-risk framework](https://www.bis.org/basel_framework/chapter/MAR/10.htm).

Read the model card and validation report before relying on any output.

## Copyright and permitted use

Copyright © 2026 Dante Li. All rights reserved.

This is a not an open-source project. Use is limited to the terms stated in the LICENSE file. Third-party and public-domain data remain subject to their original terms and attribution requirements.
