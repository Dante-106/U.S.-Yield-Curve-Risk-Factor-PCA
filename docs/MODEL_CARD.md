# Model Card: U.S. Treasury Yield-Curve PCA 3.3.0

## Status and ownership boundary

**Repository technical disposition:** SHARE WITH CAVEATS.

**Inherent capability risk:** at least Tier 2 / Moderate because the package exposes risk and hedge APIs; any proposed limit, trading, hedge, VaR/ES, stress, management-forecast, or capital use is Tier 1 / High and currently **NOT ELIGIBLE**.

This card documents a controlled research implementation. It does not assign an institutional model ID, model owner, independent validator, business-use owner, risk classification, approval date, expiry, or next review date. Those fields must be completed in the adopting firm's model inventory before any controlled use.

## Purpose

The package:

1. constructs synchronized weekly U.S. Treasury CMT par-yield proxy curves;
2. estimates equal-weight covariance PCA, correlation-PCA challenge, and recency-weighted covariance PCA;
3. identifies the first three components conditionally as Level, Slope, and Curvature;
4. challenges algebra, reconstruction, sampling uncertainty, regime stability, methodology sensitivity, and chronological out-of-sample behavior;
5. evaluates simple one-week forecasting challengers against a no-change benchmark;
6. maps a controlled key-rate DV01 vector into first-order factor, residual, tail, and scenario diagnostics.

It is not a pricing model, a SOFR/OIS curve, an executable Treasury curve, a valuation engine, an official risk engine, or a trading strategy.

## Data

The source is Federal Reserve H.15 Treasury constant-maturity data obtained through FRED. Eleven source tenors are stored; the default PCA uses the complete synchronized nine-tenor panel `3M, 6M, 1Y, 2Y, 3Y, 5Y, 7Y, 10Y, 20Y` to avoid the shorter 30Y history and sparse 1M history.

The bundled artifact contains 6,783 daily rows, 1,356 weekly level curves, and 1,355 changes. The manifest binds exact compressed and uncompressed bytes, series IDs, units, row/column counts, acquisition time, and source-as-of date. Package resources and repository resources must be byte-identical.

The run uses one hash-verified frozen/latest-revised history. It is **not** a historical point-in-time vintage panel. It cannot reproduce the information set available on each historical decision date and cannot exclude revision look-ahead. Snapshot acquisition was 2026-08-25; the latest modeled weekly curve is 2025-12-26 and was 243 calendar days stale on the 2026-08-26 New York release-verification date.

## Mathematical specification

Let weekly yield changes in basis points be rows of (X\in\mathbb{R}^{n\times p}). Equal-weight covariance PCA centers (X), forms the sample covariance with denominator (n-1), and applies a symmetric eigendecomposition. Eigenpairs are sorted descending. The fitted object owns the oriented components, scores, transform, inverse transform, physical shock basis, and one-sigma shocks; presentation copies cannot diverge from the estimator.

Correlation PCA divides centered tenor changes by their sample standard deviations before eigendecomposition. Its model-space eigenvectors and standardized scores are distinguished from its physical basis in bp per score. EWMA PCA uses normalized positional weights, the unbiased weighted-covariance correction (1-\sum w_i^2), and reports effective observations (1/\sum w_i^2).

Economic labels use maturity-based Level/Slope/Curvature templates, maximum-total-similarity assignment, orientation, a minimum matched cosine of 0.70, and a minimum dominance margin over the next-best template of 0.10. A failure produces `UNIDENTIFIED`; labels are identification conventions, not causal economic factors.

## Primary assumptions, monitoring, and breach action

| Assumption | Monitor/evidence | Breach action | Uses not supported by the assumption |
|---|---|---|---|
| H.15 CMT par yields adequately proxy the target Treasury curve and bump | Source ID, exact tenor/unit schema, KRD self-attested curve ID | Stop; obtain desk curve mapping and independent sensitivity reconciliation | Pricing, OIS/SOFR, spread/basis or official same-curve risk |
| The complete nine-tenor panel is representative | Missingness, universe challenge, full residual/KRD reporting | Expand approved curve/key rates; retain residual limits | Portfolios concentrated in omitted tenors or instruments |
| Six-, seven-, and eight-day holiday horizons are acceptable for weekly research | Count every non-seven-day change; default 76/1,355 WARN | Review/exclude only under approved horizon policy; do not silently rescale | Daily horizon or exact-horizon capital claims |
| Linear covariance PCA captures decision-relevant second-order structure | Full-rank algebra, top-K residuals, OOS reconstruction, distribution diagnostics | Retain key-rate/local stresses; escalate nonlinear or tail model | Nonlinear, option, basis, volatility, liquidity or causal claims |
| Orthogonality in sample is not economic independence | Normality/IID tests and regime monitors | Do not map sigma to probability or assume independent factors | Return periods, joint scenario likelihood, Gaussian scaling |
| Template sign and label conventions identify factors without ambiguity | Similarity and dominance gates | Mark `UNIDENTIFIED`; prohibit factor-name limit use | Stable/causal factor taxonomy |
| Structural and EWMA covariance describe their own weighting regimes | Effective observations; structural/EWMA volatility comparison | Review limits and recalibrate under governed windows | Cross-regime stationarity or direct coordinate equality |
| The covariance structure is sufficiently stable | Retrospective rolling and sequential adjacent-window angle/cosine | Escalate; suspend factor-limit use; preserve KRD limits | Universal Level/Slope/Curvature limit taxonomy |
| The 2021 Treasury methodology change does not dominate conclusions | Pre/post comparison of physical loadings, subspace, and sigma | Review source-methodology mapping; do not infer causality | Unqualified regime-causation claims |
| Expanding-origin forecast and one-week Clark-West/HAC gate are appropriate | `training_end < target`; disjoint selection/confirmation; HAC sample bounds | Retain no-change benchmark; no forecast action | Forecast model adoption without both gates |
| The 52-week confirmation sample has enough power | Report p-values and sample; outcomes analysis | Treat non-passage as no evidence, not proof of equality | Claim that challengers can never add value |
| Residual empirical quantiles transfer under local stationarity/dependence assumptions | Independent 260-week interval window; block lengths 4/13/26; coverage status | INCONCLUSIVE/WARN; develop regime-conditional or validated alternatives | Calibrated production bands or simultaneous curve regions |
| KRD is first-order, fixed-position, correct-unit, same-curve, and correctly dated | Exact tenors; USD; 1 bp; sign; signed as-of gap; syntactically validated caller-supplied portfolio/position/engine IDs; external lineage/P&L reconciliation | `NEEDS REVISION`; stop official-risk use | Gamma, convexity, carry/roll, options, basis, spreads, volatility, liquidity, funding |
| Historical shocks are representative but not exhaustive | Recent and full-history scenarios/tails, tail mass, stress inventory | Add approved hypothetical/full-revaluation scenarios | Exhaustive stress, capital, future worst-case claims |
| One frozen/latest-revised vintage is adequate for research only | Vintage status warning and source hashes | Obtain point-in-time archive before historical decision replay | PIT backtest, historical alert performance, revision-free claims |

## Stability and uncertainty

The 260-week rolling analysis compares each window with the terminal full-sample reference and is explicitly retrospective. It is not a historical alert. Sequential analysis compares adjacent non-overlapping 260-week windows and uses no observation after each monitoring-window end; all windows still come from one latest-revised vintage and therefore are not a point-in-time replay.

Sampling uncertainty uses 2,000 circular moving-block replications at 13 weeks. The 2.5% and 97.5% tails each have 50 expected order statistics. A separate 2,000-replication sensitivity uses 4, 13, and 26-week blocks and independent deterministic seeds. Bootstrap inference is conditional on the empirical sample and resampling design; rolling/sequential evidence remains the regime-instability control. EWMA bootstrap is rejected because the implemented resampling design is equal-weight.

## Forecast specification and decision rule

Models are no-change, historical mean change, PCA-factor AR(1), and PCA-factor VAR(1). Every target is forecast using only prior rows. The default holdout has three non-overlapping decision periods: 523-week selection, 52-week confirmation, and 260-week independent interval evaluation.

A challenger can be adopted only if it:

1. improves selection RMSE by at least 1%;
2. has positive Clark-West adjusted advantage;
3. passes a one-sided HAC test at `0.05 / 3` in selection;
4. independently repeats the 1% improvement, positive adjusted advantage, and one-sided `p < 0.05` in confirmation.

No challenger passes. `No-change benchmark` is a fallback benchmark, not an approved forecasting model.

Intervals are marginal tenor bands from prequential selected-model residuals. The formal repository gate is a row-wise circular moving-block bootstrap that preserves cross-tenor dependence, applies a two-tail familywise adjustment across 18 tenor-band metrics, and takes the conservative envelope over 4/13/26-week blocks. Wilson bounds and the exact-binomial undercoverage p-value assume IID observations and are descriptive, unadjusted diagnostics only. The 260-week evaluation is intentionally an independent severe-miscalibration challenge; its power is insufficient to certify normal calibration. Full-history interval results are conditional/post-selection retrospective contra-evidence. Neither output is a simultaneous curve region.

## Risk specification

DV01 is positive USD price gain for a one-basis-point fall in yield. For shock vector (\Delta y_{bp}), first-order P&L is

\[
\text{P&L}_{USD}=-\text{DV01}_{USD/bp}^{\mathsf T}\Delta y_{bp}.
\]

Factor exposure is derived in physical bp-per-score space. Factor variance uses squared exposure times factor eigenvalue. Residual variance is computed directly from omitted full-rank factor contributions and must reconcile to physical covariance within absolute and relative tolerances.

Historical VaR uses the finite-sample next-observation order statistic with ascending rank `ceil((n+1) * confidence)`, capped at `n`. The reported strict-exceedance upper bound is `(n+1-rank)/(n+1)`; unattainable confidence targets are `NOT EVALUATED`. ES is the exact fractional empirical upper-tail average. Recent and full-history tails are both reported. The default minimum effective tail-mass review level is 20.

Rolling VaR coverage applies the same finite-sample threshold and reports Kupiec, Christoffersen independence, and conditional-coverage diagnostics. These tests do not turn fixed illustrative KRD/latest-revised linear P&L into actual-position, official-P&L, hypothetical-P&L, PLA, FRTB, or regulatory backtesting.

Hedge optimization minimizes full covariance risk, includes omitted-factor residual risk, rejects redundant/underdetermined or ill-conditioned hedge sets, enforces finite quantity bounds and maximum quantity, and requires separate positive modeled-risk floors for the pre-hedge portfolio and each candidate plus a finite minimum variance improvement. Zero-risk portfolios fail closed instead of returning an undefined reduction. It does not include price, bid/offer, liquidity, funding, inventory, lot size, carry/roll, nonlinear risk, or execution constraints and cannot autonomously recommend a trade.

## Validation evidence

- full-rank reconstruction maximum absolute error: approximately `7.1e-14` bp;
- transform and inverse transform share the oriented estimator state;
- covariance and correlation physical mappings reconcile;
- template ambiguity, shuffled dates, future mutation, partition overlap, undersized HAC, unattainable VaR, zero/all/clustered exceptions, malformed data/KRD/config, corrupted artifacts, and failure-ledger paths are tested;
- top-three cumulative variance: 96.8656%; OOS worst-tenor RMSE: 2.3783 bp;
- historical stability thresholds are breached materially;
- current-EWMA versus structural maximum top-three angle is 5.5977° and maximum symmetric factor-sigma ratio is 1.2267, both below their 30°/1.25 review levels;
- methodology PC1 sigma ratio is 1.2785 versus a 1.25 review level;
- no forecast challenger passes adoption; 18/18 independent interval results are INCONCLUSIVE;
- recent VaR and ES tail masses are below 20, so a controlled KRD CLI run is `NEEDS REVISION` even when its aggregate VaR coverage test passes.

The alternate SVD/algebra implementation is an implementation challenge, not an organizationally independent validation function.

## Required institutional controls before any expanded use

An adopting firm must supply named data/model/technology/business/risk owners; an independent validator; inventory ID and tier; documented approved and prohibited uses; thresholds tied to risk appetite and decision loss; official position, sensitivity, valuation, and P&L lineage; point-in-time EOD market-data archives; nonlinear/full-revaluation benchmarks; access and entitlement controls; override approval and expiry; monitoring frequency; issue owners and due dates; immutable centralized audit retention; dependency/SBOM and signed-release controls; capacity, storage, backup, restore, disaster recovery, and retirement procedures.

Until those controls and use-specific validation exist, outputs remain restricted research or descriptive diagnostics.
