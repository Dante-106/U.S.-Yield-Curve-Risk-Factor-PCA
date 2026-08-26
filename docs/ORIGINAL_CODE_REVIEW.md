# Review of the Supplied Notebook

## Conclusion

The supplied `U.S. Yield Curve Risk Factor PCA(8).ipynb` contained useful research structure—official H.15 data, weekly changes, covariance PCA, basic reconstruction, rolling summaries, simple forecasts and charts—but it was not safe for risk or forecast decisions. The most severe defect was an internally inconsistent sign orientation. Several later cells also overstated validation, mixed research and operational semantics, and omitted portfolio-risk controls required in a fixed-income market-risk workflow.

Version 3.0.0 is a ground-up controlled research implementation. It does not claim that engineering remediation creates production approval.

## Critical findings and remediation

| Severity | Original finding | Consequence | Remediation |
|---|---|---|---|
| P0 | Components and copied scores were sign-flipped for display, but the fitted sklearn estimator was not updated | Tables/charts and `transform`/`inverse_transform` referred to different factors; transform mismatch reached 211.99 bp and full-rank inverse error 143.33 bp | One coherent `eigh` calibration owns oriented components, scores, transforms, inverse transforms and physical shocks; full-rank error ≈`7.1e-14` bp |
| P0 | In-sample reconstruction and self-consistency were presented as validation | Could not detect temporal leakage, regime failure or specification risk | Expanding-OOS reconstruction, chronological audits, alternative algebra, correlation/EWMA/methodology challenges, independent forecast periods |
| P0 | Forecast comparison did not enforce a genuine benchmark/adoption protocol | A negligible or lucky challenger could be promoted | No-change baseline; 1% materiality; one-sided Clark-West/HAC; Bonferroni selection; new confirmation; no challenger adopted |
| P0 | No decision boundary between historical research and current risk | Stale/latest-revised data could be mistaken for current or PIT evidence | Explicit source-as-of/acquisition/model-as-of/freshness/vintage controls and prohibited-use matrix |
| P1 | Notebook embedded a large compressed payload without a separately governed manifest | Weak provenance, hard review and reuse | Repository and package resource snapshot with exact compressed/uncompressed hashes, schema, series, units and acquisition metadata |
| P1 | Boundary weeks and sample counts were hard-coded | Partial final period produced 1,356 changes and brittle assertions | Explicit `drop`/`include_and_flag` policy; default has 1,355 complete changes and dynamic controls |
| P1 | Weekly resampling could mix asynchronous tenor observations | Artificial curves and covariance | Complete same-day curve selection before weekly sampling; actual observation dates retained |
| P1 | Cache could truncate history and lacked exact upstream replay | Silent sample drift | Query identity, exact payload + canonical bytes, hashes, age/coverage/QA admission, atomic commit and platform locks |
| P1 | Factor signs alone were treated as economic identification | Near-tied templates could be force-labelled | Global assignment plus similarity and dominance margin; `UNIDENTIFIED` is preserved through generic PC monitoring labels |
| P1 | Rolling analysis emphasized variance and used a terminal reference without clear semantics | Large rotations were hidden; retrospective result could look PIT | Principal angles, projection distance, eigenvalue gap, cosines, explicit retrospective label, plus adjacent non-overlapping sequential windows |
| P1 | Bootstrap uncertainty was too thin and block choice unchallenged | Unstable 2.5%/97.5% tails | 2,000 primary and 2,000 per block sensitivity at 4/13/26 weeks; seed, reps and 50 expected tail order stats reported |
| P1 | Covariance/correlation/physical units could be conflated | Incorrect shock and risk mapping | Model-space and physical-bp bases separated; rolling API inherits and enforces like-for-like specification |
| P1 | Forecast intervals used full-sample residual information and weak calibration claims | Leakage/post-selection optimism | Prequential residuals, disjoint 260-week interval window, dependence-aware familywise block gate, conditional full-history contra-evidence |
| P1 | No KRD sign/units/as-of/curve schema, residual risk, tail controls or scenarios | Not decision-relevant for a risk manager | Exact KRD contract; positive-DV01 convention; full variance/residual reconciliation; recent/full tails; exact historical scenarios; rolling VaR diagnostics |
| P1 | Historical VaR used an ambiguous library quantile convention | Nominal coverage was not finite-sample explicit | Predictive order rank `ceil((n+1)c)`; exception bound; target-achievability gate; fractional ES tail mass |
| P1 | `±σ` scenarios could be read probabilistically despite non-normal/non-IID factors | Misleading likelihood and return periods | Explicit historical-covariance scaling only; no probability, return period or joint plausibility claim |
| P1 | Output directories/manifests lacked full integrity and invocation semantics | Partial/tampered results or warning labels could be misread | Atomic content address; independent expected controls/artifact set; hashes plus structural contracts; success/failure ledger; exit 0/2/3/4 |
| P2 | Monolithic notebook mixed acquisition, model, forecast, risk and presentation | Hard to test, review and operate | Typed package modules, strict configs, CLI, deterministic notebook builder and focused tests |
| P2 | Several labels implied “approved”, “production” or stable taxonomy | Governance overclaim | Restricted technical disposition and explicit institutional-control gap throughout |

## Sample reconciliation

The supplied notebook's boundary convention included a partial final period and produced 1,356 changes. The controlled default drops incomplete first/final boundary weeks and produces 1,355 changes from 2000-01-14 through 2025-12-26, supported by the 2000-01-07 baseline curve. Top-three cumulative variance remains 96.8656%; top-three MAE/RMSE are 1.3421/1.9882 bp.

## What the rewrite does not solve

The repository still does not contain point-in-time historical vintages, a current approved market feed, an executable pricing curve, official positions/sensitivities/P&L, nonlinear/full revaluation, desk limit calibration, hypothetical stresses, independent organizational validation, authorization, WORM retention, or release attestation. Those are external prerequisites, not code comments to be simulated.
