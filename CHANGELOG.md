# Changelog

## 3.3.0 — 2026-08-26

- Fixed the first-cell Colab failure caused by rejecting every Python runtime at or above 3.13.
- Expanded formal package support, classifiers, tests, and GitHub Actions coverage to Python 3.10–3.13.
- Changed the standalone bootstrap to fail only below Python 3.10 and to label future, newer-than-validated runtimes as guarded execution instead of failing before dependency and integrity checks.
- Reproduced the reported path on CPython 3.13.14 from an isolated directory with neither the project package nor Jinja2 installed; all 17 code cells, 59 outputs, and five figures completed without error.
- Added a permanent regression gate against reintroducing an upper-version hard rejection and ran all 152 cases on CPython 3.10, 3.11, 3.12, and 3.13.
- Reconciled the v3.2.0 and v3.3.0 complete normalized analytical results byte for byte across Python 3.12 and 3.13, proving no mathematical or financial output drift.

## 3.2.0 — 2026-08-26

- Fixed the notebook-only Colab failure caused by requiring repository-local `src/` and `data/` directories.
- Added a deterministic embedded runtime containing the exact package source and frozen resources, with archive, member, size, path-containment, implementation, and snapshot integrity gates.
- Kept complete-repository execution as a separate hash-bound mode while removing any need for Git clone, network access, or project installation in standalone mode.
- Fixed the no-Jinja2 table fallback for Pandas versions that raise `AttributeError` from the `.style` accessor.
- Added isolated-directory bootstrap tests, standalone full-notebook execution checks, and dual-mode notebook execution to the Python 3.10–3.12 GitHub Actions matrix.

## 3.1.0 — 2026-08-26

- Removed every executable Python and notebook-code inline comment while retaining API docstrings, model documentation, and decision-use disclosures.
- Replaced explanatory comments with descriptive constants, variable names, and explicit provenance-field collections so intent remains visible in code.
- Added release tests that tokenize every Python source and committed notebook code cell and fail on any future inline comment.
- Rebuilt the deterministic notebook and all distributions under a new non-breaking version to preserve artifact and package traceability.

## 3.0.0 — 2026-08-25

- Replaced the inconsistent notebook/sklearn sign orientation with a coherent symmetric eigendecomposition and exact transform/inverse-transform state.
- Added physical-unit covariance/correlation/EWMA mappings, similarity plus dominance identification gates, and generic monitoring labels that preserve `UNIDENTIFIED` status.
- Added strict H.15 snapshot/live/cache integrity, byte bounds, manifest/resource hashes, synchronized complete curves, freshness, holiday-horizon, zero-bound, methodology and PIT-vintage controls.
- Added retrospective rolling and chronologically separated sequential stability, 2,000-replication block bootstrap, 2,000-replication 4/13/26-week sensitivity, methodology-regime challenge, and expanding-OOS reconstruction with source-position audit.
- Rebuilt forecasting as expanding-origin no-change/mean/PCA-AR/PCA-VAR challenge with disjoint selection, confirmation and interval evaluation; one-sided Clark-West/HAC plus materiality gates; dependence-aware familywise block-bootstrap marginal coverage; and joint-hit diagnostics that are not presented as simultaneous bands.
- Added positive-DV01 mapping, direct factor/residual variance reconciliation, structural/EWMA comparison, exact historical scenarios, recent/full-history finite-sample VaR and fractional ES, rolling VaR coverage tests, and bounded/conditioned hedge optimization with finite pre-risk and improvement contracts.
- Centralized risk tail-mass review, made unattainable finite-sample VaR targets `NOT EVALUATED`, and made inadequate recent tail evidence produce KRD `NEEDS REVISION`.
- Added strict bounded configuration and KRD schemas, required caller-attested portfolio/position-snapshot/sensitivity-engine identifiers, duplicate JSON-key rejection, cross-field sample-size constraints, atomic content-addressed outputs, fixed artifact-name reconciliation, SHA-256 plus schema-v3 structural contracts, and invocation success/failure lineage.
- Added POSIX/Windows crash-released cache locks, exact/fsynced POSIX ledger writes, argument-parse failure events, and per-invocation data warning/fallback/cache context.
- Added deterministic notebook generation with source/data hash binding, execution/output checks, Python 3.10–3.12 CI, job timeout, extracted-sdist tests, and isolated wheel/sdist smoke tests.
- Rewrote the model card, validation report, runbook, original-code review, and Chinese executive review around restricted research scope and explicit organizational-control gaps.
- Recorded the final lint, 146-test, executed-notebook, complete-sdist, isolated-install, CLI-identity, and role-based P0/P1 verification evidence.

Final release test count, artifact hashes, and notebook execution state are recorded by the release verification run rather than hard-coded here.
