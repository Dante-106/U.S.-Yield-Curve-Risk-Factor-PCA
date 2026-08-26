# Release Verification Record

## Release candidate

- Version: `3.3.0`
- Verification date: `2026-08-26` (America/New_York)
- Python verification runtimes: `3.10.20`, `3.11.15`, `3.12.13`, and `3.13.14`
- Implementation SHA-256: `00bacee9e61bfb8d44ee268a48ea7ed0e3cca722ae68d41085d3d0d9794e9887`
- Frozen snapshot SHA-256: `910dab1695a286a242d4d8832838938dc3cbec1795d048c6eaebb14c5ae4f08e`
- Embedded runtime archive SHA-256: `5c680a2e6bc2ee1adced3677a3a27b15d4f74097dc0590d9ce3cf4bd66fc0661`
- v3.0.0/v3.1.0 normalized full-result SHA-256: `8838bcbc07a60919726cffd668be589069630d07aa35e849799e95321e3e45d3`
- v3.1.0/v3.2.0 normalized pipeline plus structural/current-risk SHA-256: `d74a201c74e1e93e4b95ffeafb985d3b2974cc1c31adce688e8c73a3f071204b`
- v3.2.0/v3.3.0 normalized pipeline plus structural/current-risk SHA-256: `d74a201c74e1e93e4b95ffeafb985d3b2974cc1c31adce688e8c73a3f071204b`
- Final weekly market-data date / release-date staleness: `2025-12-26 / 243 calendar days`

## Required checks and observed results

| Check | Observed result |
|---|---|
| Ruff over `src`, `tests`, and `scripts` | PASS |
| Python and notebook-code comment-token gate | PASS; zero comment tokens |
| Unit/integration/failure-path suite | 152 PASS independently on each of Python 3.10.20, 3.11.15, 3.12.13, and 3.13.14 |
| Deterministic notebook source synchronization | 29/29 cells synchronized |
| Complete-repository notebook execution | Python 3.13.14; 17/17 code cells; 59 outputs; zero errors; `repository` binding |
| Isolated single-file notebook execution | Python 3.13.14; 17/17 code cells; 59 outputs including five embedded PNG figures; zero errors; `standalone embedded` binding |
| Clean standalone environment | PASS; project package and Jinja2 both absent; exact reported Python 3.13 path reproduced |
| Runtime compatibility boundary | Python 3.10–3.13 formally supported; below 3.10 fails; future versions are visibly guarded rather than upper-bound rejected |
| Embedded runtime tamper rejection | PASS; altered payload fails SHA-256 before import |
| Complete source-distribution test from extracted archive | 152 PASS on Python 3.13.14 |
| Wheel install in a fresh virtual environment and `pip check` | PASS |
| Source-distribution build/install in a fresh virtual environment and `pip check` | PASS |
| Installed wheel full snapshot CLI run | PASS |
| Installed source-distribution full snapshot CLI run | PASS |
| Wheel/sdist analytic identity | Same run ID `5a376b6c…01786` and implementation SHA-256 |
| Wheel/sdist artifact contracts | 53/53 names, hashes, and structural contracts identical |
| v3.1.0/v3.2.0 full-result equivalence | 3,718,192 normalized bytes identical; same SHA-256 |
| v3.2.0/v3.3.0 full-result equivalence | 3,718,192 normalized bytes identical across Python 3.12/3.13; same SHA-256 |
| Colab-style notebook-only execution | PASS on Python 3.13.14 from an isolated directory with no project installation |
| Notebook figure visual QA | PASS; five figures, no clipping or heading collisions |
| Independent role-based audit | P0 = 0; P1 = 0 at release freeze |

The two isolated CLI runs intentionally differ only in invocation/load/completion timestamps and warning-ticket labels stored outside the immutable analytic result. After removing those invocation-time fields, stable run ID, implementation, provenance, configuration, validation disposition, 53 artifact hashes, and 53 structural contracts reconcile byte for byte.

## Reproduction commands

```bash
python -m pip install -e ".[dev,notebook]"
ruff check src tests scripts
pytest -q
python scripts/check_executed_notebook.py notebooks/us_treasury_yield_curve_pca.ipynb
generated_notebook="$(mktemp --suffix=.ipynb)"
python scripts/build_notebook.py --output "$generated_notebook"
python scripts/check_notebook_sync.py notebooks/us_treasury_yield_curve_pca.ipynb "$generated_notebook"
python scripts/execute_notebook.py notebooks/us_treasury_yield_curve_pca.ipynb --timeout 300
python scripts/check_standalone_notebook.py notebooks/us_treasury_yield_curve_pca.ipynb --timeout 300
python -m build
```

This record demonstrates repository-level technical verification, not organizational model approval, independent institutional validation, production authorization, or acceptance of any prohibited use.
