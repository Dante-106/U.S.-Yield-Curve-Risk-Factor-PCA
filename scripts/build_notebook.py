"""Build the reader-facing notebook from deterministic cell sources."""

from __future__ import annotations

import argparse
import os
from base64 import b64encode
from hashlib import sha1, sha256
from io import BytesIO
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

import nbformat as nbf

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _implementation_sha256() -> str:
    package_root = PROJECT_ROOT / "src/yield_curve_pca"
    digest = sha256()
    for source in sorted(package_root.rglob("*.py")):
        relative = source.relative_to(package_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        digest.update(source.read_bytes())
    return digest.hexdigest()


EXPECTED_IMPLEMENTATION_SHA256 = _implementation_sha256()
EXPECTED_SNAPSHOT_COMPRESSED_SHA256 = sha256(
    (PROJECT_ROOT / "data/h15_treasury_cmt_2000_2025.csv.gz").read_bytes()
).hexdigest()


def _embedded_runtime() -> tuple[bytes, tuple[str, ...]]:
    package_root = PROJECT_ROOT / "src/yield_curve_pca"
    included_suffixes = frozenset({".py", ".gz", ".json"})
    sources = sorted(
        path for path in package_root.rglob("*") if path.is_file() and path.suffix in included_suffixes
    )
    buffer = BytesIO()
    member_names: list[str] = []
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for source in sources:
            member_name = (Path("src/yield_curve_pca") / source.relative_to(package_root)).as_posix()
            member = ZipInfo(member_name, date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = ZIP_DEFLATED
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            archive.writestr(member, source.read_bytes())
            member_names.append(member_name)
    return buffer.getvalue(), tuple(member_names)


def _tuple_literal(values: tuple[str, ...], indentation: int = 0) -> str:
    prefix = " " * indentation
    return "(\n" + "".join(f"{prefix}    {value!r},\n" for value in values) + f"{prefix})"


EMBEDDED_RUNTIME_BYTES, EMBEDDED_RUNTIME_MEMBERS = _embedded_runtime()
EXPECTED_EMBEDDED_RUNTIME_SHA256 = sha256(EMBEDDED_RUNTIME_BYTES).hexdigest()
EMBEDDED_RUNTIME_BASE64 = b64encode(EMBEDDED_RUNTIME_BYTES).decode("ascii")
EMBEDDED_RUNTIME_CHUNKS = tuple(
    EMBEDDED_RUNTIME_BASE64[index : index + 96]
    for index in range(0, len(EMBEDDED_RUNTIME_BASE64), 96)
)
EMBEDDED_RUNTIME_MEMBERS_LITERAL = _tuple_literal(EMBEDDED_RUNTIME_MEMBERS)
EMBEDDED_RUNTIME_CHUNKS_LITERAL = _tuple_literal(EMBEDDED_RUNTIME_CHUNKS)
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Notebook output path.")
    arguments = parser.parse_args()
    requested_output = arguments.output
else:
    requested_output = None
OUTPUT = Path(
    requested_output
    or os.environ.get(
        "YIELD_CURVE_PCA_NOTEBOOK_OUTPUT",
        PROJECT_ROOT / "notebooks/us_treasury_yield_curve_pca.ipynb",
    )
).resolve()


def _cell_id(source: str, prefix: str) -> str:
    return f"{prefix}-{sha1(source.encode()).hexdigest()[:10]}"


def md(source: str):
    cell = nbf.v4.new_markdown_cell(source.strip() + "\n")
    cell["id"] = _cell_id(source, "md")
    return cell


def code(source: str):
    normalized = source.strip() + "\n"
    cell_id = _cell_id(source, "code")
    compile(normalized, f"<generated-notebook-{cell_id}>", "exec")
    cell = nbf.v4.new_code_cell(normalized)
    cell["id"] = cell_id
    return cell


cells = [
    md(
        r"""
# U.S. Treasury Yield-Curve PCA, Forecast Validation & Linear Risk Mapping

**Version 3.3.0 | controlled deterministic research notebook | frozen default run**

This notebook separates three questions that are often incorrectly conflated:

1. **Risk representation:** which orthogonal yield-curve shock directions explain historical covariance?
2. **Forecasting:** can past factor dynamics beat an unchanged-curve benchmark out of sample?
3. **Portfolio relevance:** how do key-rate DV01s convert curve shocks into dollar P&L and residual risk?

Repository scope is restricted historical U.S. Treasury CMT research, model challenge, and illustrative linear KRD diagnostics. `SHARE WITH CAVEATS` is a technical disposition—not organizational model approval, production authorization, or management acceptance of risk. This is **not** a SOFR/OIS pricing curve, full-revaluation engine, limit model, trading signal, or VaR/FRTB capital implementation.
"""
    ),
    md(
        """
## 0. Setup and configuration

Upload this notebook by itself to Colab and choose **Runtime → Run all**. The notebook carries a hash-verified, offline runtime containing the exact model implementation and frozen data snapshot, so no repository clone, network download, or project installation is required. When the complete repository is present, the notebook instead binds to its matching `src/` and `data/` files. `YIELD_CURVE_PCA_PROJECT_ROOT` may explicitly select that repository.
"""
    ),
    code(
        f"""
from base64 import b64decode
from hashlib import sha256
from io import BytesIO
from pathlib import Path, PurePosixPath
import os
import sys
from tempfile import TemporaryDirectory
from time import perf_counter
from zipfile import BadZipFile, ZipFile

runtime_python_version = sys.version_info[:2]
if runtime_python_version < (3, 10):
    raise RuntimeError(
        "This release requires Python 3.10 or newer. "
        "Select a compatible Colab or Jupyter runtime."
    )
runtime_validation_status = (
    "validated Python runtime"
    if runtime_python_version < (3, 14)
    else "newer-than-validated Python runtime; guarded execution"
)

try:
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy
    from IPython import get_ipython
    from IPython.display import Markdown, display
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The notebook requires numpy, pandas, scipy, matplotlib, and IPython. "
        "A standard Google Colab Python runtime includes these dependencies."
    ) from exc

ipython_shell = get_ipython()
if ipython_shell is not None:
    ipython_shell.run_line_magic("matplotlib", "inline")

configured_root = os.environ.get("YIELD_CURVE_PCA_PROJECT_ROOT")
def repository_layout_is_complete(path):
    return (
        (path / "src/yield_curve_pca").is_dir()
        and (path / "data/h15_treasury_cmt_2000_2025.csv.gz").is_file()
        and (path / "data/source_manifest.json").is_file()
    )


if configured_root is not None and not repository_layout_is_complete(Path(configured_root).expanduser()):
    raise FileNotFoundError(
        "YIELD_CURVE_PCA_PROJECT_ROOT does not contain the complete src and data layout: "
        + configured_root
    )
root_candidates = [
    *([Path(configured_root).expanduser()] if configured_root else []),
    Path.cwd(),
    *Path.cwd().parents,
]
PROJECT_ROOT = next(
    (path.resolve() for path in root_candidates if repository_layout_is_complete(path)),
    None,
)

EXPECTED_EMBEDDED_RUNTIME_SHA256 = "{EXPECTED_EMBEDDED_RUNTIME_SHA256}"
EXPECTED_EMBEDDED_RUNTIME_MEMBERS = {EMBEDDED_RUNTIME_MEMBERS_LITERAL}
EMBEDDED_RUNTIME_CHUNKS = {EMBEDDED_RUNTIME_CHUNKS_LITERAL}
MAX_EMBEDDED_RUNTIME_ARCHIVE_BYTES = 2_000_000
MAX_EMBEDDED_RUNTIME_EXPANDED_BYTES = 4_000_000

RUNTIME_DIRECTORY = None
if PROJECT_ROOT is not None:
    RUNTIME_MODE = "repository"
    PACKAGE_ROOT = PROJECT_ROOT / "src/yield_curve_pca"
    SNAPSHOT_PATH = PROJECT_ROOT / "data/h15_treasury_cmt_2000_2025.csv.gz"
    PIPELINE_PROJECT_ROOT = PROJECT_ROOT
else:
    RUNTIME_MODE = "standalone embedded"
    runtime_payload = b64decode("".join(EMBEDDED_RUNTIME_CHUNKS), validate=True)
    if len(runtime_payload) > MAX_EMBEDDED_RUNTIME_ARCHIVE_BYTES:
        raise RuntimeError("Embedded runtime archive exceeds its size control.")
    if sha256(runtime_payload).hexdigest() != EXPECTED_EMBEDDED_RUNTIME_SHA256:
        raise RuntimeError("Embedded runtime archive failed SHA-256 validation.")
    RUNTIME_DIRECTORY = TemporaryDirectory(prefix="yield_curve_pca_v330_")
    runtime_root = Path(RUNTIME_DIRECTORY.name).resolve()
    try:
        with ZipFile(BytesIO(runtime_payload)) as archive:
            members = archive.infolist()
            observed_member_names = tuple(member.filename for member in members)
            if observed_member_names != EXPECTED_EMBEDDED_RUNTIME_MEMBERS:
                raise RuntimeError("Embedded runtime archive member contract failed.")
            if sum(member.file_size for member in members) > MAX_EMBEDDED_RUNTIME_EXPANDED_BYTES:
                raise RuntimeError("Embedded runtime archive exceeds its expanded-size control.")
            for member in members:
                relative = PurePosixPath(member.filename)
                if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                    raise RuntimeError("Embedded runtime archive contains an unsafe path.")
                destination = runtime_root.joinpath(*relative.parts).resolve()
                if runtime_root not in destination.parents:
                    raise RuntimeError("Embedded runtime extraction escaped its temporary directory.")
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(archive.read(member))
    except BadZipFile as exc:
        raise RuntimeError("Embedded runtime archive is invalid.") from exc
    PACKAGE_ROOT = runtime_root / "src/yield_curve_pca"
    SNAPSHOT_PATH = PACKAGE_ROOT / "resources/h15_treasury_cmt_2000_2025.csv.gz"
    PIPELINE_PROJECT_ROOT = None

loaded_package_modules = [
    name for name in sys.modules if name == "yield_curve_pca" or name.startswith("yield_curve_pca.")
]
for module_name in loaded_package_modules:
    del sys.modules[module_name]
package_source_parent = str(PACKAGE_ROOT.parent)
sys.path = [entry for entry in sys.path if entry != package_source_parent]
sys.path.insert(0, package_source_parent)

from yield_curve_pca import DataConfig, ForecastConfig, PCAConfig, PipelineConfig, RiskConfig
from yield_curve_pca.pipeline import run_pipeline
from yield_curve_pca.reporting import (
    format_table,
    plot_curve_history,
    plot_forecast_rmse,
    plot_pca_structure,
    plot_rolling_stability,
    plot_variance_reconciliation,
)
from yield_curve_pca.risk import map_linear_curve_risk

pd.set_option("display.max_columns", 30)
pd.set_option("display.width", 180)
pd.set_option("display.float_format", lambda value: f"{{value:,.4f}}")
dependency_versions = (
    runtime_validation_status
    + " | Python " + ".".join(map(str, sys.version_info[:3]))
    + " | NumPy " + np.__version__
    + " | pandas " + pd.__version__
    + " | SciPy " + scipy.__version__
    + " | Matplotlib " + matplotlib.__version__
)
display(Markdown(
    "**Runtime ready:** " + RUNTIME_MODE + "; " + dependency_versions
    + ". Exact implementation and snapshot checks follow."
))
"""
    ),
    code(
        f"""
from hashlib import sha256

EXPECTED_IMPLEMENTATION_SHA256 = "{EXPECTED_IMPLEMENTATION_SHA256}"
EXPECTED_SNAPSHOT_COMPRESSED_SHA256 = "{EXPECTED_SNAPSHOT_COMPRESSED_SHA256}"

package_root = PACKAGE_ROOT
implementation_digest = sha256()
for source in sorted(package_root.rglob("*.py")):
    relative = source.relative_to(package_root).as_posix().encode("utf-8")
    implementation_digest.update(len(relative).to_bytes(4, "big"))
    implementation_digest.update(relative)
    implementation_digest.update(source.read_bytes())
observed_implementation_sha256 = implementation_digest.hexdigest()
observed_snapshot_sha256 = sha256(SNAPSHOT_PATH.read_bytes()).hexdigest()
assert observed_implementation_sha256 == EXPECTED_IMPLEMENTATION_SHA256
assert observed_snapshot_sha256 == EXPECTED_SNAPSHOT_COMPRESSED_SHA256
display(Markdown(
    "**Execution binding verified (" + RUNTIME_MODE + "):** implementation `" + observed_implementation_sha256
    + "`; compressed snapshot `" + observed_snapshot_sha256 + "`."
))
"""
    ),
    code(
        """
CONFIG = PipelineConfig(
    data=DataConfig(
        start_date="2000-01-01",
        end_date="2025-12-31",
        source_mode="snapshot",
        weekly_rule="W-FRI",
        boundary_week_policy="drop",
        maximum_staleness_days=None,
    ),
    pca=PCAConfig(
        retained_factors=3,
        minimum_template_similarity=0.70,
        minimum_template_dominance_margin=0.10,
        current_halflife_weeks=52.0,
        rolling_window_years=5,
        bootstrap_replications=2_000,
        bootstrap_block_weeks=13,
        random_seed=20260825,
    ),
    forecast=ForecastConfig(
        enabled=True,
        minimum_training_weeks=520,
        retained_factors=3,
        hac_lags=4,
        minimum_model_selection_observations=52,
        confirmation_observations=52,
        interval_evaluation_observations=260,
        interval_history_weeks=260,
        minimum_interval_observations=104,
        interval_bootstrap_replications=20_000,
        interval_bootstrap_block_lengths=(4, 13, 26),
        adoption_alpha=0.05,
        minimum_rmse_improvement=0.01,
    ),
    risk=RiskConfig(
        historical_window_weeks=520,
        var_confidence=0.99,
        es_confidence=0.975,
    ),
)

ILLUSTRATIVE_KRD_USD_PER_BP = pd.Series(
    {
        "3M": 0.0, "6M": 0.0, "1Y": 25_000.0, "2Y": 100_000.0,
        "3Y": 150_000.0, "5Y": 200_000.0, "7Y": 150_000.0,
        "10Y": 100_000.0, "20Y": 50_000.0,
    },
    name="DV01 (USD per bp fall in yield)",
)

started = perf_counter()
RESULT = run_pipeline(CONFIG, project_root=PIPELINE_PROJECT_ROOT)
RUNTIME_SECONDS = perf_counter() - started
display(Markdown(
    f"**Pipeline completed:** {RUNTIME_SECONDS:.2f} seconds  |  "
    f"**data mode:** {RESULT.data.provenance.mode}"
))
"""
    ),
    md("## tl;dr"),
    code(
        '''
pca = RESULT.structural_pca
warnings_count = int((RESULT.data.quality_table["status"] == "WARN").sum())
max_angle = RESULT.rolling_stability["Maximum principal angle (deg)"].max()
sequential_worst = RESULT.sequential_stability["Maximum principal angle (deg)"].max()
sequential_latest = RESULT.sequential_stability["Maximum principal angle (deg)"].iloc[-1]
forecast_decision = RESULT.forecast_note if RESULT.forecast is None else RESULT.forecast.selected_model
partition_text = "not run"
if RESULT.forecast is not None:
    partition_counts = RESULT.forecast.audit_trail["Evaluation partition"].value_counts()
    partition_text = (
        f"{partition_counts.get('Selection', 0)} selection / "
        f"{partition_counts.get('Confirmation', 0)} confirmation / "
        f"{partition_counts.get('Interval evaluation', 0)} interval-evaluation"
    )
coverage_warning = (
    "not evaluated"
    if RESULT.forecast is None
    else (
        f"{(RESULT.forecast.interval_coverage['Status'] == 'INCONCLUSIVE').sum()} of "
        f"{len(RESULT.forecast.interval_coverage)} gates INCONCLUSIVE; "
        f"minimum independent gap {RESULT.forecast.interval_coverage['Coverage gap'].min():.1%}"
    )
)

summary = f"""
### Executive result

- The controlled weekly sample contains **{len(RESULT.data.weekly_changes_bp):,}** change observations from **{RESULT.data.weekly_changes_bp.index.min().date()}** to **{RESULT.data.weekly_changes_bp.index.max().date()}**, supported by the baseline level curve on **{RESULT.data.weekly_yields_pct.index.min().date()}**. It is one frozen/latest-revised history—not a point-in-time vintage panel—and is stale for current decisions.
- The first three covariance PCs explain **{pca.explained_ratio[:3].sum():.2%}** of equal-tenor historical curve-change variance. Template similarity is **{pca.template_similarity[0]:.3f} / {pca.template_similarity[1]:.3f} / {pca.template_similarity[2]:.3f}** and dominance margin is **{pca.template_dominance_margin[0]:.3f} / {pca.template_dominance_margin[1]:.3f} / {pca.template_dominance_margin[2]:.3f}**.
- Compression does **not** prove factor stability: retrospective full-sample-reference worst angle is **{max_angle:.1f}°**; chronologically separated adjacent-window worst/latest are **{sequential_worst:.1f}° / {sequential_latest:.1f}°**. Historical evidence rejects a universally stable factor-limit taxonomy.
- Walk-forward adoption uses disjoint **{partition_text}** observations and selects **{forecast_decision}** as the fallback benchmark; no PCA challenger passed both economic and one-sided Clark–West gates.
- Marginal-band calibration is **{coverage_warning}**. Familywise block-bootstrap coverage inference is not a simultaneous curve band and does not authorize limit use.
- Numerical validation is **{RESULT.validation.overall_assessment}**; the data layer reports **{warnings_count}** review items. Warnings are visible controls, not silently converted into PASS.

**Repository-scope disposition:** restricted historical research and illustrative linear KRD diagnostics only. Factor limits, current risk, official VaR/ES, hedge sizing, trading, capital, and management forecasting are **NOT ELIGIBLE**. External owner, validation, approval, entitlement, and data-vintage controls remain prerequisites.
"""
display(Markdown(summary))
'''
    ),
    md(
        r"""
## 1. Context & methods

### Key assumptions

- H.15 constant-maturity yields are nominal Treasury **par-yield risk proxies**, not tradable zero rates or collateralized derivatives discount factors.
- The primary PCA uses same-date complete curves and weekly changes in basis points: $\Delta y_{t,i}=100(y_{t,i}-y_{t-1,i})$ because H.15 observations are quoted in percent.
- For $X\in\mathbb{R}^{T\times p}$, $\Sigma=V\Lambda V^\top$ and factor scores are $F=(X-\mu)V$. Unit-norm eigenvectors $V$, one-standard-deviation shocks $V\sqrt{\Lambda}$, and tenor-factor correlations are reported separately.
- PCA is a covariance representation, not a forecast. Forecast models are re-estimated at every origin using only information available before the target week.
- Positive DV01 means price increases when yield falls 1 bp, so linear P&L is $\Delta P\approx-d^\top\Delta y_{bp}$.
- Chronological row splits prevent future-row leakage, but the frozen/latest-revised source is not a historical vintage archive; revision look-ahead remains possible.
"""
    ),
    md("## 2. Data, provenance & quality"),
    code(
        """
provenance = pd.DataFrame([RESULT.data.provenance.to_dict()]).T
provenance.columns = ["Value"]
display(format_table(provenance, caption="Run-level data provenance"))

quality = RESULT.data.quality_table.copy()
display(format_table(quality, caption="Data-quality controls and decision impact"))

gap_distribution = (
    RESULT.data.observation_gap_days.value_counts().sort_index().rename("Weekly changes").to_frame()
)
gap_distribution.index.name = "Calendar gap (days)"
display(format_table(gap_distribution, formats={"Weekly changes": "{:,.0f}"}, caption="Actual weekly observation horizons"))
"""
    ),
    code(
        """
fig = plot_curve_history(RESULT.data.weekly_yields_pct)
display(fig)
plt.close(fig)
"""
    ),
    md("## 3. Structural and recency-weighted PCA"),
    code(
        """
components = []
for index, name in enumerate(pca.component_names):
    components.append(
        {
            "Component": name,
            "Economic shape": pca.assigned_shapes[index] if index < 3 else "Residual",
            "Suggested shape": pca.suggested_shapes[index] if index < 3 else "—",
            "Identification": pca.identification_status[index] if index < 3 else "—",
            "Eigenvalue": pca.eigenvalues[index],
            "Variance share": pca.explained_ratio[index],
            "Cumulative share": pca.explained_ratio[: index + 1].sum(),
            "Template similarity": pca.template_similarity[index] if index < 3 else np.nan,
            "Template dominance margin": pca.template_dominance_margin[index] if index < 3 else np.nan,
        }
    )
component_table = pd.DataFrame(components).set_index("Component")
display(format_table(
    component_table,
    formats={
        "Eigenvalue": "{:.3f}", "Variance share": "{:.2%}",
        "Cumulative share": "{:.2%}", "Template similarity": "{:.3f}",
        "Template dominance margin": "{:.3f}",
    },
    caption="Structural covariance PCA | eigenvalues in bp²",
))

display(format_table(pca.loading_table(3), formats="{:+.4f}", caption="Unit-norm eigenvectors V"))
display(format_table(pca.sigma_shock_table(3), formats="{:+.2f}", caption="One-standard-deviation physical curve shocks V√Λ (bp)"))
display(format_table(pca.tenor_factor_correlation(3), formats="{:+.3f}", caption="Tenor-to-factor correlations"))
"""
    ),
    code(
        """
fig = plot_pca_structure(pca, CONFIG.pca.retained_factors)
display(fig)
plt.close(fig)
"""
    ),
    code(
        """
current = RESULT.current_ewma_pca
current_comparison = pd.DataFrame(
    {
        "Structural variance share": pca.explained_ratio[:3],
        "Current EWMA variance share": current.explained_ratio[:3],
        "Structural shape": pca.assigned_shapes,
        "Current shape": current.assigned_shapes,
        "Current identification": current.identification_status,
        "Current template similarity": current.template_similarity,
        "Current template dominance margin": current.template_dominance_margin,
    },
    index=["PC1", "PC2", "PC3"],
)
display(format_table(
    current_comparison,
    formats={
        "Structural variance share": "{:.2%}",
        "Current EWMA variance share": "{:.2%}",
        "Current template similarity": "{:.3f}",
        "Current template dominance margin": "{:.3f}",
    },
    caption=f"Structural versus recency-weighted PCA as of {current.training_end.date()} | EWMA effective n={current.effective_observations:.1f}",
))
display(format_table(
    RESULT.structural_current_ewma_comparison,
    formats={
        "Physical loading cosine": "{:.4f}",
        "Ordered subspace principal angle (deg)": "{:.3f}",
        "Reference one-sigma norm (bp)": "{:.3f}",
        "Candidate one-sigma norm (bp)": "{:.3f}",
        "Candidate/reference sigma ratio": "{:.4f}",
    },
    caption=(
        "Structural versus current-EWMA absolute factor shocks | "
        "candidate aligned to structural physical coordinates"
    ),
))
"""
    ),
    md("## 4. Reconstruction, specification challenge & model stability"),
    code(
        """
display(format_table(RESULT.reconstruction_summary, formats={"Value (bp)": "{:.4f}"}, caption="In-sample reconstruction controls"))
display(format_table(RESULT.reconstruction_by_tenor, formats="{:.3f}", caption="In-sample top-3 residual by tenor"))
display(format_table(RESULT.oos_reconstruction_metrics, formats="{:.3f}", caption="Strictly out-of-time top-3 reconstruction by tenor"))
assert (RESULT.oos_reconstruction_audit["Training end"] < RESULT.oos_reconstruction_audit.index).all()
oos_audit_excerpt = pd.concat(
    [RESULT.oos_reconstruction_audit.head(3), RESULT.oos_reconstruction_audit.tail(3)]
)
display(format_table(
    oos_audit_excerpt,
    formats={"Target position": "{:,.0f}", "Basis refit sequence": "{:,.0f}",
             "Basis refit trigger position": "{:,.0f}", "Maximum source position": "{:,.0f}",
             "Training observations": "{:,.0f}"},
    caption="OOS basis-vintage audit excerpt | every training end precedes its target",
))

challenge_formats = {
    "Effective observations": "{:.1f}", "PC1": "{:.2%}", "PC2": "{:.2%}",
    "PC3": "{:.2%}", "Top-3 cumulative": "{:.2%}", "Minimum template similarity": "{:.3f}",
    "Minimum template dominance margin": "{:.3f}",
}
display(format_table(RESULT.specification_challenge, formats=challenge_formats, caption="Controlled PCA specification challenge"))
display(format_table(
    RESULT.methodology_sensitivity,
    formats={
        "Physical loading cosine": "{:.3f}",
        "Ordered subspace principal angle (deg)": "{:.2f}",
        "Reference one-sigma norm (bp)": "{:.2f}",
        "Candidate one-sigma norm (bp)": "{:.2f}",
        "Candidate/reference sigma ratio": "{:.3f}",
        "Pre top-3 variance": "{:.2%}",
        "Post top-3 variance": "{:.2%}",
    },
    caption="Pre/post 2021-12-06 Treasury methodology sensitivity (cross-boundary week excluded)",
))
display(Markdown(
    "**Methodology interpretation:** ordered principal angles describe the three-dimensional "
    "subspace as a set; they are not PC-specific Level/Slope/Curvature angles. Matched-factor "
    "identity is assessed separately by physical loading cosine. Market-regime and source-method "
    "effects remain confounded."
))
"""
    ),
    code(
        """
rolling_columns = [
    "Window start", "Observations", "Top-3 variance", "Maximum principal angle (deg)",
    "Projection distance", "Cutoff eigenvalue gap", "Minimum template similarity",
    "PC1 loading cosine", "PC2 loading cosine", "PC3 loading cosine",
]
rolling_summary = RESULT.rolling_stability[rolling_columns].agg(["min", "median", "max"])
display(format_table(
    rolling_summary,
    formats={
        column: "{:.4f}"
        for column in rolling_summary.select_dtypes(include="number").columns
    },
    caption="Quarterly-step five-year loading/subspace stability summary",
))
display(format_table(
    RESULT.bootstrap_stability,
    formats={
        "2.5%": "{:.4f}", "Median": "{:.4f}", "97.5%": "{:.4f}",
        "Replications": "{:,.0f}", "Block length (weeks)": "{:,.0f}",
        "Random seed": "{:,.0f}",
        "Expected order statistics per 2.5% tail": "{:.1f}",
    },
    caption="Circular 13-week block-bootstrap uncertainty",
))
display(format_table(
    RESULT.bootstrap_block_sensitivity,
    formats={
        "Replications": "{:,.0f}", "Angle median (deg)": "{:.2f}",
        "Angle 97.5% (deg)": "{:.2f}", "Top-3 variance median": "{:.2%}",
        "Top-3 variance 2.5%": "{:.2%}", "Minimum median loading cosine": "{:.3f}",
    },
    caption=(
        "Block-length sensitivity — conditional sampling uncertainty only; "
        "rolling windows remain the regime-instability control"
    ),
))
display(Markdown(
    "**Reference discipline:** the chart below is retrospective against the frozen full-sample "
    "basis. It is not a historical alert or point-in-time replay. The adjacent-window table is "
    "chronologically separated and uses no rows after each monitoring-window end, but every row "
    "comes from one frozen/latest-revised vintage; it is not a point-in-time-vintage replay."
))
display(format_table(
    RESULT.sequential_stability,
    formats={
        "Observations per window": "{:,.0f}",
        "Monitoring top-3 variance": "{:.2%}",
        "Maximum principal angle (deg)": "{:.2f}",
        "Minimum aligned loading cosine": "{:.3f}",
        "Factor 1 cosine": "{:.3f}", "Factor 2 cosine": "{:.3f}",
        "Factor 3 cosine": "{:.3f}",
    },
    caption="Sequential adjacent non-overlapping five-year windows (reference end < monitor start)",
))

fig = plot_rolling_stability(
    RESULT.rolling_stability,
    CONFIG.pca.retained_factors,
    loading_cosine_review_level=CONFIG.pca.minimum_loading_cosine_warning,
)
display(fig)
plt.close(fig)
"""
    ),
    code(
        """
display(format_table(
    RESULT.validation.distribution_checks,
    formats={
        column: "{:.4g}"
        for column in RESULT.validation.distribution_checks.select_dtypes(include="number").columns
    },
    caption="Factor distribution and serial-dependence diagnostics",
))
print(
    "Interpretation: formal skew/kurtosis, Jarque–Bera, and Ljung–Box diagnostics—not the mere "
    "existence of extreme observations—support non-Gaussian and time-varying-risk caveats."
)
"""
    ),
    md("## 5. One-week-ahead forecast challenge"),
    code(
        """
if RESULT.forecast is None:
    display(Markdown(f"**Forecast status:** {RESULT.forecast_note}"))
else:
    forecast = RESULT.forecast
    partition_summary = pd.DataFrame(
        [
            {
                "Evaluation partition": label,
                "Start": frame.index.min(),
                "End": frame.index.max(),
                "Observations": len(frame),
                "Maximum training end": frame["Training end"].max(),
            }
            for label, frame in forecast.audit_trail.groupby(
                "Evaluation partition", sort=False
            )
        ]
    ).set_index("Evaluation partition")
    display(Markdown(
        f"**Forecast timing:** as-of **{forecast.forecast_as_of.date()}** → target weekly "
        f"period end **{forecast.target_period_end.date()}**. This target is historical/stale, "
        "not a current trading or management forecast."
    ))
    display(format_table(
        partition_summary,
        formats={"Observations": "{:,.0f}"},
        caption="Disjoint adoption and interval-evaluation partitions | every training end < target",
    ))
    display(format_table(
        forecast.model_comparison,
        formats={
            "Selection curve RMSE (bp)": "{:.4f}",
            "Selection RMSE improvement vs no-change": "{:+.3%}",
            "Mean weekly squared-loss advantage": "{:+.4f}",
            "HAC statistic": "{:+.3f}",
            "Two-sided p-value": "{:.4f}",
            "Clark-West adjusted advantage": "{:+.4f}",
            "Clark-West HAC statistic": "{:+.3f}",
            "Clark-West one-sided p-value": "{:.4f}",
            "Multiplicity-adjusted alpha": "{:.4f}",
            "Confirmation curve RMSE (bp)": "{:.4f}",
            "Confirmation RMSE improvement": "{:+.3%}",
            "Confirmation Clark-West one-sided p-value": "{:.4f}",
        },
        caption=(
            "Expanding-origin selection and chronologically held-out confirmation | "
            "interval evaluation is not used for model adoption"
        ),
    ))
    fig = plot_forecast_rmse(forecast)
    display(fig)
    plt.close(fig)

    display(format_table(
        forecast.prediction_intervals_pct.T,
        formats="{:.3f}%",
        caption=(
            f"Historical next-period curve forecast as of {forecast.forecast_as_of.date()} "
            f"for {forecast.target_period_end.date()} | fallback: {forecast.selected_model}"
        ),
    ))
    coverage_columns = [
        "Hits", "Evaluated forecasts", "Observed coverage", "Coverage gap",
        "Acceptable coverage floor", "Hit lag-1 autocorrelation",
        "Familywise block-bootstrap coverage lower bound",
        "Familywise block-bootstrap coverage upper bound", "Status",
    ]
    coverage_formats = {
        "Hits": "{:,.0f}", "Evaluated forecasts": "{:,.0f}",
        "Observed coverage": "{:.1%}", "Coverage gap": "{:+.1%}",
        "Acceptable coverage floor": "{:.1%}", "Hit lag-1 autocorrelation": "{:+.3f}",
        "Familywise block-bootstrap coverage lower bound": "{:.1%}",
        "Familywise block-bootstrap coverage upper bound": "{:.1%}",
    }
    interval_observations = int(forecast.interval_coverage["Evaluated forecasts"].min())
    bootstrap_replications = int(forecast.interval_coverage["Bootstrap replications"].min())
    bootstrap_blocks = str(forecast.interval_coverage["Bootstrap block lengths"].iloc[0])
    bootstrap_block_count = len(CONFIG.forecast.interval_bootstrap_block_lengths)
    inference_family = len(forecast.interval_coverage)
    display(format_table(
        forecast.interval_coverage[coverage_columns],
        formats=coverage_formats,
        caption=(
            f"Independent {interval_observations}-week marginal-tenor coverage challenge | "
            f"{bootstrap_replications:,} circular moving-block replications, blocks "
            f"{bootstrap_blocks}, Bonferroni family of {inference_family}; coverage-inference "
            "bounds, not a simultaneous curve band"
        ),
    ))
    display(format_table(
        forecast.full_history_interval_diagnostic[coverage_columns],
        formats=coverage_formats,
        caption=(
            "Long-history conditional/post-selection marginal coverage diagnostic | retained as "
            "contra-evidence; not an independent adoption gate"
        ),
    ))
    joint_hit_diagnostic = pd.concat(
        {
            "Independent interval evaluation": forecast.simultaneous_interval_diagnostic,
            "Conditional full history": forecast.full_history_simultaneous_diagnostic,
        },
        names=("Diagnostic sample", "Marginal nominal coverage"),
    )
    display(format_table(
        joint_hit_diagnostic,
        formats={
            "Joint hit rate of marginal tenor bands": "{:.1%}",
            "Evaluated forecasts": "{:,.0f}",
        },
        caption="Joint hit diagnostic of marginal bands—not a calibrated joint interval",
    ))
    display(Markdown(
        f"**Inference limitation:** the {interval_observations}-week bootstrap challenge "
        "preserves weekly cross-tenor dependence and "
        f"challenges serial dependence across {bootstrap_block_count} block lengths, but assumes "
        "local stationarity "
        "and circular wrap. Wilson and IID-binomial fields are auxiliary only. This evaluation sample "
        "has low familywise power to certify a 3 pp non-inferiority margin; INCONCLUSIVE is "
        "therefore escalated rather than treated as PASS."
    ))
"""
    ),
    md("## 6. Illustrative portfolio risk mapping"),
    code(
        """
risk = map_linear_curve_risk(
    RESULT.data.weekly_changes_bp,
    ILLUSTRATIVE_KRD_USD_PER_BP,
    pca,
    CONFIG.risk,
    CONFIG.pca.retained_factors,
)
recency_weighted_risk = map_linear_curve_risk(
    RESULT.data.weekly_changes_bp,
    ILLUSTRATIVE_KRD_USD_PER_BP,
    RESULT.current_ewma_pca,
    CONFIG.risk,
    CONFIG.pca.retained_factors,
)
display(Markdown(
    "**Important:** the KRD below is a transparent demonstration input, not the user's position, "
    "not a limit, and not a trade recommendation. Any external KRD must first pass same-curve, "
    "as-of, portfolio lineage, sensitivity-engine, official-position, and organizational approval controls."
))
display(format_table(risk.key_rate_dv01_usd_per_bp.to_frame(), formats="${:,.0f}", caption="Illustrative key-rate DV01"))
display(format_table(risk.factor_exposure_usd_per_score.to_frame(), formats="${:,.0f}", caption="PCA factor exposure"))
display(format_table(
    pd.DataFrame(
        {
            "Structural": risk.factor_exposure_usd_per_score,
            "Recency-weighted EWMA": recency_weighted_risk.factor_exposure_usd_per_score,
        }
    ),
    formats="${:,.0f}",
    caption=(
        f"Structural versus recency-weighted exposure as of {RESULT.data.weekly_yields_pct.index.max().date()} "
        "(calibration-specific factor coordinates)"
    ),
))
display(format_table(
    risk.variance_reconciliation,
    formats={"Variance (USD²)": "${:,.0f}", "Variance share": "{:.2%}"},
    caption="Factor and residual linear-variance reconciliation",
))
risk_calibration = pd.DataFrame(
    {
        "Modeled variance (USD²)": {
            "Structural covariance": risk.variance_reconciliation.loc["Total", "Variance (USD²)"],
            "Current EWMA covariance": recency_weighted_risk.variance_reconciliation.loc["Total", "Variance (USD²)"],
        }
    }
)
risk_calibration["Weekly modeled volatility (USD)"] = np.sqrt(
    risk_calibration["Modeled variance (USD²)"]
)
risk_calibration["Ratio to structural"] = (
    risk_calibration["Weekly modeled volatility (USD)"]
    / risk_calibration.loc["Structural covariance", "Weekly modeled volatility (USD)"]
)
display(format_table(
    risk_calibration,
    formats={
        "Modeled variance (USD²)": "${:,.0f}",
        "Weekly modeled volatility (USD)": "${:,.0f}",
        "Ratio to structural": "{:.4f}",
    },
    caption="Illustrative KRD absolute structural versus recency-weighted risk",
))
tail_diagnostic = pd.concat(
    [risk.tail_risk_summary, risk.full_history_tail_risk_summary],
    keys=["Recent configured window", "Full available history"],
    names=["History scope", "Measure"],
)
display(format_table(
    tail_diagnostic,
    formats={"Confidence": "{:.1%}", "Loss (USD)": "${:,.0f}", "Effective tail mass": "{:.1f}", "History weeks": "{:,.0f}"},
    caption=(
        "Illustrative recent and full-history weekly linear tail loss—finite-sample "
        "predictive VaR convention; not regulatory VaR/ES"
    ),
))
exception_dates = risk.var_backtest_detail.loc[
    risk.var_backtest_detail["Exception"],
    ["Estimation start", "Estimation end", "Historical VaR (USD)", "Realized loss (USD)"],
]
display(format_table(
    exception_dates,
    formats={"Historical VaR (USD)": "${:,.0f}", "Realized loss (USD)": "${:,.0f}"},
    caption="Fixed-KRD exception dates | concentration by regime remains visible despite aggregate test PASS",
))
display(Markdown(
    "**Use boundary:** this is a fixed illustrative KRD applied to a single latest-revised history. "
    "It is not an actual-position, official-P&L, hypothetical-P&L, ES, PLA, or FRTB backtest. "
    "It omits convexity, options, spreads, basis, volatility, liquidity, funding, and cross-curve risk."
))
display(format_table(
    risk.var_backtest_summary,
    formats={
        "Confidence": "{:.1%}", "Expected exceptions": "{:.2f}",
        "Observed exception rate": "{:.2%}", "Kupiec p-value": "{:.4f}",
        "Christoffersen independence p-value": "{:.4f}",
        "Conditional coverage p-value": "{:.4f}",
    },
    caption=(
        "Rolling VaR diagnostic with fixed illustrative KRD and latest-revised linear scenario "
        "P&L—not actual-position/official-P&L backtesting"
    ),
))
recent_tail_statuses = set(risk.tail_risk_summary["Tail-mass review status"])
rolling_var_status = str(
    risk.var_backtest_summary.loc["Rolling historical VaR", "Status"]
)
illustrative_krd_evidence_status = (
    "SHARE WITH CAVEATS"
    if rolling_var_status == "PASS" and recent_tail_statuses == {"REVIEW LEVEL MET"}
    else "NEEDS REVISION"
)
illustrative_krd_evidence_reason = (
    "Recent VaR/ES tail-mass review level is not met"
    if illustrative_krd_evidence_status == "NEEDS REVISION"
    else "Statistical evidence gate met; external lineage and approval remain absent"
)
display(Markdown(
    f"### Illustrative fixed-KRD evidence gate: **{illustrative_krd_evidence_status}**\\n\\n"
    "This gate uses the rolling VaR status and every recent-window VaR/ES tail-mass "
    "status. It is separate from the repository-scope technical disposition and is "
    "necessary but not sufficient for any organizational use approval. "
    f"Observed reason: {illustrative_krd_evidence_reason}."
))
"""
    ),
    code(
        """
fig = plot_variance_reconciliation(risk.variance_reconciliation)
display(fig)
plt.close(fig)

scenario_columns = ["Factor", "Sigma multiple", "Linear P&L (USD)", "3M shock (bp)", "2Y shock (bp)", "10Y shock (bp)", "20Y shock (bp)"]
display(format_table(
    risk.pure_factor_scenarios[scenario_columns],
    formats={"Sigma multiple": "{:+.0f}", "Linear P&L (USD)": "${:,.0f}", **{column: "{:+.2f}" for column in scenario_columns if "shock" in column}},
    caption=(
        "Historical-covariance-scaled pure PCA factor shocks | not probability, return-period, "
        "or jointly plausible scenario statements"
    ),
))
display(format_table(
    risk.historical_scenarios.head(10),
    formats={"Linear P&L (USD)": "${:,.0f}", **{column: "{:+.2f}" for column in risk.historical_scenarios if "shock" in column}},
    caption="Worst exact contemporaneous shocks in the configured 520-week tail window",
))
display(format_table(
    risk.full_history_scenarios.head(10),
    formats={"Linear P&L (USD)": "${:,.0f}", **{column: "{:+.2f}" for column in risk.full_history_scenarios if "shock" in column}},
    caption="Worst exact contemporaneous shocks across the full available history",
))
"""
    ),
    md("## 7. Numerical challenge and technical disposition"),
    code(
        """
display(Markdown(
    f"### Repository-scope technical disposition: **{RESULT.validation.overall_assessment}**\\n\\n"
    "This is neither organizational model approval nor management acceptance of risk. "
    "The alternate SVD/algebra route is an implementation check, not an organizationally "
    "independent model validation function."
))
display(format_table(
    RESULT.validation.algebra_checks,
    formats={"Result": "{:.4g}", "Threshold": "{:.4g}"},
    caption="Alternate-route algebra/API challenge",
))
display(format_table(
    RESULT.validation.model_health,
    formats={"Result": "{:.4f}", "Threshold": "{:.4f}"},
    caption="Model-health thresholds and required actions",
))
"""
    ),
    md("## 8. Takeaways"),
    code(
        '''
residual_share = risk.variance_reconciliation.loc["Residual key-rate risk", "Variance share"]
selected = "not run" if RESULT.forecast is None else RESULT.forecast.selected_model
eligibility = pd.DataFrame(
    [
        ("Restricted historical research/education", "SHARE WITH CAVEATS", "Single frozen/latest-revised vintage"),
        ("Illustrative fixed-KRD linear mapping", illustrative_krd_evidence_status, illustrative_krd_evidence_reason),
        ("Fresh-feed non-limit factor/KRD monitor", "NOT ELIGIBLE", "External data, model, and use approval missing"),
        ("Factor limits / risk appetite", "NOT ELIGIBLE", "Historical factor instability breaches review levels"),
        ("Trading / hedge sizing", "NO-TRADE", "No costs, liquidity, lots, capacity, full revaluation, or OOS efficacy"),
        ("Official VaR/ES/FRTB/capital", "NOT ELIGIBLE", "Fixed illustrative KRD and non-official P&L"),
        ("Management forecast", "NOT ELIGIBLE", "No challenger edge; stale target; interval gates inconclusive"),
    ],
    columns=("Use", "Eligibility", "Blocking control"),
).set_index("Use")
display(format_table(eligibility, caption="Decision-use eligibility and hard stops"))
takeaways = f"""
1. **Use PCA as compression, not as truth.** The top three factors explain **{pca.explained_ratio[:3].sum():.2%}** of equal-tenor covariance, but factor identities rotate materially in some five-year windows.
2. **Retain key-rate residuals.** For the illustrative KRD, omitted-factor linear variance is **{residual_share:.2%}**. A different portfolio can have far higher residual concentration even when market-wide top-3 variance is high.
3. **No forecast edge is the correct possible answer.** The conservative fallback benchmark is **{selected}**; no PCA factor forecast is adopted without a statistically and economically defensible benchmark win.
4. **Separate structural and recency-weighted views.** Full-history PCA supplies a structural reference taxonomy subject to documented regime instability; EWMA is only recency-weighted relative to the frozen model date. Neither establishes stable factor limits or substitutes for stressed, liquidity-horizon, or full-revaluation risk.
5. **Escalate before any decision use.** Obtain a controlled point-in-time EOD vintage archive, correct desk curve mapping, official positions/sensitivities and P&L, nonlinear full revaluation, named owners, independent validation, approval authority, override expiry, and retention controls.
"""
display(Markdown(takeaways))
display(Markdown(
    f"**Figure lifecycle check:** open matplotlib figures after controlled display/close = "
    f"`{plt.get_fignums()}`"
))
'''
    ),
    md(
        """
## 9. Sources, governance & hard limitations

Primary sources:

- [Federal Reserve Board — H.15 Selected Interest Rates](https://www.federalreserve.gov/releases/h15/)
- [FRED — H.15 release](https://fred.stlouisfed.org/release?rid=18)
- [U.S. Treasury — Yield Curve Methodology](https://home.treasury.gov/policy-issues/financing-the-government/interest-rate-statistics/treasury-yield-curve-methodology)
- [Federal Reserve — Revised Guidance on Model Risk Management (2026)](https://www.federalreserve.gov/frrs/guidance/supervisory-guidance-on-model-risk-management.htm)
- [Federal Reserve SR 26-2 — Supervisory Guidance on Model Risk Management](https://www.federalreserve.gov/supervisionreg/srletters/SR2602.htm)
- [OCC Bulletin 2026-13 — Model Risk Management](https://www.occ.treas.gov/news-issuances/bulletins/2026/bulletin-2026-13.html)
- [Basel Framework — Market risk terminology](https://www.bis.org/basel_framework/chapter/MAR/10.htm)

Hard limitations:

- H.15 is latest-revised unless a frozen snapshot/vintage is used. Historical decision backtests require point-in-time archives.
- The sample spans Treasury's December 2021 curve-methodology change; the reported pre/post challenge cannot causally separate source-methodology effects from market regimes.
- Weekly covariance mixes normal holiday rolls; every actual calendar gap is retained and reported. No square-root-of-time conversion is applied.
- PCA is linear and backward-looking. Callable bonds, MBS, options, convexity, basis, volatility, funding, liquidity, and cross-curve risk require governed pricing and independently validated full revaluation.
- Historical linear VaR/ES and rolling exception tests for the fixed illustrative KRD are diagnostics only. They do not use actual historical positions or official P&L and do not implement FRTB RFET, PLA, NMRF, liquidity horizons, stressed scaling, ES calibration tests, or capital aggregation.
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "colab": {"provenance": [], "toc_visible": True},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.x"},
    },
)
nbf.validate(notebook)
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
nbf.write(notebook, OUTPUT)
print(OUTPUT)
