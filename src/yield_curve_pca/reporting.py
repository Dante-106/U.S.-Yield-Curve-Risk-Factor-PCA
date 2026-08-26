"""Deterministic tables and static, notebook-oriented risk figures."""

from __future__ import annotations

from hashlib import sha1

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import PercentFormatter

from .forecast import ForecastResult
from .pca import PCAFit

INK = "#172433"
BLUE = "#2F6B9A"
GOLD = "#C28E2C"
ORANGE = "#D97941"
OLIVE = "#748650"
PINK = "#B95D79"
GRAY = "#6E7D8B"
LIGHT = "#DCE5EC"
FACTOR_COLORS = (BLUE, GOLD, ORANGE)


PLOT_STYLE = {
    "figure.figsize": (11.0, 5.2),
    "figure.dpi": 120,
    "axes.facecolor": "white",
    "figure.facecolor": "white",
    "axes.edgecolor": "#C7D1DA",
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "axes.titleweight": "bold",
    "axes.titlesize": 12,
    "xtick.color": GRAY,
    "ytick.color": GRAY,
    "grid.color": "#DDE4EA",
    "grid.alpha": 0.75,
    "font.size": 10,
    "legend.frameon": False,
}


def _set_title_and_context(ax: Axes, title: str, context: str) -> None:
    """Render a consistent two-line chart heading without visual collision."""

    ax.set_title(title, loc="left", pad=24)
    ax.text(
        0,
        1.01,
        context,
        transform=ax.transAxes,
        color=GRAY,
        fontsize=9,
        verticalalignment="bottom",
    )


def format_table(
    frame: pd.DataFrame,
    *,
    formats: dict[str, str] | str | None = None,
    caption: str = "",
) -> object:
    """Return a deterministic Styler when Jinja2 is present, else the frame."""

    try:
        styler = frame.style
    except (AttributeError, ImportError):
        return frame
    if formats is not None:
        styler = styler.format(formats, na_rep="—")
    if caption:
        styler = styler.set_caption(caption)
    token = sha1((caption + "|" + "|".join(map(str, frame.columns))).encode()).hexdigest()[:12]
    styler = styler.set_uuid(token)
    return styler.set_table_styles(
        [
            {
                "selector": "caption",
                "props": [
                    ("font-size", "13px"),
                    ("font-weight", "700"),
                    ("color", INK),
                    ("text-align", "left"),
                    ("padding", "8px 0"),
                ],
            },
            {
                "selector": "th",
                "props": [
                    ("background-color", INK),
                    ("color", "white"),
                    ("font-weight", "600"),
                    ("padding", "6px"),
                ],
            },
            {
                "selector": "td",
                "props": [("padding", "6px"), ("border-bottom", "1px solid #E5EAF0")],
            },
        ]
    )


def plot_curve_history(weekly_yields_pct: pd.DataFrame) -> Figure:
    selected = [tenor for tenor in ("3M", "2Y", "5Y", "10Y", "20Y") if tenor in weekly_yields_pct]
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(12.5, 5.0))
        colors = (GRAY, GOLD, BLUE, ORANGE, PINK)
        for tenor, color in zip(selected, colors, strict=False):
            ax.plot(weekly_yields_pct.index, weekly_yields_pct[tenor], label=tenor, color=color, lw=1.25)
        _set_title_and_context(
            ax,
            "Synchronized U.S. Treasury constant-maturity yields",
            f"Percent per annum | {weekly_yields_pct.index.min().date()} to {weekly_yields_pct.index.max().date()}",
        )
        ax.set_ylabel("Yield (%)")
        ax.set_xlabel("Actual observation date")
        ax.grid(True)
        ax.legend(ncol=len(selected), loc="upper center")
        fig.tight_layout()
    return fig


def plot_pca_structure(fit: PCAFit, retained_factors: int = 3) -> Figure:
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0), gridspec_kw={"width_ratios": (1.35, 1.0)})
        loadings = fit.loading_table(retained_factors)
        for column, color in zip(loadings, FACTOR_COLORS, strict=False):
            axes[0].plot(loadings.index, loadings[column], marker="o", lw=2.0, color=color, label=column)
        axes[0].axhline(0, color=GRAY, lw=0.8)
        _set_title_and_context(
            axes[0],
            "Unit-norm PCA eigenvectors",
            f"{'Correlation' if fit.standardize else 'Covariance'} basis | score units: {fit.score_unit}",
        )
        axes[0].set_ylabel("Eigenvector weight")
        axes[0].set_xlabel("Maturity")
        axes[0].grid(True)
        axes[0].legend()

        count = min(6, len(fit.eigenvalues))
        x = np.arange(count)
        shares = fit.explained_ratio[:count] * 100.0
        bars = axes[1].bar(
            x, shares, color=[*FACTOR_COLORS[: min(3, count)], *([LIGHT] * max(0, count - 3))], edgecolor=INK
        )
        axes[1].set_xticks(x, [f"PC{i + 1}" for i in range(count)])
        axes[1].set_ylim(0, max(85.0, shares.max() * 1.18))
        _set_title_and_context(
            axes[1],
            "Explained variance by component",
            f"Top-{retained_factors} cumulative: {fit.explained_ratio[:retained_factors].sum():.1%}",
        )
        axes[1].set_ylabel("Variance share")
        axes[1].yaxis.set_major_formatter(PercentFormatter())
        axes[1].grid(True, axis="y")
        axes[1].bar_label(bars, labels=[f"{value:.1f}%" for value in shares], padding=3, fontsize=8)
        fig.tight_layout()
    return fig


def plot_rolling_stability(
    rolling: pd.DataFrame,
    retained_factors: int = 3,
    *,
    loading_cosine_review_level: float = 0.80,
    variance_reference_level: float | None = None,
) -> Figure:
    cosine_columns = [column for column in rolling if column.endswith(" cosine")]
    with plt.rc_context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 1, figsize=(12.5, 7.5), sharex=True)
        axes[0].plot(
            rolling.index,
            rolling[f"Top-{retained_factors} variance"] * 100.0,
            color=BLUE,
            lw=1.8,
            marker="o",
            ms=3,
        )
        if variance_reference_level is not None:
            axes[0].axhline(
                variance_reference_level * 100.0,
                color=GRAY,
                ls="--",
                lw=1.0,
                label=f"{variance_reference_level:.0%} reference",
            )
        axes[0].set_title("Rolling retained-factor variance")
        axes[0].set_ylabel("Variance share")
        axes[0].yaxis.set_major_formatter(PercentFormatter())
        axes[0].grid(True)
        if variance_reference_level is not None:
            axes[0].legend()

        for column, color in zip(cosine_columns, FACTOR_COLORS, strict=False):
            axes[1].plot(
                rolling.index, rolling[column], label=column.replace(" cosine", ""), color=color, lw=1.5
            )
        axes[1].axhline(
            loading_cosine_review_level,
            color=GRAY,
            ls="--",
            lw=1.0,
            label=f"{loading_cosine_review_level:.2f} review level",
        )
        _set_title_and_context(
            axes[1],
            "Aligned loading cosine to the full-sample reference",
            f"Maximum principal angle observed: {rolling['Maximum principal angle (deg)'].max():.1f}°",
        )
        axes[1].set_ylabel("Cosine similarity")
        axes[1].set_xlabel("Rolling-window end date")
        axes[1].set_ylim(0, 1.05)
        axes[1].grid(True)
        axes[1].legend(ncol=4, loc="lower center")
        fig.tight_layout()
    return fig


def plot_forecast_rmse(result: ForecastResult) -> Figure:
    comparison = result.model_comparison.sort_values("Selection curve RMSE (bp)", ascending=True)
    colors = [BLUE if model == result.selected_model else LIGHT for model in comparison.index]
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10.5, 4.5))
        bars = ax.barh(
            comparison.index,
            comparison["Selection curve RMSE (bp)"],
            color=colors,
            edgecolor=INK,
        )
        ax.set_xlim(0, comparison["Selection curve RMSE (bp)"].max() * 1.15)
        _set_title_and_context(
            ax,
            "One-week-ahead model-selection curve RMSE",
            f"Expanding origin | Clark–West selection + untouched confirmation: {result.selected_model}",
        )
        ax.set_xlabel("Selection-sample RMSE (bp across tenors and forecasts)")
        ax.grid(True, axis="x")
        ax.bar_label(
            bars,
            labels=[f"{value:.2f}" for value in comparison["Selection curve RMSE (bp)"]],
            padding=3,
        )
        fig.tight_layout()
    return fig


def plot_variance_reconciliation(variance_table: pd.DataFrame) -> Figure:
    detail = variance_table.drop(index="Total")
    with plt.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(10.5, 4.5))
        bars = ax.barh(
            detail.index[::-1],
            detail["Variance share"][::-1] * 100.0,
            color=[*FACTOR_COLORS, LIGHT][: len(detail)][::-1],
            edgecolor=INK,
        )
        ax.set_xlim(0, max(5.0, detail["Variance share"].max() * 115.0))
        _set_title_and_context(
            ax,
            "Illustrative portfolio linear variance by PCA risk source",
            "Key-rate DV01 mapping | residual risk retained",
        )
        ax.set_xlabel("Share of modeled linear weekly P&L variance")
        ax.xaxis.set_major_formatter(PercentFormatter())
        ax.grid(True, axis="x")
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in detail["Variance share"][::-1]], padding=3)
        fig.tight_layout()
    return fig
