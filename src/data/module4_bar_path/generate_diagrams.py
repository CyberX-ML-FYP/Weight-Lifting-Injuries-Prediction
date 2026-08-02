"""
Module 4 (bar_path) — generate static report diagrams:

  1. reports/figures/module4/diagrams/module4_architecture.png
     Pipeline architecture diagram (video -> extraction -> cleaning ->
     features -> RF+XGBoost ensemble -> risk report), styled as
     colour-coded stage panels with icon-labelled boxes and arrows,
     under the project banner.

  2. reports/figures/module4/diagrams/module4_charts_panel.png
     The two real analysis charts from the app (bar path trace, left/right
     wrist deviation) for one demo lift, side by side under a header --
     these are rendered with the SAME drawing code as
     app_module4_bar_path.py's render_bar_path_chart() /
     render_wrist_deviation_chart(), extracted here so both the live app
     and this static export stay visually identical and can't drift apart.

Run:
    python -m src.data.module4_bar_path.generate_diagrams
"""
from __future__ import annotations

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyBboxPatch

from .config import BarPathConfig
from .utils import setup_logger

logger = setup_logger(__name__)

DIAGRAM_DIRNAME = "diagrams"
DEFAULT_CHART_LIFT = "2good"

BANNER_TITLE = "Multi-View Mathematical Analysis and AI-Based Performance\n& Risk Assessment System for Clean & Jerk Weightlifting"
BANNER_SUBTITLE = "Module 4 — Bar Path, Multi-View Sync & ML Pipeline"
BANNER_TEAM = "Group 18 — Team CyberX  |  214147B Perera A.K.A.K.K.  ·  214188B Savindu R.H.  ·  214027H Bandara H.G.P.M.  ·  214189E Senarathna G.G.P.C.  |  Supervisor: Dr. Lochandaka Ranathunga"

# ── Shared banner ────────────────────────────────────────────────────────────


def _draw_banner(fig, y0: float, height: float) -> None:
    """Dark title band matching the reference poster's banner style."""
    banner_ax = fig.add_axes([0.0, y0, 1.0, height])
    banner_ax.set_xlim(0, 1)
    banner_ax.set_ylim(0, 1)
    banner_ax.axis("off")
    banner_ax.add_patch(
        FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0",
                        facecolor="#0f1e4d", edgecolor="none", zorder=0, transform=banner_ax.transAxes)
    )
    banner_ax.text(0.5, 0.72, BANNER_TITLE, ha="center", va="center", fontsize=15,
                    fontweight="bold", color="white", transform=banner_ax.transAxes, linespacing=1.4)
    banner_ax.text(0.5, 0.32, BANNER_SUBTITLE, ha="center", va="center", fontsize=11,
                    color="#93c5fd", transform=banner_ax.transAxes)
    banner_ax.text(0.5, 0.10, BANNER_TEAM, ha="center", va="center", fontsize=7.5,
                    color="#cbd5e1", transform=banner_ax.transAxes)


# ── Architecture diagram ─────────────────────────────────────────────────────

STAGE_BOX = dict(boxstyle="round,pad=0.4,rounding_size=0.08", linewidth=1.4)


def _draw_stage_box(ax, xy, w, h, label, icon, facecolor, edgecolor, textcolor="#111827", fontsize=9):
    x, y = xy
    box = FancyBboxPatch((x, y), w, h, facecolor=facecolor, edgecolor=edgecolor,
                          linewidth=1.4, boxstyle="round,pad=0.012,rounding_size=0.02",
                          transform=ax.transAxes, zorder=2)
    ax.add_patch(box)
    ax.text(x + w / 2, y + h / 2, f"{icon}  {label}", ha="center", va="center",
            fontsize=fontsize, color=textcolor, transform=ax.transAxes, zorder=3, wrap=True)


def _draw_panel(ax, xy, w, h, title, facecolor, edgecolor):
    x, y = xy
    panel = FancyBboxPatch((x, y), w, h, facecolor=facecolor, edgecolor=edgecolor,
                            linewidth=1.6, boxstyle="round,pad=0.01,rounding_size=0.015",
                            transform=ax.transAxes, zorder=0, alpha=0.55)
    ax.add_patch(panel)
    ax.text(x + 0.012, y + h - 0.035, title, ha="left", va="top", fontsize=9.5,
            fontweight="bold", color=edgecolor, transform=ax.transAxes, zorder=1)


def _draw_arrow(ax, start, end, color="#374151"):
    ax.annotate(
        "", xy=end, xytext=start, xycoords="axes fraction", textcoords="axes fraction",
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.6, mutation_scale=16,
                         path_effects=[pe.withStroke(linewidth=3, foreground="white")]),
        zorder=4,
    )


PAGE_BACKGROUND = "#f4f7ff"


def render_architecture_diagram(out_path):
    fig = plt.figure(figsize=(13, 8.5), facecolor=PAGE_BACKGROUND)
    banner_h = 0.16
    _draw_banner(fig, 1 - banner_h, banner_h)

    fig.text(0.5, 1 - banner_h - 0.045, "Bar Path Analysis & Injury-Risk Pipeline",
              ha="center", va="top", fontsize=13, fontweight="bold", color="#111827")

    ax = fig.add_axes([0.02, 0.03, 0.96, 1 - banner_h - 0.13])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor(PAGE_BACKGROUND)

    # Input
    _draw_stage_box(ax, (0.005, 0.42), 0.11, 0.16, "Front-view\nvideo input", "▶",
                     "#e0e7ff", "#4338ca")

    # Stage 1 panel — raw extraction
    _draw_panel(ax, (0.14, 0.10), 0.19, 0.80, "Pass 1 — Raw Extraction", "#fef3c7", "#b45309")
    _draw_stage_box(ax, (0.155, 0.62), 0.16, 0.14, "MediaPipe pose\nlandmarks (33pt)", "●",
                     "#fffbeb", "#b45309", fontsize=8)
    _draw_stage_box(ax, (0.155, 0.42), 0.16, 0.14, "select_lifter()\ncentre+close+low+cont.", "◎",
                     "#fffbeb", "#b45309", fontsize=8)
    _draw_stage_box(ax, (0.155, 0.22), 0.16, 0.14, "find_lift_window()\nwalk-in/out trim", "✂",
                     "#fffbeb", "#b45309", fontsize=8)

    # Stage 2 panel — cleaning
    _draw_panel(ax, (0.36, 0.10), 0.19, 0.80, "Pass 2 — Cleaning", "#dcfce7", "#15803d")
    _draw_stage_box(ax, (0.375, 0.62), 0.16, 0.14, "Lift-phase +\nvisibility filter", "▤",
                     "#f0fdf4", "#15803d", fontsize=8)
    _draw_stage_box(ax, (0.375, 0.42), 0.16, 0.14, "Wrist-swap fix +\noutlier removal", "⇄",
                     "#f0fdf4", "#15803d", fontsize=8)
    _draw_stage_box(ax, (0.375, 0.22), 0.16, 0.14, "Savitzky-Golay\nsmoothing", "∿",
                     "#f0fdf4", "#15803d", fontsize=8)

    # Stage 3 panel — features
    _draw_panel(ax, (0.58, 0.10), 0.17, 0.80, "Pass 3 — Features", "#fce7f3", "#be185d")
    _draw_stage_box(ax, (0.593, 0.55), 0.145, 0.20,
                     "7 per-lift features:\ndeviation, smoothness,\nvelocity, displacement,\njerk count", "▦",
                     "#fdf2f8", "#be185d", fontsize=7.5)
    _draw_stage_box(ax, (0.593, 0.25), 0.145, 0.20,
                     "T=150 resampled\n(x,y) sequence\n(for LSTM)", "↻",
                     "#fdf2f8", "#be185d", fontsize=7.5)

    # Stage 4 panel — ensemble
    _draw_panel(ax, (0.775, 0.10), 0.20, 0.80, "Model Ensemble", "#e0e7ff", "#4338ca")
    _draw_stage_box(ax, (0.79, 0.62), 0.17, 0.14, "Random Forest\n81% CV acc.", "▲",
                     "#eef2ff", "#4338ca", fontsize=8)
    _draw_stage_box(ax, (0.79, 0.42), 0.17, 0.14, "XGBoost\n79% CV acc.", "◆",
                     "#eef2ff", "#4338ca", fontsize=8)
    _draw_stage_box(ax, (0.79, 0.22), 0.17, 0.14, "Attention-LSTM\n(evaluated, excluded)", "◇",
                     "#f1f5f9", "#64748b", fontsize=8, textcolor="#64748b")

    # arrows between the four panels
    _draw_arrow(ax, (0.115, 0.5), (0.155, 0.5))
    _draw_arrow(ax, (0.315, 0.5), (0.375, 0.5))
    _draw_arrow(ax, (0.535, 0.5), (0.593, 0.5))
    _draw_arrow(ax, (0.738, 0.5), (0.79, 0.5))

    # Output — risk report, below the ensemble
    _draw_panel(ax, (0.775, -0.03), 0.20, 0.0, "", "none", "none")  # spacer no-op kept minimal
    out_ax_y = 0.02
    _draw_stage_box(ax, (0.79, out_ax_y), 0.17, 0.14,
                     "Injury-risk report\n(score + band + why)", "▣", "#fff7ed", "#c2410c", fontsize=8)
    _draw_arrow(ax, (0.875, 0.22), (0.875, 0.16))

    fig.savefig(out_path, dpi=160, facecolor=PAGE_BACKGROUND)
    plt.close(fig)
    logger.info("Saved architecture diagram to %s", out_path)


# ── Charts panel (reuses the app's exact drawing logic) ──────────────────────


def draw_bar_path_trace(ax, cleaned_df: pd.DataFrame) -> None:
    """Identical drawing logic to app_module4_bar_path.render_bar_path_chart,
    factored out so the static export and the live app can't visually drift
    apart -- kept here (not imported from the app) because the app module
    isn't meant to be imported as a library (it runs top-level Streamlit
    calls on import)."""
    x = cleaned_df["x"].to_numpy(dtype=float)
    y = cleaned_df["y"].to_numpy(dtype=float)
    progress = np.linspace(0.0, 1.0, len(x))

    ax.axvline(x[0], color="#9ca3af", linewidth=1, linestyle="--", zorder=1)

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, cmap="viridis", linewidth=2.5, zorder=2)
    lc.set_array(progress)
    ax.add_collection(lc)
    ax.figure.colorbar(lc, ax=ax, label="Progress through lift", shrink=0.6, pad=0.02)

    n_arrows = min(7, max(1, len(x) // 12))
    span = max(2, len(x) // (n_arrows + 1) // 3)
    if len(x) > span:
        arrow_idx = np.linspace(0, len(x) - 1 - span, n_arrows, dtype=int)
        for i in arrow_idx:
            j = i + span
            ax.annotate(
                "", xy=(x[j], y[j]), xytext=(x[i], y[i]),
                arrowprops=dict(arrowstyle="-|>", color="#1f2937", lw=1.5, mutation_scale=18,
                                 shrinkA=0, shrinkB=0,
                                 path_effects=[pe.withStroke(linewidth=3, foreground="white")]),
                zorder=4,
            )

    ax.scatter([x[0]], [y[0]], color="#16a34a", s=110, zorder=5, edgecolor="white", linewidth=2)
    ax.annotate("Start", (x[0], y[0]), textcoords="offset points", xytext=(10, 8),
                fontsize=11, fontweight="bold", color="#166534")
    ax.scatter([x[-1]], [y[-1]], color="#dc2626", s=110, zorder=5, edgecolor="white", linewidth=2, marker="s")
    ax.annotate("Finish", (x[-1], y[-1]), textcoords="offset points", xytext=(10, 8),
                fontsize=11, fontweight="bold", color="#991b1b")

    ax.set_xlim(max(0.0, x.min() - 0.08), min(1.0, x.max() + 0.08))
    ax.set_ylim(1.02, -0.02)
    ax.set_xlabel("Horizontal position (0.5 = frame centre)")
    ax.set_ylabel("Height (ground → overhead)")
    ax.set_title("Bar path trace", fontsize=11, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d1d5db")
    ax.tick_params(colors="#6b7280")
    ax.set_facecolor("#fafafa")


def draw_wrist_deviation(ax, cleaned_df: pd.DataFrame) -> None:
    """Identical drawing logic to app_module4_bar_path.render_wrist_deviation_chart."""
    df = cleaned_df.sort_values("frame_id").reset_index(drop=True)
    left_x = df["left_wrist_x"].to_numpy(dtype=float) / df["frame_width"].to_numpy(dtype=float)
    right_x = df["right_wrist_x"].to_numpy(dtype=float) / df["frame_width"].to_numpy(dtype=float)
    left_dev = left_x - left_x[0]
    right_dev = right_x - right_x[0]

    ax.axhline(0.0, color="#9ca3af", linewidth=1, linestyle="--", zorder=1)
    ax.plot(df["frame_id"], left_dev, color="#2563eb", linewidth=2, label="Left wrist", zorder=2)
    ax.plot(df["frame_id"], right_dev, color="#ea580c", linewidth=2, label="Right wrist", zorder=2)

    ax.set_xlabel("Frame")
    ax.set_ylabel("Horizontal deviation\n(fraction of frame width)")
    ax.set_title("Left vs right wrist deviation", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color("#d1d5db")
    ax.tick_params(colors="#6b7280")
    ax.set_facecolor("#fafafa")


def render_charts_panel(out_path, config: BarPathConfig, video_id: str = DEFAULT_CHART_LIFT):
    cleaned_path = config.processed_output_dir / f"{video_id}_cleaned.csv"
    if not cleaned_path.exists():
        raise FileNotFoundError(
            f"No cleaned coordinates for '{video_id}' at {cleaned_path} -- "
            "run raw_extractor.py + raw_cleaner.py for this video first."
        )
    cleaned_df = pd.read_csv(cleaned_path)

    fig = plt.figure(figsize=(13, 6.2), facecolor=PAGE_BACKGROUND)
    banner_h = 0.20
    _draw_banner(fig, 1 - banner_h, banner_h)

    fig.text(0.5, 1 - banner_h - 0.035, f"Example lift analysis — {video_id}", ha="center",
              fontsize=12, fontweight="bold", color="#111827")

    ax1 = fig.add_axes([0.06, 0.08, 0.34, 1 - banner_h - 0.18], facecolor=PAGE_BACKGROUND)
    ax2 = fig.add_axes([0.50, 0.14, 0.46, 1 - banner_h - 0.30], facecolor=PAGE_BACKGROUND)

    draw_bar_path_trace(ax1, cleaned_df)
    draw_wrist_deviation(ax2, cleaned_df)

    fig.savefig(out_path, dpi=160, facecolor=PAGE_BACKGROUND)
    plt.close(fig)
    logger.info("Saved charts panel (lift=%s) to %s", video_id, out_path)


def main() -> None:
    config = BarPathConfig()
    out_dir = config.root_dir / "reports" / "figures" / "module4" / DIAGRAM_DIRNAME
    out_dir.mkdir(parents=True, exist_ok=True)

    render_architecture_diagram(out_dir / "module4_architecture.png")
    render_charts_panel(out_dir / "module4_charts_panel.png", config)


if __name__ == "__main__":
    main()
