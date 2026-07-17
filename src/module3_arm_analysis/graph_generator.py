"""
Module 3 — Graph Generator
Reads the analysis CSV and produces research-quality charts.

Author: Pasindu (214027H)
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

from src.module3_arm_analysis.config import (
    SYMMETRY_THRESHOLD, LOCKOUT_THRESHOLD, OUTPUT_DIR,
)


def _style_axes(ax, title, ylabel, xlabel="Frame"):
    """Apply dark research-theme styling to a matplotlib axes."""
    ax.set_facecolor("#0f0f1a")
    ax.set_title(title, color="white", fontsize=13, fontweight="bold", pad=12)
    ax.set_xlabel(xlabel, color="#aaaaaa", fontsize=10)
    ax.set_ylabel(ylabel, color="#aaaaaa", fontsize=10)
    ax.tick_params(colors="#aaaaaa")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333355")
    ax.grid(axis="y", color="#222244", linewidth=0.5)


def generate_graphs(csv_path, output_dir=None):
    """
    Read the analysis CSV and generate 4 research charts.

    Args:
        csv_path:   Path to the per-frame analysis CSV.
        output_dir: Folder to save PNG outputs (defaults to reports/figures/module3).
    """
    if output_dir is None:
        output_dir = OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)

    df = pd.read_csv(csv_path)
    frames = df["frame"]
    print(f"Loaded {len(df)} frames from {csv_path}")

    # ── Graph 1 — Elbow Angles Over Time ──────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.plot(frames, df["left_elbow_angle"],  color="#00d4ff", linewidth=1.5, label="Left Elbow")
    ax.plot(frames, df["right_elbow_angle"], color="#ff6b6b", linewidth=1.5, label="Right Elbow")
    ax.axhline(y=LOCKOUT_THRESHOLD, color="#ffffff", linestyle="--",
               linewidth=0.8, alpha=0.5, label=f"Lockout Threshold ({LOCKOUT_THRESHOLD:.0f}°)")
    _style_axes(ax, "Elbow Angle Over Time — Upper Limb Analysis", "Angle (degrees)")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white", fontsize=9)
    ax.set_ylim(0, 200)
    plt.tight_layout()
    p1 = os.path.join(output_dir, "1_elbow_angles.png")
    plt.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p1}")

    # ── Graph 2 — Symmetry Difference Over Time ───────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f1a")
    colors = ["#ff4444" if v > SYMMETRY_THRESHOLD else "#00cc66" for v in df["symmetry_diff"]]
    ax.bar(frames, df["symmetry_diff"], color=colors, width=1.0, alpha=0.85)
    ax.axhline(y=SYMMETRY_THRESHOLD, color="#ff4444", linestyle="--",
               linewidth=1, alpha=0.85, label=f"Deviation Threshold ({SYMMETRY_THRESHOLD:.0f}°)")
    _style_axes(ax, "Arm Symmetry Difference — Deviation Detection", "Difference (degrees)")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white", fontsize=9)
    ax.set_ylim(0, max(df["symmetry_diff"].max() + 5, 30))
    plt.tight_layout()
    p2 = os.path.join(output_dir, "2_symmetry_diff.png")
    plt.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p2}")

    # ── Graph 3 — Shoulder Angles Over Time ───────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    fig.patch.set_facecolor("#0f0f1a")
    ax.plot(frames, df["left_shoulder_angle"],  color="#a78bfa", linewidth=1.5, label="Left Shoulder")
    ax.plot(frames, df["right_shoulder_angle"], color="#fbbf24", linewidth=1.5, label="Right Shoulder")
    _style_axes(ax, "Shoulder Angle Over Time", "Angle (degrees)")
    ax.legend(facecolor="#1a1a2e", edgecolor="#333355", labelcolor="white", fontsize=9)
    ax.set_ylim(0, 200)
    plt.tight_layout()
    p3 = os.path.join(output_dir, "3_shoulder_angles.png")
    plt.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p3}")

    # ── Graph 4 — Summary Statistics Bar Chart ────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0f0f1a")
    metrics = ["Avg L Elbow", "Avg R Elbow", "Avg L Shoulder", "Avg R Shoulder", "Avg Symmetry Diff"]
    values = [
        df["left_elbow_angle"].mean(),
        df["right_elbow_angle"].mean(),
        df["left_shoulder_angle"].mean(),
        df["right_shoulder_angle"].mean(),
        df["symmetry_diff"].mean(),
    ]
    colors = ["#00d4ff", "#ff6b6b", "#a78bfa", "#fbbf24", "#ff4444"]
    bars = ax.bar(metrics, values, color=colors, width=0.5, edgecolor="#0f0f1a", linewidth=1.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f"{val:.1f}°", ha="center", color="white", fontsize=10, fontweight="bold")
    _style_axes(ax, "Average Measurements Summary — Module 3", "Angle (degrees)", xlabel="")
    ax.tick_params(axis="x", labelsize=9)
    ax.set_ylim(0, max(values) + 20)
    plt.tight_layout()
    p4 = os.path.join(output_dir, "4_summary_stats.png")
    plt.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ {p4}")

    print(f"\nAll graphs saved to → {output_dir}")
    return [p1, p2, p3, p4]


if __name__ == "__main__":
    # Example usage
    CSV_INPUT = "data/processed/arm_analysis_sample.csv"
    generate_graphs(CSV_INPUT)