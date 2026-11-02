"""Build the system architecture diagram for the MAD-format report."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).resolve().parent / "figures" / "architecture.png"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 8.5,
        "axes.linewidth": 0.0,
    }
)

fig, ax = plt.subplots(figsize=(8.4, 3.9), dpi=300)
ax.set_xlim(0, 8.4)
ax.set_ylim(0, 3.9)
ax.axis("off")

COLORS = {
    "image": "#E8F1FA",
    "cnn": "#D6E6F5",
    "feature": "#EAF3E7",
    "attention": "#FBE7D7",
    "object": "#F3E8F6",
    "decoder": "#FFF2CC",
    "output": "#E8E8E8",
}
EDGE = "#4B5D6E"


def box(x: float, y: float, w: float, h: float, text: str, color: str, bold: bool = False) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.025,rounding_size=0.06",
        linewidth=0.9,
        edgecolor=EDGE,
        facecolor=color,
        zorder=2,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=8.2,
        fontweight="bold" if bold else "normal",
        color="#1E2A33",
        zorder=3,
        linespacing=1.18,
    )


def arrow(x1: float, y1: float, x2: float, y2: float, dashed: bool = False, color: str = "#44515C") -> None:
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.9,
            color=color,
            linestyle="--" if dashed else "-",
            shrinkA=0,
            shrinkB=2,
            zorder=1,
        )
    )


# Top row: visual pipeline.
box(0.12, 2.65, 0.95, 0.62, "COCO\nimage", COLORS["image"], bold=True)
box(1.42, 2.65, 1.10, 0.62, "ResNet-18\n(frozen)", COLORS["cnn"], bold=True)
box(2.88, 2.80, 1.22, 0.52, "spatial grid\n7 × 7 × 512", COLORS["feature"])
box(4.48, 2.65, 1.15, 0.62, "Bahdanau\nattention", COLORS["attention"], bold=True)
box(6.00, 2.65, 0.88, 0.62, "LSTM\ndecoder", COLORS["decoder"], bold=True)
box(7.25, 2.65, 1.00, 0.62, "caption", COLORS["output"], bold=True)

arrow(1.07, 2.96, 1.40, 2.96)
arrow(2.52, 2.96, 2.86, 3.04)
arrow(4.10, 3.06, 4.46, 2.96)
arrow(5.63, 2.96, 5.98, 2.96)
arrow(6.88, 2.96, 7.23, 2.96)

# Detector / object-label branch.
box(0.12, 0.78, 1.10, 0.62, "Faster R-CNN\ndetector", COLORS["object"], bold=True)
box(1.52, 0.78, 1.10, 0.62, "object labels\n+ scores", COLORS["object"])
box(2.92, 0.78, 1.10, 0.62, "object\nembedding", COLORS["object"], bold=True)
arrow(1.22, 1.09, 1.50, 1.09)
arrow(2.62, 1.09, 2.90, 1.09)
arrow(4.02, 1.09, 6.30, 2.61, color="#7A5B85")

# Global feature initialisation.
box(2.88, 1.88, 1.22, 0.52, "global vector\n512-d", COLORS["feature"])
arrow(2.52, 2.70, 3.10, 2.42, dashed=True)
arrow(4.10, 2.14, 6.06, 2.62, dashed=True, color="#5B7F63")

# Evaluation strip.
box(5.55, 0.10, 2.65, 0.42, "BLEU-1–4  ·  CIDEr-D  ·  CHAIR", "#F7F7F7")
arrow(7.72, 2.63, 7.72, 0.54, dashed=True, color="#666666")

# Small explanatory labels.
ax.text(3.49, 3.47, "spatial features", ha="center", va="center", fontsize=7.2, color="#4F5F6D")
ax.text(5.81, 2.79, "context", ha="center", va="center", fontsize=7.0, color="#4F5F6D", style="italic", bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
ax.text(2.28, 2.34, "initial state", ha="center", va="center", fontsize=7.0, color="#4F5F6D", style="italic")
ax.text(5.18, 1.58, "object evidence", ha="center", va="center", fontsize=7.0, color="#6B556F", style="italic", bbox=dict(facecolor="white", edgecolor="none", pad=0.4))
ax.text(0.24, 0.35, "Auxiliary object evidence", fontsize=7.3, color="#5D4A66", fontweight="bold")
ax.text(0.24, 3.58, "Visual pipeline", fontsize=7.3, color="#334E66", fontweight="bold")

fig.savefig(OUT, bbox_inches="tight", facecolor="white", pad_inches=0.02)
plt.close(fig)
print(OUT)
