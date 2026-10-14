"""
Explainability figure. Every value plotted here comes from explainability.py,
which computes it from the model. There are no synthetic fallbacks.
"""

import io
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from explainability import upsample_cam

FAKE_C = "#C0392B"
REAL_C = "#1E8449"
BAR_C = "#6C4AB6"
TXT_C = "#2C3E50"


def create_explainability_figure(raw_crops, cams, temporal, gat_edges, gat_stats,
                                 pred_class, probs, peak_frame=None,
                                 show_structural=False):
    """Plot computed CNN attribution, frame sensitivity and optional graph diagnostics."""
    if peak_frame is None:
        peak_frame = int(np.argmax(temporal))
    peak_frame = int(np.clip(peak_frame, 0, len(raw_crops) - 1))

    frame = raw_crops[peak_frame]
    h, w = frame.shape[:2]
    cam = upsample_cam(cams[peak_frame], (h, w))

    label = "PREDICTED FAKE" if pred_class == 1 else "PREDICTED REAL"
    colour = FAKE_C if pred_class == 1 else REAL_C

    rows = 3 if show_structural else 2
    heights = [1.15, 0.9, 1.0][:rows]
    fig = plt.figure(figsize=(16, 6.4 if rows == 2 else 12.5))
    gs = fig.add_gridspec(rows, 3, height_ratios=heights, hspace=0.46,
                          wspace=0.20)

    fig.suptitle("Attribution report", fontsize=17, fontweight="bold",
                 color=TXT_C, x=0.125, ha="left",
                 y=0.985 if rows == 2 else 0.965)
    fig.text(0.125, 0.935 if rows == 2 else 0.928,
             f"{label}   |   Real score {probs[0]:.3f}   "
             f"Fake score {probs[1]:.3f}   |   frame {peak_frame} of "
             f"{len(raw_crops)}",
             ha="left", fontsize=11.5, color=colour, fontweight="bold",
             family="monospace")

    # ---------------- spatial ----------------
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(frame)
    ax.set_title("Analysed region", fontsize=12, fontweight="bold", color=TXT_C)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(frame)
    ax.imshow(cam, cmap="jet", alpha=0.45, vmin=0, vmax=1)
    ax.set_title("CNN Grad-CAM", fontsize=12, fontweight="bold",
                 color=TXT_C)
    ax.axis("off")

    ax = fig.add_subplot(gs[0, 2])
    hm = ax.imshow(cam, cmap="jet", vmin=0, vmax=1)
    ax.set_title("Attribution intensity", fontsize=12, fontweight="bold",
                 color=TXT_C)
    ax.axis("off")
    cb = fig.colorbar(hm, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Normalised", fontsize=9.5)

    # ---------------- temporal ----------------
    ax = fig.add_subplot(gs[1, :])
    x = np.arange(len(temporal))
    ax.plot(x, temporal, lw=2.2, color=colour, marker="o", ms=4)
    ax.fill_between(x, temporal, alpha=0.16, color=colour)
    ax.axvline(peak_frame, ls="--", lw=1.8, color="#3A4F9B",
               label=f"Selected sample: {peak_frame}")
    ax.set_title("Visual frame sensitivity", fontsize=12, fontweight="bold",
                 color=TXT_C, loc="left")
    ax.set_xlabel("Sampled frame", fontsize=10.5)
    ax.set_ylabel("Relative sensitivity", fontsize=10.5)
    ax.set_xlim(0, max(len(temporal) - 1, 1))
    ax.set_ylim(0, max(float(np.max(temporal)) * 1.25, 1e-6))
    ax.grid(alpha=0.22, ls=":")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(fontsize=10, loc="upper left", frameon=False)

    # ---------------- structural (opt-in only) ----------------
    if show_structural:
        ax = fig.add_subplot(gs[2, :2])
        ranked = sorted(gat_edges, key=lambda e: abs(e["deviation"]),
                        reverse=True)[:8][::-1]
        vals = [e["deviation"] for e in ranked]
        cols = [FAKE_C if v > 0 else "#3A4F9B" for v in vals]
        ax.barh(range(len(vals)), vals, color=cols, alpha=0.85,
                edgecolor="white", lw=1.2)
        ax.set_yticks(range(len(ranked)))
        ax.set_yticklabels([e["label"] for e in ranked], fontsize=9.5)
        ax.axvline(0, color="#555555", lw=1.2)
        ax.set_xlabel("Attention minus uniform baseline (1/degree)", fontsize=10.5)
        ax.set_title("GAT attention diagnostic: how", fontsize=12,
                     fontweight="bold", color=TXT_C, loc="left")
        ax.grid(axis="x", alpha=0.22, ls=":")

        ax = fig.add_subplot(gs[2, 2])
        ax.axis("off")
        ax.text(0.02, 0.97,
                "GAT ATTENTION\n"
                "--------------------------\n"
                f"Edges          : {gat_stats['n_edges']}\n"
                f"Range          : [{gat_stats['att_min']:.4f}, "
                f"{gat_stats['att_max']:.4f}]\n"
                f"Mean           : {gat_stats['att_mean']:.4f}\n"
                f"Max |deviation|: {gat_stats['max_abs_deviation']:.4f}",
                transform=ax.transAxes, va="top", fontsize=10,
                family="monospace", color=TXT_C,
                bbox=dict(boxstyle="round,pad=0.7", facecolor="#F2F5F8",
                          edgecolor="#D5DEE7", lw=1.2))

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()
