"""Regenerate every figure in the report from the result JSON files.

Palette and mark rules follow the dataviz reference palette (light mode).
Colour is assigned by job: categorical for condition identity, diverging
blue/red for signed quantities, muted ink for reference lines and chrome.

Run from the experiment root:  python report/make_figures.py
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "report", "figures")
os.makedirs(OUT, exist_ok=True)

# ---------------------------------------------------------------- palette
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"      # categorical 1 / diverging positive pole
ORANGE = "#eb6834"    # categorical 2
AQUA = "#1baf7a"      # categorical 3
RED = "#e34948"       # diverging negative pole
NEUTRAL = "#f0efec"   # diverging midpoint

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
    "font.size": 9,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK2,
    "axes.titlecolor": INK,
    "axes.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": INK2,
    "ytick.labelcolor": INK2,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "legend.frameon": False,
    "savefig.facecolor": SURFACE,
    "savefig.bbox": "tight",
    "savefig.dpi": 200,
})

W = 6.6  # full text width in inches for A4 with 18mm margins


def load(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as f:
        return json.load(f)


def tidy(ax, grid_axis="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis=grid_axis, zorder=0)
    ax.set_axisbelow(True)


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path)
    plt.close(fig)
    print(f"  wrote {name}")


# ============================================================ Figure 1
def fig1_design():
    """Design schematic: the 2x2 quadruplet and the three prompt conditions."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(W, 2.5),
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    # ---- left: the quadruplet
    axL.set_xlim(0, 10); axL.set_ylim(0, 10); axL.axis("off")
    axL.set_title("One scenario, four sentences", fontsize=10, loc="left",
                  color=INK, pad=8)

    rows = [
        ("q1", "Node-A", "severing",  "self harm",    -1),
        ("q2", "Node-A", "restoring", "self benefit",  1),
        ("q3", "Node-B", "severing",  "other harm",    0),
        ("q4", "Node-B", "restoring", "other benefit", 0),
    ]
    colour = {-1: RED, 1: BLUE, 0: NEUTRAL}
    for i, (tag, ent, verb, role, lab) in enumerate(rows):
        y = 7.6 - i * 1.85
        axL.add_patch(FancyBboxPatch((0.2, y), 7.4, 1.45,
                                     boxstyle="round,pad=0,rounding_size=0.12",
                                     facecolor=colour[lab], alpha=0.16 if lab else 1.0,
                                     edgecolor=colour[lab] if lab else AXIS, linewidth=1.0))
        axL.text(0.55, y + 0.95, tag, fontsize=8, color=MUTED, weight="bold")
        axL.text(1.5, y + 0.95, f"Target: {ent}", fontsize=8.5, color=INK)
        axL.text(1.5, y + 0.3, f"Action: {verb} fluid loop", fontsize=8.5, color=INK2)
        axL.text(8.0, y + 0.62, f"{lab:+d}" if lab else "0", fontsize=12,
                 color=colour[lab] if lab else INK2, weight="bold", ha="center",
                 va="center")
        axL.text(8.9, y + 0.62, role, fontsize=7.5, color=MUTED, va="center")
    axL.text(0.2, 0.15, "Every sentence shares one locked template.\n"
                        "q1 and q3 differ by a single token.",
             fontsize=7.5, color=MUTED, style="italic")

    # ---- right: the three conditions
    axR.set_xlim(0, 10); axR.set_ylim(0, 10); axR.axis("off")
    axR.set_title("Three prompt conditions", fontsize=10, loc="left", color=INK, pad=8)

    conds = [
        ("Ungrounded", "no prefix", "control", MUTED),
        ("SelfA", "\"you are Node-A\"", "self = A", BLUE),
        ("SelfB", "\"you are Node-B\"", "self = B", ORANGE),
    ]
    for i, (name, prefix, note, c) in enumerate(conds):
        y = 7.0 - i * 2.4
        axR.add_patch(Rectangle((0.3, y), 0.16, 1.7, facecolor=c, edgecolor="none"))
        axR.text(0.95, y + 1.28, name, fontsize=9.5, color=INK, weight="bold")
        axR.text(0.95, y + 0.68, prefix, fontsize=8, color=INK2)
        axR.text(0.95, y + 0.13, note, fontsize=7.5, color=MUTED)
    axR.text(0.3, 0.15, "Identical sentences in all three.\n"
                        "In SelfB the labels swap with the designation.",
             fontsize=7.5, color=MUTED, style="italic")

    save(fig, "fig1_design.png")


# ============================================================ Figure 2
def fig2_layer_sweep():
    """Probe R2 against depth, both conditions, with the sentiment ceiling."""
    d = load("probe_sweep.json")
    sweep = d["sweep"]
    ceiling = d["sentiment_ceiling"]
    x = np.arange(len(sweep["SelfA"]))

    fig, ax = plt.subplots(figsize=(W, 2.45))
    tidy(ax)

    ax.axhspan(0, ceiling, color=NEUTRAL, alpha=0.6, zorder=0)
    ax.axhline(ceiling, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=2)
    ax.text(1, ceiling + 0.018, f"sentiment ceiling {ceiling:.2f}",
            fontsize=7.5, color=INK2)

    ax.plot(x, sweep["SelfA"], color=BLUE, linewidth=2, zorder=4, label="SelfA")
    ax.plot(x, sweep["SelfB"], color=ORANGE, linewidth=2, zorder=3, label="SelfB")

    chosen = 34
    ax.axvline(chosen, color=AXIS, linewidth=1, zorder=1)
    ax.plot([chosen], [sweep["SelfA"][chosen]], "o", color=BLUE, markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2, zorder=5)
    ax.annotate(f"layer {chosen}\nR² {sweep['SelfA'][chosen]:.3f}",
                xy=(chosen, sweep["SelfA"][chosen]), xytext=(chosen + 2.5, 0.40),
                fontsize=8, color=INK, ha="left",
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9))

    ax.set_xlabel("hidden state index (0 = embeddings, 48 = final)")
    ax.set_ylabel("probe R², grouped CV")
    ax.set_xlim(-1, 49); ax.set_ylim(-0.03, 0.92)
    ax.legend(loc="lower right", fontsize=8.5, labelcolor=INK2)
    save(fig, "fig2_layer_sweep.png")


# ============================================================ Figure 3
def fig3_sign_flip():
    """The headline: swap cosine flips sign once the model is told who it is."""
    d = load("artifact_control.json")
    doms = ["Hardware", "Weights", "Runtime"]
    floor = d["artifact_floor"]
    grounded = d["grounded"]

    fig, ax = plt.subplots(figsize=(W, 2.6))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="y", zorder=0)
    ax.set_axisbelow(True)

    xs = np.arange(len(doms))
    bw = 0.36
    for i, (f, g) in enumerate(zip(floor, grounded)):
        ax.bar(i - bw / 2 - 0.01, f, bw, color=RED, zorder=3,
               label="Ungrounded (no prefix)" if i == 0 else None)
        ax.bar(i + bw / 2 + 0.01, g, bw, color=BLUE, zorder=3,
               label="Grounded (told who it is)" if i == 0 else None)
        ax.text(i - bw / 2 - 0.01, f - 0.055, f"{f:+.2f}", ha="center", va="top",
                fontsize=8.5, color=RED, weight="bold")
        ax.text(i + bw / 2 + 0.01, g + 0.045, f"{g:+.2f}", ha="center", va="bottom",
                fontsize=8.5, color=BLUE, weight="bold")

    ax.axhline(0, color=INK, linewidth=1.4, zorder=4)
    ax.set_xticks(xs); ax.set_xticklabels(doms)
    ax.set_ylabel("cosine between the two probe directions")
    ax.set_ylim(-0.85, 0.85)
    ax.set_yticks([-0.8, -0.4, 0, 0.4, 0.8])

    ax.text(2.66, -0.44, "point\nopposite", fontsize=8.5, color=RED, ha="center",
            va="center", weight="bold")
    ax.text(2.66, 0.44, "point\ntogether", fontsize=8.5, color=BLUE, ha="center",
            va="center", weight="bold")
    # Legend below the axis: inside the plot it collided with the bar labels.
    ax.legend(loc="upper center", bbox_to_anchor=(0.44, -0.10), ncol=2,
              fontsize=8.5, labelcolor=INK2, handlelength=1.2, columnspacing=1.6)
    ax.set_xlim(-0.62, 3.05)
    save(fig, "fig3_sign_flip.png")


# ============================================================ Figure 4
def fig4_heldout():
    """Zero-shot transfer to the held-out domain, mean prediction by role."""
    d = load("evaluate_report.json")
    roles = ["self_harm", "other_harm", "other_benefit", "self_benefit"]
    labels = ["harm to me", "harm to other", "benefit to other", "benefit to me"]
    targets = [-1, 0, 0, 1]

    fig, ax = plt.subplots(figsize=(W, 2.6))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="x", zorder=0)
    ax.set_axisbelow(True)

    # The two conditions land within about 0.05 of each other, so plotting them at
    # the same y hides one marker behind the other. Offset them within each band.
    ys = np.arange(len(roles))[::-1].astype(float)
    off = 0.19
    for y, role, t in zip(ys, roles, targets):
        ax.plot([t, t], [y - 0.34, y + 0.34], color=MUTED, linewidth=2.2, zorder=2)
        a = d["SelfA"]["mean_by_role"][role]
        b = d["SelfB"]["mean_by_role"][role]
        for val, yy, c, name in ((a, y + off, BLUE, "SelfA"), (b, y - off, ORANGE, "SelfB")):
            ax.plot([val], [yy], "o", color=c, markersize=8.5,
                    markeredgecolor=SURFACE, markeredgewidth=2, zorder=4,
                    label=name if y == ys[0] else None)
            ax.text(val + 0.075, yy, f"{val:+.2f}", fontsize=8, color=INK2,
                    va="center")
        ax.axhline(y - 0.5, color=GRID, linewidth=0.6, zorder=0)

    ax.axvline(0, color=AXIS, linewidth=1, zorder=1)
    ax.set_yticks(ys); ax.set_yticklabels(labels)
    ax.set_xlabel("mean predicted value on the held-out domain "
                  "(grey bar = training target)")
    ax.set_xlim(-1.18, 1.32)
    ax.set_ylim(-0.55, 3.55)
    # Above the axes: below it collided with the x-axis label.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
              fontsize=8.5, labelcolor=INK2, handlelength=1.0, columnspacing=1.8)
    save(fig, "fig4_heldout.png")


# ============================================================ Figure 5
def fig5_cosine_groups():
    """Every cosine comparison against its measured null and the ceiling."""
    p = load("probe_report.json")["probes"]["cosine_groups"]
    rel = load("reliability_report.json")
    ev = load("evaluate_report.json")
    ceiling = rel["ceiling_measured"]

    bars = [
        ("same domain,\nswapped self", p["swap_same_domain"]["mean"],
         p["swap_same_domain"]["null_abs_p95"]),
        ("cross domain,\nsame self", p["within_condition"]["mean"],
         p["within_condition"]["null_abs_p95"]),
        ("cross domain,\nswapped self", p["swap_cross_domain"]["mean"],
         p["swap_cross_domain"]["null_abs_p95"]),
        ("domain-averaged,\nswapped self", ev["cos_w_pure_SelfA_SelfB"], None),
    ]

    fig, ax = plt.subplots(figsize=(W, 2.4))
    tidy(ax)
    xs = np.arange(len(bars))

    ax.axhspan(0, max(b[2] for b in bars if b[2]), color=RED, alpha=0.10, zorder=1)
    ax.axhline(ceiling, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.3, zorder=2)
    ax.text(-0.42, ceiling + 0.02,
            f"measured reliability ceiling {ceiling:.3f}", fontsize=7.5, color=INK2)
    ax.text(-0.42, 0.095, "null band (shuffled labels)", fontsize=7.5, color=RED)

    for i, (_, val, _) in enumerate(bars):
        ax.bar(i, val, 0.52, color=BLUE, zorder=3)
        ax.text(i, val + 0.022, f"{val:.3f}", ha="center", fontsize=8.5,
                color=INK, weight="bold")
        ax.text(i, val - 0.05, f"{val/ceiling*100:.0f}% of\nceiling", ha="center",
                va="top", fontsize=7.5, color=SURFACE, weight="bold")

    ax.set_xticks(xs); ax.set_xticklabels([b[0] for b in bars], fontsize=8.5)
    ax.set_ylabel("cosine between probe directions")
    ax.set_ylim(0, 0.85)
    save(fig, "fig5_cosine_groups.png")


# ============================================================ Figure 6
def fig6_magnitude():
    """Activation magnitude grows about 1000x with depth."""
    d = load("activation_summary.json")["domains"]

    fig, ax = plt.subplots(figsize=(W, 2.15))
    tidy(ax)
    for i, (dom, v) in enumerate(d.items()):
        m = v["max_abs_per_state"]
        ax.plot(np.arange(len(m)), m, color=BLUE, linewidth=1.6,
                alpha=0.55, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("hidden state index")
    ax.set_ylabel("max |activation|")
    ax.set_xlim(-1, 49)

    first = list(d.values())[0]["max_abs_per_state"]
    ax.annotate(f"{first[0]:.2f}", xy=(0, first[0]), xytext=(2.5, 1.1),
                fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9))
    peak = int(np.argmax(first))
    ax.annotate(f"{first[peak]:.0f} at layer {peak}", xy=(peak, first[peak]),
                xytext=(peak - 17, 430), fontsize=8, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, linewidth=0.9))
    ax.text(0.5, 0.06,
            "all four domains overlap; log scale spans three orders of magnitude",
            transform=ax.transAxes, ha="center", fontsize=7.5, color=MUTED,
            style="italic")
    save(fig, "fig6_magnitude.png")


# ============================================================ Figure 7
def fig7_leace():
    """Sentiment decodability before and after erasure, held-out rows."""
    a = load("scrub_layer34.json")
    b = load("scrub_report.json")
    rows = []
    for src in (a, b):
        for layer in src["layers"]:
            for cond in ("SelfA", "SelfB"):
                r = src["results"][cond][str(layer)]
                rows.append((layer, cond, r["before"], r["after"]))
    rows.sort(key=lambda r: (r[0], r[1]))

    fig, ax = plt.subplots(figsize=(W, 2.6))
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.grid(True, axis="x", zorder=0)
    ax.set_axisbelow(True)

    ys = np.arange(len(rows))[::-1]
    for y, (layer, cond, before, after) in zip(ys, rows):
        ax.plot([after, before], [y, y], color=AXIS, linewidth=1.4, zorder=2)
        ax.plot([before], [y], "o", color=RED, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=4,
                label="before erasure" if y == ys[0] else None)
        ax.plot([after], [y], "o", color=BLUE, markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=1.8, zorder=4,
                label="after erasure" if y == ys[0] else None)

    ax.axvline(0.5, color=MUTED, linestyle=(0, (4, 3)), linewidth=1.2, zorder=1)
    ax.text(0.5, len(rows) - 0.2, " chance 0.50", fontsize=7.5, color=INK2)
    ax.set_yticks(ys)
    ax.set_yticklabels([f"layer {l}  {c}" for l, c, _, _ in rows], fontsize=8)
    ax.set_xlabel("accuracy of a sentiment probe on rows the eraser never saw")
    ax.set_xlim(0.42, 1.03)
    ax.legend(loc="lower left", fontsize=8.5, labelcolor=INK2)
    save(fig, "fig7_leace.png")


if __name__ == "__main__":
    print("generating figures ->", OUT)
    fig1_design()
    fig2_layer_sweep()
    fig3_sign_flip()
    fig4_heldout()
    fig5_cosine_groups()
    fig6_magnitude()
    fig7_leace()
    print("done")
