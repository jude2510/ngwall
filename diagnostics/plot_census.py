"""README figure: the census v2 resolution check of self-contacts.

Usage: python plot_census.py [DATA_DIR] [OUT_PNG]
Defaults: ../data/approach_census_v2 and ../figures/census_summary.png.
"""
import glob
import os
import json
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "data", "approach_census_v2")
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "figures", "census_summary.png")
VC, TCUT = 0.2598, 1.5
SURF, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3de"
COL = {4: "#2a78d6", 5: "#eb6834"}          # palette slots 1 (blue), 2 (orange)
LMAX = (4, 8, 12)

plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": INK2,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})

runs = [json.load(open(f)) for f in glob.glob(f"{DATA}/census_*.json")]
ev = {(n, l): [] for n in (4, 5) for l in LMAX}
per_run = {(n, l): [] for n in (4, 5) for l in LMAX}
for r in runs:
    e = [x for x in r["events"] if x["t"] < TCUT]
    ev[(r["nsub"], r["l_max"])] += e
    per_run[(r["nsub"], r["l_max"])].append(len(e))

fig, axs = plt.subplots(1, 3, figsize=(14, 4.3))
for ax in axs:
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

# (a) contacts per run: bars = mean over 8 seeds, dots = individual seeds
ax = axs[0]
x = np.arange(len(LMAX))
w = 0.36
for k, n in enumerate((4, 5)):
    vals = [np.mean(per_run[(n, l)]) for l in LMAX]
    xs = x + (k - 0.5) * (w + 0.02)
    ax.bar(xs, vals, w, color=COL[n], label=f"nsub = {n}", zorder=2)
    for xi, l in zip(xs, LMAX):
        pts = per_run[(n, l)]
        ax.plot(np.full(len(pts), xi), pts, "o", ms=3.2, mfc=SURF, mec=INK2, mew=0.8, zorder=3)
    for xi, v, l in zip(xs, vals, LMAX):
        top = max(v, max(per_run[(n, l)]))
        ax.text(xi, top + 18, f"{v:.0f}", ha="center", va="bottom", fontsize=9, color=INK)
ax.set_xticks(x, [f"l_max = {l}" for l in LMAX])
ax.set_ylabel(f"self-contacts per run (t < {TCUT:g})")
ax.set_title("Counts converge with resolution\n(a mesh artifact would grow ~4×)", fontsize=10.5,
             color=INK, loc="left")
ax.legend(frameon=False, loc="upper left")

# (b) along-surface separation of the two sheets at onset: median and IQR
ax = axs[1]
for k, n in enumerate((4, 5)):
    xs = x + (k - 0.5) * 0.22
    med = [np.median([e["geo_min"] for e in ev[(n, l)]]) for l in LMAX]
    lo = [np.percentile([e["geo_min"] for e in ev[(n, l)]], 25) for l in LMAX]
    hi = [np.percentile([e["geo_min"] for e in ev[(n, l)]], 75) for l in LMAX]
    ax.vlines(xs, lo, hi, color=COL[n], lw=2, zorder=2)
    ax.plot(xs, med, "o", ms=8, color=COL[n], mec=SURF, mew=2, zorder=3, label=f"nsub = {n}")
    for xi, m in zip(xs, med):
        dx, ha = (-0.07, "right") if n == 4 else (0.07, "left")
        ax.text(xi + dx, m, f"{m:.3f}", va="center", ha=ha, fontsize=8.5, color=INK)
ax.set_xticks(x, [f"l_max = {l}" for l in LMAX])
ax.set_xlim(-0.5, len(LMAX) - 0.4)
ax.set_ylim(0, None)
ax.set_ylabel("separation along the surface at onset")
ax.set_title("Separation halves with the mesh spacing\n→ every contact starts at a fold",
             fontsize=10.5, color=INK, loc="left")
ax.legend(frameon=False, loc="upper left")

# (c) closing speed, all short-range contacts pooled over l_max
ax = axs[2]
bins = np.linspace(0, 1, 41)
for n in (4, 5):
    u = np.array([e["u_med"] for l in LMAX for e in ev[(n, l)]])
    h, _ = np.histogram(u, bins=bins, density=True)
    ax.step(bins[:-1], h, where="post", color=COL[n], lw=2,
            label=f"nsub = {n}: {np.mean(u < VC):.0%} below v_c")
ax.axvline(VC, color=INK2, ls="--", lw=1.2)
ax.text(VC + 0.015, ax.get_ylim()[1] * 0.92, "φ⁴ kink–antikink\nv_c = 0.26", fontsize=8.5,
        color=INK2, va="top")
ax.set_xlabel("per-sheet approach speed u")
ax.set_ylabel("density")
ax.set_xlim(0, 1)
ax.set_title("Speed distribution unchanged;\nmost folds close slowly", fontsize=10.5,
             color=INK, loc="left")
ax.legend(frameon=False, loc="center right", fontsize=9)

fig.suptitle("Random walls meet themselves only at folds — converged between resolutions",
             fontsize=12.5, color=INK, x=0.01, ha="left", y=0.995)
fig.text(0.01, -0.06, "24 random walls (l_max 4/8/12 × seeds 0–7, ω = 0), pure Nambu-Goto (bare evolver, "
         "sheets pass through), each at nsub = 4 and 5; energy conserved to ~2e-6. Dots in (a): "
         "individual walls. Contacts between distant parts of a wall: none at nsub = 5.",
         fontsize=8.5, color=INK2, ha="left")
fig.tight_layout()
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
