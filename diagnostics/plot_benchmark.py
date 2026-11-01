"""README figure: bare vs production evolver on the prolate collapse of
arXiv:2605.22974, against eikonal ray tracing.

Needs the two benchmark runs (from diagnostics/):
    python bench_ellipsoid.py 4 bare 1.5 0.004 OUTDIR
    python bench_ellipsoid.py 4 prod 1.5 0.004 OUTDIR
Usage: python plot_benchmark.py BARE.npz PROD.npz [OUT_PNG]
Default output: ../figures/benchmark_comparison.png.
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BARE, PROD = sys.argv[1], sys.argv[2]
OUT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "benchmark_comparison.png")
SURF, INK, INK2, GRID, BLUE = "#fcfcfb", "#0b0b0b", "#52514e", "#c9c8c2", "#2a78d6"
plt.rcParams.update({"font.size": 9.5, "text.color": INK, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "axes.edgecolor": GRID,
                     "figure.facecolor": SURF, "axes.facecolor": SURF})


def load(p):
    d = np.load(p, allow_pickle=True)
    return d, {c: d["rows"][:, i] for i, c in enumerate(d["cols"])}


def first_cross(t, y, level):
    i = np.nonzero(y >= level)[0][0]
    return np.interp(level, [y[i - 1], y[i]], [t[i - 1], t[i]])


def events(R):
    t = R["t"]
    t099 = first_cross(t, R["vpole"], 0.99)
    C, A = np.interp(t099, t, R["zpole"]), np.interp(t099, t, R["rho_mid"])
    fold = t[np.nonzero(R["nfold"] > 0)[0][0]]
    return dict(t099=t099, C=C, A=A, fold=fold, side=t[np.nanargmin(R["rho_mid"])],
                caps=t[np.argmin(R["zpole"] - R["zbot"])], dE=np.abs(R["dE"]).max(),
                nv=int(R["nv"][-1]))


def slice_y0(X, F):
    y = X[:, 1]
    out = []
    for tri in F:
        pts = [X[a] + y[a] / (y[a] - y[b]) * (X[b] - X[a])
               for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
               if (y[a] > 0) != (y[b] > 0)]
        if len(pts) == 2:
            out.append(pts)
    return np.array(out)


db, Rb = load(BARE)
dp, Rp = load(PROD)
eb, ep = events(Rb), events(Rp)
A, C, t0 = eb["A"], eb["C"], eb["t099"]
th = np.linspace(0, 2 * np.pi, 800)
ex, ez = A * np.sin(th), C * np.cos(th)
nx, nz = -C * np.sin(th), -A * np.cos(th)
nn = np.hypot(nx, nz)
idx = [1, 3, 4, 5, 7, 8]                     # snapshot times 0.44 ... 1.19

fig, axs = plt.subplots(2, len(idx), figsize=(15, 6.2), sharex=True, sharey=True)
for row, (d, name) in enumerate(((db, "bare evolver"), (dp, "production settings"))):
    for col, i in enumerate(idx):
        ax = axs[row, col]
        arr = d[f"snapX_{i}"]
        tt, X = arr[0], arr[1:].reshape(-1, 3)
        ax.plot(ex + (tt - t0) * nx / nn, ez + (tt - t0) * nz / nn, "--", color=BLUE, lw=1.4)
        for sg in slice_y0(X, d[f"snapF_{i}"]):
            ax.plot(sg[:, 0], sg[:, 2], "-", color=INK, lw=1.1)
        ax.set_aspect("equal")
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.8, 0.8)
        ax.tick_params(length=2)
        if row == 0:
            ax.set_title(f"t = {tt:.2f}   (paper t = {tt * 80:.0f})", fontsize=9.5, color=INK)
        if col == 0:
            ax.set_ylabel(name, fontsize=10.5, color=INK)
axs[0, 0].plot([], [], "-", color=INK, lw=1.1, label="our mesh (y = 0 slice)")
axs[0, 0].plot([], [], "--", color=BLUE, lw=1.4, label="eikonal ray tracing (paper's approximation)")
fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper right", ncol=2, frameon=False,
           bbox_to_anchor=(0.995, 1.0))
fig.suptitle("Prolate wall released from rest (arXiv:2605.22974, lengths scaled by 1/80)",
             fontsize=12.5, color=INK, x=0.01, ha="left", y=0.995)

rows = [("pole reaches v = 0.99", "0.31", f"{eb['t099']:.2f}", f"{ep['t099']:.2f}"),
        ("pole swallowtail forms", "≈ 0.50", f"{eb['fold']:.2f}", f"{ep['fold']:.2f}"),
        ("sides cross", "≈ 0.75", f"{eb['side']:.2f}", f"{ep['side']:.2f}"),
        ("caps cross", "≈ 1.25", f"{eb['caps']:.2f}", f"{ep['caps']:.2f}"),
        ("max energy error |ΔE/E|", "—", f"{eb['dE']:.0e}", f"{ep['dE']:.1f}"),
        ("vertices at the end", "—", f"{eb['nv']}", f"{ep['nv']}")]
tab = fig.add_axes([0.08, -0.29, 0.55, 0.30])
tab.axis("off")
t = tab.table(cellText=[r[1:] for r in rows], rowLabels=[r[0] for r in rows],
              colLabels=["paper", "bare evolver", "production settings"], loc="center",
              cellLoc="center", edges="horizontal")
t.auto_set_font_size(False)
t.set_fontsize(9.5)
t.scale(1, 1.35)
for (r, c), cell in t.get_celld().items():
    cell.set_edgecolor(GRID)
    cell.get_text().set_color(INK if r > 0 else INK2)
fig.text(0.66, -0.14, "The bare evolver (no smoothing or coarsening, γ-weighted time step)\n"
         "follows the paper through the swallowtails, the sides crossing and\n"
         "the caps crossing, conserving energy to ~1e-6. The production\n"
         "settings tear the cusp regions apart; energy grows ~9×.",
         fontsize=9.5, color=INK2, va="center")
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(OUT, dpi=150, bbox_inches="tight")
print("wrote", OUT)
