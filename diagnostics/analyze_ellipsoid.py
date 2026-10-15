"""Compare bench_ellipsoid.py runs with arXiv:2605.22974 (paper times / 80).

Usage: python analyze_ellipsoid.py RUN.npz [RUN2.npz ...]
Prints event times against the paper and writes RUN.png: y=0 cross-sections
of the mesh against the eikonal (ray-tracing) prediction started from the
run's own state when the pole reaches v = 0.99 (cf. the paper's Fig. 3).
"""
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAPER = dict(t099=25 / 80, C099=61.8 / 80, A099=30.9 / 80, t_pole=40 / 80,
             t_side=60 / 80, t_caps=100 / 80)


def first_cross(t, y, level):
    i = np.nonzero(y >= level)[0]
    if not len(i):
        return np.nan
    i = i[0]
    return t[i] if i == 0 else np.interp(level, [y[i - 1], y[i]], [t[i - 1], t[i]])


def slice_y0(X, F):
    """Segments where the mesh crosses the plane y = 0."""
    y = X[:, 1]
    segs = []
    for tri in F:
        pts = [X[a] + y[a] / (y[a] - y[b]) * (X[b] - X[a])
               for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0]))
               if (y[a] > 0) != (y[b] > 0)]
        if len(pts) == 2:
            segs.append(pts)
    return np.array(segs)


for path in sys.argv[1:]:
    d = np.load(path, allow_pickle=True)
    R = {c: d["rows"][:, i] for i, c in enumerate(d["cols"])}
    t = R["t"]
    t099 = first_cross(t, R["vpole"], 0.99)
    C = np.interp(t099, t, R["zpole"])
    A = np.interp(t099, t, R["rho_mid"])
    zext = R["zpole"] - R["zbot"]
    fold = np.nonzero(R["nfold"] > 0)[0]
    print(f"\n=== {path}: max |dE/E| = {np.abs(R['dE']).max():.1e}, final verts {int(R['nv'][-1])}")
    print(f"  pole speed 0.99 at t = {t099:.4f}  (paper {PAPER['t099']:.4f})")
    print(f"  shape then {C:.4f} x {A:.4f}  (paper {PAPER['C099']:.4f} x {PAPER['A099']:.4f})")
    print(f"  first fold (swallowtail) t = {t[fold[0]] if len(fold) else np.nan:.4f};"
          f" eikonal from own t0: {t099 + A**2 / C:.4f}  (paper ~{PAPER['t_pole']:.3f})")
    print(f"  sides cross t = {t[np.nanargmin(R['rho_mid'])]:.4f}  (paper ~{PAPER['t_side']:.3f})")
    print(f"  caps cross t = {t[np.argmin(zext)]:.4f}  (paper ~{PAPER['t_caps']:.3f})")

    th = np.linspace(0, 2 * np.pi, 800)
    ex, ez = A * np.sin(th), C * np.cos(th)
    nx, nz = -C * np.sin(th), -A * np.cos(th)       # inward normal of the ellipse
    nn = np.hypot(nx, nz)
    fig, axs = plt.subplots(2, 5, figsize=(15, 6.5), sharex=True, sharey=True)
    for ax, i in zip(axs.ravel(), range(1, 11)):
        if f"snapX_{i}" not in d:
            ax.axis("off")
            continue
        arr = d[f"snapX_{i}"]
        tt, X = arr[0], arr[1:].reshape(-1, 3)
        for sg in slice_y0(X, d[f"snapF_{i}"]):
            ax.plot(sg[:, 0], sg[:, 2], "k-", lw=0.8)
        tau = tt - t099
        ax.plot(ex + tau * nx / nn, ez + tau * nz / nn, "--", color="tab:blue", lw=1)
        ax.plot(ex, ez, ":", color="tab:orange", lw=0.8)
        ax.set_title(f"t = {tt:.3f}  (paper t = {tt * 80:.0f})", fontsize=9)
        ax.set_aspect("equal")
        ax.set_xlim(-0.6, 0.6)
        ax.set_ylim(-0.85, 0.85)
    fig.suptitle("mesh y=0 slice (black) vs eikonal ray tracing from the run's own "
                 "v=0.99 state (blue dashed); orange dotted = that state", fontsize=10)
    fig.tight_layout()
    fig.savefig(path.replace(".npz", ".png"), dpi=110)
