"""Benchmark: prolate wall released from rest vs Blanco-Pillado et al.,
"Cuspidal Singularities in Collapsing Domain Walls" (arXiv:2605.22974).

Their case a=40 (equatorial), c=80 (polar) is scaled by 1/80 to a=0.5, c=1;
expected times (paper time / 80): pole speed 0.99 at 0.3125 with semi-axes
(0.7725, 0.386); pole swallowtail ~0.50; sides cross ~0.75; caps cross ~1.25;
ring cuspidal edges meet on the equator ~1.375.

Pure Nambu-Goto, no contact detection or surgery. MODE "bare" uses the bare
evolver (no smoothing or coarsening, gamma-weighted dt); MODE "prod" uses the
production cascade settings (cfl 0.2, smoothing every 5 steps, coarsening
every 20 at lmin = 0.1 * median edge).
Usage: python bench_ellipsoid.py NSUB MODE [T_MAX] [DT_CAP] [OUTDIR]
"""
import os
import sys
import time
import numpy as np
from ngtools import crossing_detail, fold_edges, gamma_dt
from membrane import Membrane, icosphere

nsub, mode = int(sys.argv[1]), sys.argv[2]
t_max = float(sys.argv[3]) if len(sys.argv) > 3 else 1.5
cap = float(sys.argv[4]) if len(sys.argv) > 4 else 0.004
outdir = sys.argv[5] if len(sys.argv) > 5 else "."
assert mode in ("bare", "prod")
a, c = 0.5, 1.0

# rotate the icosphere so a vertex sits exactly on each pole
V, F = icosphere(nsub)
v = np.array([0.0, 1.0, (1 + 5 ** 0.5) / 2]) / np.sqrt(1 + ((1 + 5 ** 0.5) / 2) ** 2)
k = np.cross(v, [0, 0, 1.0])
s = np.linalg.norm(k)
k /= s
K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
V = V @ (np.eye(3) + s * K + (1 - v[2]) * K @ K).T
X = V * [a, a, c]
mem = Membrane(X, F)
lmin = 0.1 * np.median(np.linalg.norm(X[F[:, 0]] - X[F[:, 1]], axis=1))
E0 = mem.energy()
snap_t = [0.3125, 0.4375, 0.5, 0.5625, 0.6875, 0.8125, 0.9375, 1.0625, 1.1875,
          1.3125, 1.4375]
snaps, rows = {}, []
n = 0
t0 = time.time()
while mem.t < t_max:
    if n % 4 == 0:
        Xc, Fc = mem.X, mem.faces
        sp = np.linalg.norm(mem.velocity(Xc, mem.p), axis=1)
        top, bot = np.argmax(Xc[:, 2]), np.argmin(Xc[:, 2])
        band = np.abs(Xc[:, 2]) < 0.03
        rho = np.hypot(Xc[:, 0], Xc[:, 1])
        rows.append((mem.t, mem.energy() / E0 - 1, sp.max(), sp[top], Xc[top, 2],
                     Xc[bot, 2], rho[band].max() if band.any() else np.nan,
                     len(fold_edges(Xc, Fc)), len(crossing_detail(Xc, Fc)["tri"]),
                     len(Xc)))
    for ts in snap_t:
        if ts not in snaps and mem.t >= ts:
            snaps[ts] = (mem.t, mem.X.copy(), mem.faces.copy())
    if mode == "bare":
        mem.step(gamma_dt(mem, cap=cap))
        n += 1
    else:
        mem.step(0.2 * mem.min_edge(mem.X))
        n += 1
        if n % 5 == 0:
            mem.tangential_smooth(0.25)
        if n % 20 == 0:
            mem.collapse_short_edges(lmin)
    if not np.isfinite(mem.X).all():
        print("NON-FINITE at", mem.t)
        break
tag = f"ellipsoid_n{nsub}_{mode}" + (f"_cap{cap:g}" if mode == "bare" else "")
cols = "t dE vmax vpole zpole zbot rho_mid nfold nx nv".split()
np.savez(os.path.join(outdir, f"{tag}.npz"), rows=np.array(rows), cols=cols,
         snap_keys=np.array(list(snaps.keys())),
         **{f"snapX_{i}": np.concatenate([[v_[0]], v_[1].ravel()])
            for i, v_ in enumerate(snaps.values())},
         **{f"snapF_{i}": v_[2] for i, v_ in enumerate(snaps.values())})
print(f"{tag}: {n} steps, t={mem.t:.4f}, {time.time() - t0:.0f}s, "
      f"final |dE/E|={abs(mem.energy() / E0 - 1):.2e}")
