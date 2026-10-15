"""Approach-speed census: how fast do pieces of a random wall meet?

Evolves the production initial data (run_random.random_wall, nsub=4) as pure
Nambu-Goto with the bare evolver (no smoothing, coarsening or surgery), so
sheets pass through each other, and logs every new self-contact.

Every 2 steps all mesh self-crossings (edge pierces triangle) are found.
Crossings farther than 2 median edge lengths from every crossing of the
previous check are new contact; new crossings are clustered (single linkage,
same radius) into contact events. Per crossing:
  u    per-sheet speed in the frame where the two sheets approach
       symmetrically, from their relativistic closing speed along the
       triangle normal (compare the phi^4 kink-antikink v_c = 0.2598)
  dot  n_edge . n_triangle: < -0.7 head-on, > 0.7 co-moving (fold overlap)
  local  the two sheets are within 6 mesh hops (fold) or not (distant)

Usage: python approach_census.py L_MAX SEED OMEGA_Z T_MAX [OUTDIR]
"""
import os
import sys
import json
import time
import numpy as np
from scipy import sparse
from scipy.spatial import cKDTree
from ngtools import (crossing_detail, face_normals, symmetric_speed, gamma_dt,
                     vertex_normals, _khop)
import run_random

l_max, seed = int(sys.argv[1]), int(sys.argv[2])
wz, t_max = float(sys.argv[3]), float(sys.argv[4])
outdir = sys.argv[5] if len(sys.argv) > 5 else "."
os.makedirs(outdir, exist_ok=True)
tag = f"census_l{l_max}_s{seed}_w{wz:g}"

mem = run_random.random_wall(l_max=l_max, seed=seed, omega=(0, 0, wz), nsub=4)
E0 = mem.energy()
R6, R6_faces = None, None
prevP = np.zeros((0, 3))
events, sweep, series = [], [], []
n, next_report = 0, 0.1
t0 = time.time()
while mem.t < t_max:
    mem.step(gamma_dt(mem))
    n += 1
    if not (np.isfinite(mem.X).all() and np.isfinite(mem.p).all()):
        print(f"{tag}: NON-FINITE at t={mem.t:.4f}", flush=True)
        break
    if mem.t >= next_report:
        print(f"{tag}: t={mem.t:.2f} steps={n} events={len(events)} "
              f"dE/E={mem.energy() / E0 - 1:+.1e} wall={time.time() - t0:.0f}s", flush=True)
        next_report += 0.1
    if n % 2:
        continue
    X, F = mem.X, mem.faces
    cd = crossing_detail(X, F)
    ext = float((X.max(0) - X.min(0)).max())
    series.append((round(mem.t, 5), len(cd["tri"]), round(ext, 5), mem.energy() / E0))
    if len(cd["tri"]) == 0:
        prevP = np.zeros((0, 3))
        continue
    rnew = 2 * float(np.median(np.linalg.norm(X[F[:, 0]] - X[F[:, 1]], axis=1)))
    new = (cKDTree(prevP).query(cd["P"])[0] > rnew) if len(prevP) else \
        np.ones(len(cd["tri"]), bool)
    prevP = cd["P"]
    if not new.any():
        continue
    if R6_faces is not F:
        R6, R6_faces = _khop(F, len(X), 6), F
    V = mem.velocity(X, mem.p)
    vn = vertex_normals(X, F)
    idx = np.nonzero(new)[0]
    e, tri, s, bary = cd["e"][idx], cd["tri"][idx], cd["s"][idx], cd["bary"][idx]
    Ve = (1 - s)[:, None] * V[e[:, 0]] + s[:, None] * V[e[:, 1]]
    ne = (1 - s)[:, None] * vn[e[:, 0]] + s[:, None] * vn[e[:, 1]]
    ne /= np.clip(np.linalg.norm(ne, axis=1), 1e-300, None)[:, None]
    VT = np.einsum('kc,kcj->kj', bary, V[F[tri]])
    nT = face_normals(X, F[tri])
    a_ = np.einsum('ij,ij->i', Ve, nT)
    b_ = np.einsum('ij,ij->i', VT, nT)
    u = symmetric_speed((a_ - b_) / (1 - a_ * b_))
    dot = np.einsum('ij,ij->i', ne, nT)
    local = np.zeros(len(idx), bool)
    for i0 in (0, 1):
        for j0 in range(3):
            local |= np.asarray(R6[e[:, i0], F[tri, j0]]).ravel().astype(bool)
    vmax = np.maximum(np.linalg.norm(Ve, axis=1), np.linalg.norm(VT, axis=1))
    sweep += [(round(mem.t, 5), round(float(u[k]), 4), int(local[k]),
               round(float(dot[k]), 3)) for k in range(len(idx))]
    P = cd["P"][idx]
    pr = cKDTree(P).query_pairs(rnew, output_type="ndarray")
    G = sparse.csr_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(P),) * 2)
    _, lab = sparse.csgraph.connected_components(G, directed=False)
    for cid in np.unique(lab):
        m = lab == cid
        events.append(dict(t=round(mem.t, 5), n=int(m.sum()), local=bool(local[m].all()),
                           any_local=bool(local[m].any()),
                           u_med=float(np.median(u[m])), u_min=float(u[m].min()),
                           u_max=float(u[m].max()), dot_med=float(np.median(dot[m])),
                           vmax=float(vmax[m].max()), where=P[m].mean(0).round(5).tolist(),
                           extent=round(ext, 5)))
with open(os.path.join(outdir, f"{tag}.json"), "w") as f:
    json.dump(dict(l_max=l_max, seed=seed, omega=wz, t_max=t_max, steps=n,
                   t_end=mem.t, E_drift=mem.energy() / E0 - 1, events=events,
                   sweep=sweep, series=series), f)
print(f"{tag}: DONE {n} steps, t={mem.t:.3f}, {len(events)} events, "
      f"dE/E={mem.energy() / E0 - 1:+.1e}, {time.time() - t0:.0f}s", flush=True)
