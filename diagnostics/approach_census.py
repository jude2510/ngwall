"""Approach-speed census (v2): how fast, how far apart, and how relativistic
are the pieces of a random wall when they meet?

Evolves the production initial data (run_random.random_wall) as pure
Nambu-Goto with the bare evolver (no smoothing, coarsening or surgery), so
sheets pass through each other, and logs every new self-contact.

Every CHECK_DT of simulation time, all mesh self-crossings (edge pierces
triangle) are found. Crossings farther than R_NEW from every crossing of the
previous check are new contact; new crossings are clustered (single linkage
at R_NEW) into contact events. Both are fixed physical lengths/times, so an
event means the same thing at every mesh resolution. Per crossing:
  u      per-sheet speed in the frame where the two sheets approach
         symmetrically, from their relativistic closing speed along the
         triangle normal (phi^4 kink-antikink v_c = 0.2598)
  dot    n_edge . n_triangle (only |dot| is physical for a Z2 wall:
         > 0.7 nearly parallel, the 1D kink-antikink regime)
  geo    distance between the two sheets along the surface: shortest path
         over mesh edges (weighted by length), capped at GEO_LIMIT
  gamma  Lorentz factor of each sheet at the crossing, from p and the
         vertex mass (exact near cusps, where |V| rounds to 1)

Usage: python approach_census.py L_MAX SEED OMEGA_Z T_MAX [OUTDIR] [--nsub N]
"""
import argparse
import os
import json
import time
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra, connected_components
from scipy.spatial import cKDTree
from ngtools import (crossing_detail, face_normals, symmetric_speed, gamma_dt,
                     vertex_normals, unique_edges)
import run_random

CHECK_DT = 0.008    # contact-check cadence (simulation time)
R_NEW = 0.13        # new-contact and clustering radius (2 median edges at nsub=4)
GEO_LIMIT = 1.0     # along-surface distances beyond this are stored as GEO_LIMIT

ap = argparse.ArgumentParser()
ap.add_argument("l_max", type=int)
ap.add_argument("seed", type=int)
ap.add_argument("omega_z", type=float)
ap.add_argument("t_max", type=float)
ap.add_argument("outdir", nargs="?", default=".")
ap.add_argument("--nsub", type=int, default=4)
a = ap.parse_args()
os.makedirs(a.outdir, exist_ok=True)
tag = f"census_l{a.l_max}_s{a.seed}_w{a.omega_z:g}_n{a.nsub}"

mem = run_random.random_wall(l_max=a.l_max, seed=a.seed, omega=(0, 0, a.omega_z),
                             nsub=a.nsub)
E0 = mem.energy()
prevP = np.zeros((0, 3))
events, sweep, series = [], [], []
n, next_check, next_report = 0, CHECK_DT, 0.1
t0 = time.time()
while mem.t < a.t_max:
    mem.step(gamma_dt(mem))
    n += 1
    if not (np.isfinite(mem.X).all() and np.isfinite(mem.p).all()):
        print(f"{tag}: NON-FINITE at t={mem.t:.4f}", flush=True)
        break
    if mem.t >= next_report:
        print(f"{tag}: t={mem.t:.2f} steps={n} events={len(events)} "
              f"dE/E={mem.energy() / E0 - 1:+.1e} wall={time.time() - t0:.0f}s", flush=True)
        next_report += 0.1
    if mem.t < next_check:
        continue
    next_check += CHECK_DT * np.ceil((mem.t - next_check) / CHECK_DT + 1e-12)
    X, F = mem.X, mem.faces
    cd = crossing_detail(X, F)
    ext = float((X.max(0) - X.min(0)).max())
    series.append((round(mem.t, 5), len(cd["tri"]), round(ext, 5), mem.energy() / E0))
    if len(cd["tri"]) == 0:
        prevP = np.zeros((0, 3))
        continue
    new = (cKDTree(prevP).query(cd["P"])[0] > R_NEW) if len(prevP) else \
        np.ones(len(cd["tri"]), bool)
    prevP = cd["P"]
    if not new.any():
        continue
    idx = np.nonzero(new)[0]
    e, tri, s, bary = cd["e"][idx], cd["tri"][idx], cd["s"][idx], cd["bary"][idx]

    # closing speed and crossing angle
    V = mem.velocity(X, mem.p)
    vn = vertex_normals(X, F)
    Ve = (1 - s)[:, None] * V[e[:, 0]] + s[:, None] * V[e[:, 1]]
    ne = (1 - s)[:, None] * vn[e[:, 0]] + s[:, None] * vn[e[:, 1]]
    ne /= np.clip(np.linalg.norm(ne, axis=1), 1e-300, None)[:, None]
    VT = np.einsum('kc,kcj->kj', bary, V[F[tri]])
    nT = face_normals(X, F[tri])
    a_ = np.einsum('ij,ij->i', Ve, nT)
    b_ = np.einsum('ij,ij->i', VT, nT)
    u = symmetric_speed((a_ - b_) / (1 - a_ * b_))
    dot = np.einsum('ij,ij->i', ne, nT)

    # Lorentz factors from p and the vertex mass: gamma = E_v / m_v
    m = np.clip(mem.sigma * mem.vertex_areas(X), 1e-300, None)
    gam = np.sqrt(m ** 2 + np.einsum('ij,ij->i', mem.p, mem.p)) / m
    gam_e = (1 - s) * gam[e[:, 0]] + s * gam[e[:, 1]]
    gam_T = np.einsum('kc,kc->k', bary, gam[F[tri]])

    # distance between the two sheets along the surface (from the edge's
    # first endpoint to the nearest triangle corner; error < one edge)
    E = unique_edges(F)
    el = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
    G = sparse.csr_matrix((np.r_[el, el], (np.r_[E[:, 0], E[:, 1]], np.r_[E[:, 1], E[:, 0]])),
                          shape=(len(X), len(X)))
    src, inv = np.unique(e[:, 0], return_inverse=True)
    geo = np.empty(len(idx))
    for c0 in range(0, len(src), 200):
        D = dijkstra(G, indices=src[c0:c0 + 200], limit=GEO_LIMIT)
        rows = np.nonzero((inv >= c0) & (inv < c0 + 200))[0]
        Dr = D[inv[rows] - c0]                       # (rows, n_vertices)
        geo[rows] = Dr[np.arange(len(rows))[:, None], F[tri[rows]]].min(axis=1)
    geo = np.minimum(geo, GEO_LIMIT)

    sweep += [(round(mem.t, 5), round(float(u[k]), 4), round(float(dot[k]), 3),
               round(float(geo[k]), 4), round(float(max(gam_e[k], gam_T[k])), 2))
              for k in range(len(idx))]
    P = cd["P"][idx]
    pr = cKDTree(P).query_pairs(R_NEW, output_type="ndarray")
    Gc = sparse.csr_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(len(P),) * 2)
    _, lab = connected_components(Gc, directed=False)
    gmax = np.maximum(gam_e, gam_T)
    for cid in np.unique(lab):
        k = lab == cid
        events.append(dict(t=round(mem.t, 5), n=int(k.sum()),
                           u_med=float(np.median(u[k])), u_min=float(u[k].min()),
                           u_max=float(u[k].max()), dot_med=float(np.median(dot[k])),
                           adot_med=float(np.median(np.abs(dot[k]))),
                           geo_med=float(np.median(geo[k])), geo_min=float(geo[k].min()),
                           geo_max=float(geo[k].max()),
                           gam_med=float(np.median(gmax[k])), gam_max=float(gmax[k].max()),
                           where=P[k].mean(0).round(5).tolist(), extent=round(ext, 5)))
with open(os.path.join(a.outdir, f"{tag}.json"), "w") as f:
    json.dump(dict(census_version=2, l_max=a.l_max, seed=a.seed, omega=a.omega_z,
                   nsub=a.nsub, t_max=a.t_max, check_dt=CHECK_DT, r_new=R_NEW,
                   geo_limit=GEO_LIMIT, steps=n, t_end=mem.t,
                   E_drift=mem.energy() / E0 - 1, events=events,
                   sweep_cols=["t", "u", "dot", "geo", "gamma"], sweep=sweep,
                   series_cols=["t", "n_crossings", "extent", "E/E0"], series=series), f)
print(f"{tag}: DONE {n} steps, t={mem.t:.3f}, {len(events)} events, "
      f"dE/E={mem.energy() / E0 - 1:+.1e}, {time.time() - t0:.0f}s", flush=True)
