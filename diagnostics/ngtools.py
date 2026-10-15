"""Shared helpers for the cusp / self-contact diagnostics.

The "bare" evolver used here is the transverse-gauge Lagrangian mesh with no
tangential smoothing and no coarsening, stepped with a gamma-weighted CFL
condition. It follows Nambu-Goto evolution through cuspidal edges (validated
against arXiv:2605.22974 in bench_ellipsoid.py), whereas smoothing and
coarsening corrupt the energy there.
"""
import os
import sys
for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.spatial import cKDTree
from contact import vertex_normals, _khop  # noqa: F401  (re-exported)


def unique_edges(F):
    return np.unique(np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]),
                             axis=1), axis=0)


def gamma_dt(mem, cfl=0.2, cap=0.004):
    """Time step from the gamma-weighted CFL condition.

    The stiffest mode of the discrete membrane has frequency ~ 1/(gamma*h),
    so the stable step grows with gamma near a cusp, where edges shrink but
    gamma diverges. Capped for accuracy in slow regions."""
    V = mem.velocity(mem.X, mem.p)
    gam = 1.0 / np.sqrt(np.clip(1 - np.einsum('ij,ij->i', V, V), 1e-300, None))
    F = mem.faces
    E = np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]])
    le = np.linalg.norm(mem.X[E[:, 0]] - mem.X[E[:, 1]], axis=1)
    return min(cfl * np.min(le * np.minimum(gam[E[:, 0]], gam[E[:, 1]])), cap)


def crossing_detail(X, F):
    """Every (edge, triangle) pair where a mesh edge pierces a triangle it
    shares no vertex with -- empty for an embedded surface. Returns arrays:
    e (k,2) edge vertices, tri (k,) face index, s (k,) edge parameter,
    bary (k,3) barycentric coordinates on the triangle, P (k,3) pierce point."""
    E = unique_edges(F)
    cen = X[F].mean(axis=1)
    trad = np.linalg.norm(X[F] - cen[:, None], axis=2).max(axis=1)
    el = np.linalg.norm(X[E[:, 0]] - X[E[:, 1]], axis=1)
    mid = 0.5 * (X[E[:, 0]] + X[E[:, 1]])
    # an edge can only pierce a triangle if their bounding spheres overlap,
    # |mid - cen| <= len/2 + trad. Binning triangles by size keeps the query
    # radius local instead of set by the single largest triangle.
    bins = np.digitize(trad, np.quantile(trad, [0.5, 0.8, 0.95, 0.99]))
    ei_l, ti_l = [], []
    for b in np.unique(bins):
        tb = np.nonzero(bins == b)[0]
        lists = cKDTree(cen[tb]).query_ball_point(mid, el / 2 + trad[tb].max())
        ei_l.append(np.repeat(np.arange(len(E)), [len(l) for l in lists]))
        ti_l.append(tb[np.concatenate([np.asarray(l, int) for l in lists])])
    ei, ti = np.concatenate(ei_l), np.concatenate(ti_l)
    share = ((F[ti] == E[ei, 0:1]) | (F[ti] == E[ei, 1:2])).any(axis=1)
    ei, ti = ei[~share], ti[~share]
    # Moller-Trumbore segment-triangle test
    P0, P1 = X[E[ei, 0]], X[E[ei, 1]]
    A, B, C = X[F[ti, 0]], X[F[ti, 1]], X[F[ti, 2]]
    d = P1 - P0
    e1, e2 = B - A, C - A
    h = np.cross(d, e2)
    a = np.einsum('ij,ij->i', e1, h)
    ok = np.abs(a) > 1e-18
    f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
    s0 = P0 - A
    u = f * np.einsum('ij,ij->i', s0, h)
    q = np.cross(s0, e1)
    v = f * np.einsum('ij,ij->i', d, q)
    t = f * np.einsum('ij,ij->i', e2, q)
    tol = 1e-9
    hit = ok & (u > tol) & (v > tol) & (u + v < 1 - tol) & (t > tol) & (t < 1 - tol)
    ei, ti, u, v, t = ei[hit], ti[hit], u[hit], v[hit], t[hit]
    return dict(e=E[ei], tri=ti, s=t, bary=np.stack([1 - u - v, u, v], axis=1),
                P=X[E[ei, 0]] + t[:, None] * (X[E[ei, 1]] - X[E[ei, 0]]))


def face_normals(X, F):
    n = np.cross(X[F[:, 1]] - X[F[:, 0]], X[F[:, 2]] - X[F[:, 0]])
    return n / np.clip(np.linalg.norm(n, axis=1), 1e-300, None)[:, None]


def fold_edges(X, F, thresh=-0.5):
    """Midpoints of interior edges whose two faces' normals have dot <
    thresh: the surface is folded back on itself (cuspidal edges, creases)."""
    fn = face_normals(X, F)
    ea = np.sort(np.vstack([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]]), axis=1)
    fid = np.tile(np.arange(len(F)), 3)
    key = ea[:, 0].astype(np.int64) * (len(X) + 1) + ea[:, 1]
    o = np.argsort(key, kind="stable")
    key, fid, ea = key[o], fid[o], ea[o]
    i = np.nonzero(key[1:] == key[:-1])[0]
    d = np.einsum('ij,ij->i', fn[fid[i]], fn[fid[i + 1]])
    e = ea[i[d < thresh]]
    return 0.5 * (X[e[:, 0]] + X[e[:, 1]])


def symmetric_speed(w):
    """Per-sheet speed in the frame where two sheets approach symmetrically,
    from their relativistic closing speed |w| < 1 (half the rapidity)."""
    w = np.clip(np.abs(w), 0, 1 - 1e-15)
    return np.tanh(0.5 * np.arctanh(w))
