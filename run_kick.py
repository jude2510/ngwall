"""Kicked-waist dumbbell: the controlled single-pinch intercommutation test.

A rest-start dumbbell never pinches before global collapse (the waist bounces
outward on the axial saddle curvature while the whole wall implodes into a
near-axis spindle -- see peanut runs). Here the waist ring is given an initial
inward kick along the local surface normal, v_n = -v0 exp(-z^2/w^2) (n.rho),
which preserves the transverse gauge exactly (V parallel to n) and drives a
clean central pinch at t ~ 1 while the bulbs are still extended. At the pinch
the wall is split by intercommutation surgery (surgery.py) and both daughter
walls are evolved independently.
"""

import numpy as np
from membrane import Membrane
from run_peanut import peanut
from surgery import detect_pinch, pinch_split
from vtk_out import VtkSeries, wall_block


def kicked_peanut(nsub=4, d=1.1, s=0.35, rho_w=0.10, v0=0.7, w=0.25):
    X, F = peanut(nsub, d, s, rho_w)
    mem = Membrane(X, F, sigma=1.0)

    # outward vertex normals (area-weighted face normals)
    a, b, c = (X[F[:, k]] for k in range(3))
    fn = np.cross(b - a, c - a)
    vn = np.zeros_like(X)
    for k in range(3):
        np.add.at(vn, F[:, k], fn)
    vn /= np.linalg.norm(vn, axis=1)[:, None]

    rho = np.hypot(X[:, 0], X[:, 1])
    rho_hat = np.zeros_like(X)
    nz = rho > 1e-12
    rho_hat[nz, :2] = X[nz, :2] / rho[nz, None]
    vmag = -v0 * np.exp(-X[:, 2] ** 2 / w ** 2) * \
        np.einsum('ij,ij->i', vn, rho_hat)
    V = vmag[:, None] * vn
    gam = 1.0 / np.sqrt(1 - np.einsum('ij,ij->i', V, V))
    m = mem.sigma * mem.vertex_areas(X)
    mem.p = (m * gam)[:, None] * V
    return mem


def _speed(mem):
    V = mem.velocity(mem.X, mem.p)
    return np.sqrt(np.einsum('ij,ij->i', V, V))


def evolve(mem, mask, cfl=0.2, vstop=0.999999, smooth_every=5, smooth_lam=0.25,
           rho_c=None, t_end=None, max_steps=20000, tag="", capture=None,
           coarsen=None, coarsen_every=20):
    """Evolve to pinch detection (rho_c), t_end, vstop or step limit.

    capture=(dt_snap, out_list) appends (t, X, speed, faces) snapshots every
    dt_snap of simulation time, plus the final state, for VTK output (faces
    per frame because coarsening can change the topology mid-run).
    coarsen=lmin enables edge-collapse coarsening every coarsen_every steps;
    its energy cost is reported (dissipation, like surgery), not rescaled.
    mask is dropped after any coarsening: vertex indices no longer match."""
    if coarsen and rho_c is not None:
        raise ValueError("coarsening during a pinch-detection phase is not "
                         "supported: collapse invalidates the vertex mask "
                         "(needs generic self-contact detection)")
    E0 = mem.energy()
    hist = {"t": [], "rmin": [], "E": [], "vmax": []}
    nstep = 0
    why = "steps"
    tsnap = mem.t
    ncol, col_dE = 0, 0.0
    while True:
        rho = np.hypot(mem.X[:, 0], mem.X[:, 1])
        hist["t"].append(mem.t)
        # daughters (mask None): ignore their own on-axis cap vertex
        hist["rmin"].append(rho[mask].min() if mask is not None
                            else rho[rho > 1e-6].min())
        hist["E"].append(mem.energy())
        vmax = mem.vmax()
        hist["vmax"].append(vmax)
        if capture is not None and mem.t >= tsnap - 1e-12:
            capture[1].append((mem.t, mem.X.copy(), _speed(mem), mem.faces))
            tsnap += capture[0]
        if rho_c is not None and detect_pinch(mem, mask, rho_c) is not None:
            why = "pinch"
            break
        if t_end is not None and mem.t >= t_end:
            why = "t_end"
            break
        if vmax > vstop:
            why = "vstop"
            break
        if nstep >= max_steps:
            break
        mem.step(cfl * mem.min_edge(mem.X))
        nstep += 1
        if smooth_every and nstep % smooth_every == 0:
            mem.tangential_smooth(smooth_lam)
        if coarsen and nstep % coarsen_every == 0:
            nc, dE, dP = mem.collapse_short_edges(coarsen)
            if nc:
                ncol += nc
                col_dE += dE
                mask = None    # vertex indices changed; rmin falls back safely
    if capture is not None and (not capture[1]
                                or capture[1][-1][0] < mem.t - 1e-12):
        capture[1].append((mem.t, mem.X.copy(), _speed(mem), mem.faces))
    hist = {k: np.array(v) for k, v in hist.items()}
    drift = np.abs(hist["E"] / E0 - 1.0).max()
    extra = (f", coarsened {ncol} edges (cum |dE|/E = {col_dE:.2e})"
             if ncol else "")
    print(f"[{tag}] stop on {why} at t = {mem.t:.4f}: rmin = {hist['rmin'][-1]:.4f}, "
          f"vmax = {hist['vmax'][-1]:.4f}, steps = {nstep}, drift = {drift:.2e}"
          f"{extra}")
    return hist, why


def run(rho_c=0.02, r_cut=0.04, h=0.03, t_after=0.25, vtk="kick_vtk",
        dt_snap=0.02, coarsen=None):
    mem = kicked_peanut()
    z0, r0 = mem.X[:, 2], np.linalg.norm(mem.X, axis=1)
    mask = np.abs(z0) / r0 < 0.97          # exclude on-axis polar caps

    capP = [] if vtk else None
    hist, why = evolve(mem, mask, rho_c=rho_c, tag="parent",
                       capture=(dt_snap, capP) if vtk else None)
    assert why == "pinch", f"no pinch reached (stopped on {why})"
    z_star = detect_pinch(mem, mask, rho_c)

    daughters, rep = pinch_split(mem, z_star, r_cut=r_cut, h=h)
    print(f"surgery at t = {mem.t:.4f}, z* = {z_star:+.4f}: "
          f"removed {rep['n_removed']} verts, loops {rep['loop_sizes']}")
    print(f"  E_before = {rep['E_before']:.4f}, E_removed = {rep['E_removed']:.4f}")
    print(f"  E_daughters = {[f'{e:.4f}' for e in rep['E_daughters']]}, "
          f"dE/E = {rep['dE'] / rep['E_before']:+.2e}, "
          f"dP/E = {rep['dP'] / rep['E_before']:.2e}")

    out = {"parent_" + k: v for k, v in hist.items()}
    out["snap_parent"] = mem.X
    out["snap_parent_faces"] = mem.faces
    capD = [[] if vtk else None for _ in daughters]
    for i, d in enumerate(daughters):
        if coarsen:
            nc, dE, dP = d.collapse_short_edges(coarsen)
            if nc:
                print(f"  daughter{i} seam coarsening: {nc} edges, "
                      f"|dE|/E = {dE:.2e}, |dP|/E = {dP:.2e}")
        dh, _ = evolve(d, None, t_end=d.t + t_after, tag=f"daughter{i}",
                       capture=(dt_snap, capD[i]) if vtk else None,
                       coarsen=coarsen)
        zc = d.X[:, 2]
        print(f"  daughter{i}: verts = {len(d.X)}, z in "
              f"[{zc.min():+.3f}, {zc.max():+.3f}], E = {d.energy():.4f}")
        out.update({f"d{i}_" + k: v for k, v in dh.items()})
        out[f"snap_d{i}"] = d.X
        out[f"snap_d{i}_faces"] = d.faces
    np.savez("kick_hist.npz", **out)

    if vtk:
        series = VtkSeries(vtk, "kick")
        for t, X, sp, F in capP:
            series.add(t, [wall_block(X, F, sp, 0)])
        # daughters share the snapshot grid from the split; a track that ends
        # early (v -> 1 guard) stays frozen at its last state rather than
        # vanishing, which would read as annihilation
        nfr = max(len(c) for c in capD)
        for k in range(nfr):
            blocks, t = [], -np.inf
            for i, c in enumerate(capD):
                tk, X, sp, F = c[min(k, len(c) - 1)]
                blocks.append(wall_block(X, F, sp, i + 1))
                t = max(t, tk)
            series.add(t, blocks)
        print(f"VTK series: {series.close()}  "
              f"({len(series.frames)} frames — open in ParaView)")
    return daughters, rep


if __name__ == "__main__":
    run()
