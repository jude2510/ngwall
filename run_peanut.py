"""Dumbbell ('peanut') closed wall: does the neck pinch before global collapse?

Initial shape: isosurface of a two-blob Gaussian metaball,
    F(x) = exp(-|x - d ez|^2/s^2) + exp(-|x + d ez|^2/s^2) = c,
with the level c fixed by the requested waist radius rho_w. This gives round
(blunt) bulbs joined by a thin smooth neck, so the neck pinch -- the
self-intersection event where intercommutation would split the wall into two
closed daughter walls -- happens well before the bulbs themselves collapse.
Meshed by pushing icosphere vertices along their rays onto the isosurface
(the shape is star-shaped about the origin).
"""

import numpy as np
from membrane import Membrane, icosphere


def peanut(nsub=4, d=0.9, s=0.5, rho_w=0.25):
    V, F = icosphere(nsub)
    c = 2.0 * np.exp(-(rho_w ** 2 + d ** 2) / s ** 2)

    def g(r):
        # F(r*n_hat) - c for all directions at once
        P = V * r[:, None]
        dp = np.sum((P - [0, 0, d]) ** 2, axis=1)
        dm = np.sum((P + [0, 0, d]) ** 2, axis=1)
        return np.exp(-dp / s ** 2) + np.exp(-dm / s ** 2) - c

    # bracket the outermost root along each ray by marching inward ...
    lo = np.full(len(V), np.nan)
    hi = np.full(len(V), np.nan)
    done = np.zeros(len(V), bool)
    grid = np.linspace(d + 4 * s, 1e-3, 600)
    for rprev, r in zip(grid[:-1], grid[1:]):
        inside = (g(np.full(len(V), r)) > 0) & ~done
        lo[inside], hi[inside], done[inside] = r, rprev, True
    assert done.all(), "some rays never crossed the isosurface"
    # ... then refine by bisection
    for _ in range(50):
        mid = 0.5 * (lo + hi)
        pos = g(mid) > 0
        lo, hi = np.where(pos, mid, lo), np.where(pos, hi, mid)
    return V * (0.5 * (lo + hi))[:, None], F


def run(nsub=4, d=0.9, s=0.5, rho_w=0.25, cfl=0.2, vstop=0.999, neck_stop=0.02,
        nsnap=6, smooth_every=5, smooth_lam=0.25, max_steps=150000,
        vtk=None, dt_snap=0.02):
    X, F = peanut(nsub, d, s, rho_w)
    mem = Membrane(X, F, sigma=1.0)
    E0 = mem.energy()
    series = None
    if vtk:
        from vtk_out import VtkSeries, wall_block
        series = VtkSeries(vtk, "peanut")

    def vtk_add():
        V = mem.velocity(mem.X, mem.p)
        sp = np.sqrt(np.einsum('ij,ij->i', V, V))
        series.add(mem.t, [wall_block(mem.X, mem.faces, sp, 0)])

    def neck_radius():
        m = np.abs(mem.X[:, 2]) < 0.1
        rho = np.hypot(mem.X[m, 0], mem.X[m, 1])
        return rho.min()

    hist = {"t": [], "neck": [], "zmax": [], "E": [], "vmax": [], "minedge": []}
    snaps = [(0.0, mem.X.copy())]
    tsnap = 0.0
    tvtk = 0.0
    nstep = 0
    smooth_dE, smooth_dP = 0.0, 0.0
    while True:
        vmax = mem.vmax()
        nk = neck_radius()
        hist["t"].append(mem.t)
        hist["neck"].append(nk)
        hist["zmax"].append(np.abs(mem.X[:, 2]).max())
        hist["E"].append(mem.energy())
        hist["vmax"].append(vmax)
        hist["minedge"].append(mem.min_edge(mem.X))
        if series is not None and mem.t >= tvtk - 1e-12:
            vtk_add()
            tvtk += dt_snap
        if nk < neck_stop:
            print(f"NECK PINCH at t = {mem.t:.4f} (neck radius {nk:.4f})")
            break
        if vmax > vstop:
            print(f"stopped on vmax = {vmax:.4f} at t = {mem.t:.4f}")
            break
        if nstep >= max_steps:
            print(f"stopped on step limit at t = {mem.t:.4f}")
            break
        if mem.t >= tsnap:
            if mem.t > 0:
                snaps.append((mem.t, mem.X.copy()))
            tsnap += 0.12
        dt = cfl * hist["minedge"][-1]
        mem.step(dt)
        nstep += 1
        if smooth_every and nstep % smooth_every == 0:
            dE, dP = mem.tangential_smooth(smooth_lam)
            smooth_dE += dE
            smooth_dP += dP
    snaps.append((mem.t, mem.X.copy()))
    if series is not None:
        if series.last_t() < mem.t - 1e-12:
            vtk_add()
        print(f"VTK series: {series.close()}  "
              f"({len(series.frames)} frames — open in ParaView)")
    hist = {k: np.array(v) for k, v in hist.items()}
    drift = np.abs(hist["E"] / E0 - 1.0).max()
    print(f"steps = {len(hist['t'])},  max energy drift (total) = {drift:.2e}")
    print(f"smoothing: cumulative |dE|/E = {smooth_dE:.2e}, |dP|/E = {smooth_dP:.2e}")
    print(f"bulb extent z_max: {hist['zmax'][0]:.3f} -> {hist['zmax'][-1]:.3f}")
    np.savez("peanut_hist.npz", **hist,
             snap_t=np.array([s[0] for s in snaps]),
             **{f"snap{i}": s[1] for i, s in enumerate(snaps)})
    return hist, snaps


if __name__ == "__main__":
    run()
