"""Validation: collapse of a spherical NG wall from rest.

Exact solution (from energy conservation gamma R^2 = R0^2):
    Rddot = -(2/R)(1 - Rdot^2),  collapse at t* = sqrt(pi)*Gamma(5/4)/Gamma(3/4).
"""

import numpy as np
from scipy.special import gamma as Gamma
from membrane import Membrane, icosphere


def exact_sphere(R0=1.0, dt=1e-4):
    """RK4 for Rddot = -(2/R)(1 - Rdot^2), from rest at R0."""
    def f(y):
        R, Rd = y
        return np.array([Rd, -(2.0 / R) * (1.0 - Rd ** 2)])
    y = np.array([R0, 0.0])
    ts, Rs = [0.0], [R0]
    t = 0.0
    while y[0] > 1e-3 * R0:
        k1 = f(y); k2 = f(y + 0.5 * dt * k1)
        k3 = f(y + 0.5 * dt * k2); k4 = f(y + dt * k3)
        y = y + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += dt
        ts.append(t); Rs.append(y[0])
    return np.array(ts), np.array(Rs)


def run(nsub=3, cfl=0.2, vstop=0.999):
    V, F = icosphere(nsub)
    mem = Membrane(V, F, sigma=1.0)
    E0 = mem.energy()
    hist = {"t": [], "R": [], "E": [], "vmax": []}
    while True:
        vmax = mem.vmax()
        hist["t"].append(mem.t)
        hist["R"].append(np.linalg.norm(mem.X, axis=1).mean())
        hist["E"].append(mem.energy())
        hist["vmax"].append(vmax)
        if vmax > vstop:
            break
        dt = cfl * mem.min_edge(mem.X)
        mem.step(dt)
    return {k: np.array(v) for k, v in hist.items()}, E0


if __name__ == "__main__":
    hist, E0 = run()
    te, Re = exact_sphere()
    tstar = np.sqrt(np.pi) * Gamma(1.25) / Gamma(0.75)

    # interpolate exact R onto simulation times and compare
    Rex = np.interp(hist["t"], te, Re)
    err = np.abs(hist["R"] - Rex)
    drift = np.abs(hist["E"] / E0 - 1.0)

    np.savez("sphere_hist.npz", **hist, te=te, Re=Re, tstar=tstar)
    print(f"exact collapse time t* = {tstar:.5f}")
    print(f"reached t = {hist['t'][-1]:.5f}  with R = {hist['R'][-1]:.5f}, "
          f"vmax = {hist['vmax'][-1]:.5f}  ({len(hist['t'])} steps)")
    print(f"max |R_num - R_exact| over run = {err.max():.2e}")
    print(f"max relative energy drift     = {drift.max():.2e}")
