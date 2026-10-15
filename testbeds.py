"""Test geometries and the exact sphere solution.

  exact_sphere   reference R(t) for a spherical wall collapsing from rest
  peanut         two-blob dumbbell with a thin smooth neck (star-shaped)
  kicked_peanut  dumbbell whose waist is kicked inward: a clean neck pinch
  torus          genus-1 torus mesh
  oblate         oblate spheroid (pancake)
  kicked_oblate  pancake with its faces kicked toward each other: face-face
                 contact before the rim cusp
"""

import numpy as np
from membrane import Membrane, icosphere
from contact import vertex_normals


def exact_sphere(R0=1.0, dt=1e-4):
    """RK4 for Rddot = -(2/R)(1 - Rdot^2), from rest at R0.

    Exact NG solution of a spherical wall (from energy conservation
    gamma R^2 = R0^2); collapse at t* = sqrt(pi)*Gamma(5/4)/Gamma(3/4)*R0.
    Returns (t, R) arrays until R < 1e-3 R0."""
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


def peanut(nsub=4, d=0.9, s=0.5, rho_w=0.25):
    """Dumbbell: isosurface of a two-blob Gaussian metaball,

        F(x) = exp(-|x - d ez|^2/s^2) + exp(-|x + d ez|^2/s^2) = c,

    with c fixed by the requested waist radius rho_w: round bulbs joined by
    a thin smooth neck. Meshed by pushing icosphere vertices along their rays
    onto the isosurface (the shape is star-shaped about the origin)."""
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


def kicked_peanut(nsub=4, d=1.1, s=0.35, rho_w=0.10, v0=0.7, w=0.25):
    """Dumbbell whose waist ring gets an inward kick along the local normal,
    v_n = -v0 exp(-z^2/w^2) (n.rho_hat), driving a clean central pinch at
    t ~ 1 while the bulbs are still extended. A rest-start dumbbell never
    pinches before global collapse: its waist bounces outward."""
    X, F = peanut(nsub, d, s, rho_w)
    mem = Membrane(X, F, sigma=1.0)
    vn = vertex_normals(X, F)
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


def torus(R=1.0, a=0.45, nu=64, nv=32):
    """Torus mesh: tube radius a around a circle of radius R (genus 1)."""
    iu, iv = np.meshgrid(np.arange(nu), np.arange(nv), indexing="ij")
    U = 2 * np.pi * iu / nu
    V = 2 * np.pi * iv / nv
    X = np.stack([(R + a * np.cos(V)) * np.cos(U),
                  (R + a * np.cos(V)) * np.sin(U),
                  a * np.sin(V)], axis=-1).reshape(-1, 3)
    vid = (iu * nv + iv).reshape(nu, nv)
    F = []
    for i in range(nu):
        for j in range(nv):
            v00, v01 = vid[i, j], vid[i, (j + 1) % nv]
            v10, v11 = vid[(i + 1) % nu, j], vid[(i + 1) % nu, (j + 1) % nv]
            F += [[v00, v10, v11], [v00, v11, v01]]
    return X, np.array(F)


def oblate(c=0.3, nsub=4):
    """Oblate spheroid (pancake): unit sphere squashed to height c."""
    X, F = icosphere(nsub)
    X = X.copy()
    X[:, 2] *= c
    return X, F


def kicked_oblate(c=0.3, v0=0.6, w=0.6, nsub=4):
    """Pancake with its flat faces kicked toward each other along the local
    normal, magnitude -v0 exp(-rho^2/w^2) |n.z| -- full strength at the face
    centers, zero at the rim -- so face-face contact (the tunnel case) comes
    before the rim cusp."""
    X, F = oblate(c, nsub)
    mem = Membrane(X, F)
    vn = vertex_normals(X, F)
    rho2 = X[:, 0] ** 2 + X[:, 1] ** 2
    vmag = -v0 * np.exp(-rho2 / w ** 2) * np.abs(vn[:, 2])
    V = vmag[:, None] * vn
    gam = 1.0 / np.sqrt(1 - np.einsum('ij,ij->i', V, V))
    m = mem.sigma * mem.vertex_areas(X)
    mem.p = (m * gam)[:, None] * V
    mem.genus = 0
    return mem
