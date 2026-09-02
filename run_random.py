"""Random-wall cascades: the domain-wall analog of Copi-Vachaspati.

Initial data is a sphere perturbed by random real spherical harmonics up to
l_max (the analog of CV's M loop harmonics, flat spectrum like their Type A)
in BOTH shape and transverse velocity, plus an optional coherent spin field
(omega x X).n -- the normal projection of rigid rotation, which carries
genuine angular momentum on an asymmetric shape. Only the normal velocity
component is physical in transverse gauge; tangential motion is gauge.

Each wall is evolved through the full contact/surgery cascade and the census
(fragmentation tree with energy, momentum, angular momentum, inertia
eigenvalues, genus, survival) is written to JSON.

Determinism: BLAS is pinned to one thread (set before numpy import) so a
given seed reproduces its tree bit-for-bit; multithreaded reductions were
observed to flip knife-edge retirement branches.
"""

import os
for _v in ("VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
           "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import json
import numpy as np
from membrane import Membrane, icosphere
from contact import vertex_normals
from run_cascade import cascade, summarize, ang_mom, GENUS_MAX

try:
    from scipy.special import sph_harm_y

    def _ylm(m, l, theta, phi):
        return sph_harm_y(l, m, theta, phi)
except ImportError:                            # older scipy
    from scipy.special import sph_harm

    def _ylm(m, l, theta, phi):
        return sph_harm(m, l, phi, theta)


def _random_field(nhat, l_max, rng):
    """Random real-harmonic field on directions nhat, unit RMS.

    Flat spectrum: every (l, m) coefficient uniform in [-1, 1] (CV Type A --
    maximal small-scale power, maximal fragmentation)."""
    theta = np.arccos(np.clip(nhat[:, 2], -1, 1))
    phi = np.arctan2(nhat[:, 1], nhat[:, 0])
    f = np.zeros(len(nhat))
    for l in range(2, l_max + 1):
        for m in range(0, l + 1):
            y = _ylm(m, l, theta, phi)
            f += rng.uniform(-1, 1) * y.real
            if m > 0:
                f += rng.uniform(-1, 1) * y.imag
    return f / f.std()


def random_wall(l_max=8, eps=0.25, v_amp=0.4, omega=(0.0, 0.0, 0.0),
                seed=0, nsub=4, R0=1.0):
    """Randomly perturbed sphere with random transverse velocities.

    eps: RMS fractional radius perturbation; v_amp: RMS of the random
    normal-velocity field; omega: coherent spin vector (its normal
    projection is added to the velocity). Speeds are clipped at 0.95."""
    rng = np.random.default_rng(seed)
    V0, F = icosphere(nsub)
    f = _random_field(V0, l_max, rng)
    # unlucky seeds whose mode sum would push the surface through the origin
    # get a deterministically shrunken amplitude instead of a crash; the
    # effective eps is recorded in the census (mem.eps_eff)
    eps_eff = eps
    if 1.0 + eps * f.min() <= 0.3:
        eps_eff = 0.9 * 0.7 / (-f.min())
    r = R0 * (1.0 + eps_eff * f)
    X = V0 * r[:, None]
    mem = Membrane(X, F)
    mem.genus = 0
    mem.eps_eff = eps_eff

    vn_hat = vertex_normals(X, F)
    v_n = v_amp * _random_field(V0, l_max, rng)
    v_n += np.einsum('ij,ij->i', np.cross(np.asarray(omega, float), X),
                     vn_hat)
    v_n = np.clip(v_n, -0.95, 0.95)
    V = v_n[:, None] * vn_hat
    gam = 1.0 / np.sqrt(1 - np.einsum('ij,ij->i', V, V))
    m = mem.sigma * mem.vertex_areas(X)
    mem.p = (m * gam)[:, None] * V
    return mem


def _jsonable(x):
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def run(l_max=8, eps=0.25, v_amp=0.4, omega=(0.0, 0.0, 0.0), seed=0,
        t_max=3.0, nsub=4, vtk=None, out=None):
    mem = random_wall(l_max, eps, v_amp, omega, seed, nsub)
    F = mem.faces
    e = np.linalg.norm(mem.X[F[:, 0]] - mem.X[F[:, 1]], axis=1)
    lmin = 0.1 * np.median(e)
    L0 = ang_mom(mem)
    print(f"random wall l_max={l_max} seed={seed} v_amp={v_amp} "
          f"omega={tuple(omega)}: E = {mem.energy():.3f}, "
          f"|L| = {np.linalg.norm(L0):.4f}, lmin = {lmin:.4f}")
    out = out or f"random_l{l_max}_s{seed}.json"
    records, events = cascade(mem, t_max=t_max, lmin=lmin, vtk_dir=vtk,
                              save_states=out.rsplit(".json", 1)[0])
    summarize(records, events)

    survivors = [r for r in records if r["cause"] == "survived"]
    foam = [r for r in records if r["cause"] == "foam"]
    print(f"census: {len(records)} walls, {len(events)} surgeries, "
          f"{len(survivors)} survived to t = {t_max}, "
          f"{len(foam)} foam (E = {sum(r['E1'] for r in foam):.2f})")
    with open(out, "w") as f:
        json.dump(dict(
            params=dict(l_max=l_max, eps=eps,
                        eps_eff=getattr(mem, "eps_eff", eps), v_amp=v_amp,
                        omega=list(omega), seed=seed, t_max=t_max,
                        nsub=nsub, lmin=lmin, genus_max=GENUS_MAX),
            E0=mem.energy(), L0=L0.tolist(),
            records=[{k: _jsonable(v) for k, v in r.items()}
                     for r in records],
            events=[{k: _jsonable(v) for k, v in e.items()}
                    for e in events]), f, indent=1)
    print(f"census written to {out}")
    return records, events


if __name__ == "__main__":
    run()
