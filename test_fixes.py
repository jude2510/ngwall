"""Regression tests for the numerical guards and the cascade path.

Run directly: `python3 test_fixes.py` (exits non-zero on failure). Importing
the modules also runs their import-time self-tests.
"""
import os
for _v in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from membrane import Membrane, icosphere      # noqa: E402
from contact import ContactDetector           # noqa: E402
import run_cascade                            # noqa: E402
import run_random                             # noqa: E402

fails = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        fails.append(name)


print("T1: a degenerate triangle no longer poisons the mesh")
V, F = icosphere(2)
Xd = V.copy()
Xd[F[0, 2]] = Xd[F[0, 0]]        # collapse one triangle to zero area
memd = Membrane(Xd, F)
Ad, gAd = memd.tri_geom(memd.X)
check("tri_geom output is all finite",
      np.isfinite(Ad).all() and np.isfinite(gAd).all())
check("the degenerate triangle has area exactly 0", Ad[0] == 0.0)
check("the degenerate triangle exerts zero force", np.all(gAd[0] == 0.0))
for _ in range(5):
    memd.step(0.2 * memd.min_edge(memd.X))
nbad = int((~np.isfinite(memd.p)).any(axis=1).sum())
check("momentum stays finite through 5 RK4 steps", nbad == 0,
      f"{nbad}/{len(memd.p)} non-finite (unguarded: all of them by step 3)")

print("\nT2: energy drift through a collapse with smoothing stays small")
V, F = icosphere(3)
mem = Membrane(V, F)
mem.p[:] = 0.0
E0 = mem.energy()
drift_max = 0.0
for i in range(300):
    mem.step(0.2 * mem.min_edge(mem.X))
    if (i + 1) % 5 == 0:
        mem.tangential_smooth(0.25)
    drift_max = max(drift_max, abs(mem.energy() / E0 - 1.0))
# a rest-start sphere collapsing toward a caustic; 1.3e-8 measured, flat
# from step 50 to 300 (a one-time smoothing offset, not a secular leak)
check("max |dE/E| < 1e-7", drift_max < 1e-7, f"max |dE/E| = {drift_max:.2e}")

print("\nT3: the k-ring cache is sound across topology changes")
Vs, Fs = icosphere(3)
Xs = Vs.copy()
Xs[:, 2] *= 0.01                 # squash to a pancake: sheets nearly touch
mem2 = Membrane(Xs, Fs)
cached = ContactDetector()
agree = True
for _ in range(6):
    got = cached.find(mem2, 0.05)
    want = ContactDetector().find(mem2, 0.05)      # ground truth: never cached
    if [set(g.tolist()) for g in got] != [set(w.tolist()) for w in want]:
        agree = False
    mem2.X[:, 0] *= 1.02                           # move, same topology
check("6 repeated finds: cached result == freshly built result", agree)

faces_before = mem2.faces
nv_before = len(mem2.X)
mem2.collapse_short_edges(0.05)
check("coarsening reassigned the faces array",
      mem2.faces is not faces_before, f"{nv_before} -> {len(mem2.X)} verts")
cached.find(mem2, 0.05)
check("k-ring table was rebuilt to the new vertex count",
      cached._R.shape[0] == len(mem2.X),
      f"_R is {cached._R.shape[0]}^2, mesh has {len(mem2.X)}")
check("cache holds the live faces array, so its id cannot be reused",
      cached._faces is mem2.faces)

print("\nT4: end-to-end smoke test of the cascade path")
mem_s = run_random.random_wall(l_max=4, eps=0.25, v_amp=0.4, seed=0, nsub=3)
lmin_s = 0.1 * np.median(np.linalg.norm(
    mem_s.X[mem_s.faces[:, 0]] - mem_s.X[mem_s.faces[:, 1]], axis=1))
recs, evs = run_cascade.cascade(mem_s, t_max=0.6, lmin=lmin_s)
causes = {}
for r in recs:
    causes[r.get("cause", "?")] = causes.get(r.get("cause", "?"), 0) + 1
check("short cascade completes", len(recs) >= 1,
      f"{len(recs)} walls, {len(evs)} surgeries, causes={causes}")
check("no non-finite energies and no step-limit deaths",
      all(np.isfinite(r["E0"]) for r in recs) and "step-limit" not in causes)

print("\n" + "=" * 62)
print("ALL PASS" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
