"""Cascade driver: evolve a population of walls with generic contact surgery.

Each wall evolves independently (fragments fly apart; wall-wall re-collision
is neglected, as in Copi-Vachaspati). When the ContactDetector fires, the
wall undergoes contact_surgery and its pieces recurse. A wall retires when
it hits the v -> 1 guard (a caustic: the zero-thickness description ends,
physically annihilation) or survives to t_max. The full fragmentation tree
is returned as a list of wall records plus surgery events.

Initial data helpers: kicked dumbbell (import from run_kick), torus, oblate.
"""

import numpy as np
from membrane import Membrane
from contact import ContactDetector
from surgery import contact_surgery
from vtk_out import VtkSeries, wall_block


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
    from membrane import icosphere
    X, F = icosphere(nsub)
    X = X.copy()
    X[:, 2] *= c
    return X, F


def kicked_oblate(c=0.3, v0=0.6, w=0.6, nsub=4):
    """Pancake with its flat faces kicked toward each other.

    Kick along the local normal (transverse gauge exact), magnitude
    -v0 exp(-rho^2/w^2) |n.z| -- full strength at the face centers, zero at
    the rim -- so face-face contact (the tunnel-branch test) happens before
    the rim cusp.
    """
    from contact import vertex_normals
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


def _speed(mem):
    V = mem.velocity(mem.X, mem.p)
    return np.sqrt(np.einsum('ij,ij->i', V, V))


def ang_mom(mem):
    """Total angular momentum L = sum x cross p (conserved by evolution;
    changed only by surgery, which removes material)."""
    return np.cross(mem.X, mem.p).sum(axis=0)


def inertia_eigs(mem):
    """Eigenvalues (ascending) of the energy-weighted inertia tensor about
    the center of energy -- the CV planarity/shape statistic for walls."""
    m = mem.sigma * mem.vertex_areas(mem.X)
    Ev = np.sqrt(m ** 2 + np.einsum('ij,ij->i', mem.p, mem.p))
    xc = (Ev[:, None] * mem.X).sum(axis=0) / Ev.sum()
    d = mem.X - xc
    r2 = np.einsum('ij,ij->i', d, d)
    I = (Ev * r2).sum() * np.eye(3) - np.einsum('i,ij,ik->jk', Ev, d, d)
    if not np.isfinite(I).all():   # NaN p poisons the whole tensor; some
        return np.full(3, np.nan)  # numpy builds raise on it, others don't
    return np.linalg.eigvalsh(I)


def load_wall(path):
    """Rebuild a Membrane from a survivor state dump (see cascade's
    save_survivors) so stability candidates can be re-evolved."""
    d = np.load(path)
    mem = Membrane(d["X"], d["faces"])
    mem.p = d["p"].copy()
    mem.t = float(d["t"])
    return mem


GENUS_MAX = 25      # above this a wall is resolution-scale foam (see below)


def cascade(mem, t_max, lmin, d_c=None, grace=0, cfl=0.2,
            vstop=0.999999, smooth_every=5, smooth_lam=0.25, coarsen_every=20,
            max_steps=None, wall_id="0", records=None, events=None,
            vtk_dir=None, dt_snap=0.02, extent_min=None, save_states=None,
            genus_max=GENUS_MAX):
    """Evolve one wall to contact/caustic/t_max, recursing on fragments.

    Returns (records, events): records have one entry per wall (birth/death
    time, cause, energy, vertex count, genus, extent); events one entry per
    surgery. lmin is the absolute coarsening scale, inherited by fragments;
    d_c defaults to 2*lmin (the resolution floor = contact condition).

    Detection cadence is an adaptive travel budget: two sheets close at most
    2*dt per step (c = 1), so checking whenever the accumulated 2*dt reaches
    d_c/2 guarantees no pair can cross undetected -- worst case they arrive
    still d_c/2 apart and flag on the next check. A fixed every-N-steps
    cadence has no such bound and lets fast sheets interpenetrate.

    Retirement is cusp-tolerant: v > vstop on a LARGE wall is a transient
    cusp (walls, like strings, momentarily reach v = 1 at points during
    smooth evolution) and evolution continues, with coarsening dissipating
    the cusp-generated sub-grid structure; only a wall that is relativistic
    AND small (extent < extent_min, default 20*lmin) retires as a caustic.
    """
    if records is None:
        records = []
    if events is None:
        events = []
    d_c = d_c if d_c is not None else 2 * lmin
    extent_min = extent_min if extent_min is not None else 20 * lmin
    # dt ~ cfl*min_edge scales with lmin, so halving the resolution doubles
    # the steps needed to reach t_max. A fixed budget silently truncates deep
    # branches as "step-limit" at finer resolution; scale it instead.
    if max_steps is None:
        max_steps = int(10 * t_max / (cfl * lmin))
    det = ContactDetector()
    series = VtkSeries(vtk_dir, f"wall{wall_id}") if vtk_dir else None
    genus = getattr(mem, "genus", None)
    rec = dict(id=wall_id, born=mem.t, E0=mem.energy(), nv0=len(mem.X),
               genus=genus, Pv0=mem.p.sum(axis=0), L0=ang_mom(mem),
               I0=inertia_eigs(mem))
    nstep = 0
    tsnap = mem.t
    E0 = rec["E0"]
    travel = d_c        # newborn walls are checked immediately: their other
    # contact sites are already within d_c and must not evolve unhandled

    def extent():
        return (mem.X.max(axis=0) - mem.X.min(axis=0)).max()

    def retire(cause):
        rec.update(died=mem.t, cause=cause, E1=mem.energy(),
                   P1=np.linalg.norm(mem.p.sum(axis=0)), nv1=len(mem.X),
                   Pv1=mem.p.sum(axis=0), L1=ang_mom(mem),
                   I1=inertia_eigs(mem),
                   extent=extent(), steps=nstep,
                   drift=abs(mem.energy() / E0 - 1.0))
        if cause in ("survived", "foam") and save_states:
            fn = f"{save_states}_{cause}_wall{wall_id}.npz"
            np.savez(fn, X=mem.X, p=mem.p, faces=mem.faces, t=mem.t)
            rec["state_file"] = fn
        records.append(rec)
        if series:
            series.close()
        print(f"[{wall_id}] {cause} at t = {mem.t:.4f}: E = {rec['E1']:.4f}, "
              f"extent = {rec['extent']:.3f}, verts = {rec['nv1']}, "
              f"steps = {nstep}")

    # a wall born past genus_max is resolution-scale foam: sheets folded
    # within d_c everywhere, annihilating in the field theory, beyond the
    # zero-thickness description at this resolution. Retire it (its energy
    # is the census's "ended as foam" column) instead of letting
    # micro-surgeries churn until one becomes unresolvable.
    if genus is not None and genus > genus_max:
        retire("foam")
        return records, events

    while True:
        # a NaN in X or p (rare degenerate coarsening / smoothing edge case,
        # only seen deep in violent cascades) otherwise propagates silently
        # until step-limit and then crashes census bookkeeping, taking the
        # whole recursive tree's records with it. Catch it here: retire the
        # fragment honestly and log where it happened so the source can be
        # traced if it recurs (e.g. under mesh refinement).
        if not (np.isfinite(mem.X).all() and np.isfinite(mem.p).all()):
            nbx = int((~np.isfinite(mem.X)).any(axis=1).sum())
            nbp = int((~np.isfinite(mem.p)).any(axis=1).sum())
            print(f"[{wall_id}] NON-FINITE at t={mem.t:.4f} step={nstep} "
                  f"gen={wall_id.count('.')} nv={len(mem.X)}: "
                  f"{nbx} bad X, {nbp} bad p")
            retire("nonfinite")
            return records, events
        if series and mem.t >= tsnap - 1e-12:
            series.add(mem.t, [wall_block(mem.X, mem.faces, _speed(mem), 0)])
            tsnap += dt_snap
        if mem.vmax() > vstop and extent() < extent_min:
            retire("caustic")
            return records, events
        if mem.t >= t_max:
            retire("survived")
            return records, events
        if nstep >= max_steps:
            retire("step-limit")
            return records, events
        dt = cfl * mem.min_edge(mem.X)
        mem.step(dt)
        travel += 2.0 * dt
        nstep += 1
        if smooth_every and nstep % smooth_every == 0:
            mem.tangential_smooth(smooth_lam)
        if lmin and nstep % coarsen_every == 0:
            mem.collapse_short_edges(lmin)
        if nstep >= grace and travel >= 0.5 * d_c:
            travel = 0.0
            clusters = det.find(mem, d_c)
            if clusters:
                # one surgery for the union of all simultaneous contact
                # sites: a pressed-sheet front crosses d_c nearly everywhere
                # at once, and handling sites one by one drills disconnected
                # holes instead of reconnecting the whole front
                L_par = ang_mom(mem)
                try:
                    walls, rep = contact_surgery(
                        mem, np.concatenate(clusters), d_c=d_c)
                except ValueError as ex:
                    # unresolvable contact zone: the wall IS foam (see above)
                    rec["surgery_error"] = str(ex)
                    retire("foam")
                    return records, events
                rep["dL"] = np.linalg.norm(
                    L_par - sum(ang_mom(w) for w in walls)) \
                    if walls else np.linalg.norm(L_par)
                rec.update(died=mem.t, cause=rep["mode"], E1=rep["E_before"],
                           nv1=len(mem.X), extent=extent(), steps=nstep,
                           drift=abs(rep["E_before"] / E0 - 1.0))
                records.append(rec)
                if series:
                    series.add(mem.t, [wall_block(mem.X, mem.faces,
                                                  _speed(mem), 0)])
                    series.close()
                events.append(dict(t=mem.t, parent=wall_id, **{
                    k: rep[k] for k in
                    ("mode", "dE", "dP", "dL", "E_removed", "E_debris",
                     "genus")}))
                deb = (f", debris E = {rep['E_debris']:.4f}"
                       if rep["E_debris"] else "")
                print(f"[{wall_id}] {rep['mode']} at t = {mem.t:.4f}: "
                      f"{len(walls)} pieces, genus {rep['genus']}, "
                      f"dE/E = {rep['dE'] / rep['E_before']:+.2e}, "
                      f"dP/E = {rep['dP'] / rep['E_before']:.2e}{deb}")
                for i, w in enumerate(walls):
                    if lmin:
                        w.collapse_short_edges(lmin)   # seam cleanup
                    cascade(w, t_max, lmin, d_c, grace, cfl,
                            vstop, smooth_every, smooth_lam, coarsen_every,
                            max_steps, f"{wall_id}.{i}", records, events,
                            vtk_dir, dt_snap, extent_min, save_states,
                            genus_max)
                return records, events


def summarize(records, events):
    print(f"\n=== cascade: {len(records)} walls, {len(events)} surgeries ===")
    for r in sorted(records, key=lambda r: r["id"]):
        g = f", genus {r['genus']}" if r.get("genus") is not None else ""
        print(f"  {r['id']:<8s} t: {r['born']:.3f} -> {r['died']:.3f}  "
              f"{r['cause']:<10s} E: {r['E0']:.3f} -> {r['E1']:.3f}{g}")
    if events:
        dE = sum(e["dE"] for e in events)
        print(f"  total surgery dE = {dE:+.4f}")


if __name__ == "__main__":
    from run_kick import kicked_peanut
    mem = kicked_peanut()
    F = mem.faces
    e = np.linalg.norm(mem.X[F[:, 0]] - mem.X[F[:, 1]], axis=1)
    lmin = 0.1 * np.median(e)
    print(f"kicked dumbbell via generic cascade (lmin = {lmin:.4f}, "
          f"d_c = {2 * lmin:.4f})")
    records, events = cascade(mem, t_max=1.6, lmin=lmin)
    summarize(records, events)
