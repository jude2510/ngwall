"""Summarize approach-speed census runs.

Usage: python analyze_census.py DIR [DIR ...]   (reads DIR/census_*.json)

u is the per-sheet approach speed in the symmetric frame; v_c = 0.2598 is the
phi^4 kink-antikink critical velocity (re-separate above, annihilate below).
Geometry uses |n.n| only (the sign of a mesh normal carries no physics for a
Z2 wall): nearly parallel (> 0.7) vs oblique. Separation along the surface
(v2 runs) is binned against the shortest wrinkle wavelength 2 pi / l_max and
the wall scale; v1 runs only carry a within-6-mesh-hops flag. Contacts are
counted up to fixed times (the time of minimum extent is not a collapse
marker for random walls).
"""
import glob
import json
import os
import sys
import numpy as np

VC = 0.2598
T_CUTS = (1.0, 1.5, 2.0)

runs = []
for d in sys.argv[1:] or ["."]:
    runs += [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "census_*.json")))]
if not runs:
    sys.exit("no census_*.json found")


def dist_class(r, e):
    """'sub-wrinkle' / 'wrinkle-to-wall' / 'distant' (v2), or 'local' /
    'distant' from the 6-hop flag (v1)."""
    if "geo_min" in e:
        lam = 2 * np.pi / r["l_max"]
        if e["geo_min"] < lam / 2:
            return "sub-wrinkle"
        return "distant" if e["geo_min"] >= r.get("geo_limit", 1.0) else "wrinkle-to-wall"
    return "local(<=6 hops)" if e["any_local"] else "distant"


def adot(e):
    return e.get("adot_med", abs(e["dot_med"]))


groups = {}
for r in runs:
    groups.setdefault((r.get("nsub", 4), r["l_max"]), []).append(r)
drift = np.array([abs(r["E_drift"]) for r in runs])
print(f"{len(runs)} runs; |dE/E| median {np.median(drift):.1e}, max {drift.max():.1e}")

for (nsub, l_max), rs in sorted(groups.items()):
    t_end = min(r["t_end"] for r in rs)
    tc = max(c for c in T_CUTS if c <= t_end + 1e-9) if t_end >= T_CUTS[0] else t_end
    ev = [(r, e) for r in rs for e in r["events"] if e["t"] < tc]
    print(f"\n== nsub={nsub} l_max={l_max}: {len(rs)} runs, contacts up to t < {tc:g}: "
          f"{len(ev) / len(rs):.1f} per run")
    classes = sorted({dist_class(r, e) for r, e in ev})
    for cls in classes:
        for par, pname in ((True, "nearly parallel"), (False, "oblique")):
            sel = [e for r, e in ev if dist_class(r, e) == cls and (adot(e) > 0.7) == par]
            if not sel:
                continue
            u = np.array([e["u_med"] for e in sel])
            line = (f"   {cls:<16} {pname:<16} {len(sel) / len(rs):7.2f}/run  slow {np.mean(u < VC):.2f}"
                    f"  u 25/50/75% {np.percentile(u, 25):.2f} {np.median(u):.2f} {np.percentile(u, 75):.2f}")
            if "gam_med" in sel[0]:
                g = np.array([e["gam_med"] for e in sel])
                line += f"  gamma median {np.median(g):.1f}, >3: {np.mean(g > 3):.2f}"
            print(line)

# side-by-side resolution comparison for seeds run at more than one nsub
keys = {}
for r in runs:
    keys.setdefault((r["l_max"], r["seed"], r["omega"]), {})[r.get("nsub", 4)] = r
pairs = {k: v for k, v in keys.items() if len(v) > 1}
if pairs:
    ns = sorted({n for v in pairs.values() for n in v})
    print(f"\n== resolution comparison ({len(pairs)} initial conditions run at nsub {ns}),"
          f" contacts per run up to t < {min(r['t_end'] for v in pairs.values() for r in v.values()):.2f}")
    tc = min(r["t_end"] for v in pairs.values() for r in v.values())
    for l_max in sorted({k[0] for k in pairs}):
        out = []
        for n_ in ns:
            rs = [v[n_] for k, v in pairs.items() if k[0] == l_max and n_ in v]
            ev = [(r, e) for r in rs for e in r["events"] if e["t"] < tc]
            by = {}
            for r, e in ev:
                by.setdefault(dist_class(r, e), []).append(e["u_med"])
            out.append(f"nsub={n_}: " + ", ".join(
                f"{c} {len(u) / len(rs):.1f}/run (slow {np.mean(np.array(u) < VC):.2f})"
                for c, u in sorted(by.items())))
        print(f"   l_max={l_max:2d}  " + "  |  ".join(out))
