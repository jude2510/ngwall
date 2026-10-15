"""Summarize approach-speed census runs.

Usage: python analyze_census.py [DIR]   (reads DIR/census_*.json)

u is the per-sheet approach speed in the symmetric frame; v_c = 0.2598 is the
phi^4 kink-antikink critical velocity (re-separate above, annihilate below).
Contacts are cut at the run's minimum extent (global collapse; later NG
evolution is a re-expansion that field theory would have radiated away) and
wherever energy has drifted by more than 1%.
"""
import glob
import json
import os
import sys
import numpy as np

VC = 0.2598
d = sys.argv[1] if len(sys.argv) > 1 else "."
runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(d, "census_*.json")))]
print(f"{len(runs)} runs in {d}\n")
print(f"{'run':<16}{'t_end':>6}{'|dE/E|':>9}{'t_minext':>9}{'events':>8}"
      f"{'head-on':>8}{'slow':>6}{'distant':>8}{'first distant t, u':>20}")
rows = []
for r in runs:
    S = np.array(r["series"])
    t, ext, eE = S[:, 0], S[:, 2], S[:, 3]
    bad = np.nonzero(np.abs(eE - 1) > 1e-2)[0]
    tcut = min(t[bad[0]] if len(bad) else np.inf, t[np.argmin(ext)])
    ev = [e for e in r["events"] if e["t"] < tcut]
    for e in ev:
        rows.append((r["l_max"], r["omega"], e["dot_med"], e["u_med"], e["any_local"], e["t"]))
    head = [e for e in ev if e["dot_med"] < -0.7]
    dist = sorted((e for e in ev if not e["any_local"]), key=lambda e: e["t"])
    name = f"l{r['l_max']} s{r['seed']} w{r['omega']:g}"
    fd = f"{dist[0]['t']:.3f}, {dist[0]['u_med']:.2f}" if dist else "-"
    print(f"{name:<16}{r['t_end']:>6.2f}{abs(r.get('E_drift', np.nan)):>9.1e}{t[np.argmin(ext)]:>9.3f}"
          f"{len(ev):>8d}{len(head):>8d}{sum(e['u_med'] < VC for e in head):>6d}"
          f"{len(dist):>8d}{fd:>20}")
A = np.array(rows, float) if rows else np.zeros((0, 6))
print("\nContact onsets by crossing geometry (per-event median u):")
for lo, hi, name in [(-1.01, -0.7, "head-on   (n.n < -0.7)"),
                     (-0.7, 0.7, "oblique   (|n.n| < 0.7)"),
                     (0.7, 1.01, "co-moving (n.n > 0.7)")]:
    m = (A[:, 2] >= lo) & (A[:, 2] < hi)
    if not m.any():
        continue
    u = A[m, 3]
    q = np.percentile(u, [10, 25, 50, 75, 90])
    print(f"  {name}: n={m.sum():5d}  slow (u<v_c) {np.mean(u < VC):.2f}   "
          f"u 10/25/50/75/90% = " + " ".join(f"{x:.2f}" for x in q))
    for l in sorted(set(A[m, 0])):
        for w in sorted(set(A[m, 1])):
            mm = m & (A[:, 0] == l) & (A[:, 1] == w)
            if mm.any():
                print(f"      l_max={int(l):2d} omega={w:.1f}: n={mm.sum():4d}  slow "
                      f"{np.mean(A[mm, 3] < VC):.2f}  median u {np.median(A[mm, 3]):.2f}")
m = (A[:, 2] < -0.7) & (A[:, 4] == 0)
if m.any():
    print(f"\nHead-on contacts between DISTANT sheets (the reconnection candidates): "
          f"n={m.sum()}, slow fraction {np.mean(A[m, 3] < VC):.2f}, "
          f"median u {np.median(A[m, 3]):.2f}")
