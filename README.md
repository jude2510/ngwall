# ngwall — relativistic Nambu-Goto domain wall evolver

Evolves closed domain walls as relativistic Nambu-Goto membranes on a triangle
mesh, with intercommutation surgery when a wall touches itself, and drives
recursive fragmentation cascades.

The scientific question is the domain-wall analog of Copi & Vachaspati
([arXiv:1010.4030](https://arxiv.org/abs/1010.4030)): random string loops
fragment into stable, non-self-intersecting daughters. Do arbitrary closed
domain walls likewise cascade into stable remnants, or does every fragment
collapse? Strings never shrink; walls at rest collapse, so the endpoint may be
qualitatively different.

## Status (October 2026)

**The production census (144 runs, July 2026) is not valid.** A code review
with instrumented runs found:

- **Walls passed through themselves undetected.** A typical random wall first
  self-intersects at t ≈ 0.3, where it reaches v → 1 and folds over; the first
  contact is detected at t ≈ 0.9, with ~180 crossings already present. The
  folds lie within the detector's 4-hop exclusion, and vertex-pair detection at
  `d_c` (a fifth of an edge) lets sheets cross between vertices.
- **Surgery chose the wrong topology** in controlled tests: a clean neck pinch
  became a tunnel (genus 0 → 1), and a pressed pancake went to genus 26.
- **Almost no fragmentation happened.** 6859 of 6881 surgeries produced one
  wall; a typical run is one wall gaining handles until genus > 25 ("foam").
- **Tangential smoothing manufactures angular momentum** (|L| +1.7% in 0.4 time
  units on a spinning wall; RK4 alone conserves it to 1e-10).

**Cusps are physical.** Collapsing walls generically develop cuspidal edges
and swallowtails from smooth initial data, and lattice field theory follows
Nambu-Goto through them
([arXiv:2509.19457](https://arxiv.org/abs/2509.19457),
[arXiv:2605.22974](https://arxiv.org/abs/2605.22974)). The bare evolver — no
smoothing, no coarsening, γ-weighted time step — reproduces the latter
paper's prolate-ellipsoid collapse through its cusps (event times within
~2–4%, energy to 2e-6); the production settings do not.

**Working model:** evolve Nambu-Goto through cusps and fast contacts, and
reconnect only slow head-on contacts (φ⁴ kink-antikink critical velocity
v_c ≈ 0.26), pending the approach-speed census in `diagnostics/`.

## Layout

| Path | Role |
| --- | --- |
| `membrane.py` | Mesh evolver: RK4, tangential smoothing with energy rescale, edge-collapse coarsening |
| `contact.py` | Self-contact detection: KDTree vertex pairs within `d_c`, k-ring exclusion, opposing-normal test, clustering |
| `surgery.py` | Intercommutation: excision, validity-constrained min-area caps and tubes, dilation retries |
| `run_cascade.py` | Recursive population driver; fragmentation trees, retirement causes, census records |
| `run_random.py` | Random initial data: flat-spectrum harmonics `l=2..l_max`, shape `eps`, normal velocity `v_amp`, optional spin `omega` |
| `testbeds.py` | Test geometries (kicked dumbbell, torus, pancake) and the exact sphere solution |
| `vtk_out.py` | ParaView output (per-snapshot faces, since topology changes) |
| `diagnostics/` | Ellipsoid benchmark against arXiv:2605.22974; approach-speed census and its Sol job script |
| `test_fixes.py` | Regression tests for the numerical guards and a cascade smoke test |
| `data/census/` | Census JSONs: `prod_*` (144 production runs), `random_*` (12 earlier pilot runs) — from the invalid runs, kept for reference |
| `data/states/` | Terminal mesh states of those runs; records' `state_file` fields name the old root location, so look files up by name here |

## Numerical behaviour

- RK4 conserves energy to ~1e-9 and angular momentum to ~1e-10 in smooth
  evolution, and matches the exact spherical collapse.
- `tangential_smooth` restores energy after each pass by a global rescale of
  the proper velocity. The resulting ~1e-9 energy conservation is enforced, not
  measured; the rescale inflates |L| and P. At cusps the pass is not gauge and
  can change energy by 10–100% per pass.
- `collapse_short_edges` conserves momentum exactly and reports its energy
  change as dissipation. At a fixed absolute `lmin` it erases cusp structure
  where the wall compresses, and nothing refines the mesh again.
- Surgery output is checked to be a closed manifold (no duplicate faces, every
  edge in exactly two triangles, even Euler characteristic).

## Open issues

1. Contact detection works on vertex pairs and cannot see sheets crossing
   between vertices or folds near cusps.
2. Surgery decides tube-vs-cap with a mean-normal heuristic instead of the
   topology of the excised region.
3. The smoothing energy rescale does not conserve L or P and fails at cusps.
4. No mesh refinement: stretched regions lose resolution.
5. Once sheets may pass through each other, a "fragment" must be defined by
   vacuum domains, not mesh connectivity.
6. Census bookkeeping: JSON written only at the end of a run; state filenames
   embed the full wall id (the 255-byte limit is reached near depth 112);
   records of walls ending in surgery lack `L1`/`P1`/`I1`; seam-coarsening
   energy after surgery is not recorded.
7. Initial data: a flat spectrum per (l, m) at fixed RMS `eps` puts power
   ∝ 2l+1 per l, so `l_max=12` walls are far more crumpled and energetic than
   `l_max=4`; CV instead fix the loop length.

## Next

1. Approach-speed census on Sol (`diagnostics/approach_census.py`, the 144
   production seeds) to settle the reconnection rule.
2. Tests first: surgery topology (neck → two genus-0 walls, pancake → genus 1,
   no crossings left), conservation of E and L, the ellipsoid benchmark.
3. Rewrite contact detection, the surgery decision and the evolver.
4. Rerun the ensemble on Sol, then a resolution-convergence check.
