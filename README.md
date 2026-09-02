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

## Layout

| File | Role |
| --- | --- |
| `membrane.py` | Mesh evolver: RK4, gauge tangential smoothing with energy-restoring Newton rescale, edge-collapse coarsening |
| `surgery.py` | Intercommutation: throat excision, validity-constrained min-area cap/tube triangulation, dilation retries |
| `contact.py` | Self-contact detection: KDTree pairs within `d_c`, k-ring graph exclusion, opposing-normal test, clustering |
| `run_cascade.py` | Recursive population driver; fragmentation trees, retirement causes, census to JSON |
| `run_random.py` | Random initial data: flat-spectrum harmonics `l=2..l_max`, shape `eps`, normal velocity `v_amp`, optional spin `omega` |
| `vtk_out.py` | ParaView output (per-snapshot faces, since topology changes) |
| `run_peanut.py`, `run_kick.py`, `run_sphere.py` | Legacy symmetric testbeds; predate the generic contact/surgery path |

`prod_*.json` are the census records of the production ensemble;
`prod_*_{survived,foam}_wall*.npz` are terminal mesh states, reloadable with
`run_cascade.load_wall` for re-evolution.

## Numerical invariants

- Energy conserved to ~1e-9 relative during smooth evolution. Smoothing was
  historically the only drift source: energy is convex in the gauge parameter
  `u`, so linear interpolation is systematically lossy (Jensen). Fixed by a
  global Newton rescale of `u` after each pass.
- `collapse_short_edges` conserves momentum exactly; its energy change is
  reported as dissipation, by design.
- After surgery the mesh stays a closed manifold: no duplicate faces, every
  edge in exactly two triangles, even Euler characteristic.
- Surgery energy changes are physical (excised throat + seam de-crumpling) and
  are bookkept deliberately, not a conservation failure.

## Design choices that look surprising but are intentional

- **Loop pairing.** Surgery excises the union of *all* simultaneous contact
  clusters at once. A pressed-together front crosses the contact threshold
  everywhere at the same time; drilling one site at a time caused genus runaway.
- **Travel-budget detection cadence.** Contact is checked once accumulated
  `2*dt` reaches `d_c/2`, rather than every N steps. Fixed cadences caused all
  the early failures.
- **Validity-constrained min-area caps.** Existing faces and interior edges
  cost infinity in the DP; infeasibility triggers dilation. Unconstrained
  min-area duplicated triangles over crumpled folds; centroid fans inserted
  spurious area and produced a fake energy *gain*.
- **Coarsening is daughters-only.** It invalidates the pinch-detection vertex
  mask, so combining the two raises deliberately.
- Sub-resolution components are dropped as annihilated debris; genus > 25
  retires a wall as `foam`. Both are modeling decisions.

## Results so far

Production ensemble: 144 runs (16 seeds x omega {0, 0.3, 0.6} x l_max {4, 8,
12}), `t_max=8`, at `nsub=4`. 7049 walls.

- **No permanently stable remnants.** Two walls survived to `t=8`, and both
  died on re-evolution (`t=12.86` and `t=21.69`, the latter after ~80 crossing
  times with energy conserved to every digit). Everything annihilates.
- **But the decay has a deep metastability tail** — compact, flattened,
  high-genus fragments living 50-80+ of their own crossing times.
- **Imposed spin does not create survivors** (0/48 at each of omega=0.3 and
  0.6; both survivors came from omega=0). Spin delays fragmentation; longevity
  instead comes from cascade-distilled specific angular momentum.
- **Foam fraction is 64-95% and rises monotonically with `l_max`**, which is
  circumstantial evidence that foam is partly a resolution artifact. This is
  the main open caveat and the motivation for the convergence campaign.

## Open issues under review

Findings from a code review, **not yet fixed** — recorded here so they are not
rediscovered:

1. `membrane.py` `tri_geom` normalizes the face normal without an epsilon
   guard, so a zero-area triangle yields NaN silently and poisons the whole
   mesh within ~3 RK4 steps. Reproduced. Probable source of the untraced
   production NaN. Every sibling near-singularity in the file is guarded.
2. `contact.py` k-ring exclusion (`k_ring=4`, `d_c=2*lmin`) is structurally
   blind to a coarsened neck pinching shut: at the contact threshold such a
   neck has ~6 vertices around its ring, putting opposite points ~3 mesh-hops
   apart. Scale-invariant, so refinement will not fix it.
3. `contact.py` caches the k-ring table on `id(faces)`, which CPython can reuse
   after `collapse_short_edges` reallocates.
4. The vertex-normal normalization is duplicated three times, guarded in only
   one of them.
5. `max_steps`, `GENUS_MAX`, and `min_faces` are absolute constants that do not
   scale with `lmin`, and will need attention at `nsub=5`.

## Next

Resolution-convergence campaign at `nsub=5` (lmin halved, ~4x vertices). The
deciding observable is foam energy fraction per cell against `nsub=4`: if it
falls, foam is numerical and the survivor population should grow; if flat,
foam is physical.
