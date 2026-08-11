# Plan

Totality at Colmenar Viejo: **2026-08-12, 20:31:xx CEST**, lasting ~37 s,
with the Sun 7.55° above the horizon at azimuth 283°.

Written 2026-08-11. That is roughly **35 hours**, of which maybe 20 are
usable. The plan is built backwards from the deadline, and every phase has a
stated fallback so that a failure at any stage still leaves an image.

## The hard constraint

**A worse image that exists beats a better image that does not.** Every
decision below resolves in favour of shipping.

Ship order, best first:
1. Full ensemble, both `B` and `pB`, quantitative PSI comparison.
2. Single best-guess member, `B` only, visual PSI comparison.
3. One PFSS solve, one render, one PNG.

Level 3 must exist by **18:00 on 12 August**. Everything after that is upside.

---

## Phase A — boundary condition

Fetch the most recent ADAPT-GONG map. Read all 12 realisations. Normalise
coordinates, check flux balance (`∮ Br dA ≈ 0` — a large imbalance means the
map is wrong, and PFSS will happily solve it anyway). Plot `Br(θ, φ)`.
Record the observation timestamp in a manifest.

**Out:** `magnetogram.png`, `data/processed/br_*.npz` + manifests
**Fallback:** GONG synoptic map. Last resort: a sunkit-magex sample map, clearly labelled as not a prediction.
**Risk:** NSO FTP unavailable. Mitigate by fetching *early* and caching.

## Phase B — PFSS

Solve with sunkit-magex. Verify the lower boundary reproduces the input.
Verify sign and coordinate conventions against a known coronal-hole location —
sign errors here are silent and produce a plausible, wrong corona.
Then the ensemble: 5 `Rss` × 12 realisations, cached by content hash.

**Out:** `pfss_rss_{1.3,1.5,2.0,2.5,3.0}.png`, cached solutions
**Fallback:** single realisation, `Rss = 2.0` and `2.5` only.
**Risk:** 60 solves too slow. Measure one solve first, then decide the grid.

## Phase C — topology

Seed the photosphere, trace, classify open/closed, keep expansion factors.
3-D render coloured by polarity and openness.

**Out:** `field_lines_3d.png`
**Fallback:** classification only, no 3-D render. The renderer needs the
topology; the pretty picture is optional.

## Phase D — density proxy

Implement `n0(r)`, normalise against a published K-corona radial profile,
apply topological modulation, inspect 2-D slices for discontinuities.

**Out:** `electron_density_slice.png`
**Fallback:** radial profile only, no modulation. Produces a boring circular
corona — but it produces a corona, and it isolates renderer bugs from
density-proxy bugs.

## Phase E — the renderer

The critical path. Camera and ray construction, van de Hulst coefficients,
LOS quadrature, tiling. NumPy reference first, then JAX. Tests from
`architecture.md` — the point-source limit is the one that catches sign and
factor errors.

**Out:** `synthetic_corona.png` (B and pB, north-up and horizon-referenced)
**Fallback:** 512×512, 256 samples per ray, NumPy, CPU. Minutes, not seconds,
and entirely sufficient for a figure.
**Risk:** this is where the physics bugs are. Budget the most time here and
write the tests before the optimisation.

## Phase F — versus PSI

Fetch the PSI prediction (they publish a Palencia-oriented view — same
totality track, close enough in geometry to be directly comparable). Match
angular scale and orientation. Apply the *same* radial filter to both.
Side-by-side, then streamer position angles on rings at 1.5/2.0/2.5 R☉.

**Out:** `ours_vs_psi.png`, `streamer_angles.csv`
**Fallback:** visual side-by-side, no metrics.

## Phase G — versus the Sun

After totality. Same processing applied to the observed image.

**Out:** `final_comparison.png` — ours | PSI | real
**Note:** at 7.55° altitude the photograph will carry heavy atmospheric
extinction and reddening. Compare *shape*, not colour or absolute brightness.

---

## Timeline

**11 Aug (today).** Repo, docs, environment — done. Then Phase A immediately
(the download is the only step with an external dependency that can fail
outside our control). Phase B baseline solve. Phase C classification.

**11 Aug evening.** Phase D. Start Phase E: camera, geometry, van de Hulst
coefficients with the point-source test passing.

**12 Aug morning.** Finish Phase E. First synthetic corona. **Refetch the
magnetogram** — the freshest possible boundary condition is the single
cheapest accuracy improvement available. Re-solve.

**12 Aug midday.** Ensemble if time allows. Phase F. Figures. README numbers.

**12 Aug 18:00.** Hard stop on code. Push. Whatever exists is the prediction,
and it is timestamped before totality — which is the only thing that makes it
a prediction rather than a retrodiction.

**12 Aug 19:30.** Leave for the observing site. Verify the WNW horizon is
clear well before totality.

**12 Aug 20:31.** Look up. 37 seconds.

**13 Aug onwards.** Phase G, write-up, V1.1.

## Non-negotiable

The prediction must be **committed and pushed before totality**, with the
input magnetogram timestamp recorded. A prediction published after the event
is not a prediction. The git history is the evidence, and that is the whole
scientific value of the exercise.

## Traps

- Writing the JAX PFSS solver first. It is the most interesting part and it is
  not on the critical path.
- Chasing resolution. 512² proves the physics; 2048² only makes it prettier.
- Installing ROCm on eclipse day.
- Tuning the density proxy until it matches PSI. That is fitting to the
  baseline, and it destroys the experiment. Parameters get fixed **before**
  the PSI comparison and do not move afterwards.
- Hiding the approximations. They are the most interesting part of the result.
