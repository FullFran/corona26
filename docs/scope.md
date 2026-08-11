# Scope

## The question

> **Can I predict tomorrow's solar corona from today's magnetic field?**

The eclipse of 12 August 2026 provides something unusual: a hard deadline and
an unimpeachable validation dataset. The Sun does not care what our model says.

## Done, defined

V1 is finished when this runs end to end and produces `final_comparison.png`:

```
ADAPT-GONG magnetogram
  → PFSS ensemble (5 × Rss, 12 × ADAPT realisation)
  → open/closed field-line topology
  → electron-density proxy
  → Thomson scattering + LOS integration
  → synthetic K-corona, oriented for Colmenar Viejo
  → comparison against PSI/MAS
  → comparison against the real eclipse
```

## In scope for V1

- Real photospheric magnetic field, with its uncertainty ensemble.
- Global coronal field by PFSS, across a source-surface ensemble.
- Open/closed topology; streamers, helmet streamers, coronal holes.
- Radius- and topology-dependent electron density proxy.
- Exact single-scattering Thomson radiative transfer, written from scratch.
- Total brightness `B` and polarised brightness `pB`.
- Real observer geometry for Colmenar Viejo at totality.
- Quantitative comparison: streamer position angles, not just eyeballs.

## Explicitly out of scope for V1

Full thermodynamic MHD. Riemann solvers. Constrained transport. Resistivity.
Anisotropic conduction. Radiative losses. Wave-turbulence-driven heating.
Self-consistent solar wind. Time evolution. F-corona and E-corona.
Multiple scattering. Absolute photometric calibration.

Each of these is a legitimate research programme. Starting any of them before
20:31 CEST on 12 August guarantees there is no image at all.

## The standing rule

**Every result is published against an explicit baseline, including where we
lose.** Carried over from `snow-mcrt`. Here the baselines are PSI/MAS and the
Sun itself, and we already know we will lose to both. The value is in
characterising *how*.

## Deliverables

| # | Artefact |
|---|---|
| 1 | `magnetogram.png` — the boundary condition actually used, with timestamp |
| 2 | `pfss_rss_*.png` — the source-surface ensemble |
| 3 | `field_lines_3d.png` — open/closed topology |
| 4 | `electron_density_slice.png` — the proxy, visualised |
| 5 | `synthetic_corona.png` — B and pB, north-up and horizon-referenced |
| 6 | `ensemble_spread.png` — which features survive all 60 members |
| 7 | `ours_vs_psi.png` — against the professional prediction |
| 8 | `final_comparison.png` — ours \| PSI \| the real Sun |
| 9 | Reproducible repo + README that states its own approximations |

Items 1–7 are due **before** totality. Item 8 is due after. Item 9 is
continuous.

## Success criteria

V1 succeeds if the pipeline runs, the figures exist, and the approximations
are stated honestly — **regardless of whether the prediction is any good**.

If the corona matches, that is a result. If it does not, the analysis of why
(far-side field? `Rss`? density proxy? missing plasma physics?) is a better
result, because it is the one that teaches something.

The failure mode is not a wrong prediction. It is no prediction.

## After the eclipse

Not before, none of them blocking:

- **V1.1** — own spherical-harmonic PFSS in JAX, validated against sunkit-magex.
- **V1.2** — local GPU (ROCm on gfx1032, or a PyTorch backend).
- **V2** — relax the source surface; compare against outflow-field equilibria.
- **V3** — mini-MHD in JAX on a reduced grid.
- **V4/V5** — thermodynamics; Alfvén-wave turbulent heating.
