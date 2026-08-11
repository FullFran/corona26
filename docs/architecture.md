# Architecture

## Shape

A pipeline of pure stages. Each stage takes typed inputs, returns arrays plus a
manifest, and writes nothing outside its output directory. Every stage is
runnable and inspectable on its own; a broken renderer must not require
re-downloading a magnetogram.

```
corona26/
├── data/          boundary conditions      ADAPT / GONG / HMI  → Br(θ, φ)
├── magnetic/      field reconstruction     Br                  → B(r, θ, φ)
├── plasma/        density proxy            B + topology        → ne(r, θ, φ)
├── radiation/     Thomson renderer         ne + camera         → I(x, y)
├── validation/    baselines and metrics    I + PSI + real      → numbers
└── plotting/      figures                  everything          → PNG
```

### Module responsibilities

| Module | Contents |
|---|---|
| `data/adapt.py` | Fetch ADAPT via `Fido`, select realisation, normalise to a sunpy `GenericMap` |
| `data/gong.py` | GONG synoptic fallback |
| `data/hmi.py` | HMI cross-check (post-eclipse) |
| `magnetic/pfss.py` | sunkit-magex wrapper; `Rss` ensemble; caching |
| `magnetic/harmonics.py` | *V1.1* own JAX spherical-harmonic solver |
| `magnetic/trace.py` | Seeding, tracing, open/closed classification, expansion factor |
| `plasma/radial_density.py` | `n0(r) = A r⁻² + B r⁻⁴ + C r⁻⁶` |
| `plasma/topology_density.py` | Modulation by topology; smooth blend |
| `radiation/thomson.py` | van de Hulst `A,B,C,D`; `I_tan`, `I_rad`, `B`, `pB` |
| `radiation/camera.py` | Observer geometry, `P`/`B0`/`L0`, ray construction |
| `radiation/render.py` | Tiled LOS integration; `jit` + `vmap`; CPU/GPU backends |
| `validation/psi.py` | Fetch and align the PSI prediction |
| `validation/metrics.py` | Streamer position angle, angular correlation, IoU, SSIM |
| `plotting/figures.py` | Radial filter; consistent styling across all three images |

## Decisions

### D1 — sunkit-magex for V1 PFSS, own solver in V1.1

The spherical-harmonic expansion is the most tempting thing to write first and
the most reliable way to miss the deadline. Using the maintained library first
and validating an own implementation against it afterwards is both faster and a
better story than the reverse.

### D2 — the renderer is ours, always

This is where the physics is exact and where the compute lives. Writing it is
the point of the project. No library shortcut here.

### D3 — NumPy first, JAX behind a seam

`radiation/render.py` takes an array namespace. The physics is written once
against an `xp`-style interface; NumPy for correctness and small images, JAX
for `jit`/`vmap`/GPU on the large render. The NumPy path is the reference
implementation the JAX path is checked against — the same discipline
`snow-mcrt` used with closed-form solutions.

### D4 — GPU is Kaggle today, local later

**Verified on this machine:** Navi 23 / gfx1032 (RX 6600), `amdgpu` kernel
module loaded, **no ROCm userspace installed** — no `/opt/rocm`, no
`rocminfo`, no `hipcc`.

Installing ROCm is multi-gigabyte and slow, gfx1032 is not officially
supported (it needs `HSA_OVERRIDE_GFX_VERSION=10.3.0`), and there is a known
segfault regression on gfx1031/1032 in recent ROCm releases. Spending eclipse
day on a driver stack is the wrong trade.

So: **NumPy/JAX-CPU locally at reduced resolution, Kaggle T4 for the full
render.** Kaggle is compute, not architecture — everything must run on CPU at
lower resolution. Local GPU is V1.2; if it happens, PyTorch-ROCm is a more
travelled path on this card than JAX-ROCm.

### D5 — float64 for validation, float32 for production

`jax_enable_x64` on while checking the renderer against analytic limits, then
measure whether float32 changes the image. It almost certainly will not — the
dynamic range is handled by log scaling, not by mantissa bits — but that is a
measurement, not an assumption. No float16/bfloat16 in the physics.

### D6 — caching by content hash

PFSS solves are the slow CPU step and the ensemble runs 60 of them. Cache to
`data/processed/` keyed on a hash of (magnetogram id, realisation, `Rss`,
`nr`, `lmax`). Re-runs of downstream stages must never re-solve.

### D7 — every artefact carries a manifest

Every output PNG/NPZ is written next to a JSON manifest recording input file,
observation timestamp, ensemble member, all parameters, and code version.
An eclipse prediction whose input magnetogram date is unknown is worthless.
Same pattern as `snow-mcrt`'s `data/reference/*.manifest.json`.

## Environment

Verified working:

```
Python 3.12  (system python is 3.14 — scientific wheels are not reliable there)
sunpy 8.0.0 · sunkit-magex 1.1.0 · astropy 8.0.1 · numpy 2.5.2 · scipy 1.18.0
```

Everything runs through `uv`. Verified present: `pfss.Input/Output/pfss`,
`pfss.tracing.{FortranTracer, PerformanceTracer, PythonTracer}`,
`FieldLine.{is_open, polarity, expansion_factor, solar_footpoint,
source_surface_footpoint}`, `sunpy.net.dataretriever.ADAPTClient`.

## Testing

Behaviour-first, following the project's standing TDD discipline. The renderer
is the part that must be tested hardest, because it is the part we wrote:

- **Point-source limit.** As `r → ∞`, `Ω → 0`, and the van de Hulst kernel must
  reduce to the classical point-source Thomson formula `∝ (1 + cos²χ)`.
- **Thomson surface.** For a given LOS, `pB` must peak where the scattering
  angle is 90°.
- **Radial profile.** A spherically symmetric `ne` must render a spherically
  symmetric image, and its radial brightness slope must match the analytic
  result for a power-law density.
- **Quadrature convergence.** Doubling LOS samples changes the image by less
  than a stated tolerance.
- **PFSS sanity.** Solved `Br` at the lower boundary reproduces the input
  magnetogram; `∇·B ≈ 0`; total open flux is positive and finite.
- **Backend agreement.** NumPy and JAX paths agree to float32 precision.
- **Geometry.** `P`, `B0`, `L0` at a known epoch match published ephemerides.
