# corona26

Predicting the corona of the 12 August 2026 total solar eclipse from real
photospheric magnetic-field data — PFSS reconstruction, a topology-informed
electron-density proxy, and an exact Thomson-scattering renderer.

> **Status: planning complete, implementation starting.**
> No results yet. Every figure referenced below is a commitment, not a claim.
> Written 11 August 2026, the day before totality.

## The question

> **Can I predict tomorrow's solar corona from today's magnetic field?**

Tomorrow evening I am going to stand in a field near Colmenar Viejo, north of
Madrid, and for 37 seconds the Moon will remove the Sun and leave the corona
visible. Today the Sun's magnetic field has already been measured.

Those two facts are separated by about thirty hours and a great deal of
physics. This repository is an attempt to cross that gap and then find out, by
looking up, how badly I got it wrong.

The appeal is the deadline. Most models are validated against archived data
you could, in principle, have peeked at. This one gets graded by an event that
has not happened yet, against a dataset nobody can argue with.

## How this started

I built [`snow-mcrt`](https://github.com/FullFran/snow-mcrt) to answer whether
I could simulate light in a disordered medium correctly enough to be believed —
photons scattering thousands of times inside snow, `τ ≫ 1`, multiple scattering
as the whole problem.

The corona is the same physics standing on its head. Free electrons instead of
ice grains, and an optical depth of about **10⁻⁶**. A photon that scatters
once effectively never scatters again. There is no transport equation to
solve — just one exact integral along each line of sight, over hundreds of
millions of sample points that do not talk to each other.

Which is a way of saying: the hard part is not the radiation. It is knowing
where the electrons are. And nothing measures that directly.

## What this is

A five-stage pipeline, each stage stating its own approximation:

```
ADAPT-GONG magnetogram          real measurement, far side modelled
  → PFSS ensemble               current-free field, 5 source-surface radii
  → open/closed topology        streamers, coronal holes, the streamer belt
  → electron density proxy      empirical — this is the weak point, and it is labelled
  → Thomson scattering + LOS    exact, single-scattering, written from scratch
  → synthetic K-corona          oriented for Colmenar Viejo, 20:31 CEST
```

Full derivations, every equation, and a table of all eight approximations
ranked by risk: [`docs/physics.md`](docs/physics.md).

## What it is not

**This is not a thermodynamic MHD model.** It does not solve for plasma. It
has no heating, no wind, no time evolution. Where
[Predictive Science](https://eclipse.predsci.com/) run a time-dependent MHD
model assimilating far-side magnetograms from Solar Orbiter/PHI, this runs a
static potential-field extrapolation and multiplies the answer by an empirical
density profile.

They will be better than us. That is the point of having them as a baseline.
The interesting question is *where* the cheap model breaks and by how much.

## The uncomfortable part, stated in advance

The dominant error is not the renderer and it is not even PFSS. **It is that
we cannot see the far side of the Sun.**

Global magnetic maps reconstruct the hidden hemisphere with a surface
flux-transport model. In August 2026 the Sun is near cycle maximum and active
regions evolve in days. If a region large enough to anchor a helmet streamer
emerged out of sight, our streamer belt is wrong, and no amount of resolution
fixes it.

So we do not publish one image. ADAPT ships **12 realisations** of the boundary
condition; we run **5 source-surface radii**. That is a 60-member ensemble, and
its spread is the error bar. A streamer that survives all 60 members is a
prediction. One that appears in three is noise.

## Where the compute goes

One place: the line-of-sight integral. A 1024² image at 512 samples per ray is
~5×10⁸ independent kernel evaluations with no coupling between them — an
embarrassingly parallel map-reduce, and the one part of the pipeline where a
GPU is obviously correct.

Written against NumPy first as the reference implementation, then JAX
`jit`/`vmap` for the same equations at speed, rendered in tiles.
The two backends must agree.

Local hardware is an RX 6600 (gfx1032) with no ROCm userspace installed and a
known segfault regression on that architecture, so the full render runs on a
Kaggle T4. **Kaggle is compute, not architecture** — everything runs on CPU at
reduced resolution, and does so before it runs anywhere else.

## Where we will be looking

Computed with astropy/sunpy for Colmenar Viejo (40.659° N, 3.768° W, 1004 m)
at 2026-08-12 18:31 UTC:

| | |
|---|---|
| Solar north position angle `P` | 15.19° |
| Heliographic latitude of disk centre `B0` | 6.51° |
| Carrington longitude `L0` | 224.53° |
| Solar angular radius | 15.78′ |
| **Sun altitude** | **7.55°** |
| **Sun azimuth** | **283.04°** (WNW) |

`B0 = 6.51°` means we look slightly down on the north pole: the southern polar
coronal hole is foreshortened, the northern one better exposed. `P = 15.19°`
sets the rotation of the whole corona on the sky — an image with solar north up
is not what a camera on a tripod records, so both are produced.

And 7.55° of altitude means the corona will appear low over the horizon, deep
in atmosphere, in the direction of the Sierra de Guadarrama. The observing site
has to be picked against a horizon profile, not a map.

## Validation

Three baselines, in increasing order of severity:

1. **Predictive Science / MAS** — the professional MHD prediction, which
   publishes a Spain-oriented view.
2. **The Sun**, at 20:31:xx CEST on 12 August. No appeal.
3. **Ourselves** — the 60-member ensemble spread.

Compared with numbers, not vibes: streamer position angles extracted from
angular intensity profiles on rings at 1.5, 2.0 and 2.5 R☉, plus angular
correlation and IoU of binarised bright regions. Identical radial filtering
applied to all three images, because comparing differently-processed pictures
is the easiest way to produce a meaningless figure.

The standing rule from `snow-mcrt` applies: **publish against an explicit
baseline, including where we lose.** We already know we lose. The result is
characterising how.

## The prediction is timestamped

The synthetic corona and the magnetogram it came from are committed and pushed
**before totality**, with the input observation time recorded in a manifest.
The git history is the evidence. A prediction published afterwards is not a
prediction.

## Success criteria

V1 succeeds if the pipeline runs, the figures exist and the approximations are
stated honestly — **regardless of whether the prediction is any good.**

If the corona matches, that is a result. If it does not, working out why
(far-side field? source surface? density proxy? missing plasma physics?) is a
better one. The failure mode is not a wrong prediction. It is no prediction.

## Documentation

| | |
|---|---|
| [`docs/physics.md`](docs/physics.md) | Every equation, every approximation, ranked by risk |
| [`docs/scope.md`](docs/scope.md) | What V1 does and explicitly does not do |
| [`docs/architecture.md`](docs/architecture.md) | Pipeline design and the seven decisions behind it |
| [`docs/plan.md`](docs/plan.md) | Phases, fallbacks, and the hour-by-hour run to totality |
| [`docs/bib/references.md`](docs/bib/references.md) | Sources, from Minnaert 1930 to Rice 2026 |

## Running it

```bash
uv sync
uv run python -m corona26.pipeline --help    # not yet implemented
```

Python 3.12 (the 3.14 system interpreter has no reliable scientific wheels),
sunpy 8.0, sunkit-magex 1.1, astropy 8.0.

## Licence

MIT.
