# References

## Direct precedent

**Mikić, Downs, Linker et al. (2018)** — *Predicting the corona for the
21 August 2017 total solar eclipse*, Nature Astronomy 2, 913.
A global thermodynamic MHD prediction published before the event and compared
against the observed corona afterwards. This is the paper corona26 is a small,
honest imitation of.
<https://www.predsci.com/corona/aug2017eclipse/home.php>

**Predictive Science — 2026 eclipse prediction.** Time-dependent MHD, updated
in near-real-time, HMI vector magnetograms with Solar Orbiter/PHI far-side
assimilation. Publishes a view oriented for Palencia, Spain, plus synthetic
coronagraph views (LASCO C2, Metis, COR2) and PUNCH products.
<https://eclipse.predsci.com/> · archive: <https://www.predsci.com/corona/>

**MAS** — the MHD code behind the above. <https://github.com/predsci/MAS>

## PFSS

**Altschuler & Newkirk (1969)** — the original potential-field source-surface
model, and the origin of the `Rss = 2.5 R☉` convention.

**Schatten, Wilcox & Ness (1969)** — independent, concurrent formulation.

**Benavitz, Boe & Habbal (2024)** — *Total Solar Eclipse White Light Images as
a Benchmark for PFSS Coronal Magnetic Field Models*. Uses eclipse images to
constrain `Rss` and shows the best value varies with the solar cycle. The
closest published precedent to our `Rss` ensemble.
<https://arxiv.org/abs/2408.16149>

**Wang & Sheeley (1990)** — flux-tube expansion factor `f_s` as a predictor of
solar wind speed. Motivates using `expansion_factor` to modulate density
rather than a binary open/closed flag.

**sunkit-magex** — maintained successor to `pfsspy`; the V1 PFSS engine.
<https://docs.sunpy.org/projects/sunkit-magex/en/latest/>

**NSO PFSS products** — independent cross-check of our field.
<https://nso.edu/data/nisp-data/pfss/>

## Thomson scattering

**Minnaert (1930)** — first inclusion of limb darkening in coronal scattering.

**van de Hulst (1950)** — the `A, B, C, D` coefficients accounting for the
finite, limb-darkened solar disk. The core of `radiation/thomson.py`.

**Billings (1966)** — *A Guide to the Solar Corona*. The formulation in
standard use; the source of the closed forms we implement.

**Inhester (2015)** — *Thomson Scattering in the Solar Corona*. Modern,
self-contained derivation with explicit formulas for `I_tan` and `I_rad`.
Primary implementation reference.
<https://arxiv.org/abs/1512.00651>

**Howard & DeForest (2012)** — *The Thomson Surface I: Reality and Myth*, and
DeForest et al. (2013) *II: Polarization*. Clarify what the "Thomson surface"
does and does not mean — useful for the `pB` sanity test.

## Coronal electron density

**Baumbach (1937)**, **Allen (1947)** — the classic
`A r⁻² + B r⁻⁴ + C r⁻⁶`-type empirical fits to eclipse brightness.

**Saito, Poland & Munro (1977)** — density model separated by coronal
structure (streamer vs hole), the empirical antecedent of our topological
modulation.

**Guhathakurta et al. (1996)** — density in polar coronal holes vs streamers,
useful for choosing the open/closed contrast factors.

## Magnetogram data

**ADAPT** (Air Force Data Assimilative Photospheric flux Transport) —
GONG magnetograms plus a surface flux transport model; 12 realisations per
file. Primary boundary condition.
<https://nso.edu/data/nisp-data/adapt-maps/>

**GONG** — near-real-time magnetograms and synoptic maps.
<https://nso.edu/data/nisp-data/magnetograms/>

**SDO/HMI** — high-quality magnetograms; the source PSI use.
<https://jsoc.stanford.edu/HMI/>

## Beyond V1

**Lionello, Linker & Mikić (2014)** — *Validating a Time-Dependent
Wave-Turbulence-Driven Model of the Solar Wind*.
<https://arxiv.org/abs/1402.4188>

**Downs et al. (2016)** — *Closed-Field Coronal Heating Driven by Wave
Turbulence*. <https://arxiv.org/abs/1610.02113>

**AWSoM-R / SWMF** — Alfvén-wave solar model; 2024 eclipse prediction.
<https://arxiv.org/abs/2503.10974>

**Rice et al. (2026)** — *Global Coronal Equilibria with Solar Wind Outflow
II*. Replaces the artificial source surface with a wind-outflow equilibrium.
The natural V2. <https://arxiv.org/html/2603.22159v1>

## Software

**SunPy** — solar data access and coordinate frames. <https://sunpy.org>
**Astropy** — units, time, ephemerides.
**JAX** — `jit`, `vmap`, GPU for the LOS render. <https://docs.jax.dev/>

## Sibling project

**snow-mcrt** — Monte Carlo photon transport in snow, validated against
closed-form radiative transfer. Same scattering physics in the opposite
optical-depth regime: `τ ≫ 1` and multiple scattering there, `τ ≈ 10⁻⁶` and
single scattering here.
<https://github.com/FullFran/snow-mcrt>
