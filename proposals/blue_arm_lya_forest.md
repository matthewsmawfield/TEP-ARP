# Observing Proposal Draft — The Lyα Forest Census of High-Redshift Companions

**Working title:** A definitive test of spatial proximity in discordant quasar–galaxy
pairs: the Lyα forest census of four z > 1.9 companions

**Requested configuration (primary):** VLT/UVES, DIC1 390+580 standard setting,
1.0″ slit, or VLT/X-shooter UVB+VIS arms (R ≈ 5,400–6,700). Backup instruments:
Keck/LRIS-B (600/4000 grism), Keck/ESI, Gemini-N/GMOS.

**Total time request:** ~6 hr including overheads (≈ 0.7 night), gray/dark time.

---

## 1. Science case

Discordant quasar–galaxy pairs — compact high-redshift objects lying within
arcseconds to arcminutes of low-redshift galaxies, several on luminous bridges or
filaments — have been debated for fifty years. Two readings remain open:

- **Standard reading.** The companions are background quasars at cosmological
  distance. Photons from a z ≈ 2.1 quasar traverse ~3 Gpc of intervening
  intergalactic medium and must intersect dozens to hundreds of Lyα absorbers.
  The observed-frame Lyα forest band (λ ≈ 3240–3690 Å at z = 2.114) must
  therefore be densely populated.

- **Temporal-well reading (TEP).** If the companion is embedded in a localized
  proper-time well at the host's distance (~90 Mpc for NGC 7319), its redshift is
  a clock-rate offset, not a path length. The sightline samples only the local
  ~90 Mpc, and the forest band should be essentially empty — at most a handful of
  systems from the local large-scale structure.

**There is no intermediate outcome.** At z ≈ 2.1 the mean forest line density is
dN/dz ≈ 15–25 for log N_HI ≳ 14; a background quasar shows of order 30–50
absorbers across the ~450 Å band, while a 90-Mpc-distant companion shows ≲ 2.
The discrimination is not statistical in character: a single spectrum of adequate
resolution and signal-to-noise settles it. A census rate across several targets
converts the test into a population measurement.

**Existing controls.** The reduction pipeline has already demonstrated the
positive control on this exact measurement: for the z = 0.53 companion 3C 232
(toward NGC 3067), a HST/COS spectrum was reduced and a forest census performed —
13 absorption systems were detected at the host-frame positions, confirming that
the pipeline finds a forest where one exists and that the host-frame absorbers
track the galaxy association. A non-detection on the z > 2 companions is
therefore interpretable; it is not a sensitivity statement.

**Archival coverage has been exhausted.** A systematic audit of MAST, ESO, and the
Keck Observatory Archive across all 18 quasar-class companions establishes that
no usable spectrum of the Lyα forest band of any z > 1.7 companion exists:

- NGC 7319 z = 2.114 companion: the 2003 LRIS-B long-slit exposure (programme
  U44L, "Spectroscopy of Quasar Candidates Near AGN Galaxies") was pointed at the
  separate "NGC 7319 ULX" X-ray source 9.4″ away; the slit position angle (115°)
  places the companion ~6.7″ perpendicular off the 0.7″ slit. The 2011 LRIS
  slitmask of the Stephan's Quintet field allocated slitlets only to low-redshift
  star-forming regions; no slitlet covers the companion position. The only
  spectral product overlapping the band is an IUE LWP large-aperture exposure of
  the NGC 7319 nucleus with S/N ≈ 0.3 per pixel in the overlap window — no
  continuum detection, let alone absorption census.
- NGC 3516 and NGC 1073 companions: nearest KOA long-slit pointings lie 94–101″
  away, beyond the 175″ slit half-length.
- WISP 257 129 (NGC 1097, z = 2.334): no covering spectrum in any queried
  archive.

A dedicated observation is the only remaining route.

## 2. Targets

| # | Companion | Host | z_comp | RA (J2000) | Dec (J2000) | R mag | Forest band (Å) | t_exp est. |
|---|-----------|------|--------|------------|-------------|-------|------------------|------------|
| 1 | [VV2006] J223603.7+335824 (Q 2233+337) | NGC 7319 | 2.114 | 22 36 03.70 | +33 58 24.0 | 21.8 | 3242–3690 | ~3.5–4.0 hr |
| 2 | 1WGA J1107.7+7232 | NGC 3516 | 2.10 | 11 07 41.7 | +72 32 35.5 | 18.1 | 3227–3673 | ~0.5 hr |
| 3 | NGC 1073 U1 ([VV96] J024333.6+012222) | NGC 1073 | 1.945 | 02 43 33.6 | +01 22 21.6 | 19.7 | 3066–3490 | ~1.0 hr |
| 4 | WISP 257 129 | NGC 1097 | 2.334 | 02 46 34.4 | −30 32 42.9 | TBC | 3471–3951 | ~1.0 hr |

Target 1 is the highest-value single object: it is the X-ray–detected
companion on the NGC 7319 filament (CXOG J223603.6+335825), the pair carries the
deepest inferred well in the sample (Δφ_int ≈ 1.11), and its forest band is
fully above the atmospheric cutoff. Targets 2–4 are brighter and provide the
rate estimate. NGC 3516 (δ = +72°) requires a northern site; WISP 257 129
(δ = −30°) requires a southern site — the programme is naturally split, and a
partial allocation on either hemisphere remains scientifically decisive for its
covered targets.

## 3. Technical justification

- **Spectral resolution.** Forest absorbers have Doppler b ≈ 20–30 km/s;
  R ≳ 3,000 resolves them cleanly. UVES DIC1 390 (R ≈ 40,000–80,000) or
  X-shooter UVB (R ≈ 6,700) are more than sufficient; LRIS-B 600/4000
  (R ≈ 1,300) is adequate for a census of lines with N_HI ≳ 10^14 cm⁻².
- **Signal-to-noise.** The census requires continuum S/N ≳ 8–10 per resolution
  element at λ ≈ 3,600–3,900 Å. For the faintest target (R = 21.8) this is
  ~3.5–4 hr on UVES/X-shooter under standard conditions; the brighter targets
  need ≲ 1 hr.
- **Calibration.** Standard arcs and flats; no telluric-critical features in the
  band beyond the standard B-star correction. Slit should be oriented to include
  the host nucleus on the slit where feasible (NGC 7319: companion–nucleus
  separation 8.9″), enabling a simultaneous host-frame absorption cross-check.
- **Scheduling.** No time-critical constraints; any dark/gray period with the
  target above airmass 2.

## 4. Pre-registered predictions

Stated prior to any observation:

1. **Standard prediction.** For each companion at cosmological distance, the
   observed-frame band [(1+z)·1050 Å, (1+z)·1215.67 Å] contains a forest at the
   z ≈ 2 mean density, dN/dz ≈ 15–25 above N_HI ≈ 10^14 cm⁻².
2. **TEP prediction.** Each companion at the host distance shows ≲ 2 absorbers
   in the same band, with any residual systems at the host's redshift or
   traceable to the local ~100 Mpc environment.
3. **Falsifiability.** A normal-density forest in the primary target falsifies
   the proximity hypothesis for that pair. A systematically empty forest across
   the target list constitutes the population-level result; the 3C 232 census
   already fixes the pipeline's detection efficiency at a much lower column
   density. The binary outcome is pre-registered: three or more normal-density
   forests among the four targets refutes the proximity reading; a systematically
   empty forest supports it.

## 5. Fallback and archival leverage

The NGC 7603 field additionally motivates a follow-up MUSE/NIRSpec IFU pointing
at the filament knots (existing MUSE coverage terminates 0.9″ short of the knot
positions), which resolves the proximity question kinematically on that system.
That observation is complementary, not a substitute: the Lyα census is the only
test that is binary, model-independent, and achievable in under a night.

---

*Prepared from the TEP-ARP analysis pipeline outputs: coverage audit
(step_29), archival census attempt (step_36), and the companion sample
(steps_00–10). All positions, redshifts, magnitudes, and archive-geometry
statements above trace to `results/outputs/` and `data/processed/`.*
