# Temporal Equivalence Principle: Localized Proper-Time Gradients and the Spatial Proximity of Discordant Pairs

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22938153.svg)](https://doi.org/10.5281/zenodo.22938153)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

![TEP-ARP: Localized Proper-Time Gradients and the Spatial Proximity of Discordant Pairs](site/public/image.png)

**Author:** Matthew Lukin Smawfield  
**Version:** v0.1 (Pasadena)  
**Date:** First published: 24 September 2026 · Last updated: 24 September 2026  
**Status:** Preprint (Draft)  
**DOI:** [10.5281/zenodo.22938153](https://doi.org/10.5281/zenodo.22938153)  
**Website:** [https://mlsmawfield.com/tep/arp/](https://mlsmawfield.com/tep/arp/)  
**Paper Series:** TEP Research Series: Paper 37 (Discordant Pairs)

## Abstract

Halton Arp catalogued low-redshift galaxies apparently joined by luminous bridges, filaments, and foreground absorption to companions at redshifts one to two orders of magnitude larger; under the standard redshift–distance identification every such pair is dismissed as a chance superposition. This paper applies the Temporal Equivalence Principle (TEP) — the bi-metric scalar-tensor framework in which matter, clocks, and light propagate on the causal metric $\tilde{g}_{\mu\nu} = A^2(\phi)\,g_{\mu\nu} + B(\phi)\,\nabla_\mu\phi\,\nabla_\nu\phi$ with universal coupling $A(\phi) = e^{-\phi}$ — to a twelve-pair catalogue drawn from the published literature and verified against the NASA/IPAC Extragalactic Database (NED). In TEP the observed redshift factorises into background temporal shear and intrinsic local shear; for objects at a common distance the intrinsic conformal factor $A_{\rm int}(Q) = (1+z_G)/(1+z_Q)$ measures the companion's local clock-rate offset with no distance dependence.

## Key Findings

1. **Distance-independent diagnostic**: $A_{\rm int}(Q)$ spans $0.33$ (NGC 7319) to $0.98$ (NGC 1232A) across the twelve pairs — a continuous distribution of local proper-time depths, not a quantised ladder. The standard reading instead places the companions a median of $\sim 90$, and up to $\sim 360$, times farther away than their apparent hosts.

2. **Joint chance-alignment probability**: Following the falsification of one pair by a Ly$\alpha$ path-length audit, the remaining eleven systems evaluate to $P_{\rm joint} = 3.6 \times 10^{-13}$ ($\log_{10} P = -12.4$) under the conservative empirical field density ($6.6 \times 10^{-14}$ under the published SDSS DR16 density), evaluated at archive-measured nucleus-to-companion separations. Conditioning on documented selection strengthens the result: axis-corridor pricing of the five morphology-claimed pairs gives $4.9 \times 10^{-19}$, and X-ray-selection-aware pricing at the measured $\Sigma_X = 3.8\ {\rm deg}^{-2}$ gives $7.2 \times 10^{-22}$. The look-elsewhere-corrected catalogue probability $P_{\rm cat}(N) = \prod_i[1-e^{-Np_i}]$ stays below 0.05 for parent search volumes under ~310 fields. Under the specified chance-superposition model, the surviving eleven-system configuration is highly improbable.

3. **Foreground absorption ordering**: Ca II K and H I 21-cm absorption at the host redshift in the spectrum of 3C 232, and the silhouette companion of NGC 1199, fix the line-of-sight ordering of two pairs independently of any redshift model.

4. **Bridge-axis detections in archival imaging**: every companion resolves to an archive position; on modern CCD imaging (DESI Legacy DR10 r-band; SDSS r for NGC 7319) the transect along the resolved NGC 7603 filament path — through both archive-measured emission knots — exceeds rotated control paths by $+3.9\sigma$ with a $+5.7\sigma$ localised peak, and the Mrk 205 bridge direction carries a persistent low-level excess above its control envelope (sign-test $p = 0.020$).

5. **Geometric coherence beyond the selection criteria**: companions of the five axis-claimed systems cluster on host minor axes (8/12 within 30°, $p = 0.019$); member redshifts show a trend ($p = 0.058$) to order radially with the nearer object at higher $z$ in 13/16 pairs, the well-gradient direction rather than an ejection sequence; and projected separations at redshift-independent distances cluster at the halo scale, 9.7–53 kpc over host distances spanning 7.5–113 Mpc (exact permutation $p = 0.019$).

6. **Nested-well field structure**: the measured NGC 7603 four-point redshift transect is non-monotonic — interior filament knots sit deeper in the field than the companion — resolving into the two-component structure $A_{\rm obs}(x) = A_{\rm shared}(x)\,w_i$ predicted by the framework's clock-rate hierarchy.

## The TEP Research Program

| Paper | Repository | Title |
|-------|-----------|-------|
| **Paper 0** | [TEP](https://github.com/matthewsmawfield/TEP) | Temporal Equivalence Principle: Dynamic Time & Emergent Light Speed |
| **Paper 9** | [TEP-EXP](https://github.com/matthewsmawfield/TEP-EXP) | What Do Precision Tests of General Relativity Actually Measure? |
| **Paper 11** | [TEP-H0](https://github.com/matthewsmawfield/TEP-H0) | The Cepheid Bias: Resolving the Hubble Tension |
| **Paper 12** | [TEP-JWST](https://github.com/matthewsmawfield/TEP-JWST) | A Unified Resolution to the JWST High-Redshift Anomalies |
| **Paper 31** | [TEP-VOID](https://github.com/matthewsmawfield/TEP-VOID) | Cosmological Voids vs Temporal Shear |
| **Paper 32** | [TEP-AGN](https://github.com/matthewsmawfield/TEP-AGN) | Scalar Proper-Time Gradients as a Geometric Alternative to Relativistic Smearing |
| **Paper 37** | **TEP-ARP** (This repo) | Localized Proper-Time Gradients and the Spatial Proximity of Discordant Pairs |

## Pipeline Structure

The pipeline (26 registered steps) tests the TEP spatial-proximity hypothesis against chance superposition across five blocks:

```
Block 0 (Steps 00-05): Data Ingestion & Target Catalog
  - Arp discordant-pair catalog (published redshifts, separations, connection evidence)
  - NED redshift verification
  - Optical imaging ingestion (DESI Legacy DR10 r-band primary; SDSS r / DSS2 Red via SkyView as fallbacks)
  - HEASARC / VizieR multi-wavelength coverage manifest
  - SDSS spectroscopic ingestion
  - Redshift-independent distances (Tully-Fisher and related calibrators)

Block I (Steps 10-13): TEP Formalism
  - Intrinsic conformal factor A_int(Q) = (1+z_G)/(1+z_Q) per pair
  - Redshift decomposition: background shear vs intrinsic local shear
  - Candidate scalar-field profile families across the bridge coordinate
  - Proper-time budget and fictitious distance gaps

Block II (Steps 20-28): Physical-Connection Evidence
  - Luminous bridge surface-brightness transects vs control axes
  - Foreground absorption systems at the host redshift
  - Quasar overdensity around Arp hosts vs seeded random fields
  - Poisson chance-alignment probabilities per pair and joint
  - Forward galaxy-quasar cross-correlation (2MRS parent sample)
  - Satellite redshift asymmetry (Tully 2MASS groups)
  - SIS lensing magnification-bias budget (bounded correction to joint P)
  - Geometric/spectroscopic coherence tests (minor-axis anisotropy, z-ordering, X-ray repricing)
  - Archive audit for resolved-kinematics coverage (ESO IFU, JWST, MaNGA, fibers)

Block III (Steps 30-33): Field-Gradient Inference
  - NGC 7603 four-point transect profile fits
  - MCMC posterior on the profile shape parameter
  - Pair-sample statistics (distribution, separation scaling)
  - Residuals analysis of the profile fits

Block IV (Steps 40-41): Synthesis & Figures
  - Falsification summary
  - Manuscript figure generation
```

## Running the Pipeline

```bash
cd "/Users/matthewsmawfield/www/Temporal Equivalence Principle/TEP-ARP"
python3 scripts/run_pipeline.py                     # Run full pipeline
python3 scripts/run_pipeline.py --block I           # Run only Block I
python3 scripts/run_pipeline.py --step 10           # Run only step 10
python3 scripts/run_pipeline.py --continue-on-error # Don't stop on failures
```

All outputs are step-number-prefixed (e.g., `step_10_intrinsic_conformal_factor.json`) and stored in `results/outputs/`. Figures are in `results/figures/`. Logs are in `logs/`.

The pipeline is deterministic: seeded where randomness enters (MCMC chains, control-field placement). A verification test suite re-derives the headline numbers from the processed tables offline:

```bash
python3 -m pytest scripts/tests/ -v
```

## Repository Structure

```
TEP-ARP/
├── site/                           # Academic manuscript site
│   ├── components/                 # HTML section files (edit these)
│   ├── public/                     # Static assets
│   └── dist/                       # Built site (generated)
├── scripts/steps/                  # Reproducible analysis pipeline (26 steps)
├── core/                           # Shared TEP framework modules
│   ├── constants.py                # Physical constants
│   └── ...
├── data/                           # Data (raw, interim, processed)
├── results/                        # Pipeline outputs and figures
├── logs/                           # Pipeline logs
└── VERSION.json                    # Version metadata
```

## Building the Site

```bash
cd site
npm install
npm run build
```

The built site will be in `site/dist/`. The build also regenerates the manuscript markdown at the repository root. The development server (`npm run dev`) watches the components and rebuilds automatically at http://localhost:55537.

## Manuscript Editing

Edit `site/components/*.html` files only. The markdown and `site/dist/` files are auto-generated by `npm run build`. Do not edit generated files directly.

## Citation

```bibtex
@misc{smawfield2026arp,
  title        = {Temporal Equivalence Principle: Localized Proper-Time Gradients and the Spatial Proximity of Discordant Pairs},
  author       = {Smawfield, Matthew Lukin},
  year         = {2026},
  doi          = {10.5281/zenodo.22938153},
  url          = {https://doi.org/10.5281/zenodo.22938153},
  note         = {Preprint, Version v0.1 (Pasadena)}
}
```

---

## Open Science Statement

These are working preprints shared in the spirit of open science — all manuscripts, analysis code, and data products are openly available under a Creative Commons license to encourage and facilitate replication. The complete analysis pipeline, data ingestion code, and all intermediate products are released as open-source software. Feedback and collaboration are warmly invited and welcome.

---

**Contact:** matthew@mlsmawfield.com  
**ORCID:** [0009-0003-8219-3159](https://orcid.org/0009-0003-8219-3159)
