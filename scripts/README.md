# TEP-ARP Scripts

## Analysis Pipeline

```bash
cd "/Users/matthewsmawfield/www/Temporal Equivalence Principle/TEP-ARP"
python3 scripts/run_pipeline.py
```

Options:
```bash
python3 scripts/run_pipeline.py --block I           # Run only Block I (steps 10-13)
python3 scripts/run_pipeline.py --step 23           # Run only step 23
python3 scripts/run_pipeline.py --continue-on-error # Don't stop on failures
```

The pipeline is deterministic: seeded where randomness enters (MCMC chains,
control-field placement). It requires live access to NED, Simbad, VizieR,
HEASARC, SkyView, and SDSS for the ingestion steps; every step fails loudly
when a required input is missing.

## Pipeline Structure

The pipeline consists of 26 registered steps in 5 blocks:

### Block 0: Data Ingestion & Target Catalog (Steps 00-05)
- `step_00_arp_pair_catalog.py` — Twelve discordant pairs with published redshifts, separations, and connection evidence
- `step_01_ned_redshift_verification.py` — NED redshift verification for all catalog members (archive values supersede literature where discrepant)
- `step_02_optical_imaging_ingestion.py` — SkyView DSS2 Red / SDSS r imaging per pair field
- `step_03_multiwavelength_manifest.py` — HEASARC (Chandra/XMM/ROSAT) and VizieR (NVSS/FIRST) coverage manifest
- `step_04_spectroscopic_ingestion.py` — SDSS spectroscopic ingestion and independent redshift measurement where spectra exist
- `step_05_redshift_independent_distances.py` — Redshift-independent host distances (Cosmicflows-4 / NED compilations) for the physical-scale tests

### Block I: TEP Formalism (Steps 10-13)
- `step_10_intrinsic_conformal_factor.py` — A_int(Q) = (1+z_G)/(1+z_Q) per pair; distance-independent diagnostic
- `step_11_redshift_decomposition.py` — Background vs intrinsic shear decomposition; Hubble distance-inflation factors
- `step_12_scalar_field_profiles.py` — Candidate A_int(x) profile families (exponential, tanh, Yukawa, nested wells)
- `step_13_proper_time_budget.py` — Clock-rate ratios and fictitious distance/lookback gaps

### Block II: Physical-Connection Evidence (Steps 20-27)
- `step_20_bridge_morphology.py` — Archive-resolved companion axes; surface-brightness transects vs rotated controls; sign test, peak z, silhouette annulus
- `step_21_absorption_systems.py` — Published foreground-absorption and silhouette ordering evidence
- `step_22_quasar_galaxy_association.py` — Milliquas confirmed quasar-class counts around hosts vs seeded control fields; companions excluded (non-circular), inclusive counts alongside; member-level persistence to qso_field_members.csv
- `step_23_chance_alignment_probability.py` — Poisson chance-alignment probabilities per pair and joint, under SDSS and empirical densities; axis-corridor conditioning on morphology-claimed pairs; compact-subsample joint (theta < 300 arcsec) reported separately from wide-field systems; look-elsewhere search-volume correction P_cat(N) = prod_i[1-exp(-N p_i)]
- `step_24_forward_crosscorrelation.py` — Forward galaxy-quasar cross-correlation on the predefined 2MRS parent sample vs controls
- `step_25_satellite_redshift_asymmetry.py` — Satellite redshift asymmetry on Tully 2015 2MASS groups (Arp & Sulentic signature)
- `step_26_lens_magnification_budget.py` — SIS lensing magnification-bias budget: bounded correction to the joint P
- `step_27_geometric_coherence.py` — Structure tests the catalogue was not selected on: minor-axis anisotropy, radial redshift ordering, halo-scale clustering, paired-redshift excess, symmetric configurations, and X-ray-selection-aware repricing
- `step_28_archive_kinematic_audit.py` — Live audit of resolved-kinematics coverage (ESO MUSE/VIMOS/SINFONI/GIRAFFE, JWST IFU, MaNGA, fiber spectroscopy) with client-side in-cone filtering and program-level provenance
- `step_29_lya_forest_audit.py` — Ly-alpha forest path-length audit: per-companion forest-band coverage across SDSS/MAST/ESO/KOA gated by actual grating wavelength ranges and on-target slit requirements, plus a direct Lyman-confirmed forest census on the archived HST/COS spectra of 3C 232

### Block III: Field-Gradient Inference (Steps 30-33)
- `step_30_bridge_redshift_transect.py` — Four-point NGC 7603 transect fits across all profile families
- `step_31_mcmc_field_profile.py` — emcee MCMC posterior on nested-wells depths and shared width (Metropolis fallback)
- `step_32_pair_sample_statistics.py` — A_int distribution and separation-scaling statistics
- `step_33_residuals_analysis.py` — Weighted residuals, chi-squared, residual RMS, lag-1 autocorrelation, Lilliefors normality check

### Block IV: Synthesis & Figures (Steps 40-41)
- `step_40_falsification_summary.py` — Thirteen-test falsification table: TEP proximity vs chance superposition
- `step_41_manuscript_figures.py` — Manuscript figures from all results

## Verification

```bash
python3 -m pytest scripts/tests/ -v
```

Offline tests re-derive the headline numbers from the processed tables:
conformal-factor arithmetic, Poisson chance-alignment probabilities under
both density assumptions, the look-elsewhere catalogue probability and its
5% crossover volume, transect knot arithmetic, profile boundary conditions,
MCMC posterior recovery of Delta_phi, falsification-table integrity, and
residual-diagnostic presence.

## PDF Generation

```bash
python3 scripts/generate_site_pdf.py
python3 scripts/generate_site_pdf.py --quality high --wait-time 10
```

Generates `37-TEP-ARP-v0.1-Pasadena.pdf` from the built static site (root and
`site/public/docs/`). Requires the site to be built first (`cd site && npm run build`).

## Utilities

- `utils/logger.py` — TEPLogger: color-coded console + file logging
- `utils/plot_style.py` — Matplotlib style for consistent figures
- `utils/html_to_pdf.py` — HTML to PDF converter (Playwright)
- `utils/compress_pdf.py` — PDF compression
- `utils/process_pdf.py` — PDF metadata embedding
- `utils/setup_pdf_converter.sh` — Setup PDF converter dependencies
