#!/usr/bin/env python3
"""
TEP-ARP Analysis Pipeline Master Script
========================================
Orchestrates the analysis pipeline for the TEP-ARP paper:

  "Localized Proper-Time Gradients and the Spatial Proximity of
   Discordant Pairs"

This paper applies the Temporal Equivalence Principle (TEP) to
Halton Arp's discordant quasar-galaxy pairs.  The central claim
is that the luminous bridges, filaments, and absorption
associations documented by Arp are physical connections between
objects at a common distance, and that the redshift differential
of each pair is a localized proper-time gradient of the TEP
scalar field rather than a gap in cosmological distance.

The pipeline implements:
  1. Discordant-pair catalog with published redshifts (step 00)
  2. NED redshift verification and archival data ingestion
     (steps 01-05: NED, SkyView imaging, HEASARC/VizieR
     multi-wavelength coverage, SDSS spectroscopy,
     Cosmicflows-4 redshift-independent distances)
  3. TEP redshift decomposition: the intrinsic conformal factor
     A_int(Q) = (1+z_G)/(1+z_Q) and the proper-time budget
     (steps 10-13)
  4. Physical-connection evidence: bridge morphology transects,
     foreground absorption systems, quasar-galaxy association,
     and chance-alignment probabilities (steps 20-23)
  5. Field-gradient inference: transect fits of candidate
     A_int(x) profiles, MCMC posterior on the profile shape,
     pair-sample statistics, and residuals analysis (steps 30-33)
  6. Synthesis: falsification summary and manuscript figures
     (steps 40-41)

Pipeline Blocks:
  Block 0 (Steps 00-04): Data Ingestion & Target Catalog
    - Arp discordant-pair catalog (published redshifts, separations)
    - NED redshift verification
    - SkyView optical imaging ingestion
    - HEASARC / VizieR multi-wavelength coverage manifest
    - SDSS spectroscopic ingestion
    - Cosmicflows-4 redshift-independent distance anchoring

  Block I (Steps 10-13): TEP Formalism
    - Intrinsic conformal factor A_int(Q) per pair
    - Redshift decomposition into background and local shear
    - Candidate scalar-field profile families
    - Proper-time budget and fictitious distance gaps

  Block II (Steps 20-23): Physical-Connection Evidence
    - Luminous bridge surface-brightness transects
    - Foreground absorption systems at the host redshift
    - Quasar overdensity around Arp hosts vs control fields
    - Poisson chance-alignment probabilities

  Block III (Steps 30-33): Field-Gradient Inference
    - NGC 7603 four-point transect profile fits
    - MCMC posterior on the profile shape parameter
    - Pair-sample statistics (distribution, separation scaling)
    - Residuals analysis of the profile fits

  Block IV (Steps 40-41): Synthesis & Figures
    - Falsification summary
    - Manuscript figure generation

Usage:
    python3 scripts/run_pipeline.py
    python3 scripts/run_pipeline.py --block I
    python3 scripts/run_pipeline.py --step 10
    python3 scripts/run_pipeline.py --continue-on-error

Author: Matthew Lukin Smawfield
Date: September 2026
"""

import sys
import time
import json
import argparse
import platform
import subprocess
import traceback
from datetime import datetime, timezone
from pathlib import Path

# Ensure project root is in path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

# Pipeline definition: (block, step_num, module_name, class_name, description)
PIPELINE = [
    # Block 0: Data Ingestion & Target Catalog
    ("0", 0, "scripts.steps.step_00_arp_pair_catalog", "Step00ArpPairCatalog",
     "Arp discordant-pair catalog: published redshifts, separations, connection evidence"),
    ("0", 1, "scripts.steps.step_01_ned_redshift_verification", "Step01NEDVerification",
     "NED redshift verification for all catalog members"),
    ("0", 2, "scripts.steps.step_02_optical_imaging_ingestion", "Step02OpticalImaging",
     "SkyView optical imaging ingestion (DSS2 Red / SDSS r) for each pair field"),
    ("0", 3, "scripts.steps.step_03_multiwavelength_manifest", "Step03MultiwavelengthManifest",
     "HEASARC (Chandra/XMM/ROSAT) and VizieR (NVSS/FIRST) coverage manifest"),
    ("0", 4, "scripts.steps.step_04_spectroscopic_ingestion", "Step04SpectroscopicIngestion",
     "SDSS spectroscopic ingestion for pair members in the footprint"),
    ("0", 5, "scripts.steps.step_05_redshift_independent_distances", "Step05RedshiftIndependentDistances",
     "Cosmicflows-4 redshift-independent distances for catalog members"),

    # Block I: TEP Formalism
    ("I", 10, "scripts.steps.step_10_intrinsic_conformal_factor", "Step10IntrinsicConformalFactor",
     "Intrinsic conformal factor A_int(Q) = (1+z_G)/(1+z_Q) per pair"),
    ("I", 11, "scripts.steps.step_11_redshift_decomposition", "Step11RedshiftDecomposition",
     "Redshift decomposition: background shear vs intrinsic local shear"),
    ("I", 12, "scripts.steps.step_12_scalar_field_profiles", "Step12ScalarFieldProfiles",
     "Candidate scalar-field profile families across the bridge coordinate"),
    ("I", 13, "scripts.steps.step_13_proper_time_budget", "Step13ProperTimeBudget",
     "Proper-time budget: clock-rate ratios and fictitious distance gaps"),

    # Block II: Physical-Connection Evidence
    ("II", 20, "scripts.steps.step_20_bridge_morphology", "Step20BridgeMorphology",
     "Luminous bridge surface-brightness transects vs control axes"),
    ("II", 21, "scripts.steps.step_21_absorption_systems", "Step21AbsorptionSystems",
     "Foreground absorption systems at the host-galaxy redshift"),
    ("II", 22, "scripts.steps.step_22_quasar_galaxy_association", "Step22QuasarGalaxyAssociation",
     "Quasar overdensity around Arp hosts vs seeded random fields"),
    ("II", 23, "scripts.steps.step_23_chance_alignment_probability", "Step23ChanceAlignment",
     "Poisson chance-alignment probabilities per pair and joint"),
    ("II", 24, "scripts.steps.step_24_forward_crosscorrelation", "Step24ForwardCrossCorrelation",
     "Forward galaxy-quasar cross-correlation: predefined 2MRS parent sample vs controls"),
    ("II", 25, "scripts.steps.step_25_satellite_redshift_asymmetry", "Step25SatelliteRedshiftAsymmetry",
     "Satellite redshift asymmetry on Tully 2015 groups (Arp & Sulentic signature)"),
    ("II", 26, "scripts.steps.step_26_lens_magnification_budget", "Step26LensMagnificationBudget",
     "SIS lensing magnification-bias budget: bounded correction to joint P"),
    ("II", 27, "scripts.steps.step_27_geometric_coherence", "Step27GeometricCoherence",
     "Geometric/spectroscopic coherence: minor-axis anisotropy, z-ordering, halo scale, pairing, X-ray repricing"),
    ("II", 28, "scripts.steps.step_28_archive_kinematic_audit", "Step28ArchiveKinematicAudit",
     "Archive audit for resolved-kinematics coverage (ESO IFU, JWST, MaNGA, fibers)"),
    ("II", 29, "scripts.steps.step_29_lya_forest_audit", "Step29LyaForestAudit",
     "Ly-alpha forest path-length audit: archive coverage plus direct forest census where spectra exist"),

    # Block III: Field-Gradient Inference
    ("III", 30, "scripts.steps.step_30_bridge_redshift_transect", "Step30BridgeRedshiftTransect",
     "Redshift transect fits: A_int(x) families on the NGC 7603 four-point transect"),
    ("III", 31, "scripts.steps.step_31_mcmc_field_profile", "Step31MCMCFieldProfile",
     "MCMC posterior on the scalar-field profile shape parameter"),
    ("III", 32, "scripts.steps.step_32_pair_sample_statistics", "Step32PairSampleStatistics",
     "Pair-sample statistics: A_int distribution and separation scaling"),
    ("III", 33, "scripts.steps.step_33_residuals_analysis", "Step33ResidualsAnalysis",
     "Residuals analysis of the transect profile fits"),

    # Block IV: Synthesis & Figures
    ("IV", 40, "scripts.steps.step_40_falsification_summary", "Step40FalsificationSummary",
     "Falsification summary: TEP spatial proximity vs chance superposition"),
    ("IV", 41, "scripts.steps.step_41_manuscript_figures", "Step41ManuscriptFigures",
     "Manuscript figure generation from all results"),
]


def collect_environment():
    """Assemble the software-provenance record for the run manifest."""
    env = {
        "generated_utc": datetime.now(timezone.utc).isoformat(
            timespec="seconds"
        ),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "packages": {},
    }
    import importlib
    for pkg in (
        "numpy", "pandas", "astropy", "astroquery", "scipy",
        "matplotlib", "emcee", "corner", "requests",
    ):
        try:
            mod = importlib.import_module(pkg)
            env["packages"][pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            env["packages"][pkg] = None
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            capture_output=True, text=True, timeout=10,
        )
        env["git_commit"] = rev.stdout.strip() if rev.returncode == 0 else None
        env["git_dirty"] = (
            bool(dirty.stdout.strip()) if dirty.returncode == 0 else None
        )
    except Exception:
        env["git_commit"] = None
        env["git_dirty"] = None
    version_path = PROJECT_ROOT / "VERSION.json"
    if version_path.exists():
        try:
            env["manuscript_version"] = json.loads(
                version_path.read_text()
            )
        except Exception:
            env["manuscript_version"] = None
    return env


def run_step(block, step_num, module_name, class_name, description, master_logger):
    """Execute a single pipeline step with proper logging and error handling."""

    if isinstance(step_num, int):
        step_id = f"step_{step_num:02d}"
    else:
        step_id = f"step_{step_num}"
    set_step_logger(None)
    print_status(f"Starting {step_id}: {description}", "TITLE")

    start_time = time.time()

    try:
        import importlib
        module = importlib.import_module(module_name)
        if class_name and hasattr(module, class_name):
            step_class = getattr(module, class_name)
            step_instance = step_class()
            step_instance.run()
        elif hasattr(module, "run"):
            module.run()
        elif hasattr(module, "main"):
            module.main()
        else:
            raise AttributeError(
                f"Module {module_name} has neither class '{class_name}' nor function 'run'/'main'"
            )

        elapsed = time.time() - start_time
        print_status(f"{step_id} completed in {elapsed:.1f}s", "SUCCESS")
        master_logger.info(f"  {step_id}: OK ({elapsed:.1f}s)")
        return True, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        error_msg = f"{step_id} FAILED after {elapsed:.1f}s: {e}"
        print_status(error_msg, "ERROR")
        master_logger.error(f"  {step_id}: FAILED ({elapsed:.1f}s) - {e}")
        traceback.print_exc()
        return False, elapsed


def main():
    parser = argparse.ArgumentParser(
        description="TEP-ARP Analysis Pipeline"
    )
    parser.add_argument(
        "--block",
        choices=["0", "I", "II", "III", "IV"],
        help="Run only a specific block",
    )
    parser.add_argument(
        "--step",
        type=str,
        help="Run only a specific step (by number, e.g. 10)",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running subsequent steps even if one fails",
    )
    args = parser.parse_args()

    # Create output directories
    (PROJECT_ROOT / "results" / "figures").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "results" / "outputs").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "logs").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "raw").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "interim").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "processed").mkdir(parents=True, exist_ok=True)

    # Master logger
    master_log = PROJECT_ROOT / "logs" / "pipeline_master.log"
    master_logger = TEPLogger("pipeline_master", log_file_path=master_log, reset_log=True)
    set_step_logger(master_logger)

    print_status("TEP-ARP Analysis Pipeline", "TITLE")
    print_status("Localized Proper-Time Gradients and the Spatial Proximity of Discordant Pairs", "PROCESS")
    print_status(f"Project root: {PROJECT_ROOT}", "INFO")
    print_status(f"Results: {PROJECT_ROOT / 'results'}", "INFO")
    print_status(f"Logs: {PROJECT_ROOT / 'logs'}", "INFO")

    # Filter steps
    steps_to_run = PIPELINE
    if args.block:
        steps_to_run = [s for s in PIPELINE if s[0] == args.block]
    elif args.step is not None:
        try:
            step_val = int(args.step)
        except ValueError:
            step_val = args.step
        steps_to_run = [s for s in PIPELINE if s[1] == step_val]

    print_status(f"Running {len(steps_to_run)} step(s)", "PROCESS")

    succeeded = 0
    failed = 0
    total_start = time.time()
    step_records = []

    for block, step_num, module_name, class_name, description in steps_to_run:
        step_id = (
            f"step_{step_num:02d}"
            if isinstance(step_num, int)
            else f"step_{step_num}"
        )
        ok, elapsed = run_step(
            block, step_num, module_name, class_name, description,
            master_logger,
        )
        step_records.append({
            "step": step_id,
            "block": block,
            "description": description,
            "status": "ok" if ok else "failed",
            "elapsed_s": round(elapsed, 2),
        })
        if ok:
            succeeded += 1
        else:
            failed += 1
            if not args.continue_on_error:
                print_status("Stopping pipeline (use --continue-on-error to skip failures)", "WARNING")
                break

    total_elapsed = time.time() - total_start
    summary = f"Pipeline complete: {succeeded} succeeded, {failed} failed, {total_elapsed:.1f}s total"
    print_status(summary, "TITLE")
    master_logger.info("")
    master_logger.info("=" * 80)
    master_logger.info(f"   {summary}")
    master_logger.info("=" * 80)
    master_logger.info("")

    # Provenance manifest: environment, per-step status, output inventory.
    outputs_dir = PROJECT_ROOT / "results" / "outputs"
    figures_dir = PROJECT_ROOT / "results" / "figures"
    manifest = {
        "pipeline": "TEP-ARP",
        "title": (
            "Temporal Equivalence Principle: Localized Proper-Time "
            "Gradients and the Spatial Proximity of Discordant Pairs"
        ),
        "run_mode": (
            "full" if len(steps_to_run) == len(PIPELINE)
            else f"partial ({len(steps_to_run)} steps)"
        ),
        "environment": collect_environment(),
        "n_steps_run": len(step_records),
        "n_succeeded": succeeded,
        "n_failed": failed,
        "total_elapsed_s": round(total_elapsed, 2),
        "steps": step_records,
        "outputs": {
            "results": sorted(
                p.name for p in outputs_dir.glob("*") if p.is_file()
            ),
            "figures": sorted(
                p.name for p in figures_dir.glob("*") if p.is_file()
            ),
        },
    }
    manifest_path = outputs_dir / "pipeline_run_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print_status(f"Saved run manifest: {manifest_path}", "SUCCESS")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
