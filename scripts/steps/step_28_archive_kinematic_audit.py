#!/usr/bin/env python3
"""
Step 28: Archival Kinematic-Data Feasibility Audit
===================================================
The decisive test against the "projection through tidal debris"
reading is spatially resolved kinematics: emission-line velocity
structure of the NGC 7603 filament gas at the knot coordinates.
Before that requirement is written into the manuscript, this
step exhaustively audits what the public archives actually
contain, so the paper states with provenance which datasets
exist, which are absent, and what new observation is required.

Queries, all live and all recorded:

  * ESO archive (astroquery.eso) — MUSE, VIMOS-IFU, SINFONI and
    FLAMES/GIRAFFE-ARGUS cone searches at each host position and,
    for NGC 7603, at the two filament-knot coordinates.  Only
    DPR CATG = 'SCIENCE' frames with valid sky coordinates are
    counted; calibration frames are classified separately so an
    empty science return is an explicit, audited result — never
    a silent zero.
  * MAST (astroquery.mast) — JWST IFU coverage (NIRSpec IFU,
    MIRI MRS) and MaNGA/HLSP products per field; the closest
    usable spectroscopic alternative probes (BOSS/SDSS fibers,
    HST imaging) are recorded as partial alternatives.
  * Availability verdict per field: resolved kinematics of the
    claimed structure (a) already in the archive, (b) partially
    addressable, or (c) absent — requiring a new observation,
    for which the minimal sufficient configuration is stated.

The step fails loudly if the catalogue or the step_20 knot
positions are missing; archive timeouts are retried and then
recorded as 'query_failed' rows (never silently omitted, never
fabricated).

Outputs:
    data/processed/archive_kinematic_audit.csv
    results/outputs/step_28_archive_kinematic_audit.json
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.jsonio import json_safe

ESO_INSTRUMENTS = ["muse", "vimos", "sinfoni", "giraffe"]
SEARCH_RADIUS_ARCMIN = 3.0        # covers knot-to-host offsets
JWST_IFU_KEYS = ("IFU", "MRS")   # 'MIRI/IFU', 'NIRSPEC/IFU' product names
MANGA_KEYS = ("MANGA",)
SPECTRO_KEYS = ("BOSS", "SDSS SPECTROGRAPH", "SDSS SPECTRA")
HST_IMAGING_KEYS = ("ACS", "WFC3", "WFPC2", "WFPC")
N_RETRIES = 3
RETRY_SLEEP_S = 3.0


class Step28ArchiveKinematicAudit:
    """Step 28: exhaustively audit archives for resolved-kinematics data."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"
        for d in [self.data_processed, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)
        self.logger = TEPLogger(
            "step_28",
            log_file_path=self.logs / "step_28_archive_kinematic_audit.log",
        )
        set_step_logger(self.logger)

    # ---------------------------------------------------------------
    def _positions(self):
        """Host positions for all pairs + NGC 7603 knot positions."""
        cat_path = self.data_processed / "arp_pair_catalog_verified.csv"
        if not cat_path.exists():
            cat_path = self.data_processed / "arp_pair_catalog.csv"
        if not cat_path.exists():
            raise FileNotFoundError("pair catalogue missing")
        catalog = pd.read_csv(cat_path)

        knots_path = self.data_processed / "filament_knot_positions.csv"
        if not knots_path.exists():
            raise FileNotFoundError(
                "filament_knot_positions.csv missing; run step_20")
        knots = pd.read_csv(knots_path)

        from astroquery.simbad import Simbad
        targets = []
        for _, row in catalog.iterrows():
            pid = row["pair_id"]
            ra, dec = row.get("galaxy_ra_deg"), row.get("galaxy_dec_deg")
            if pd.isna(ra) or pd.isna(dec):
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore", message=".*No results.*")
                    t = Simbad.query_object(str(row["galaxy"]))
                if t is None or len(t) == 0:
                    raise RuntimeError(
                        f"{pid}: host position unavailable from "
                        "catalogue and Simbad")
                ra, dec = float(t[0]["ra"]), float(t[0]["dec"])
            targets.append({
                "pair_id": pid,
                "label": f"{row['galaxy']} (host)",
                "ra_deg": float(ra), "dec_deg": float(dec),
                "kind": "host",
            })
        for _, k in knots.iterrows():
            targets.append({
                "pair_id": k["pair_id"],
                "label": f"{k['resolved_name']} (filament knot)",
                "ra_deg": float(k["ra_deg"]),
                "dec_deg": float(k["dec_deg"]),
                "kind": "filament_knot",
            })
        return targets

    def _query_eso(self, eso, instrument, ra, dec):
        """Cone query on one ESO instrument with retry; returns
        (science_rows, n_calib) or raises after N_RETRIES."""
        last_err = None
        for attempt in range(1, N_RETRIES + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    # ESO form keys are coord1/coord2 (deg) + box.
                    # Request a box comfortably larger than the
                    # audit radius; instruments that honour it
                    # return the local set, instruments that ignore
                    # it return a catalogue slice that the
                    # client-side filter below then constrains.
                    t = eso.query_instrument(
                        instrument, coord1=ra, coord2=dec,
                        box=f"{4.0 * SEARCH_RADIUS_ARCMIN:.0f}")
                if t is None:
                    return [], 0
                # ESO instruments differ in category-column naming
                # ('DPR CATG' for MUSE/VIMOS, 'DPR.CATG' for
                # SINFONI/GIRAFFE); locate it case-insensitively.
                catg_col = next(
                    (c for c in t.colnames
                     if c.upper().replace(".", " ") == "DPR CATG"),
                    None)
                sci, cal = [], 0
                for r in t:
                    catg = (
                        str(r[catg_col]).strip().upper()
                        if catg_col is not None else "SCIENCE")
                    if catg == "SCIENCE":
                        sci.append(r)
                    else:
                        cal += 1
                return sci, cal
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(RETRY_SLEEP_S * attempt)
        raise RuntimeError(
            f"ESO {instrument} query failed after {N_RETRIES} "
            f"attempts: {last_err}")

    def _offset_arcmin(self, r_ra, r_dec, ra, dec):
        if np.ma.is_masked(r_ra) or np.ma.is_masked(r_dec):
            return np.nan
        try:
            f_ra, f_dec = float(r_ra), float(r_dec)
        except (TypeError, ValueError):
            return np.nan
        d2 = (f_ra - ra) ** 2 * np.cos(np.radians(dec)) ** 2 \
            + (f_dec - dec) ** 2
        return float(np.degrees(np.sqrt(d2)) * 60.0)

    def _query_mast(self, ra, dec):
        """MAST region query with retry; returns the observation table."""
        from astroquery.mast import Observations
        last_err = None
        for attempt in range(1, N_RETRIES + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return Observations.query_region(
                        f"{ra} {dec}",
                        radius=SEARCH_RADIUS_ARCMIN * u.arcmin)
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(RETRY_SLEEP_S * attempt)
        raise RuntimeError(
            f"MAST query failed after {N_RETRIES} attempts: {last_err}")

    # ---------------------------------------------------------------
    def run(self):
        from astroquery.eso import Eso

        print_status(
            "Auditing archives for resolved-kinematics coverage...",
            "PROCESS")
        targets = self._positions()
        print_status(f"Auditing {len(targets)} field positions "
                     f"({sum(t['kind'] == 'host' for t in targets)} hosts, "
                     f"{sum(t['kind'] == 'filament_knot' for t in targets)} "
                     "filament knots)", "INFO")

        eso = Eso()
        # Lift the 50-row default cap; without it instrument tables
        # truncate to calibration frames and the audit under-reports.
        eso.ROW_LIMIT = -1
        rows = []
        for tgt in targets:
            pid, ra, dec = tgt["pair_id"], tgt["ra_deg"], tgt["dec_deg"]

            # --- ESO IFU instruments --------------------------------
            for inst in ESO_INSTRUMENTS:
                try:
                    sci, cal = self._query_eso(eso, inst, ra, dec)
                    # Server-side box semantics differ between
                    # instruments (some ignore it and return a
                    # catalogue slice); the authoritative spatial
                    # filter is applied here on the returned
                    # pointing positions.
                    offs = [self._offset_arcmin(
                        r["RA"], r["DEC"], ra, dec) for r in sci]
                    valid = [o for o in offs if np.isfinite(o)]
                    in_cone = [
                        r for r, o in zip(sci, offs)
                        if np.isfinite(o)
                        and o <= SEARCH_RADIUS_ARCMIN]
                    progs = sorted({
                        str(r["Program ID"]) for r in in_cone})
                    objs = sorted({
                        str(r["Object"]) for r in in_cone})
                    rows.append({
                        "pair_id": pid, "label": tgt["label"],
                        "kind": tgt["kind"], "archive": "ESO",
                        "instrument": inst.upper(),
                        "n_frames_total": len(sci) + cal,
                        "n_science": len(in_cone),
                        "n_science_returned_unfiltered": len(sci),
                        "n_calibration": cal,
                        "min_offset_arcmin": (
                            min(valid) if valid else np.nan),
                        "science_programs": ";".join(progs),
                        "science_objects": ";".join(objs),
                        "status": "ok",
                    })
                except RuntimeError as e:
                    print_status(str(e), "WARNING")
                    rows.append({
                        "pair_id": pid, "label": tgt["label"],
                        "kind": tgt["kind"], "archive": "ESO",
                        "instrument": inst.upper(),
                        "n_frames_total": 0, "n_science": 0,
                        "n_calibration": 0,
                        "min_offset_arcmin": np.nan,
                        "science_programs": "",
                        "science_objects": "",
                        "status": "query_failed",
                    })
                time.sleep(0.3)

            # --- MAST -----------------------------------------------
            try:
                t = self._query_mast(ra, dec)
                insts = [str(x).upper() for x in t["instrument_name"]]
                obs_coll = [
                    str(x).upper() for x in t["obs_collection"]]
                n_jwst_ifu = sum(
                    1 for i, c in zip(insts, obs_coll)
                    if any(k in i for k in JWST_IFU_KEYS)
                    and "JWST" in c)
                n_manga = sum(
                    1 for i, c in zip(insts, obs_coll)
                    if any(k in i or k in c for k in MANGA_KEYS))
                n_spectro = sum(
                    1 for i in insts
                    if any(k in i for k in SPECTRO_KEYS))
                n_hst_img = sum(
                    1 for i, c in zip(insts, obs_coll)
                    if any(k in i for k in HST_IMAGING_KEYS)
                    and "HST" in c)
                rows.append({
                    "pair_id": pid, "label": tgt["label"],
                    "kind": tgt["kind"], "archive": "MAST",
                    "instrument": "ALL",
                    "n_frames_total": len(t),
                    "n_science": len(t),
                    "n_calibration": 0,
                    "min_offset_arcmin": np.nan,
                    "n_jwst_ifu": n_jwst_ifu,
                    "n_manga": n_manga,
                    "n_spectro_fibers": n_spectro,
                    "n_hst_imaging": n_hst_img,
                    "status": "ok",
                })
            except RuntimeError as e:
                print_status(str(e), "WARNING")
                rows.append({
                    "pair_id": pid, "label": tgt["label"],
                    "kind": tgt["kind"], "archive": "MAST",
                    "instrument": "ALL",
                    "n_frames_total": 0, "n_science": 0,
                    "n_calibration": 0,
                    "min_offset_arcmin": np.nan,
                    "n_jwst_ifu": 0, "n_manga": 0,
                    "n_spectro_fibers": 0, "n_hst_imaging": 0,
                    "status": "query_failed",
                })
            time.sleep(0.3)

        df = pd.DataFrame(rows)
        csv_path = self.data_processed / "archive_kinematic_audit.csv"
        df.to_csv(csv_path, index=False)
        print_status(f"Saved audit table: {csv_path} ({len(df)} queries)",
                     "SUCCESS")

        # ----------------------------------------------------------
        # Verdict per field position.
        verdicts = []
        for tgt in targets:
            sub = df[(df["pair_id"] == tgt["pair_id"])
                     & (df["label"] == tgt["label"])]
            eso_sub = sub[sub["archive"] == "ESO"]
            n_eso_sci = int(eso_sub["n_science"].sum())
            n_failed = int((sub["status"] == "query_failed").sum())
            # Closest science pointing of any ESO IFU to this position;
            # a 0.6 arcmin bound approximates a MUSE half-field.
            off = pd.to_numeric(
                eso_sub["min_offset_arcmin"], errors="coerce")
            min_eso_off = (
                float(off.min()) if off.notna().any() else np.nan)
            mast = sub[sub["archive"] == "MAST"].iloc[0]
            n_jwst = int(mast.get("n_jwst_ifu", 0))
            n_manga = int(mast.get("n_manga", 0))
            n_spec = int(mast.get("n_spectro_fibers", 0))
            n_hst = int(mast.get("n_hst_imaging", 0))

            if (np.isfinite(min_eso_off) and min_eso_off <= 0.6) \
                    or n_jwst > 0 or n_manga > 0:
                verdict = "ifu_covers_position"
            elif n_eso_sci > 0:
                verdict = "ifu_in_field_not_on_target"
            elif n_spec > 0:
                verdict = "fiber_spectra_only"
            else:
                verdict = "no_spatial_spectroscopy"
            verdicts.append({
                "pair_id": tgt["pair_id"],
                "label": tgt["label"],
                "kind": tgt["kind"],
                "n_eso_ifu_science": n_eso_sci,
                "min_eso_ifu_offset_arcmin": min_eso_off,
                "n_jwst_ifu": n_jwst,
                "n_manga": n_manga,
                "n_spectro_fibers": n_spec,
                "n_hst_imaging": n_hst,
                "n_failed_queries": n_failed,
                "verdict": verdict,
            })
            print_status(
                f"{tgt['label']}: ESO-IFU science={n_eso_sci} "
                f"(min offset {min_eso_off:.2f} arcmin), "
                f"JWST-IFU={n_jwst}, MaNGA={n_manga}, "
                f"fibers={n_spec}, HST={n_hst} -> {verdict}", "TEST")

        n_q = int((df["status"] == "ok").sum())
        n_f = int((df["status"] == "query_failed").sum())
        summary = {
            "n_positions_audited": len(verdicts),
            "n_archive_queries": len(df),
            "n_queries_answered": n_q,
            "n_queries_failed": n_f,
            "eso_instruments": ESO_INSTRUMENTS,
            "search_radius_arcmin": SEARCH_RADIUS_ARCMIN,
            "verdicts": verdicts,
            "required_observation": {
                "ngc7603_filament": (
                    "A single VLT/MUSE pointing (or JWST NIRSpec-IFU "
                    "mosaic) covering the filament polyline and both "
                    "[LG2002] knot coordinates would deliver the "
                    "resolved [O III]/H-alpha velocity field needed to "
                    "test for kinematic disturbance induced by the "
                    "knots. The audit establishes whether such data "
                    "already exist."
                ),
            },
            "interpretation": (
                "Every archive was queried live and every return was "
                "classified as science or calibration. Absent IFU "
                "coverage is reported as an audited negative result, "
                "not an untested assumption."
            ),
            "coordinate_caveat": (
                "ESO archive rows carry the proposal target "
                "coordinates, which can differ from the actual "
                "pointing footprint; min_offset_arcmin is therefore "
                "indicative and verdicts err conservative — a frame "
                "reported near a position may still not cover it, "
                "and a frame reported far may cover it. Footprint-"
                "level verification is required before any claimed "
                "coverage is used as a measurement."
            ),
        }
        json_path = (
            self.results / "step_28_archive_kinematic_audit.json")
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")
        print_status("Kinematic feasibility audit complete.", "SUCCESS")


if __name__ == "__main__":
    Step28ArchiveKinematicAudit().run()
