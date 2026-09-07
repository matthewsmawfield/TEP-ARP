#!/usr/bin/env python3
"""
Step 03: Multi-Wavelength Observation Manifest
==============================================
Queries the HEASARC master catalogues (Chandra, XMM-Newton, ROSAT)
and VizieR (NVSS, FIRST) for archival coverage of each Arp pair
field, producing a provenance manifest of multi-wavelength data.

Multi-band coverage is the cross-spectrum proof demanded by the
analysis plan: if the claimed connecting structures are physical,
their signatures should imprint across radio continuum, optical,
and X-ray bands rather than appearing only in one detector.

This step requires network access to HEASARC and VizieR.

Outputs:
    data/processed/multiwavelength_manifest.csv
    results/outputs/step_03_multiwavelength_manifest.json
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.jsonio import json_safe

# HEASARC master mission tables queried per field.
XRAY_TABLES = {
    "chanmaster": "Chandra",
    "xmmmaster": "XMM-Newton",
    "rosmaster": "ROSAT",
}

# VizieR radio catalogues queried per field.
RADIO_CATALOGS = {
    "VIII/65/nvss": "NVSS",
    "VIII/92/first14": "FIRST",
}


class Step03MultiwavelengthManifest:
    """Step 03: Compile multi-wavelength archive coverage manifest."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_03",
            log_file_path=self.logs / "step_03_multiwavelength_manifest.log",
        )
        set_step_logger(self.logger)

    def _resolve(self, name):
        """Resolve an object name to coordinates via the resolver chain
        NED -> SIMBAD.  Permanent name-resolution failures are reported
        once at INFO level; transient transport failures are retried
        before SIMBAD is tried.  Returns None only when every resolver
        fails — a genuine unresolvable-field condition."""
        from astroquery.ipac.ned import Ned
        from astroquery.simbad import Simbad
        from astroquery.exceptions import NoResultsWarning

        for attempt in (1, 2, 3):
            try:
                tbl = Ned.query_object(name)
                if tbl is not None and len(tbl) > 0:
                    return SkyCoord(
                        ra=float(tbl[0]["RA"]) * u.deg,
                        dec=float(tbl[0]["DEC"]) * u.deg,
                    )
                break
            except Exception as e:
                msg = str(e)
                if "not currently recognized" in msg:
                    break
                short = msg.strip().splitlines()[0]
                if attempt < 3:
                    self.logger.info(
                        f"NED resolve attempt {attempt}/3 for {name} "
                        f"failed ({short}); retrying."
                    )
                    time.sleep(2.0 * attempt)
                else:
                    self.logger.info(
                        f"NED resolve for {name} failed after 3 attempts "
                        f"({short}); trying SIMBAD."
                    )
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=NoResultsWarning)
                tbl = Simbad.query_object(name)
            if tbl is not None and len(tbl) > 0:
                coord = SkyCoord(
                    ra=float(tbl[0]["ra"]) * u.deg,
                    dec=float(tbl[0]["dec"]) * u.deg,
                )
                self.logger.info(
                    f"{name} resolved via SIMBAD fallback "
                    f"({coord.ra.deg:.5f}, {coord.dec.deg:.5f})."
                )
                return coord
        except Exception as e:
            short = str(e).strip().splitlines()[0]
            self.logger.warning(
                f"SIMBAD resolve failed for {name}: {short}"
            )
        return None

    def run(self):
        print_status("Building multi-wavelength coverage manifest...", "PROCESS")

        catalog_path = self.data_processed / "arp_pair_catalog.csv"
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"Pair catalog not found: {catalog_path}. Run step_00 first."
            )
        catalog = pd.read_csv(catalog_path)

        try:
            from astroquery.heasarc import Heasarc
            from astroquery.vizier import Vizier
            from astroquery.exceptions import NoResultsWarning
        except ImportError:
            raise ImportError(
                "astroquery is required for archive queries. "
                "Install it with: pip install astroquery"
            )

        records = []
        n_queried = 0
        n_unresolved = 0
        for _, row in catalog.iterrows():
            pair_id = row["pair_id"]
            coord = self._resolve(row["galaxy"])
            if coord is None:
                n_unresolved += 1
                self.logger.error(
                    f"{pair_id}: galaxy '{row['galaxy']}' could not be "
                    "resolved by NED or SIMBAD; archive coverage for "
                    "this field is unavailable."
                )
                continue
            radius_arcmin = max(5.0, float(row["separation_arcsec"]) / 60.0 * 1.4)
            radius = radius_arcmin * u.arcmin

            # X-ray mission coverage
            for table, mission in XRAY_TABLES.items():
                n_queried += 1
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter(
                            "ignore", category=NoResultsWarning
                        )
                        res = Heasarc.query_region(
                            coord, catalog=table, radius=radius
                        )
                    n_obs = len(res) if res is not None else 0
                    status = "ok" if n_obs > 0 else "no_match"
                except Exception as e:
                    short = str(e).strip().splitlines()[0]
                    self.logger.warning(
                        f"{pair_id} {table} query failed: {short}"
                    )
                    n_obs = -1
                    status = "query_failed"
                records.append({
                    "pair_id": pair_id,
                    "band": "X-ray",
                    "archive": mission,
                    "catalog_or_table": table,
                    "radius_arcmin": radius_arcmin,
                    "n_records": n_obs,
                    "status": status,
                })
                time.sleep(0.3)

            # Radio continuum coverage
            for cat, label in RADIO_CATALOGS.items():
                n_queried += 1
                try:
                    v = Vizier(columns=["*", "+_r"], row_limit=200)
                    with warnings.catch_warnings():
                        warnings.simplefilter(
                            "ignore", category=NoResultsWarning
                        )
                        res = v.query_region(
                            coord, radius=radius, catalog=cat
                        )
                    n_src = len(res[0]) if res else 0
                    status = "ok" if n_src > 0 else "no_match"
                except Exception as e:
                    short = str(e).strip().splitlines()[0]
                    self.logger.warning(
                        f"{pair_id} {cat} query failed: {short}"
                    )
                    n_src = -1
                    status = "query_failed"
                records.append({
                    "pair_id": pair_id,
                    "band": "radio",
                    "archive": label,
                    "catalog_or_table": cat,
                    "radius_arcmin": radius_arcmin,
                    "n_records": n_src,
                    "status": status,
                })
                time.sleep(0.3)

            print_status(
                f"{pair_id}: queried {len(XRAY_TABLES)} X-ray + "
                f"{len(RADIO_CATALOGS)} radio archives", "TEST"
            )

        df = pd.DataFrame(records)
        csv_path = self.data_processed / "multiwavelength_manifest.csv"
        df.to_csv(csv_path, index=False)
        print_status(f"Saved manifest: {csv_path}", "SUCCESS")

        n_ok = int((df["status"] == "ok").sum())
        n_no_match = int((df["status"] == "no_match").sum())
        n_failed = int((df["status"] == "query_failed").sum())
        n_answered = n_ok + n_no_match
        summary = {
            "n_pairs": len(catalog),
            "n_queries": n_queried,
            "n_successful": n_answered,
            "n_with_records": n_ok,
            "n_no_match": n_no_match,
            "n_query_failed": n_failed,
            "n_unresolved_fields": n_unresolved,
            "records": records,
            "data_policy": (
                "Archive coverage queried live from HEASARC and CDS VizieR; "
                "negative n_records denotes a failed query, never a fabricated count."
            ),
        }
        json_path = self.results / "step_03_multiwavelength_manifest.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")

        if n_answered == 0:
            raise RuntimeError(
                "All archive queries failed; check network access to HEASARC/VizieR."
            )
        print_status(
            f"Multi-wavelength manifest complete: {n_answered}/{n_queried} "
            f"queries answered ({n_ok} with records, {n_no_match} empty, "
            f"{n_failed} failed; {n_unresolved} fields unresolved).",
            "SUCCESS",
        )
