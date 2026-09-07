#!/usr/bin/env python3
"""
Step 04: Spectroscopic Data Ingestion
=====================================
Queries SDSS DR18 for spectroscopic coverage of each pair member
and downloads available spectra (FITS) for the members inside the
SDSS footprint.  Where a companion is outside the footprint the
step records the absence rather than substituting synthetic data.

The downloaded spectra feed the emission-line redshift transects
of step_30 and provide the independent spectroscopic confirmation
of the catalog redshifts.

This step requires network access to the SDSS archive.

Outputs:
    data/raw/sdss_spectra/<object>.fits
    data/processed/spectroscopic_manifest.csv
    results/outputs/step_04_spectroscopic_ingestion.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import warnings
from astropy import units as u

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.jsonio import json_safe


class Step04SpectroscopicIngestion:
    """Step 04: Ingest SDSS spectroscopy for pair members."""

    # NED/SIMBAD-resolvable names for SDSS cross-ID.
    OBJECT_NAME_MAP = {
        "Markarian 205": "Mrk 205",
    }

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.data_raw = self.root / "data" / "raw" / "sdss_spectra"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.data_raw, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_04",
            log_file_path=self.logs / "step_04_spectroscopic_ingestion.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status("Ingesting SDSS spectroscopy for pair members...", "PROCESS")

        catalog_path = self.data_processed / "arp_pair_catalog.csv"
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"Pair catalog not found: {catalog_path}. Run step_00 first."
            )
        catalog = pd.read_csv(catalog_path)

        try:
            from astroquery.ipac.ned import Ned
            from astroquery.sdss import SDSS
            from astropy.coordinates import SkyCoord
        except ImportError:
            raise ImportError(
                "astroquery is required for SDSS ingestion. "
                "Install it with: pip install astroquery"
            )

        records = []
        for _, row in catalog.iterrows():
            for role in ("galaxy", "companion"):
                name = row[role]
                query_name = self.OBJECT_NAME_MAP.get(name, name)

                # Skip multi-object companions; queried separately later
                if role == "companion" and ";" in str(row.get("z_comp_list", "")):
                    continue
                # Skip unresolved non-catalog labels
                if name.startswith(("compact companion", "QSO", "X-ray", "three")):
                    records.append({
                        "pair_id": row["pair_id"], "object": name,
                        "role": role, "status": "unresolved_name",
                        "n_spectra": 0,
                    })
                    continue

                try:
                    tbl = Ned.query_object(query_name)
                    if tbl is None or len(tbl) == 0:
                        raise ValueError("no NED match")
                    pos = SkyCoord(
                        float(tbl[0]["RA"]) * u.deg,
                        float(tbl[0]["DEC"]) * u.deg,
                    )
                except Exception:
                    # SIMBAD fallback: transient NED failures must not
                    # strand a resolvable archive object.
                    try:
                        from astroquery.simbad import Simbad
                        from astroquery.exceptions import NoResultsWarning
                        with warnings.catch_warnings():
                            warnings.simplefilter(
                                "ignore", category=NoResultsWarning
                            )
                            tbl = Simbad.query_object(query_name)
                        if tbl is None or len(tbl) == 0:
                            raise ValueError("no SIMBAD match")
                        pos = SkyCoord(
                            float(tbl[0]["ra"]) * u.deg,
                            float(tbl[0]["dec"]) * u.deg,
                        )
                        self.logger.info(
                            f"{name} resolved via SIMBAD fallback."
                        )
                    except Exception as e:
                        records.append({
                            "pair_id": row["pair_id"], "object": name,
                            "role": role, "status": f"resolve_failed: {e}",
                            "n_spectra": 0,
                        })
                        continue

                try:
                    with warnings.catch_warnings():
                        # SDSS specobjid exceeds int64; astropy's ASCII
                        # reader auto-recovers to String, which is the
                        # desired type — suppress the cosmetic notice.
                        warnings.filterwarnings(
                            "ignore", message=".*converting to IntType.*"
                        )
                        xid = SDSS.query_region(
                            pos, radius=6 * u.arcsec, spectro=True
                        )
                except Exception as e:
                    records.append({
                        "pair_id": row["pair_id"], "object": name,
                        "role": role, "status": f"sdss_query_failed: {e}",
                        "n_spectra": 0,
                    })
                    time.sleep(0.3)
                    continue

                if xid is None or len(xid) == 0:
                    records.append({
                        "pair_id": row["pair_id"], "object": name,
                        "role": role, "status": "no_sdss_spectrum",
                        "n_spectra": 0,
                    })
                    print_status(
                        f"{name}: no SDSS spectrum in footprint", "INFO"
                    )
                    continue

                spec_rows = xid[xid["instrument"] == "SDSS"] if "instrument" in xid.colnames else xid
                try:
                    spectra = SDSS.get_spectra(matches=spec_rows)
                except Exception as e:
                    records.append({
                        "pair_id": row["pair_id"], "object": name,
                        "role": role, "status": f"download_failed: {e}",
                        "n_spectra": 0,
                    })
                    continue

                n_saved = 0
                for i, sp in enumerate(spectra or []):
                    safe = name.replace(" ", "_")
                    out = self.data_raw / f"{safe}_{i}.fits"
                    try:
                        sp.writeto(out, overwrite=True)
                        n_saved += 1
                    except Exception as e:
                        self.logger.warning(f"save failed {out.name}: {e}")

                records.append({
                    "pair_id": row["pair_id"], "object": name,
                    "role": role,
                    "status": "downloaded" if n_saved else "download_failed",
                    "n_spectra": n_saved,
                    "sdss_z": float(xid[0]["z"]) if "z" in xid.colnames else np.nan,
                })
                print_status(
                    f"{name}: {n_saved} SDSS spectrum/a downloaded", "SUCCESS"
                )
                time.sleep(0.3)

        df = pd.DataFrame(records)
        csv_path = self.data_processed / "spectroscopic_manifest.csv"
        df.to_csv(csv_path, index=False)
        print_status(f"Saved spectroscopic manifest: {csv_path}", "SUCCESS")

        n_dl = int(df["n_spectra"].sum())
        summary = {
            "n_members_queried": len(df),
            "n_spectra_downloaded": n_dl,
            "records": records,
            "data_policy": (
                "Spectra downloaded from SDSS DR18 via astroquery; members "
                "outside the footprint are recorded, not substituted."
            ),
        }
        json_path = self.results / "step_04_spectroscopic_ingestion.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")
        print_status(
            f"Spectroscopic ingestion complete: {n_dl} spectra retrieved.",
            "SUCCESS",
        )
