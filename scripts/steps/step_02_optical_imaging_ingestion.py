#!/usr/bin/env python3
"""
Step 02: Optical Imaging Ingestion
==================================
Downloads archival optical imaging for each Arp pair field.
Surveys are attempted in preference order:

  1. DESI Legacy Imaging Surveys DR10 r-band (direct FITS cutout
     service; deep CCD imaging, 0.262 arcsec/pixel, covering the
     full northern sky) — the primary survey;
  2. SDSS r-band via SkyView (drift-scan CCD, 0.396 arcsec/pixel,
     partial footprint);
  3. DSS2 Red via SkyView (digitised photographic plates) as the
     universal fallback.

The images provide the spatial base on which the bridge transects
of step_20 are drawn.  Each pair field is centred on the host
galaxy and sized to cover the catalogued companion separation
with margin.

This step requires network access to the Legacy Survey cutout
service and the SkyView service.  If no image can be retrieved
for any field the step fails loudly rather than fabricating
morphology data.

Outputs:
    data/raw/skyview/<pair_id>_<survey>.fits
    results/outputs/step_02_optical_imaging.json
    data/processed/imaging_manifest.csv
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.jsonio import json_safe

# SkyView survey names, in preference order after the DESI Legacy
# primary.
SURVEYS = ["SDSSr", "DSS2 Red"]

# DESI Legacy Imaging Surveys DR10 cutout service (primary survey).
# r-band, native 0.262 arcsec/pixel — modern CCD imaging an order of
# magnitude deeper than the DSS2 photographic plates.
LS_PIXSCALE = 0.262            # arcsec/pixel
LS_LAYER = "ls-dr10"
LS_URL = "https://www.legacysurvey.org/viewer/fits-cutout"
LS_TIMEOUT = 180


class Step02OpticalImaging:
    """Step 02: Download optical imaging for Arp pair fields."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.data_raw = self.root / "data" / "raw" / "skyview"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.data_raw, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_02",
            log_file_path=self.logs / "step_02_optical_imaging_ingestion.log",
        )
        set_step_logger(self.logger)

    def _fetch_desi_legacy(self, position, radius_arcmin, out_path,
                           band="r"):
        """DESI Legacy Imaging DR10 FITS cutout.

        Resolves the target name through Simbad (CDS), then requests a
        native-pixel cutout from the Legacy Survey viewer service in the
        requested band.  Returns the download size in bytes, or None on
        failure (caller falls through to the SkyView surveys)."""
        import requests
        from astroquery.simbad import Simbad
        from astropy.io import fits

        try:
            import warnings as _warnings
            from astroquery.exceptions import NoResultsWarning
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore", category=NoResultsWarning)
                tbl = Simbad.query_object(position)
            ra, dec = float(tbl[0]["ra"]), float(tbl[0]["dec"])
        except Exception as e:
            short = str(e).strip().splitlines()[0]
            self.logger.info(
                f"Simbad resolution of {position} failed ({short}); "
                "falling back to SkyView."
            )
            return None
        size_px = int(np.ceil(radius_arcmin * 60.0 / LS_PIXSCALE) * 2)
        url = (
            f"{LS_URL}?ra={ra:.6f}&dec={dec:.6f}&layer={LS_LAYER}"
            f"&pixscale={LS_PIXSCALE}&size={size_px}&bands={band}"
        )
        try:
            r = requests.get(url, timeout=LS_TIMEOUT)
            r.raise_for_status()
            # Verify the payload is a real FITS file with actual
            # coverage before writing: the LS-DR10 layer has holes
            # (e.g. at the BASS/DECaLS boundary) that return
            # all-zero cutouts.
            with fits.open(__import__("io").BytesIO(r.content)) as hdul:
                data = hdul[0].data
                if data is None or float(np.mean(data != 0)) < 0.5:
                    self.logger.info(
                        f"{position} LS-DR10{band} cutout is blank "
                        "(coverage hole); falling back to SkyView."
                    )
                    return None
            out_path.write_bytes(r.content)
            return len(r.content)
        except Exception as e:
            short = str(e).strip().splitlines()[0]
            self.logger.info(
                f"{position} LS-DR10{band} request failed ({short}); "
                "falling back to SkyView."
            )
            return None

    def run(self):
        print_status("Downloading optical imaging for Arp pair fields...", "PROCESS")

        catalog_path = self.data_processed / "arp_pair_catalog.csv"
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"Pair catalog not found: {catalog_path}. Run step_00 first."
            )
        catalog = pd.read_csv(catalog_path)

        try:
            from astroquery.skyview import SkyView
        except ImportError:
            raise ImportError(
                "astroquery is required for archival imaging ingestion. "
                "Install it with: pip install astroquery"
            )

        manifest = []
        for _, row in catalog.iterrows():
            pair_id = row["pair_id"]
            target = row["galaxy"]
            # Field radius: separation + 40% margin, minimum 4 arcmin
            radius_arcmin = max(4.0, float(row["separation_arcsec"]) / 60.0 * 1.4)

            got_image = False

            # Primary survey: DESI Legacy Imaging DR10 r-band.
            ls_path = self.data_raw / f"{pair_id}_LS-DR10r.fits"
            if ls_path.exists() and ls_path.stat().st_size > 10_000:
                print_status(f"[SKIP] {ls_path.name} already present", "INFO")
                manifest.append({
                    "pair_id": pair_id, "survey": "LS-DR10r",
                    "file": str(ls_path.relative_to(self.root)),
                    "status": "skip_exists",
                    "bytes": ls_path.stat().st_size,
                })
                got_image = True
            else:
                nbytes = self._fetch_desi_legacy(
                    target, radius_arcmin, ls_path
                )
                if nbytes:
                    print_status(
                        f"[OK] {pair_id} LS-DR10r: {nbytes:,} bytes "
                        f"({radius_arcmin:.1f} arcmin radius)", "SUCCESS"
                    )
                    manifest.append({
                        "pair_id": pair_id, "survey": "LS-DR10r",
                        "file": str(ls_path.relative_to(self.root)),
                        "status": "downloaded",
                        "bytes": nbytes,
                        "radius_arcmin": radius_arcmin,
                    })
                    got_image = True

            # Independent second band: DESI Legacy DR10 g-band.
            # Detection of a claimed structure in two independent
            # bands excludes single-band detector/plate artifacts.
            g_path = self.data_raw / f"{pair_id}_LS-DR10g.fits"
            if g_path.exists() and g_path.stat().st_size > 10_000:
                print_status(f"[SKIP] {g_path.name} already present", "INFO")
                manifest.append({
                    "pair_id": pair_id, "survey": "LS-DR10g",
                    "file": str(g_path.relative_to(self.root)),
                    "status": "skip_exists",
                    "bytes": g_path.stat().st_size,
                })
            else:
                nbytes = self._fetch_desi_legacy(
                    target, radius_arcmin, g_path, band="g"
                )
                if nbytes:
                    print_status(
                        f"[OK] {pair_id} LS-DR10g: {nbytes:,} bytes",
                        "SUCCESS"
                    )
                    manifest.append({
                        "pair_id": pair_id, "survey": "LS-DR10g",
                        "file": str(g_path.relative_to(self.root)),
                        "status": "downloaded",
                        "bytes": nbytes,
                        "radius_arcmin": radius_arcmin,
                    })
                else:
                    manifest.append({
                        "pair_id": pair_id, "survey": "LS-DR10g",
                        "file": None, "status": "no_coverage", "bytes": 0,
                    })

            # Fallback: SkyView surveys (SDSS r-band, then DSS2 Red).
            for survey in ([] if got_image else SURVEYS):
                safe_survey = survey.replace(" ", "_")
                out_path = self.data_raw / f"{pair_id}_{safe_survey}.fits"
                if out_path.exists() and out_path.stat().st_size > 10_000:
                    print_status(f"[SKIP] {out_path.name} already present", "INFO")
                    manifest.append({
                        "pair_id": pair_id, "survey": survey,
                        "file": str(out_path.relative_to(self.root)),
                        "status": "skip_exists",
                        "bytes": out_path.stat().st_size,
                    })
                    got_image = True
                    break
                try:
                    imgs = SkyView.get_images(
                        position=target,
                        survey=[survey],
                        radius=radius_arcmin * u.arcmin,
                        pixels=1200,
                    )
                    if imgs and len(imgs) > 0:
                        imgs[0].writeto(out_path, overwrite=True)
                        size = out_path.stat().st_size
                        print_status(
                            f"[OK] {pair_id} {survey}: {size:,} bytes "
                            f"({radius_arcmin:.1f} arcmin radius)", "SUCCESS"
                        )
                        manifest.append({
                            "pair_id": pair_id, "survey": survey,
                            "file": str(out_path.relative_to(self.root)),
                            "status": "downloaded",
                            "bytes": size,
                            "radius_arcmin": radius_arcmin,
                        })
                        got_image = True
                        break
                except Exception as e:
                    short = str(e).strip().splitlines()[0]
                    self.logger.info(
                        f"{pair_id} {survey} request failed ({short}); "
                        "trying next survey."
                    )
                time.sleep(0.5)

            if not got_image:
                manifest.append({
                    "pair_id": pair_id, "survey": None,
                    "file": None, "status": "failed", "bytes": 0,
                })
                print_status(f"[FAIL] no imaging retrieved for {pair_id}", "ERROR")

        df = pd.DataFrame(manifest)
        csv_path = self.data_processed / "imaging_manifest.csv"
        df.to_csv(csv_path, index=False)
        print_status(f"Saved imaging manifest: {csv_path}", "SUCCESS")

        n_ok = int(df["status"].isin(["downloaded", "skip_exists"]).sum())
        n_fail = int((df["status"] == "failed").sum())

        summary = {
            "n_fields": len(catalog),
            "n_images": n_ok,
            "n_failed": n_fail,
            "surveys": ["LS-DR10r"] + SURVEYS,
            "manifest": manifest,
            "data_policy": (
                "All images retrieved from the DESI Legacy Imaging Surveys "
                "DR10 cutout service or NASA SkyView (SDSS/DSS2); no "
                "synthetic imaging."
            ),
        }
        json_path = self.results / "step_02_optical_imaging.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")

        if n_ok == 0:
            raise RuntimeError(
                "No archival imaging retrieved for any pair field. "
                "Check network access to SkyView; refusing to fabricate morphology."
            )
        print_status(
            f"Imaging ingestion complete: {n_ok} fields imaged, {n_fail} failed.",
            "SUCCESS",
        )
