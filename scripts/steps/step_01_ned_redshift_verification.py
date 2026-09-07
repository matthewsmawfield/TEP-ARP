#!/usr/bin/env python3
"""
Step 01: NED Redshift Verification
==================================
Verifies every catalog redshift against the NASA/IPAC Extragalactic
Database (NED) using astroquery.ned.  For each galaxy and companion
the step queries the object record, extracts the heliocentric
redshift, and compares it to the literature value in the catalog.

Discrepancies larger than the tolerance are reported as TEST
failures and written to the manifest; entries flagged
``literature_approx`` in step_00 are replaced by the NED value
where a secure identification exists, including when the archive
value differs from the literature value beyond tolerance (recorded
as ``corrected``).

Companions entered under descriptive (non-resolvable) labels are
verified through the catalogue's ``companion_ned_name`` field or,
failing that, a NED cone search at the catalogue's resolved
companion coordinates (``companion_ra_deg``/``companion_dec_deg``),
taking the nearest object carrying a redshift.

This step requires network access to NED.  If the archive cannot
be reached the step fails loudly rather than fabricating values.

Outputs:
    results/outputs/step_01_ned_verification.json
    data/processed/ned_verification.csv
    data/processed/arp_pair_catalog_verified.csv
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.jsonio import json_safe

Z_TOLERANCE = 0.01  # fractional |Delta z| / (1+z) tolerance for verification


class Step01NEDVerification:
    """Step 01: Verify catalog redshifts against NED."""

    # NED object names where they differ from the catalog label.
    NED_NAME_MAP = {
        "Markarian 205": "MRK 205",
        "NGC 7603B": "NGC 7603B",
        "3C 232": "3C 232",
    }

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_01",
            log_file_path=self.logs / "step_01_ned_redshift_verification.log",
        )
        set_step_logger(self.logger)

    def _query_ned_redshift(self, name):
        """Return (redshift, object_type) for a NED object, or (None, None).

        Permanent name-resolution failures ("not currently recognized by
        the NED name interpreter") are detected on the first attempt and
        reported once at INFO level — retrying them cannot succeed.
        Transient transport/service failures are retried up to three
        times with a short backoff, logged at INFO level, since the
        caller provides further fallback resolvers (Simbad, cone
        search).  A warning is emitted only if every resolver fails."""
        from astroquery.ipac.ned import Ned

        ned_name = self.NED_NAME_MAP.get(name, name)
        tbl = None
        for attempt in (1, 2, 3):
            try:
                tbl = Ned.query_object(ned_name)
                break
            except Exception as e:
                msg = str(e)
                if "not currently recognized" in msg:
                    self.logger.info(
                        f"NED name lookup: '{ned_name}' is not a "
                        "resolvable catalogue identifier; trying "
                        "alternate resolvers."
                    )
                    return None, None
                short = msg.strip().splitlines()[0]
                if attempt < 3:
                    self.logger.info(
                        f"NED query for '{ned_name}' attempt "
                        f"{attempt}/3 failed ({short}); retrying."
                    )
                    time.sleep(2.0 * attempt)
                else:
                    self.logger.info(
                        f"NED query for '{ned_name}' failed after 3 "
                        f"attempts ({short}); trying alternate resolvers."
                    )
                    return None, None
        if tbl is None or len(tbl) == 0:
            return None, None
        row = tbl[0]
        z = None
        for col in ("Redshift", "z"):
            if col in tbl.colnames and row[col] is not np.ma.masked:
                try:
                    z = float(row[col])
                except (TypeError, ValueError):
                    z = None
                break
        otype = str(row["Type"]) if "Type" in tbl.colnames else ""
        return z, otype

    def _query_simbad_redshift(self, name):
        """Return (redshift, object_type) for a SIMBAD object, or
        (None, None).  Used as the secondary resolver when a NED name
        lookup fails or the NED service is unreachable."""
        from astroquery.simbad import Simbad
        from astroquery.exceptions import NoResultsWarning
        import warnings

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=NoResultsWarning)
                tbl = Simbad.query_object(name)
        except Exception as e:
            short = str(e).strip().splitlines()[0]
            self.logger.info(
                f"SIMBAD lookup for '{name}' unavailable ({short})."
            )
            return None, None
        if tbl is None or len(tbl) == 0:
            return None, None
        row = tbl[0]
        z = None
        for col in ("rvz_redshift", "redshift"):
            if col in tbl.colnames and not np.ma.is_masked(row[col]):
                try:
                    z = float(row[col])
                except (TypeError, ValueError):
                    z = None
                break
        otype = str(row["otype"]) if "otype" in tbl.colnames else ""
        if z is not None:
            self.logger.info(
                f"'{name}' resolved via SIMBAD fallback "
                f"(z={z:.6f}, type={otype or 'n/a'})."
            )
        return z, otype

    def _query_ned_cone_redshift(self, ra_deg, dec_deg, radius_arcsec=15.0):
        """Return (redshift, object_type, object_name) of the nearest
        NED object carrying a redshift within ``radius_arcsec`` of the
        given position, or (None, None, None)."""
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        from astroquery.ipac.ned import Ned

        pos = SkyCoord(float(ra_deg), float(dec_deg), unit="deg")
        tbl = None
        for attempt in (1, 2, 3):
            try:
                tbl = Ned.query_region(pos, radius=radius_arcsec * u.arcsec)
                break
            except Exception as e:
                short = str(e).strip().splitlines()[0]
                if attempt < 3:
                    self.logger.info(
                        f"NED cone query at ({ra_deg:.5f}, {dec_deg:.5f}) "
                        f"attempt {attempt}/3 failed ({short}); retrying."
                    )
                    time.sleep(2.0 * attempt)
                else:
                    self.logger.info(
                        f"NED cone query at ({ra_deg:.5f}, {dec_deg:.5f}) "
                        f"failed after 3 attempts ({short})."
                    )
                    return None, None, None
        if tbl is None or len(tbl) == 0:
            return None, None, None
        df = tbl.to_pandas()
        if "Redshift" not in df.columns:
            return None, None, None
        df = df[pd.to_numeric(df["Redshift"], errors="coerce").notna()]
        if len(df) == 0:
            return None, None, None
        df = df.sort_values("Separation")
        row = df.iloc[0]
        return (
            float(row["Redshift"]),
            str(row.get("Type", "")),
            str(row.get("Object Name", "")),
        )

    def _resolve_redshift(self, name):
        """Resolve an object's redshift through the resolver chain
        NED name -> SIMBAD name.  Returns (z, otype, method)."""
        ned_z, otype = self._query_ned_redshift(name)
        if ned_z is not None:
            return ned_z, otype, "name"
        time.sleep(0.3)
        sim_z, sotype = self._query_simbad_redshift(name)
        if sim_z is not None:
            return sim_z, sotype, "simbad_name"
        return None, None, "unresolved"

    def _companion_ned_redshift(self, row):
        """Verify a companion redshift, falling back from the catalogue
        name to ``companion_ned_name``, then SIMBAD, and finally to a
        cone search at the catalogue's resolved companion coordinates.
        Returns (z, otype, method)."""
        ned_z, otype = self._query_ned_redshift(str(row["companion"]))
        if ned_z is not None:
            return ned_z, otype, "name"
        alt = row.get("companion_ned_name")
        if isinstance(alt, str) and alt.strip():
            time.sleep(0.3)
            ned_z, otype = self._query_ned_redshift(alt.strip())
            if ned_z is not None:
                return ned_z, otype, "ned_name"
        time.sleep(0.3)
        sim_z, sotype = self._query_simbad_redshift(str(row["companion"]))
        if sim_z is not None:
            return sim_z, sotype, "simbad_name"
        ra, dec = row.get("companion_ra_deg"), row.get("companion_dec_deg")
        if pd.notna(ra) and pd.notna(dec):
            time.sleep(0.3)
            ned_z, otype, obj = self._query_ned_cone_redshift(ra, dec)
            if ned_z is not None:
                self.logger.info(
                    f"Companion of {row['pair_id']} verified by cone "
                    f"search as {obj} (z={ned_z})."
                )
                return ned_z, otype, f"cone:{obj}"
        return None, None, "unresolved"

    def run(self):
        print_status("Verifying redshifts against NED...", "PROCESS")

        catalog_path = self.data_processed / "arp_pair_catalog.csv"
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"Pair catalog not found: {catalog_path}. Run step_00 first."
            )
        catalog = pd.read_csv(catalog_path)

        records = []
        verified = catalog.copy()
        for _, row in catalog.iterrows():
            for role, name_key, z_key in (
                ("galaxy", "galaxy", "z_gal"),
                ("companion", "companion", "z_comp"),
            ):
                name = row[name_key]
                lit_z = float(row[z_key])

                # Companions that are multiple objects cannot be queried as one
                if ";" in str(row.get("z_comp_list", "")) and role == "companion":
                    print_status(
                        f"Skipping multi-object companion '{name}' "
                        "(queried individually in later steps)", "INFO"
                    )
                    continue

                if role == "companion":
                    ned_z, otype, method = self._companion_ned_redshift(row)
                else:
                    ned_z, otype, method = self._resolve_redshift(name)
                time.sleep(0.3)

                if ned_z is None:
                    status = "no_ned_match"
                    frac = np.nan
                    self.logger.warning(
                        f"{role} '{name}' could not be verified against "
                        "NED or SIMBAD; recorded as no_ned_match."
                    )
                else:
                    frac = abs(ned_z - lit_z) / (1.0 + lit_z)
                    status = "verified" if frac <= Z_TOLERANCE else "discrepant"

                # Replace approximate literature values with the NED value
                # wherever a secure archive identification exists — including
                # when the archive redshift differs beyond tolerance, in which
                # case the literature value is recorded as corrected.
                if (
                    ned_z is not None
                    and row["redshift_source"] == "literature_approx"
                ):
                    verified.loc[verified["pair_id"] == row["pair_id"], z_key] = ned_z
                    verified.loc[
                        verified["pair_id"] == row["pair_id"], "redshift_source"
                    ] = "ned_verified"
                    if status == "discrepant":
                        status = "corrected"

                records.append({
                    "pair_id": row["pair_id"],
                    "role": role,
                    "object": name,
                    "literature_z": lit_z,
                    "ned_z": ned_z,
                    "ned_type": otype,
                    "match_method": method,
                    "frac_diff": frac,
                    "status": status,
                })
                print_status(
                    f"{name}: lit z={lit_z:.4f}  NED z="
                    f"{ned_z if ned_z is not None else 'n/a'}  [{status}]",
                    "TEST",
                )

        dfv = pd.DataFrame(records)
        csv_path = self.data_processed / "ned_verification.csv"
        dfv.to_csv(csv_path, index=False)
        print_status(f"Saved verification table: {csv_path}", "SUCCESS")

        # Recompute derived columns for the verified catalog
        verified["one_plus_z_ratio"] = (1.0 + verified["z_comp"]) / (
            1.0 + verified["z_gal"]
        )
        verified["hubble_distance_factor"] = verified["z_comp"] / verified["z_gal"]
        vpath = self.data_processed / "arp_pair_catalog_verified.csv"
        verified.to_csv(vpath, index=False)
        print_status(f"Saved verified catalog: {vpath}", "SUCCESS")

        n_verified = int((dfv["status"] == "verified").sum())
        n_discrepant = int((dfv["status"] == "discrepant").sum())
        n_corrected = int((dfv["status"] == "corrected").sum())
        n_missing = int((dfv["status"] == "no_ned_match").sum())

        summary = {
            "n_queried": len(dfv),
            "n_verified": n_verified,
            "n_discrepant": n_discrepant,
            "n_corrected": n_corrected,
            "n_no_match": n_missing,
            "tolerance_frac": Z_TOLERANCE,
            "records": records,
        }
        json_path = self.results / "step_01_ned_verification.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")

        if n_verified + n_corrected == 0 and len(dfv) > 0:
            raise RuntimeError(
                "No NED redshifts could be retrieved; archive unreachable or "
                "query interface changed. Refusing to continue on unverified data."
            )
        print_status(
            f"NED verification complete: {n_verified} verified, "
            f"{n_corrected} corrected, "
            f"{n_discrepant} discrepant, {n_missing} unmatched.",
            "SUCCESS",
        )
