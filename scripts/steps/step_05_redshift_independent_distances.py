#!/usr/bin/env python3
"""
Step 05: Redshift-Independent Distance Anchoring
================================================
Queries the Cosmicflows-4 distance catalogue (VizieR J/ApJ/944/94;
Tully et al. 2023, ApJ 944, 94) for redshift-independent distance
moduli of every catalogue member — host galaxies resolved through
Simbad and companions at the archive positions resolved by step_20.

Cosmicflows-4 aggregates the standard primary and secondary
redshift-independent methods (Cepheids, tip of the red giant
branch, masers, surface-brightness fluctuations, Tully-Fisher,
fundamental plane, Type Ia supernovae) into a combined distance
modulus DM with uncertainty e_DM, plus the method-specific moduli
where available.  A matched entry converts each member's projected
separation and every physical scale in the analysis (transect
lengths, well-width bounds) from a redshift-inferred quantity into
a measured one, and tests whether the host distance implied by
its redshift is consistent with the independent measurement.

Companions are quasars and compact knots and are not expected to
carry TF/SBF/TRGB measurements; where a companion's nearest
catalogue entry is the host's own PGC record this is recorded as
an absent independent distance rather than a match.  The coverage
table thereby also enumerates which members could, in principle,
settle the association question with a future primary-method
distance.

This step requires network access to CDS (Simbad, VizieR).  It
fails loudly if the archive is unreachable; legitimately absent
catalogue entries are recorded, not substituted.

Outputs:
    data/processed/redshift_independent_distances.csv
    results/outputs/step_05_redshift_independent_distances.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.jsonio import json_safe

CF4_CATALOG = "J/ApJ/944/94"          # Cosmicflows-4, Tully et al. 2023
CONE_RADIUS_ARCMIN = 2.0              # cone-search radius per member
MATCH_RADIUS_ARCSEC = 30.0            # positional match threshold
QUERY_TIMEOUT = 60                    # per-query timeout (seconds)
C_KM_S = 299792.458
H0_KM_S_MPC = 70.0                    # pipeline convention (step_13)

# Method-specific distance-modulus columns in CF4.
METHOD_COLUMNS = {
    "DMtf": "tully_fisher",
    "DMsbf": "surface_brightness_fluctuation",
    "DMfp": "fundamental_plane",
    "DMceph": "cepheid",
    "DMtrgb": "tip_red_giant_branch",
    "DMsnIa": "sn_ia",
    "DMmas": "maser",
}


class Step05RedshiftIndependentDistances:
    """Step 05: CF4 redshift-independent distances for members."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_05",
            log_file_path=self.logs
            / "step_05_redshift_independent_distances.log",
        )
        set_step_logger(self.logger)

    def _resolve_name(self, name):
        """Resolve an object name to J2000 coordinates via Simbad."""
        from astroquery.simbad import Simbad
        from astroquery.exceptions import NoResultsWarning
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=NoResultsWarning)
            tbl = Simbad.query_object(name)
        if tbl is None or len(tbl) == 0:
            return None
        return SkyCoord(float(tbl[0]["ra"]) * u.deg,
                        float(tbl[0]["dec"]) * u.deg)

    def _query_cf4(self, coord):
        """Nearest Cosmicflows-4 entry within the cone, or None.
        Returns the sentinel string 'unreachable' on query failure."""
        from astroquery.vizier import Vizier

        viz = Vizier(row_limit=50, timeout=QUERY_TIMEOUT)
        for attempt in (1, 2):
            try:
                res = viz.query_region(
                    coord, radius=CONE_RADIUS_ARCMIN * u.arcmin,
                    catalog=CF4_CATALOG,
                )
                break
            except Exception as e:
                short = str(e).strip().splitlines()[0]
                if attempt < 2:
                    self.logger.info(
                        f"VizieR CF4 query attempt {attempt}/2 failed "
                        f"({short}); retrying."
                    )
                else:
                    self.logger.warning(
                        f"VizieR CF4 query failed after 2 attempts: "
                        f"{short}"
                    )
                res = None
                if attempt == 2:
                    return "unreachable"
                time.sleep(1.0)
        if res is None or len(res.keys()) == 0:
            return None
        tbl = res[res.keys()[0]]
        if len(tbl) == 0:
            return None
        best, best_sep = None, np.inf
        for row in tbl:
            pos = SkyCoord(float(row["RAJ2000"]) * u.deg,
                           float(row["DEJ2000"]) * u.deg)
            sep = coord.separation(pos).to(u.arcsec).value
            if sep < best_sep:
                best, best_sep = row, sep
        return best, best_sep

    def run(self):
        print_status(
            "Querying Cosmicflows-4 redshift-independent distances...",
            "PROCESS",
        )

        catalog_path = self.data_processed / "arp_pair_catalog_verified.csv"
        if not catalog_path.exists():
            catalog_path = self.data_processed / "arp_pair_catalog.csv"
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"Pair catalog not found: {catalog_path}. Run step_00 first."
            )
        catalog = pd.read_csv(catalog_path)

        comp_path = self.data_processed / "companion_positions.csv"
        comp_coords = {}
        if comp_path.exists():
            dfc = pd.read_csv(comp_path)
            for pid, grp in dfc.groupby("pair_id"):
                comp_coords[pid] = [
                    (str(r.resolved_name),
                     SkyCoord(r.ra_deg * u.deg, r.dec_deg * u.deg))
                    for r in grp.itertuples()
                ]
        else:
            self.logger.warning(
                "companion_positions.csv not found; companions will be "
                "resolved by name only."
            )

        records = []
        n_unreachable = 0
        n_members = 0

        for _, row in catalog.iterrows():
            pair_id = row["pair_id"]
            members = [("host", str(row["galaxy"]), None)]
            for cname, cpos in comp_coords.get(pair_id, []):
                members.append(("companion", cname, cpos))
            if pair_id not in comp_coords:
                members.append(("companion", str(row["companion"]), None))

            host_pgc = None
            for role, name, pos in members:
                n_members += 1
                coord = pos
                if coord is None:
                    try:
                        coord = self._resolve_name(name)
                    except Exception as e:
                        short = str(e).strip().splitlines()[0]
                        self.logger.info(
                            f"Simbad resolve failed for {name} "
                            f"({short}); recorded as unresolved."
                        )
                        coord = None
                if coord is None:
                    records.append({
                        "pair_id": pair_id, "role": role, "object": name,
                        "status": "unresolved",
                    })
                    continue

                hit = self._query_cf4(coord)
                time.sleep(0.3)
                if hit == "unreachable":
                    n_unreachable += 1
                    records.append({
                        "pair_id": pair_id, "role": role, "object": name,
                        "status": "query_failed",
                    })
                    continue
                if hit is None:
                    records.append({
                        "pair_id": pair_id, "role": role, "object": name,
                        "status": "not_in_cf4",
                    })
                    print_status(f"{name}: no CF4 entry", "INFO")
                    continue

                cf4row, sep_arcsec = hit
                dm = float(cf4row["DM"])
                e_dm = float(cf4row["e_DM"])
                d_mpc = float(10.0 ** ((dm - 25.0) / 5.0))
                # Symmetrised distance uncertainty from the modulus error.
                e_mpc = float(d_mpc * np.log(10.0) / 5.0 * e_dm)
                methods = {
                    col: METHOD_COLUMNS[col]
                    for col in METHOD_COLUMNS
                    if col in cf4row.colnames
                    and str(cf4row[col]) not in ("--", "")
                    and np.isfinite(float(cf4row[col]))
                }
                pgc = str(cf4row["PGC"])
                if role == "host" and sep_arcsec <= MATCH_RADIUS_ARCSEC:
                    host_pgc = pgc
                # Positional match threshold: beyond MATCH_RADIUS_ARCSEC
                # the nearest entry is a different object (e.g. a compact-
                # group neighbour), not a measurement of the member.
                # A companion whose nearest CF4 entry is the host's own
                # record has no independent distance measurement.
                if sep_arcsec > MATCH_RADIUS_ARCSEC:
                    status = "no_cf4_match"
                elif role == "companion" and host_pgc is not None \
                        and pgc == host_pgc:
                    status = "no_independent_entry_nearest_is_host"
                else:
                    status = "measured"

                z_lit = (float(row["z_gal"]) if role == "host"
                         else float(row["z_comp"]))
                z_dist_mpc = C_KM_S * z_lit / H0_KM_S_MPC
                rec = {
                    "pair_id": pair_id, "role": role, "object": name,
                    "status": status,
                    "nearest_entry_is_host": bool(
                        host_pgc is not None and pgc == host_pgc
                    ),
                    "pgc": pgc,
                    "match_separation_arcsec": float(sep_arcsec),
                    "vcmb_kms": float(cf4row["Vcmb"]),
                    "distance_modulus": dm,
                    "e_distance_modulus": e_dm,
                    "distance_mpc": d_mpc,
                    "e_distance_mpc": e_mpc,
                    "methods": ";".join(methods.values()),
                    "literature_z": z_lit,
                    "z_distance_mpc_h70": float(z_dist_mpc),
                    "distance_ratio_cf4_over_z": (
                        float(d_mpc / z_dist_mpc) if z_dist_mpc > 0 else None
                    ),
                }
                records.append(rec)
                print_status(
                    f"{name}: DM = {dm:.2f} ± {e_dm:.2f} -> "
                    f"{d_mpc:.1f} ± {e_mpc:.1f} Mpc "
                    f"({rec['methods'] or 'combined'}; "
                    f"z-distance {z_dist_mpc:.1f} Mpc) [{status}]",
                    "TEST",
                )

        if n_unreachable == n_members:
            raise RuntimeError(
                "All Cosmicflows-4 queries failed; archive unreachable. "
                "Refusing to write distance table."
            )

        df = pd.DataFrame(records)
        csv_path = self.data_processed / "redshift_independent_distances.csv"
        df.to_csv(csv_path, index=False)
        print_status(f"Saved distance table: {csv_path}", "SUCCESS")

        measured = df[df["status"] == "measured"]
        n_hosts_measured = int((measured["role"] == "host").sum())
        n_comp_measured = int((measured["role"] == "companion").sum())

        summary = {
            "catalog": (
                "Cosmicflows-4 (VizieR J/ApJ/944/94; Tully et al. 2023, "
                "ApJ 944, 94)"
            ),
            "cone_radius_arcmin": CONE_RADIUS_ARCMIN,
            "h0_km_s_mpc": H0_KM_S_MPC,
            "n_members_queried": n_members,
            "n_members_measured": int(len(measured)),
            "n_hosts_measured": n_hosts_measured,
            "n_companions_measured": n_comp_measured,
            "members": records,
            "method": (
                "Simbad name resolution (hosts; companions use step_20 "
                f"archive positions); VizieR cone search of Cosmicflows-4 "
                f"within {CONE_RADIUS_ARCMIN} arcmin; nearest entry "
                f"adopted only inside {MATCH_RADIUS_ARCSEC} arcsec, beyond "
                "which it is a different object (compact-group "
                "neighbour). A companion whose nearest entry shares the "
                "host PGC is recorded as lacking an independent "
                "measurement. Distances derive from the catalogue "
                "combined modulus DM ± e_DM; the method-specific moduli "
                "present are listed per member. The z-distance uses "
                f"c·z/H0 with H0 = {H0_KM_S_MPC} km/s/Mpc, the pipeline "
                "convention."
            ),
        }
        json_path = self.results / "step_05_redshift_independent_distances.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")
        print_status(
            f"Distance anchoring: {n_hosts_measured} hosts measured, "
            f"{n_comp_measured} companions with independent distances.",
            "SUCCESS",
        )


if __name__ == "__main__":
    Step05RedshiftIndependentDistances().run()
