#!/usr/bin/env python3
"""
Step 48: Falsification Adjudication Ledger
==========================================

Separates three logically distinct quantities that must not be pooled:

1. catalogue-pair adjudication by a decisive path-length observation;
2. archival coverage that is present but non-discriminating; and
3. the population control supplied by SDSS and DESI projected quasars.

The step also de-duplicates repeated galaxy projections of the same spectrum,
de-duplicates the SDSS/DESI overlap on the sky, and excludes SDSS catalogue
redshifts rejected by the multi-line audit from the strong-forest denominator.

Outputs:
    data/processed/falsification_adjudication.csv
    results/outputs/step_48_falsification_adjudication.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.jsonio import json_safe
from scripts.utils.logger import TEPLogger, print_status, set_step_logger


class Step48FalsificationAdjudication:
    """Build the pair-level and population-level falsification ledger."""

    def __init__(self):
        self.data = PROJECT_ROOT / "data" / "processed"
        self.results = PROJECT_ROOT / "results" / "outputs"
        self.logs = PROJECT_ROOT / "logs"
        for directory in (self.data, self.results, self.logs):
            directory.mkdir(parents=True, exist_ok=True)
        self.logger = TEPLogger(
            "step_48",
            log_file_path=self.logs / "step_48_falsification_adjudication.log",
        )
        set_step_logger(self.logger)

    @staticmethod
    def _load_json(path):
        if not path.exists():
            raise FileNotFoundError(f"Required pipeline artifact missing: {path}")
        return json.loads(path.read_text())

    def _pair_ledger(self):
        catalog = pd.read_csv(self.data / "arp_pair_catalog_verified.csv")
        s44 = self._load_json(
            self.results / "step_44_survey_erasure_audit.json"
        )

        rows = []
        for pair_id in catalog["pair_id"]:
            if pair_id == "NGC3067-3C232":
                status = "falsified"
                evidence = (
                    "HST/COS forest: 9 Lyman-confirmed intervening systems "
                    "against 2.3 expected; cosmological path detected"
                )
            elif pair_id == "NGC4319-Mrk205":
                status = "archival_non_discriminating"
                mrk = s44["mrk205_forest_band_audit"]
                evidence = (
                    f"HST/STIS E140M audit: {mrk['n_candidates']} candidate "
                    "intervening Ly-alpha features over a short path; host-z "
                    "window continuum-starved and non-discriminating"
                )
            elif pair_id == "NGC7319-QSO":
                status = "archival_non_discriminating"
                evidence = (
                    "Keck/LRIS spectrum verifies z=2.114 and excludes only "
                    "strong absorption; R~1400 and S/N~5/A are insufficient "
                    "for a forest or intervening-metal census"
                )
            else:
                status = "no_decisive_path_data"
                evidence = "No public spectrum supplies a decisive path-length test"
            rows.append({
                "pair_id": pair_id,
                "adjudication": status,
                "evidence": evidence,
            })
        return pd.DataFrame(rows)

    def _strong_population(self):
        sdss = pd.read_csv(self.data / "extended_forest_audit.csv")
        desi = pd.read_csv(self.data / "desi_forest_audit.csv")

        # A rejected pipeline redshift cannot define a high-z path length.
        sdss_strong = sdss[
            sdss["verdict"].isin(["forest_normal", "forest_candidate"])
            & (sdss["transmission_expected"] < 0.5)
        ].copy()
        sdss_strong = sdss_strong.drop_duplicates("sp_id")
        sdss_strong["survey"] = "SDSS"
        sdss_strong["spectrum_id"] = sdss_strong["sp_id"].astype(str)
        sdss_strong["ra"] = sdss_strong["ra_qso"]
        sdss_strong["dec"] = sdss_strong["dec_qso"]

        desi_strong = desi[
            (desi["verdict"] == "measured")
            & (desi["transmission_expected"] < 0.5)
        ].copy()
        # One DESI target may project within 100 arcsec of more than one 2MRS
        # galaxy; it remains one independent forest sightline.
        desi_strong = desi_strong.drop_duplicates("targetid")
        desi_strong["survey"] = "DESI"
        desi_strong["spectrum_id"] = desi_strong["targetid"].astype(str)
        desi_strong["ra"] = desi_strong["ra_q"]
        desi_strong["dec"] = desi_strong["dec_q"]

        # Remove spectra repeated across surveys.  A one-arcsec match and
        # |Delta z| < 0.02 identify the same astrophysical sightline.
        cross_matches = []
        keep_desi = np.ones(len(desi_strong), dtype=bool)
        if len(sdss_strong) and len(desi_strong):
            cs = SkyCoord(
                sdss_strong["ra"].to_numpy() * u.deg,
                sdss_strong["dec"].to_numpy() * u.deg,
            )
            cd = SkyCoord(
                desi_strong["ra"].to_numpy() * u.deg,
                desi_strong["dec"].to_numpy() * u.deg,
            )
            idx, sep, _ = cd.match_to_catalog_sky(cs)
            z_sdss = sdss_strong["z_qso"].to_numpy()[idx]
            duplicate = (sep.arcsec < 1.0) & (
                np.abs(desi_strong["z_qso"].to_numpy() - z_sdss) < 0.02
            )
            keep_desi = ~duplicate
            for j in np.flatnonzero(duplicate):
                cross_matches.append({
                    "desi_targetid": str(desi_strong.iloc[j]["targetid"]),
                    "sdss_sp_id": str(sdss_strong.iloc[idx[j]]["sp_id"]),
                    "separation_arcsec": float(sep.arcsec[j]),
                })

        combined = pd.concat(
            [
                sdss_strong[[
                    "survey", "spectrum_id", "ra", "dec", "z_qso",
                    "verdict", "transmission", "transmission_expected",
                ]],
                desi_strong.loc[keep_desi, [
                    "survey", "spectrum_id", "ra", "dec", "z_qso",
                    "verdict", "transmission", "transmission_expected",
                ]],
            ],
            ignore_index=True,
        )

        n = len(combined)
        n_candidate = int((combined["verdict"] == "forest_candidate").sum())
        if n_candidate != 0:
            raise RuntimeError(
                "Strong-forest population contains a proximity candidate; "
                "the zero-event upper limit is no longer applicable."
            )
        upper_95 = 1.0 - 0.05 ** (1.0 / n)
        return combined, {
            "selection": (
                "redshift-confirmed spectra with expected mean forest "
                "transmission below 0.5"
            ),
            "n_sdss_unique": int(len(sdss_strong)),
            "n_desi_projection_rows": int(len(desi[
                (desi["verdict"] == "measured")
                & (desi["transmission_expected"] < 0.5)
            ])),
            "n_desi_unique": int(len(desi_strong)),
            "n_cross_survey_duplicates": int(len(cross_matches)),
            "cross_survey_duplicates": cross_matches,
            "n_unique_sightlines": int(n),
            "n_proximity_candidates": n_candidate,
            "empty_forest_fraction_upper_95_one_sided": float(upper_95),
            "interval_method": "exact binomial zero-event bound: 1-alpha^(1/n)",
        }

    def run(self):
        print_status("Building falsification adjudication ledger...", "PROCESS")
        pairs = self._pair_ledger()
        population, population_summary = self._strong_population()

        csv_path = self.data / "falsification_adjudication.csv"
        pairs.to_csv(csv_path, index=False)

        counts = pairs["adjudication"].value_counts().to_dict()
        out = {
            "step": "step_48_falsification_adjudication",
            "catalogue_pair_ledger": {
                "n_pairs": int(len(pairs)),
                "n_decisively_tested": int(counts.get("falsified", 0)),
                "n_falsified": int(counts.get("falsified", 0)),
                "n_confirmed_codistant": 0,
                "n_archival_non_discriminating": int(
                    counts.get("archival_non_discriminating", 0)
                ),
                "n_without_decisive_path_data": int(
                    counts.get("no_decisive_path_data", 0)
                ),
                "tested_result": "1 falsified in 1 decisive catalogue-pair test",
                "inference_boundary": (
                    "The 1/1 result is the observed adjudication count, not a "
                    "false-positive-rate estimate for the heterogeneous untested "
                    "catalogue. The remaining pairs are unadjudicated, not survivors."
                ),
                "pairs": pairs.to_dict("records"),
            },
            "population_control": population_summary,
            "logical_separation": {
                "configuration_severity": (
                    "The chance-alignment product measures the rarity of the "
                    "catalogued angular and morphology-conditioned configuration."
                ),
                "distance_adjudication": (
                    "Only a path-length or independent-distance measurement "
                    "adjudicates co-distance for an individual pair."
                ),
                "population_prior": (
                    "The random-projection forest control bounds the prevalence "
                    "of empty-forest sightlines in the surveyed population; it "
                    "does not adjudicate an unobserved curated companion."
                ),
            },
            "strong_population_rows": population.to_dict("records"),
            "csv": str(csv_path),
        }
        out_path = self.results / "step_48_falsification_adjudication.json"
        out_path.write_text(json.dumps(json_safe(out), indent=2))
        print_status(
            f"Adjudication complete: {out['catalogue_pair_ledger']['tested_result']}; "
            f"{population_summary['n_unique_sightlines']} unique strong controls, "
            f"95% upper bound {population_summary['empty_forest_fraction_upper_95_one_sided']:.3f}.",
            "SUCCESS",
        )
        return out


if __name__ == "__main__":
    Step48FalsificationAdjudication().run()
