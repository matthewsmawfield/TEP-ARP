#!/usr/bin/env python3
"""
Step 13: Proper-Time Budget
===========================
Translates the intrinsic conformal factors into the physical
proper-time budget of each pair: the fractional clock-rate
difference between the companion and its host, the corresponding
scalar-field offset in Planck units, and the photon-flight
conditions implied by the disformal map.

Under A(phi) = exp(beta_A phi) with beta_A = -1, every matter
clock in the companion runs at the rate A_int relative to the
host's clocks.  The step also computes the observable
equivalents: the velocity that a standard Doppler interpretation
would assign to the intrinsic component,

    v_intrinsic = c * z_intrinsic,

and the fictitious lookback-time gap the Hubble interpretation
inserts between the two objects.

Outputs:
    results/outputs/step_13_proper_time_budget.json
    data/processed/proper_time_budget.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.cosmology import FlatLambdaCDM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.jsonio import json_safe
from core.constants import C_LIGHT

COSMO = FlatLambdaCDM(H0=70.0, Om0=0.3)


class Step13ProperTimeBudget:
    """Step 13: Compute the proper-time budget for each pair."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_13",
            log_file_path=self.logs / "step_13_proper_time_budget.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status("Computing proper-time budgets...", "PROCESS")

        cf_path = self.data_processed / "intrinsic_conformal_factors.csv"
        if not cf_path.exists():
            raise FileNotFoundError(
                f"Conformal factor table not found: {cf_path}. Run step_10 first."
            )
        cf = pd.read_csv(cf_path)

        c_kms = C_LIGHT / 1000.0
        rows = []
        for _, row in cf.iterrows():
            a_int = float(row["a_int"])
            delta_phi = float(row["delta_phi"])
            z_int = float(row["z_intrinsic"])
            z_g, z_q = float(row["z_gal"]), float(row["z_comp"])

            # Fictitious Doppler velocity of the intrinsic component
            v_int_kms = c_kms * z_int

            # Fictitious gap inserted by the standard cosmological interpretation
            # between galaxy and companion, computed exactly using LambdaCDM
            # (H0=70, Om0=0.3) rather than the linear Hubble approximation.
            if z_q > 0 and z_g > 0:
                d_c_q = COSMO.comoving_distance(z_q).value
                d_c_g = COSMO.comoving_distance(z_g).value
                d_gap_mpc = d_c_q - d_c_g
                
                t_l_q = COSMO.lookback_time(z_q).value
                t_l_g = COSMO.lookback_time(z_g).value
                t_gap_gyr = t_l_q - t_l_g
                
                inflation_factor = d_c_q / d_c_g
            else:
                d_gap_mpc = 0.0
                t_gap_gyr = 0.0
                inflation_factor = 1.0

            # Proper-time lag of companion clocks over one Gyr of
            # host coordinate time
            lag_gyr_per_gyr = 1.0 - a_int

            rows.append({
                "pair_id": row["pair_id"],
                "a_int": a_int,
                "delta_phi": delta_phi,
                "clock_rate_ratio": a_int,
                "clock_lag_per_gyr": lag_gyr_per_gyr,
                "v_intrinsic_doppler_kms": v_int_kms,
                "fictitious_distance_gap_mpc": d_gap_mpc,
                "fictitious_lookback_gap_gyr": t_gap_gyr,
                "distance_inflation_factor": inflation_factor,
            })

            print_status(
                f"{row['pair_id']}: clocks at {a_int:.3f}x host rate; "
                f"fictitious gap {d_gap_mpc:.0f} Mpc / {t_gap_gyr:.2f} Gyr",
                "TEST",
            )

        df = pd.DataFrame(rows)
        csv_path = self.data_processed / "proper_time_budget.csv"
        df.to_csv(csv_path, index=False)
        print_status(f"Saved proper-time budget: {csv_path}", "SUCCESS")

        summary = {
            "n_pairs": len(df),
            "clock_rate_ratio_range": [
                float(df["clock_rate_ratio"].min()),
                float(df["clock_rate_ratio"].max()),
            ],
            "max_fictitious_gap_mpc": float(df["fictitious_distance_gap_mpc"].max()),
            "pairs": df.to_dict(orient="records"),
            "interpretation": (
                "A_int is the relative clock rate of companion matter "
                "versus host matter. The 'fictitious' columns quantify the "
                "distance and lookback-time gaps that the standard "
                "interpretation inserts between the two members."
            ),
        }
        json_path = self.results / "step_13_proper_time_budget.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")
        print_status("Proper-time budgets computed.", "SUCCESS")
