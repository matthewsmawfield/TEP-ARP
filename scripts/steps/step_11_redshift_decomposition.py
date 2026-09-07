#!/usr/bin/env python3
"""
Step 11: Redshift Decomposition
===============================
Decomposes each companion's observed redshift into the shared
background temporal shear and the intrinsic local shear, then
quantifies the distance misinterpretation that follows from
assigning the full redshift to the Hubble expansion.

Under the standard interpretation the comoving distance is a
monotone function of z, so a companion at z_Q sitting at
separation theta from a galaxy at z_G is placed a factor
~ z_Q / z_G farther away.  Under TEP both objects share the
same coordinate distance r; the observed redshift is

    1 + z_obs = A_0 / (A_bg(r) * A_int)

with A_bg(r) fixed by the galaxy's redshift
(A_bg = A_0 / (1 + z_G) once the pair distance is set) and
A_int carrying the entire discordance.

The step computes, for every pair:
  * the Hubble-law distances D_H(G) = c z_G / H0 and D_H(Q)
    implied by each redshift separately (h = 0.70);
  * the standard-LambdaCDM comoving distances D_C(G) and D_C(Q)
    (flat cosmology, H0 = 70 km/s/Mpc, Om0 = 0.3);
  * the distance inflation factor D_C(Q)/D_C(G) (the comoving
    ratio is the primary quantity; the linear-Hubble ratio is
    retained for reference);
  * the comoving-distance gap and lookback-time gap the standard
    reading assigns between the two members of each pair;
  * the implied background-shear rate contribution of the pair's
    actual distance when the intrinsic component is removed.

Outputs:
    results/outputs/step_11_redshift_decomposition.json
    data/processed/redshift_decomposition.csv
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

H0_KM_S_MPC = 70.0  # reference Hubble constant for distance illustration
# Standard flat-LambdaCDM cosmology used to price the distances the
# conventional redshift-distance reading assigns to each member.
LCDM = FlatLambdaCDM(H0=H0_KM_S_MPC, Om0=0.3)


class Step11RedshiftDecomposition:
    """Step 11: Decompose redshift into background and intrinsic shear."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_11",
            log_file_path=self.logs / "step_11_redshift_decomposition.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status("Decomposing redshifts into background and intrinsic shear...", "PROCESS")

        cf_path = self.data_processed / "intrinsic_conformal_factors.csv"
        if not cf_path.exists():
            raise FileNotFoundError(
                f"Conformal factor table not found: {cf_path}. Run step_10 first."
            )
        cf = pd.read_csv(cf_path)

        c_kms = C_LIGHT / 1000.0  # km/s
        rows = []
        for _, row in cf.iterrows():
            z_g, z_q = row["z_gal"], row["z_comp"]

            # Hubble-law distances each redshift implies separately
            d_h_g = c_kms * z_g / H0_KM_S_MPC
            d_h_q = c_kms * z_q / H0_KM_S_MPC
            inflation_hubble = d_h_q / d_h_g

            # Standard-LambdaCDM comoving distances and the gap the
            # conventional reading assigns between the two members
            d_c_g = float(LCDM.comoving_distance(z_g).to("Mpc").value)
            d_c_q = float(LCDM.comoving_distance(z_q).to("Mpc").value)
            inflation = d_c_q / d_c_g
            lookback_gap = float(
                (LCDM.lookback_time(z_q) - LCDM.lookback_time(z_g))
                .to("Gyr").value
            )

            # Under TEP the pair sits at the galaxy distance; the
            # background shear at that distance reproduces z_G, and
            # the remainder of z_Q is entirely intrinsic.
            # A_bg = 1/(1+z_G) in units where A_0 = 1.
            a_bg = 1.0 / (1.0 + z_g)
            a_int = row["a_int"]

            # Physical projected separation at the galaxy's
            # angular-diameter distance
            theta_rad = np.deg2rad(float(row["separation_arcsec"]) / 3600.0)
            d_a_g = float(LCDM.angular_diameter_distance(z_g).to("Mpc").value)
            r_proj_kpc = theta_rad * d_a_g * 1000.0

            rows.append({
                "pair_id": row["pair_id"],
                "d_hubble_gal_mpc": d_h_g,
                "d_hubble_comp_mpc": d_h_q,
                "d_comoving_gal_mpc": d_c_g,
                "d_comoving_comp_mpc": d_c_q,
                "distance_inflation_factor": inflation,
                "hubble_inflation_factor": inflation_hubble,
                "comoving_gap_mpc": d_c_q - d_c_g,
                "lookback_gap_gyr": lookback_gap,
                "a_bg_at_pair": a_bg,
                "a_int": a_int,
                "projected_sep_kpc": r_proj_kpc,
            })

            print_status(
                f"{row['pair_id']}: D_C(G)={d_c_g:.1f} Mpc, "
                f"D_C(Q)={d_c_q:.1f} Mpc, inflation={inflation:.1f}x, "
                f"r_proj={r_proj_kpc:.1f} kpc",
                "TEST",
            )

        df = pd.DataFrame(rows)
        csv_path = self.data_processed / "redshift_decomposition.csv"
        df.to_csv(csv_path, index=False)
        print_status(f"Saved decomposition table: {csv_path}", "SUCCESS")

        summary = {
            "h0_km_s_mpc": H0_KM_S_MPC,
            "cosmology": "FlatLambdaCDM(H0=70, Om0=0.3)",
            "n_pairs": len(df),
            "median_distance_inflation": float(df["distance_inflation_factor"].median()),
            "max_distance_inflation": float(df["distance_inflation_factor"].max()),
            "max_comoving_gap_mpc": float(df["comoving_gap_mpc"].max()),
            "max_lookback_gap_gyr": float(df["lookback_gap_gyr"].max()),
            "projected_separations_kpc": df[
                ["pair_id", "projected_sep_kpc"]
            ].to_dict(orient="records"),
            "pairs": df.to_dict(orient="records"),
            "interpretation": (
                "Under the standard LambdaCDM redshift-distance reading "
                "each companion is placed distance_inflation_factor "
                "times farther than its host (comoving-distance ratio). "
                "Under TEP the pair shares the galaxy distance; the excess "
                "redshift is carried by the intrinsic conformal factor."
            ),
        }

        json_path = self.results / "step_11_redshift_decomposition.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")
        print_status("Redshift decomposition complete.", "SUCCESS")
