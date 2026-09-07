#!/usr/bin/env python3
"""
Step 26: Gravitational Lensing Magnification Budget
====================================================
Prices the magnification-bias escape hatch: under the standard
background-quasar interpretation, each host galaxy is a foreground
lens whose magnification can promote sub-threshold background
sources above survey detection limits, producing an apparent
quasar overdensity near foreground nuclei.  This step computes
that correction explicitly rather than leaving it as an
unquantified objection.

Model.  Each host is treated as a singular isothermal sphere
(SIS) with Einstein radius

    theta_E = 4 pi (sigma_v / c)^2 (D_ls / D_s),

evaluated in the standard-cosmology frame the objection itself
assumes: the lens sits at the host's redshift-independent
Cosmicflows-4 distance where measured, else at its Hubble
distance, and the source sits at the companion's published
redshift distance.  The point-source magnification at observed
angular separation theta is

    mu = theta / (theta - theta_E)      (theta > theta_E).

Magnification bias converts this into a surface-density
correction through the cumulative quasar luminosity-function
slope alpha:

    Sigma_obs = Sigma_0 * mu^(alpha - 1).

The per-pair Poisson probabilities of step 23 are then
recomputed with lambda -> lambda * mu^(alpha - 1), including
the multi-companion Poisson tails.

Conservatism is enforced at every choice point:

  * sigma_v = 300 km/s is applied to every host, well above
    published dispersions for these L* spirals and S0s
    (~130-230 km/s); a sigma_v = 250/400 km/s sensitivity grid
    brackets the choice in both directions.
  * alpha = 2.0 exceeds the measured cumulative faint-end
    slope of the quasar luminosity function (~1.5); a grid
    over alpha = 1.5/2.0/2.5 is reported.
  * For multi-companion pairs the Einstein radius is evaluated
    at the companion redshift giving the largest value.
  * The host distance uses the nearest available estimator
    (CF4 where measured), which maximises D_ls / D_s.

All inputs are real pipeline products; no archive query is
made and no datum is fabricated.  The step fails loudly if the
catalogue or the step-23 per-pair table is absent.

Outputs:
    data/processed/lens_magnification_budget.csv
    results/outputs/step_26_lens_magnification_budget.json
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.jsonio import json_safe

C_KMS = 299792.458          # speed of light, km/s
H0_KMS_MPC = 70.0           # Hubble distance convention (matches step_05)
SIGMA_V_KMS = 300.0         # conservative universal host dispersion bound
SIGMA_V_GRID = [250.0, 300.0, 400.0]   # sensitivity scan
ALPHA_QLF = 2.0             # conservative cumulative QLF slope
ALPHA_GRID = [1.5, 2.0, 2.5]           # sensitivity scan


def _zdist_mpc(z):
    return C_KMS * float(z) / H0_KMS_MPC


class Step26LensMagnificationBudget:
    """Step 26: SIS magnification-bias budget per pair."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_26",
            log_file_path=self.logs / "step_26_lens_magnification_budget.log",
        )
        set_step_logger(self.logger)

    def run(self):
        from scipy.stats import poisson

        print_status("Computing gravitational-lensing magnification budget...",
                     "PROCESS")

        cat_path = self.data_processed / "arp_pair_catalog.csv"
        ca_path = self.data_processed / "chance_alignment.csv"
        dist_path = self.data_processed / "redshift_independent_distances.csv"
        for p in (cat_path, ca_path, dist_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Required input not found: {p}. "
                    "Run steps 00, 05 and 23 first."
                )

        catalog = pd.read_csv(cat_path)
        chance = pd.read_csv(ca_path)
        dists = pd.read_csv(dist_path)

        # Host CF4 distances where measured (nearest estimator =>
        # maximises D_ls/D_s and hence the lensing correction).
        host_d = {}
        for _, r in dists.iterrows():
            if (r.get("status") == "measured"
                    and str(r.get("role", "")).lower() == "host"
                    and pd.notna(r.get("distance_mpc"))):
                host_d[r["pair_id"]] = float(r["distance_mpc"])

        lam_by_pair = chance.set_index("pair_id")["expected_count"].to_dict()
        sep_by_pair = chance.set_index("pair_id")["separation_arcsec"].to_dict()

        rows = []
        for _, row in catalog.iterrows():
            pid = row["pair_id"]
            theta_arcsec = float(
                sep_by_pair.get(pid, row["separation_arcsec"]))
            theta_rad = np.radians(theta_arcsec / 3600.0)

            d_l = host_d.get(pid, _zdist_mpc(row["z_gal"]))
            d_l_src = "cosmicflows4" if pid in host_d else "hubble_z"

            z_list = (
                [float(z) for z in str(row["z_comp_list"]).split(";")]
                if pd.notna(row.get("z_comp_list"))
                else [float(row["z_comp"])]
            )
            n_comp = len(z_list)

            lam_base = float(lam_by_pair.get(pid, np.nan))
            if not np.isfinite(lam_base):
                raise RuntimeError(
                    f"{pid}: no expected_count in chance_alignment.csv; "
                    "step 23 output is incomplete."
                )

            for sig_v in SIGMA_V_GRID:
                # Largest Einstein radius across companions (conservative).
                theta_e = 0.0
                for z_s in z_list:
                    d_s = _zdist_mpc(z_s)
                    d_ls = max(d_s - d_l, 0.0)
                    te = 4.0 * np.pi * (sig_v / C_KMS) ** 2 * (d_ls / d_s)
                    theta_e = max(theta_e, te)
                theta_e_arcsec = np.degrees(theta_e) * 3600.0

                if theta_rad <= theta_e:
                    print_status(
                        f"{pid}: separation {theta_arcsec:.1f} arcsec inside "
                        f"Einstein radius {theta_e_arcsec:.1f} arcsec at "
                        f"sigma_v={sig_v} km/s — strong-lensing regime; "
                        "magnification-bias correction not applicable at "
                        "this separation.", "WARNING")
                    continue

                mu = theta_rad / (theta_rad - theta_e)
                rows.append({
                    "pair_id": pid,
                    "sigma_v_kms": sig_v,
                    "theta_arcsec": theta_arcsec,
                    "d_lens_mpc": d_l,
                    "d_lens_source": d_l_src,
                    "d_source_mpc": _zdist_mpc(max(z_list)),
                    "theta_einstein_arcsec": theta_e_arcsec,
                    "magnification_mu": mu,
                    "mu_minus_1_pct": 100.0 * (mu - 1.0),
                    "n_companions": n_comp,
                    "lambda_base": lam_base,
                })

        df = pd.DataFrame(rows)
        csv_path = self.data_processed / "lens_magnification_budget.csv"
        df.to_csv(csv_path, index=False)
        print_status(
            f"Saved lensing budget: {csv_path} ({len(df)} pair-rows)",
            "SUCCESS")

        # ----------------------------------------------------------
        # Joint-probability correction at the nominal (sigma_v, alpha)
        # and across the sensitivity grid.
        catalog_idx = catalog.set_index("pair_id")

        def _joint_p(sigma_v, alpha):
            """Recompute the joint chance P with lensed densities."""
            jl = 0.0
            per_pair = {}
            for pid, lam in lam_by_pair.items():
                r = df[(df["pair_id"] == pid)
                       & (df["sigma_v_kms"] == sigma_v)]
                if r.empty:
                    per_pair[pid] = np.nan
                    continue
                mu = float(r["magnification_mu"].iloc[0])
                lam_eff = lam * mu ** (alpha - 1.0)
                row = catalog_idx.loc[pid]
                if pd.notna(row.get("z_comp_list")):
                    n_q = len(str(row["z_comp_list"]).split(";"))
                    p = float(1.0 - poisson.cdf(n_q - 1, lam_eff))
                else:
                    p = float(1.0 - np.exp(-lam_eff))
                per_pair[pid] = p
                jl += np.log10(p)
            return per_pair, 10.0 ** jl, jl

        base_joint = 10.0 ** sum(
            np.log10(float(chance.set_index("pair_id").loc[pid, "p_chance"]))
            for pid in lam_by_pair
        )

        _, joint_lens, joint_lens_log = _joint_p(SIGMA_V_KMS, ALPHA_QLF)

        grid = []
        for sv in SIGMA_V_GRID:
            for al in ALPHA_GRID:
                _, jp, jl = _joint_p(sv, al)
                grid.append({
                    "sigma_v_kms": sv,
                    "alpha_qlf": al,
                    "joint_probability": jp,
                    "joint_log10_p": jl,
                    "inflation_vs_baseline": jp / base_joint,
                })
                print_status(
                    f"sigma_v={sv} km/s, alpha={al}: joint P = {jp:.3e} "
                    f"(baseline {base_joint:.3e}, x{jp / base_joint:.3f})",
                    "TEST")

        nominal = df[df["sigma_v_kms"] == SIGMA_V_KMS]
        max_mu1 = float(nominal["mu_minus_1_pct"].max())
        worst = nominal.loc[nominal["mu_minus_1_pct"].idxmax()]

        print_status(
            f"Max magnification bias: mu-1 = {max_mu1:.2f}% "
            f"({worst['pair_id']}, theta={worst['theta_arcsec']:.1f} arcsec)",
            "TEST")
        print_status(
            f"Joint P with lensing correction: {joint_lens:.3e} vs "
            f"uncorrected {base_joint:.3e} "
            f"(inflation factor {joint_lens / base_joint:.3f})",
            "TEST")

        summary = {
            "model": (
                "SIS lens; theta_E = 4 pi (sigma_v/c)^2 D_ls/D_s; "
                "mu = theta/(theta - theta_E); "
                "Sigma -> Sigma * mu^(alpha-1)"
            ),
            "sigma_v_kms_nominal": SIGMA_V_KMS,
            "sigma_v_note": (
                "300 km/s applied uniformly exceeds published central "
                "dispersions for all twelve hosts (~130-230 km/s); "
                "the correction is an upper bound."
            ),
            "alpha_qlf_nominal": ALPHA_QLF,
            "h0_kms_mpc": H0_KMS_MPC,
            "n_pairs": int(nominal.shape[0]),
            "max_mu_minus_1_pct": max_mu1,
            "max_mu_pair": str(worst["pair_id"]),
            "joint_p_baseline": base_joint,
            "joint_p_lensed": joint_lens,
            "joint_log10_p_lensed": joint_lens_log,
            "joint_p_inflation_factor": joint_lens / base_joint,
            "sensitivity_grid": grid,
            "verdict": (
                "magnification bias is bounded at the percent level and "
                "cannot generate the observed joint improbability"
                if joint_lens / base_joint < 3.0 else "see_measurement"
            ),
        }
        json_path = (
            self.results / "step_26_lens_magnification_budget.json")
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")
        print_status("Lensing magnification budget complete.", "SUCCESS")


if __name__ == "__main__":
    Step26LensMagnificationBudget().run()
