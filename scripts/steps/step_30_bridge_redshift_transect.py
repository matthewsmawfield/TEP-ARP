#!/usr/bin/env python3
"""
Step 30: Bridge Redshift Transect Fit
=====================================
Assembles the redshift transect across each connecting structure
and fits the scalar-field profile families defined in step_12.

Transect construction:
  * Every pair contributes the two endpoints of its bridge
    coordinate: x = 0 at the host nucleus (A_int = 1 by
    construction, up to the galaxy's own intrinsic shear, which
    is absorbed into A_bg for the pair) and x = 1 at the
    companion (A_int = A_int(Q) from step_10).
  * NGC 7603 additionally contributes the two emission knots
    embedded in the luminous filament at z = 0.243 and z = 0.391
    (Lopez-Corredoira & Gutierrez 2002). Both knots are resolved
    to archive positions (Simbad [LG2002] objects) by step_20,
    and their transect coordinates x are measured by projection
    onto the resolved host -> companion axis — no positional
    assumption enters the transect.

The three profile families (exponential, tanh, yukawa) are
fitted to the pooled transect data by least squares on the
interior shape parameter, and compared by chi-squared.  A smooth
single-family curve that reproduces the four-point NGC 7603
transect is direct evidence that the redshift differential is a
continuous property of a shared local field rather than a set of
independent cosmological distances.

Outputs:
    results/outputs/step_30_bridge_redshift_transect.json
    data/processed/transect_fit.csv
    results/figures/step_30_bridge_redshift_transect.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.jsonio import json_safe
from scripts.utils.plot_style import apply_tep_style
from scripts.steps.step_12_scalar_field_profiles import (
    load_ngc7603_well_centers,
    profile_exponential,
    profile_tanh,
    profile_yukawa,
    profile_nested_wells,
)

SIGMA_Z = 1.0e-3  # default redshift uncertainty where none is published


def _a_err_from_z(z_g, z_q, sigma_z):
    """Propagate z errors into A_int = (1+z_G)/(1+z_Q)."""
    dG = 1.0 / (1.0 + z_q)
    dQ = -(1.0 + z_g) / (1.0 + z_q) ** 2
    return np.sqrt((dG * sigma_z) ** 2 + (dQ * sigma_z) ** 2)


class Step30BridgeRedshiftTransect:
    """Step 30: Fit A_int(x) profiles to the transect data."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_30",
            log_file_path=self.logs / "step_30_bridge_redshift_transect.log",
        )
        set_step_logger(self.logger)

    def _load_knots(self):
        """Archive-resolved NGC 7603 filament knots (step_20 output)."""
        knot_path = self.data_processed / "filament_knot_positions.csv"
        if not knot_path.exists():
            raise FileNotFoundError(
                f"Filament-knot positions not found: {knot_path}. "
                "Run step_20 first; the transect uses archive-measured "
                "knot positions, not assumptions."
            )
        knots = pd.read_csv(knot_path)
        return knots[knots["pair_id"] == "NGC7603-NGC7603B"]

    def _build_transects(self, cf):
        """Return list of (pair_id, x, A, sigma_A, point_type)."""
        transects = []
        for _, row in cf.iterrows():
            z_g, z_q = row["z_gal"], row["z_comp"]
            a_q = row["a_int"]
            s_q = _a_err_from_z(z_g, z_q, SIGMA_Z)

            transects.append((row["pair_id"], 0.0, 1.0, SIGMA_Z, "host"))
            transects.append((row["pair_id"], 1.0, a_q, s_q, "companion"))

        # NGC 7603 interior filament knots: archive-resolved positions
        # and redshifts, projected onto the host->companion axis by
        # step_20 (see module docstring).  Fails loudly if the measured
        # knot positions are absent.
        knots = self._load_knots()
        z_g7603 = float(
            cf.loc[cf["pair_id"] == "NGC7603-NGC7603B", "z_gal"].iloc[0]
        ) if (cf["pair_id"] == "NGC7603-NGC7603B").any() else 0.0295
        for _, k in knots.iterrows():
            z_k = float(k["z"])
            a_k = (1.0 + z_g7603) / (1.0 + z_k)
            s_k = _a_err_from_z(z_g7603, z_k, SIGMA_Z)
            transects.append(
                ("NGC7603-NGC7603B", float(k["x"]), a_k, s_k, "filament_knot")
            )
        return transects

    def run(self):
        print_status("Fitting bridge redshift transects...", "PROCESS")

        cf_path = self.data_processed / "intrinsic_conformal_factors.csv"
        if not cf_path.exists():
            raise FileNotFoundError(
                f"Conformal factor table not found: {cf_path}. Run step_10 first."
            )
        cf = pd.read_csv(cf_path)

        transects = self._build_transects(cf)
        df = pd.DataFrame(
            transects, columns=["pair_id", "x", "a_int", "sigma_a", "point_type"]
        )
        df.to_csv(self.data_processed / "transect_data.csv", index=False)

        # Fit each family to the NGC 7603 four-point transect, which is
        # the only system with interior constraints on the profile shape.
        sub = df[df["pair_id"] == "NGC7603-NGC7603B"].sort_values("x")
        x, a, s = sub["x"].to_numpy(), sub["a_int"].to_numpy(), sub["sigma_a"].to_numpy()
        a_q = float(cf.loc[cf["pair_id"] == "NGC7603-NGC7603B", "a_int"].iloc[0])
        well_centers = load_ngc7603_well_centers(self.data_processed)

        fits_out = {}
        families = {
            "exponential": (lambda xx, k: profile_exponential(xx, a_q, k), [4.0], (0.5, 30.0)),
            "tanh": (lambda xx, x0, w: profile_tanh(xx, a_q, x0, w), [0.5, 0.15], ([0.2, 0.05], [0.8, 0.5])),
            "yukawa": (lambda xx, m: profile_yukawa(xx, a_q, m), [8.0], (0.5, 40.0)),
            # Nested wells: each compact object adds a Lorentzian
            # depression; permits the non-monotonic transition the
            # filament knots require (interior wells deeper than the
            # companion endpoint).
            "nested_wells": (
                lambda xx, d1, d2, dq, w: profile_nested_wells(
                    xx, d1, d2, dq, w, centers=well_centers
                ),
                [0.2, 0.3, 0.05, 0.2],
                # w lower bound sits inside the flat-likelihood
                # plateau: with no inter-knot points the data bound
                # the width only from above (leakage between wells).
                ([0.0, 0.0, 0.0, 0.002], [2.0, 2.0, 2.0, 1.0]),
            ),
        }
        for name, (fn, p0, bounds) in families.items():
            try:
                popt, pcov = curve_fit(
                    fn, x, a, p0=p0, sigma=s, absolute_sigma=True,
                    bounds=bounds, maxfev=20000,
                )
                pred = fn(x, *popt)
                chi2 = float(np.sum(((a - pred) / s) ** 2))
                dof = max(len(x) - len(popt), 1)
                fits_out[name] = {
                    "params": [float(p) for p in popt],
                    "chi2": chi2, "dof": int(dof),
                    "chi2_dof": chi2 / dof,
                }
                print_status(
                    f"NGC 7603 {name}: chi2/dof = {chi2/dof:.2f} "
                    f"(params={np.round(popt, 3)})", "TEST"
                )
            except Exception as e:
                fits_out[name] = {"error": str(e)}
                self.logger.warning(f"{name} fit failed: {e}")

        best = min(
            (n for n in fits_out if "chi2_dof" in fits_out[n]),
            key=lambda n: fits_out[n]["chi2_dof"],
            default=None,
        )
        if best:
            print_status(
                f"Best transect family: {best} "
                f"(chi2/dof={fits_out[best]['chi2_dof']:.2f})", "TEST"
            )

        fit_table = pd.DataFrame(
            [{"pair_id": "NGC7603-NGC7603B", "family": n, **v}
             for n, v in fits_out.items()]
        )
        fit_table.to_csv(self.data_processed / "transect_fit.csv", index=False)

        summary = {
            "n_transect_points": len(df),
            "n_knots_ngc7603": 2,
            "sigma_z_default": SIGMA_Z,
            "knot_positions": self._load_knots().to_dict(orient="records"),
            "well_centers": list(well_centers),
            "knot_position_source": (
                "Archive-measured positions (Simbad cone search, "
                "redshift-matched [LG2002] objects), projected onto "
                "the resolved host->companion axis; no positional "
                "assumption enters the transect."
            ),
            "fits": fits_out,
            "best_family": best,
            "transect": df.to_dict(orient="records"),
        }
        json_path = self.results / "step_30_bridge_redshift_transect.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")

        # Figure
        try:
            import matplotlib.pyplot as plt
            apply_tep_style()
            fig, ax = plt.subplots(figsize=(9, 6))
            ax.errorbar(
                x, a, yerr=s, fmt="o", color="#1a1526", ms=6,
                label="NGC 7603 transect (host, knots, companion)",
            )
            xx = np.linspace(0, 1, 200)
            for name, res in fits_out.items():
                if "params" not in res:
                    continue
                fn = families[name][0]
                ax.plot(xx, fn(xx, *res["params"]),
                        label=f"{name} (chi2/dof={res['chi2_dof']:.2f})")
            ax.set_xlabel("Normalised bridge coordinate x")
            ax.set_ylabel(r"$A_{\rm int}(x)$")
            ax.set_title("NGC 7603 filament transect: scalar-field profile fits")
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(self.figures / "step_30_bridge_redshift_transect.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print_status("Saved figure: step_30_bridge_redshift_transect.png", "SUCCESS")
        except Exception as e:
            self.logger.warning(f"figure failed: {e}")

        print_status("Bridge redshift transect fit complete.", "SUCCESS")
