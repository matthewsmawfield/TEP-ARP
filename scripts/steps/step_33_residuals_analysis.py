#!/usr/bin/env python3
"""
Step 33: Residuals Analysis
===========================
Performs the residual diagnostics for the transect profile fits
of step_30, following the project's requirement that model
comparison is supported by residual analysis rather than
parameter-count arguments alone.

For each fitted profile family the step computes, on the
NGC 7603 four-point transect:
  * weighted residuals r_i = (A_obs - A_model) / sigma_i
  * chi-squared and reduced chi-squared
  * residual RMS and lag-1 autocorrelation (structure test)
  * the Lilliefors normality check on the weighted residuals
    (small-sample caveat noted)

A family whose weighted residuals are consistent with white
noise at the error level is an adequate description of the
measured field transition; systematic structure would indicate
the profile family is incomplete.

Outputs:
    results/outputs/step_33_residuals_analysis.json
    data/processed/transect_residuals.csv
    results/figures/step_33_residuals_analysis.png
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

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


class Step33ResidualsAnalysis:
    """Step 33: Residual diagnostics of the transect fits."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_33",
            log_file_path=self.logs / "step_33_residuals_analysis.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status("Running residuals analysis...", "PROCESS")

        fit_path = self.results / "step_30_bridge_redshift_transect.json"
        transect_path = self.data_processed / "transect_data.csv"
        cf_path = self.data_processed / "intrinsic_conformal_factors.csv"
        for p in (fit_path, transect_path, cf_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Required input not found: {p}. Run step_30 first."
                )

        with open(fit_path) as f:
            s30 = json.load(f)
        transect = pd.read_csv(transect_path)
        cf = pd.read_csv(cf_path)

        sub = transect[transect["pair_id"] == "NGC7603-NGC7603B"].sort_values("x")
        x = sub["x"].to_numpy()
        a = sub["a_int"].to_numpy()
        s = sub["sigma_a"].to_numpy()
        a_q = float(cf.loc[cf["pair_id"] == "NGC7603-NGC7603B", "a_int"].iloc[0])

        well_centers = load_ngc7603_well_centers(self.data_processed)
        fns = {
            "exponential": lambda xx, p: profile_exponential(xx, a_q, *p),
            "tanh": lambda xx, p: profile_tanh(xx, a_q, *p),
            "yukawa": lambda xx, p: profile_yukawa(xx, a_q, *p),
            "nested_wells": lambda xx, p: profile_nested_wells(
                xx, *p, centers=well_centers
            ),
        }

        records = []
        resid_rows = []
        for name, res in s30["fits"].items():
            if "params" not in res or name not in fns:
                continue
            pred = fns[name](x, res["params"])
            resid = (a - pred) / s
            rms = float(np.sqrt(np.mean(resid**2)))
            chi2 = float(np.sum(resid**2))
            dof = max(len(x) - len(res["params"]), 1)
            ac1 = (
                float(np.corrcoef(resid[:-1], resid[1:])[0, 1])
                if len(resid) > 2
                else np.nan
            )
            # When the fit interpolates the points exactly (residuals
            # at the numerical floor), a lag-1 autocorrelation of the
            # residual rounding noise is not a meaningful diagnostic.
            ac1_report = ac1 if np.isfinite(ac1) and rms > 0.05 else None
            # Lilliefors normality check on the weighted residuals
            # (Lilliefors 1967): KS statistic against a normal law with
            # estimated mean and variance.  With n = 4 transect points
            # the test is reported for completeness but is
            # underpowered; at the interpolating family's numerical
            # floor it is vacuous and flagged as such.
            try:
                from statsmodels.stats.diagnostic import lilliefors
                d_stat, lillie_p = lilliefors(resid)
                lillie = {
                    "D": float(d_stat),
                    "p_value": float(lillie_p),
                    "n": int(len(resid)),
                    "informative": bool(len(resid) >= 8 and rms > 0.05),
                    "note": (
                        "n=4: underpowered, reported for completeness"
                        + ("; residuals at interpolating floor"
                           if rms <= 0.05 else "")
                    ),
                }
            except ImportError:
                lillie = None
            records.append({
                "family": name,
                "chi2": chi2,
                "dof": int(dof),
                "chi2_dof": chi2 / dof,
                "resid_rms_sigma": rms,
                "resid_lag1_autocorr": ac1_report,
                "lilliefors": lillie,
                "params": res["params"],
            })
            for xi, ri, oi, pi in zip(x, resid, a, pred):
                resid_rows.append({
                    "family": name, "x": float(xi), "resid_sigma": float(ri),
                    "a_obs": float(oi), "a_pred": float(pi),
                })
            lillie_str = (
                f", Lilliefors p={lillie['p_value']:.3f}"
                if lillie is not None else ""
            )
            print_status(
                f"{name}: chi2/dof={chi2/dof:.2f}, resid RMS={rms:.2f} sigma, "
                f"lag-1={ac1_report if ac1_report is not None else 'n/a'}"
                f"{lillie_str}", "TEST",
            )

        dfr = pd.DataFrame(records)
        dfp = pd.DataFrame(resid_rows)
        dfp.to_csv(self.data_processed / "transect_residuals.csv", index=False)

        best = dfr.loc[dfr["chi2_dof"].idxmin(), "family"] if len(dfr) else None
        summary = {
            "n_points": int(len(x)),
            "families": records,
            "best_family_by_chi2": best,
            "interpretation": (
                "Weighted residuals consistent with unit RMS and no "
                "lag-1 autocorrelation indicate the profile family is an "
                "adequate description of the measured field transition."
            ),
        }
        json_path = self.results / "step_33_residuals_analysis.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")

        # Figure
        try:
            import matplotlib.pyplot as plt
            apply_tep_style()
            fig, ax = plt.subplots(figsize=(9, 5))
            markers = {"exponential": "o", "tanh": "s", "yukawa": "^", "nested_wells": "D"}
            for name in dfr["family"]:
                sub_r = dfp[dfp["family"] == name]
                ax.plot(
                    sub_r["x"], sub_r["resid_sigma"],
                    marker=markers.get(name, "o"), ls="--", label=name,
                )
            ax.axhline(0, color="gray", lw=1)
            ax.axhline(1, color="gray", lw=0.5, ls=":")
            ax.axhline(-1, color="gray", lw=0.5, ls=":")
            ax.set_xlabel("Normalised bridge coordinate x")
            ax.set_ylabel(r"Weighted residual $(\sigma)$")
            ax.set_title("NGC 7603 transect fit residuals")
            ax.legend(fontsize=9)
            fig.tight_layout()
            fig.savefig(self.figures / "step_33_residuals_analysis.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print_status("Saved figure: step_33_residuals_analysis.png", "SUCCESS")
        except Exception as e:
            self.logger.warning(f"figure failed: {e}")

        print_status("Residuals analysis complete.", "SUCCESS")
