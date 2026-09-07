#!/usr/bin/env python3
"""
Step 32: Pair-Sample Statistics
===============================
Analyses the statistical structure of the intrinsic conformal
factor across the full discordant-pair sample.

Tests performed:
  * Distribution of A_int: the TEP expectation is a smooth
    distribution of local field depths rather than the discrete
    quantisation claimed by some intrinsic-redshift models.
  * Separation dependence: Arp's empirical claim that the
    discordance declines with projected separation is tested via
    the Spearman correlation of z_intrinsic (equivalently
    1 - A_int) against angular separation in arcseconds.
  * Connection-type comparison: whether absorption-verified pairs
    (where physical proximity is independently demonstrated)
    occupy the same A_int range as morphological-bridge pairs.

Outputs:
    results/outputs/step_32_pair_sample_statistics.json
    data/processed/pair_sample_statistics.csv
    results/figures/step_32_pair_sample_statistics.png
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


class Step32PairSampleStatistics:
    """Step 32: Statistical structure of the A_int distribution."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_32",
            log_file_path=self.logs / "step_32_pair_sample_statistics.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status("Analysing pair-sample statistics...", "PROCESS")

        cf_path = self.data_processed / "intrinsic_conformal_factors.csv"
        if not cf_path.exists():
            raise FileNotFoundError(
                f"Conformal factor table not found: {cf_path}. Run step_10 first."
            )
        cf = pd.read_csv(cf_path)

        a = cf["a_int"].to_numpy()
        sep = cf["separation_arcsec"].to_numpy()
        z_int = cf["z_intrinsic"].to_numpy()

        # Spearman: intrinsic redshift vs angular separation
        rho, p_rho = stats.spearmanr(sep, z_int)
        print_status(
            f"Spearman(z_intrinsic, separation): rho={rho:.2f}, p={p_rho:.3f}",
            "TEST",
        )

        # Distribution shape
        desc = stats.describe(a)
        skew = float(desc.skewness)
        kurt = float(desc.kurtosis)
        print_status(
            f"A_int distribution: mean={a.mean():.3f}, sd={a.std(ddof=1):.3f}, "
            f"skew={skew:.2f}, kurtosis={kurt:.2f}", "TEST",
        )

        # Absorption-verified vs bridge pairs
        cat_path = self.data_processed / "arp_pair_catalog_verified.csv"
        if not cat_path.exists():
            cat_path = self.data_processed / "arp_pair_catalog.csv"
        merged = cf.copy()
        if cat_path.exists():
            merged = cf.merge(
                pd.read_csv(cat_path)[["pair_id", "absorption_system"]],
                on="pair_id", how="left",
            )
        grp = merged.groupby("absorption_system")["a_int"]
        grp_summary = grp.agg(["count", "mean", "std"]).to_dict()
        print_status(f"A_int by absorption flag: {grp_summary}", "TEST")

        out = merged.copy()
        out_path = self.data_processed / "pair_sample_statistics.csv"
        out.to_csv(out_path, index=False)

        summary = {
            "n_pairs": len(cf),
            "a_int_mean": float(a.mean()),
            "a_int_std": float(a.std(ddof=1)),
            "a_int_skew": skew,
            "a_int_kurtosis": kurt,
            "spearman_zint_sep": {"rho": float(rho), "p_value": float(p_rho)},
            "a_int_by_absorption_flag": grp_summary,
            "interpretation": (
                "Arp reported that discordance declines with projected "
                "separation for ejection-aged companions (a negative "
                "Spearman rho between intrinsic redshift and separation); "
                "the test evaluates this scaling on the catalogued sample. "
                "The measured rho is reported with its p-value and the "
                "sample size bounds the power of the test."
            ),
        }
        json_path = self.results / "step_32_pair_sample_statistics.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")

        # Figure
        try:
            import matplotlib.pyplot as plt
            apply_tep_style()
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            axes[0].bar(cf["pair_id"], a, color="#46557c")
            axes[0].axhline(1.0, color="gray", ls="--")
            axes[0].set_ylabel(r"$A_{\rm int}(Q)$")
            axes[0].set_title("Intrinsic conformal factor by pair")
            axes[0].tick_params(axis="x", rotation=45, labelsize=8)
            for lbl in axes[0].get_xticklabels():
                lbl.set_ha("right")

            axes[1].scatter(sep, z_int, s=60, color="#c1913f")
            for _, r in cf.iterrows():
                axes[1].annotate(
                    r["pair_id"],
                    (r["separation_arcsec"], r["z_intrinsic"]),
                    fontsize=7, xytext=(4, 4), textcoords="offset points",
                )
            axes[1].set_xscale("log")
            axes[1].set_xlabel("Angular separation (arcsec)")
            axes[1].set_ylabel(r"$z_{\rm intrinsic}$")
            axes[1].set_title(
                f"Intrinsic redshift vs separation (rho={rho:.2f})"
            )
            fig.tight_layout()
            fig.savefig(self.figures / "step_32_pair_sample_statistics.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print_status("Saved figure: step_32_pair_sample_statistics.png", "SUCCESS")
        except Exception as e:
            self.logger.warning(f"figure failed: {e}")

        print_status("Pair-sample statistics complete.", "SUCCESS")
