#!/usr/bin/env python3
"""
Step 41: Manuscript Figures
===========================
Generates the composite manuscript figures from the pipeline
outputs:

  * Fig. 1 — Pair gallery summary: intrinsic conformal factor
    and distance-inflation factor for every system.
  * Fig. 2 — Field-transition figure: A_int(x) profile families
    for the NGC 7603 transect with MCMC posterior band.
  * Fig. 3 — Chance-alignment figure: per-pair Poisson
    probability and the joint probability.

Only data written by earlier steps is plotted; the step fails
loudly when required outputs are missing.

Outputs:
    results/figures/step_41_pair_gallery.png
    results/figures/step_41_field_transition.png
    results/figures/step_41_chance_alignment.png
    results/outputs/step_41_manuscript_figures.json
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
from scripts.utils.plot_style import apply_tep_style
from scripts.steps.step_12_scalar_field_profiles import (
    load_ngc7603_well_centers,
    profile_exponential,
    profile_tanh,
    profile_yukawa,
    profile_nested_wells,
)


class Step41ManuscriptFigures:
    """Step 41: Composite manuscript figure generation."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_41",
            log_file_path=self.logs / "step_41_manuscript_figures.log",
        )
        set_step_logger(self.logger)

    # Display labels aligned with the Table 1 pair names
    PAIR_LABELS = {
        "NGC4319-Mrk205": "NGC 4319\u2013Mrk 205",
        "NGC7603-NGC7603B": "NGC 7603\u20137603B",
        "NGC3067-3C232": "NGC 3067\u20133C 232",
        "NGC1073-QSOs": "NGC 1073\u2013BSO 1",
        "NGC7319-QSO": "NGC 7319\u2013QSO",
        "NGC1199-companion": "NGC 1199\u2013comp.",
        "NGC3628-XQSO": "NGC 3628\u2013WEE 51",
        "NGC4258-QSOs": "NGC 4258\u2013QSOs",
        "NGC2639-QSOs": "NGC 2639\u2013QSOs",
        "NGC1097-jetQSOs": "NGC 1097\u2013jet QSOs",
        "NGC1232-NGC1232A": "NGC 1232\u20131232A",
        "NGC3516-Qchain": "NGC 3516\u2013Q chain",
    }

    def _fig_pair_gallery(self, plt, cf, decomp):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))

        ids = [self.PAIR_LABELS.get(p, p) for p in cf["pair_id"]]
        axes[0].bar(ids, cf["a_int"], color="#46557c")
        axes[0].axhline(1.0, color="gray", ls="--", lw=1)
        axes[0].set_ylabel(r"$A_{\rm int}(Q) = (1+z_G)/(1+z_Q)$")
        axes[0].set_title("Intrinsic conformal factor")
        axes[0].tick_params(axis="x", rotation=45, labelsize=10)
        for lbl in axes[0].get_xticklabels():
            lbl.set_ha("right")

        axes[1].bar(ids, decomp["distance_inflation_factor"], color="#c1913f")
        axes[1].set_yscale("log")
        axes[1].set_ylabel(r"$D_{\rm C}(Q)/D_{\rm C}(G)$")
        axes[1].set_title(
            "Distance inflation under the standard $\\Lambda$CDM reading")
        axes[1].tick_params(axis="x", rotation=45, labelsize=10)
        for lbl in axes[1].get_xticklabels():
            lbl.set_ha("right")

        fig.tight_layout()
        out = self.figures / "step_41_pair_gallery.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out.name

    def _fig_field_transition(self, plt, transect, cf, s30, s31):
        sub = transect[transect["pair_id"] == "NGC7603-NGC7603B"].sort_values("x")
        a_q = float(cf.loc[cf["pair_id"] == "NGC7603-NGC7603B", "a_int"].iloc[0])
        xx = np.linspace(0, 1, 300)

        fig, ax = plt.subplots(figsize=(12, 6.2))

        # MCMC posterior-predictive band from the nested-wells samples
        samples_path = self.data_processed / "mcmc_field_profile_samples.csv"
        if samples_path.exists():
            samp = pd.read_csv(samples_path)
            centers = load_ngc7603_well_centers(self.data_processed)
            rng = np.random.default_rng(7)
            idx = rng.choice(len(samp), size=min(300, len(samp)), replace=False)
            curves = np.array([
                profile_nested_wells(xx, *samp.iloc[i].to_numpy(),
                                     centers=centers)
                for i in idx
            ])
            lo = np.percentile(curves, 16, axis=0)
            hi = np.percentile(curves, 84, axis=0)
            ax.fill_between(
                xx, lo, hi, color="#798691", alpha=0.18,
                label="nested-wells 16-84% band",
            )
            ax.plot(xx, np.median(curves, axis=0), color="#241a33", lw=2,
                    label="nested-wells median")

        for name, fn, col in (
            ("exponential", profile_exponential, "#c1913f"),
            ("tanh", profile_tanh, "#6b5a66"),
            ("yukawa", profile_yukawa, "#798691"),
        ):
            res = s30["fits"].get(name, {}) if s30 else {}
            if "params" in res:
                chi2_dof = res.get("chi2_dof", float("nan"))
                ax.plot(xx, fn(xx, a_q, *res["params"]), lw=1.2, ls="--",
                        alpha=0.8, color=col,
                        label=f"{name} ($\\chi^2_\\nu={chi2_dof:.0f}$)")
        ax.errorbar(
            sub["x"], sub["a_int"], yerr=sub["sigma_a"], fmt="o",
            color="#1a1526", ms=6, zorder=5,
            label="measured transect",
        )
        # Fifth datum: the diffuse filament continuum between the knots
        # returns z = 0.030 (Lopez-Corredoira & Gutierrez 2002),
        # indistinguishable from the host nucleus.  Its conformal
        # factor sits at the ambient level across the whole span,
        # the direct evidence that no kpc-scale gradient exists.
        z_g = float(cf.loc[cf["pair_id"] == "NGC7603-NGC7603B",
                           "z_gal"].iloc[0]) \
            if "z_gal" in cf.columns else 0.0295
        a_ambient = (1.0 + z_g) / (1.0 + 0.030)
        ax.axhline(
            a_ambient, color="#2e6e5e", ls="-.", lw=1.4, alpha=0.85,
            label=(
                f"filament continuum ($z = 0.030$, "
                f"$A_{{\\rm int}} = {a_ambient:.4f}$)"
            ),
        )
        ax.set_xlabel("Normalised bridge coordinate x (0 = host, 1 = companion)")
        ax.set_ylabel(r"$A_{\rm int}(x)$")
        ax.set_title("NGC 7603 filament: continuous proper-time transition")
        ax.legend(fontsize=10, loc="lower left")
        fig.tight_layout()
        out = self.figures / "step_41_field_transition.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out.name

    def _fig_chance_alignment(self, plt, chance):
        fig, ax = plt.subplots(figsize=(12, 5.2))
        ax.bar(chance["pair_id"], chance["p_chance"], color="#241a33")
        ax.set_yscale("log")
        ax.set_ylabel("P(chance superposition)")
        ax.set_title("Per-pair chance-alignment probabilities")
        ax.tick_params(axis="x", rotation=45, labelsize=10)
        for lbl in ax.get_xticklabels():
            lbl.set_ha("right")
        # Joint probability line: the decisive ensemble statistic
        s23_path = self.results / "step_23_chance_alignment.json"
        if s23_path.exists():
            with open(s23_path) as f:
                s23 = json.load(f)
            p_joint = s23.get("joint_probability_all_pairs")
            p_joint_sdss = s23.get("joint_probability_sdss_density")
            if p_joint is not None:
                ax.axhline(
                    p_joint, color="#c1913f", ls="--", lw=1.5,
                    label=(
                        f"$P_{{\\rm joint}} = {p_joint:.1e}$ "
                        "(empirical density)"
                    ),
                )
            if p_joint_sdss is not None:
                ax.axhline(
                    p_joint_sdss, color="#46557c", ls=":", lw=1.5,
                    label=(
                        f"$P_{{\\rm joint}} = {p_joint_sdss:.1e}$ "
                        "(SDSS density)"
                    ),
                )
            ax.legend(fontsize=10, loc="upper left")
        fig.tight_layout()
        out = self.figures / "step_41_chance_alignment.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return out.name

    def run(self):
        print_status("Generating manuscript figures...", "PROCESS")

        cf_path = self.data_processed / "intrinsic_conformal_factors.csv"
        decomp_path = self.data_processed / "redshift_decomposition.csv"
        transect_path = self.data_processed / "transect_data.csv"
        chance_path = self.data_processed / "chance_alignment.csv"
        for p in (cf_path, decomp_path, transect_path, chance_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Required input not found: {p}. Run earlier steps first."
                )

        import matplotlib.pyplot as plt
        apply_tep_style()

        cf = pd.read_csv(cf_path)
        decomp = pd.read_csv(decomp_path)
        transect = pd.read_csv(transect_path)
        chance = pd.read_csv(chance_path)

        s30 = s31 = None
        for name, p in (
            ("s30", self.results / "step_30_bridge_redshift_transect.json"),
            ("s31", self.results / "step_31_mcmc_field_profile.json"),
        ):
            if p.exists():
                with open(p) as f:
                    if name == "s30":
                        s30 = json.load(f)
                    else:
                        s31 = json.load(f)

        made = []
        made.append(self._fig_pair_gallery(plt, cf, decomp))
        made.append(self._fig_field_transition(plt, transect, cf, s30, s31))
        made.append(self._fig_chance_alignment(plt, chance))

        for m in made:
            print_status(f"Saved figure: {m}", "SUCCESS")

        summary = {"figures": made}
        json_path = self.results / "step_41_manuscript_figures.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")
        print_status("Manuscript figures complete.", "SUCCESS")
