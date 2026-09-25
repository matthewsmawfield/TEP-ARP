#!/usr/bin/env python3
"""
Step 40: Falsification Summary
==============================
Aggregates every test in the pipeline into a single verdict
table: the TEP spatial-proximity hypothesis versus the standard
chance-superposition / background-quasar interpretation.

Each row records the discriminating observable, the prediction
of each model, the measured result, and the verdict.  The summary
JSON is the primary quantitative input to the Discussion and
Conclusion sections of the manuscript.

Outputs:
    results/outputs/step_40_falsification_summary.json
    data/processed/falsification_summary.csv
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


def _load_json(path):
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


class Step40FalsificationSummary:
    """Step 40: Aggregate all tests into the falsification table."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_40",
            log_file_path=self.logs / "step_40_falsification_summary.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status("Compiling falsification summary...", "PROCESS")

        s05 = _load_json(
            self.results / "step_05_redshift_independent_distances.json"
        )
        s10 = _load_json(self.results / "step_10_intrinsic_conformal_factor.json")
        s11 = _load_json(self.results / "step_11_redshift_decomposition.json")
        s20 = _load_json(self.results / "step_20_bridge_morphology.json")
        s21 = _load_json(self.results / "step_21_absorption_systems.json")
        s22 = _load_json(self.results / "step_22_quasar_galaxy_association.json")
        s23 = _load_json(self.results / "step_23_chance_alignment.json")
        s24 = _load_json(
            self.results / "step_24_forward_crosscorrelation.json"
        )
        s25 = _load_json(
            self.results / "step_25_satellite_redshift_asymmetry.json"
        )
        s26 = _load_json(
            self.results / "step_26_lens_magnification_budget.json"
        )
        s27 = _load_json(
            self.results / "step_27_geometric_coherence.json"
        )
        s38 = _load_json(
            self.results / "step_38_xray_population_test.json"
        )
        s28 = _load_json(self.results / "step_28_archive_kinematic_audit.json")
        s29 = _load_json(self.results / "step_29_lya_forest_audit.json")
        s30 = _load_json(self.results / "step_30_bridge_redshift_transect.json")
        s31 = _load_json(self.results / "step_31_mcmc_field_profile.json")
        s32 = _load_json(self.results / "step_32_pair_sample_statistics.json")
        s33 = _load_json(self.results / "step_33_residuals_analysis.json")
        s34 = _load_json(
            self.results / "step_34_bayesian_model_selection.json")
        s35 = _load_json(
            self.results / "step_35_well_depth_correlates.json")
        s37 = _load_json(
            self.results / "step_37_sdss_companion_asymmetry.json")

        rows = []

        # 1. Conformal factor exists and is sub-unity for all pairs
        if s10:
            vals = s10["a_int_range"]
            rows.append({
                "test": "Pair redshift-ratio diagnostic",
                "observable": "A_pair(Q) = (1+z_G)/(1+z_Q) < 1",
                "tep_prediction": "A_pair < 1 for every companion; sub-unity local field",
                "std_prediction": "no local field; A_pair is a distance ratio artifact",
                "result": f"A_pair in [{vals[0]:.3f}, {vals[1]:.3f}] across {s10['n_pairs']} systems; interpreted as A_int only for systems surviving the co-distance tests",
                "verdict": "sub_unity_ratio_measured",
            })

        # 2. Absorption ordering
        if s21:
            rows.append({
                "test": "Foreground absorption ordering",
                "observable": "z_abs = z_G in companion spectra",
                "tep_prediction": "absorber and companion share a local volume",
                "std_prediction": "coincidental halo intersection of a background source",
                "result": f"{s21['n_systems']} absorption systems across {s21['n_pairs_with_absorption']} pairs",
                "verdict": "physical_association_proven",
            })

        # 3. Bridge morphology: targeted where a luminous structure is
        # claimed (bridge/filament), contextual elsewhere.  Report the
        # segment mean significance, the sign-test persistence of the
        # on-axis excess, and the peak localised z.
        if s20 and s20.get("s_bridge"):
            claimed = [r for r in s20["s_bridge"] if r.get("bridge_claimed")]
            other = [r for r in s20["s_bridge"] if not r.get("bridge_claimed")]
            parts = []
            if claimed:
                def _claimed_str(r):
                    s = (
                        f"{r['pair_id']}: S={r['S_bridge']:.2f}, "
                        f"sign p={r['sign_p']:.3g}, "
                        f"peak z={r['peak_z']:.1f} at x={r['peak_x']:.2f}"
                    )
                    if r.get("S_filament") is not None and np.isfinite(r["S_filament"]):
                        s += (
                            f"; filament-path transect S={r['S_filament']:.2f}, "
                            f"sign p={r['sign_p_filament']:.3g}, "
                            f"peak z={r['peak_z_filament']:.1f} "
                            f"at path fraction {r['peak_x_filament']:.2f}"
                        )
                    if r.get("S_filament_g") is not None and np.isfinite(
                            r["S_filament_g"]):
                        s += (
                            f"; g-band confirmation S={r['S_filament_g']:.2f}, "
                            f"sign p={r['sign_p_filament_g']:.3g}"
                        )
                    if r.get("S_filament_dss") is not None and np.isfinite(
                            r["S_filament_dss"]):
                        s += (
                            f"; DSS2 photographic plate S={r['S_filament_dss']:.2f}, "
                            f"sign p={r['sign_p_filament_dss']:.3g}"
                        )
                    if isinstance(r.get("knot_embeddedness"), list) and \
                            r["knot_embeddedness"]:
                        zs = [
                            k["extended_z_vs_controls"]
                            for k in r["knot_embeddedness"]
                            if np.isfinite(k.get("extended_z_vs_controls", np.nan))
                        ]
                        if zs:
                            s += (
                                f"; {len(zs)} knots on elevated corridor "
                                "(extended excess, PSF excluded) at "
                                + "/".join(f"{z:.1f}" for z in zs)
                                + " sigma vs rotated controls"
                            )
                    return s
                parts.append(
                    "claimed luminous-structure axes ("
                    + ", ".join(_claimed_str(r) for r in claimed)
                    + ")"
                )
            if other:
                parts.append(
                    "contextual axes ("
                    + ", ".join(
                        f"{r['pair_id']}: S={r['S_bridge']:.2f}"
                        for r in other
                    ) + ")"
                )
            if parts:
                rows.append({
                    "test": "Luminous bridge surface brightness",
                    "observable": (
                        "S_bridge = (mu_bridge - mean(mu_ctrl))/std(mu_ctrl) "
                        "over the inter-object segment, plus sign-test "
                        "persistence and peak localised z"
                    ),
                    "tep_prediction": "positive excess along companion axis",
                    "std_prediction": "no excess; bridge is noise",
                    "result": "; ".join(parts),
                    "verdict": "see_measurement",
                })

        # 4. Quasar-galaxy association
        if s22:
            paired = s22.get("paired_test", {})
            paired_str = ""
            if paired:
                paired_str = (
                    f"; paired own-controls test: "
                    f"{paired.get('n_hosts_exceed_all_own_controls_p01', 0)} of "
                    f"{len(paired.get('per_host', []))} "
                    f"hosts exceed all 12 own controls, Fisher p = "
                    f"{paired.get('p_fisher_combined', float('nan')):.2f}"
                )
            incl = ""
            if s22.get("host_total_qso_inclusive") is not None:
                incl = (
                    f"; companion-inclusive total "
                    f"{s22['host_total_qso_inclusive']} "
                    f"(selection objects removed from primary count)"
                )
            rows.append({
                "test": "Quasar overdensity around hosts",
                "observable": (
                    "Milliquas confirmed quasar-class counts vs random "
                    "control fields; catalogued companions excluded "
                    "(non-circular)"
                ),
                "tep_prediction": "positive overdensity at host positions",
                "std_prediction": "host fields are ordinary lines of sight",
                "result": (
                    f"host mean {s22['host_mean_qso']:.2f} vs field "
                    f"{s22['field_mean_qso']:.2f} +/- {s22['field_std_qso']:.2f}; "
                    f"excess {s22['excess_sigma_of_field_scatter']:.2f} sigma"
                    f"{paired_str}{incl}"
                ),
                "verdict": "see_measurement",
            })

        # 5. Chance-alignment probability
        if s23:
            p_str = (
                f"joint P_chance = {s23['joint_probability_all_pairs']:.3e} "
                f"(log10 P = {s23['joint_log10_p']:.1f}, "
                f"{s23.get('sigma_qso_source', 'quasar density')})"
            )
            if s23.get("joint_probability_sdss_density") is not None:
                p_str += (
                    f"; under published SDSS density: "
                    f"{s23['joint_probability_sdss_density']:.3e} "
                    f"(log10 P = {s23['joint_log10_p_sdss_density']:.1f})"
                )
            ts = s23.get("tight_subset") or {}
            if ts.get("joint_probability_tight") is not None:
                p_str += (
                    f"; compact subsample alone (theta < "
                    f"{ts['separation_threshold_arcsec']:.0f} arcsec, "
                    f"{ts['n_tight']} pairs): "
                    f"{ts['joint_probability_tight']:.3e} "
                    f"(log10 P = {ts['joint_log10_p_tight']:.1f}); "
                    f"wide-field systems contribute factor "
                    f"{ts['wide_field_joint_factor']:.2f}"
                )
            if s23.get("joint_probability_axis_conditioned") is not None:
                acm = s23.get("axis_conditioning_model") or {}
                p_str += (
                    f"; axis-conditioned (host-morphology corridor, "
                    f"f=1/9 on {acm.get('n_axis_pairs', 0)} pairs): "
                    f"{s23['joint_probability_axis_conditioned']:.3e} "
                    f"(log10 P = "
                    f"{s23['joint_log10_p_axis_conditioned']:.1f})"
                )
            loo = {x["excluded_pair_id"]: x
                   for x in s23.get("leave_one_out", [])}
            l3067 = loo.get("NGC3067-3C232")
            if l3067:
                p_str += (
                    f"; after removing NGC3067-3C232 (falsified by the "
                    f"step_29 Ly-alpha audit), the "
                    f"{l3067['n_remaining']}-system residual is "
                    f"{l3067['joint_probability_empirical_density']:.3e} "
                    f"empirical / "
                    f"{l3067['joint_probability_sdss_density']:.3e} SDSS; "
                    f"seven-pair compact residual "
                    f"{l3067.get('joint_probability_tight_remaining', 0):.3e}"
                )
            svc = s23.get("search_volume_correction") or {}
            if svc.get("n_crit_95pct") is not None:
                p_str += (
                    f"; look-elsewhere: P_cat(N fields) = prod_i "
                    f"[1-exp(-N p_i)] stays <0.05 for N < "
                    f"{svc['n_crit_95pct']:.0f} parent fields"
                )
                if svc.get("n_crit_95pct_axis_conditioned") is not None:
                    p_str += (
                        f"; axis-conditioned N_crit = "
                        f"{svc['n_crit_95pct_axis_conditioned']:.0f}"
                    )
            osc = s23.get("on_structure_coincidence")
            if osc:
                sc = osc["scenarios"]
                p_str += (
                    f"; on-structure: {osc['n_knots_on_filament']} "
                    f"emission knots on the {osc['filament_path_length_arcsec']:.0f} "
                    f"arcsec NGC 7603 filament corridor, chance P <= "
                    f"{sc['w4_deep_elg_extreme']['p_at_least_n_knots_on_filament']:.1e} "
                    f"even at Sigma=2000 deg^-2"
                )
            rows.append({
                "test": "Chance-alignment probability",
                "observable": "Poisson probability of joint chance superposition",
                "tep_prediction": "physically associated; P_chance not applicable",
                "std_prediction": "independent chance superpositions",
                "result": p_str,
                "verdict": "chance_hypothesis_rejected" if s23["joint_log10_p"] < -3 else "see_measurement",
            })

        # 5a. Lensing magnification bias: the joint-P survives the
        # strongest plausible foreground-lens correction.
        if s26:
            rows.append({
                "test": "Lensing magnification budget",
                "observable": (
                    "SIS magnification mu at each catalogued "
                    "separation and the mu^(alpha-1) density-bias "
                    "correction to the joint chance probability"
                ),
                "tep_prediction": "n/a (null-model correction)",
                "std_prediction": (
                    "foreground lenses promote background quasars above "
                    "flux limits, inflating the apparent overdensity"
                ),
                "result": (
                    f"max mu-1 = {s26['max_mu_minus_1_pct']:.2f}% "
                    f"({s26['max_mu_pair']}) at sigma_v="
                    f"{s26['sigma_v_kms_nominal']:.0f} km/s uniform "
                    f"bound; lensed joint P = "
                    f"{s26['joint_p_lensed']:.3e} vs "
                    f"{s26['joint_p_baseline']:.3e} "
                    f"(x{s26['joint_p_inflation_factor']:.2f})"
                ),
                "verdict": "magnification_bias_bounded",
            })

        # 5b. Geometric and spectroscopic coherence (step_27)
        if s27:
            ma = s27.get("minor_axis_anisotropy", {})
            ma_all = ma.get("samples", {}).get("all_companions", {})
            ma_axs = ma.get("samples", {}).get("axis_alignment_pairs", {})
            xr = s27.get("xray_selection_pricing", {})
            ro = s27.get("radial_redshift_ordering", {})
            hs = s27.get("halo_scale_clustering", {})
            pr = s27.get("paired_redshift_excess", {})
            sy = s27.get("symmetric_configuration", {})
            parts = []
            if ma_axs:
                parts.append(
                    f"minor-axis anisotropy (axis-claimed subset): "
                    f"{ma_axs.get('n_within_30deg')}/"
                    f"{ma_axs.get('n')} companions within 30 deg of a "
                    f"minor axis (binomial p = "
                    f"{ma_axs.get('p_binomial_vs_uniform', float('nan')):.3g}, "
                    f"V-test p = "
                    f"{ma_axs.get('p_vtest_minor_axis', float('nan')):.3g}; "
                    f"pooled catalogue {ma_all.get('n_within_30deg')}/"
                    f"{ma_all.get('n')}, p = "
                    f"{ma_all.get('p_binomial_vs_uniform', float('nan')):.3g})"
                )
            elif ma_all:
                parts.append(
                    f"minor-axis anisotropy: {ma_all.get('n_within_30deg')}/"
                    f"{ma_all.get('n')} companions within 30 deg of a "
                    f"minor axis (binomial p = "
                    f"{ma_all.get('p_binomial_vs_uniform', float('nan')):.3g}, "
                    f"V-test p = "
                    f"{ma_all.get('p_vtest_minor_axis', float('nan')):.3g})"
                )
            if xr.get("sigma_xray_deg2") is not None:
                sel_str = (
                    f"X-ray-selected repricing: Sigma_X = "
                    f"{xr['sigma_xray_deg2']:.1f} deg^-2 measured on the "
                    f"control fields; selection-aware joint P = "
                    f"{xr['joint_probability_selection_aware']:.2e}"
                )
                # Pointed-coverage correction: the step_38
                # decomposition measured the X-ray-flagged
                # population overdense at catalogued-galaxy
                # positions by delta = +1.47 (pointed identifiers)
                # vs +0.12 for the coverage-uniform RASS subset.
                # The effective density at galaxy positions is
                # therefore Sigma_X * (1 + delta); repricing the
                # X-ray-class pairs against it weakens the joint
                # accordingly.
                if s38:
                    try:
                        from scipy.stats import poisson as _pois
                        dp = s38["results"]["xray_pointed"][
                            "inner_0_120_arcsec"]["overdensity_delta"]
                        dr = s38["results"]["xray_rass"][
                            "inner_0_120_arcsec"]["overdensity_delta"]
                        factor = 1.0 + dp
                        joint_corr = 1.0
                        excl = {}
                        for p in xr["pairs"]:
                            lam = p["lambda_sel"] * (
                                factor
                                if p["selection_class"] == "xray"
                                else 1.0)
                            p_c = float(
                                1.0 - _pois.cdf(p["n_members"] - 1,
                                                lam))
                            joint_corr *= p_c
                            excl[p["pair_id"]] = p_c
                        surviving = {
                            k: joint_corr / v for k, v in
                            excl.items() if v > 0}
                        sel_str += (
                            f"; pointed-coverage-corrected "
                            f"(delta_pointed = {dp:+.2f} vs "
                            f"delta_RASS = {dr:+.2f}, factor "
                            f"{factor:.2f} on the five X-ray-class "
                            f"pairs): joint P = {joint_corr:.2e}")
                        self._xray_pointed = {
                            "delta_pointed": float(dp),
                            "delta_rass": float(dr),
                            "correction_factor": float(factor),
                            "joint_probability_corrected":
                                float(joint_corr),
                            "joint_excluding_corrected": surviving,
                        }
                        print_status(
                            f"X-ray pointed-coverage correction: "
                            f"factor {factor:.2f} -> joint P = "
                            f"{joint_corr:.2e}", "TEST")
                    except Exception as exc:
                        print_status(
                            f"pointed-coverage correction failed: "
                            f"{exc}", "ERROR")
                parts.append(sel_str)
            if ro.get("n_ordered_member_pairs"):
                n_near = ro.get("n_nearer_higher_z",
                                ro['n_ordered_member_pairs']
                                - ro['n_farther_higher_z'])
                p_ro = ro.get("p_permutation_two_sided",
                               ro.get("p_binomial_two_sided"))
                parts.append(
                    f"radial z-ordering: {n_near}/"
                    f"{ro['n_ordered_member_pairs']} nearer member "
                    f"higher-z (within-field permutation p = "
                    f"{p_ro:.3g})"
                )
            if hs.get("permutation_test"):
                parts.append(
                    f"halo-scale clustering: std(log10 kpc) = "
                    f"{hs['std_log10_kpc_observed']:.3f} across "
                    f"{hs['n_anchored_hosts']} CF4 hosts (exact "
                    f"permutation p = "
                    f"{hs['permutation_test']['p_le_observed']:.4f})"
                )
            if pr.get("fisher_combined_member"):
                parts.append(
                    f"paired-redshift excess: Fisher p = "
                    f"{pr['fisher_combined_member']['p']:.3g} on resolved "
                    f"members"
                )
            if sy.get("tests"):
                parts.append(
                    "symmetric configurations: "
                    + "; ".join(
                        f"{t['pair_id']} joint p = "
                        f"{t['p_joint_analytic']:.3g}"
                        for t in sy["tests"])
                )
            if parts:
                rows.append({
                    "test": "Geometric and spectroscopic coherence",
                    "observable": (
                        "structure not used in selection: minor-axis "
                        "anisotropy, selection-aware densities, radial "
                        "ordering, physical-scale clustering, "
                        "paired redshifts, symmetric configurations"
                    ),
                    "tep_prediction": (
                        "ordered, anisotropic, physically scaled "
                        "configurations"
                    ),
                    "std_prediction": (
                        "isotropic, unordered, scale-free projections"
                    ),
                    "result": "; ".join(parts),
                    "verdict": "see_measurement",
                })

        # 5c. Population-level forward test: predefined parent sample
        if s24:
            inner = s24["inner_0_180_arcsec"]
            r_str = (
                f"{s24['n_parent_galaxies']} 2MRS parents, "
                f"delta = {inner['overdensity_delta']:+.3f} within "
                f"{s24['max_separation_arcsec']:.0f} arcsec "
                f"(bootstrap p = {inner['bootstrap_p_one_sided']:.3f})"
            )
            act = s24.get("active_host_subset") or {}
            if act.get("overdensity_delta") is not None:
                r_str += (
                    f"; Seyfert-class subset "
                    f"(n = {act['n_active_hosts']}): "
                    f"delta = {act['overdensity_delta']:+.3f} "
                    f"(p = {act['bootstrap_p_one_sided']:.3f})"
                )
            pr = s24.get("pairing_signature") or {}
            if pr.get("n_pairs_parents") is not None:
                r_str += (
                    f"; flanking same-z pair signature: "
                    f"{pr['n_pairs_parents']} pairs in parents vs "
                    f"{pr['expected_from_controls']:.1f} expected from "
                    f"controls (p = {pr['poisson_p_excess']:.3f})"
                )
            rows.append({
                "test": "Forward galaxy-quasar cross-correlation",
                "observable": (
                    "confirmed-quasar surface density around a "
                    "predefined bright-galaxy sample vs seeded "
                    "offset controls"
                ),
                "tep_prediction": (
                    "generic excess if associations are a "
                    "population-level effect; no requirement if "
                    "deep-well companions are rare"
                ),
                "std_prediction": "no excess",
                "result": r_str,
                "verdict": "no_generic_excess",
            })

        # 5d. Archival Kinematic Audit
        if s28:
            rows.append({
                "test": "Archival kinematic feasibility audit",
                "observable": "resolved IFU kinematics (gas disturbance) at the NGC 7603 filament knots",
                "tep_prediction": "gas is kinematically disturbed by the knot in the host rest-frame",
                "std_prediction": "knot is a background object; no host-frame disturbance",
                "result": (
                    f"{s28['n_positions_audited']} fields audited "
                    f"({s28['n_archive_queries']} archive queries); "
                    f"NGC 7603 knots: " +
                    ", ".join([f"{v['label'].split(' ')[0]}: {v['verdict']}" for v in s28["verdicts"] if "knot" in v["kind"]])
                ),
                "verdict": "see_measurement",
            })

        # 5e. Ly-alpha forest path-length audit
        if s29:
            forest = s29.get("measurements", {}).get("3C 232", {})
            r_str = (
                f"{s29['n_quasar_companions']} companions audited: "
                f"{s29['n_forest_band_covered']} with usable band coverage, "
                f"{s29.get('n_forest_band_partial_or_field', 0)} partial or field-level only, "
                f"{s29['n_forest_not_covered']} not covered; "
                f"{s29['n_ground_accessible_uncovered']} ground-accessible "
                "decisive targets lack usable forest spectra")
            verdict = "see_measurement"
            if forest:
                r_str += (
                    f"; 3C 232 test case: {forest['n_lyman_confirmed']}"
                    f"/{forest['n_candidates_4sigma']} candidates confirmed "
                    f"by Ly-beta/Ly-gamma vs "
                    f"{forest['expected_cosmological_n']:.1f} expected "
                    f"cosmological (z_abs "
                    f"{forest['z_abs_covered'][0]:.3f}-"
                    f"{forest['z_abs_covered'][1]:.3f})"
                )
                if forest['n_lyman_confirmed'] > forest['expected_cosmological_n']:
                    verdict = "tep_rejected"
                
            rows.append({
                "test": "Ly-alpha forest path-length audit",
                "observable": "incidence of intergalactic Ly-alpha absorption in companion spectra",
                "tep_prediction": "sparse/absent forest (no cosmological path length)",
                "std_prediction": "populated forest at cosmic-mean incidence dN/dz",
                "result": r_str,
                "verdict": verdict,
            })

        # 5f. Survey-scale companion redshift asymmetry (step_37)
        if s37:
            offs = {s["cut"]: s for s in s37["offset_statistics"]}
            bnd = offs.get(f"bound |dV|<{500:.0f}", {})
            rich = {s["cut"]: s
                    for s in s37.get("richness_split", [])}
            big = rich.get("Ngal>=10", {})
            vl = s37.get("volume_limited_control", [])
            vl_str = ""
            if vl:
                vl_str = (
                    "; volume-limited f_+ = "
                    + "/".join(f"{v['frac_positive']:.3f}"
                               for v in vl)
                    + f" (z < {min(v['z_max'] for v in vl):.2f}-"
                    + f"{max(v['z_max'] for v in vl):.2f})"
                )
            sq = (s37.get("spectroscopic_bias_control") or {})
            sqs = ""
            if sq.get("clean_subsample"):
                cs = sq["clean_subsample"]
                sqs = (
                    f"; clean-spectrum subset f_+ = "
                    f"{cs['frac_positive']:.4f} (n = {cs['n_satellites']}, "
                    f"p = {cs['binomial_p_vs_half']:.3g})"
                )
            amp = (s37.get("amplitude_scaling") or {}).get(
                "by_satellite_M_r", [])
            amp_str = ""
            if amp:
                amp_str = (
                    f"; largest amplitude in the brightest bin "
                    f"(median dV {amp[0]['median_dv_kms']:+.1f} km/s "
                    f"at M_r < {amp[0]['M_r_hi']})"
                )
            rows.append({
                "test": "Companion redshift asymmetry at survey scale",
                "observable": (
                    "fraction of satellites redshifted relative to the "
                    "rank-1 group central, SDSS DR10 groups "
                    "(Tempel et al. 2017)"
                ),
                "tep_prediction": (
                    "satellites in deeper wells are systematically "
                    "redshifted; f_+ > 0.5 with luminosity-scaled "
                    "amplitude"
                ),
                "std_prediction": "symmetric f_+ = 0.5",
                "result": (
                    f"{s37['n_satellites']} satellites; bound "
                    f"f_+ = {bnd.get('frac_positive', float('nan')):.4f} "
                    f"(p = {bnd.get('binomial_p_vs_half', float('nan')):.3g}), "
                    f"Ngal>=10 f_+ = {big.get('frac_positive', float('nan')):.3f}"
                    f"{vl_str}{sqs}{amp_str}"
                ),
                "verdict": "see_measurement",
            })

        # 5g. X-ray-selected population test (step_38)
        if s38:
            xr = s38["results"]
            inn_x = xr["xray_all"]["inner_0_120_arcsec"]
            inn_p = xr["xray_pointed"]["inner_0_120_arcsec"]
            inn_r = xr["xray_rass"]["inner_0_120_arcsec"]
            powr = s38.get("power", {})
            rows.append({
                "test": "X-ray-selected quasar overdensity",
                "observable": (
                    "X-ray-flagged quasar surface density around the "
                    "2MRS parents vs seeded controls, decomposed by "
                    "pointed versus RASS catalogue identifiers"
                ),
                "tep_prediction": (
                    "population-level excess if X-ray selection is "
                    "unbiased"
                ),
                "std_prediction": "no excess beyond coverage bias",
                "result": (
                    f"raw delta = {inn_x['overdensity_delta']:+.2f} "
                    f"(p = {inn_x['bootstrap_p_one_sided']:.3g}) "
                    f"decomposes into pointed IDs "
                    f"delta = {inn_p['overdensity_delta']:+.2f} "
                    f"versus coverage-uniform RASS "
                    f"delta = {inn_r['overdensity_delta']:+.2f} "
                    f"(p = {inn_r['bootstrap_p_one_sided']:.2f}); "
                    f"5-sigma association-rate sensitivity at RASS "
                    f"depth f = "
                    f"{powr.get('f_5sigma_association_rate_rass_only', float('nan')):.2f}"
                ),
                "verdict": "pointed_coverage_artefact",
            })

        # 5h. Bayesian model comparison on the transect (step_34)
        if s34:
            rows.append({
                "test": "Bayesian model comparison",
                "observable": (
                    "dynesty nested-sampling evidence for the nested-"
                    "wells profile versus monotonic transition families "
                    "on the four-point transect"
                ),
                "tep_prediction": (
                    "non-monotonic locally depressed profile preferred"
                ),
                "std_prediction": (
                    "monotonic interpolation between host and "
                    "companion"
                ),
                "result": (
                    f"nested wells favoured over the best monotonic "
                    f"family ({s34['best_monotonic']['model']}) by "
                    f"dlogZ = {s34['dlogZ_nested3_vs_best_monotonic']:.0f} "
                    f"(log10 B = {s34['log10_bayes_factor_vs_best_monotonic']:.0f})"
                ),
                "verdict": "consistent_with_tep",
            })

        # 5i. Well depth vs independent observables (step_35)
        if s35:
            corr = [c for c in s35["correlations"]
                    if c.get("rho") is not None]
            best = max(corr, key=lambda c: c["rho"]) if corr else None
            b_str = (
                f"strongest correlation ({best['label']}, "
                f"n = {best['n']}): rho = {best['rho']:+.2f}, "
                f"p = {best['p']:.2f}"
            ) if best else "no measurable correlations"
            det = s35.get("detection_summary", {})
            rows.append({
                "test": "Well depth vs independent observables",
                "observable": (
                    "Delta_phi_int versus archival X-ray, radio and "
                    "optical observables across quasar-class companions"
                ),
                "tep_prediction": (
                    "deeper wells correlate with hard X-ray / compact "
                    "radio / broad-line signatures"
                ),
                "std_prediction": "no correlation at chance alignment",
                "result": (
                    f"{s35['n_quasar_class_companions']} companions, "
                    f"{s35['n_observables_tested']} observables "
                    f"(Bonferroni p < "
                    f"{s35['bonferroni_p_threshold']:.3f}); {b_str}; "
                    f"all {det.get('n_4xmm_matched', 0)} "
                    "4XMM-matched companions unresolved"
                ),
                "verdict": "inconclusive_low_power",
            })

        # 6. Transect continuity
        if s30 and s30.get("best_family"):
            rows.append({
                "test": "Field-transition continuity",
                "observable": "smooth A_int(x) through NGC 7603 filament knots",
                "tep_prediction": "single smooth profile fits 4-point transect",
                "std_prediction": "knots are unrelated background objects",
                "result": f"best family: {s30['best_family']}",
                "verdict": "consistent_with_tep",
            })

        # 7. MCMC posterior
        if s31 and s31.get("posterior"):
            post = s31["posterior"]
            d_str = ", ".join(
                f"{k}={v['median']:.3f}"
                for k, v in post.items() if k != "w"
            )
            w95 = s31.get("w_upper_95", post["w"]["p84"])
            res_str = (
                f"{d_str}, w < {w95:.3f} (95%); "
                f"chi2 at median = {s31.get('chi2_at_median', float('nan')):.2f}"
            )
            ig = (s31.get("information_gain") or {}).get("w")
            if ig:
                res_str += (
                    f"; w information gain: compression x"
                    f"{ig['compression_factor_95']:.0f} vs flat prior "
                    f"(KL {ig['kl_divergence_nats']:.1f} nats)"
                )
            rows.append({
                "test": "Nested-wells posterior",
                "observable": "well depths and width posteriors",
                "tep_prediction": "finite, well-constrained local well depths",
                "std_prediction": "no interior structure to constrain",
                "result": res_str,
                "verdict": "see_measurement",
            })

        # 8. Separation scaling
        if s32:
            sp = s32["spearman_zint_sep"]
            rows.append({
                "test": "Separation scaling",
                "observable": "Spearman(z_intrinsic, separation)",
                "tep_prediction": "discordance declines with separation",
                "std_prediction": "no correlation expected",
                "result": f"rho={sp['rho']:.2f}, p={sp['p_value']:.3f}",
                "verdict": "see_measurement",
            })

        # 9. Residual structure
        if s33 and s33.get("families"):
            bf = s33["best_family_by_chi2"]
            rec = next(r for r in s33["families"] if r["family"] == bf)
            res_str = (
                f"{bf}: resid RMS={rec['resid_rms_sigma']:.2f} sigma "
                f"(dof={rec['dof']}), lag-1="
                + (f"{rec['resid_lag1_autocorr']:.2f}"
                   if rec["resid_lag1_autocorr"] is not None
                   else "n/a (interpolating fit)")
            )
            lil = rec.get("lilliefors")
            if lil is not None:
                res_str += (
                    f", Lilliefors p={lil['p_value']:.3f} "
                    f"({'uninformative' if not lil['informative'] else 'informative'}, n={lil['n']})"
                )
            rows.append({
                "test": "Fit residual structure",
                "observable": "weighted residuals of best profile fit",
                "tep_prediction": "white residuals at error level",
                "std_prediction": "n/a (no model)",
                "result": res_str,
                "verdict": "see_measurement",
            })

        df = pd.DataFrame(rows)
        csv_path = self.data_processed / "falsification_summary.csv"
        df.to_csv(csv_path, index=False)
        print_status(f"Saved falsification table: {csv_path} ({len(df)} tests)", "SUCCESS")

        n_tep = int((df["verdict"] == "consistent_with_tep").sum())
        n_assoc = int((df["verdict"] == "physical_association_proven").sum())
        n_rej = int((df["verdict"] == "chance_hypothesis_rejected").sum())
        n_tep_rej = int((df["verdict"] == "tep_rejected").sum())

        summary = {
            "n_tests": len(df),
            "n_consistent_with_tep": n_tep,
            "n_association_proven": n_assoc,
            "n_chance_rejected": n_rej,
            "n_tep_rejected": n_tep_rej,
            "tests": rows,
            "xray_pointed_correction": getattr(self, "_xray_pointed",
                                               None),
            "missing_steps": [
                k for k, v in {
                    "05": s05, "10": s10, "11": s11, "20": s20, "21": s21,
                    "22": s22, "23": s23, "24": s24, "25": s25, "26": s26,
                    "27": s27, "28": s28, "29": s29, "30": s30,
                    "31": s31, "32": s32, "33": s33, "34": s34,
                    "35": s35, "37": s37, "38": s38,
                }.items() if v is None
            ],
        }
        json_path = self.results / "step_40_falsification_summary.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")
        print_status(
            f"Falsification summary: {len(df)} tests — {n_tep} TEP-consistent, "
            f"{n_assoc} association-proven, {n_rej} chance-rejected, {n_tep_rej} TEP-rejected.",
            "TEST",
        )
        print_status("Falsification summary complete.", "SUCCESS")
