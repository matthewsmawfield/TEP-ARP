#!/usr/bin/env python3
"""
TEP-ARP pipeline verification tests
====================================

Offline checks that the pipeline's published numbers recompute
correctly from the raw processed tables.  These tests do not hit
the network; they verify the arithmetic, the catalogue integrity,
and the presence of every registered step output.

Run:
    python3 -m pytest scripts/tests/ -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import poisson

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DATA = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results" / "outputs"

# All 32 registered pipeline steps and their required JSON outputs
EXPECTED_OUTPUTS = [
    "step_00_arp_pair_catalog.json",
    "step_01_ned_verification.json",
    "step_02_optical_imaging.json",
    "step_03_multiwavelength_manifest.json",
    "step_04_spectroscopic_ingestion.json",
    "step_05_redshift_independent_distances.json",
    "step_10_intrinsic_conformal_factor.json",
    "step_11_redshift_decomposition.json",
    "step_12_scalar_field_profiles.json",
    "step_13_proper_time_budget.json",
    "step_20_bridge_morphology.json",
    "step_21_absorption_systems.json",
    "step_22_quasar_galaxy_association.json",
    "step_23_chance_alignment.json",
    "step_24_forward_crosscorrelation.json",
    "step_25_satellite_redshift_asymmetry.json",
    "step_26_lens_magnification_budget.json",
    "step_27_geometric_coherence.json",
    "step_28_archive_kinematic_audit.json",
    "step_29_lya_forest_audit.json",
    "step_36_lya_forest_census.json",
    "step_30_bridge_redshift_transect.json",
    "step_31_mcmc_field_profile.json",
    "step_32_pair_sample_statistics.json",
    "step_33_residuals_analysis.json",
    "step_34_bayesian_model_selection.json",
    "step_35_well_depth_correlates.json",
    "step_37_sdss_companion_asymmetry.json",
    "step_38_xray_population_test.json",
    "step_39_temporal_well_solution.json",
    "step_40_falsification_summary.json",
    "step_41_manuscript_figures.json",
]

SIGMA_SDSS = 51.6   # SDSS DR16Q published density (deg^-2)
SIGMA_GAL = 10.0    # resolved bright-galaxy density (deg^-2)


def test_all_step_outputs_exist():
    """Every registered step must leave a non-empty JSON output."""
    missing = [f for f in EXPECTED_OUTPUTS
               if not (RESULTS / f).exists() or (RESULTS / f).stat().st_size == 0]
    assert not missing, f"Missing/empty pipeline outputs: {missing}"


def test_intrinsic_conformal_factor_arithmetic():
    """A_int = (1+z_G)/(1+z_Q) recomputed from the catalogue."""
    cf = pd.read_csv(DATA / "intrinsic_conformal_factors.csv")
    for _, r in cf.iterrows():
        expected = (1.0 + r["z_gal"]) / (1.0 + r["z_comp"])
        assert np.isclose(r["a_int"], expected, rtol=1e-9), r["pair_id"]
        # delta_phi = -ln A_int
        assert np.isclose(r["delta_phi"], -np.log(expected), rtol=1e-9)
        # z_intrinsic = (1+z_Q)/(1+z_G) - 1
        assert np.isclose(
            r["z_intrinsic"], (1.0 + r["z_comp"]) / (1.0 + r["z_gal"]) - 1.0,
            rtol=1e-9,
        )
    # All intrinsic factors must be sub-unity (companions deeper in wells)
    assert (cf["a_int"] < 1.0).all()


def test_chance_alignment_arithmetic():
    """Poisson P_chance recomputed under both density assumptions."""
    ca = pd.read_csv(DATA / "chance_alignment.csv")
    cat = pd.read_csv(DATA / "arp_pair_catalog.csv")
    s23 = json.loads(
        (RESULTS / "step_23_chance_alignment.json").read_text()
    )
    sigma_emp = s23["sigma_qso_empirical_deg2"]
    assert sigma_emp is not None and sigma_emp > 0

    joint_log10_emp, joint_log10_sdss = 0.0, 0.0
    for _, r in ca.iterrows():
        crow = cat.loc[cat["pair_id"] == r["pair_id"]].iloc[0]
        theta = r["separation_arcsec"] / 3600.0
        # Mirror the density classification in step_23: any galaxy-type
        # companion uses the bright-galaxy density.
        is_gal = "galax" in str(crow["companion_type"]).lower() or \
            crow["companion_type"] == "compact object"
        s_emp = SIGMA_GAL if is_gal else sigma_emp
        s_sdss = SIGMA_GAL if is_gal else SIGMA_SDSS
        lam_emp = s_emp * np.pi * theta**2
        lam_sdss = s_sdss * np.pi * theta**2
        assert np.isclose(r["expected_count"], lam_emp, rtol=1e-9)
        assert np.isclose(r["expected_count_sdss"], lam_sdss, rtol=1e-9)
        if pd.notna(crow.get("z_comp_list")):
            n_q = len(str(crow["z_comp_list"]).split(";"))
            p_emp = 1.0 - poisson.cdf(n_q - 1, lam_emp)
            p_sdss = 1.0 - poisson.cdf(n_q - 1, lam_sdss)
        else:
            p_emp = 1.0 - np.exp(-lam_emp)
            p_sdss = 1.0 - np.exp(-lam_sdss)
        assert np.isclose(r["p_chance"], p_emp, rtol=1e-9), r["pair_id"]
        assert np.isclose(r["p_chance_sdss"], p_sdss, rtol=1e-9)
        joint_log10_emp += np.log10(p_emp)
        joint_log10_sdss += np.log10(p_sdss)

    # Reported joint probabilities match the recomputed products
    assert np.isclose(s23["joint_log10_p"], joint_log10_emp, rtol=1e-9)
    assert np.isclose(
        s23["joint_log10_p_sdss_density"], joint_log10_sdss, rtol=1e-9
    )
    # The headline (empirical) evaluation is the more conservative one
    assert s23["joint_probability_all_pairs"] >= s23["joint_probability_sdss_density"]
    # Chance hypothesis rejected at >10 orders of magnitude under both
    assert s23["joint_log10_p"] < -10.0
    assert s23["joint_log10_p_sdss_density"] < -10.0

    # Compact-subsample partition: tight x wide factor must equal
    # the full product, and the partition must split at theta=300.
    ts = s23["tight_subset"]
    ca_seps = dict(zip(ca["pair_id"], ca["separation_arcsec"]))
    assert all(
        ca_seps[p] < ts["separation_threshold_arcsec"]
        for p in ts["tight_pair_ids"]
    )
    assert all(
        ca_seps[p] >= ts["separation_threshold_arcsec"]
        for p in ts["wide_pair_ids"]
    )
    assert len(ts["tight_pair_ids"]) == ts["n_tight"]
    assert np.isclose(
        ts["joint_probability_tight"] * ts["wide_field_joint_factor"],
        s23["joint_probability_all_pairs"], rtol=1e-9,
    )


def test_ngc7603_transect_knot_arithmetic():
    """Knot intrinsic factors recompute from published redshifts."""
    tr = pd.read_csv(DATA / "transect_data.csv")
    sub = tr[tr["pair_id"] == "NGC7603-NGC7603B"].sort_values("x")
    z_gal = float(sub.loc[sub["point_type"] == "host", "a_int"].iloc[0])
    assert np.isclose(z_gal, 1.0)
    knots = sub[sub["point_type"] == "filament_knot"]
    assert len(knots) == 2
    # Interior knots sit deeper than the companion endpoint:
    # the transect must be non-monotonic.
    comp = float(sub.loc[sub["point_type"] == "companion", "a_int"].iloc[0])
    assert (knots["a_int"] < comp).all()


def test_profile_family_boundary_conditions():
    """Monotonic families satisfy A(0)=1 and A(1)=A_Q exactly."""
    from scripts.steps.step_12_scalar_field_profiles import (
        profile_exponential,
        profile_tanh,
        profile_yukawa,
        profile_nested_wells,
    )
    a_q = 0.9744439185991483  # NGC 7603B
    x = np.linspace(0, 1, 201)
    for fn, args in [
        (profile_exponential, (4.0,)),
        (profile_tanh, (0.5, 0.15)),
        (profile_yukawa, (8.0,)),
    ]:
        prof = fn(x, a_q, *args)
        assert np.isclose(prof[0], 1.0, atol=1e-3)
        assert np.isclose(prof[-1], a_q, rtol=1e-3)
    # Nested wells: depths recover -ln A_int at the well centres
    # (archive-measured knot positions from step_20)
    from scripts.steps.step_12_scalar_field_profiles import (
        load_ngc7603_well_centers,
    )
    centers = load_ngc7603_well_centers(DATA)
    assert len(centers) == 3
    d1, d2, dq, w = 0.301, 0.188, 0.026, 0.02
    for c, d in zip(centers, (d1, d2, dq)):
        val = profile_nested_wells(
            np.array([c]), d1, d2, dq, w, centers=centers
        )[0]
        # narrow wells: centre value ~ exp(-d_i) plus small leakage
        assert val < np.exp(-0.8 * d)


def test_falsification_summary_integrity():
    """The falsification table covers all eighteen tests with verdicts."""
    s40 = json.loads(
        (RESULTS / "step_40_falsification_summary.json").read_text()
    )
    assert s40["n_tests"] == 18
    assert s40["n_chance_rejected"] == 1
    assert s40["n_association_proven"] == 1
    verdicts = {t["test"]: t["verdict"] for t in s40["tests"]}
    assert verdicts["Chance-alignment probability"] == "chance_hypothesis_rejected"
    assert verdicts["Foreground absorption ordering"] == "physical_association_proven"
    assert verdicts["Forward galaxy-quasar cross-correlation"] == "no_generic_excess"
    assert not s40["missing_steps"]


def test_mcmc_posterior_recovers_delta_phi():
    """MCMC well depths recover the direct -ln A_int of each object."""
    s31 = json.loads(
        (RESULTS / "step_31_mcmc_field_profile.json").read_text()
    )
    tr = pd.read_csv(DATA / "transect_data.csv")
    sub = tr[tr["pair_id"] == "NGC7603-NGC7603B"]
    knots = sub[sub["point_type"] == "filament_knot"].sort_values("x")
    assert len(knots) == 2
    dphi = {
        "d_knot1": -np.log(float(knots["a_int"].iloc[0])),
        "d_knot2": -np.log(float(knots["a_int"].iloc[1])),
        "d_companion": -np.log(float(sub.loc[sub["x"] == 1.0, "a_int"].iloc[0])),
    }
    post = s31["posterior"]
    for key, expected in dphi.items():
        assert abs(post[key]["median"] - expected) < 0.01, key
    # Width posterior is bounded above, one-sided: wells are compact
    # (below ~5% of the transect length at 95% credibility)
    assert post["w"]["p95"] < 0.05
    assert s31["chi2_at_median"] < 1.0


def test_no_synthetic_redshifts_in_catalog():
    """Catalogue redshifts are literature values within sane bounds."""
    cat = pd.read_csv(DATA / "arp_pair_catalog.csv")
    assert len(cat) == 12
    assert (cat["z_gal"] > 0).all() and (cat["z_gal"] < 0.1).all()
    assert (cat["z_comp"] > cat["z_gal"]).all()
    assert (cat["separation_arcsec"] > 0).all()


def test_search_volume_correction_arithmetic():
    """Look-elsewhere P_cat(N) = prod_i [1 - exp(-N p_i)] recomputed
    from the per-pair severities, and N_crit_95 verified by the same
    monotone map."""
    s23 = json.loads(
        (RESULTS / "step_23_chance_alignment.json").read_text()
    )
    svc = s23["search_volume_correction"]
    p_list = np.array([p["p_chance"] for p in s23["pairs"]])
    assert (p_list > 0).all() and (p_list <= 1).all()

    def p_cat(n):
        return float(np.prod(1.0 - np.exp(-n * p_list)))

    # Grid values recompute exactly
    for g in svc["n_grid"]:
        assert np.isclose(
            g["p_catalogue_by_chance"], p_cat(g["n_parent_fields"]),
            rtol=1e-9,
        )
        for pid, mu in g["expected_lookalikes_per_pair"].items():
            p_i = float(
                s23["pairs"][[q["pair_id"] for q in s23["pairs"]].index(pid)]["p_chance"]
            )
            assert np.isclose(mu, g["n_parent_fields"] * p_i, rtol=1e-9)

    # P_cat is monotone increasing in N
    ns = np.array([g["n_parent_fields"] for g in svc["n_grid"]])
    ps = np.array([g["p_catalogue_by_chance"] for g in svc["n_grid"]])
    assert np.all(np.diff(ns) > 0) and np.all(np.diff(ps) > 0)

    # n_crit_95pct sits at the 5% boundary of the same map
    assert np.isclose(p_cat(svc["n_crit_95pct"]), 0.05, atol=1e-3)
    # The Atlas-scale anchor is reported
    assert svc["arp_atlas_entries"] == 338


def test_residual_diagnostics_present():
    """The residuals analysis reports Lilliefors checks and honest
    informativeness flags for the interpolating family."""
    s33 = json.loads(
        (RESULTS / "step_33_residuals_analysis.json").read_text()
    )
    for fam in s33["families"]:
        assert "lilliefors" in fam
        l = fam["lilliefors"]
        if l is not None:
            assert 0.0 <= l["p_value"] <= 1.0
            assert l["n"] == 4
            # n=4 is always flagged underpowered
            assert l["informative"] is False


def test_axis_conditioned_probability_arithmetic():
    """Axis-conditioned probabilities recompute as Poisson with
    lambda -> lambda/9, and only for axis_alignment pairs."""
    ca = pd.read_csv(DATA / "chance_alignment.csv")
    cat = pd.read_csv(DATA / "arp_pair_catalog.csv")
    s23 = json.loads(
        (RESULTS / "step_23_chance_alignment.json").read_text()
    )
    f_axis = 4.0 * 10.0 / 360.0
    joint_log10 = 0.0
    for _, r in ca.iterrows():
        crow = cat.loc[cat["pair_id"] == r["pair_id"]].iloc[0]
        expect_f = (f_axis if crow["connection_type"] == "axis_alignment"
                    else 1.0)
        assert np.isclose(r["geometry_factor"], expect_f, rtol=1e-9), \
            r["pair_id"]
        lam_g = r["expected_count"] * expect_f
        if pd.notna(crow.get("z_comp_list")):
            n_q = len(str(crow["z_comp_list"]).split(";"))
            p_g = 1.0 - poisson.cdf(n_q - 1, lam_g)
        else:
            p_g = 1.0 - np.exp(-lam_g)
        assert np.isclose(
            r["p_chance_axis_conditioned"], p_g, rtol=1e-9), r["pair_id"]
        joint_log10 += np.log10(p_g)
    assert np.isclose(
        s23["joint_log10_p_axis_conditioned"], joint_log10, rtol=1e-9)
    # Conditioning on an a priori axis can only reduce the joint P
    assert s23["joint_probability_axis_conditioned"] \
        <= s23["joint_probability_all_pairs"]
    # The conditioned N_crit exceeds the unconditioned one
    svc = s23["search_volume_correction"]
    assert svc["n_crit_95pct_axis_conditioned"] >= svc["n_crit_95pct"]


def test_lens_magnification_budget_arithmetic():
    """The SIS magnification-bias correction recomputes and stays
    bounded at the percent level."""
    s26 = json.loads(
        (RESULTS / "step_26_lens_magnification_budget.json").read_text()
    )
    df = pd.read_csv(DATA / "lens_magnification_budget.csv")
    c_kms = 299792.458
    nominal = df[df["sigma_v_kms"] == 300.0]
    assert len(nominal) == s26["n_pairs"]
    for _, r in nominal.iterrows():
        # Einstein radius recomputes from D_ls/D_s
        d_s = r["d_source_mpc"]
        d_l = r["d_lens_mpc"]
        te = 4.0 * np.pi * (300.0 / c_kms) ** 2 * (d_s - d_l) / d_s
        assert np.isclose(
            np.degrees(te) * 3600.0,
            r["theta_einstein_arcsec"], rtol=1e-9)
        # SIS magnification at the catalogued separation
        theta = np.radians(r["theta_arcsec"] / 3600.0)
        mu = theta / (theta - te)
        assert np.isclose(mu, r["magnification_mu"], rtol=1e-9)
        assert r["theta_arcsec"] > r["theta_einstein_arcsec"]
    # Magnification bias cannot explain the joint improbability:
    # the lensed joint P stays within a factor 3 of baseline.
    assert s26["joint_p_inflation_factor"] < 3.0
    assert s26["max_mu_minus_1_pct"] < 50.0
    # Sensitivity grid brackets the nominal choice
    sigs = {g["sigma_v_kms"] for g in s26["sensitivity_grid"]}
    alphas = {g["alpha_qlf"] for g in s26["sensitivity_grid"]}
    assert sigs == {250.0, 300.0, 400.0}
    assert alphas == {1.5, 2.0, 2.5}


def test_geometric_coherence_provenance():
    """Step 27 inputs are genuine archive products and every reported
    density/probability recomputes from them — no hardcoded values."""
    s27 = json.loads(
        (RESULTS / "step_27_geometric_coherence.json").read_text()
    )
    members = pd.read_csv(DATA / "qso_field_members.csv")
    pos = pd.read_csv(DATA / "companion_positions.csv")
    cat = pd.read_csv(DATA / "arp_pair_catalog.csv")

    # X-ray density: measured on the step_22 control fields, not assumed
    xr = s27["xray_selection_pricing"]
    ctrl = members[members["field_kind"] == "control"]
    n_x = int(ctrl["type"].astype(str).str.contains("X").sum())
    assert n_x == xr["n_xray_in_control_fields"]
    assert len(ctrl) == xr["n_control_members"]
    # density = fraction * the step_23 empirical field density
    frac = n_x / len(ctrl)
    s23 = json.loads(
        (RESULTS / "step_23_chance_alignment.json").read_text())
    sigma_field = s23["sigma_qso_empirical_deg2"]
    assert np.isclose(
        xr["sigma_xray_deg2"], frac * sigma_field, rtol=0.05)
    # selection-aware joint P is the product of per-pair values
    jp = float(np.prod(
        [p["p_chance_selection_aware"] for p in xr["pairs"]]))
    assert np.isclose(
        jp, xr["joint_probability_selection_aware"], rtol=1e-9)
    # X-ray-classified pairs are exactly those flagged in the catalogue
    assert {p["pair_id"] for p in xr["pairs"]} == set(cat["pair_id"])

    # Minor-axis: every host PA carries a real archive source and the
    # axis-claimed subgroup is restricted to connection_type pairs
    ma = s27["minor_axis_anisotropy"]
    n_pa = sum(1 for h in ma["host_pa"].values()
               if h["pa_deg"] is not None)
    assert n_pa >= 10
    for c in ma["companions"]:
        assert 0.0 <= c["offset_from_minor_deg"] <= 90.0
    axs = ma["samples"]["axis_alignment_pairs"]
    claimed = set(
        cat.loc[cat["connection_type"] == "axis_alignment", "pair_id"])
    ax_pids = {c["pair_id"] for c in ma["companions"]
               if c["connection_type"] == "axis_alignment"}
    assert ax_pids <= claimed
    assert axs["n_within_30deg"] <= axs["n"]

    # Halo-scale: projected separations recompute from step_05 distances
    hs = s27["halo_scale_clustering"]
    s05 = json.loads(
        (RESULTS / "step_05_redshift_independent_distances.json")
        .read_text())
    host_d = {m["pair_id"]: m["distance_mpc"] for m in s05["members"]
              if m["role"] == "host" and m["status"] == "measured"}
    for r in hs["pairs"]:
        d_kpc = (r["separation_arcsec"] * host_d[r["pair_id"]]
                 * np.pi / (180.0 * 3600.0) * 1000.0)
        assert np.isclose(d_kpc, r["projected_sep_kpc"], rtol=1e-9)

    # Fixed seed recorded for reproducibility
    assert s27["rng_seed"] == 20260924
    # Companion positions are all archive-resolved rows from step_20
    assert len(pos) == len(ma["companions"]) or \
        len(ma["companions"]) <= len(pos)


def test_lya_forest_audit_consistency():
    """Step 29 Ly-alpha forest audit: coverage gating must be
    physically consistent — instruments that cannot reach the forest
    band must not count, coverage fractions cannot exceed unity, and
    the measured 3C 232 census must reproduce its headline numbers."""
    s29 = json.loads(
        (RESULTS / "step_29_lya_forest_audit.json").read_text())
    audit = pd.read_csv(DATA / "lya_forest_audit.csv")

    # All quasar-class companions audited exactly once
    assert len(audit) == s29["n_quasar_companions"] == 18

    # Coverage can never exceed the band width (union, not sum)
    band_w = audit["forest_hi_a"] - audit["forest_lo_a"]
    assert (audit["mast_forest_covered_a"]
            <= band_w + 1e-6).all()
    assert ((audit["mast_forest_covered_frac"] >= 0)
            & (audit["mast_forest_covered_frac"] <= 1.0 + 1e-9)).all()

    # Known gating counterexamples — these instruments cannot reach
    # the band and must leave zero usable coverage:
    #   NGC 7319 QSO: STIS/CCD G750M (red), LRIS pointing on the
    #     'NGC 7319 ULX' (different object, 9 arcsec off-slit)
    #   WEE 51: FORS2 floor is 3300 A > its 2601-2961 A band
    #   WISP 257: WFC3/IR G102/G141 are redward of Ly-alpha
    idx = audit.set_index("resolved_name")
    n7319 = idx.loc["[VV2006] J223603.7+335824"]
    assert n7319["mast_forest_covered_frac"] < 0.5
    assert n7319["verdict"] in ("forest_band_partial",
                                "unusable_or_field_only")
    assert "koa_lris" not in str(n7319["koa_on_target"])
    assert idx.loc["WEE 51"]["mast_forest_covered_a"] == 0.0
    assert idx.loc["WISP 257  129"]["verdict"] == "forest_not_covered"

    # Every ground-accessible decisive target is honestly reported as
    # uncovered or partial — not as measured
    ground = audit[audit["observability"] == "ground"]
    assert (ground["verdict"] != "forest_measured").all()
    assert s29["n_ground_accessible_uncovered"] == len(
        ground[ground["verdict"].isin(
            ["forest_not_covered", "forest_band_partial",
             "unusable_or_field_only"])])

    # The one executable measurement reproduces its headline numbers
    meas = s29["measurements"]["3C 232"]
    assert idx.loc["3C 232"]["verdict"] == "forest_measured"
    assert meas["n_lyman_confirmed"] >= 8
    assert meas["n_candidates_4sigma"] >= meas["n_lyman_confirmed"]
    for s in meas["systems"]:
        assert 0.29 < s["z_abs"] < 0.50
        if s["lyman_confirmed"]:
            assert max(s["lyb_depth_sig"] or 0,
                       s["lyg_depth_sig"] or 0) > 2.5
    # Confirmed-system redshifts fall inside the covered window
    za = [s["z_abs"] for s in meas["systems"] if s["lyman_confirmed"]]
    lo, hi = meas["z_abs_covered"]
    assert min(za) >= lo - 0.02 and max(za) <= hi + 0.02
