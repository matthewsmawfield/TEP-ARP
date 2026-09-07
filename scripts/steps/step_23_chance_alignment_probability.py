#!/usr/bin/env python3
"""
Step 23: Chance-Alignment Probability
=====================================
Computes the a priori probability that each discordant pair is a
chance superposition of a background quasar on a foreground
galaxy, under the standard interpretation.

The Poisson probability of finding at least one quasar within
projected radius theta of an arbitrary bright galaxy is

    P = 1 - exp(-Sigma_Q * pi * theta^2)

where Sigma_Q is the quasar surface density to the relevant
magnitude limit.  The angular scale theta is the archive-measured
nucleus-to-companion separation where step_20 resolved the
companion (the largest member separation for multi-object
companions), and the catalogued value otherwise.  The probability
is evaluated under two
assumptions and both are reported: the published SDSS DR16
quasar-catalogue value (~750,414 quasars over 14,555 deg^2,
Lyke et al. 2020), i.e. Sigma_Q ~ 51.6 deg^-2 for the bright
subset, and the more conservative empirical field density
measured by the step_22 association test from the same
catalogue in the same sight-lines (which inherits the elevated
spectroscopic completeness of these famous fields).  The
headline joint probability is the larger of the two
evaluations, so the reported rejection is the conservative one.

For pairs whose 'background' companion is luminous and resolved
(NGC 7603B) rather than a point quasar, the surface density of
comparable-brightness galaxies is used instead (~10 deg^-2 to
r < 18, conservative).

A search-volume (look-elsewhere) correction is appended: under
the null each of N parent bright-galaxy fields is an independent
Poisson trial, so the probability that a search reproduces every
observed severity class simultaneously is
P_cat(N) = prod_i [1 - exp(-N p_i)], evaluated on a grid of N
(anchored at 338, the entry count of Arp's Atlas of Peculiar
Galaxies) with the 5% crossover volume n_crit_95pct reported.

Outputs:
    results/outputs/step_23_chance_alignment.json
    data/processed/chance_alignment.csv
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.jsonio import json_safe

# Published quasar surface densities (per square degree).
SIGMA_QSO_DEG2 = 51.6    # SDSS DR16Q catalogue density (Lyke et al. 2020)
SIGMA_GAL_DEG2 = 10.0    # comparable-brightness galaxy density (conservative)


class Step23ChanceAlignment:
    """Step 23: Poisson chance-alignment probabilities."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_23",
            log_file_path=self.logs / "step_23_chance_alignment.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status("Computing chance-alignment probabilities...", "PROCESS")

        catalog_path = self.data_processed / "arp_pair_catalog_verified.csv"
        if not catalog_path.exists():
            catalog_path = self.data_processed / "arp_pair_catalog.csv"
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"Pair catalog not found: {catalog_path}. Run step_00 first."
            )
        catalog = pd.read_csv(catalog_path)

        # Archive-measured separations from the companion-resolution
        # table written by step_20 take precedence over the catalogued
        # (published or approximate) values; where a companion was not
        # resolved the catalogued value is retained and flagged.  For
        # multi-object companions the enclosing radius — the largest
        # measured member separation — is used.
        meas_sep = {}
        comp_path = self.data_processed / "companion_positions.csv"
        if comp_path.exists():
            dfc = pd.read_csv(comp_path)
            for pid, grp in dfc.groupby("pair_id"):
                meas_sep[pid] = float(grp["measured_separation_arcsec"].max())
        else:
            self.logger.warning(
                "companion_positions.csv not found; using catalogued "
                "separations for all pairs."
            )

        # Quasar surface density is evaluated under two assumptions,
        # both reported so the result is independent of the choice:
        #   * the published SDSS DR16 quasar-catalogue density
        #     (51.6 deg^-2, Lyke et al. 2020);
        #   * the empirical field density measured by the step_22
        #     association test from the same catalogue in the same
        #     fields (more conservative: it inherits the elevated
        #     spectroscopic completeness of these famous sight-lines).
        # The headline joint probability is the larger (more
        # conservative) of the two evaluations.
        empirical_path = self.results / "step_22_quasar_galaxy_association.json"
        empirical_density = None
        if empirical_path.exists():
            try:
                with open(empirical_path) as f:
                    s22 = json.load(f)
                r_deg = s22["search_radius_arcmin"] / 60.0
                area = np.pi * r_deg**2
                val = s22["field_mean_qso"] / area
                if val > 0:
                    empirical_density = float(val)
                    print_status(
                        f"Empirical QSO density from step_22: "
                        f"{empirical_density:.1f} deg^-2", "INFO"
                    )
            except (KeyError, json.JSONDecodeError):
                pass

        from scipy.stats import poisson

        # Structure-conditioned geometry factor.  For pairs whose
        # claim is that companions lie along a host-morphology axis
        # (minor axis / jet axis), the target direction is fixed by
        # the host's own light distribution BEFORE any companion
        # position is examined — an a priori corridor, not a
        # post-hoc one.  The axis corridor is taken as a +/-10 deg
        # azimuthal wedge about each of the two axis directions,
        # i.e. 40/360 = 1/9 of the full annulus.  Published
        # alignment tolerances for these systems are tighter
        # (~3-7 deg), so 10 deg is conservative.  Only
        # "axis_alignment" pairs receive the factor; bridges and
        # filaments are excluded because their claimed direction is
        # partially defined by the companion itself (their
        # structure-conditioned probability is priced separately by
        # the NGC 7603 corridor statistic below).
        AXIS_WEDGE_HALF_DEG = 10.0
        F_AXIS = 4.0 * AXIS_WEDGE_HALF_DEG / 360.0   # = 1/9

        def _eval(sig_qso):
            """Evaluate per-pair and joint P under quasar density sig_qso."""
            out_rows = []
            jl = 0.0
            jl_geom = 0.0
            for _, row in catalog.iterrows():
                pid = row["pair_id"]
                if pid in meas_sep:
                    sep_arcsec = meas_sep[pid]
                    sep_src = "archive_astrometry"
                else:
                    sep_arcsec = float(row["separation_arcsec"])
                    sep_src = "catalog_published"
                theta_deg = sep_arcsec / 3600.0
                is_gal = "galax" in str(row["companion_type"]).lower() or \
                    row["companion_type"] == "compact object"
                sigma = SIGMA_GAL_DEG2 if is_gal else sig_qso
                lam = sigma * np.pi * theta_deg**2
                f_geom = (F_AXIS
                          if row["connection_type"] == "axis_alignment"
                          else 1.0)
                lam_geom = lam * f_geom
                if "z_comp_list" in row and pd.notna(row.get("z_comp_list")):
                    n_q = len(str(row["z_comp_list"]).split(";"))
                    p = float(1.0 - poisson.cdf(n_q - 1, lam))
                    p_geom = float(1.0 - poisson.cdf(n_q - 1, lam_geom))
                else:
                    p = float(1.0 - np.exp(-lam))
                    p_geom = float(1.0 - np.exp(-lam_geom))
                out_rows.append({
                    "pair_id": pid,
                    "separation_arcsec": sep_arcsec,
                    "separation_source": sep_src,
                    "surface_density_deg2": sigma,
                    "expected_count": lam,
                    "geometry_factor": f_geom,
                    "p_chance": p,
                    "p_chance_axis_conditioned": p_geom,
                })
                if p > 0:
                    jl += np.log10(p)
                if p_geom > 0:
                    jl_geom += np.log10(p_geom)
            return out_rows, jl, jl_geom

        rows_sdss, joint_log10_sdss, _ = \
            _eval(SIGMA_QSO_DEG2)
        joint_p_sdss = 10.0**joint_log10_sdss
        print_status(
            f"Joint P under SDSS DR16 density ({SIGMA_QSO_DEG2} deg^-2): "
            f"{joint_p_sdss:.3e} (log10 P = {joint_log10_sdss:.1f})", "TEST",
        )

        if empirical_density is not None:
            rows_emp, joint_log10_emp, _ = \
                _eval(empirical_density)
            joint_p_emp = 10.0**joint_log10_emp
            print_status(
                f"Joint P under empirical field density "
                f"({empirical_density:.1f} deg^-2): {joint_p_emp:.3e} "
                f"(log10 P = {joint_log10_emp:.1f})", "TEST",
            )
            # Merge empirical columns into the primary table
            emp_by_pair = {r["pair_id"]: r for r in rows_emp}
            rows = []
            for r in rows_sdss:
                e = emp_by_pair[r["pair_id"]]
                rows.append({
                    **r,
                    "p_chance_sdss": r["p_chance"],
                    "expected_count_sdss": r["expected_count"],
                    "surface_density_deg2": e["surface_density_deg2"],
                    "expected_count": e["expected_count"],
                    "p_chance": e["p_chance"],
                    "p_chance_axis_conditioned":
                        e["p_chance_axis_conditioned"],
                })
            use_emp = True
        else:
            rows = rows_sdss
            joint_log10_emp = None
            joint_p_emp = None
            use_emp = False

        df = pd.DataFrame(rows)
        csv_path = self.data_processed / "chance_alignment.csv"
        df.to_csv(csv_path, index=False)
        print_status(f"Saved table: {csv_path}", "SUCCESS")

        # Headline: the more conservative (larger) joint probability
        if use_emp and joint_p_emp > joint_p_sdss:
            joint_p, joint_log10 = joint_p_emp, joint_log10_emp
            sigma_q = empirical_density
            sigma_src = "step_22 empirical field density (conservative)"
        else:
            joint_p, joint_log10 = joint_p_sdss, joint_log10_sdss
            sigma_q = SIGMA_QSO_DEG2
            sigma_src = "SDSS DR16Q published density (Lyke et al. 2020)"

        # Axis-conditioned joint probability on the headline density:
        # companions of axis_alignment pairs are priced against the
        # morphology-defined axis wedge rather than the full annulus.
        joint_log10_geom = float(sum(
            np.log10(r["p_chance_axis_conditioned"]) for r in rows
            if r["p_chance_axis_conditioned"] > 0
        ))
        joint_p_geom = 10.0 ** joint_log10_geom
        n_axis = int(sum(
            1 for r in rows if r["geometry_factor"] < 1.0))
        print_status(
            f"Axis-conditioned joint P (f=1/9 wedge on {n_axis} "
            f"axis pairs): {joint_p_geom:.3e} "
            f"(log10 P = {joint_log10_geom:.1f})", "TEST")

        # ----------------------------------------------------------
        # Tight-subset joint probability.  The Poisson proximity
        # statistic is the operative test only for pairs compact
        # enough that a background draw at the measured separation
        # is intrinsically improbable; the wide-field symmetric/jet
        # systems carry their evidence in structure and selection
        # rather than angular proximity, and enter the ensemble
        # product at weights near unity.  The subset product is
        # reported so the headline number cannot be read as a claim
        # that every member is a rare angular coincidence.
        TIGHT_SEP_ARCSEC = 300.0
        tight_rows = [r for r in rows
                      if r["separation_arcsec"] < TIGHT_SEP_ARCSEC]
        wide_rows = [r for r in rows
                     if r["separation_arcsec"] >= TIGHT_SEP_ARCSEC]
        joint_log10_tight = float(sum(
            np.log10(r["p_chance"]) for r in tight_rows
            if r["p_chance"] > 0))
        joint_p_tight = 10.0 ** joint_log10_tight
        log10_wide_factor = float(sum(
            np.log10(r["p_chance"]) for r in wide_rows
            if r["p_chance"] > 0))
        print_status(
            f"Tight-subset joint P ({len(tight_rows)} pairs, "
            f"theta < {TIGHT_SEP_ARCSEC:.0f} arcsec): "
            f"{joint_p_tight:.3e} (log10 P = {joint_log10_tight:.1f}); "
            f"wide-field systems contribute a factor of "
            f"{10.0**log10_wide_factor:.2f}", "TEST")

        # ----------------------------------------------------------
        # Leave-one-out sensitivity.  The joint product over the
        # remaining ensemble after each pair is dropped, priced
        # under both density assumptions; for tight-subset members
        # the residual tight product is also reported.  This is the
        # honest repricing required whenever an independent audit
        # removes a pair from the proximity ensemble (e.g. the
        # Ly-alpha path-length audit of step_29).
        leave_one_out = []
        for r in rows:
            others = [x for x in rows if x["pair_id"] != r["pair_id"]]
            loo = {
                "excluded_pair_id": r["pair_id"],
                "n_remaining": len(others),
                "joint_probability_empirical_density": float(10.0 ** sum(
                    np.log10(x["p_chance"]) for x in others
                    if x["p_chance"] > 0)),
                "joint_probability_sdss_density": float(10.0 ** sum(
                    np.log10(x["p_chance_sdss"]) for x in others
                    if x["p_chance_sdss"] > 0)),
                "joint_probability_axis_conditioned": float(10.0 ** sum(
                    np.log10(x["p_chance_axis_conditioned"])
                    for x in others
                    if x["p_chance_axis_conditioned"] > 0)),
            }
            tight_others = [x for x in tight_rows
                            if x["pair_id"] != r["pair_id"]]
            if r in tight_rows:
                loo["joint_probability_tight_remaining"] = float(
                    10.0 ** sum(
                        np.log10(x["p_chance"]) for x in tight_others
                        if x["p_chance"] > 0))
                loo["n_tight_remaining"] = len(tight_others)
            leave_one_out.append(loo)
        loo_ngc3067 = next(
            x for x in leave_one_out
            if x["excluded_pair_id"] == "NGC3067-3C232")
        print_status(
            f"Leave-one-out repricing (dropping NGC3067-3C232, "
            f"falsified by the step_29 Ly-alpha audit): "
            f"{loo_ngc3067['joint_probability_empirical_density']:.2e} "
            f"empirical / "
            f"{loo_ngc3067['joint_probability_sdss_density']:.2e} "
            f"SDSS density; tight subset "
            f"{loo_ngc3067.get('joint_probability_tight_remaining', 0):.2e}",
            "TEST")

        # ----------------------------------------------------------
        # Search-volume (look-elsewhere) correction.
        #
        # The per-pair probabilities price each observed angular
        # configuration; their product is the a priori improbability
        # of the full configuration set.  The catalogue was assembled
        # from a finite parent population of bright-galaxy fields
        # examined for discordant companions, so the honest ensemble
        # question under the null is the probability that a search of
        # N fields reproduces the observed severity profile by chance.
        # Each field is an independent Poisson trial: the expected
        # number of chance lookalikes of severity <= p_i in N fields
        # is mu_i = N p_i, and a chance catalogue matching every
        # observed severity class requires every slot to be filled
        # simultaneously,
        #
        #     P_cat(N) = prod_i [1 - exp(-N p_i)] .
        #
        # The correction is reported as a function of N so that no
        # assumption about the effective search volume is hidden.
        # N = 338 anchors the grid at the entry count of Arp's Atlas
        # of Peculiar Galaxies, the natural parent-sample scale for
        # the discordant-pair literature.
        p_list = [r["p_chance"] for r in rows]
        p_geom_list = [r["p_chance_axis_conditioned"] for r in rows]

        def _p_cat(nv, plist):
            return float(np.prod([1.0 - np.exp(-nv * p) for p in plist]))

        N_GRID = [10, 30, 50, 100, 200, 338, 1000, 10000]
        search_volume = []
        for n_par in N_GRID:
            search_volume.append({
                "n_parent_fields": int(n_par),
                "p_catalogue_by_chance": _p_cat(n_par, p_list),
                "p_catalogue_by_chance_axis_conditioned":
                    _p_cat(n_par, p_geom_list),
                "expected_lookalikes_per_pair": {
                    r["pair_id"]: float(n_par * r["p_chance"])
                    for r in rows
                },
            })
        # Largest parent search volume for which the ensemble remains
        # improbable at the 5% level; P_cat(N) is monotone increasing,
        # so bisection on the log scale converges to the boundary.
        def _n_crit(plist):
            lo, hi = 1.0, 1.0e8
            for _ in range(200):
                mid = np.sqrt(lo * hi)
                if _p_cat(mid, plist) < 0.05:
                    lo = mid
                else:
                    hi = mid
            return float(lo)

        n_crit_95 = _n_crit(p_list)
        n_crit_95_geom = _n_crit(p_geom_list)

        # ----------------------------------------------------------
        # On-structure coincidence: the NGC 7603 filament knots.
        #
        # The per-pair probabilities above price proximity to the
        # host.  A sharper question prices proximity to the
        # measured connecting structure itself: the two [LG2002]
        # emission knots sit on the detected filament polyline
        # (step_20), a corridor of length ~75 arcsec and a few
        # arcsec width — orders of magnitude smaller than the
        # pi*theta^2 pair circle.  The probability that two
        # independent background compact emission-line objects
        # land inside that corridor is Poisson with
        # lam = Sigma * L * w.  The knot population is
        # emission-line compact objects; Sigma is bracketed
        # between the measured confirmed-quasar-class density and
        # a generous faint emission-line-galaxy density, so the
        # reported value spans the plausible source populations.
        on_structure = None
        knots_path = self.data_processed / "filament_knot_positions.csv"
        comp_path = self.data_processed / "companion_positions.csv"
        if knots_path.exists() and comp_path.exists():
            from astropy.coordinates import SkyCoord
            knots = pd.read_csv(knots_path)
            comps = pd.read_csv(comp_path)
            knots7603 = knots[knots["pair_id"] == "NGC7603-NGC7603B"]
            comp7603 = comps[comps["pair_id"] == "NGC7603-NGC7603B"]
            host_row = catalog[catalog["pair_id"] == "NGC7603-NGC7603B"]
            if len(knots7603) and len(comp7603) and len(host_row):
                host = SkyCoord(
                    float(host_row["galaxy_ra_deg"].iloc[0]) * u.deg
                    if "galaxy_ra_deg" in host_row.columns else None,
                    float(host_row["galaxy_dec_deg"].iloc[0]) * u.deg
                    if "galaxy_dec_deg" in host_row.columns else None,
                ) if "galaxy_ra_deg" in host_row.columns else None
                if host is None:
                    from astroquery.simbad import Simbad
                    t = Simbad.query_object(str(host_row["galaxy"].iloc[0]))
                    host = SkyCoord(float(t[0]["ra"]) * u.deg,
                                    float(t[0]["dec"]) * u.deg)
                cb = comp7603.iloc[0]
                # Filament polyline: host -> knots (transect order)
                # -> companion.
                path_pts = [host]
                for _, k in knots7603.sort_values("x").iterrows():
                    path_pts.append(SkyCoord(
                        k["ra_deg"] * u.deg, k["dec_deg"] * u.deg))
                path_pts.append(SkyCoord(
                    cb["ra_deg"] * u.deg, cb["dec_deg"] * u.deg))
                length = float(sum(
                    a.separation(b).to(u.arcsec).value
                    for a, b in zip(path_pts[:-1], path_pts[1:])))
                n_knots = len(knots7603)
                widths = [2.0, 3.0, 4.0]      # arcsec corridor
                densities = {
                    # Same quasar-class surface density used for the
                    # angular statistic above (step_22 empirical
                    # measurement, falling back to the SDSS DR16Q
                    # published value); the corridor is not priced
                    # under a friendlier density than the rest of
                    # the sample.
                    "quasar_class_empirical": float(sigma_q),
                    "faint_elg_conservative": 500.0,
                    "deep_elg_extreme": 2000.0,
                }
                scen = {}
                for w in widths:
                    a_deg2 = length * w / 3600.0 ** 2
                    for dn, sv in densities.items():
                        lam = sv * a_deg2
                        from scipy.stats import poisson as _poi
                        p_geq = float(
                            1.0 - _poi.cdf(n_knots - 1, lam))
                        scen[f"w{w:.0f}_{dn}"] = {
                            "lambda": float(lam),
                            "p_at_least_n_knots_on_filament": p_geq,
                        }
                on_structure = {
                    "pair_id": "NGC7603-NGC7603B",
                    "n_knots_on_filament": n_knots,
                    "filament_path_length_arcsec": length,
                    "corridor_widths_arcsec": widths,
                    "surface_densities_deg2": densities,
                    "scenarios": scen,
                    "interpretation": (
                        "Poisson probability that >= the measured "
                        "number of emission knots land inside the "
                        "detected filament corridor by chance; the "
                        "conservative bracket (w=4 arcsec, Sigma=2000 "
                        "deg^-2) still prices the configuration at "
                        "~1e-3."
                    ),
                }
                # Full-configuration probability: the knots are
                # objects distinct from companion B, so the joint
                # NGC 7603 configuration prices the companion's
                # angular proximity AND the knots' corridor
                # occupancy as separate coincidences.  The ensemble
                # product inherits this through the 7603 angular
                # term times the corridor occupancy.
                p7603_ang = None
                for rr in rows:
                    if rr.get("pair_id") == "NGC7603-NGC7603B":
                        p7603_ang = rr.get("p_chance")
                if p7603_ang is not None and joint_p_emp is not None:
                    comb = {}
                    for w in widths:
                        for dn in densities:
                            pc = scen[f"w{w:.0f}_{dn}"][
                                "p_at_least_n_knots_on_filament"]
                            comb[f"w{w:.0f}_{dn}"] = {
                                "p_pair_angular": p7603_ang,
                                "p_corridor_occupancy": pc,
                                "p_ngc7603_configuration": p7603_ang * pc,
                                "p_ensemble_configuration": (
                                    joint_p_emp * pc
                                ),
                            }
                    on_structure["combined_configuration"] = comb
                    on_structure["interpretation"] += (
                        " The corridor was traced through the knot "
                        "positions, so these values price the "
                        "conditional configuration: given the "
                        "detected continuous filament (independently "
                        "supported by the g-band and DSS2 detections "
                        "and the PSF-excluded corridor excess), the "
                        "probability that unrelated background "
                        "emitters also furnish the corridor's two "
                        "brightest emission objects."
                    )
                print_status(
                    f"On-structure coincidence: {n_knots} knots on a "
                    f"{length:.0f} arcsec filament corridor; "
                    f"P(>=2) ranges {scen['w2_quasar_class_empirical']['p_at_least_n_knots_on_filament']:.1e} "
                    f"(quasar-class) to "
                    f"{scen['w4_deep_elg_extreme']['p_at_least_n_knots_on_filament']:.1e} "
                    "(deep-ELG extreme)", "TEST",
                )

        summary = {
            "sigma_qso_deg2": sigma_q,
            "sigma_qso_source": sigma_src,
            "sigma_qso_sdss_deg2": SIGMA_QSO_DEG2,
            "sigma_qso_empirical_deg2": empirical_density,
            "sigma_gal_deg2": SIGMA_GAL_DEG2,
            "n_pairs": len(df),
            "joint_probability_all_pairs": float(joint_p),
            "joint_log10_p": float(joint_log10),
            "joint_probability_sdss_density": float(joint_p_sdss),
            "joint_log10_p_sdss_density": float(joint_log10_sdss),
            "joint_probability_empirical_density": (
                float(joint_p_emp) if joint_p_emp is not None else None
            ),
            "joint_log10_p_empirical_density": (
                float(joint_log10_emp) if joint_log10_emp is not None else None
            ),
            "joint_probability_axis_conditioned": float(joint_p_geom),
            "joint_log10_p_axis_conditioned": float(joint_log10_geom),
            "tight_subset": {
                "description": (
                    "Joint product restricted to pairs whose "
                    "nucleus-companion separation is below 300 arcsec "
                    "- the subsample for which angular proximity is "
                    "the operative statistic. The wide-field "
                    "symmetric/jet systems (theta >= 300 arcsec) "
                    "carry their evidence in structure and selection "
                    "rather than proximity and enter the full product "
                    "at weights near unity; their joint factor is "
                    "reported separately so the headline number is "
                    "not read as a proximity claim about them."
                ),
                "separation_threshold_arcsec": TIGHT_SEP_ARCSEC,
                "n_tight": len(tight_rows),
                "tight_pair_ids": [r["pair_id"] for r in tight_rows],
                "wide_pair_ids": [r["pair_id"] for r in wide_rows],
                "joint_probability_tight": float(joint_p_tight),
                "joint_log10_p_tight": float(joint_log10_tight),
                "wide_field_joint_factor": float(10.0**log10_wide_factor),
            },
            "axis_conditioning_model": {
                "description": (
                    "For pairs whose published claim places companions on "
                    "a host-morphology axis (minor axis or jet axis), the "
                    "target direction is fixed by the host's own light "
                    "distribution before any companion position is "
                    "examined. The probability is therefore priced "
                    "against the axis corridor — a +/-10 deg azimuthal "
                    "wedge about each axis direction, f = 1/9 of the "
                    "annulus — rather than the full pair circle. "
                    "Published alignment tolerances are tighter than "
                    "10 deg, so the factor is conservative. Only "
                    "connection_type == 'axis_alignment' pairs are "
                    "weighted; bridges/filaments are excluded because "
                    "their direction is partially defined by the "
                    "companion itself."
                ),
                "wedge_half_angle_deg": AXIS_WEDGE_HALF_DEG,
                "geometry_factor": F_AXIS,
                "n_axis_pairs": n_axis,
            },
            "pairs": rows,
            "leave_one_out": leave_one_out,
            "on_structure_coincidence": on_structure,
            "search_volume_correction": {
                "description": (
                    "Look-elsewhere correction to the ensemble statistic. "
                    "Under the chance hypothesis each of N parent "
                    "bright-galaxy fields is an independent Poisson "
                    "trial; mu_i = N p_i is the expected number of "
                    "chance lookalikes of pair i, and P_cat(N) = "
                    "prod_i [1 - exp(-N p_i)] is the probability that "
                    "the search reproduces every observed severity "
                    "class simultaneously. N = 338 anchors the grid "
                    "at the entry count of Arp's Atlas of Peculiar "
                    "Galaxies. The angular statistic alone decides the "
                    "ensemble only for search volumes below "
                    "n_crit_95pct; above it the conjunction with the "
                    "model-independent ordering evidence carries the "
                    "argument."
                ),
                "n_grid": search_volume,
                "n_crit_95pct": n_crit_95,
                "n_crit_95pct_axis_conditioned": n_crit_95_geom,
                "arp_atlas_entries": 338,
            },
            "interpretation": (
                "Under the chance-superposition hypothesis each pair is an "
                "independent draw. The joint probability that all "
                f"{len(catalog)} systems arise by chance is the product of the individual "
                "probabilities, evaluated under both the published SDSS "
                "DR16 quasar density and the empirical density measured in "
                "the same fields by the association test; the headline "
                "value is the more conservative of the two."
            ),
        }
        json_path = self.results / "step_23_chance_alignment.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")
        print_status(
            f"Joint chance-alignment probability: {joint_p:.3e} "
            f"(log10 P = {joint_log10:.1f})", "TEST",
        )
        print_status("Chance-alignment analysis complete.", "SUCCESS")
