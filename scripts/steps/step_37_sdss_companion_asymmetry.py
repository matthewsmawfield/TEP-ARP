#!/usr/bin/env python3
"""
Step 37: SDSS-Scale Companion Redshift Asymmetry
==============================================
Scales the Arp-Sulentic companion redshift-excess test (step 25,
2.8k Tully nests) to the Tempel et al. (2017) SDSS DR10 flux-
limited group catalogue (VizieR J/A+A/602/A100): ~5.9e5 galaxies
in ~9e4 groups.  Under the symmetric-bound-satellite null the
distribution of dV = cz_sat - cz_cen is sign-symmetric; under the
TEP proper-time-well picture companions sitting in shallower wells
than their dominant galaxy carry systematically positive offsets.

This is the test the discordant-pair catalogue cannot be accused of
selecting: group membership here is assigned by a FoF algorithm
blind to the hypothesis under test.

Controls implemented, none assumed:

  1. Interloper baseline.  Satellites at |dV| > 2000 km/s cannot be
     bound to a group with R200 ~ 1 Mpc; their sign asymmetry is a
     direct measurement of the survey selection function (rising
     field density vs dust obscuration of background members).
     The bound-regime statistic is tested against this measured
     baseline, not against p0 = 0.5.

  2. Dust/extinction control (Girardi et al. 1993 alternative).
     If the positive asymmetry were produced by dust dimming
     background satellites behind the dominant galaxy, the effect
     must (a) concentrate at projected radii within the host disk
     scale (~tens of kpc), and (b) preferentially cull the faint
     background population, so surviving positive-dV satellites at
     small R_proj should be systematically brighter relative to the
     central than matched negative-dV satellites.  Both signatures
     are measured: f_+ vs R_proj (in R200 units and in Mpc) and
     the rmag-difference asymmetry within matched |dV| and R_proj
     bins.

  3. Reference-frame check.  Offsets are recomputed against the
     group's luminosity-weighted mean velocity as well as the
     rank-1 central, quantifying construction bias.

  4. Volume-limited control.  Satellites are restricted to
     (z_max, M_r <= M_lim(z_max)) windows in which the r < 17.77
     flux limit cannot preferentially cull far-side members.

  5. Mock-interloper forward model.  The observed dV histogram is
     fitted with a bound-Gaussian plus interloper-component model
     (uniform and broad-Gaussian variants); the fitted interloper
     mass inside the bound window, carrying the measured tail
     sign rate, is Monte-Carlo-injected into a symmetric bound
     population to predict the f_+ a pure-contamination origin
     would produce.

  6. Amplitude scaling.  Under TEP the companion offset is the
     depth of the emitter's own well, so the asymmetry amplitude
     should scale with satellite luminosity (compactness proxy).
     f_+ and mean signed dV are measured in absolute-magnitude
     and central-relative-magnitude bins.

Velocity offsets use dV = c(z_sat - z_cen)/(1+z_cen).  Projected
separations use the group comoving distance D_c: R_proj =
theta * D_c/(1+z).  Bootstrap confidence intervals use a fixed
seed.  Requires VizieR network access; fails loudly otherwise.

Outputs:
    data/processed/sdss_satellite_redshift_offsets.csv
    results/outputs/step_37_sdss_companion_asymmetry.json
    results/figures/step_37_sdss_companion_asymmetry.png
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

CATALOG = "J/A+A/602/A100"   # Tempel+2017 SDSS DR10 groups
C_KMS = 299792.458
N_GAL_MIN = 2                # pairs included; N>=3 reported too
DV_INTERLOPER = 2000.0       # km/s: surely-unbound reference tail
DV_BOUND = 500.0             # km/s: bound-member cut
RNG_SEED = 20261108
N_BOOT = 2000


class Step37SdssCompanionAsymmetry:
    """Step 37: companion redshift asymmetry at SDSS scale."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.data_raw = self.root / "data" / "raw"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"
        for d in [self.data_processed, self.data_raw, self.results,
                  self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)
        self.logger = TEPLogger(
            "step_37",
            log_file_path=self.logs
            / "step_37_sdss_companion_asymmetry.log")
        set_step_logger(self.logger)

    # --------------------------------------------------------------
    def _load_catalog(self):
        gcache = self.data_raw / "tempel2017_galaxies.csv"
        pcache = self.data_raw / "tempel2017_groups.csv"
        if gcache.exists() and pcache.exists():
            return pd.read_csv(gcache), pd.read_csv(pcache)

        from astroquery.vizier import Vizier
        viz = Vizier(row_limit=-1, timeout=600,
                     columns=["GroupID", "Ngal", "Rank", "zobs",
                              "Dist", "RAJ2000", "DEJ2000",
                              "rmag", "gmag"])
        res = viz.get_catalogs(CATALOG)
        if not res:
            raise RuntimeError(
                f"Tempel+2017 catalogue {CATALOG} unreachable on "
                "VizieR.")
        tab = res["J/A+A/602/A100/table1"]
        gal = pd.DataFrame({
            c: np.asarray(tab[c]) for c in
            ("GroupID", "Ngal", "Rank", "zobs", "Dist",
             "RAJ2000", "DEJ2000", "rmag", "gmag")})

        viz2 = Vizier(row_limit=-1, timeout=300,
                      columns=["GroupID", "Ngal", "RAJ2000",
                               "DEJ2000", "Dist.c", "R200", "M200"])
        res2 = viz2.get_catalogs(CATALOG)
        tab2 = res2["J/A+A/602/A100/table2"]
        grp = pd.DataFrame({
            "GroupID": np.asarray(tab2["GroupID"]),
            "Ngal": np.asarray(tab2["Ngal"]),
            "RAJ2000": np.asarray(tab2["RAJ2000"], dtype=float),
            "DEJ2000": np.asarray(tab2["DEJ2000"], dtype=float),
            "Distc": np.asarray(tab2["Dist.c"], dtype=float),
            "R200": np.asarray(tab2["R200"], dtype=float),
            "M200": np.asarray(tab2["M200"], dtype=float),
        })
        gal.to_csv(gcache, index=False)
        grp.to_csv(pcache, index=False)
        return gal, grp

    # --------------------------------------------------------------
    def run(self):
        print_status(
            "Building SDSS-scale satellite offset sample...", "PROCESS")
        gal, grp = self._load_catalog()
        gal = gal.dropna(subset=["zobs", "Rank", "rmag"])
        grp = grp.set_index("GroupID")
        print_status(
            f"Tempel+2017: {len(grp)} groups, {len(gal)} galaxies.",
            "INFO")

        # rank-1 = central (brightest); where a group carries tied
        # rank-1 rows, take the brightest in r.  Isolated galaxies
        # (Ngal=1) also carry Rank=1; they drop out when satellites
        # are restricted to groups present in the group table.
        n_cen = int((gal["Rank"] == 1).sum())
        if n_cen < 10000:
            raise RuntimeError(
                f"Only {n_cen} rank-1 centrals found; catalogue "
                "schema unexpected.")
        print_status(f"{n_cen} rank-1 rows verified.", "INFO")

        cen = (gal[gal["Rank"] == 1]
               .sort_values("rmag")
               .drop_duplicates("GroupID")
               .set_index("GroupID"))
        sat = gal[gal["Rank"] > 1].copy()
        sat = sat[sat["GroupID"].isin(cen.index)]
        sat = sat[sat["Ngal"] >= N_GAL_MIN]

        c_z = cen["zobs"].reindex(sat["GroupID"]).to_numpy()
        c_ra = cen["RAJ2000"].reindex(sat["GroupID"]).to_numpy()
        c_de = cen["DEJ2000"].reindex(sat["GroupID"]).to_numpy()
        c_rm = cen["rmag"].reindex(sat["GroupID"]).to_numpy()
        c_gm = cen["gmag"].reindex(sat["GroupID"]).to_numpy()
        g_dc = grp["Distc"].reindex(sat["GroupID"]).to_numpy()
        g_r200 = grp["R200"].reindex(sat["GroupID"]).to_numpy()
        g_m200 = grp["M200"].reindex(sat["GroupID"]).to_numpy()

        sat["dv_kms"] = C_KMS * (sat["zobs"].to_numpy() - c_z) \
            / (1.0 + c_z)
        # angular separation via haversine-free small-angle form
        d_ra = (sat["RAJ2000"].to_numpy() - c_ra) \
            * np.cos(np.radians(sat["DEJ2000"].to_numpy()))
        d_de = sat["DEJ2000"].to_numpy() - c_de
        sat["theta_arcsec"] = np.hypot(d_ra, d_de) * 3600.0
        sat["rproj_mpc"] = np.radians(np.hypot(d_ra, d_de)) \
            * g_dc / (1.0 + c_z)
        sat["rproj_r200"] = sat["rproj_mpc"] / g_r200
        # virial-scaled velocity: sigma_200 ~ sqrt(G M200 / R200)
        # with G = 4.302e-3 Mpc Msun^-1 (km/s)^2
        sig200 = np.sqrt(4.302e-3 * g_m200 / np.maximum(g_r200, 0.05))
        sat["sig200_kms"] = sig200
        sat["dv_sig"] = sat["dv_kms"] / np.maximum(sig200, 30.0)
        sat["drmag"] = sat["rmag"].to_numpy() - c_rm
        sat["g_r_color"] = sat["gmag"].to_numpy() \
            - sat["rmag"].to_numpy()
        sat["cen_g_r"] = c_gm - c_rm

        # luminosity-weighted mean velocity reference (control):
        # the group's luminosity-weighted mean redshift replaces the
        # rank-1 central's, testing central misidentification
        lum_w = 10.0 ** (-0.4 * sat["rmag"].to_numpy())
        sat["lumw"] = lum_w
        sat["zlumw"] = sat["zobs"].to_numpy() * lum_w
        cen_lum = 10.0 ** (-0.4 * c_rm)
        wsum = sat.groupby("GroupID")["lumw"].transform("sum") + \
            cen_lum
        zsum = sat.groupby("GroupID")["zlumw"].transform("sum") + \
            c_z * cen_lum
        sat["z_lumw"] = zsum / wsum
        sat["dv_lumw_kms"] = C_KMS * (sat["zobs"].to_numpy()
                                      - sat["z_lumw"]) \
            / (1.0 + sat["z_lumw"])
        sat = sat.drop(columns=["lumw", "zlumw", "z_lumw"])

        # absolute magnitude: Dist is the group comoving distance
        # (Tempel+2017); luminosity distance is D_L = D_C (1+z)
        sat["M_r"] = sat["rmag"] - 5.0 * np.log10(
            sat["Dist"].to_numpy() * (1.0 + sat["zobs"].to_numpy())) \
            - 25.0

        df = sat.reset_index(drop=True)
        df.to_csv(
            self.data_processed / "sdss_satellite_redshift_offsets.csv",
            index=False)
        print_status(
            f"{len(df)} satellites in {df['GroupID'].nunique()} "
            "groups with a rank-1 central.", "INFO")

        from scipy.stats import binomtest, wilcoxon
        rng = np.random.default_rng(RNG_SEED)

        def _boot_frac(v):
            """Seeded bootstrap 95% CI on the positive fraction."""
            n = len(v)
            if n < 10:
                return (np.nan, np.nan)
            pos = (v > 0).astype(int).to_numpy()
            draws = rng.integers(0, n, (N_BOOT, n))
            f = pos[draws].mean(axis=1)
            return (float(np.percentile(f, 2.5)),
                    float(np.percentile(f, 97.5)))

        def _stats(sub, label, p0=0.5):
            v = sub["dv_kms"]
            n_pos = int((v > 0).sum())
            n_tot = int(((v > 0) | (v < 0)).sum())
            lo, hi = _boot_frac(v)
            return {
                "cut": label,
                "n_satellites": n_tot,
                "n_positive": n_pos,
                "frac_positive": n_pos / n_tot if n_tot else np.nan,
                "frac_positive_ci95": [lo, hi],
                "median_dv_kms": float(v.median()) if n_tot else np.nan,
                "binomial_p_vs_half": float(
                    binomtest(n_pos, n_tot, 0.5).pvalue)
                if n_tot else np.nan,
                "binomial_p_vs_interloper": float(
                    binomtest(n_pos, n_tot, p0).pvalue)
                if n_tot else np.nan,
            }

        # ---- empirical selection baseline from the unbound tail ---
        tail = df[np.abs(df["dv_kms"]) > DV_INTERLOPER]
        f_tail = float((tail["dv_kms"] > 0).mean())
        n_tail = len(tail)
        print_status(
            f"Interloper baseline (|dV|>{DV_INTERLOPER:.0f} km/s): "
            f"f_+ = {f_tail:.4f} on n = {n_tail}.", "TEST")

        p0 = f_tail
        stats = [
            _stats(df, "all_offsets", p0),
            _stats(df[np.abs(df["dv_kms"]) < DV_BOUND],
                   f"bound |dV|<{DV_BOUND:.0f}", p0),
            _stats(df[np.abs(df["dv_kms"]) < 150.0],
                   "tightly bound |dV|<150", p0),
            _stats(df[np.abs(df["dv_kms"]) > 30.0],
                   "|dV|>30", p0),
            _stats(tail, f"interlopers |dV|>{DV_INTERLOPER:.0f}", 0.5),
        ]

        dv_bands = [(30, 100), (100, 300), (300, 500),
                    (500, 1000), (1000, 2000), (2000, 6000)]
        bands = []
        for lo, hi in dv_bands:
            sub = df[np.abs(df["dv_kms"]).between(lo, hi)]
            if len(sub) < 200:
                continue
            bands.append(_stats(sub, f"|dV| {lo}-{hi} km/s", p0))

        # ---- virial-normalized bands: the physical bound/interloper
        # discriminator (|dV|/sigma_200).  Sigma_200 from M200/R200.
        sig_bands = [(0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, 2.0),
                     (2.0, 4.0), (4.0, 100.0)]
        sigstats = []
        for lo, hi in sig_bands:
            sub = df[np.abs(df["dv_sig"]).between(lo, hi)]
            if len(sub) < 200:
                continue
            s = _stats(sub, f"|dV|/sig200 {lo}-{hi}", p0)
            s["sig_lo"], s["sig_hi"] = lo, hi
            # implied interloper fraction if all asymmetry came from
            # unbound contamination with f_+ = f_tail:
            denom = f_tail - 0.5
            s["implied_interloper_frac"] = (
                (s["frac_positive"] - 0.5) / denom
                if abs(denom) > 1e-6 and s["n_satellites"] else np.nan)
            sigstats.append(s)
        for s in sigstats:
            print_status(
                f"{s['cut']}: f_+ = {s['frac_positive']:.4f} "
                f"(n={s['n_satellites']}), implied interloper frac = "
                f"{s['implied_interloper_frac']:.2f}", "TEST")

        # ---- dust control 1: f_+ vs projected separation ----------
        bound = df[np.abs(df["dv_kms"]) < DV_BOUND]
        r_bins_mpc = [(0, 0.05), (0.05, 0.15), (0.15, 0.3),
                      (0.3, 0.6), (0.6, 1.5)]
        rproj = []
        for lo, hi in r_bins_mpc:
            sub = bound[bound["rproj_mpc"].between(lo, hi)]
            if len(sub) < 200:
                continue
            s = _stats(sub, f"R_proj {lo}-{hi} Mpc", p0)
            s["rproj_lo_mpc"] = lo
            s["rproj_hi_mpc"] = hi
            rproj.append(s)
        r200_bins = [(0, 0.1), (0.1, 0.3), (0.3, 0.6), (0.6, 1.0),
                     (1.0, 2.0)]
        rproj200 = []
        for lo, hi in r200_bins:
            sub = bound[bound["rproj_r200"].between(lo, hi)]
            if len(sub) < 200:
                continue
            s = _stats(sub, f"R_proj {lo}-{hi} R200", p0)
            s["r200_lo"] = lo
            s["r200_hi"] = hi
            rproj200.append(s)

        # ---- dust control 2: magnitude-difference asymmetry -------
        # Under dust dimming, observed +dV satellites at small R_proj
        # should be systematically brighter vs the central than
        # matched -dV satellites.
        mag_ctrl = []
        inner = bound[bound["rproj_mpc"] < 0.15]
        for dlo, dhi in [(30, 150), (150, 300), (300, 500)]:
            p = inner[(inner["dv_kms"] > dlo)
                      & (inner["dv_kms"] < dhi)]
            m = inner[(inner["dv_kms"] < -dlo)
                      & (inner["dv_kms"] > -dhi)]
            if len(p) < 100 or len(m) < 100:
                continue
            mag_ctrl.append({
                "dv_band_kms": [dlo, dhi],
                "rproj_cut_mpc": 0.15,
                "n_pos": len(p), "n_neg": len(m),
                "mean_drmag_pos": float(p["drmag"].mean()),
                "mean_drmag_neg": float(m["drmag"].mean()),
                "diff_pos_minus_neg": float(
                    p["drmag"].mean() - m["drmag"].mean()),
                "mean_gr_pos": float(p["g_r_color"].mean()),
                "mean_gr_neg": float(m["g_r_color"].mean()),
            })

        # ---- group-richness split ---------------------------------
        richness = [
            _stats(df[df["Ngal"] == 2], "pairs Ngal=2", p0),
            _stats(df[df["Ngal"].between(3, 9)], "Ngal 3-9", p0),
            _stats(df[df["Ngal"] >= 10], "Ngal>=10", p0),
        ]

        # ---- per-group mean offset ---------------------------------
        gmean = df[np.abs(df["dv_kms"]) < DV_BOUND].groupby(
            "GroupID")["dv_kms"].mean()
        gmean = gmean[gmean.notna()]
        try:
            p_grp = float(wilcoxon(
                gmean.to_numpy(), alternative="greater").pvalue)
        except Exception:
            p_grp = np.nan

        # ---- luminosity-weighted reference control -----------------
        dfw = df[np.abs(df["dv_lumw_kms"]) < DV_BOUND]
        lumw = _stats(
            dfw.assign(dv_kms=dfw["dv_lumw_kms"]),
            "lumw-mean reference, |dV|<500", p0)

        # ---- control 4: volume-limited subsamples ------------------
        # M_r completeness limit at each z_max under the SDSS main
        # sample flux bound r < 17.77; margin 0.05 mag for k-term.
        vol_lim = []
        for z_max in (0.06, 0.08, 0.10, 0.12):
            near = df[np.abs(df["zobs"] - z_max) < 0.01]
            if len(near) < 100:
                continue
            d_edge = float(near["Dist"].max()) * (1.0 + z_max)
            m_lim = 17.77 - 5.0 * np.log10(d_edge) - 25.0
            m_cut = np.floor((m_lim - 0.05) * 10.0) / 10.0
            sub = df[(df["zobs"] <= z_max) & (df["M_r"] <= m_cut)]
            sub_b = sub[np.abs(sub["dv_kms"]) < DV_BOUND]
            if len(sub_b) < 200:
                continue
            s = _stats(sub_b,
                       f"vol-limited z<{z_max}, M_r<{m_cut}", p0)
            s["z_max"], s["M_r_cut"] = z_max, float(m_cut)
            s["completeness_M_lim"] = float(m_lim)
            s["n_all_offsets"] = int(len(sub))
            vol_lim.append(s)
            print_status(
                f"vol-limited z<{z_max} M_r<{m_cut}: f_+ = "
                f"{s['frac_positive']:.4f} (n={s['n_satellites']}), "
                f"p(0.5)={s['binomial_p_vs_half']:.3g}", "TEST")

        # ---- control 5: mock-interloper forward model -------------
        # Shape decomposition of the dV histogram is degenerate, so
        # contamination is measured geometrically: FoF interlopers
        # are field/correlated-structure galaxies whose catalogued
        # surface density is measured in the interloper-dominated
        # annulus r = 3-10 R200 at the same |dV| < 500 window, then
        # scaled by area into r < R200.  Two radial profiles bracket
        # the unknown central concentration: uniform (far-region
        # mean density) and 1/r (filament-projected, strongly
        # centrally concentrated).  A third estimator extrapolates
        # the |dV| 2000-3600 tail flat in velocity.  Each is
        # Monte-Carlo-injected at its measured sign rate into a
        # symmetric bound population of the observed size.
        bound_mask = np.abs(df["dv_kms"]) < DV_BOUND
        n_bound = int(bound_mask.sum())
        n_pos_obs = int((df.loc[bound_mask, "dv_kms"] > 0).sum())
        f_obs = n_pos_obs / n_bound

        far = df[df["rproj_r200"].between(3.0, 10.0)]
        far_bound = far[np.abs(far["dv_kms"]) < DV_BOUND]
        n_far = len(far_bound)
        f_far = float((far_bound["dv_kms"] > 0).mean())
        area_factor_flat = 1.0 / (10.0 ** 2 - 3.0 ** 2)
        area_factor_1overr = 1.0 / (10.0 - 3.0)   # rho ~ 1/r
        # velocity-tail estimator: interlopers flat in dV across the
        # catalogue's +/-3600 km/s linking window
        tail_v = df[(np.abs(df["dv_kms"]) > 2000.0)
                    & (df["rproj_r200"] < 1.0)]
        n_int_vel = len(tail_v) * (2 * DV_BOUND) \
            / (2 * (3600.0 - 2000.0))

        interloper_models = {}
        for name, c_meas, f_int in (
                ("geometric_uniform_far_region",
                 n_far * area_factor_flat / n_bound, f_far),
                ("geometric_1overr_central_concentration",
                 n_far * area_factor_1overr / n_bound, f_far),
                ("velocity_tail_flat",
                 n_int_vel / n_bound, f_tail)):
            f_pred = 0.5 * (1 - c_meas) + f_int * c_meas
            n_inj = int(round(c_meas * n_bound))
            f_mc = []
            for _ in range(500):
                s_b = rng.choice([-1, 1], n_bound - n_inj)
                s_i = (rng.choice([-1, 1], n_inj,
                                  p=[1 - f_int, f_int])
                       if n_inj else np.empty(0, int))
                f_mc.append((np.concatenate([s_b, s_i]) > 0).mean())
            f_mc = np.asarray(f_mc)
            c_needed = (f_obs - 0.5) / (f_int - 0.5) \
                if abs(f_int - 0.5) > 1e-6 else np.nan
            interloper_models[name] = {
                "interloper_frac_in_bound": float(c_meas),
                "interloper_sign_rate": float(f_int),
                "f_plus_predicted_pure_contamination": float(f_pred),
                "f_plus_mc_p99": float(np.percentile(f_mc, 99)),
                "f_plus_observed": float(f_obs),
                "contamination_fraction_needed": float(c_needed),
                "shortfall_factor": float(c_needed / c_meas)
                if c_meas > 0 else np.nan,
                "observed_exceeds_mc_p99": bool(
                    f_obs > np.percentile(f_mc, 99)),
            }
            print_status(
                f"mock interlopers ({name}): c_bound = "
                f"{c_meas:.5f} (sign rate {f_int:.4f}), predicted "
                f"f_+ = {f_pred:.4f} vs observed {f_obs:.4f}; "
                f"needed c = {c_needed:.3f}, shortfall "
                f"{c_needed / c_meas:.0f}x", "TEST")
        interloper_models["far_region_reference"] = {
            "region": "3-10 R200, |dV|<500 km/s",
            "n": int(n_far),
            "frac_positive": f_far,
            "note": ("measured sign rate of interloper-dominated "
                     "correlated-structure members at bound "
                     "velocities; the extreme-tail baseline 0.531 "
                     "does not extend to |dV|<500"),
        }

        # ---- control 6: amplitude scaling --------------------------
        # TEP-specific: the offset is the emitter's own well depth,
        # so the asymmetry amplitude should scale with satellite
        # luminosity (and, differentially, with drmag).
        bound_all = df[np.abs(df["dv_kms"]) < DV_BOUND]
        lum_bins = [(-99, -21.5), (-21.5, -21.0), (-21.0, -20.5),
                    (-20.5, -20.0), (-20.0, -19.5), (-19.5, -19.0),
                    (-19.0, -18.0)]
        lum_scale = []
        for lo, hi in lum_bins:
            sub = bound_all[
                bound_all["M_r"].between(lo, hi)]
            if len(sub) < 300:
                continue
            s = _stats(sub, f"M_r {lo} to {hi}", p0)
            s["M_r_lo"], s["M_r_hi"] = lo, hi
            s["mean_dv_kms"] = float(sub["dv_kms"].mean())
            s["mean_dv_se_kms"] = float(
                sub["dv_kms"].std() / np.sqrt(len(sub)))
            lum_scale.append(s)
        drm_bins = [(0, 1.0), (1.0, 2.0), (2.0, 3.0), (3.0, 20.0)]
        drm_scale = []
        for lo, hi in drm_bins:
            sub = bound_all[
                bound_all["drmag"].between(lo, hi)]
            if len(sub) < 300:
                continue
            s = _stats(sub, f"drmag {lo}-{hi}", p0)
            s["drmag_lo"], s["drmag_hi"] = lo, hi
            s["mean_dv_kms"] = float(sub["dv_kms"].mean())
            s["mean_dv_se_kms"] = float(
                sub["dv_kms"].std() / np.sqrt(len(sub)))
            drm_scale.append(s)
        for s in lum_scale:
            print_status(
                f"amp scaling {s['cut']}: f_+ = "
                f"{s['frac_positive']:.4f}, <dV> = "
                f"{s['mean_dv_kms']:+.2f} km/s "
                f"(n={s['n_satellites']})", "TEST")
        for s in drm_scale:
            print_status(
                f"amp scaling {s['cut']}: f_+ = "
                f"{s['frac_positive']:.4f}, <dV> = "
                f"{s['mean_dv_kms']:+.2f} km/s "
                f"(n={s['n_satellites']})", "TEST")

        # ---- control 7: spectroscopic redshift-bias channel -------
        # The remaining systematic: satellite spectra are fainter
        # than rank-1 centrals, so a magnitude/S/N-dependent
        # pipeline bias could mimic a positive sign asymmetry.
        # Internal proxies first (no network), then the decisive
        # per-object redshift-quality fields from SDSS DR16
        # (VizieR V/154/sdss16) via CDS XMatch.
        spec_ctrl = {}
        # 7a. apparent-magnitude proxy: a S/N-driven bias grows at
        # the faint end
        rmag_bins = [(10, 16.5), (16.5, 17.0), (17.0, 17.3),
                     (17.3, 17.6), (17.6, 17.77)]
        spec_ctrl["f_plus_vs_apparent_rmag"] = [
            _stats(bound_all[bound_all["rmag"].between(lo, hi)],
                   f"rmag {lo}-{hi}", p0)
            for lo, hi in rmag_bins
            if len(bound_all[bound_all["rmag"].between(lo, hi)])
            >= 200]
        # 7b. brightness-matched subset: at drmag ~ 0 the satellite
        # and central carry equal S/N and template weight, so a
        # differential pipeline bias vanishes by construction
        spec_ctrl["f_plus_vs_drmag_fine"] = [
            _stats(bound_all[bound_all["drmag"].between(lo, hi)],
                   f"drmag {lo}-{hi}", p0)
            for lo, hi in [(-1, 0), (0, 0.2), (0.2, 0.5),
                           (0.5, 1.0), (1.0, 2.0), (2.0, 20.0)]
            if len(bound_all[bound_all["drmag"].between(lo, hi)])
            >= 200]
        # 7c. spectral-type proxy: template bias would separate by
        # colour (absorption- vs emission-line dominated)
        spec_ctrl["f_plus_vs_g_minus_r"] = [
            _stats(bound_all[bound_all["g_r_color"].between(lo, hi)],
                   f"g-r {lo}-{hi}", p0)
            for lo, hi in [(0, 0.65), (0.65, 0.95), (0.95, 3.0)]
            if len(bound_all[bound_all["g_r_color"].between(lo, hi)])
            >= 200]

        # 7d. per-object SDSS redshift quality via XMatch
        qcache = (self.data_processed
                  / "sdss_satellite_specquality.csv")
        if qcache.exists():
            qm = pd.read_csv(qcache)
        else:
            try:
                import time
                from astropy.table import Table
                from astroquery.xmatch import XMatch
                import astropy.units as u
                up = bound_all[["RAJ2000", "DEJ2000"]].copy()
                up.insert(0, "idx", np.arange(len(up)))
                up.columns = ["idx", "ra", "dec"]
                # XMatch rejects very large uploads; chunk
                chunks = []
                step_n = 20000
                for s0 in range(0, len(up), step_n):
                    part = up.iloc[s0:s0 + step_n]
                    for attempt in range(3):
                        try:
                            xm = XMatch.query(
                                cat1=Table.from_pandas(part),
                                cat2="vizier:V/154/sdss16",
                                max_distance=2.0 * u.arcsec,
                                colRA1="ra", colDec1="dec")
                            chunks.append(xm.to_pandas())
                            break
                        except Exception:
                            if attempt == 2:
                                raise
                            time.sleep(20)
                    print_status(
                        f"XMatch chunk {s0 // step_n + 1}: "
                        f"{len(part)} rows done", "INFO")
                xmd = pd.concat(chunks, ignore_index=True)
                xmd = (xmd.sort_values("angDist")
                          .drop_duplicates("idx"))
                qm = xmd[["idx", "e_zsp", "f_zsp", "spCl",
                          "angDist"]].copy()
                qm.to_csv(qcache, index=False)
            except Exception as exc:
                qm = None
                spec_ctrl["xmatch_error"] = str(exc)
                print_status(
                    f"spec-quality XMatch failed: {exc}", "ERROR")

        if qm is not None and len(qm):
            m = bound_all.reset_index(drop=True).merge(
                qm, left_index=True, right_on="idx", how="left")
            m["matched"] = m["e_zsp"].notna()
            n_mat = int(m["matched"].sum())
            spec_ctrl["xmatch_n"] = n_mat
            spec_ctrl["xmatch_rate"] = float(n_mat / len(m))
            spec_ctrl["xmatch_rate_pos"] = float(
                m.loc[m["dv_kms"] > 0, "matched"].mean())
            spec_ctrl["xmatch_rate_neg"] = float(
                m.loc[m["dv_kms"] < 0, "matched"].mean())
            mm = m[m["matched"] & m["e_zsp"].notna()].copy()
            mm["e_zsp"] = mm["e_zsp"].astype(float)
            # e_zsp <= 0 is the pipeline no-redshift sentinel;
            # count it (pos vs neg) then exclude from quantiles
            spec_ctrl["e_zsp_sentinel_frac_pos"] = float(
                (mm.loc[mm["dv_kms"] > 0, "e_zsp"] <= 0).mean())
            spec_ctrl["e_zsp_sentinel_frac_neg"] = float(
                (mm.loc[mm["dv_kms"] < 0, "e_zsp"] <= 0).mean())
            mm = mm[mm["e_zsp"] > 0]
            # f_+ in redshift-error quartiles: a pipeline bias
            # concentrated in poor redshifts would track e_zsp
            ez_bins = np.nanpercentile(
                mm["e_zsp"], [0, 25, 50, 75, 100])
            ez_rows = []
            for lo, hi in zip(ez_bins[:-1], ez_bins[1:]):
                sub = mm[mm["e_zsp"].between(lo, hi)]
                if len(sub) < 200:
                    continue
                s = _stats(sub, f"e_zsp {lo:.2e}-{hi:.2e}", p0)
                s["e_zsp_lo"], s["e_zsp_hi"] = float(lo), float(hi)
                ez_rows.append(s)
            spec_ctrl["f_plus_vs_e_zsp"] = ez_rows
            # clean subsample: no warning flag, error below median
            clean = mm[(mm["f_zsp"].astype(float) == 0)
                       & (mm["e_zsp"] <= np.nanmedian(mm["e_zsp"]))]
            spec_ctrl["clean_subsample"] = _stats(
                clean, "clean: f_zsp=0 & e_zsp<=median", p0)
            # error distributions of + vs - satellites at fixed
            # apparent magnitude
            ezpm = []
            for lo, hi in [(16.0, 17.0), (17.0, 17.3),
                           (17.3, 17.6), (17.6, 17.77)]:
                sub = mm[mm["rmag"].between(lo, hi)]
                p_, n_ = sub[sub["dv_kms"] > 0], \
                    sub[sub["dv_kms"] < 0]
                if len(p_) < 200 or len(n_) < 200:
                    continue
                ezpm.append({
                    "rmag_bin": [lo, hi],
                    "median_e_zsp_pos": float(p_["e_zsp"].median()),
                    "median_e_zsp_neg": float(n_["e_zsp"].median()),
                    "warnrate_pos": float(
                        (p_["f_zsp"].astype(float) != 0).mean()),
                    "warnrate_neg": float(
                        (n_["f_zsp"].astype(float) != 0).mean()),
                })
            spec_ctrl["e_zsp_and_warnrate_pos_vs_neg"] = ezpm
            spec_ctrl["spclass_frac_pos"] = float(
                (mm.loc[mm["dv_kms"] > 0, "spCl"].astype(str)
                 == "GALAXY").mean())
            spec_ctrl["spclass_frac_neg"] = float(
                (mm.loc[mm["dv_kms"] < 0, "spCl"].astype(str)
                 == "GALAXY").mean())
            for s in spec_ctrl["f_plus_vs_e_zsp"]:
                print_status(
                    f"spec-bias {s['cut']}: f_+ = "
                    f"{s['frac_positive']:.4f} "
                    f"(n={s['n_satellites']})", "TEST")
            cs = spec_ctrl["clean_subsample"]
            print_status(
                f"spec-bias clean subsample: f_+ = "
                f"{cs['frac_positive']:.4f} "
                f"(n={cs['n_satellites']}, "
                f"p={cs['binomial_p_vs_half']:.3g})", "TEST")
            for s in spec_ctrl["f_plus_vs_drmag_fine"]:
                print_status(
                    f"spec-bias {s['cut']}: f_+ = "
                    f"{s['frac_positive']:.4f} "
                    f"(n={s['n_satellites']})", "TEST")

        # ---- control 8: spectral-class / same-feature-set channel --
        # The magnitude-differential template bias remains open after
        # control 7 because quality flags cannot see a systematic
        # template offset.  Two closures: (i) does the asymmetry
        # track spectral class, or luminosity within a class;
        # (ii) restrict satellites and centrals to the same feature
        # set so both redshifts are measured the same way.
        # Spectral class = SDSS DR16 subCl (emission vs passive) and
        # spectroscopic S/N from VizieR V/154/sdss16 via chunked
        # CDS XMatch; colour-matched quadrants need no cross-match.
        bnd = bound_all.reset_index(drop=True)
        cls_ctrl = {}
        # 8a. colour same-feature-set: both members red / both blue
        cls_ctrl["f_plus_both_red_gr_gt_0.7"] = _stats(
            bnd[(bnd["g_r_color"] > 0.7) & (bnd["cen_g_r"] > 0.7)],
            "sat & cen g-r > 0.7", p0)
        cls_ctrl["f_plus_both_blue_gr_le_0.7"] = _stats(
            bnd[(bnd["g_r_color"] <= 0.7) & (bnd["cen_g_r"] <= 0.7)],
            "sat & cen g-r <= 0.7", p0)
        cls_ctrl["f_plus_sat_blue_cen_red"] = _stats(
            bnd[(bnd["g_r_color"] <= 0.7) & (bnd["cen_g_r"] > 0.7)],
            "sat blue, cen red", p0)
        cls_ctrl["f_plus_sat_red_cen_blue"] = _stats(
            bnd[(bnd["g_r_color"] > 0.7) & (bnd["cen_g_r"] <= 0.7)],
            "sat red, cen blue", p0)

        def _xmatch_chunked(coords, cols, cache, tag):
            """Chunked CDS XMatch to V/154/sdss16 with on-disk cache."""
            if cache.exists():
                return pd.read_csv(cache)
            up = coords.reset_index(drop=True).copy()
            up.insert(0, "idx", np.arange(len(up)))
            up.columns = ["idx", "ra", "dec"]
            chunks = []
            for s0 in range(0, len(up), 20000):
                part = up.iloc[s0:s0 + 20000]
                for attempt in range(3):
                    try:
                        xm = XMatch.query(
                            cat1=Table.from_pandas(part),
                            cat2="vizier:V/154/sdss16",
                            max_distance=2.0 * u.arcsec,
                            colRA1="ra", colDec1="dec")
                        chunks.append(xm.to_pandas())
                        break
                    except Exception:
                        if attempt == 2:
                            raise
                        time.sleep(20)
                print_status(
                    f"{tag} XMatch chunk {s0 // 20000 + 1}: "
                    f"{len(part)} rows", "INFO")
            xmd = pd.concat(chunks, ignore_index=True)
            xmd = xmd.sort_values("angDist").drop_duplicates("idx")
            out = xmd[["idx"] + cols].copy()
            out.to_csv(cache, index=False)
            return out

        try:
            import time
            from astropy.table import Table
            from astroquery.xmatch import XMatch
            import astropy.units as u
            sat_cls = _xmatch_chunked(
                bnd[["RAJ2000", "DEJ2000"]],
                ["subCl", "spCl", "e_zsp"],
                self.data_processed / "sdss_satellite_specclass.csv",
                "satellite")
            cen_ids = bnd["GroupID"].unique()
            cen_coords = cen.loc[cen_ids, ["RAJ2000", "DEJ2000"]] \
                .reset_index()
            cen_coords["idx"] = np.arange(len(cen_coords))
            cen_cls = _xmatch_chunked(
                cen_coords[["RAJ2000", "DEJ2000"]],
                ["subCl", "spCl", "e_zsp"],
                self.data_processed / "sdss_central_specclass.csv",
                "central")
            cen_cls = cen_coords.merge(cen_cls, on="idx")[
                ["GroupID", "subCl", "spCl", "e_zsp"]] \
                .rename(columns={"subCl": "cen_subCl",
                                 "e_zsp": "cen_e_zsp",
                                 "spCl": "cen_spCl"})
            ms = bnd.merge(sat_cls, left_index=True, right_on="idx",
                           how="left").merge(cen_cls, on="GroupID",
                                             how="left")
        except Exception as exc:
            ms = None
            cls_ctrl["xmatch_error"] = str(exc)
            print_status(
                f"spec-class XMatch failed: {exc}", "ERROR")

        if ms is not None and len(ms):
            def _emission(v):
                s = v.astype(str).str.upper()
                return (s.str.contains("STARFORM|STARBURST|AGN"
                                       "|BROADLINE")
                        & ~s.str.contains(r"E\+A"))
            ms["sat_emission"] = _emission(ms["subCl"])
            ms["cen_emission"] = _emission(ms["cen_subCl"])
            ms["sat_passive"] = ms["subCl"].isna() | (
                ms["subCl"].astype(str).str.strip()
                .isin(["", "--", "NAN", "NONE"]))
            ms["cen_passive"] = ms["cen_subCl"].isna() | (
                ms["cen_subCl"].astype(str).str.strip()
                .isin(["", "--", "NAN", "NONE"]))
            ms["matched_subcl"] = ms["subCl"].notna()
            msp = ms[ms["spCl"].astype(str) == "GALAXY"].copy()

            cls_ctrl["xmatch_sat_n"] = int(ms["subCl"].notna()
                                           .sum())
            cls_ctrl["xmatch_cen_n"] = int(ms["cen_subCl"].notna()
                                           .sum())
            cls_ctrl["subcl_counts_satellite"] = (
                ms["subCl"].astype(str).value_counts()
                .head(10).to_dict())

            # 8b. f_+ by satellite spectral class
            cls_ctrl["f_plus_sat_passive"] = _stats(
                msp[msp["sat_passive"]], "sat passive (no subCl)",
                p0)
            cls_ctrl["f_plus_sat_emission"] = _stats(
                msp[msp["sat_emission"]], "sat emission-line", p0)
            cls_ctrl["f_plus_sat_EA"] = _stats(
                msp[msp["subCl"].astype(str).str.upper()
                    .str.contains(r"E\+A", na=False)],
                "sat E+A", p0)

            # 8c. same feature set on both sides: class quadrants
            for sat_e, cen_e, key, lab in [
                    (True, True, "f_plus_both_emission",
                     "both emission"),
                    (False, False, "f_plus_both_passive",
                     "both passive"),
                    (True, False, "f_plus_sat_emit_cen_passive",
                     "sat emit, cen passive"),
                    (False, True, "f_plus_sat_passive_cen_emit",
                     "sat passive, cen emit")]:
                sub = msp[(msp["sat_emission"] == sat_e)
                          & (msp["cen_emission"] == cen_e)]
                cls_ctrl[key] = _stats(sub, lab, p0)

            # 8d. matched-error same-feature channel: restrict to
            # pairs where satellite and central redshift errors are
            # comparable, so any magnitude-linked template offset is
            # applied symmetrically on both sides
            eok = msp.dropna(subset=["e_zsp", "cen_e_zsp"]).copy()
            eok["e_ratio"] = (eok["e_zsp"].astype(float)
                              / eok["cen_e_zsp"].astype(float))
            for lo, hi in [(0.5, 2.0), (0.75, 1.33)]:
                sub = eok[eok["e_ratio"].between(lo, hi)]
                cls_ctrl[f"f_plus_err_ratio_{lo}_{hi}"] = _stats(
                    sub, f"e_zsp sat/cen in [{lo}, {hi}]", p0)

            # 8e. within-class amplitude scaling: if the luminosity
            # trend survives inside the passive class, the class
            # channel is closed
            def _amp(bins_df, label, lo, hi):
                s = _stats(bins_df, label, p0)
                s["mean_dv_kms"] = float(bins_df["dv_kms"].mean())
                s["mean_dv_se_kms"] = float(
                    bins_df["dv_kms"].std()
                    / np.sqrt(len(bins_df)))
                s["M_r_lo"], s["M_r_hi"] = lo, hi
                return s

            passive = msp[msp["sat_passive"]]
            cls_ctrl["within_passive_scaling"] = [
                _amp(passive[passive["M_r"].between(lo, hi)],
                     f"passive M_r {lo} to {hi}", lo, hi)
                for lo, hi in [(-24, -21.5), (-21.5, -21.0),
                               (-21.0, -20.5), (-20.5, -20.0),
                               (-20.0, -19.5), (-19.5, -18.0)]
                if len(passive[passive["M_r"].between(lo, hi)])
                >= 200]
            emission = msp[msp["sat_emission"]]
            cls_ctrl["within_emission_scaling"] = [
                _amp(emission[emission["M_r"].between(lo, hi)],
                     f"emission M_r {lo} to {hi}", lo, hi)
                for lo, hi in [(-24, -21.0), (-21.0, -20.0),
                               (-20.0, -18.0)]
                if len(emission[emission["M_r"].between(lo, hi)])
                >= 200]

            for key in ["f_plus_sat_passive", "f_plus_sat_emission",
                        "f_plus_both_emission", "f_plus_both_passive",
                        "f_plus_sat_emit_cen_passive",
                        "f_plus_sat_passive_cen_emit"]:
                s = cls_ctrl.get(key)
                if isinstance(s, dict) and s.get("n_satellites"):
                    print_status(
                        f"spec-class {s['cut']}: f_+ = "
                        f"{s['frac_positive']:.4f} "
                        f"(n={s['n_satellites']})", "TEST")
            for s in cls_ctrl.get("f_plus_vs_spSN", []):
                print_status(
                    f"spec-class {s['cut']}: f_+ = "
                    f"{s['frac_positive']:.4f} "
                    f"(n={s['n_satellites']})", "TEST")
            for s in cls_ctrl.get("within_passive_scaling", []):
                print_status(
                    f"within-class {s['cut']}: f_+ = "
                    f"{s['frac_positive']:.4f}, <dV> = "
                    f"{s['mean_dv_kms']:+.2f} km/s "
                    f"(n={s['n_satellites']})", "TEST")

        spec_ctrl["spectral_class_control"] = cls_ctrl

        for s in stats + bands + rproj + richness:
            print_status(
                f"{s['cut']}: f_+ = {s['frac_positive']:.4f} "
                f"(n={s['n_satellites']}), p(0.5)="
                f"{s['binomial_p_vs_half']:.3g}, p(interloper)="
                f"{s['binomial_p_vs_interloper']:.3g}", "TEST")
        for m in mag_ctrl:
            print_status(
                f"dust ctrl |dV| {m['dv_band_kms'][0]}-"
                f"{m['dv_band_kms'][1]}: <dmag> +{m['mean_drmag_pos']:.3f}"
                f" vs -{m['mean_drmag_neg']:.3f}, "
                f"diff={m['diff_pos_minus_neg']:+.3f}, "
                f"<g-r> +{m['mean_gr_pos']:.3f} vs "
                f"{m['mean_gr_neg']:.3f}", "TEST")

        summary = {
            "catalog": (
                "Tempel et al. 2017 SDSS DR10 flux-limited groups "
                "(VizieR J/A+A/602/A100); central = rank-1 galaxy; "
                "dV = c(z_sat - z_cen)/(1+z_cen)"
            ),
            "n_groups": int(df["GroupID"].nunique()),
            "n_satellites": int(len(df)),
            "interloper_baseline": {
                "cut_kms": DV_INTERLOPER,
                "n": n_tail,
                "frac_positive": f_tail,
                "note": (
                    "measured survey selection asymmetry of "
                    "surely-unbound members; used as the null "
                    "probability p0 for all bound-regime tests"),
            },
            "offset_statistics": stats,
            "dv_bands": bands,
            "virial_normalized_bands": sigstats,
            "rproj_binned_mpc": rproj,
            "rproj_binned_r200": rproj200,
            "dust_magnitude_control": mag_ctrl,
            "richness_split": richness,
            "lumw_reference_control": lumw,
            "volume_limited_control": vol_lim,
            "mock_interloper_forward_model": interloper_models,
            "spectroscopic_bias_control": spec_ctrl,
            "amplitude_scaling": {
                "by_satellite_M_r": lum_scale,
                "by_satellite_minus_central_rmag": drm_scale,
            },
            "per_group_bound": {
                "n_groups": int(len(gmean)),
                "frac_groups_mean_dv_positive": float(
                    (gmean > 0).mean()),
                "wilcoxon_p_positive": p_grp,
            },
            "method": (
                "FoF group membership is blind to the hypothesis. "
                "The sign asymmetry of satellite velocity offsets is "
                "tested against the empirically measured interloper "
                "baseline (|dV|>2000 km/s tail), not the symmetric "
                "0.5; dust-obscuration alternatives are controlled by "
                "the R_proj dependence and by the satellite-central "
                "magnitude and g-r colour asymmetry of matched +/-dV "
                "populations."
            ),
        }
        json_path = (
            self.results / "step_37_sdss_companion_asymmetry.json")
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")
        self._figure(stats, bands, rproj200, f_tail)
        print_status(
            f"SDSS asymmetry complete: bound f_+ = "
            f"{stats[1]['frac_positive']:.4f} vs interloper baseline "
            f"{f_tail:.4f}.", "SUCCESS")

    # --------------------------------------------------------------
    def _figure(self, stats, bands, rproj200, f_tail):
        try:
            import warnings as _w
            with _w.catch_warnings():
                _w.filterwarnings("ignore",
                                  message=".*extended precision.*")
                import matplotlib.pyplot as plt
            apply_tep_style()
            fig, (ax1, ax2) = plt.subplots(
                1, 2, figsize=(11, 4.4), constrained_layout=True)

            # panel 1: f_+ vs |dV| band
            los = [b["cut"] for b in bands]
            fs = [b["frac_positive"] for b in bands]
            e_lo = [b["frac_positive"] - b["frac_positive_ci95"][0]
                    for b in bands]
            e_hi = [b["frac_positive_ci95"][1] - b["frac_positive"]
                    for b in bands]
            xs = np.arange(len(los))
            ax1.errorbar(xs, fs, yerr=[e_lo, e_hi], fmt="o",
                         ms=5, color="#241a33", capsize=3)
            ax1.axhline(0.5, color="#888", lw=0.8, ls="--")
            ax1.axhline(f_tail, color="#c1913f", lw=1.0, ls=":")
            ax1.set_xticks(xs)
            ax1.set_xticklabels(
                [l.replace("|dV| ", "").replace(" km/s", "")
                 for l in los], rotation=30, ha="right", fontsize=8)
            ax1.set_ylabel(r"$f_+$")
            ax1.set_xlabel(r"$|\Delta v|$ band (km/s)")
            ax1.set_title("sign asymmetry vs velocity offset", fontsize=10)

            # panel 2: f_+ vs R/R200
            xs2 = [0.5 * (r["r200_lo"] + r["r200_hi"]) for r in rproj200]
            fs2 = [r["frac_positive"] for r in rproj200]
            el2 = [r["frac_positive"] - r["frac_positive_ci95"][0]
                   for r in rproj200]
            eh2 = [r["frac_positive_ci95"][1] - r["frac_positive"]
                   for r in rproj200]
            ax2.errorbar(xs2, fs2, yerr=[el2, eh2], fmt="s-",
                         ms=5, color="#241a33", capsize=3)
            ax2.axhline(0.5, color="#888", lw=0.8, ls="--")
            ax2.axhline(f_tail, color="#c1913f", lw=1.0, ls=":")
            ax2.set_xlabel(r"$R_{\rm proj}/R_{200}$")
            ax2.set_ylabel(r"$f_+$ (bound, $|\Delta v|<500$ km/s)")
            ax2.set_title("sign asymmetry vs projected radius", fontsize=10)

            fig.savefig(
                self.figures / "step_37_sdss_companion_asymmetry.png",
                dpi=300, bbox_inches="tight")
            plt.close(fig)
            print_status(
                "Saved figure: step_37_sdss_companion_asymmetry.png",
                "SUCCESS")
        except Exception as e:
            self.logger.warning(f"figure failed: {e}")


if __name__ == "__main__":
    Step37SdssCompanionAsymmetry().run()
