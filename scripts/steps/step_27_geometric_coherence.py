#!/usr/bin/env python3
"""
Step 27: Geometric and Spectroscopic Coherence Tests
=====================================================
The catalogue was selected for discordance and claimed
connections, so the angular proximity test of step_23 prices a
quantity the sample was chosen on.  This step instead evaluates
structure the selection never used — orthogonal signatures that
the chance-superposition hypothesis cannot produce even inside
a deliberately selected sample:

  1. X-ray-selection repricing.  The companions of the
     X-ray-selected systems (NGC 7319, NGC 3628, NGC 4258,
     NGC 2639, NGC 3516) were identified in X-ray surveys, not
     optical quasar catalogues.  The correct comparison
     population for their angular statistic is therefore the
     surface density of X-ray-detected quasar-class objects,
     measured empirically from the Milliquas association flags
     (type suffix 'X') inside the same control fields used by
     step_22 — the same catalogue, the same sight-lines, the
     same completeness.  The joint probability is recomputed
     with each pair priced against the population it was
     actually selected from.

  2. Minor-axis anisotropy.  Arp's ejection geometry predicts
     companions concentrated on host minor axes.  For every
     resolved companion the position angle of the
     host-to-companion offset is compared against the host's
     major-axis PA (NED diameters table; literature photometric
     PAs), folded to axial coordinates, and tested for
     clustering at the minor axis (binomial test on the
     |offset| < 30 deg fraction versus the 1/3 uniform
     expectation, and the circular V-test on doubled angles).

  3. Radial redshift ordering.  For fields with two or more
     resolved members, every within-field member pair
     contributes a sign: does the more distant member carry the
     higher redshift?  A pooled two-sided binomial test asks
     whether the ordering departs from random in either
     direction; fields with >= 3 members additionally get an
     exact-permutation Spearman test of z against projected
     nuclear distance.

  4. Halo-scale clustering.  For the six hosts carrying
     redshift-independent Cosmicflows-4 distances the projected
     companion separation is converted to kpc.  Under chance
     projection the angular separation is unrelated to the
     host distance, so a random reassignment of separations to
     distances should produce physical scales as tight as the
     true pairing; an exact permutation test over all 6! = 720
     assignments measures whether the observed kpc-scale
     dispersion is unusually small.

  5. Paired-redshift excess.  For each multi-member field the
     closest pairwise |dz| among resolved members (and, in a
     second variant, among all confirmed quasar-class objects
     in the field cone) is priced against Monte Carlo draws
     from the empirical field-quasar redshift distribution
     measured on the step_22 control fields — the distribution
     with the same catalogue selection function.

  6. Symmetric-configuration probability.  For two-member
     fields the opposition angle between the two
     host-to-companion directions and the radius ratio are
     priced analytically (uniform directions: phi uniform on
     [0,180]; uniform disk positions: ratio density 2*rho) and
     verified by Monte Carlo.

All inputs are pipeline products with archive provenance:
companion_positions.csv (step_20 archive resolution),
qso_field_members.csv (step_22 member-level cones),
arp_pair_catalog_verified.csv (step_01 NED verification),
bridge_morphology.csv (step_20 measured separations),
step_05 distances, step_23 density framework.  Host PAs come
from the NED diameters table (literature photometry), with the
per-host source recorded.  Monte Carlo uses a fixed seed.

Outputs:
    results/outputs/step_27_geometric_coherence.json
    data/processed/geometric_coherence_tests.csv
    data/processed/host_position_angles.csv
    data/processed/companion_axis_offsets.csv
"""

import json
import math
import sys
import time
from itertools import combinations, permutations
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.jsonio import json_safe

RNG_SEED = 20260924
N_MC = 200_000          # Monte Carlo draws for redshift-pairing test
N_MC_GEOM = 50_000      # Monte Carlo draws for symmetric-configuration check
MINOR_AXIS_HALF_WINDOW_DEG = 30.0   # |offset from minor axis| window
MILLIQUAS_CATALOG = "VII/290/catalog"

CONFIRMED_CLASSES = ("Q", "A", "B", "K")


def is_confirmed_quasar(type_str):
    """Mirror of step_22's Milliquas class filter."""
    t = str(type_str).strip()
    return len(t) > 0 and t[0] in CONFIRMED_CLASSES


def is_xray_detected(type_str):
    """Milliquas association flag: 'X' suffix marks an X-ray
    detection of the source (any mission)."""
    return "X" in str(type_str).strip()


def _pa_east_of_north(host, comp):
    """Position angle (deg E of N, [0,360)) of comp as seen from host."""
    dra = (comp.ra.rad - host.ra.rad) * math.cos(host.dec.rad)
    ddec = comp.dec.rad - host.dec.rad
    return math.degrees(math.atan2(dra, ddec)) % 360.0


def _circular_median_axial(pas_deg):
    """Median direction for axial data: fold to [0,180), double,
    circular-mean, halve."""
    arr = np.asarray(pas_deg, dtype=float) % 180.0
    psi = np.radians(2.0 * arr)
    mx = np.mean(np.cos(psi))
    my = np.mean(np.sin(psi))
    mean2 = math.degrees(math.atan2(my, mx)) % 360.0
    return mean2 / 2.0


class Step27GeometricCoherence:
    """Step 27: structure tests the catalogue was not selected on."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"
        for d in [self.data_processed, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)
        self.logger = TEPLogger(
            "step_27",
            log_file_path=self.logs / "step_27_geometric_coherence.log",
        )
        set_step_logger(self.logger)
        self.rng = np.random.default_rng(RNG_SEED)

    # ---------------------------------------------------------------
    # Input loading (all with provenance; fail loudly when missing)
    # ---------------------------------------------------------------
    def _load_inputs(self):
        cat_path = self.data_processed / "arp_pair_catalog_verified.csv"
        if not cat_path.exists():
            cat_path = self.data_processed / "arp_pair_catalog.csv"
        if not cat_path.exists():
            raise FileNotFoundError("pair catalogue missing; run step_00/01")
        self.catalog = pd.read_csv(cat_path)

        pos_path = self.data_processed / "companion_positions.csv"
        if not pos_path.exists():
            raise FileNotFoundError(
                "companion_positions.csv missing; run step_20")
        self.positions = pd.read_csv(pos_path)

        mem_path = self.data_processed / "qso_field_members.csv"
        if not mem_path.exists():
            raise FileNotFoundError(
                "qso_field_members.csv missing; rerun step_22 "
                "(member-level storage required)")
        self.members = pd.read_csv(mem_path)

        fld_path = self.data_processed / "qso_field_counts.csv"
        if not fld_path.exists():
            raise FileNotFoundError("qso_field_counts.csv missing")
        self.field_counts = pd.read_csv(fld_path)

        s23_path = self.results / "step_23_chance_alignment.json"
        if not s23_path.exists():
            raise FileNotFoundError("step_23 output missing")
        self.s23 = json.loads(s23_path.read_text())
        self.s23_pairs = {
            p["pair_id"]: p for p in self.s23["pairs"]}

        s05_path = (
            self.results / "step_05_redshift_independent_distances.json")
        self.s05 = (
            json.loads(s05_path.read_text()) if s05_path.exists()
            else None
        )

    def _resolve_host(self, name):
        """Simbad host resolution (same service as step_20)."""
        from astroquery.simbad import Simbad
        import warnings as _w
        from astroquery.exceptions import NoResultsWarning
        try:
            with _w.catch_warnings():
                _w.simplefilter("ignore", category=NoResultsWarning)
                tbl = Simbad.query_object(name)
        except Exception:
            return None
        if tbl is None or len(tbl) == 0:
            return None
        return SkyCoord(
            float(tbl[0]["ra"]) * u.deg, float(tbl[0]["dec"]) * u.deg)

    def _host_pa_ned(self, name):
        """Host major-axis PA from the NED diameters table: all
        literature photometric position angles are folded to axial
        [0,180) and combined by circular median; the median axis
        ratio is kept for the face-on quality flag.  Returns
        (pa_deg, axis_ratio_median, n_measurements) or (None,...)."""
        from astroquery.ipac.ned import Ned
        t = None
        for attempt in (1, 2):
            try:
                t = Ned.get_table(name, table="diameters")
                break
            except Exception as e:
                if attempt == 2:
                    self.logger.info(
                        f"NED diameters query failed for {name}: "
                        f"{str(e).strip().splitlines()[0]}; "
                        "falling back to HyperLEDA (VizieR VII/237)")
                else:
                    time.sleep(5.0)
        if t is None or len(t) == 0:
            return self._host_pa_leda(name)
        pas, bas = [], []
        for r in t:
            pa = r["Position Angle"]
            if not np.ma.is_masked(pa):
                try:
                    paf = float(pa)
                    if np.isfinite(paf):
                        pas.append(paf)
                except (TypeError, ValueError):
                    pass
            ba = r["Axis Ratio"]
            if not np.ma.is_masked(ba):
                try:
                    baf = float(ba)
                    if np.isfinite(baf):
                        bas.append(baf)
                except (TypeError, ValueError):
                    pass
        if not pas:
            return None, np.nan, 0, "unavailable"
        pa_med = _circular_median_axial(pas)
        ba_med = float(np.median(bas)) if bas else np.nan
        return pa_med, ba_med, len(pas), \
            "NED diameters (literature photometry)"

    def _host_pa_leda(self, name):
        """Fallback host PA from HyperLEDA (VizieR catalogue VII/237,
        Paturel et al. 2003): mean photometric PA and the logR25 axis
        ratio (b/a = 10^-logR25).  Returns (pa_deg, axis_ratio, 1)."""
        from astroquery.vizier import Vizier
        try:
            Vizier.ROW_LIMIT = 5
            cats = Vizier.query_object(name, catalog="VII/237")
        except Exception as e:
            self.logger.info(
                f"HyperLEDA query failed for {name}: "
                f"{str(e).strip().splitlines()[0]}")
            return None, np.nan, 0, "unavailable"
        if not cats:
            return None, np.nan, 0, "unavailable"
        t = cats[0]
        pa = t["PA"][0]
        if np.ma.is_masked(pa):
            return None, np.nan, 0, "unavailable"
        ba = np.nan
        lr = t["logR25"][0]
        if not np.ma.is_masked(lr):
            ba = float(10.0 ** (-float(lr)))
        return float(pa), ba, 1, \
            "HyperLEDA mean PA (VizieR VII/237, Paturel et al. 2003)"

    # ---------------------------------------------------------------
    # Test 1: X-ray-selection repricing of the angular statistic
    # ---------------------------------------------------------------
    def _xray_pricing(self):
        from scipy.stats import poisson

        ctrl = self.members[self.members["field_kind"] == "control"]
        n_fields = int(self.field_counts.shape[0])
        area_deg2 = math.pi * (6.0 / 60.0) ** 2
        n_x = int(sum(is_xray_detected(t) for t in ctrl["type"]))
        n_all = len(ctrl)
        sigma_x = n_x / (n_fields * area_deg2)
        sigma_x_err = math.sqrt(n_x) / (n_fields * area_deg2)
        frac_x = n_x / n_all if n_all else np.nan
        self.logger.info(
            f"X-ray-detected confirmed quasar-class objects in control "
            f"fields: {n_x}/{n_all} -> Sigma_X = {sigma_x:.3f} "
            f"+/- {sigma_x_err:.3f} deg^-2 "
            f"(fraction {frac_x:.3f} of the field density)")

        sig_emp = float(self.s23["sigma_qso_empirical_deg2"])
        sig_gal = float(self.s23["sigma_gal_deg2"])
        out_pairs = []
        joint_sel = 1.0
        for _, row in self.catalog.iterrows():
            pid = row["pair_id"]
            p23 = self.s23_pairs.get(pid)
            if p23 is None:
                continue
            theta = float(p23["separation_arcsec"]) / 3600.0
            ctype = str(row["companion_type"]).lower()
            is_gal = "galax" in ctype or ctype == "compact object"
            is_xray = "x-ray" in ctype
            if isinstance(row.get("z_comp_list"), str) and row["z_comp_list"]:
                n_mem = len(row["z_comp_list"].split(";"))
            else:
                n_mem = 1
            if is_gal:
                sig_sel = sig_gal
                basis = "resolved-galaxy density"
            elif is_xray:
                sig_sel = sigma_x
                basis = "X-ray-detected quasar-class density"
            else:
                sig_sel = sig_emp
                basis = "quasar-class density"
            lam = sig_sel * math.pi * theta * theta
            p_sel = float(1.0 - poisson.cdf(n_mem - 1, lam))
            joint_sel *= p_sel
            out_pairs.append({
                "pair_id": pid,
                "selection_class": (
                    "xray" if is_xray else ("galaxy" if is_gal else "quasar")),
                "sigma_used_deg2": sig_sel,
                "density_basis": basis,
                "n_members": n_mem,
                "lambda_sel": lam,
                "p_chance_selection_aware": p_sel,
            })
        return {
            "sigma_xray_deg2": sigma_x,
            "sigma_xray_err_deg2": sigma_x_err,
            "n_xray_in_control_fields": n_x,
            "n_control_members": n_all,
            "xray_fraction_of_field": frac_x,
            "density_note": (
                "X-ray-detected fraction of confirmed quasar-class "
                "Milliquas objects measured on the step_22 control "
                "fields; the flag spans all X-ray missions (deeper "
                "than the ROSAT-era selection of the companions), so "
                "Sigma_X is an upper bound and the repriced "
                "probabilities are conservative"),
            "pairs": out_pairs,
            "joint_probability_selection_aware": joint_sel,
            "joint_log10_selection_aware": math.log10(joint_sel),
            "joint_excluding": {
                p["pair_id"]: joint_sel / p["p_chance_selection_aware"]
                for p in out_pairs if p["p_chance_selection_aware"] > 0
            },
        }

    # ---------------------------------------------------------------
    # Test 2: minor-axis anisotropy
    # ---------------------------------------------------------------
    def _minor_axis(self):
        host_pa = {}
        pa_rows = []
        for _, row in self.catalog.iterrows():
            name = row["galaxy"]
            pa, ba, n, src = self._host_pa_ned(name)
            host_pa[row["pair_id"]] = {
                "galaxy": name, "pa_deg": pa, "axis_ratio": ba,
                "n_measurements": n,
                "source": src,
            }
            pa_rows.append({
                "pair_id": row["pair_id"], "galaxy": name,
                "pa_major_deg": pa, "axis_ratio_median": ba,
                "n_pa_measurements": n,
                "pa_source": host_pa[row["pair_id"]]["source"],
            })
            self.logger.info(
                f"{name}: PA = {pa:.1f} deg (axis_ratio={ba:.2f}, n={n})"
                if pa is not None else
                f"{name}: PA unavailable")
            time.sleep(0.3)

        conn_type = {
            r["pair_id"]: r["connection_type"]
            for _, r in self.catalog.iterrows()
        }
        comp_rows = []
        for pid, grp in self.positions.groupby("pair_id"):
            info = host_pa.get(pid)
            if info is None or info["pa_deg"] is None:
                continue
            gpos = self._resolve_host(info["galaxy"])
            if gpos is None:
                continue
            for r in grp.itertuples():
                cpos = SkyCoord(ra=r.ra_deg * u.deg, dec=r.dec_deg * u.deg)
                pa_c = _pa_east_of_north(gpos, cpos)
                theta_rel = (pa_c - info["pa_deg"]) % 180.0
                dmin = abs(theta_rel - 90.0)
                ba_raw = info["axis_ratio"]
                # NED diameter references report the axis ratio in
                # mixed conventions (b/a vs a/b); the elongation that
                # controls PA quality is symmetric under inversion.
                ba_eff = (min(ba_raw, 1.0 / ba_raw)
                          if np.isfinite(ba_raw) and ba_raw > 0
                          else np.nan)
                comp_rows.append({
                    "pair_id": pid,
                    "resolved_name": r.resolved_name,
                    "is_primary": bool(r.is_primary),
                    "connection_type": conn_type.get(pid, ""),
                    "pa_companion_deg": pa_c,
                    "pa_major_deg": info["pa_deg"],
                    "theta_rel_deg": theta_rel,
                    "offset_from_minor_deg": dmin,
                    "offset_from_major_deg": 90.0 - dmin,
                    "host_axis_ratio": info["axis_ratio"],
                    "host_axis_ratio_folded": ba_eff,
                    "separation_arcsec": float(r.measured_separation_arcsec),
                })
            time.sleep(0.2)

        if not comp_rows:
            raise RuntimeError("no companion PAs computable")

        def _stats(sub):
            deltas = np.asarray(sub["offset_from_minor_deg"], dtype=float)
            n = len(deltas)
            k = int(np.sum(deltas <= MINOR_AXIS_HALF_WINDOW_DEG))
            from scipy.stats import binomtest
            p_bin = float(
                binomtest(k, n, 1.0 / 3.0, alternative="greater").pvalue)
            # Axial V-test toward the minor axis: psi = 2*theta_rel
            # (theta_rel measured from major axis); minor axis maps to
            # psi = 180 deg.  u = sqrt(2/n) * sum cos(psi - mu0).
            theta_rel = np.asarray(sub["theta_rel_deg"], dtype=float)
            psi = np.radians(2.0 * theta_rel)
            u_stat = math.sqrt(2.0 / n) * float(
                np.sum(np.cos(psi - math.pi)))
            from scipy.stats import norm
            p_v = float(norm.sf(u_stat))
            return {
                "n": n,
                "n_within_30deg": k,
                "frac_within_30deg": k / n,
                "mean_offset_deg": float(np.mean(deltas)),
                "median_offset_deg": float(np.median(deltas)),
                "p_binomial_vs_uniform": p_bin,
                "v_statistic": u_stat,
                "p_vtest_minor_axis": p_v,
            }

        allc = pd.DataFrame(comp_rows)
        samples = {"all_companions": _stats(allc)}
        prim = allc[allc["is_primary"]]
        if len(prim) >= 3:
            samples["primary_companions"] = _stats(prim)
        # Predeclared subgroup: only pairs whose catalogue claim places
        # companions on a host-morphology axis (connection_type ==
        # "axis_alignment").  Bridge/filament companions are claimed
        # on their own structure axes, not the host minor axis, so
        # pooling them dilutes the signal the hypothesis actually
        # predicts.
        axs = allc[allc["connection_type"] == "axis_alignment"]
        if len(axs) >= 3:
            samples["axis_alignment_pairs"] = _stats(axs)
        edge = allc[np.isfinite(allc["host_axis_ratio_folded"])
                   & (allc["host_axis_ratio_folded"] <= 0.85)]
        if len(edge) >= 3:
            samples["well_defined_pa_subset"] = _stats(edge)

        return {
            "host_pa": host_pa,
            "companions": comp_rows,
            "samples": samples,
            "method": (
                "Host major-axis PA = circular median of all literature "
                "photometric PAs in the NED diameters table; companion "
                "PA = host-to-companion direction E of N at archive "
                "positions (step_20).  Offset folded to axial [0,90] "
                "deg from the minor axis.  Uniform-null checks: "
                "binomial on |offset|<=30 deg (expectation 1/3) and "
                "circular V-test on doubled angles."
            ),
        }

    # ---------------------------------------------------------------
    # Test 3: radial redshift ordering within multi-member fields
    # ---------------------------------------------------------------
    def _radial_ordering(self):
        multi = {
            pid: grp for pid, grp in self.positions.groupby("pair_id")
            if len(grp) >= 2
        }
        sign_pairs = []
        per_field = []
        # Per-field null distributions of the within-field count of
        # nearer=higher-z pairs, used to build the exact within-field
        # permutation null.  Member pairs within a single field share
        # members and are correlated; the pooled binomial that treats
        # them as independent Bernoulli trials is accordingly
        # anti-conservative.  The correct null permutes redshifts
        # within each field independently and convolves the per-field
        # distributions.
        per_field_dist = []
        for pid, grp in multi.items():
            rr = grp["measured_separation_arcsec"].to_numpy(dtype=float)
            zz = grp["z"].to_numpy(dtype=float)
            for i, j in combinations(range(len(rr)), 2):
                if abs(rr[j] - rr[i]) < 1e-9 or abs(zz[j] - zz[i]) < 1e-12:
                    continue
                s = np.sign((rr[j] - rr[i]) * (zz[j] - zz[i]))
                sign_pairs.append({
                    "pair_id": pid,
                    "r_near": float(min(rr[i], rr[j])),
                    "r_far": float(max(rr[i], rr[j])),
                    "z_near": float(zz[i] if rr[i] < rr[j] else zz[j]),
                    "z_far": float(zz[j] if rr[i] < rr[j] else zz[i]),
                    "sign": int(s),   # +1: farther member has higher z
                })
            rec = {"pair_id": pid, "n_members": int(len(grp))}
            if len(grp) >= 3:
                rho_obs = float(np.corrcoef(
                    np.argsort(np.argsort(rr)),
                    np.argsort(np.argsort(zz)))[0, 1])
                if np.isfinite(rho_obs):
                    idx = np.arange(len(grp))
                    cnt = 0
                    tot = 0
                    for perm in permutations(idx):
                        rho_p = float(np.corrcoef(
                            np.argsort(np.argsort(rr)),
                            np.argsort(np.argsort(zz[list(perm)])))[0, 1])
                        tot += 1
                        if (np.isfinite(rho_p) and
                                abs(rho_p) >= abs(rho_obs) - 1e-12):
                            cnt += 1
                    rec["spearman_rho"] = rho_obs
                    rec["p_perm_two_sided"] = cnt / tot
                    rec["n_perms"] = tot
                else:
                    rec["status"] = (
                        "skipped: constant z or separation among "
                        "members (Spearman undefined)")
            per_field.append(rec)

            # Per-field distribution of the nearer=higher-z count
            # (sign = -1) over all n! redshift permutations.
            idx = np.arange(len(rr))
            dist = {}
            for perm in permutations(idx):
                zp = zz[list(perm)]
                k = 0
                for i, j in combinations(range(len(rr)), 2):
                    if (abs(rr[j] - rr[i]) < 1e-9
                            or abs(zp[j] - zp[i]) < 1e-12):
                        continue
                    if (rr[j] - rr[i]) * (zp[j] - zp[i]) < 0:
                        k += 1  # nearer member has higher z
                dist[k] = dist.get(k, 0) + 1
            n_perm = sum(dist.values())
            per_field_dist.append(
                {k: v / n_perm for k, v in dist.items()})

        n_pos = sum(1 for s in sign_pairs if s["sign"] > 0)
        n_neg = sum(1 for s in sign_pairs if s["sign"] < 0)
        n_tot = len(sign_pairs)
        from scipy.stats import binomtest
        p_binom = float(binomtest(
            n_pos, n_tot, 0.5, alternative="two-sided").pvalue) \
            if n_tot else np.nan

        # Convolve per-field distributions to obtain the exact null
        # distribution of the total nearer=higher-z count.
        if per_field_dist:
            total_dist = per_field_dist[0]
            for d in per_field_dist[1:]:
                new_dist = {}
                for k1, p1 in total_dist.items():
                    for k2, p2 in d.items():
                        k = k1 + k2
                        new_dist[k] = new_dist.get(k, 0.0) + p1 * p2
                total_dist = new_dist
            observed_neg = n_neg
            expected = n_tot / 2.0
            tail = sum(p for k, p in total_dist.items()
                       if k >= observed_neg or k <= n_tot - observed_neg)
            p_perm = float(tail)
        else:
            p_perm = np.nan

        return {
            "n_ordered_member_pairs": n_tot,
            "n_farther_higher_z": n_pos,
            "n_nearer_higher_z": n_neg,
            "frac_farther_higher_z": (n_pos / n_tot if n_tot else np.nan),
            "p_binomial_two_sided": p_binom,
            "p_permutation_two_sided": p_perm,
            "per_field": per_field,
            "sign_pairs": sign_pairs,
            "method": (
                "Every unordered within-field member pair contributes "
                "sign(dr*dz): +1 when the member farther from the "
                "nucleus carries the higher redshift.  No directional "
                "claim is registered a priori.  Because pairs within "
                "a field share members and are correlated, the null "
                "distribution is built by permuting redshifts within "
                "each field independently and convolving the per-field "
                "distributions; the two-sided p_permutation_two_sided "
                "is reported as the primary result.  The pooled "
                "binomial p_binomial_two_sided is retained for "
                "reference but is anti-conservative.  Fields with "
                ">=3 members additionally report Spearman rho with an "
                "exact n! permutation p-value."
            ),
        }

    # ---------------------------------------------------------------
    # Test 4: halo-scale clustering under CF4 distances
    # ---------------------------------------------------------------
    def _halo_scale(self):
        if self.s05 is None:
            return {"status": "skipped: step_05 distances unavailable"}
        host_d = {
            m["pair_id"]: float(m["distance_mpc"])
            for m in self.s05["members"]
            if m["role"] == "host" and m["status"] == "measured"
        }
        rows = []
        for pid, d_mpc in host_d.items():
            grp = self.positions[self.positions["pair_id"] == pid]
            prim = grp[grp["is_primary"]]
            sep = float(
                prim["measured_separation_arcsec"].iloc[0]
                if len(prim) else grp["measured_separation_arcsec"].min())
            d_kpc = sep * d_mpc * math.pi / (180.0 * 3600.0) * 1000.0
            rows.append({
                "pair_id": pid, "host_distance_mpc": d_mpc,
                "separation_arcsec": sep, "projected_sep_kpc": d_kpc,
            })
        if len(rows) < 4:
            return {"status": "skipped: fewer than 4 anchored hosts"}

        thetas = np.array([r["separation_arcsec"] for r in rows])
        dists = np.array([r["host_distance_mpc"] for r in rows])
        conv = math.pi / (180.0 * 3600.0) * 1000.0

        def _disp(a, b):
            return float(np.std(np.log10(a * b * conv), ddof=1))

        obs = _disp(thetas, dists)
        idx = np.arange(len(rows))
        n_le = 0
        tot = 0
        for perm in permutations(idx):
            tot += 1
            if _disp(thetas, dists[list(perm)]) <= obs + 1e-12:
                n_le += 1
        return {
            "n_anchored_hosts": len(rows),
            "pairs": rows,
            "projected_sep_kpc": [r["projected_sep_kpc"] for r in rows],
            "std_log10_kpc_observed": obs,
            "permutation_test": {
                "statistic": "std(log10 projected kpc)",
                "n_permutations": tot,
                "p_le_observed": n_le / tot,
                "description": (
                    "Exact enumeration of all 6! reassignments of the "
                    "measured angular separations to the measured host "
                    "distances; under chance projection theta and D are "
                    "unrelated, so a true pairing tighter than random "
                    "indicates a preferred physical scale."),
            },
        }

    # ---------------------------------------------------------------
    # Test 5: paired-redshift excess
    # ---------------------------------------------------------------
    def _field_quasar_census(self, gpos, radius_arcmin):
        """All confirmed quasar-class Milliquas objects in a cone —
        the full-field pairing census (not only catalogued members)."""
        from astroquery.vizier import Vizier
        import warnings as _w
        from astroquery.exceptions import NoResultsWarning
        viz = Vizier(row_limit=-1, timeout=60)
        res = None
        for attempt in (1, 2):
            try:
                with _w.catch_warnings():
                    _w.simplefilter("ignore", category=NoResultsWarning)
                    res = viz.query_region(
                        gpos, radius=radius_arcmin * u.arcmin,
                        catalog=MILLIQUAS_CATALOG)
                break
            except Exception as e:
                if attempt == 2:
                    self.logger.info(
                        f"Milliquas census query failed after retry: "
                        f"{str(e).strip().splitlines()[0]}")
                    return None
                time.sleep(5.0)
        if res is None or MILLIQUAS_CATALOG not in res.keys():
            return []
        out = []
        for r in res[MILLIQUAS_CATALOG]:
            if not is_confirmed_quasar(r["Type"]):
                continue
            try:
                zv = r["z"]
                zf = float(zv) if not np.ma.is_masked(zv) else np.nan
            except Exception:
                zf = np.nan
            if np.isfinite(zf):
                out.append({"name": str(r["Name"]), "z": zf,
                            "type": str(r["Type"])})
        return out

    def _paired_redshift(self):
        ctrl = self.members[self.members["field_kind"] == "control"]
        z_pool = ctrl["z"].to_numpy(dtype=float)
        z_pool = z_pool[np.isfinite(z_pool) & (z_pool > 0.01)]
        if len(z_pool) < 50:
            raise RuntimeError(
                "too few control-field redshifts for the empirical "
                "pairing null; check qso_field_members.csv")
        zmin, zmax = float(z_pool.min()), float(z_pool.max())

        def _min_dz(zs):
            zs = np.sort(np.asarray(zs, dtype=float))
            return float(np.min(np.diff(zs))) if len(zs) >= 2 else np.nan

        def _mc_p(n, dz_obs):
            if not np.isfinite(dz_obs) or n < 2:
                return np.nan
            draws = self.rng.choice(
                z_pool, size=(N_MC, int(n)), replace=True)
            draws.sort(axis=1)
            mins = np.diff(draws, axis=1).min(axis=1)
            return float((1.0 + np.sum(mins <= dz_obs)) / (1.0 + N_MC))

        member_tests = []
        census_tests = []
        multi = {
            pid: grp for pid, grp in self.positions.groupby("pair_id")
            if len(grp) >= 2
        }
        for pid, grp in multi.items():
            zs = grp["z"].to_numpy(dtype=float)
            dz = _min_dz(zs)
            member_tests.append({
                "pair_id": pid, "n_members": int(len(zs)),
                "member_z": [float(z) for z in zs],
                "min_pair_dz": dz,
                "p_mc_vs_field_z": _mc_p(len(zs), dz),
            })

            row = self.catalog[self.catalog["pair_id"] == pid].iloc[0]
            gpos = self._resolve_host(row["galaxy"])
            if gpos is None:
                continue
            r_field = max(
                grp["measured_separation_arcsec"].max() / 60.0, 6.0)
            cen = self._field_quasar_census(gpos, r_field)
            time.sleep(0.3)
            if cen is None or len(cen) < 2:
                census_tests.append({
                    "pair_id": pid, "status": "census_too_small"})
                continue
            zf = [q["z"] for q in cen]
            dz_f = _min_dz(zf)
            census_tests.append({
                "pair_id": pid,
                "radius_arcmin": r_field,
                "n_field_quasars": len(zf),
                "field_z": zf,
                "min_pair_dz": dz_f,
                "p_mc_vs_field_z": _mc_p(len(zf), dz_f),
            })

        def _fisher(ps):
            from scipy.stats import chi2
            ps = np.clip(np.asarray([p for p in ps
                                     if np.isfinite(p)]), 1e-16, 1.0)
            if len(ps) == 0:
                return np.nan, np.nan
            stat = float(-2.0 * np.sum(np.log(ps)))
            return stat, float(chi2.sf(stat, 2 * len(ps)))

        f_stat_m, f_p_m = _fisher(
            [t["p_mc_vs_field_z"] for t in member_tests])
        f_stat_c, f_p_c = _fisher(
            [t.get("p_mc_vs_field_z") for t in census_tests])
        return {
            "z_pool_n": int(len(z_pool)),
            "z_pool_range": [zmin, zmax],
            "n_mc": N_MC,
            "member_tests": member_tests,
            "field_census_tests": census_tests,
            "fisher_combined_member": {
                "chi2": f_stat_m, "p": f_p_m},
            "fisher_combined_census": {
                "chi2": f_stat_c, "p": f_p_c},
            "method": (
                "Null: members are independent draws from the "
                "empirical confirmed-quasar redshift distribution "
                "measured on the step_22 control fields (same "
                "catalogue, same selection function).  p_i = fraction "
                "of Monte-Carlo n_i-draws whose closest pair is at "
                "least as tight as observed.  The census variant "
                "repeats the test on every confirmed quasar in the "
                "field cone, not only the catalogued members."
            ),
        }

    # ---------------------------------------------------------------
    # Test 6: symmetric-configuration probability (two-member fields)
    # ---------------------------------------------------------------
    def _symmetric_configuration(self):
        out = []
        for pid, grp in self.positions.groupby("pair_id"):
            if len(grp) != 2:
                continue
            row = self.catalog[self.catalog["pair_id"] == pid].iloc[0]
            gpos = self._resolve_host(row["galaxy"])
            if gpos is None:
                continue
            c1 = SkyCoord(ra=grp.iloc[0].ra_deg * u.deg,
                          dec=grp.iloc[0].dec_deg * u.deg)
            c2 = SkyCoord(ra=grp.iloc[1].ra_deg * u.deg,
                          dec=grp.iloc[1].dec_deg * u.deg)
            pa1 = math.radians(_pa_east_of_north(gpos, c1))
            pa2 = math.radians(_pa_east_of_north(gpos, c2))
            phi = math.degrees(
                math.acos(np.clip(
                    math.cos(pa1) * math.cos(pa2)
                    + math.sin(pa1) * math.sin(pa2), -1.0, 1.0)))
            r1 = float(grp.iloc[0].measured_separation_arcsec)
            r2 = float(grp.iloc[1].measured_separation_arcsec)
            rho = min(r1, r2) / max(r1, r2)
            # Analytic nulls for two uniform points in the field disk:
            # opposition angle uniform on [0,180] -> p = (180-phi)/180;
            # radius CDF r^2/R^2 -> ratio density 2*rho on [0,1]
            # -> P(rho_obs' >= rho) = 1 - rho^2.
            p_opp = (180.0 - phi) / 180.0
            p_ratio = 1.0 - rho ** 2
            p_joint = p_opp * p_ratio

            # Monte Carlo cross-check of the analytic null
            r_max = max(r1, r2)
            us = self.rng.uniform(0, 2 * math.pi, (N_MC_GEOM, 2))
            rs = r_max * np.sqrt(
                self.rng.uniform(0, 1, (N_MC_GEOM, 2)))
            dphi = np.degrees(np.abs(us[:, 0] - us[:, 1]))
            dphi = np.minimum(dphi, 360.0 - dphi)
            rr = np.minimum(rs[:, 0], rs[:, 1]) / np.maximum(
                rs[:, 0], rs[:, 1])
            p_mc = float(np.mean(
                (dphi >= phi) & (rr >= rho)))
            out.append({
                "pair_id": pid,
                "members": [str(grp.iloc[0].resolved_name),
                            str(grp.iloc[1].resolved_name)],
                "opposition_angle_deg": phi,
                "deg_from_exact_opposition": 180.0 - phi,
                "radius_ratio": rho,
                "p_opposition": p_opp,
                "p_radius_ratio": p_ratio,
                "p_joint_analytic": p_joint,
                "p_joint_mc": p_mc,
            })
            time.sleep(0.2)
        return {
            "tests": out,
            "method": (
                "Two-member fields only.  Null: companions are two "
                "uniform-random points in the field disk bounded by "
                "the widest member.  Opposition angle is uniform on "
                "[0,180] deg; radius ratio has density 2*rho; the "
                "joint price is the product, verified by Monte Carlo.  "
                "Computed for all resolved two-member fields and "
                "interpreted only where a symmetric pairing is the "
                "published claim (NGC 4258, NGC 2639)."
            ),
        }

    # ---------------------------------------------------------------
    def run(self):
        print_status("Running geometric-coherence tests...", "PROCESS")
        self._load_inputs()

        xray = self._xray_pricing()
        print_status(
            f"Sigma_X = {xray['sigma_xray_deg2']:.2f} deg^-2 "
            f"({xray['n_xray_in_control_fields']}/"
            f"{xray['n_control_members']} control members X-ray "
            f"detected); selection-aware joint P = "
            f"{xray['joint_probability_selection_aware']:.2e}",
            "TEST")

        minor = self._minor_axis()
        s = minor["samples"]["all_companions"]
        print_status(
            f"Minor-axis anisotropy: {s['n_within_30deg']}/{s['n']} "
            f"within 30 deg (binomial p={s['p_binomial_vs_uniform']:.3g}; "
            f"V-test p={s['p_vtest_minor_axis']:.3g})",
            "TEST")

        ordering = self._radial_ordering()
        print_status(
            f"Radial z-ordering: {ordering['n_nearer_higher_z']}/"
            f"{ordering['n_ordered_member_pairs']} nearer=higher-z "
            f"(perm p={ordering['p_permutation_two_sided']:.3g}; "
            f"binomial p={ordering['p_binomial_two_sided']:.3g})",
            "TEST")

        halo = self._halo_scale()
        if "permutation_test" in halo:
            print_status(
                f"Halo-scale clustering: std(log10 kpc) = "
                f"{halo['std_log10_kpc_observed']:.3f}, permutation "
                f"p = {halo['permutation_test']['p_le_observed']:.4f}",
                "TEST")

        pairing = self._paired_redshift()
        print_status(
            f"Paired-redshift excess: Fisher p = "
            f"{pairing['fisher_combined_member']['p']:.3g} (members) / "
            f"{pairing['fisher_combined_census']['p']:.3g} (field census)",
            "TEST")

        sym = self._symmetric_configuration()
        for t in sym["tests"]:
            print_status(
                f"Symmetry {t['pair_id']}: phi={t['opposition_angle_deg']:.1f} "
                f"deg, ratio={t['radius_ratio']:.2f}, "
                f"joint p={t['p_joint_analytic']:.3g}",
                "TEST")

        summary = {
            "rng_seed": RNG_SEED,
            "xray_selection_pricing": xray,
            "minor_axis_anisotropy": minor,
            "radial_redshift_ordering": ordering,
            "halo_scale_clustering": halo,
            "paired_redshift_excess": pairing,
            "symmetric_configuration": sym,
        }
        # strip bulky per-object lists into CSVs
        pd.DataFrame(minor["companions"]).to_csv(
            self.data_processed / "companion_axis_offsets.csv",
            index=False)
        pd.DataFrame([
            {"pair_id": pid, **{k: v for k, v in info.items()
                               if k != "pair_id"}}
            for pid, info in minor["host_pa"].items()
        ]).to_csv(
            self.data_processed / "host_position_angles.csv",
            index=False)
        test_rows = [
            {"test": "xray_selection_repriced_joint_p",
             "value": xray["joint_probability_selection_aware"]},
            {"test": "minor_axis_binomial_p_all",
             "value": s["p_binomial_vs_uniform"]},
            {"test": "minor_axis_vtest_p_all",
             "value": s["p_vtest_minor_axis"]},
            {"test": "radial_ordering_binomial_p",
             "value": ordering["p_binomial_two_sided"]},
            {"test": "radial_ordering_permutation_p",
             "value": ordering["p_permutation_two_sided"]},
            {"test": "halo_scale_permutation_p",
             "value": halo.get("permutation_test", {}).get(
                 "p_le_observed")},
            {"test": "paired_redshift_fisher_p_members",
             "value": pairing["fisher_combined_member"]["p"]},
            {"test": "paired_redshift_fisher_p_census",
             "value": pairing["fisher_combined_census"]["p"]},
        ]
        for t in sym["tests"]:
            test_rows.append({
                "test": f"symmetric_config_joint_p:{t['pair_id']}",
                "value": t["p_joint_analytic"]})
        pd.DataFrame(test_rows).to_csv(
            self.data_processed / "geometric_coherence_tests.csv",
            index=False)

        json_path = self.results / "step_27_geometric_coherence.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")
        print_status("Geometric-coherence tests complete.", "SUCCESS")


if __name__ == "__main__":
    Step27GeometricCoherence().run()
