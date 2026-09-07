#!/usr/bin/env python3
"""
Step 24: Forward Galaxy-Quasar Cross-Correlation
================================================
The population-level association test that removes the
morphology-selection critique of the eight-pair catalogue: a
predefined parent sample of bright nearby galaxies is
cross-matched against the confirmed quasar population, and the
quasar surface density around those galaxies is compared with
seeded control fields at matched sky positions.

Parent sample: the 2MRS all-sky redshift survey (VizieR
J/ApJS/199/26, table3; Huchra et al. 2012), magnitude-limited at
K_s = 11.75.  Selection is by predefined criteria only —
500 < cz < 10000 km/s (the velocity range spanned by the Arp
hosts) and Galactic latitude |b| > 15 deg — and makes no
reference to the Arp catalogue.  The Arp hosts satisfy the
same cuts and remain in the sample.

Quasar catalogue: Milliquas v7.2 (VizieR VII/290; Flesch 2021),
restricted to the same confirmed quasar-class objects used in
step_22 (type-I broad-line Q, Seyfert/AGN A, BL Lac B,
narrow-line K; N/L classes and candidates excluded).

Cross-matching: CDS XMatch bulk service at a fixed maximum
separation of 180 arcsec, returning every galaxy-quasar pair with
its angular separation (angDist); matches closer than 3 arcsec
are removed as self-identifications of the parent's own AGN
catalogue entry.  Controls: four positions per parent galaxy,
offset by a fixed 1.5 deg at seeded random position angles
(declination-corrected), cross-matched identically — the same
completeness-robust paired null used in step_22, now applied to
the full parent sample.

Statistics: per-annulus (0-60, 60-120, 120-180 arcsec) quasar
overdensity delta = mean_host/mean_control - 1 with the
uncertainty propagated from the per-field count variances; a
galaxy-unit bootstrap (resampling galaxies with replacement) on
the mean paired difference within 180 arcsec; and a per-galaxy
empirical tail combined across the sample by Fisher's method.

This step requires network access to CDS (VizieR, XMatch) and
fails loudly if either is unreachable.

Outputs:
    data/processed/xcorr_parent_sample.csv
    data/processed/xcorr_annulus_statistics.csv
    results/outputs/step_24_forward_crosscorrelation.json
    results/figures/step_24_radial_overdensity.png
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.jsonio import json_safe

TWOMRS_CATALOG = "J/ApJS/199/26/table3"   # Huchra et al. 2012
MILLIQUAS_CATALOG = "vizier:VII/290/catalog"  # Flesch 2021 (XMatch id)

CZ_MIN_KMS = 500            # parent velocity range (km/s)
CZ_MAX_KMS = 10000
GLAT_MIN_DEG = 15.0         # |b| cut
N_CONTROLS = 4              # control fields per parent galaxy
CONTROL_OFFSET_DEG = 1.5    # control offset distance
MAX_SEP_ARCSEC = 180.0      # XMatch radius
SELF_MATCH_ARCSEC = 3.0     # self-identification exclusion
ANNULI_ARCSEC = [(0.0, 60.0), (60.0, 120.0), (120.0, 180.0)]
PAIR_OPPOSITION_DEG = 135.0  # folded PA difference >= this = flanking
PAIR_DZ_MAX = 0.1            # matching-redshift tolerance
N_BOOTSTRAP = 2000
RNG_SEED = 20260924

CONFIRMED_CLASSES = ("Q", "A", "B", "K")


def _is_confirmed(type_str):
    t = str(type_str).strip()
    return len(t) > 0 and t[0] in CONFIRMED_CLASSES


class Step24ForwardCrossCorrelation:
    """Step 24: Predefined-sample galaxy-quasar cross-correlation."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.data_raw = self.root / "data" / "raw"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.data_raw,
                  self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_24",
            log_file_path=self.logs
            / "step_24_forward_crosscorrelation.log",
        )
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Data acquisition
    # ------------------------------------------------------------------
    def _load_parent_sample(self):
        """2MRS galaxies meeting the predefined cz and |b| cuts."""
        cache = self.data_raw / "2mrs_parent_sample.csv"
        if cache.exists():
            df = pd.read_csv(cache)
            print_status(
                f"Parent sample loaded from cache ({len(df)} galaxies).",
                "INFO",
            )
            return df

        from astroquery.vizier import Vizier

        viz = Vizier(
            columns=["ID", "RAJ2000", "DEJ2000", "cz", "Kcmag"],
            row_limit=-1, timeout=300,
        )
        res = viz.query_constraints(
            catalog=TWOMRS_CATALOG,
            cz=f"{CZ_MIN_KMS}..{CZ_MAX_KMS}",
        )
        if not res or len(res) == 0:
            raise RuntimeError(
                "2MRS query returned nothing; VizieR unreachable. "
                "Refusing to build parent sample."
            )
        t = res[0]
        df = pd.DataFrame({
            "id": [str(x) for x in t["ID"]],
            "ra_deg": np.asarray(t["RAJ2000"], dtype=float),
            "dec_deg": np.asarray(t["DEJ2000"], dtype=float),
            "cz_kms": np.asarray(t["cz"], dtype=float),
        })
        coords = SkyCoord(df["ra_deg"].to_numpy() * u.deg,
                          df["dec_deg"].to_numpy() * u.deg,
                          frame="icrs")
        df["glat_deg"] = coords.galactic.b.deg
        df = df[np.abs(df["glat_deg"]) > GLAT_MIN_DEG].reset_index(drop=True)
        df.to_csv(cache, index=False)
        print_status(
            f"2MRS parent sample: {len(df)} galaxies "
            f"({CZ_MIN_KMS} < cz < {CZ_MAX_KMS} km/s, "
            f"|b| > {GLAT_MIN_DEG:.0f} deg).",
            "TEST",
        )
        return df

    def _control_positions(self, df):
        """K seeded offset positions per parent galaxy."""
        rng = np.random.default_rng(RNG_SEED)
        rows = []
        pas = rng.uniform(0.0, 360.0, size=(len(df), N_CONTROLS))
        for i, r in enumerate(df.itertuples()):
            for k in range(N_CONTROLS):
                pa = np.radians(pas[i, k])
                dec_off = CONTROL_OFFSET_DEG * np.sin(pa)
                ra_off = (CONTROL_OFFSET_DEG * np.cos(pa)
                          / np.cos(np.radians(r.dec_deg)))
                rows.append({
                    "parent_index": i,
                    "control_index": k,
                    "ra_deg": (r.ra_deg + ra_off) % 360.0,
                    "dec_deg": np.clip(r.dec_deg + dec_off, -89.9, 89.9),
                })
        return pd.DataFrame(rows)

    def _xmatch(self, positions, label):
        """Bulk XMatch of positions against Milliquas at MAX_SEP."""
        from astroquery.xmatch import XMatch

        cache = self.data_raw / f"xmatch_{label}.csv"
        if cache.exists():
            df = pd.read_csv(cache)
            if "qso_ra" in df.columns:
                return df
            cache.unlink()

        cat1 = Table({
            "idx": np.arange(len(positions)),
            "ra": np.asarray(positions["ra_deg"], dtype=float),
            "dec": np.asarray(positions["dec_deg"], dtype=float),
        })
        res = None
        for attempt in (1, 2, 3):
            try:
                res = XMatch.query(
                    cat1=cat1, cat2=MILLIQUAS_CATALOG,
                    max_distance=MAX_SEP_ARCSEC * u.arcsec,
                    colRA1="ra", colDec1="dec",
                )
                break
            except Exception as e:
                short = str(e).strip().splitlines()[0]
                if attempt < 3:
                    self.logger.info(
                        f"XMatch {label} attempt {attempt}/3 failed "
                        f"({short}); retrying."
                    )
                else:
                    self.logger.warning(
                        f"XMatch {label} failed after 3 attempts: {short}"
                    )
                time.sleep(3.0)
        if res is None:
            raise RuntimeError(
                f"XMatch bulk cross-match failed for {label}; "
                "archive unreachable."
            )
        dfm = pd.DataFrame({
            "idx": np.asarray(res["idx"], dtype=int),
            "angdist_arcsec": np.asarray(res["angDist"], dtype=float),
            "qso_name": [str(x) for x in res["Name"]],
            "qso_type": [str(x) for x in res["Type"]],
            "qso_ra": np.asarray(res["RAJ2000"], dtype=float),
            "qso_dec": np.asarray(res["DEJ2000"], dtype=float),
            "qso_z": np.asarray(res["z"], dtype=float),
        })
        dfm.to_csv(cache, index=False)
        print_status(
            f"XMatch {label}: {len(dfm)} raw matches within "
            f"{MAX_SEP_ARCSEC:.0f} arcsec.", "TEST",
        )
        return dfm

    def _paired_signature(self, matches, positions):
        """Arp's pairing signature: quasar pairs flanking the field
        centre on opposite sides at matching redshifts.  For every
        field, confirmed quasars (>= SELF_MATCH arcsec, finite z)
        are combined into unordered pairs; a pair qualifies when the
        position angles differ by PAIR_OPPOSITION_DEG or more
        (folded opposition) and the redshifts agree within
        PAIR_DZ_MAX.  Returns per-field qualifying-pair counts."""
        centers = positions[["ra_deg", "dec_deg"]].to_numpy(dtype=float)
        m = matches[
            (matches["angdist_arcsec"] >= SELF_MATCH_ARCSEC)
            & (matches["qso_type"].map(_is_confirmed))
            & np.isfinite(matches["qso_z"])
            & (matches["qso_z"] > 0.01)
        ]
        counts = np.zeros(len(positions))
        for fid, grp in m.groupby("idx"):
            if len(grp) < 2:
                continue
            cra, cdec = centers[fid]
            # position angle of each quasar about the field centre
            dra = np.radians(grp["qso_ra"].to_numpy() - cra)
            ddec = np.radians(grp["qso_dec"].to_numpy() - cdec)
            pa = np.degrees(np.arctan2(
                dra * np.cos(np.radians(cdec)), ddec)) % 360.0
            z = grp["qso_z"].to_numpy()
            names = grp["qso_name"].to_numpy()
            n_pairs = 0
            for i in range(len(grp)):
                for j in range(i + 1, len(grp)):
                    if names[i] == names[j]:
                        continue
                    dpa = abs(pa[i] - pa[j]) % 360.0
                    dpa = min(dpa, 360.0 - dpa)   # folded to [0,180]
                    if dpa < PAIR_OPPOSITION_DEG:
                        continue
                    if abs(z[i] - z[j]) > PAIR_DZ_MAX:
                        continue
                    n_pairs += 1
            counts[fid] = n_pairs
        return counts

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------
    def _annulus_counts(self, matches, n_fields):
        """Confirmed-quasar counts per field per annulus."""
        m = matches[
            (matches["angdist_arcsec"] >= SELF_MATCH_ARCSEC)
            & (matches["qso_type"].map(_is_confirmed))
        ]
        counts = np.zeros((n_fields, len(ANNULI_ARCSEC)))
        for j, (lo, hi) in enumerate(ANNULI_ARCSEC):
            sel = m[(m["angdist_arcsec"] >= lo)
                    & (m["angdist_arcsec"] < hi)]
            c = sel.groupby("idx").size()
            counts[c.index.to_numpy(), j] = c.to_numpy()
        return counts

    def run(self):
        print_status(
            "Running forward galaxy-quasar cross-correlation...", "PROCESS",
        )

        parents = self._load_parent_sample()
        n_parents = len(parents)
        controls = self._control_positions(parents)
        n_ctrl = len(controls)

        m_gal = self._xmatch(parents, "parents")
        m_ctl = self._xmatch(controls, "controls")

        c_gal = self._annulus_counts(m_gal, n_parents)
        c_ctl = self._annulus_counts(m_ctl, n_ctrl)
        c_ctl_by_parent = c_ctl.reshape(n_parents, N_CONTROLS,
                                        len(ANNULI_ARCSEC))

        # Active-host subset: Arp's claim was specifically about
        # active/peculiar galaxies.  A parent is classed "active" when
        # its own position is cross-identified in Milliquas as a
        # confirmed quasar-class object (the self-identification that
        # the main analysis excludes); these parents are Seyfert-class
        # hosts by construction.  The same annulus counts are computed
        # on the subset against each galaxy's own controls.
        self_m = m_gal[
            (m_gal["angdist_arcsec"] < SELF_MATCH_ARCSEC)
            & (m_gal["qso_type"].map(_is_confirmed))
        ]
        active_idx = np.unique(self_m["idx"].to_numpy())
        print_status(
            f"Active-host subset (self-identified AGN class): "
            f"{len(active_idx)} of {n_parents} parents.", "TEST",
        )

        # Per-annulus overdensity with propagated per-field variance
        annulus_rows = []
        for j, (lo, hi) in enumerate(ANNULI_ARCSEC):
            g = c_gal[:, j]
            c = c_ctl[:, j]
            mean_g, mean_c = float(g.mean()), float(c.mean())
            delta = mean_g / mean_c - 1.0 if mean_c > 0 else np.nan
            # Propagated variance of delta = mean_g/mean_c - 1:
            #   var(delta) = var(mean_g)/mean_c^2
            #              + (mean_g/mean_c^2)^2 * var(mean_c)
            # where var(mean_g) = g.var/n_parents, var(mean_c) =
            # c.var/n_ctrl.  Each term carries exactly one factor of
            # 1/mean_c^2 from the partial derivative; the control
            # term must not be divided by mean_c^2 a second time.
            if mean_c > 0:
                var = (g.var(ddof=1) / n_parents / mean_c ** 2
                       + (mean_g / mean_c ** 2) ** 2
                       * c.var(ddof=1) / n_ctrl)
            else:
                var = np.nan
            sigma = delta / np.sqrt(var) if np.isfinite(var) and var > 0 \
                else np.nan
            annulus_rows.append({
                "annulus_arcsec": f"{lo:.0f}-{hi:.0f}",
                "mean_per_galaxy": mean_g,
                "mean_per_control": mean_c,
                "overdensity_delta": float(delta),
                "delta_sigma": float(sigma),
                "n_gal_matches": int(g.sum()),
                "n_ctrl_matches": int(c.sum()),
            })
            print_status(
                f"Annulus {lo:.0f}-{hi:.0f} arcsec: "
                f"delta = {delta:+.3f} ({sigma:+.1f} sigma; "
                f"{int(g.sum())} vs {int(c.sum())} matches)", "TEST",
            )

        # Cumulative 0-180 arcsec paired test: each galaxy vs its own
        # four controls; galaxy-unit bootstrap on the mean difference.
        inner = c_gal.sum(axis=1)
        inner_ctrl = c_ctl_by_parent.sum(axis=2)
        paired_diff = inner - inner_ctrl.mean(axis=1)
        mean_diff = float(paired_diff.mean())

        rng = np.random.default_rng(RNG_SEED + 1)
        boot = np.empty(N_BOOTSTRAP)
        for b in range(N_BOOTSTRAP):
            idx = rng.integers(0, n_parents, n_parents)
            boot[b] = paired_diff[idx].mean()
        ci = np.percentile(boot, [2.5, 97.5])
        p_boot = float((np.sum(boot <= 0) + 1) / (N_BOOTSTRAP + 1))

        # Per-galaxy empirical tail (own controls) -> Fisher combined
        own_mean = inner_ctrl.mean(axis=1)
        p_own = (1.0 + np.sum(inner_ctrl >= inner[:, None], axis=1)) \
            / (1.0 + N_CONTROLS)
        from scipy.stats import chi2 as _chi2, wilcoxon as _wilcoxon
        fisher_stat = float(-2.0 * np.sum(
            np.log(np.clip(p_own, 1e-12, 1.0))))
        p_fisher = float(1.0 - _chi2.cdf(fisher_stat, 2 * n_parents))
        nz = paired_diff[np.abs(paired_diff) > 1e-12]
        try:
            import warnings as _warnings
            with _warnings.catch_warnings(record=True) as caught:
                _warnings.simplefilter("always")
                w_res = _wilcoxon(nz, alternative="greater")
            for _w in caught:
                self.logger.info(
                    f"wilcoxon note: {str(_w.message).strip()}"
                )
            p_wilcoxon = float(w_res.pvalue)
        except Exception:
            p_wilcoxon = None

        # Active-subset statistics: same inner-aperture comparison
        # restricted to self-identified AGN parents.
        inner_a = c_gal[active_idx].sum(axis=1)
        inner_ctrl_a = c_ctl_by_parent[active_idx].sum(axis=2)
        if len(active_idx) >= 20:
            diff_a = inner_a - inner_ctrl_a.mean(axis=1)
            boot_a = np.empty(N_BOOTSTRAP)
            for b in range(N_BOOTSTRAP):
                bidx = rng.integers(0, len(active_idx), len(active_idx))
                boot_a[b] = diff_a[bidx].mean()
            ci_a = np.percentile(boot_a, [2.5, 97.5])
            p_boot_a = float(
                (np.sum(boot_a <= 0) + 1) / (N_BOOTSTRAP + 1))
            delta_a = float(
                inner_a.mean() / inner_ctrl_a.mean() - 1.0)
            active_stats = {
                "n_active_hosts": int(len(active_idx)),
                "mean_qso_per_galaxy": float(inner_a.mean()),
                "mean_qso_per_control": float(inner_ctrl_a.mean()),
                "overdensity_delta": delta_a,
                "bootstrap_ci95_mean_diff": [
                    float(ci_a[0]), float(ci_a[1])],
                "bootstrap_p_one_sided": p_boot_a,
                "definition": (
                    "parent galaxies self-identified in Milliquas as "
                    "confirmed quasar-class objects (Seyfert-class "
                    "hosts); companion counts exclude the <3 arcsec "
                    "self-match"
                ),
            }
            print_status(
                f"Active subset 0-180 arcsec: {inner_a.mean():.4f} vs "
                f"{inner_ctrl_a.mean():.4f}; delta = {delta_a:+.3f}; "
                f"bootstrap p = {p_boot_a:.4f}", "TEST",
            )
        else:
            active_stats = {
                "n_active_hosts": int(len(active_idx)),
                "note": "subset too small for paired inference",
            }

        # Arp's pairing signature: opposite-side, matching-redshift
        # quasar pairs.  Rare enough per field that even a modest
        # excess is decisive; tested against the same controls.
        pairs_gal = self._paired_signature(m_gal, parents)
        pairs_ctl = self._paired_signature(m_ctl, controls)
        n_fields_pair_gal = int((pairs_gal > 0).sum())
        n_fields_pair_ctl = int((pairs_ctl > 0).sum())
        n_pairs_gal = float(pairs_gal.sum())
        n_pairs_ctl = float(pairs_ctl.sum())
        lam_ctl = n_pairs_ctl / N_CONTROLS   # expected per-field-scale
        from scipy.stats import poisson as _poisson
        p_pairs = float(1.0 - _poisson.cdf(int(n_pairs_gal) - 1,
                                           lam_ctl)) \
            if lam_ctl > 0 else np.nan
        rate_ratio = (n_pairs_gal / lam_ctl) if lam_ctl > 0 else np.nan
        pair_stats = {
            "criterion": (
                f"two confirmed quasars within {MAX_SEP_ARCSEC:.0f} "
                f"arcsec, folded PA difference >= "
                f"{PAIR_OPPOSITION_DEG:.0f} deg, |dz| <= {PAIR_DZ_MAX}"
            ),
            "n_pairs_parents": int(n_pairs_gal),
            "n_pairs_controls": int(n_pairs_ctl),
            "n_fields_with_pair_parents": n_fields_pair_gal,
            "n_fields_with_pair_controls": n_fields_pair_ctl,
            "expected_from_controls": float(lam_ctl),
            "rate_ratio_parents_over_controls": (
                float(rate_ratio) if np.isfinite(rate_ratio) else None),
            "poisson_p_excess": p_pairs,
        }
        print_status(
            f"Pairing signature: {int(n_pairs_gal)} qualifying pairs in "
            f"{n_fields_pair_gal} parent fields vs "
            f"{int(n_pairs_ctl)} in {n_fields_pair_ctl} control fields "
            f"(expected {lam_ctl:.1f}; Poisson p = {p_pairs:.4f})",
            "TEST",
        )

        area_deg2 = np.pi * (MAX_SEP_ARCSEC / 3600.0) ** 2
        dens_gal = float(inner.mean() / area_deg2)
        dens_ctl = float(inner_ctrl.mean() / area_deg2)

        print_status(
            f"0-180 arcsec: {inner.mean():.4f} +/- per galaxy vs "
            f"{inner_ctrl.mean():.4f} per control; "
            f"delta = {dens_gal / dens_ctl - 1:+.3f}; "
            f"bootstrap p = {p_boot:.4f}", "TEST",
        )

        pd.DataFrame(annulus_rows).to_csv(
            self.data_processed / "xcorr_annulus_statistics.csv",
            index=False,
        )
        parents.to_csv(
            self.data_processed / "xcorr_parent_sample.csv", index=False)

        summary = {
            "parent_sample": (
                f"2MRS table3 (VizieR J/ApJS/199/26; Huchra et al. 2012), "
                f"{CZ_MIN_KMS} < cz < {CZ_MAX_KMS} km/s, "
                f"|b| > {GLAT_MIN_DEG:.0f} deg"
            ),
            "n_parent_galaxies": int(n_parents),
            "n_control_fields": int(n_ctrl),
            "controls_per_galaxy": N_CONTROLS,
            "control_offset_deg": CONTROL_OFFSET_DEG,
            "max_separation_arcsec": MAX_SEP_ARCSEC,
            "self_match_exclusion_arcsec": SELF_MATCH_ARCSEC,
            "quasar_definition": (
                "Milliquas v7.2 confirmed classes Q, A, B, K (same "
                "definition as step_22); matches closer than "
                f"{SELF_MATCH_ARCSEC:.0f} arcsec removed as the parent's "
                "own AGN entry."
            ),
            "annuli": annulus_rows,
            "inner_0_180_arcsec": {
                "mean_qso_per_galaxy": float(inner.mean()),
                "mean_qso_per_control": float(inner_ctrl.mean()),
                "surface_density_deg2_galaxy": dens_gal,
                "surface_density_deg2_control": dens_ctl,
                "overdensity_delta": float(dens_gal / dens_ctl - 1.0),
                "paired_mean_difference": mean_diff,
                "bootstrap_ci95": [float(ci[0]), float(ci[1])],
                "bootstrap_p_one_sided": p_boot,
                "fisher_chi2": fisher_stat,
                "p_fisher_combined": p_fisher,
                "p_wilcoxon_paired": p_wilcoxon,
            },
            "active_host_subset": active_stats,
            "pairing_signature": pair_stats,
            "rng_seed": RNG_SEED,
            "method": (
                "Predefined parent sample (2MRS, cz and |b| cuts only; "
                "no reference to the Arp catalogue) bulk cross-matched "
                "to confirmed Milliquas quasars via CDS XMatch at "
                f"{MAX_SEP_ARCSEC:.0f} arcsec; {N_CONTROLS} seeded "
                f"1.5-deg offset controls per galaxy cross-matched "
                "identically. Overdensity per annulus with per-field "
                "count-variance propagation; paired galaxy-vs-own-"
                "controls difference tested by galaxy-unit bootstrap, "
                "Fisher-combined per-galaxy empirical tails, and a "
                "paired Wilcoxon test."
            ),
        }
        json_path = self.results / "step_24_forward_crosscorrelation.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")

        self._make_figure(annulus_rows, dens_gal, dens_ctl)
        print_status("Forward cross-correlation complete.", "SUCCESS")

    def _make_figure(self, annulus_rows, dens_gal, dens_ctl):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        centers = [(lo + hi) / 2 / 60.0 for lo, hi in
                   [tuple(map(float, r["annulus_arcsec"].split("-")))
                    for r in annulus_rows]]
        deltas = [r["overdensity_delta"] for r in annulus_rows]
        errs = [abs(r["overdensity_delta"] / r["delta_sigma"])
                if r["delta_sigma"] and np.isfinite(r["delta_sigma"])
                else 0.0 for r in annulus_rows]

        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        ax.errorbar(centers, deltas, yerr=errs, fmt="o-", color="#1a5276",
                    capsize=4, lw=1.5, ms=6,
                    label="2MRS parent sample")
        ax.axhline(0.0, color="grey", ls="--", lw=1)
        ax.set_xlabel("Angular separation (arcmin)")
        ax.set_ylabel(r"Quasar overdensity $\delta$")
        ax.set_title(
            "Galaxy$-$quasar cross-correlation: predefined 2MRS sample")
        ax.legend(frameon=False)
        fig.tight_layout()
        out = self.figures / "step_24_radial_overdensity.png"
        fig.savefig(out, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print_status(f"Saved figure: {out}", "SUCCESS")


if __name__ == "__main__":
    Step24ForwardCrossCorrelation().run()
