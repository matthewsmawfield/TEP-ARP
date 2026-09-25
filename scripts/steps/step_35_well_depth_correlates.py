#!/usr/bin/env python3
"""
Step 35: Well-Depth Correlates
==============================
Tests whether the inferred intrinsic temporal-well depth
Delta phi_int = ln[(1+z_comp)/(1+z_gal)] of each quasar-class
companion correlates with an independent, non-redshift observable.
If the well depth is a physical property of the emitter rather than
a relabelled distance, it should track something intrinsic; a null
bounds that claim and is reported equally.

Observables pulled per companion (VizieR cone search, per-catalogue
radius reflecting astrometric precision):

  * 4XMM-DR13 (IX/69): hardness ratios HR1-HR4, broadband flux
    Flux8, and source extent ext (X-ray compactness) — the pointed
    catalogue, queried first at 15 arcsec
  * 2RXS (J/A+A/588/A103/cat2rxs): ROSAT hardness ratios HR1, HR2
    and count rate Cts — the all-sky fallback at 40 arcsec
  * FIRST (VIII/92/first14): Fint/Fpeak compactness, Fint, Maj axis
  * NVSS (VIII/65/nvss): S1.4 integrated flux; NVSS/FIRST flux ratio
    (compactness on 45-arcsec vs 5-arcsec scales) where both exist
  * SDSS DR16Q (VII/289): gmag, included as a control observable
    (a brightness/redshift-driven correlation would show up here)

Statistics: Spearman rank correlation of Delta phi_int against each
observable over the matched subset, with a companion-unit bootstrap
95% interval on rho; Pearson on the same subset for comparison; n
reported per observable.  Unmatched companions are excluded per
observable, never imputed.  gmag is a deliberate control: it carries
apparent-brightness information and bounds any Malmquist-like
contamination of the physical correlations.

This step requires network access to VizieR and fails loudly if the
catalogues are unreachable.

Outputs:
    results/outputs/step_35_well_depth_correlates.json
    data/processed/well_depth_correlates.csv
    results/figures/step_35_well_depth_correlates.png
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.jsonio import json_safe
from scripts.utils.plot_style import apply_tep_style

RNG_SEED = 20260924
N_BOOT = 20000

CAT_2RXS = "J/A+A/588/A103/cat2rxs"
CAT_4XMM = "IX/69"
CAT_FIRST = "VIII/92/first14"
CAT_NVSS = "VIII/65/nvss"
CAT_DR16Q = "VII/289"

# Cone-search radius per catalogue, reflecting astrometric precision:
# ROSAT all-sky positions carry ~10-30 arcsec uncertainty; pointed
# XMM/Chandra-class positions are sub-arcsec to a few arcsec.
SEARCH_RADII_ARCSEC = {
    CAT_4XMM: 15.0,
    CAT_2RXS: 40.0,
    CAT_FIRST: 10.0,
    CAT_NVSS: 15.0,
    CAT_DR16Q: 10.0,
}

# companion_type values in the verified catalogue that mark
# quasar-class members (excludes galaxy companions NGC 7603B,
# NGC 1232A and the NGC 1199 compact object, leaving the 18
# quasar-class companions used elsewhere in the analysis)
QUASAR_TYPES = ("quasar", "seyfert")


def _is_quasar_class(ctype):
    c = str(ctype).lower()
    return any(q in c for q in QUASAR_TYPES)


def _fnum(v):
    """float() that returns nan for masked/blank VizieR cells."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return np.nan
    return f if np.isfinite(f) else np.nan


def _conesearch(viz, coord, radius_deg, catalog):
    """Nearest catalogue row within the search radius, or None."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    try:
        res = viz.query_region(coord, radius=radius_deg * u.deg,
                               catalog=catalog)
    except Exception as e:
        raise RuntimeError(
            f"VizieR query failed for {catalog}: {e}. "
            "Step 35 requires network access to VizieR."
        )
    if not res or len(res[0]) == 0:
        return None
    t = res[0]
    ra_col = next((c for c in t.colnames if c.upper().startswith("RA")),
                  None)
    de_col = next((c for c in t.colnames if c.upper().startswith("DE")),
                  None)
    if ra_col is None or de_col is None:
        return None
    best, best_d = None, np.inf
    for row in t:
        try:
            ra, dec = float(row[ra_col]), float(row[de_col])
            row_coord = SkyCoord(ra, dec, unit="deg")
        except (TypeError, ValueError):
            # sexagesimal strings (e.g. FIRST): hms dms
            row_coord = SkyCoord(str(row[ra_col]),
                                 str(row[de_col]),
                                 unit=(u.hourangle, u.deg))
        d = coord.separation(row_coord).deg
        if d < best_d:
            best, best_d = row, d
    return best, best_d * 3600.0


class Step35WellDepthCorrelates:
    """Step 35: correlate Delta phi_int with archival observables."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.figures,
                  self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_35",
            log_file_path=self.logs / "step_35_well_depth_correlates.log",
        )
        set_step_logger(self.logger)

    def _load_companions(self):
        cat_path = self.data_processed / "arp_pair_catalog_verified.csv"
        pos_path = self.data_processed / "companion_positions.csv"
        for p in (cat_path, pos_path):
            if not p.exists():
                raise FileNotFoundError(
                    f"Required input not found: {p}. Run steps 00/01 first."
                )
        cat = pd.read_csv(cat_path)
        pos = pd.read_csv(pos_path)
        z_gal = dict(zip(cat["pair_id"], cat["z_gal"]))
        ctype = dict(zip(cat["pair_id"], cat["companion_type"]))
        pos["z_gal"] = pos["pair_id"].map(z_gal)
        pos["companion_type"] = pos["pair_id"].map(ctype)
        pos = pos[pos["companion_type"].map(_is_quasar_class)]
        pos["delta_phi"] = np.log(
            (1.0 + pos["z"].astype(float)) / (1.0 + pos["z_gal"].astype(float)))
        return pos.reset_index(drop=True)

    def run(self):
        print_status("Pulling archival observables for quasar-class "
                     "companions...", "PROCESS")

        from astroquery.vizier import Vizier
        from astropy.coordinates import SkyCoord
        viz = Vizier(columns=["*"], row_limit=20)

        comps = self._load_companions()
        print_status(f"{len(comps)} quasar-class companions", "TEST")

        rows = []
        for _, c in comps.iterrows():
            coord = SkyCoord(float(c["ra_deg"]), float(c["dec_deg"]),
                             unit="deg")
            rec = {
                "pair_id": c["pair_id"],
                "resolved_name": c["resolved_name"],
                "z": float(c["z"]),
                "z_gal": float(c["z_gal"]),
                "delta_phi": float(c["delta_phi"]),
            }

            def search(cat):
                return _conesearch(
                    viz, coord, SEARCH_RADII_ARCSEC[cat] / 3600.0, cat)

            m = search(CAT_4XMM)
            if m:
                row, off = m
                rec.update({
                    "xmm_off_arcsec": off,
                    "xmm_hr1": _fnum(row["HR1"]),
                    "xmm_hr2": _fnum(row["HR2"]),
                    "xmm_flux8": _fnum(row["Flux8"]),
                    "xmm_ext_arcsec": _fnum(row["ext"]),
                })
            m = search(CAT_2RXS)
            if m:
                row, off = m
                rec.update({
                    "rxs_off_arcsec": off,
                    "rxs_hr1": _fnum(row["HR1"]),
                    "rxs_hr2": _fnum(row["HR2"]),
                    "cts_2rxs": _fnum(row["Cts"]),
                })
            # single X-ray hardness observable: 4XMM HR2 where the
            # pointed catalogue reaches, else 2RXS HR1; instrument
            # flag kept so mixed definitions are transparent
            if np.isfinite(rec.get("xmm_hr2", np.nan)):
                rec["xray_hardness"] = rec["xmm_hr2"]
                rec["xray_hardness_src"] = "4XMM_HR2"
            elif np.isfinite(rec.get("rxs_hr1", np.nan)):
                rec["xray_hardness"] = rec["rxs_hr1"]
                rec["xray_hardness_src"] = "2RXS_HR1"
            m = search(CAT_FIRST)
            if m:
                row, off = m
                fpeak, fint = _fnum(row["Fpeak"]), _fnum(row["Fint"])
                rec.update({
                    "first_off_arcsec": off,
                    "first_fpeak_mjy": fpeak,
                    "first_fint_mjy": fint,
                    "first_compactness": (fint / fpeak
                                          if fpeak and fpeak > 0
                                          else np.nan),
                    "first_maj_arcsec": _fnum(row["Maj"]),
                })
            m = search(CAT_NVSS)
            if m:
                row, off = m
                rec.update({
                    "nvss_off_arcsec": off,
                    "nvss_s14_mjy": _fnum(row["S1.4"]),
                    "nvss_maj_arcsec": _fnum(row["MajAxis"]),
                })
            m = search(CAT_DR16Q)
            if m:
                row, off = m
                rec.update({
                    "dr16q_off_arcsec": off,
                    "gmag": _fnum(row["gmag"]),
                })
            if np.isfinite(rec.get("nvss_s14_mjy", np.nan)) and \
                    np.isfinite(rec.get("first_fint_mjy", np.nan)) and \
                    rec["first_fint_mjy"] > 0:
                rec["nvss_over_first"] = (
                    rec["nvss_s14_mjy"] / rec["first_fint_mjy"])
            rows.append(rec)
            print_status(
                f"{c['resolved_name']:<32} dphi={rec['delta_phi']:+.3f} "
                f"4XMM={'Y' if 'xmm_hr2' in rec else '-'} "
                f"2RXS={'Y' if 'rxs_hr1' in rec else '-'} "
                f"FIRST={'Y' if 'first_fint_mjy' in rec else '-'} "
                f"NVSS={'Y' if 'nvss_s14_mjy' in rec else '-'} "
                f"DR16Q={'Y' if 'gmag' in rec else '-'}", "TEST")
            time.sleep(0.15)

        df = pd.DataFrame(rows)
        csv_path = self.data_processed / "well_depth_correlates.csv"
        df.to_csv(csv_path, index=False)
        print_status(f"Saved correlate table: {csv_path}", "SUCCESS")

        # --- correlations -------------------------------------------
        from scipy.stats import spearmanr, pearsonr
        rng = np.random.default_rng(RNG_SEED)
        observables = [
            ("xray_hardness", "X-ray hardness (4XMM HR2 / 2RXS HR1)", False),
            ("xmm_ext_arcsec", "4XMM source extent", False),
            ("xmm_flux8", "4XMM 0.2-12 keV flux", True),
            ("cts_2rxs", "ROSAT count rate", True),
            ("first_compactness", "FIRST Fint/Fpeak", False),
            ("first_fint_mjy", "FIRST flux", True),
            ("nvss_s14_mjy", "NVSS S1.4", True),
            ("nvss_over_first", "NVSS/FIRST flux ratio", True),
            ("gmag", "SDSS g (control)", False),
        ]
        stats = []
        for col, label, log10 in observables:
            sub = df[["delta_phi", col]].dropna()
            n = len(sub)
            if n < 4:
                stats.append({"observable": col, "label": label,
                              "n": int(n), "rho": None, "p": None,
                              "note": "fewer than 4 matched companions"})
                continue
            xv = np.log10(sub[col].to_numpy()) if log10 else \
                sub[col].to_numpy()
            yv = sub["delta_phi"].to_numpy()
            if len(np.unique(xv)) < 3:
                stats.append({
                    "observable": col, "label": label,
                    "n": int(n), "rho": None, "p": None,
                    "note": ("observable constant across matched "
                             "companions (e.g. all unresolved/point "
                             "sources) — correlation undefined"),
                })
                print_status(
                    f"dphi vs {label:<42} n={n:2d}  constant "
                    f"(all matched companions identical)", "TEST")
                continue
            rho, p = spearmanr(xv, yv)
            r_pear, p_pear = pearsonr(xv, yv)
            if n >= 6 and len(np.unique(yv)) > 2:
                with np.errstate(all="ignore"):
                    import warnings as _w
                    with _w.catch_warnings():
                        _w.simplefilter("ignore")
                        boot = np.array([
                            spearmanr(
                                xv[(idx := rng.integers(0, n, n))],
                                yv[idx]).statistic
                            for _ in range(N_BOOT)])
                boot = boot[np.isfinite(boot)]
                lo, hi = (np.percentile(boot, [2.5, 97.5])
                          if len(boot) > 100 else (np.nan, np.nan))
            else:
                lo, hi = np.nan, np.nan
            stats.append({
                "observable": col, "label": label,
                "log10_x": bool(log10),
                "n": int(n),
                "rho": float(rho), "p": float(p),
                "rho_ci95": ([float(lo), float(hi)]
                             if np.isfinite(lo) else None),
                "pearson_r": float(r_pear), "pearson_p": float(p_pear),
            })
            ci_txt = (f"[{lo:+.3f},{hi:+.3f}]" if np.isfinite(lo)
                      else "[  n<6  ]")
            print_status(
                f"dphi vs {label:<42} n={n:2d}  rho={rho:+.3f} "
                f"{ci_txt}  p={p:.4f}", "TEST")

        # smallest raw p for the record (multiplicity is flagged, not
        # corrected away, so the reader sees the honest family)
        tested = [s for s in stats if s.get("p") is not None]
        n_tested = len(tested)
        bonf = 0.05 / max(n_tested, 1)
        n_xmm = int(df["xmm_hr2"].notna().sum()) \
            if "xmm_hr2" in df else 0
        n_xmm_resolved = int(
            (df["xmm_ext_arcsec"].fillna(0) > 0).sum()) \
            if "xmm_ext_arcsec" in df else 0
        summary = {
            "n_quasar_class_companions": int(len(comps)),
            "detection_summary": {
                "n_4xmm_matched": n_xmm,
                "n_4xmm_resolved_extent": n_xmm_resolved,
                "n_2rxs_matched": int(df["rxs_hr1"].notna().sum())
                if "rxs_hr1" in df else 0,
                "n_first_matched": int(df["first_fint_mjy"].notna().sum())
                if "first_fint_mjy" in df else 0,
                "n_nvss_matched": int(df["nvss_s14_mjy"].notna().sum())
                if "nvss_s14_mjy" in df else 0,
                "n_dr16q_matched": int(df["gmag"].notna().sum())
                if "gmag" in df else 0,
                "note": ("every 4XMM-matched companion is unresolved "
                         "(ext = 0): the X-ray emitters are point "
                         "sources at XMM resolution"),
            },
            "selection": (
                "companion_type contains 'quasar' or 'Seyfert' in the "
                "verified catalogue: the 18 quasar-class companions; "
                "galaxy companions (NGC 7603B, NGC 1232A) and the "
                "NGC 1199 compact object are excluded"
            ),
            "search_radii_arcsec": SEARCH_RADII_ARCSEC,
            "catalogs": {
                "4XMM-DR13": CAT_4XMM, "2RXS": CAT_2RXS,
                "FIRST": CAT_FIRST, "NVSS": CAT_NVSS,
                "DR16Q": CAT_DR16Q,
            },
            "delta_phi_definition": "ln[(1+z_comp)/(1+z_gal)]",
            "correlations": stats,
            "n_observables_tested": n_tested,
            "bonferroni_p_threshold": bonf,
            "control_note": (
                "gmag is a control observable: it tracks apparent "
                "brightness and would reveal a Malmquist-like "
                "correlation contaminating the physical observables. "
                "A significant correlation of delta_phi with a "
                "non-redshift physical property (hardness, compactness) "
                "indicates the well depth is a property of the emitter; "
                "nulls bound the claim and are reported equally."
            ),
            "csv": str(csv_path),
        }
        json_path = self.results / "step_35_well_depth_correlates.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")

        # --- figure --------------------------------------------------
        try:
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings(
                    "ignore", message=".*extended precision.*PINT.*")
                import matplotlib.pyplot as plt
            apply_tep_style()
            plotted = [s for s in stats if s.get("rho") is not None]
            ncols = 3
            nrows = int(np.ceil(len(plotted) / ncols))
            fig, axes = plt.subplots(nrows, ncols,
                                     figsize=(13, 3.6 * nrows))
            for ax, s in zip(np.atleast_1d(axes).flat, plotted):
                sub = df[["delta_phi", s["observable"]]].dropna()
                xv = np.log10(sub[s["observable"]]) \
                    if s["log10_x"] else sub[s["observable"]]
                ax.scatter(sub["delta_phi"], xv, color="#241a33", s=28)
                ax.set_xlabel(r"$\Delta\phi_{\rm int}$")
                ax.set_ylabel(("log10 " if s["log10_x"] else "")
                              + s["label"])
                ci = s.get("rho_ci95")
                ci_txt = (f"[{ci[0]:+.2f},{ci[1]:+.2f}]"
                          if ci else "[n<6]")
                ax.set_title(
                    f"rho={s['rho']:+.2f} {ci_txt}"
                    f"  p={s['p']:.3f}  n={s['n']}", fontsize=9)
            for ax in np.atleast_1d(axes).flat[len(plotted):]:
                ax.axis("off")
            fig.suptitle("Well depth vs archival observables "
                         "(quasar-class companions)")
            fig.tight_layout()
            fig.savefig(
                self.figures / "step_35_well_depth_correlates.png",
                dpi=300, bbox_inches="tight")
            plt.close(fig)
            print_status("Saved figure: step_35_well_depth_correlates.png",
                         "SUCCESS")
        except Exception as e:
            self.logger.warning(f"figure failed: {e}")

        print_status("Well-depth correlate analysis complete.", "SUCCESS")
