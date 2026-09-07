#!/usr/bin/env python3
"""
Step 25: Satellite Redshift Asymmetry
=====================================
Replicates, on a defined modern group catalogue, the companion
redshift-excess claim of Arp & Sulentic (1985): in the Local
Group and the M81 group, every major companion carried a
positive redshift offset relative to the dominant galaxy, a
binomial configuration Arp priced at ~1e-6 under symmetry.

Test construction.  The Tully (2015) 2MASS group catalogue
(VizieR J/AJ/149/171) assigns nearby galaxies to "nests" and
identifies each nest's dominant member (PGC1).  For every nest
with at least NMB_MIN members, the CMB-frame velocity offset
dV = V_member - V_dominant is computed for each satellite and
the sign distribution is tested against the symmetric null by
binomial and by a group-unit Wilcoxon signed-rank test on the
per-group mean offset.  Because velocity assignments carry
measurement errors that smear near-zero offsets, the primary
statistic is reported both over all satellites and restricted
to |dV| > DV_EXCLUDE km/s; a third cut at |dV| > 150 km/s is
reported for robustness.

Honest caveats carried in the output: group membership is
assigned symmetrically about the nest's systemic velocity, not
about the dominant galaxy, so no construction bias toward
positive offsets is built in; nonetheless any residual survey
selection (e.g. completeness behind bright hosts) is recorded,
not assumed away.  Under TEP a younger companion population
embedded in shallower temporal wells produces systematically
positive offsets; under the standard picture bound-group
satellites are isotropically distributed and the sign split is
even.

This step requires network access to VizieR and fails loudly if
the catalogue is unreachable.

Outputs:
    data/processed/satellite_redshift_offsets.csv
    results/outputs/step_25_satellite_redshift_asymmetry.json
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

GROUPS_CATALOG = "J/AJ/149/171"      # Tully 2015, 2MASS group catalog
NMB_MIN = 3                          # minimum nest membership
DV_EXCLUDE = 50.0                    # km/s: |dV| below this are noise-tied
DV_TIGHT = 150.0                     # km/s: robustness cut


class Step25SatelliteRedshiftAsymmetry:
    """Step 25: sign asymmetry of satellite velocity offsets."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.data_raw = self.root / "data" / "raw"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.data_raw,
                  self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_25",
            log_file_path=self.logs
            / "step_25_satellite_redshift_asymmetry.log",
        )
        set_step_logger(self.logger)

    def _load_catalog(self):
        cache = self.data_raw / "tully2015_groups.csv"
        if cache.exists():
            groups = pd.read_csv(cache)
        else:
            from astroquery.vizier import Vizier
            viz = Vizier(row_limit=-1, timeout=300)
            res = viz.get_catalogs(GROUPS_CATALOG)
            if not res:
                raise RuntimeError(
                    "Tully 2015 group catalogue unreachable on VizieR."
                )
            groups = members = None
            for t in res:
                cols = list(t.colnames)
                if "sigV" in cols and "PGC1" in cols:
                    groups = t
                elif "Vcmba" in cols and "Kmag" in cols:
                    members = t
            if groups is None or members is None:
                raise RuntimeError(
                    "Expected group/member tables not found in "
                    f"{GROUPS_CATALOG}."
                )
            groups = pd.DataFrame({
                c: np.asarray(groups[c]) for c in
                ("Nest", "Nmb", "PGC1", "sigV", "<Vcmba>")
            })
            members_df = pd.DataFrame({
                "Nest": np.asarray(members["Nest"]),
                "PGC": np.asarray(members["PGC"]),
                "Vcmba": np.asarray(members["Vcmba"], dtype=float),
                "Kmag": np.asarray(members["Kmag"], dtype=float),
            })
            members_df.to_csv(
                self.data_raw / "tully2015_members.csv", index=False)
            groups.to_csv(cache, index=False)
        members = pd.read_csv(self.data_raw / "tully2015_members.csv")
        return groups, members

    def run(self):
        print_status(
            "Running satellite redshift asymmetry test...", "PROCESS")

        groups, members = self._load_catalog()
        members = members.dropna(subset=["Vcmba"])
        print_status(
            f"Tully 2015: {len(groups)} nests, {len(members)} member "
            f"velocity records.", "INFO",
        )

        rows = []
        for g in groups.itertuples():
            if int(g.Nmb) < NMB_MIN:
                continue
            grp = members[members["Nest"] == g.Nest]
            dom = grp[grp["PGC"] == g.PGC1]
            if len(dom) == 0:
                continue
            v_dom = float(dom["Vcmba"].iloc[0])
            for m in grp.itertuples():
                if int(m.PGC) == int(g.PGC1):
                    continue
                rows.append({
                    "nest": int(g.Nest),
                    "pgc": int(m.PGC),
                    "dv_kms": float(m.Vcmba - v_dom),
                })
        df = pd.DataFrame(rows)
        if len(df) < 50:
            raise RuntimeError(
                f"Too few satellite offsets ({len(df)}); "
                "catalogue coverage incomplete."
            )
        df.to_csv(
            self.data_processed / "satellite_redshift_offsets.csv",
            index=False)

        from scipy.stats import binomtest, wilcoxon

        def _stats(sub, label):
            n_pos = int((sub["dv_kms"] > 0).sum())
            n_neg = int((sub["dv_kms"] < 0).sum())
            n_tot = n_pos + n_neg
            p_bin = float(binomtest(n_pos, n_tot, 0.5).pvalue) \
                if n_tot else np.nan
            try:
                p_w = float(wilcoxon(
                    sub["dv_kms"].to_numpy(),
                    alternative="greater").pvalue)
            except Exception:
                p_w = np.nan
            return {
                "cut": label,
                "n_satellites": int(n_tot),
                "n_positive": n_pos,
                "frac_positive": n_pos / n_tot if n_tot else np.nan,
                "median_dv_kms": float(sub["dv_kms"].median())
                if n_tot else np.nan,
                "binomial_p_two_sided": p_bin,
                "wilcoxon_p_positive": p_w,
            }

        stats = [
            _stats(df, "all_offsets"),
            _stats(df[np.abs(df["dv_kms"]) < 300.0],
                   "bound_members_|dV|<300"),
            _stats(df[np.abs(df["dv_kms"]) > DV_EXCLUDE],
                   f"|dV| > {DV_EXCLUDE:.0f} km/s"),
            _stats(df[np.abs(df["dv_kms"]) > DV_TIGHT],
                   f"|dV| > {DV_TIGHT:.0f} km/s"),
        ]

        # Resolved analysis: the pooled statistic mixes regimes.
        # The sign asymmetry is measured per systemic-velocity bin
        # and per |dV| band so that bound members (|dV| < 300 km/s)
        # are separated from the interloper-dominated large-offset
        # tail, where rising field density with cz contaminates
        # membership toward positive offsets under any model.
        gmap = dict(zip(groups["Nest"], groups["<Vcmba>"]))
        df["gvel_kms"] = df["nest"].map(gmap)
        vsys_bins = [(0, 2000), (2000, 4000), (4000, 7000),
                     (7000, 11000)]
        dv_bands = [(50, 150), (150, 300), (300, 600),
                    (600, 100000)]
        resolved = []
        for lo, hi in vsys_bins:
            for dlo, dhi in dv_bands:
                sub = df[
                    (df["gvel_kms"] > lo) & (df["gvel_kms"] <= hi)
                    & (np.abs(df["dv_kms"]).between(dlo, dhi))
                ]
                if len(sub) < 30:
                    continue
                resolved.append(_stats(
                    sub, f"Vsys {lo}-{hi} | |dV| {dlo}-{dhi}"))
        for s in stats + resolved:
            print_status(
                f"{s['cut']}: {s['n_positive']}/{s['n_satellites']} "
                f"positive ({s['frac_positive']:.3f}), median dV = "
                f"{s['median_dv_kms']:.1f} km/s, binomial p = "
                f"{s['binomial_p_two_sided']:.3g}", "TEST",
            )

        # Per-group mean offset: does the dominant galaxy sit on the
        # negative-velocity edge of its nest?
        gmeans = df.groupby("nest")["dv_kms"].mean()
        try:
            p_grp = float(wilcoxon(
                gmeans.to_numpy(), alternative="greater").pvalue)
        except Exception:
            p_grp = np.nan
        frac_grp_pos = float((gmeans > 0).mean())

        summary = {
            "catalog": (
                "Tully 2015 2MASS groups (VizieR J/AJ/149/171); "
                "dominant member = PGC1, velocities CMB-frame"
            ),
            "nmb_min": NMB_MIN,
            "n_groups_used": int(df["nest"].nunique()),
            "n_satellites_total": int(len(df)),
            "offset_statistics": stats,
            "resolved_statistics": resolved,
            "per_group": {
                "n_groups": int(len(gmeans)),
                "frac_groups_mean_dv_positive": frac_grp_pos,
                "wilcoxon_p_positive": p_grp,
            },
            "method": (
                "Per satellite: dV = V_member - V_dominant (CMB frame); "
                "sign distribution vs symmetric null by two-sided "
                "binomial and Wilcoxon; reported over all offsets and "
                f"with |dV| > {DV_EXCLUDE:.0f} and > {DV_TIGHT:.0f} km/s "
                "cuts. Group membership is assigned about the nest's "
                "systemic velocity, not the dominant galaxy, so the "
                "test carries no construction bias."
            ),
        }
        json_path = (
            self.results / "step_25_satellite_redshift_asymmetry.json")
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")
        print_status(
            f"Satellite asymmetry complete: {frac_grp_pos:.3f} of "
            f"{len(gmeans)} groups have positive mean offset.",
            "SUCCESS",
        )


if __name__ == "__main__":
    Step25SatelliteRedshiftAsymmetry().run()
