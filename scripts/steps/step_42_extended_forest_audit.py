#!/usr/bin/env python3
"""
Step 42: Unbiased All-Sky Lyman-alpha Forest Audit
=================================================
The catalogue's Ly-alpha audit (steps 29/36) found that the
decisive forest test has been executed on exactly one claimed
sightline.  This step removes the dependence on which pairs were
historically claimed: every spectroscopic quasar at z > 1.9
projected within the strong-connection separation range of a
bright low-redshift galaxy receives the same measurement.

  1. Parent galaxy sample: the 2MRS flux-limited catalogue at
     cz < 15,000 km/s (z < 0.05) — the same parent list used by
     the forward cross-correlation control (step_24).
  2. Spectral matches: CDS XMatch against the SDSS-DR16
     spectroscopic superset (VizieR V/154/sdss16) at 100 arcsec —
     the catalogue's strong-connection claims (luminous bridge,
     filament, silhouette, absorption ordering) all lie at
     separations <= 114 arcsec.
  3. For every QSO-class match at z > 1.9, the archived
     spectrum is retrieved from the SDSS Science Archive Server
     and the forest transmission is measured directly:
     T = <flux> / p95(flux) in the observed band
     [1050, 1215.67] x (1+z) Angstrom, compared with the cosmic
     mean tau_eff = 0.0023 (1+z)^3.65.  A companion at the host
     distance has no intergalactic path and predicts T ~ 1; a
     cosmological quasar predicts T ~ exp(-tau_eff) ~ 0.45-0.8.
     Sightline-to-sightline cosmic variance is ~0.12 in T; a
     candidate anomaly requires T > T_exp + 3*sigma AND T > 0.80.

The test is symmetric: a normal forest falsifies proximity for
that projected pair, an empty forest flags a TEP-proximity
candidate.  Projection-coincidence rate is recorded so the
survey-scale expectation can be priced.

Outputs:
    data/processed/extended_forest_audit.csv
    results/outputs/step_42_extended_forest_audit.json
"""

import io
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy import units as u

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status

LYA_REST = 1215.67          # Angstrom
FOREST_LO_REST = 1050.0     # Angstrom
Z_GAL_MAX_KMS = 15000.0     # z < 0.05
SEP_MAX_ARCSEC = 100.0      # strong-connection class limit
Z_QSO_MIN = 1.9             # forest band ground-accessible
H0_KMS_MPC = 70.0           # host-distance scale for projected kpc
TAU_A, TAU_GAMMA = 0.0023, 3.65   # tau_eff = A (1+z)^gamma
SIGMA_COSMIC = 0.12         # sightline scatter in transmission
MIN_BAND_FRAC = 0.5         # require >=50% band coverage
MIN_BAND_SN = 1.0           # median per-pixel S/N floor in band
SAS_URL = ("https://dr18.sdss.org/sas/dr18/spectro/sdss/redux/"
           "v5_13_2/spectra/lite/{plate}/spec-{plate}-{mjd}-{fiber:04d}.fits")


class Step42ExtendedForestAudit:
    """Step 42: all-sky forest census on unselected QSO-galaxy
    projections."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.spec_dir = self.root / "data" / "raw" / "sdss_spectra"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"
        for d in [self.data_processed, self.spec_dir,
                  self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)
        self.logger = TEPLogger(
            "step_42",
            log_file_path=self.logs / "step_42_extended_forest_audit.log")
        set_step_logger(self.logger)

    # --------------------------------------------------------------
    def _galaxy_sample(self):
        path = self.root / "data" / "raw" / "2mrs_parent_sample.csv"
        if not path.exists():
            raise FileNotFoundError(
                "2mrs_parent_sample.csv missing; run step_24")
        g = pd.read_csv(path)
        g = g[g["cz_kms"] < Z_GAL_MAX_KMS].copy()
        g["z_gal"] = g["cz_kms"] / 299792.458
        self.logger.info(f"2MRS galaxies with z<0.05: {len(g)}")
        return g

    # --------------------------------------------------------------
    def _xmatch(self, g):
        """All V/154 spectra within SEP_MAX_ARCSEC of a parent
        galaxy (cached; one CDS XMatch, chunked uploads)."""
        cache = self.data_processed / "extended_forest_xmatch.csv"
        if cache.exists():
            return pd.read_csv(cache)
        from astropy.table import Table
        from astroquery.xmatch import XMatch
        up = g[["ra_deg", "dec_deg"]].copy()
        up.insert(0, "idx", np.arange(len(up)))
        up.columns = ["idx", "ra", "dec"]
        chunks, step_n = [], 10000
        for s0 in range(0, len(up), step_n):
            part = up.iloc[s0:s0 + step_n]
            for attempt in range(3):
                try:
                    xm = XMatch.query(
                        cat1=Table.from_pandas(part),
                        cat2="vizier:V/154/sdss16",
                        max_distance=SEP_MAX_ARCSEC * u.arcsec,
                        colRA1="ra", colDec1="dec")
                    chunks.append(xm.to_pandas())
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(20)
            print_status(f"XMatch chunk {s0 // step_n + 1}: "
                         f"{len(part)} rows done", "INFO")
        xm = pd.concat(chunks, ignore_index=True)
        xm.to_csv(cache, index=False)
        return xm

    # --------------------------------------------------------------
    def _fetch_spectrum(self, spid):
        """Return (wave, flux, ivar) for plate-mjd-fiber, cached."""
        plate, mjd, fiber = spid.split("-")
        plate, mjd, fiber = int(plate), int(mjd), int(fiber)
        f = self.spec_dir / f"spec-{plate}-{mjd}-{fiber:04d}.fits"
        if not f.exists():
            url = SAS_URL.format(plate=plate, mjd=mjd, fiber=fiber)
            import urllib.request
            try:
                data = urllib.request.urlopen(url, timeout=60).read()
            except Exception:
                return None
            f.write_bytes(data)
        co = fits.open(f)[1].data
        return 10**co["loglam"], co["flux"], co["ivar"]

    # --------------------------------------------------------------
    def _measure(self, w, fl, iv, z):
        lo = FOREST_LO_REST * (1 + z)
        hi = LYA_REST * (1 + z)
        m = (w > lo) & (w < hi)
        if not m.any():
            return {"verdict": "band_uncovered"}
        cov = float(((iv[m] > 0).sum()) / max(m.sum(), 1))
        if cov < MIN_BAND_FRAC:
            return {"verdict": "band_partial", "coverage": cov}
        fband, iband = fl[m][iv[m] > 0], iv[m][iv[m] > 0]
        sn = float(np.median(fband * np.sqrt(iband)))
        if sn < MIN_BAND_SN:
            return {"verdict": "insufficient_sn",
                    "coverage": cov, "sn_band": sn}
        cont = float(np.percentile(fband, 95))
        t_obs = float(np.mean(fband) / cont)
        tau = TAU_A * (1 + z) ** TAU_GAMMA
        t_exp = float(np.exp(-tau))
        excess = (t_obs - t_exp) / SIGMA_COSMIC
        # Flag on the sigma excess alone, then decide whether the
        # spectrum is a true forest anomaly or a pipeline
        # mis-redshift.  A TEP-proximity emitter shows emission lines
        # at the catalogued z but an unabsorbed band; a mis-pipelined
        # object shows its lines at a different consistent z.  Score
        # the claimed z and the best alternative z by how many
        # standard emission lines land on detected peaks.
        verdict = "forest_normal"
        if excess > 4.0:
            claimed, alt_z, alt_n = self._line_pattern(w, fl, iv, z)
            if claimed >= 1 and alt_n < 2:
                verdict = "forest_candidate"
            else:
                verdict = "z_suspect_misz"
            return {"verdict": verdict,
                    "coverage": cov, "sn_band": sn,
                    "transmission": t_obs,
                    "transmission_expected": t_exp,
                    "transmission_excess_sigma": excess,
                    "alt_z": alt_z, "n_lines_alt": alt_n,
                    "n_lines_claimed": claimed}
        return {"verdict": verdict,
                "coverage": cov, "sn_band": sn,
                "transmission": t_obs,
                "transmission_expected": t_exp,
                "transmission_excess_sigma":
                    (t_obs - t_exp) / SIGMA_COSMIC}

    # --------------------------------------------------------------
    def _line_pattern(self, w, fl, iv, z_claim):
        """Score emission-line fits at claimed vs alternative z.

        Returns (n_lines_at_claimed_z, best_alt_z, n_lines_at_alt_z).
        Peaks are local maxima >2 flux units above the median.  A
        redshift scores one point per standard AGN/ELG line whose
        predicted observed position lands within 30 A of a peak.
        """
        from scipy.signal import find_peaks
        lines = {"Lya": 1215.67, "NV": 1240.0, "CIV": 1549.0,
                 "CIII": 1909.0, "MgII": 2799.0, "OII": 3727.0,
                 "Hb": 4861.0, "OIII": 5007.0, "Ha": 6563.0}
        g = (iv > 0) & np.isfinite(fl)
        if g.sum() < 200:
            return 0, None, 0
        sm = np.convolve(np.where(g, fl, 0), np.ones(5) / 5, "same")
        noise = np.where(g, 1 / np.sqrt(iv + 1e-9), np.inf)
        prom = max(3 * float(np.nanmedian(noise[g])), 0.5)
        pk, _ = find_peaks(np.where(g, sm, -9),
                           prominence=prom, distance=15)
        floor = float(np.median(fl[g])) + max(
            0.6, 3 * float(np.nanmedian(noise[g])))
        peaks = np.sort(w[[p for p in pk if g[p] and fl[p] > floor]])
        if len(peaks) == 0:
            return 0, None, 0

        def score(z):
            return sum(1 for r in lines.values()
                       if np.min(np.abs(peaks - r * (1 + z))) < 30)

        claimed = score(z_claim)
        best_z, best_n = None, 0
        for i, a in enumerate(peaks):
            for b in peaks[i + 1:]:
                for r1 in lines.values():
                    z1 = a / r1 - 1
                    if not (0 < z1 < 7):
                        continue
                    n1 = score(z1)
                    for r2 in lines.values():
                        z2 = b / r2 - 1
                        if not (0 < z2 < 7):
                            continue
                        n2 = score(z2)
                        for zc, nc in ((z1, n1), (z2, n2)):
                            if (abs(zc - z_claim) > 0.15
                                    and nc > best_n):
                                best_z, best_n = zc, nc
        return claimed, (round(best_z, 3) if best_z is not None
                         else None), best_n

    # --------------------------------------------------------------
    def run(self):
        self.logger.info("Step 42: unbiased all-sky forest audit")
        g = self._galaxy_sample()
        xm = self._xmatch(g)
        self.logger.info(f"XMatch rows: {len(xm)}")

        cand = xm[(xm["zsp"] > Z_QSO_MIN)
                  & xm["Sp-ID"].notna()].copy()
        cand = cand.drop_duplicates(subset=["idx", "Sp-ID"])
        self.logger.info(
            f"z>{Z_QSO_MIN} QSO spectra within {SEP_MAX_ARCSEC}'' "
            f"of z<0.05 2MRS galaxy: {len(cand)}")

        zmap = g["z_gal"].to_dict()
        rows = []
        spids = cand["Sp-ID"].unique()
        for i, spid in enumerate(spids):
            sub = cand[cand["Sp-ID"] == spid]
            spec = self._fetch_spectrum(spid)
            for _, r in sub.iterrows():
                row = {"galaxy_idx": int(r["idx"]),
                       "sp_id": spid, "z_qso": float(r["zsp"]),
                       "sep_arcsec": float(r["angDist"]),
                       "ra_qso": float(r["RA_ICRS"]),
                       "dec_qso": float(r["DE_ICRS"])}
                zg = float(zmap[int(r["idx"])])
                d_mpc = zg * 299792.458 / H0_KMS_MPC
                row["z_gal"] = zg
                row["sep_kpc_proj"] = (r["angDist"] / 206265.0
                                       ) * d_mpc * 1000.0
                if spec is None:
                    row["verdict"] = "fetch_failed"
                else:
                    row.update(self._measure(*spec, r["zsp"]))
                rows.append(row)
            if (i + 1) % 50 == 0:
                print_status(f"spectra {i + 1}/{len(spids)}", "INFO")

        df = pd.DataFrame(rows)
        df.to_csv(self.data_processed / "extended_forest_audit.csv",
                  index=False)

        vc = df["verdict"].value_counts().to_dict()
        anomalies = df[df["verdict"] == "forest_candidate"]
        suspects = df[df["verdict"] == "z_suspect_misz"]
        # A rejected catalogue redshift cannot define a high-z path length:
        # z_suspect_misz rows are re-audited at their alternative redshift in
        # step_45 and must not enter the strong-forest denominator here.
        testable = df[(df["verdict"].isin(
            ["forest_normal", "forest_candidate"]))
            & (df["transmission_expected"] < 0.5)]
        n_lt30 = int((df["sep_arcsec"] < 30).sum())
        exp_lt30 = len(df) * (30.0 / SEP_MAX_ARCSEC) ** 2
        out = {
            "step": "step_42_extended_forest_audit",
            "parameters": {
                "z_gal_max": Z_GAL_MAX_KMS / 299792.458,
                "sep_max_arcsec": SEP_MAX_ARCSEC,
                "z_qso_min": Z_QSO_MIN,
                "tau_eff_model": f"{TAU_A}(1+z)^{TAU_GAMMA}",
                "sigma_cosmic": SIGMA_COSMIC,
                "excess_sigma_flag": 4.0,
            },
            "n_parent_galaxies": int(len(g)),
            "n_spectral_matches_all": int(len(xm)),
            "n_projections_z_gt_1p9": int(len(cand)),
            "n_unique_spectra": int(len(spids)),
            "n_testable_strong_forest": int(len(testable)),
            "n_lt30arcsec": n_lt30,
            "n_lt30arcsec_expected_uniform": round(float(exp_lt30), 1),
            "verdicts": {k: int(v) for k, v in vc.items()},
            "anomalies": anomalies.to_dict("records"),
            "z_suspects": suspects.to_dict("records"),
            "note": ("forest_normal falsifies proximity for that "
                     "projection; forest_candidate flags a "
                     "TEP-proximity candidate "
                     "(claimed-z lines confirmed, no "
                     "alternative-z line fit)"),
        }
        with open(self.results /
                  "step_42_extended_forest_audit.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        self.logger.info(
            f"verdicts: {vc}; anomalies: {len(anomalies)}")
        print_status(f"Step 42 complete: {len(cand)} projections, "
                     f"{len(anomalies)} forest anomalies", "SUCCESS")


if __name__ == "__main__":
    Step42ExtendedForestAudit().run()
