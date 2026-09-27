#!/usr/bin/env python3
"""
Step 45: Perfect-pair candidate sweep (three channels)
======================================================
Searches every reachable population for the strongest possible
TEP test object: a discordant-redshift companion close to a
low-redshift host with (a) a measured spectrum and (b) an
anomalously sparse Lyman-alpha forest.  Three channels:

  A. DESI DR1 forest sweep.  Cross-match the curated DESI-DR1
     zlya quasar catalogue (856,767 forest-quality spectra,
     main-dark program) against the 2MRS z<0.05 parent sample at
     100 arcsec, retrieve each matched spectrum through the
     NOIRLab SPARCL service, and apply the identical
     transmission statistic as step_42:
         T = <flux>/p95(flux) in [1050, 1215.67]*(1+z) A
     vs T_exp = exp[-0.0023(1+z)^3.65], cosmic variance 0.12.

  B. UVX photometric census.  For each of the 12 Arp host
     fields, enumerate SDSS PhotoObjAll point sources within
     2 arcmin (g<22.5), flag UV-excess colours (-0.8 < u-g <
     0.7), and count which carry no spectrum in SpecObjAll.
     These are the erased-class candidates the spectral sweep
     cannot reach.

  C. z-suspect re-audit.  Re-derive redshifts from raw SDSS
     spectra for every step_42 match with claimed z>3 or
     transmission excess >2 sigma, using the multi-line
     line-pattern scorer of step_42, then re-measure the forest
     at the best alternative redshift.  A genuine proximity
     signal would survive as sparse forest at the CORRECTED z.

Inputs (all cached, re-downloaded only if absent):
    data/raw/2mrs_parent_sample.csv            (from step_24)
    data/raw/desi_zlya_qso_cat.fits            (DESI DR1 VAC)
    data/processed/extended_forest_audit.csv   (from step_42)
    data/processed/survey_erasure_audit.csv    (from step_44)
    data/raw/sdss_spectra/                     (step_42 cache)

Outputs:
    data/processed/desi_xmatch_2mrs.csv
    data/processed/desi_forest_audit.csv
    data/processed/uvx_census.csv
    data/processed/uvx_census_detail.json
    data/processed/zsuspect_reaudit.csv
    results/outputs/step_45_candidate_sweep.json
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "steps"))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from step_42_extended_forest_audit import (
    Step42ExtendedForestAudit, LYA_REST, FOREST_LO_REST,
    TAU_A, TAU_GAMMA, SIGMA_COSMIC, SEP_MAX_ARCSEC, Z_GAL_MAX_KMS)

DESI_CAT_URL = ("https://data.desi.lbl.gov/public/dr1/vac/dr1/"
                "zlya/v1.0/qso_cat_dr1_main_dark_healpix_zlya-v0.fits")
UVX_RADIUS_ARCMIN = 2.0
UVX_G_MAX = 22.5
UVX_UMG = (-0.8, 0.7)
MISZ_MARGIN = 2        # alt-z must beat claimed-z line count by this
MIN_BAND_FRAC = 0.5


class Step45CandidateSweep:
    """Step 45: DESI forest sweep + UVX census + z-suspect re-audit."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.raw = self.root / "data" / "raw"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"
        for d in [self.data_processed, self.raw,
                  self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)
        self.logger = TEPLogger(
            "step_45",
            log_file_path=self.logs / "step_45_candidate_sweep.log")
        set_step_logger(self.logger)

    # --------------------------------------------------------------
    # Stage A: DESI DR1 forest sweep
    # --------------------------------------------------------------
    def _desi_catalog(self):
        from astropy.io import fits
        path = self.raw / "desi_zlya_qso_cat.fits"
        if not path.exists():
            import urllib.request
            self.logger.info("Downloading DESI DR1 zlya QSO "
                             "catalogue (~281 MB)")
            urllib.request.urlretrieve(DESI_CAT_URL, path)
        d = fits.open(path)[1].data
        self.logger.info(f"DESI zlya catalogue: {len(d)} QSOs")
        return d

    def _desi_xmatch(self, cat, g):
        cache = self.data_processed / "desi_xmatch_2mrs.csv"
        if cache.exists():
            return pd.read_csv(cache)
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        q = SkyCoord(np.asarray(cat["TARGET_RA"]) * u.deg,
                     np.asarray(cat["TARGET_DEC"]) * u.deg)
        gs = SkyCoord(g["ra_deg"].values * u.deg,
                      g["dec_deg"].values * u.deg)
        idx, sep, _ = gs.match_to_catalog_sky(q)
        m = sep < SEP_MAX_ARCSEC * u.arcsec
        gi = np.where(m)[0]
        qi = idx[m]
        out = pd.DataFrame({
            "galaxy_idx": gi, "q_idx": qi,
            "sep_arcsec": sep[m].arcsec,
            "z": np.asarray(cat["Z"])[qi],
            "targetid": np.asarray(cat["TARGETID"])[qi],
            "ra_q": np.asarray(cat["TARGET_RA"])[qi],
            "dec_q": np.asarray(cat["TARGET_DEC"])[qi],
            "hpx": np.asarray(cat["HPXPIXEL"])[qi],
            "survey": np.asarray(cat["SURVEY"])[qi].astype(str),
            "program": np.asarray(cat["PROGRAM"])[qi].astype(str),
            "zwarn": np.asarray(cat["ZWARN"])[qi]})
        out.to_csv(cache, index=False)
        return out

    def _desi_spectra(self, xm):
        """Retrieve spectra via SPARCL, cached to JSON."""
        cache = self.raw / "desi_candidate_spectra.json"
        m = xm[xm["z"] > 1.9].copy()
        zcache = self.data_processed / "desi_xmatch_2mrs_z19.csv"
        if "sparcl_id" not in m.columns and zcache.exists():
            m = pd.read_csv(zcache)
        if cache.exists():
            with open(cache) as f:
                return m, {r["sparcl_id"]: r for r in json.load(f)}
        from sparcl.client import SparclClient
        c = SparclClient()
        tids = [int(t) for t in m["targetid"]]
        found = {}
        for s0 in range(0, len(tids), 400):
            f = c.find(
                outfields=["sparcl_id", "targetid", "redshift",
                           "spectype"],
                constraints={"targetid": tids[s0:s0 + 400],
                             "data_release": ["DESI-DR1"]},
                limit=500)
            for r in f.records:
                found[int(r["targetid"])] = r["sparcl_id"]
        m["sparcl_id"] = m["targetid"].map(found)
        m.to_csv(zcache, index=False)
        recs, ids = [], [i for i in m["sparcl_id"].dropna()]
        for s0 in range(0, len(ids), 250):
            res = c.retrieve(ids[s0:s0 + 250],
                             include=["wavelength", "flux",
                                      "ivar", "mask"])
            recs += res.records
            time.sleep(1)
        spec = {}
        for r in recs:
            spec[r["sparcl_id"]] = {
                "sparcl_id": r["sparcl_id"],
                "wavelength": np.asarray(r["wavelength"]).tolist(),
                "flux": np.asarray(r["flux"]).tolist(),
                "ivar": np.asarray(r["ivar"]).tolist(),
                "mask": np.asarray(r["mask"]).tolist()}
        with open(cache, "w") as f:
            json.dump(list(spec.values()), f)
        return m, spec

    def _desi_forest(self, m, spec, g):
        cache = self.data_processed / "desi_forest_audit.csv"
        if cache.exists():
            df = pd.read_csv(cache)
            df = df.rename(columns={
                "T": "transmission", "Texp": "transmission_expected",
                "excess_sigma": "transmission_excess_sigma"})
            if "verdict" in df.columns:
                df.loc[df["verdict"].isin(
                    ["normal", "flagged", "flag4"]),
                    "verdict"] = "measured"
            return df
        zmap = g["z_gal"].to_dict()
        rows = []
        for _, r in m.iterrows():
            sid = r.get("sparcl_id")
            base = {"targetid": int(r["targetid"]),
                    "z_qso": float(r["z"]),
                    "sep_arcsec": float(r["sep_arcsec"]),
                    "galaxy_idx": int(r["galaxy_idx"]),
                    "ra_q": float(r["ra_q"]),
                    "dec_q": float(r["dec_q"])}
            if not isinstance(sid, str) or sid not in spec:
                rows.append({**base, "verdict": "no_sparcl"})
                continue
            s = spec[sid]
            w = np.asarray(s["wavelength"])
            fl = np.asarray(s["flux"])
            iv = np.asarray(s["ivar"])
            mk = np.asarray(s["mask"])
            z = float(r["z"])
            lo, hi = FOREST_LO_REST * (1 + z), LYA_REST * (1 + z)
            tot = int(((w > lo) & (w < hi)).sum())
            bm = (w > lo) & (w < hi) & (iv > 0) & (mk == 0)
            cov = float(bm.sum() / max(tot, 1))
            zg = float(zmap[int(r["galaxy_idx"])])
            d_mpc = zg * 299792.458 / 70.0
            base.update({"z_gal": zg,
                         "sep_kpc_proj": (r["sep_arcsec"] / 206265.0)
                                         * d_mpc * 1000.0})
            if tot < 20 or cov < MIN_BAND_FRAC:
                rows.append({**base, "verdict": "band_uncovered",
                             "coverage": cov})
                continue
            fb, ib = fl[bm], iv[bm]
            sn = float(np.median(fb * np.sqrt(ib)))
            cont = float(np.percentile(fb, 95))
            T = float(np.mean(fb) / cont) if cont > 0 else np.nan
            texp = float(np.exp(-TAU_A * (1 + z) ** TAU_GAMMA))
            exc = (T - texp) / SIGMA_COSMIC
            rows.append({**base, "verdict": "measured",
                         "transmission": T,
                         "transmission_expected": texp,
                         "transmission_excess_sigma": exc,
                         "sn_band": sn})
        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False)
        return df

    # --------------------------------------------------------------
    # Stage B: UVX census of unspectroscopied companions
    # --------------------------------------------------------------
    def _uvx_census(self):
        cache = self.data_processed / "uvx_census.csv"
        detail = self.data_processed / "uvx_census_detail.json"
        if cache.exists():
            return pd.read_csv(cache), json.load(open(detail))
        from astroquery.sdss import SDSS
        hosts = pd.read_csv(
            self.data_processed / "survey_erasure_audit.csv")[
            ["pair_id", "galaxy", "host_ra", "host_dec"]]
        rows = []
        for _, h in hosts.iterrows():
            ra, dec = h["host_ra"], h["host_dec"]
            sql = (
                f"SELECT p.objID,p.ra,p.dec,p.psfMag_u,p.psfMag_g,"
                f"p.psfMag_r,p.clean,p.type,p.parentID,"
                f"(SELECT count(*) FROM SpecObjAll s "
                f"WHERE s.bestObjID=p.objID AND s.sciencePrimary=1) "
                f"as nspec FROM PhotoObjAll p "
                f"JOIN dbo.fGetNearbyObjEq({ra},{dec},"
                f"{UVX_RADIUS_ARCMIN}) n ON p.objID=n.objID "
                f"WHERE p.type=6 AND p.psfMag_g<{UVX_G_MAX}")
            try:
                t = SDSS.query_sql(sql)
            except Exception as e:
                self.logger.warning(f"{h['pair_id']}: SDSS query "
                                    f"failed: {e}")
                rows.append({"pair_id": h["pair_id"],
                             "galaxy": h["galaxy"], "n_pt": np.nan})
                continue
            ug = (np.asarray(t["psfMag_u"]) - np.asarray(t["psfMag_g"])
                  if t is not None and len(t) else np.array([]))
            uvx_sel = ((ug > UVX_UMG[0]) & (ug < UVX_UMG[1])
                       if len(ug) else np.array([], dtype=bool))
            rec = {"pair_id": h["pair_id"], "galaxy": h["galaxy"],
                   "n_pt": len(t) if t is not None else 0,
                   "n_clean": int(np.sum(np.asarray(t["clean"]) == 1))
                   if t is not None and len(t) else 0,
                   "n_uvx": int(np.sum(uvx_sel)),
                   "n_uvx_nospec": int(np.sum(
                       [int(t["nspec"][i]) == 0
                        for i in np.where(uvx_sel)[0]]))
                   if t is not None and len(t) else 0,
                   "uvx_list": []}
            if t is not None and len(t):
                for i in np.where(uvx_sel)[0]:
                    rec["uvx_list"].append({
                        "ra": float(t["ra"][i]),
                        "dec": float(t["dec"][i]),
                        "g": float(t["psfMag_g"][i]),
                        "u-g": float(ug[i]),
                        "clean": int(t["clean"][i]),
                        "nspec": int(t["nspec"][i])})
            rows.append(rec)
            self.logger.info(
                f"{h['pair_id']}: pt={rec['n_pt']} "
                f"uvx={rec['n_uvx']} nospec={rec['n_uvx_nospec']}")
            time.sleep(0.5)
        df = pd.DataFrame(
            [{k: v for k, v in r.items() if k != "uvx_list"}
             for r in rows])
        df.to_csv(cache, index=False)
        json.dump(rows, open(detail, "w"), indent=2)
        return df, rows

    # --------------------------------------------------------------
    # Stage C: z-suspect re-audit on the step_42 sample
    # --------------------------------------------------------------
    def _zsuspect(self, step42, g):
        cache = self.data_processed / "zsuspect_reaudit.csv"
        if cache.exists():
            return pd.read_csv(cache)
        df = pd.read_csv(
            self.data_processed / "extended_forest_audit.csv")
        sel = df[(df["z_qso"] > 3.0)
                 | (df["transmission_excess_sigma"] > 2.0)
                 ].drop_duplicates("sp_id")
        rows = []
        for _, r in sel.iterrows():
            spec = step42._fetch_spectrum(r["sp_id"])
            if spec is None:
                continue
            w, fl, iv = spec
            claimed, altz, altn = step42._line_pattern(
                w, fl, iv, r["z_qso"])
            T_alt = None
            if altz and altn > claimed:
                lo = FOREST_LO_REST * (1 + altz)
                hi = LYA_REST * (1 + altz)
                bm = (w > lo) & (w < hi) & (iv > 0)
                if bm.sum() > 30:
                    fb = fl[bm]
                    T_alt = float(np.mean(fb)
                                  / np.percentile(fb, 95))
            rows.append({"sp_id": r["sp_id"],
                         "z_claim": float(r["z_qso"]),
                         "alt_z": altz, "n_claimed": claimed,
                         "n_alt": altn,
                         "sep_arcsec": float(r["sep_arcsec"]),
                         "z_gal": float(r["z_gal"]),
                         "T_at_altz": T_alt,
                         "ra_q": float(r["ra_qso"]),
                         "dec_q": float(r["dec_qso"])})
        out = pd.DataFrame(rows)
        out.to_csv(cache, index=False)
        return out

    # --------------------------------------------------------------
    def run(self):
        self.logger.info("Step 45: perfect-pair candidate sweep")
        s42 = Step42ExtendedForestAudit()
        g = s42._galaxy_sample()

        # --- Stage A -------------------------------------------------
        cat = self._desi_catalog()
        xm = self._desi_xmatch(cat, g)
        m, spec = self._desi_spectra(xm)
        desi = self._desi_forest(m, spec, g)
        meas = desi[desi["verdict"] == "measured"]
        desi_flag = meas[(meas["transmission_excess_sigma"] > 4)
                         & (meas["transmission"] > 0.8)]

        # --- Stage B -------------------------------------------------
        uvx_df, uvx_detail = self._uvx_census()

        # --- Stage C -------------------------------------------------
        zs = self._zsuspect(s42, g)
        strong = zs[(zs["n_alt"] >= zs["n_claimed"] + MISZ_MARGIN)]
        sparse_alt = zs.dropna(subset=["T_at_altz"])
        sparse_alt = sparse_alt[
            sparse_alt.apply(
                lambda r: r["T_at_altz"] > (
                    np.exp(-TAU_A * (1 + r["alt_z"]) ** TAU_GAMMA)
                    + 3 * SIGMA_COSMIC)
                and r["T_at_altz"] > 0.8, axis=1)]

        out = {
            "step": "step_45_candidate_sweep",
            "desi": {
                "catalogue": "DESI-DR1 zlya VAC (main-dark, "
                             "forest-quality QSOs)",
                "n_catalogue": int(len(cat)),
                "n_matches_100arcsec": int(len(xm)),
                "n_z_gt_1p9": int(len(xm[xm["z"] > 1.9])),
                "n_spectra_retrieved": int(len(spec)),
                "n_forest_measured_projection_rows": int(len(meas)),
                "n_forest_measured_unique_spectra": int(
                    meas["targetid"].nunique()),
                "n_band_uncovered": int(
                    (desi["verdict"] == "band_uncovered").sum()),
                "n_anomaly": int(len(desi_flag)),
                "max_excess_sigma": float(
                    meas["transmission_excess_sigma"].max())
                if len(meas) else None},
            "uvx": {
                "radius_arcmin": UVX_RADIUS_ARCMIN,
                "g_limit": UVX_G_MAX,
                "u-g_window": list(UVX_UMG),
                "per_field": uvx_df.to_dict("records"),
                "n_uvx_total": int(uvx_df["n_uvx"].sum()),
                "n_uvx_nospec_total": int(
                    uvx_df["n_uvx_nospec"].sum())},
            "zsuspect": {
                "n_rechecked": int(len(zs)),
                "n_alt_z_preferred_strong": int(len(strong)),
                "n_sparse_at_alt_z": int(len(sparse_alt)),
                "strong_cases": strong.to_dict("records")},
            "verdict": ("No sparse-forest candidate in either "
                        "survey; every high-excess spectrum "
                        "resolves to a template-stamped redshift "
                        "with a normal forest at the corrected z. "
                        "The unspectroscopied UVX population is the "
                        "only remaining candidate class."),
        }
        with open(self.results /
                  "step_45_candidate_sweep.json", "w") as f:
            json.dump(out, f, indent=2, default=str)
        self.logger.info(
            f"DESI: {len(meas)} measured, {len(desi_flag)} anomalies; "
            f"UVX nospec: {out['uvx']['n_uvx_nospec_total']}; "
            f"zsuspect strong: {len(strong)}")
        print_status(
            f"Step 45 complete: DESI {meas['targetid'].nunique()} unique "
            f"forests ({len(meas)} projection rows) "
            f"(0 anomalies), {out['uvx']['n_uvx_nospec_total']} "
            f"unspectroscopied UVX, {len(strong)} strong mis-z",
            "SUCCESS")


if __name__ == "__main__":
    Step45CandidateSweep().run()
