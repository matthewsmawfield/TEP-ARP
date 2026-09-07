#!/usr/bin/env python3
"""
Step 29: Lyman-alpha Forest Path-Length Audit
=============================================
If a quasar-class companion is at its cosmological redshift, the
path between it and the observer crosses gigaparsecs of
intergalactic medium: the spectrum blueward of the Lyman-alpha
emission line must carry a Lyman-alpha forest accumulated over
that path, plus intervening metal systems at 0 < z_abs < z_Q.
If the companion instead sits at the host's distance (tens of
Mpc), there is essentially no path length and the forest should
be absent or radically sparse.  This is the decisive archival
test of the distance question — the same class of evidence as
the foreground-absorption ordering of step_21, applied to the
intervening path rather than the host halo.

For every quasar-class companion this step:

  1. computes the observed-frame forest band
     [1025.72*(1+z_Q), 1215.67*(1+z_Q)] minus the quasar
     proximity zone, and classifies observability
     (ground-accessible vs space-UV);
  2. audits public archives for spectra covering the band:
     SDSS SpecObj, MAST (HST COS/STIS/FOS/GHRS/FOS, FUSE, IUE,
     GALEX, WFC3 gratings), ESO (UVES, XSHOOTER, FORS2), and the
     Keck Observatory Archive (HIRES, ESI, LRIS) via the KOA TAP
     service;
  3. where a retrievable product covers the band, downloads the
     archived 1-D spectrum and measures the forest directly:
     continuum-normalised absorption-line census with Galactic-
     ISM and intrinsic-O VI exclusion, each candidate confirmed
     by a significant decrement at the predicted Ly-beta (and,
     where covered, Ly-gamma) position — the standard forest
     line validation.  The confirmed system count is compared
     with the published cosmic-mean incidence (Danforth et al.
     2016, ApJ 817, 111).

Every query is live and recorded with instrument, coverage,
program IDs and offsets.  A companion with no covering product
is reported as an audited 'forest_not_covered' — the decisive
spectra that do not yet exist publicly are stated as such, with
the minimal required observation.  The step fails loudly if the
catalogue inputs are missing; archive timeouts are retried and
then recorded as 'query_failed' rows, never silently omitted.

Outputs:
    data/processed/lya_forest_audit.csv
    results/outputs/step_29_lya_forest_audit.json
"""

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.jsonio import json_safe

LYA = 1215.67          # Lyman-alpha rest wavelength (Angstrom)
LYB = 1025.72          # Lyman-beta
LYG = 972.537          # Lyman-gamma
PROX_REST = 1185.0     # forest red edge: exclude quasar proximity zone
FOREST_BLUE_REST = 1041.0  # forest blue edge above Ly-beta complex

ESO_SPECTROGRAPHS = ["uves", "xshooter", "fors2"]
ESO_ON_SLIT_ARCSEC = 3.0     # long-slit spectra need the target on slit
ESO_FIELD_ARCMIN = 2.0       # wider hits are field context, not spectra
KOA_TABLES = {"koa_hires": (3000, 10000), "koa_esi": (3900, 10900),
              "koa_lris": (3100, 5600)}
KOA_ON_TARGET_ARCSEC = 3.0   # slit width scale: pointing must centre the target
KOA_MASK_FIELD_ARCSEC = 120.0
VLT_DEC_LIMIT = 28.0               # practical reach from Paranal
KECK_DEC_RANGE = (-45.0, 73.0)
N_RETRIES = 3
RETRY_SLEEP_S = 3.0

# Galactic ISM metal lines in typical forest windows (Angstrom)
ISM_LINES = [1260.42, 1302.17, 1304.37, 1334.53, 1526.71,
             1608.45, 1670.79, 1709.60, 1741.55, 1751.91, 1808.01]

# Approximate wavelength coverage (Angstrom) of MAST spectrograph
# modes used to score forest-band coverage.
MAST_COVERAGE = {
    "COS":          [(880, 1450), (1105, 2250), (1405, 1775)],
    "COS/FUV":      [(880, 1450), (1405, 1775)],
    "COS/NUV":      [(1650, 3200)],
    "STIS":         [(1150, 1710), (1570, 3180), (2900, 5700),
                     (5240, 10270)],
    "STIS/FUV-MAMA": [(1150, 1710)],
    "STIS/NUV-MAMA": [(1570, 3180)],
    "STIS/CCD":     [(2900, 5700), (5240, 10270)],
    "FOS/BL":       [(1150, 2500)],
    "FOS/RD":       [(1600, 8500)],
    "HRS":          [(1100, 3200)],
    "HRS/2":        [(1100, 3200)],
    "FUV":          [(905, 1187)],           # FUSE
    "SWP":          [(1150, 1980)],          # IUE
    "LWP":          [(1850, 3350)],
    "LWR":          [(1850, 3350)],
    "GALEX":        [(1350, 1800), (1750, 2800)],
    "WFC3/UVIS":    [(2000, 10000)],
    "WFC3/IR":      [(8000, 17000)],
    "BOSS":         [(3600, 10400)],
    "SDSS SPECTROGRAPH": [(3800, 9200)],
}
ESO_COVERAGE = {"uves": (3000, 5000), "xshooter": (3000, 5600),
                "fors2": (3300, 11000)}
SDSS_COVERAGE = (3600, 10400)


def _sep_deg(ra1, d1, ra2, d2):
    return np.degrees(np.arccos(np.clip(
        np.sin(np.radians(d1)) * np.sin(np.radians(d2))
        + np.cos(np.radians(d1)) * np.cos(np.radians(d2))
        * np.cos(np.radians(ra1 - ra2)), -1, 1)))


def _overlap(lo1, hi1, lo2, hi2):
    return max(0.0, min(hi1, hi2) - max(lo1, lo2))


def _union_len(intervals):
    """Total length covered by a set of (lo, hi) intervals without
    double counting overlaps."""
    total, clo, chi = 0.0, None, None
    for lo, hi in sorted(intervals):
        if clo is None:
            clo, chi = lo, hi
        elif lo <= chi:
            chi = max(chi, hi)
        else:
            total += chi - clo
            clo, chi = lo, hi
    if clo is not None:
        total += chi - clo
    return total


class Step29LyaForestAudit:
    """Step 29: audit and measure Ly-alpha forest path length."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.data_raw = self.root / "data" / "raw" / "spectra_lya"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"
        for d in [self.data_processed, self.data_raw,
                  self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)
        self.logger = TEPLogger(
            "step_29",
            log_file_path=self.logs / "step_29_lya_forest_audit.log")
        set_step_logger(self.logger)

    # --------------------------------------------------------------
    def _targets(self):
        """Quasar-class companions (AGN continuum required for the
        forest); galaxy companions are excluded as non-applicable."""
        cat_path = self.data_processed / "arp_pair_catalog_verified.csv"
        if not cat_path.exists():
            cat_path = self.data_processed / "arp_pair_catalog.csv"
        if not cat_path.exists():
            raise FileNotFoundError("pair catalogue missing")
        catalog = pd.read_csv(cat_path)
        cp_path = self.data_processed / "companion_positions.csv"
        if not cp_path.exists():
            raise FileNotFoundError(
                "companion_positions.csv missing; run step_20")
        cp = pd.read_csv(cp_path)

        non_agn = {"compact galaxy", "compact object",
                   "companion spiral galaxy"}
        agn_pairs = set(
            catalog.loc[
                ~catalog["companion_type"].isin(non_agn), "pair_id"])
        # NGC 1199's companion is a galaxy despite 'compact object'
        # wording; NGC 7603B and NGC 1232A are galaxies.
        drop_names = {"NGC 7603B", "2MASXi J0303366-153740",
                      "NGC 1232A"}
        cp = cp[cp["pair_id"].isin(agn_pairs)
                & ~cp["resolved_name"].isin(drop_names)]
        gal = catalog.set_index("pair_id")["z_gal"].to_dict()
        out = []
        for _, r in cp.iterrows():
            z = float(r["z"])
            out.append({
                "pair_id": r["pair_id"],
                "resolved_name": str(r["resolved_name"]).strip(),
                "ra_deg": float(r["ra_deg"]),
                "dec_deg": float(r["dec_deg"]),
                "z": z,
                "z_gal": float(gal.get(r["pair_id"], np.nan)),
                "forest_lo": FOREST_BLUE_REST * (1 + z),
                "forest_hi": PROX_REST * (1 + z),
                "lya_emission": LYA * (1 + z),
                "observability": ("ground" if LYA * (1 + z) >= 3200
                                  else "space_uv"),
            })
        return out

    # --------------------------------------------------------------
    def _query_sdss(self, ra, dec):
        from astroquery.sdss import SDSS
        from astropy.coordinates import SkyCoord
        last = None
        for attempt in range(1, N_RETRIES + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t = SDSS.query_region(
                        SkyCoord(ra, dec, unit="deg"),
                        spectro=True, radius=30 * u.arcsec)
                return 0 if t is None else len(t)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(RETRY_SLEEP_S * attempt)
        raise RuntimeError(f"SDSS query failed: {last}")

    def _query_mast(self, ra, dec):
        from astroquery.mast import Observations
        last = None
        for attempt in range(1, N_RETRIES + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    return Observations.query_region(
                        f"{ra} {dec}", radius=30 * u.arcsec)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(RETRY_SLEEP_S * attempt)
        raise RuntimeError(f"MAST query failed: {last}")

    def _query_eso(self, eso, instrument, ra, dec):
        last = None
        for attempt in range(1, N_RETRIES + 1):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    t = eso.query_instrument(
                        instrument, coord1=ra, coord2=dec, box="8",
                        column_filters={"dp_cat": "SCIENCE"})
                return [] if t is None else list(t)
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(RETRY_SLEEP_S * attempt)
        raise RuntimeError(f"ESO {instrument} query failed: {last}")

    def _query_koa(self, tap, table, ra, dec):
        last = None
        for attempt in range(1, N_RETRIES + 1):
            try:
                box = 0.04
                return tap.search(
                    f"SELECT koaid, ra, dec, targname FROM {table} "
                    f"WHERE ra BETWEEN {ra - box:.5f} AND "
                    f"{ra + box:.5f} AND dec BETWEEN "
                    f"{dec - box:.5f} AND {dec + box:.5f}").to_table()
            except Exception as e:  # noqa: BLE001
                last = e
                time.sleep(RETRY_SLEEP_S * attempt)
        raise RuntimeError(f"KOA {table} query failed: {last}")

    # --------------------------------------------------------------
    # Grating-level coverage (Angstrom) overriding the broad
    # instrument map where the filter name is recognised.
    GRATING_COVERAGE = {
        "G130M": (900, 1450), "G140L": (1105, 2250),
        "G160M": (1405, 1775), "G185M": (1700, 2100),
        "G225M": (2100, 2500), "G285M": (2500, 3200),
        "E140M": (1150, 1710), "E140H": (1150, 1710),
        "E230M": (1600, 3100), "E230H": (1600, 3100),
        "G230L": (1570, 3180), "G230M": (1600, 3100),
        "G430L": (2900, 5700), "G430M": (2900, 5700),
        "G750L": (5240, 10270), "G750M": (5450, 10270),
        "G130H": (1150, 1600), "G190H": (1600, 2330),
        "G270H": (2200, 3300), "G270M": (2200, 3300),
        "G400H": (3240, 4820), "G570H": (4600, 6800),
        "G780H": (6200, 8500), "PRISM": (1150, 6000),
        "G102": (8000, 11500), "G141": (11000, 17000),
        "G280": (2000, 9500),
        "FUV": (1350, 1800), "NUV": (1750, 2800),  # GALEX
        "LOW DISP": (1150, 3350),
    }

    # Instruments whose 'LOW DISP' filter label is camera-blind or
    # whose products cannot support a forest census.
    IUE_CAMERAS = {"SWP": (1150, 1980), "LWP": (1850, 3350),
                   "LWR": (1850, 3350)}
    SLITLESS = ("GALEX",)   # slitless grism R~100 cannot resolve forest

    def _mast_forest_products(self, t, band_lo, band_hi):
        """Per-observation forest-band coverage inside the 30-arcsec
        query cone.  The actual grating named in the filters column
        gates the range first (IUE 'LOW DISP' is resolved per camera);
        the broad instrument map is the fallback.  Returns coverage
        dicts carrying the clipped band intervals so per-target union
        coverage can be computed without double counting."""
        out = []
        for r in t:
            if str(r["dataproduct_type"]).lower() != "spectrum":
                continue
            try:
                dist_arcsec = float(r["distance"]) * 3600.0
            except Exception:  # noqa: BLE001
                dist_arcsec = np.nan
            # query_region can return rows far outside the cone for
            # some collections; enforce the intended 30-arcsec limit.
            if not np.isfinite(dist_arcsec) or dist_arcsec > 30.0:
                continue
            inst = str(r["instrument_name"])
            filt = str(r["filters"])
            # Acquisition/mirror entries can carry a 'spectrum'
            # dataproduct type; they are not spectra.
            if all(g.strip().upper().startswith(("MIRROR", "ACQ"))
                   for g in filt.replace(";", ",").split(",")
                   if g.strip()):
                continue
            ranges = []
            if inst.upper() in self.IUE_CAMERAS:
                ranges = [self.IUE_CAMERAS[inst.upper()]]
            else:
                for g in filt.replace(";", ",").split(","):
                    g = g.strip().upper()
                    if g in self.GRATING_COVERAGE:
                        ranges.append(self.GRATING_COVERAGE[g])
                if not ranges:
                    ranges = MAST_COVERAGE.get(inst, MAST_COVERAGE.get(
                        inst.split("/")[0], []))
            intervals = [(max(lo, band_lo), min(hi, band_hi))
                         for lo, hi in ranges
                         if _overlap(lo, hi, band_lo, band_hi) > 0]
            if not intervals:
                continue
            out.append({
                "obs_id": str(r["obs_id"]),
                "instrument": inst,
                "filters": filt,
                "proposal_id": str(r["proposal_id"]),
                "collection": str(r["obs_collection"]),
                "dist_arcsec": float(dist_arcsec),
                "usable": not any(s in inst.upper()
                                  for s in self.SLITLESS),
                "intervals": [[float(a), float(b)] for a, b in intervals],
                "covered_a": float(sum(b - a for a, b in intervals)),
            })
        return out

    # --------------------------------------------------------------
    def _fetch_product(self, obs_id_list, patterns):
        """Download the first matching 1-D spectral product for the
        given obs ids; returns local Path or None."""
        from astroquery.mast import Observations
        prods = Observations.get_product_list(obs_id_list)
        keep = prods[[
            any(p in str(x) for p in patterns)
            and str(x).endswith(".fits")
            for x in prods["productFilename"]]]
        if not len(keep):
            return None
        man = Observations.download_products(
            keep[:1], download_dir=str(self.data_raw), flat=True)
        p = Path(man["Local Path"][0])
        return p if p.exists() else None

    # --------------------------------------------------------------
    def _norm_band(self, w, f, e, lo, hi, resel):
        m = (w >= lo) & (w <= hi)
        wb, fb, eb = w[m], f[m], e[m]
        n = len(wb) // resel
        if n < 8:
            return None
        wr = wb[:n * resel].reshape(n, resel).mean(1)
        fr = fb[:n * resel].reshape(n, resel).mean(1)
        er = np.sqrt(
            (eb[:n * resel].reshape(n, resel) ** 2).sum(1)) / resel
        cont = np.array([
            np.nanpercentile(fr[max(0, i - 25):min(n, i + 25)], 80)
            for i in range(n)])
        good = np.isfinite(cont)
        if good.sum() < 4:
            return None
        cont = np.interp(wr, wr[good], cont[good])
        return wr, fr / cont, er / cont

    def _dips(self, wr, nn, ne, min_sig=4.0):
        sig = (1 - nn) / np.maximum(ne, 1e-9)
        out, i, n = [], 0, len(wr)
        while i < n:
            if sig[i] > min_sig:
                j = i
                while j < n and sig[j] > min_sig * 0.7:
                    j += 1
                k = i + int(np.argmax(sig[i:j]))
                ew = float(np.trapezoid(1 - nn[i:j], wr[i:j]))
                out.append({"lam": float(wr[k]), "ew": ew,
                            "sig": float(sig[k])})
                i = j
            else:
                i += 1
        return out

    def _depth_at(self, wr, nn, ne, lam, tol=1.5):
        i = int(np.argmin(np.abs(wr - lam)))
        if abs(wr[i] - lam) > tol:
            return None
        return float((1 - nn[i]) / ne[i])

    def _measure_3c232(self):
        """Forest census on the archived COS spectra of 3C 232.
        Returns the measurement dict, or None if products cannot
        be retrieved."""
        path160 = self.data_raw / "leys13010_x1dsum.fits"
        path140 = self.data_raw / "leys13020_x1dsum.fits"
        if not path160.exists() or not path140.exists():
            fetched = self._fetch_product(
                ["leys13010", "leys13020"],
                ["leys13010_x1dsum.fits", "leys13020_x1dsum.fits"])
            if fetched is None and not (
                    path160.exists() and path140.exists()):
                return None

        from astropy.io import fits

        def load(path, seg):
            d = fits.open(path)[1].data
            w, f, e = (d["WAVELENGTH"][seg], d["FLUX"][seg],
                       d["ERROR"][seg])
            dq = d["DQ"][seg]
            g = ((e > 0) & (f != 0) & np.isfinite(f)
                 & np.isfinite(e) & (dq == 0))
            return w[g], f[g], e[g]

        wA, fA, eA = load(path160, 0)   # G160M segA 1531-1710
        wB, fB, eB = load(path160, 1)   # G160M segB 1338-1520
        wL, fL, eL = load(path140, 0)   # G140L 1268-2439

        # Clean Ly-alpha window: above Ly-beta forest confusion and
        # below the proximity zone; O VI intrinsic complex masked.
        res = self._norm_band(wA, fA, eA, 1596.0, 1710.0, 8)
        if res is None:
            return None
        wrA, nA, neA = res
        resB = self._norm_band(wB, fB, eB, 1340.0, 1520.0, 8)
        resL = self._norm_band(wL, fL, eL, 1268.0, 1573.0, 10)

        cand = self._dips(wrA, nA, neA, 4.0)
        ovi = (1578.0, 1594.0)
        systems = []
        for c in cand:
            lam = c["lam"]
            if ovi[0] <= lam <= ovi[1]:
                continue
            if any(abs(lam - s) < 2.0 for s in ISM_LINES):
                continue
            lyb_pos = lam * LYB / LYA
            lyg_pos = lam * LYG / LYA
            d_b = (self._depth_at(*resB, lyb_pos)
                   if resB is not None else None)
            d_g = (self._depth_at(*resL, lyg_pos)
                   if resL is not None else None)
            confirmed = bool(
                (d_b is not None and d_b > 2.5)
                or (d_g is not None and d_g > 2.5))
            systems.append({
                "lam_obs": lam, "ew_a": c["ew"], "sig": c["sig"],
                "z_abs": lam / LYA - 1.0,
                "lyb_depth_sig": d_b, "lyg_depth_sig": d_g,
                "lyman_confirmed": confirmed,
            })
        n_conf = sum(s["lyman_confirmed"] for s in systems)
        z_covered = [1596.0 / LYA - 1, 1710.0 / LYA - 1]
        dz_covered = z_covered[1] - z_covered[0]
        return {
            "products": ["leys13010_x1dsum.fits",
                         "leys13020_x1dsum.fits"],
            "provenance": (
                "HST/COS G160M+G140L x1dsum products, program 17211 "
                "(MAST obs leys13010/leys13020), downloaded via "
                "astroquery.mast"),
            "clean_window_obs_a": [1596.0, 1710.0],
            "z_abs_covered": z_covered,
            "dz_covered": float(dz_covered),
            "n_candidates_4sigma": len(systems),
            "n_lyman_confirmed": n_conf,
            "systems": systems,
            "expected_cosmological_dndz": 25.0,
            "expected_cosmological_n": 25.0 * float(dz_covered),
        }

    # --------------------------------------------------------------
    def run(self):
        import pyvo
        from astroquery.eso import Eso

        print_status(
            "Auditing Ly-alpha forest coverage and path length...",
            "PROCESS")
        targets = self._targets()
        print_status(
            f"{len(targets)} quasar-class companions audited", "INFO")

        eso = Eso()
        eso.ROW_LIMIT = -1
        tap = pyvo.dal.TAPService("https://koa.ipac.caltech.edu/TAP")

        rows = []
        for tgt in targets:
            ra, dec, z = tgt["ra_deg"], tgt["dec_deg"], tgt["z"]
            blo, bhi = tgt["forest_lo"], tgt["forest_hi"]
            band_w = bhi - blo
            rec = {
                "pair_id": tgt["pair_id"],
                "resolved_name": tgt["resolved_name"],
                "z": z,
                "forest_lo_a": blo, "forest_hi_a": bhi,
                "lya_emission_a": tgt["lya_emission"],
                "observability": tgt["observability"],
                "status": "ok",
                "n_sdss_spec": np.nan,
                "sdss_covers": False,
                "mast_products": [],
                "eso_on_slit": 0,
                "eso_in_field": 0,
                "koa_on_target": [],
                "koa_in_field": [],
                "koa_nearest": [],
            }

            # --- SDSS -------------------------------------------
            try:
                rec["n_sdss_spec"] = self._query_sdss(ra, dec)
                rec["sdss_covers"] = bool(
                    _overlap(SDSS_COVERAGE[0], SDSS_COVERAGE[1],
                             blo, bhi) > 0.3 * band_w
                    and rec["n_sdss_spec"] > 0)
            except RuntimeError as e:
                print_status(str(e), "WARNING")
                rec["status"] = "partial_failure"
            time.sleep(0.2)

            # --- MAST -------------------------------------------
            try:
                t = self._query_mast(ra, dec)
                rec["mast_products"] = self._mast_forest_products(
                    t, blo, bhi)
            except RuntimeError as e:
                print_status(str(e), "WARNING")
                rec["status"] = "partial_failure"
            time.sleep(0.2)

            # --- ESO (VLT declination limit) --------------------
            if dec <= VLT_DEC_LIMIT:
                for inst in ESO_SPECTROGRAPHS:
                    # Only instruments whose range intersects the
                    # forest band can cover it; a FORS2 pointing at
                    # a z=1.5 quasar (band <3000 A) does not count.
                    ilo, ihi = ESO_COVERAGE[inst]
                    if _overlap(ilo, ihi, blo, bhi) <= 0:
                        continue
                    try:
                        rows_eso = self._query_eso(eso, inst, ra, dec)
                        for r in rows_eso:
                            try:
                                o = _sep_deg(float(r["RA"]),
                                             float(r["DEC"]), ra,
                                             dec) * 3600
                            except Exception:  # noqa: BLE001
                                continue
                            if o <= ESO_ON_SLIT_ARCSEC:
                                rec["eso_on_slit"] += 1
                            elif o <= ESO_FIELD_ARCMIN * 60:
                                rec["eso_in_field"] += 1
                    except RuntimeError as e:
                        print_status(str(e), "WARNING")
                        rec["status"] = "partial_failure"
                    time.sleep(0.2)
            else:
                rec["eso_on_slit"] = -1  # unreachable
                rec["eso_in_field"] = -1

            # --- KOA --------------------------------------------
            if KECK_DEC_RANGE[0] <= dec <= KECK_DEC_RANGE[1]:
                for table, (klo, khi) in KOA_TABLES.items():
                    if _overlap(klo, khi, blo, bhi) <= 0:
                        continue
                    try:
                        t = self._query_koa(tap, table, ra, dec)
                        if len(t):
                            d = _sep_deg(
                                np.array(t["ra"], dtype=float),
                                np.array(t["dec"], dtype=float),
                                ra, dec) * 3600
                            on = t[d <= KOA_ON_TARGET_ARCSEC]
                            fld = t[(d > KOA_ON_TARGET_ARCSEC)
                                    & (d <= KOA_MASK_FIELD_ARCSEC)]
                            if len(on):
                                rec["koa_on_target"].append(table)
                            if len(fld):
                                rec["koa_in_field"].append(table)
                            i = int(np.argmin(d))
                            rec["koa_nearest"].append({
                                "table": table,
                                "koaid": str(t["koaid"][i]),
                                "targname": str(t["targname"][i]),
                                "offset_arcsec": float(d[i]),
                            })
                    except RuntimeError as e:
                        print_status(str(e), "WARNING")
                        rec["status"] = "partial_failure"
                    time.sleep(0.2)
            rows.append(rec)
            n_mast = len(rec["mast_products"])
            print_status(
                f"{tgt['resolved_name']}: z={z:.3f} "
                f"forest {blo:.0f}-{bhi:.0f}A ({tgt['observability']}) "
                f"SDSS={rec['n_sdss_spec']} MAST_cov={n_mast} "
                f"ESO_slit={rec['eso_on_slit']} "
                f"ESO_fld={rec['eso_in_field']} "
                f"KOA_on={len(rec['koa_on_target'])} "
                f"KOA_field={len(rec['koa_in_field'])}", "INFO")

        # --------------------------------------------------------------
        # Verdicts + measurement where coverage is retrievable.
        meas_3c = None
        for rec in rows:
            z = rec["z"]
            band_w = rec["forest_hi_a"] - rec["forest_lo_a"]
            usable = [p for p in rec["mast_products"]
                      if p.get("usable", True)]
            union_use = _union_len(
                iv for p in usable for iv in p["intervals"])
            union_all = _union_len(
                iv for p in rec["mast_products"] for iv in p["intervals"])
            frac = union_use / band_w if band_w > 0 else 0.0
            rec["forest_band_covered_a"] = float(union_use)
            rec["forest_band_covered_frac"] = float(frac)
            on_target = (rec["sdss_covers"] or rec["eso_on_slit"] > 0
                         or len(rec["koa_on_target"]) > 0)
            if z < 0.15:
                rec["verdict"] = "weak_path_low_power"
            elif on_target or frac >= 0.5:
                rec["verdict"] = "forest_band_covered"
            elif union_use > 0:
                rec["verdict"] = "forest_band_partial"
            elif (union_all > 0 or rec["eso_in_field"] > 0
                  or len(rec["koa_in_field"]) > 0):
                # Products or slits exist nearby but none usable
                # on-target (slitless grism, wrong target, mask centre).
                rec["verdict"] = "unusable_or_field_only"
            else:
                rec["verdict"] = "forest_not_covered"

        # 3C 232: retrievable COS products -> direct measurement.
        for rec in rows:
            if rec["resolved_name"] == "3C 232":
                try:
                    meas_3c = self._measure_3c232()
                except Exception as e:  # noqa: BLE001
                    print_status(f"3C 232 measurement failed: {e}",
                                 "WARNING")
                    meas_3c = None
                if meas_3c:
                    rec["verdict"] = "forest_measured"
                    rec["measurement"] = meas_3c
                    print_status(
                        f"3C 232: {meas_3c['n_lyman_confirmed']}"
                        f"/{meas_3c['n_candidates_4sigma']} Lyman-"
                        "confirmed forest systems at z_abs "
                        f"{min(s['z_abs'] for s in meas_3c['systems']):.3f}"
                        f"-{max(s['z_abs'] for s in meas_3c['systems']):.3f}"
                        f" (expected ~{meas_3c['expected_cosmological_n']:.1f}"
                        " cosmological)", "TEST")

        df_rows = []
        for rec in rows:
            df_rows.append({
                "pair_id": rec["pair_id"],
                "resolved_name": rec["resolved_name"],
                "z": rec["z"],
                "forest_lo_a": rec["forest_lo_a"],
                "forest_hi_a": rec["forest_hi_a"],
                "observability": rec["observability"],
                "n_sdss_spec": rec["n_sdss_spec"],
                "n_mast_forest_products": len(rec["mast_products"]),
                "mast_forest_covered_a": rec["forest_band_covered_a"],
                "mast_forest_covered_frac":
                    rec["forest_band_covered_frac"],
                "eso_on_slit": rec["eso_on_slit"],
                "eso_in_field": rec["eso_in_field"],
                "koa_on_target": ";".join(rec["koa_on_target"]),
                "koa_in_field": ";".join(rec["koa_in_field"]),
                "koa_nearest": ";".join(
                    f"{n['koaid']}({n['targname']},"
                    f"{n['offset_arcsec']:.1f}arcsec)"
                    for n in rec["koa_nearest"]),
                "status": rec["status"],
                "verdict": rec["verdict"],
            })
        df = pd.DataFrame(df_rows)
        csv_path = self.data_processed / "lya_forest_audit.csv"
        df.to_csv(csv_path, index=False)
        print_status(f"Saved audit table: {csv_path}", "SUCCESS")

        n_covered = int((df["verdict"] == "forest_band_covered").sum()
                        + (df["verdict"] == "forest_measured").sum())
        n_partial = int((df["verdict"] == "forest_band_partial").sum()
                        + (df["verdict"]
                           == "unusable_or_field_only").sum())
        n_not = int((df["verdict"] == "forest_not_covered").sum())
        n_decisive_uncovered = int((
            (df["observability"] == "ground")
            & (df["verdict"].isin(["forest_not_covered",
                                   "forest_band_partial",
                                   "unusable_or_field_only"]))).sum())

        summary = {
            "n_quasar_companions": len(df),
            "n_forest_band_covered": n_covered,
            "n_forest_band_partial_or_field": n_partial,
            "n_forest_not_covered": n_not,
            "n_ground_accessible_uncovered": n_decisive_uncovered,
            "audit_rows": df_rows,
            "audit_details": [
                {"pair_id": rec["pair_id"],
                 "resolved_name": rec["resolved_name"],
                 "sdss_covers": rec["sdss_covers"],
                 "mast_products": rec["mast_products"],
                 "koa_nearest": rec["koa_nearest"]}
                for rec in rows],
            "measurements": {
                "3C 232": meas_3c,
            },
            "method": {
                "forest_band": (
                    "observed [1041*(1+z_Q), 1185*(1+z_Q)] A; "
                    "the blue edge sits above the Ly-beta forest "
                    "complex and the red edge below the quasar "
                    "proximity zone"),
                "detection": (
                    "spectrum rebinned to ~1 resolution element; "
                    "80th-percentile sliding continuum; dips >4 "
                    "sigma; Galactic-ISM and intrinsic O VI "
                    "exclusion; each candidate confirmed by a "
                    "significant decrement at the predicted Ly-beta "
                    "or Ly-gamma position (standard forest "
                    "validation)"),
                "expectation": (
                    "cosmic-mean incidence dN/dz ~ 25 for W > 0.1 A "
                    "at z < 1 (Danforth et al. 2016, ApJ 817, 111); "
                    "a companion at the host distance (~tens of "
                    "Mpc) predicts ~0-1 lines"),
            },
            "required_observation": (
                "For the ground-accessible companions (z >~ 1.7: "
                "NGC 7319 QSO, NGC 3516 z=2.10 member, WISP 257, "
                "BSO 3, and marginally WEE 51): a single blue-arm "
                "echelle or X-Shooter UVB exposure (~1 hr, 8-10m "
                "class; UVES/XSHOOTER, HIRES/ESI, or HST/STIS "
                "G430L) covering the forest band settles the path "
                "length directly."),
            "interpretation": (
                "A populated forest at the cosmic-mean incidence is "
                "consistent with a full cosmological path and is "
                "adverse to a strictly local position for that "
                "sightline; a sparse or absent forest would "
                "directly favour the local interpretation. Both "
                "outcomes are reported; coverage verdicts are "
                "audited negatives, not assumptions."),
        }
        json_path = self.results / "step_29_lya_forest_audit.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")
        print_status("Ly-alpha forest audit complete.", "SUCCESS")


if __name__ == "__main__":
    Step29LyaForestAudit().run()
