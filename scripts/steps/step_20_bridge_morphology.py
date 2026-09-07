#!/usr/bin/env python3
"""
Step 20: Bridge Morphology Analysis
===================================
Resolves the companion of each Arp pair to an archive position and
extracts surface-brightness transects from the archival imaging
downloaded in step_02 to test for the presence of the claimed
luminous bridges and filaments.

Companion resolution is archival throughout — no assumed position
angle is ever used.  For each pair the resolver tries, in order:

  1. a NED object-name query for the catalogued companion name
     (with name variants where the literature uses several);
  2. a Simbad cone search about the host nucleus out to
     ``max(1.5 x separation, 60 arcsec)``, selecting sources whose
     redshift matches the catalogued companion redshift(s);
  3. a VizieR cone search of the Milliquas catalogue (VII/290)
     over the same region, restricted to confirmed quasar-class
     objects (classes Q, A, B, K), again matched in redshift;
  4. a NED cone search over the same region (slowest; reached for
     compact-galaxy companions that Simbad does not catalogue).

Multi-object companions (the NGC 1073 disk quasars, the NGC 3628
minor-axis quasars) resolve every catalogued redshift; the member
matched to the primary ``z_comp`` carries the transect endpoint and
all matched members are written to ``companion_positions.csv`` for
the exclusion lists used by the association and chance-alignment
tests downstream.

For each pair field a primary transect is drawn along the measured
galaxy-companion axis.  Control transects of identical length and
width are drawn at eight rotated position angles that sample the
galaxy's outer isophotes away from the companion direction.  The
detection statistic is the mean surface brightness of the bridge
transect in the inter-object segment (the middle 45-80% of the
axis, which excludes both nuclear regions) compared with the mean
and scatter of the control transects over the same fractional
segment:

    S_bridge = (mu_bridge - mean(mu_ctrl)) / std(mu_ctrl)

A significant positive excess along the companion direction
indicates a morphological structure where none is expected from
the azimuthally averaged galaxy light.  ``bridge_claimed`` records
whether the pair's published connection is a luminous structure
(luminous_bridge or luminous_filament); for the remaining
morphologies the transect is a contextual measurement of the
companion sight-line, not a test of the claimed evidence.

For silhouette-type pairs a complementary decrement statistic is
computed: the surface brightness in an annulus about the companion
position (excluding the compact object itself) is compared with
control annuli at the same galactocentric radius and rotated
position angles.  A negative S_silhouette indicates the darkened
ring claimed by Arp (1978) for the NGC 1199 companion.

This step requires the FITS images produced by step_02 and fails
loudly when imaging is missing.  If a companion cannot be resolved
in any archive the pair is marked ``companion_unresolved`` and no
transect statistic is reported.

Pairs carrying catalogued filament knots (``filament_knots_z``;
currently the NGC 7603 filament) additionally resolve each knot to an
archive position, and the projected transect coordinate on the
resolved host->companion axis is written to
``filament_knot_positions.csv`` for the field-profile inference of
steps 30-31.  The step fails loudly if a catalogued knot cannot be
resolved in any archive.

Outputs:
    results/outputs/step_20_bridge_morphology.json
    data/processed/bridge_morphology.csv
    data/processed/bridge_transects.csv
    data/processed/companion_positions.csv
    data/processed/filament_knot_positions.csv
    results/figures/step_20_bridge_morphology.png
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy import units as u

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.utils.logger import TEPLogger, set_step_logger, print_status
from scripts.utils.jsonio import json_safe
from scripts.utils.plot_style import apply_tep_style

N_CONTROLS = 8          # control transect position angles
TRANSECT_HALF_WIDTH = 3  # pixels each side of the axis

# Fractional endpoints of the inter-object segment: the statistic
# is evaluated over the middle of the galaxy-companion axis so that
# neither the host nucleus nor the companion's own light enters it.
SEG_LO, SEG_HI = 0.45, 0.80

# Companion-resolution tolerances
Z_MATCH_REL = 0.10           # |z_obj - z_target| / z_target
Z_MATCH_REL_APPROX = 0.30    # widened tolerance for literature_approx entries
CONE_FACTOR = 1.5            # cone radius = factor x catalog separation
CONE_MIN_ARCSEC = 60.0       # minimum cone-search radius
NED_CONE_MAX_ARCMIN = 10.0   # NED server limit for cone searches
DEDUPE_ARCSEC = 5.0          # candidates closer than this are one object
NED_TIMEOUT_S = 120          # NED cone searches are slow

# Bridge-axis statistics are the targeted test only where the
# published connection is a luminous structure.
BRIDGE_CLAIM_TYPES = {"luminous_bridge", "luminous_filament"}

# Milliquas confirmed quasar classes (VII/290 ReadMe, note 3);
# identical definition to step_22.
MILLIQUAS_CATALOG = "VII/290/catalog"
CONFIRMED_CLASSES = ("Q", "A", "B", "K")

# Name variants for companions resolvable by NED object-name query.
# Companions entered in the catalog as descriptive labels
# (containing parentheses) are never name-resolvable and go
# straight to the cone-search stages.
NAME_VARIANTS = {
    "Markarian 205": ["Markarian 205", "Mrk 205", "MRK 205", "MRK 0205"],
    "NGC 7603B": ["NGC 7603B", "NGC7603B", "NGC 7603 B"],
    "3C 232": ["3C 232", "3C232", "3C  232"],
}


def _rel_z(z_obj, z_target):
    """Relative redshift mismatch |z_obj - z_target| / z_target.

    The redshift itself is the identity here — a z = 0.01 object is
    not a candidate for a z = 0.044 companion even though the two
    agree to three per cent in 1 + z."""
    if z_target <= 0:
        return np.inf
    return abs(z_obj - z_target) / z_target


class Step20BridgeMorphology:
    """Step 20: Companion astrometry + surface-brightness transects."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.data_raw = self.root / "data" / "raw" / "skyview"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_20",
            log_file_path=self.logs / "step_20_bridge_morphology.log",
        )
        set_step_logger(self.logger)

    # ------------------------------------------------------------------
    # Companion resolution
    # ------------------------------------------------------------------

    @staticmethod
    def _companion_z_targets(row):
        """All catalogued companion redshifts: primary z_comp plus any
        members of a ``z_comp_list`` (multi-object companions)."""
        targets = [float(row["z_comp"])]
        zl = row.get("z_comp_list")
        if isinstance(zl, str) and zl.strip():
            for tok in zl.split(";"):
                try:
                    targets.append(float(tok))
                except ValueError:
                    pass
        return targets

    def _ned_name_query(self, name):
        """NED object-name query with retries; returns SkyCoord or None."""
        from astroquery.ipac.ned import Ned
        from astropy.coordinates import SkyCoord
        from astropy import units as u

        for attempt in (1, 2, 3):
            try:
                tbl = Ned.query_object(name)
                if tbl is not None and len(tbl) > 0:
                    return SkyCoord(
                        float(tbl[0]["RA"]) * u.deg,
                        float(tbl[0]["DEC"]) * u.deg,
                    ), str(tbl[0]["Object Name"])
                return None, None
            except Exception as e:
                msg = str(e)
                if "not currently recognized" in msg:
                    self.logger.info(
                        f"NED name lookup: '{name}' is not a resolvable "
                        "catalogue identifier; proceeding to cone "
                        "searches."
                    )
                    return None, None
                short = msg.strip().splitlines()[0]
                if attempt < 3:
                    self.logger.info(
                        f"NED name query '{name}' attempt {attempt}/3 "
                        f"failed ({short}); retrying."
                    )
                    time.sleep(2.0 * attempt)
                else:
                    self.logger.info(
                        f"NED name query '{name}' failed after 3 "
                        f"attempts ({short}); proceeding to cone "
                        "searches."
                    )
                    return None, None
        return None, None

    def _simbad_cone(self, gpos, radius_deg):
        """Simbad cone search with redshift and object-type fields."""
        from astroquery.simbad import Simbad

        try:
            s = Simbad()
            s.add_votable_fields("rvz_redshift", "otype")
            t = s.query_region(gpos, radius=radius_deg * 3600.0 * u.arcsec)
            return t
        except Exception as e:
            short = str(e).strip().splitlines()[0]
            self.logger.info(
                f"Simbad cone query failed ({short}); trying "
                "Milliquas/NED."
            )
            return None

    def _milliquas_cone(self, gpos, radius_deg):
        """VizieR Milliquas cone search; returns the catalogue table."""
        from astroquery.vizier import Vizier

        try:
            viz = Vizier(row_limit=-1, timeout=60)
            res = viz.query_region(
                gpos, radius=radius_deg * u.deg, catalog=MILLIQUAS_CATALOG
            )
            if res is None or MILLIQUAS_CATALOG not in res.keys():
                return None
            return res[MILLIQUAS_CATALOG]
        except Exception as e:
            short = str(e).strip().splitlines()[0]
            self.logger.info(
                f"VizieR cone query failed ({short}); trying NED."
            )
            return None

    def _ned_cone(self, gpos, radius_deg):
        """NED cone search (slow; last resort).  Returns the table.
        The NED cone service rejects very wide apertures, so the radius
        is capped at NED_CONE_MAX_ARCMIN; the quasar-class companions
        that populate the widest fields are in any case recovered by
        the Simbad and Milliquas stages."""
        from astroquery.ipac.ned import Ned

        radius_arcmin = radius_deg * 60.0
        if radius_arcmin > NED_CONE_MAX_ARCMIN:
            self.logger.info(
                f"NED cone radius {radius_arcmin:.1f} arcmin exceeds "
                f"the service limit; capped at {NED_CONE_MAX_ARCMIN} "
                "arcmin."
            )
            radius_deg = NED_CONE_MAX_ARCMIN / 60.0

        for attempt in (1, 2, 3):
            try:
                return Ned.query_region(
                    gpos, radius=radius_deg * 3600.0 * u.arcsec
                )
            except Exception as e:
                short = str(e).strip().splitlines()[0]
                if attempt < 3:
                    self.logger.info(
                        f"NED cone query attempt {attempt}/3 failed "
                        f"({short}); retrying."
                    )
                    time.sleep(3.0 * attempt)
                else:
                    self.logger.warning(
                        f"NED cone query failed after 3 attempts: "
                        f"{short}"
                    )
        return None

    def _dedupe_candidates(self, cands, z_targets):
        """Merge candidates that coincide within DEDUPE_ARCSEC —
        the same object catalogued in both Simbad and Milliquas.
        Where the catalogues disagree on redshift, the entry whose
        redshift is closer to a catalogued companion target is kept,
        so that an aggregated-archive outlier cannot shadow the
        spectroscopic value the pair actually claims."""
        keep = []
        for c in cands:
            dup_i = None
            for i, k in enumerate(keep):
                if c["coord"].separation(k["coord"]).to(u.arcsec).value < DEDUPE_ARCSEC:
                    dup_i = i
                    break
            if dup_i is None:
                keep.append(c)
            else:
                k = keep[dup_i]
                c_best = min(_rel_z(c["z"], zt) for zt in z_targets)
                k_best = min(_rel_z(k["z"], zt) for zt in z_targets)
                if c_best < k_best:
                    keep[dup_i] = c
        return keep

    def _match_targets(self, candidates, z_targets, gpos, rel_tol):
        """Match catalogued companion redshifts to cone-search rows.

        ``candidates`` is a list of dicts with keys name, coord, z.
        Returns {z_target: best-match dict}; a match requires the
        relative redshift mismatch |z_obj - z_target|/z_target to be
        below ``rel_tol``.  Each target takes its closest-in-z
        candidate; a candidate may not be claimed by two targets
        (the better-matching target wins)."""
        matches = {}
        used = set()
        order = sorted(z_targets)  # deterministic claim order
        for zt in order:
            best, best_rel = None, rel_tol
            for i, c in enumerate(candidates):
                if i in used or c["z"] is None or not np.isfinite(c["z"]):
                    continue
                f = _rel_z(c["z"], zt)
                if f < best_rel:
                    best, best_rel = i, f
            if best is not None:
                used.add(best)
                c = dict(candidates[best])
                c["z_target"] = zt
                c["z_match_rel"] = best_rel
                c["sep_arcsec"] = float(
                    gpos.separation(c["coord"]).to(u.arcsec).value
                )
                matches[zt] = c
        return matches

    def _resolve_companions(self, row, gpos):
        """Resolve the companion(s) of one pair to archive positions.

        Returns (matches_by_target, method) where matches_by_target
        maps each catalogued companion redshift to a dict with the
        resolved coordinates, archive name and redshift."""
        from astropy.coordinates import SkyCoord

        comp_name = str(row["companion"])
        z_targets = self._companion_z_targets(row)
        sep_arcsec = float(row["separation_arcsec"])
        radius_deg = max(CONE_FACTOR * sep_arcsec, CONE_MIN_ARCSEC) / 3600.0
        rel_tol = (
            Z_MATCH_REL_APPROX
            if row.get("redshift_source") == "literature_approx"
            else Z_MATCH_REL
        )

        # Stage 1: NED object-name query (only for resolvable names —
        # descriptive labels like "QSO (z=2.114)" go to cone search).
        if "(" not in comp_name:
            for variant in NAME_VARIANTS.get(comp_name, [comp_name]):
                pos, resolved = self._ned_name_query(variant)
                if pos is not None:
                    zt = z_targets[0]
                    return {
                        zt: {
                            "name": resolved,
                            "coord": pos,
                            "z": zt,  # name-resolved; catalogue z carried
                            "z_target": zt,
                            "z_match_rel": 0.0,
                            "sep_arcsec": float(
                                gpos.separation(pos).to(u.arcsec).value
                            ),
                        }
                    }, "ned_name"
                time.sleep(0.3)

        # Stage 2: merged Simbad + Milliquas cone searches, matched
        # in redshift against every catalogued companion redshift.
        cands = []
        tbl = self._simbad_cone(gpos, radius_deg)
        if tbl is not None and len(tbl) > 0:
            for r in tbl:
                val = r["rvz_redshift"]
                if np.ma.is_masked(val):
                    continue
                try:
                    z = float(val)
                    if not np.isfinite(z):
                        continue
                except (TypeError, ValueError):
                    continue
                cands.append({
                    "name": str(r["main_id"]),
                    "coord": SkyCoord(
                        float(r["ra"]) * u.deg, float(r["dec"]) * u.deg
                    ),
                    "z": z,
                    "src": "simbad",
                })
        tbl = self._milliquas_cone(gpos, radius_deg)
        if tbl is not None and len(tbl) > 0:
            for r in tbl:
                t = str(r["Type"]).strip()
                if not t or t[0] not in CONFIRMED_CLASSES:
                    continue
                try:
                    z = float(r["z"])
                    if not np.isfinite(z):
                        continue
                except (TypeError, ValueError):
                    continue
                cands.append({
                    "name": str(r["Name"]),
                    "coord": SkyCoord(
                        float(r["RAJ2000"]) * u.deg,
                        float(r["DEJ2000"]) * u.deg,
                    ),
                    "z": z,
                    "src": "milliquas",
                })
        cands = self._dedupe_candidates(cands, z_targets)
        matches = self._match_targets(cands, z_targets, gpos, rel_tol)
        if len(matches) == len(set(z_targets)):
            return matches, "simbad_milliquas_cone_zmatch"

        # Stage 3: NED cone search (slow; reached for compact-galaxy
        # companions that Simbad and Milliquas do not catalogue).
        tbl = self._ned_cone(gpos, radius_deg)
        if tbl is not None and len(tbl) > 0:
            ncands = []
            for r in tbl:
                val = r["Redshift"]
                if np.ma.is_masked(val):
                    continue
                try:
                    z = float(val)
                    if not np.isfinite(z):
                        continue
                except (TypeError, ValueError):
                    continue
                ncands.append({
                    "name": str(r["Object Name"]),
                    "coord": SkyCoord(
                        float(r["RA"]) * u.deg, float(r["DEC"]) * u.deg
                    ),
                    "z": z,
                    "src": "ned",
                })
            nmatches = self._match_targets(ncands, z_targets, gpos, rel_tol)
            for zt, m in nmatches.items():
                matches.setdefault(zt, m)
            if matches:
                method = "ned_cone_zmatch" if nmatches else "mixed"
                return matches, method

        if matches:
            return matches, "simbad_milliquas_cone_zmatch"
        return {}, "unresolved"

    def _resolve_filament_knots(self, row, gpos, cpos):
        """Resolve the catalogued filament knots to archive positions.

        For pairs carrying a ``filament_knots_z`` field (currently only
        the NGC 7603 filament) each knot redshift is matched to a
        Simbad cone-search candidate, exactly as companion resolution
        does.  The returned records carry the measured transect
        coordinate x — the projection of the knot position onto the
        resolved host -> companion axis — and the perpendicular offset
        of the knot from the straight axis (the filament is curved, so
        non-zero offsets are expected and reported)."""
        from astropy.coordinates import SkyCoord

        raw = row.get("filament_knots_z")
        if not isinstance(raw, str) or not raw.strip():
            return []
        z_knots = [float(t) for t in raw.split(";")]

        sep_arcsec = float(gpos.separation(cpos).to(u.arcsec).value)
        radius_deg = max(CONE_FACTOR * sep_arcsec, CONE_MIN_ARCSEC) / 3600.0
        tbl = self._simbad_cone(gpos, radius_deg)
        if tbl is None or len(tbl) == 0:
            self.logger.warning(
                f"{row['pair_id']}: Simbad cone returned nothing for "
                "filament knots"
            )
            return []

        cands = []
        for r in tbl:
            val = r["rvz_redshift"]
            if np.ma.is_masked(val):
                continue
            try:
                z = float(val)
                if not np.isfinite(z):
                    continue
            except (TypeError, ValueError):
                continue
            cands.append({
                "name": str(r["main_id"]),
                "coord": SkyCoord(
                    float(r["ra"]) * u.deg, float(r["dec"]) * u.deg
                ),
                "z": z,
            })

        # Host -> companion axis in a local tangent frame
        # (delta_RA weighted by cos(dec)).
        cosd = np.cos(np.radians(gpos.dec.deg))
        d = np.array([
            (cpos.ra.deg - gpos.ra.deg) * cosd,
            cpos.dec.deg - gpos.dec.deg,
        ])
        d2 = float(d @ d)

        out = []
        for zk in z_knots:
            best, best_rel = None, Z_MATCH_REL
            for c in cands:
                f = _rel_z(c["z"], zk)
                if f < best_rel:
                    best, best_rel = c, f
            if best is None:
                self.logger.warning(
                    f"{row['pair_id']}: no archive candidate for "
                    f"filament knot z={zk}"
                )
                continue
            v = np.array([
                (best["coord"].ra.deg - gpos.ra.deg) * cosd,
                best["coord"].dec.deg - gpos.dec.deg,
            ])
            x = float((v @ d) / d2)
            perp_deg = float(np.linalg.norm(v - x * d))
            out.append({
                "pair_id": row["pair_id"],
                "resolved_name": best["name"],
                "ra_deg": float(best["coord"].ra.deg),
                "dec_deg": float(best["coord"].dec.deg),
                "z": float(best["z"]),
                "z_target": float(zk),
                "z_match_rel": float(best_rel),
                "x": x,
                "perp_offset_arcsec": perp_deg * 3600.0,
                "resolution_method": "simbad_cone_zmatch",
            })
        out.sort(key=lambda r: r["x"])
        return out

    # ------------------------------------------------------------------
    # Transect extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _transect(image, x0, y0, x1, y1, half_width=TRANSECT_HALF_WIDTH,
                  n_samples=None):
        """Mean surface brightness along a line from (x0,y0) to (x1,y1).

        Returns the 1-D profile (mean across a strip of half-width
        perpendicular to the axis) sampled at n points along the axis.
        """
        if n_samples is None:
            n = int(round(np.hypot(x1 - x0, y1 - y0)))
        else:
            n = n_samples
        n = max(n, 8)
        xs = np.linspace(x0, x1, n)
        ys = np.linspace(y0, y1, n)

        # perpendicular unit vector
        dx, dy = x1 - x0, y1 - y0
        L = np.hypot(dx, dy)
        px, py = -dy / L, dx / L

        profile = np.full(n, np.nan)
        h, w = image.shape
        for i in range(n):
            vals = []
            for off in range(-half_width, half_width + 1):
                xi = int(round(xs[i] + off * px))
                yi = int(round(ys[i] + off * py))
                if 0 <= xi < w and 0 <= yi < h:
                    vals.append(image[yi, xi])
            if vals:
                profile[i] = np.nanmean(vals)
        return profile

    def _path_transect(self, image, pts, half_width=TRANSECT_HALF_WIDTH):
        """Surface-brightness profile along a polyline of pixel
        vertices, indexed by arc length.  Used for the curved
        filament of NGC 7603, whose archive-resolved knots sit
        off the straight host->companion axis."""
        prof = []
        for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
            n = max(int(round(np.hypot(x1 - x0, y1 - y0))), 4)
            leg = self._transect(
                image, x0, y0, x1, y1, half_width, n_samples=n
            )
            if len(prof):
                leg = leg[1:]
            prof.extend(leg.tolist())
        return np.asarray(prof)

    @staticmethod
    def _rotate_points(pts, centre, ang_rad):
        """Rigidly rotate a list of pixel vertices about centre."""
        c, s = np.cos(ang_rad), np.sin(ang_rad)
        cx, cy = centre
        return [
            (
                cx + (x - cx) * c - (y - cy) * s,
                cy + (x - cx) * s + (y - cy) * c,
            )
            for x, y in pts
        ]

    @staticmethod
    def _annulus_mean(image, xc, yc, r_in, r_out):
        """Mean surface brightness in the annulus r_in < r <= r_out
        (pixels) centred on (xc, yc)."""
        h, w = image.shape
        x0 = max(int(np.floor(xc - r_out)), 0)
        x1 = min(int(np.ceil(xc + r_out)) + 1, w)
        y0 = max(int(np.floor(yc - r_out)), 0)
        y1 = min(int(np.ceil(yc + r_out)) + 1, h)
        yy, xx = np.mgrid[y0:y1, x0:x1]
        rr = np.hypot(xx - xc, yy - yc)
        mask = (rr > r_in) & (rr <= r_out)
        if not mask.any():
            return np.nan
        return float(np.nanmean(image[y0:y1, x0:x1][mask]))

    def _silhouette_stat(self, image, gx, gy, cx, cy, scale_arcsec_per_pix):
        """Decrement statistic for silhouette pairs: the surface
        brightness of an annulus around the companion (compact object
        excluded) versus control annuli at the same galactocentric
        radius and rotated position angles."""
        dist_pix = np.hypot(cx - gx, cy - gy)
        pa = np.arctan2(cy - gy, cx - gx)
        # Annulus: exclude the compact object (~4 arcsec) and sample
        # the surrounding ring to ~12 arcsec.
        r_in = 4.0 / scale_arcsec_per_pix
        r_out = 12.0 / scale_arcsec_per_pix
        mu_ann = self._annulus_mean(image, cx, cy, r_in, r_out)
        ctrl = []
        for k in range(N_CONTROLS):
            pa_c = pa + 2.0 * np.pi * (k + 1) / (N_CONTROLS + 1)
            x2 = gx + dist_pix * np.cos(pa_c)
            y2 = gy + dist_pix * np.sin(pa_c)
            v = self._annulus_mean(image, x2, y2, r_in, r_out)
            if np.isfinite(v):
                ctrl.append(v)
        if len(ctrl) < 3 or not np.isfinite(mu_ann):
            return np.nan, np.nan, np.nan
        mu_c, sd_c = float(np.mean(ctrl)), float(np.std(ctrl, ddof=1))
        s = (mu_ann - mu_c) / sd_c if sd_c > 0 else np.nan
        return mu_ann, mu_c, s

    # ------------------------------------------------------------------

    def run(self):
        from astropy.io import fits
        from astropy.wcs import WCS
        from astropy.coordinates import SkyCoord
        from astropy import units as u
        from astroquery.simbad import Simbad

        print_status("Analysing bridge morphology...", "PROCESS")

        catalog_path = self.data_processed / "arp_pair_catalog_verified.csv"
        if not catalog_path.exists():
            catalog_path = self.data_processed / "arp_pair_catalog.csv"
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"Pair catalog not found: {catalog_path}. Run step_00 first."
            )
        catalog = pd.read_csv(catalog_path)

        records = []
        profiles_out = []
        companion_rows = []
        knot_rows = []
        n_images = 0
        for _, row in catalog.iterrows():
            pair_id = row["pair_id"]

            # Prefer the deepest imaging: DESI Legacy DR10 r-band,
            # then SDSS r-band, then DSS2 Red.
            img_path = None
            for tag in ("LS-DR10r", "SDSSr", "DSS2_Red"):
                cand = self.data_raw / f"{pair_id}_{tag}.fits"
                if cand.exists():
                    img_path = cand
                    break
            if img_path is None:
                print_status(f"[MISS] no image for {pair_id}", "WARNING")
                records.append({"pair_id": pair_id, "status": "no_image"})
                continue

            n_images += 1
            with fits.open(img_path) as hdul:
                image = np.asarray(hdul[0].data, dtype=float).squeeze()
                wcs = WCS(hdul[0].header)
            if image.ndim != 2:
                image = np.squeeze(image)
                if image.ndim != 2:
                    records.append({"pair_id": pair_id, "status": "bad_image_shape"})
                    continue
            if float(np.mean(image != 0)) < 0.5:
                print_status(
                    f"[BLANK] {pair_id}: image has no coverage", "WARNING"
                )
                records.append({"pair_id": pair_id, "status": "blank_image"})
                continue

            # Resolve the host position (Simbad; CDS, not NED-dependent)
            try:
                ht = Simbad.query_object(row["galaxy"])
                gpos = SkyCoord(
                    float(ht[0]["ra"]) * u.deg, float(ht[0]["dec"]) * u.deg
                )
            except Exception as e:
                records.append(
                    {"pair_id": pair_id, "status": f"host_resolve_failed: {e}"}
                )
                continue

            # Resolve companion(s) through the archive chain
            matches, method = self._resolve_companions(row, gpos)
            z_primary = float(row["z_comp"])
            primary = matches.get(z_primary)
            if primary is None and matches:
                primary = matches[sorted(matches)[0]]

            for zt, m in matches.items():
                companion_rows.append({
                    "pair_id": pair_id,
                    "resolved_name": m["name"],
                    "ra_deg": float(m["coord"].ra.deg),
                    "dec_deg": float(m["coord"].dec.deg),
                    "z": float(m["z"]),
                    "z_target": float(zt),
                    "z_match_rel": float(m.get("z_match_rel", np.nan)),
                    "measured_separation_arcsec": float(m["sep_arcsec"]),
                    "resolution_method": method,
                    "is_primary": bool(m is primary),
                })

            if primary is None:
                records.append({
                    "pair_id": pair_id,
                    "status": "companion_unresolved",
                    "companion_resolved": False,
                    "resolution_method": method,
                })
                print_status(
                    f"{pair_id}: companion unresolved in NED/Simbad/"
                    "Milliquas — no transect drawn", "WARNING"
                )
                continue

            cpos = primary["coord"]
            meas_sep = float(gpos.separation(cpos).to(u.arcsec).value)

            # Filament knots (NGC 7603): archive-resolved positions and
            # measured transect coordinates on the host-companion axis.
            knots_here = []
            try:
                knots_here = self._resolve_filament_knots(row, gpos, cpos)
                knot_rows.extend(knots_here)
            except Exception as e:
                self.logger.warning(
                    f"{pair_id}: filament-knot resolution failed: {e}"
                )
            pa_sky = float(gpos.position_angle(cpos).to(u.deg).value)
            bridge_claimed = row["connection_type"] in BRIDGE_CLAIM_TYPES

            gx, gy = wcs.world_to_pixel(gpos)
            cx, cy = wcs.world_to_pixel(cpos)

            length = np.hypot(cx - gx, cy - gy)
            n_ax = max(int(round(length)), 8)
            bridge_prof = self._transect(image, gx, gy, cx, cy,
                                         n_samples=n_ax)
            pa_bridge = np.degrees(np.arctan2(cy - gy, cx - gx))

            # Control transects at rotated PAs, same length
            ctrl_profiles = []
            for k in range(N_CONTROLS):
                pa = pa_bridge + 360.0 * (k + 1) / (N_CONTROLS + 1)
                par = np.radians(pa)
                x2 = gx + length * np.cos(par)
                y2 = gy + length * np.sin(par)
                ctrl_profiles.append(
                    self._transect(image, gx, gy, x2, y2,
                                   n_samples=n_ax)
                )
            ctrl_profiles = np.array(ctrl_profiles)

            # Detection statistic over the inter-object segment
            # (fractional [SEG_LO, SEG_HI]: excludes both nuclei)
            i0 = int(np.floor(SEG_LO * len(bridge_prof)))
            i1 = max(i0 + 1, int(np.ceil(SEG_HI * len(bridge_prof))))
            seg = slice(i0, i1)
            mu_b = np.nanmean(bridge_prof[seg])
            mu_c = np.nanmean(ctrl_profiles[:, seg])
            sd_c = np.nanstd(ctrl_profiles[:, seg])
            s_stat = (mu_b - mu_c) / sd_c if sd_c > 0 else np.nan

            # Per-sample diagnostics inside the segment.  A genuine
            # connecting structure is a continuous low-level excess,
            # which a segment mean can dilute; two complementary
            # statistics are therefore reported alongside S_bridge:
            #   * the one-sided sign probability that the bridge axis
            #     sits above the per-sample control median as often as
            #     observed;
            #   * the largest per-sample z within the segment and its
            #     transect coordinate (a localised knot or the
            #     filament's approach to the companion).
            from scipy.stats import binomtest

            bseg = bridge_prof[seg]
            # Control profiles may contain all-NaN columns where a
            # rotated strip leaves the image footprint; restrict the
            # statistics to positions with at least one finite control.
            cseg = ctrl_profiles[:, seg]
            col_ok = np.isfinite(cseg).any(axis=0)
            cm = np.full(cseg.shape[1], np.nan)
            cs = np.full(cseg.shape[1], np.nan)
            cmd = np.full(cseg.shape[1], np.nan)
            if col_ok.any():
                cm[col_ok] = np.nanmean(cseg[:, col_ok], axis=0)
                cs[col_ok] = np.nanstd(cseg[:, col_ok], axis=0)
                cmd[col_ok] = np.nanmedian(cseg[:, col_ok], axis=0)
            # Adjacent rows of a transect strip share pixels and
            # resolution elements, so they are not independent
            # sign-test trials.  Thin the segment to one sample per
            # strip width (2*half_width+1 pixels along the axis),
            # which also approximates the seeing correlation length.
            stride = 2 * TRANSECT_HALF_WIDTH + 1
            keep = np.arange(len(bseg)) % stride == 0
            valid = keep & np.isfinite(bseg) & np.isfinite(cmd)
            n_seg = int(np.sum(valid))
            k_seg = int(np.sum(bseg[valid] > cmd[valid]))
            sign_p = (
                float(binomtest(k_seg, n_seg, 0.5, alternative="greater").pvalue)
                if n_seg >= 6 else np.nan
            )
            z_samp = np.where(cs > 0, (bseg - cm) / np.where(cs > 0, cs, 1), np.nan)
            if np.isfinite(z_samp).any():
                ipk = int(np.nanargmax(z_samp))
                peak_z = float(z_samp[ipk])
                peak_x = (i0 + ipk) / max(len(bridge_prof) - 1, 1)
            else:
                peak_z, peak_x = np.nan, np.nan

            # Curved-filament transect: the published claim for
            # NGC 7603 is the luminous filament itself, which is
            # curved — its archive-resolved knots sit off the
            # straight axis.  Where knots exist the transect follows
            # the resolved structure host -> knots -> companion;
            # control paths are the same polyline rotated about the
            # nucleus, preserving the path's shape and length.
            fil_stats = {}
            if knots_here:
                pts = [(gx, gy)] + [
                    tuple(float(v) for v in wcs.world_to_pixel(
                        SkyCoord(k["ra_deg"], k["dec_deg"], unit="deg")
                    ))
                    for k in knots_here
                ] + [(cx, cy)]
                fpath = self._path_transect(image, pts)
                fctrl = np.array([
                    self._path_transect(
                        image,
                        self._rotate_points(
                            pts, (gx, gy), 2 * np.pi * (k + 1) / (N_CONTROLS + 1)
                        ),
                    )
                    for k in range(N_CONTROLS)
                ])
                fi0 = int(np.floor(SEG_LO * len(fpath)))
                fi1 = max(fi0 + 1, int(np.ceil(SEG_HI * len(fpath))))
                fseg = slice(fi0, fi1)
                mu_fb = np.nanmean(fpath[fseg])
                mu_fc = np.nanmean(fctrl[:, fseg])
                sd_fc = np.nanstd(fctrl[:, fseg])
                s_fil = (mu_fb - mu_fc) / sd_fc if sd_fc > 0 else np.nan
                fb = fpath[fseg]
                fcm = np.nanmean(fctrl[:, fseg], axis=0)
                fcs = np.nanstd(fctrl[:, fseg], axis=0)
                fcmd = np.nanmedian(fctrl[:, fseg], axis=0)
                fvalid = (
                    (np.arange(len(fb)) % (2 * TRANSECT_HALF_WIDTH + 1) == 0)
                    & np.isfinite(fb) & np.isfinite(fcmd)
                )
                nf = int(np.sum(fvalid))
                kf = int(np.sum(fb[fvalid] > fcmd[fvalid]))
                sign_p_fil = (
                    float(binomtest(kf, nf, 0.5, alternative="greater").pvalue)
                    if nf >= 6 else np.nan
                )
                fz = np.where(fcs > 0, (fb - fcm) / np.where(fcs > 0, fcs, 1), np.nan)
                if np.isfinite(fz).any():
                    ifpk = int(np.nanargmax(fz))
                    fil_peak_z = float(fz[ifpk])
                    fil_peak_x = (fi0 + ifpk) / max(len(fpath) - 1, 1)
                else:
                    fil_peak_z, fil_peak_x = np.nan, np.nan
                # Embeddedness test: a knot that is part of the
                # structure sits on the filament's brightness
                # ridge, so a cut perpendicular to the path at the
                # knot peaks near the path point.  At every knot
                # vertex the perpendicular contrast
                # (SB at centre minus mean of the cut ends) is
                # compared with the same cut at the same
                # fractional position on each rotated control
                # path.  A projection coincidence would place the
                # knot anywhere across the corridor width; a
                # ridge-embedded knot sits at the maximum.
                pixscale = float(np.sqrt(abs(np.linalg.det(
                    wcs.wcs.cd)))) * 3600.0 if wcs.wcs.has_cd() else \
                    abs(wcs.wcs.cdelt[1]) * 3600.0
                knot_embed = []
                c_contr = []
                c_ext = []
                half_cut = 8.0 / pixscale   # +-8 arcsec across corridor
                for j in range(1, len(pts) - 1):
                    x0p, y0p = pts[j - 1]
                    x1p, y1p = pts[j + 1]
                    dx, dy = x1p - x0p, y1p - y0p
                    dd = np.hypot(dx, dy)
                    if dd == 0:
                        continue
                    px, py = -dy / dd, dx / dd
                    kx, ky = pts[j]
                    ncut = max(int(round(2 * half_cut)), 4)
                    cut = self._transect(
                        image, kx - px * half_cut, ky - py * half_cut,
                        kx + px * half_cut, ky + py * half_cut,
                        half_width=0, n_samples=ncut)
                    if len(cut) < 5 or not np.isfinite(cut).all():
                        continue
                    mid = len(cut) // 2
                    ends = np.concatenate([cut[:2], cut[-2:]])
                    # Central contrast recovers the compact knot
                    # itself; the discriminating embeddedness
                    # quantity is the mid-zone excess (|offset| ~
                    # 2-6 arcsec, outside the ~1.3 arcsec PSF): a
                    # knot embedded in diffuse filament light sits
                    # on an elevated corridor, a bare point source
                    # projected on blank sky does not.
                    npsf = max(1, int(round(2.0 / pixscale)))
                    nfil = max(npsf + 1, int(round(6.0 / pixscale)))
                    lo = slice(max(0, mid - nfil), mid - npsf)
                    hi = slice(mid + npsf + 1, min(len(cut), mid + nfil + 1))
                    ext = np.concatenate([cut[lo], cut[hi]])
                    knot_embed.append({
                        "knot_index": j - 1,
                        "perp_contrast": float(cut[mid] - np.mean(ends)),
                        "peak_offset_arcsec": float(
                            (int(np.argmax(cut)) - mid) * pixscale),
                        "extended_contrast": float(
                            np.mean(ext) - np.mean(ends)),
                    })
                if knot_embed:
                    for k in range(N_CONTROLS):
                        rpts = self._rotate_points(
                            pts, (gx, gy),
                            2 * np.pi * (k + 1) / (N_CONTROLS + 1))
                        for j in range(1, len(rpts) - 1):
                            x0p, y0p = rpts[j - 1]
                            x1p, y1p = rpts[j + 1]
                            dx, dy = x1p - x0p, y1p - y0p
                            dd = np.hypot(dx, dy)
                            if dd == 0:
                                continue
                            px, py = -dy / dd, dx / dd
                            kx, ky = rpts[j]
                            ccut = self._transect(
                                image, kx - px * half_cut,
                                ky - py * half_cut,
                                kx + px * half_cut, ky + py * half_cut,
                                half_width=0, n_samples=ncut)
                            if len(ccut) < 5 or not np.isfinite(ccut).all():
                                continue
                            cmid = len(ccut) // 2
                            cends = np.concatenate([ccut[:2], ccut[-2:]])
                            c_contr.append(
                                float(ccut[cmid] - np.mean(cends)))
                            clo = slice(max(0, cmid - nfil),
                                        cmid - npsf)
                            chi = slice(cmid + npsf + 1,
                                        min(len(ccut), cmid + nfil + 1))
                            cext = np.concatenate([ccut[clo], ccut[chi]])
                            c_ext.append(
                                float(np.mean(cext) - np.mean(cends)))
                    if c_contr:
                        cmu = float(np.mean(c_contr))
                        csd = float(np.std(c_contr, ddof=1))
                        emu = float(np.mean(c_ext)) if c_ext else np.nan
                        esd = (float(np.std(c_ext, ddof=1))
                               if len(c_ext) > 1 else np.nan)
                        for ke in knot_embed:
                            ke["contrast_z_vs_controls"] = (
                                (ke["perp_contrast"] - cmu) / csd
                                if csd > 0 else np.nan
                            )
                            ke["extended_z_vs_controls"] = (
                                (ke["extended_contrast"] - emu) / esd
                                if np.isfinite(esd) and esd > 0 else np.nan
                            )

                # Independent-band checks: the same path transect on
                # the g-band cutout and the DSS2 photographic plate,
                # where ingested.  A filament that persists across
                # bands and detectors cannot be a single-band
                # artifact — DSS2 is the photographic-era medium on
                # which the structure was originally claimed.
                band_stats = {}
                for band_tag in ("LS-DR10g", "DSS2_Red"):
                    b_path = self.data_raw / f"{pair_id}_{band_tag}.fits"
                    if not b_path.exists():
                        continue
                    try:
                        with fits.open(b_path) as ghd:
                            gimg = np.asarray(
                                ghd[0].data, dtype=float).squeeze()
                            gwcs = WCS(ghd[0].header)
                        if gimg.ndim != 2 or float(
                                np.mean(gimg != 0)) <= 0.5:
                            continue
                        gpts = [
                            tuple(float(v) for v in gwcs.world_to_pixel(
                                SkyCoord(ra, de, unit="deg")))
                            for ra, de in [
                                (gpos.ra.deg, gpos.dec.deg),
                                *[(k["ra_deg"], k["dec_deg"])
                                  for k in knots_here],
                                (cpos.ra.deg, cpos.dec.deg),
                            ]
                        ]
                        gpath = self._path_transect(gimg, gpts)
                        gctrl = np.array([
                            self._path_transect(
                                gimg,
                                self._rotate_points(
                                    gpts,
                                    gpts[0],
                                    2 * np.pi * (k + 1)
                                    / (N_CONTROLS + 1)),
                            )
                            for k in range(N_CONTROLS)
                        ])
                        gi0 = int(np.floor(SEG_LO * len(gpath)))
                        gi1 = max(gi0 + 1, int(
                            np.ceil(SEG_HI * len(gpath))))
                        gseg = slice(gi0, gi1)
                        gmu_b = np.nanmean(gpath[gseg])
                        gmu_c = np.nanmean(gctrl[:, gseg])
                        gsd_c = np.nanstd(gctrl[:, gseg])
                        s_b = (
                            float((gmu_b - gmu_c) / gsd_c)
                            if gsd_c > 0 else None
                        )
                        gb = gpath[gseg]
                        gcmd = np.nanmedian(
                            gctrl[:, gseg], axis=0)
                        gv = (
                            (np.arange(len(gb))
                             % (2 * TRANSECT_HALF_WIDTH + 1) == 0)
                            & np.isfinite(gb) & np.isfinite(gcmd)
                        )
                        ng = int(np.sum(gv))
                        kg = int(np.sum(gb[gv] > gcmd[gv]))
                        sp_b = (
                            float(binomtest(
                                kg, ng, 0.5,
                                alternative="greater").pvalue)
                            if ng >= 6 else None
                        )
                        band_stats[band_tag] = {
                            "S_filament": s_b, "sign_p": sp_b}
                    except Exception as e:
                        self.logger.warning(
                            f"{pair_id}: {band_tag} transect failed: {e}")
                s_fil_g = band_stats.get("LS-DR10g", {}).get("S_filament")
                sign_p_fil_g = band_stats.get("LS-DR10g", {}).get("sign_p")
                s_fil_dss = band_stats.get("DSS2_Red", {}).get("S_filament")
                sign_p_fil_dss = band_stats.get("DSS2_Red", {}).get("sign_p")

                # Blind ridge test: is the knot on a ridge that the
                # image itself defines?  For each knot a cut is taken
                # perpendicular to the STRAIGHT host->companion chord
                # (not the knot-informed filament path), the knot's
                # own PSF is masked out, and the surface-brightness
                # maximum of the remaining cut locates the
                # image-defined ridge.  The knot's offset from that
                # ridge is then compared with the ridge offsets found
                # at the same chord fraction on the rotated control
                # paths.  A knot lying on a real independent filament
                # shows a ridge within the corridor width; a knot on
                # blank sky sees its ridge land anywhere in the cut.
                blind = []
                ux, uy = (cx - gx), (cy - gy)
                ul = np.hypot(ux, uy)
                ux, uy = ux / ul, uy / ul
                ppx, ppy = -uy, ux
                half_blind = 8.0 / pixscale    # +-8 arcsec cut
                mask_w = 2.5 / pixscale        # masked knot PSF
                smooth_hw = max(1, int(round(0.5 / pixscale)))
                # pool of control-cut ridge offsets across all knots
                # and all rotated paths for the empirical comparison
                c_offs_all = []
                for j in range(1, len(pts) - 1):
                    kx, ky = pts[j]
                    nb = max(int(round(2 * half_blind)), 8)
                    bcut = self._transect(
                        image, kx - ppx * half_blind,
                        ky - ppy * half_blind,
                        kx + ppx * half_blind, ky + ppy * half_blind,
                        half_width=smooth_hw, n_samples=nb)
                    if len(bcut) < 8 or not np.isfinite(bcut).all():
                        continue
                    bmid = len(bcut) // 2
                    bmask = np.zeros(len(bcut), dtype=bool)
                    lo = max(0, bmid - int(mask_w))
                    hi = min(len(bcut), bmid + int(mask_w) + 1)
                    bmask[lo:hi] = True
                    unmasked = np.where(~bmask)[0]
                    # ridge locator: centroid of the brightest
                    # quartile of the masked cut (stable against
                    # single-pixel noise)
                    thr = np.nanpercentile(bcut[unmasked], 75)
                    top = unmasked[bcut[unmasked] >= thr]
                    ridge_idx = float(np.mean(top))
                    ridge_off = (ridge_idx - bmid) * pixscale
                    # control cuts: same chord fraction, rotated paths
                    c_offs = []
                    x_frac = ((kx - gx) * ux + (ky - gy) * uy) / ul
                    for k in range(N_CONTROLS):
                        ang = 2 * np.pi * (k + 1) / (N_CONTROLS + 1)
                        ca, sa = np.cos(ang), np.sin(ang)
                        cux = ux * ca - uy * sa
                        cuy = ux * sa + uy * ca
                        ccx = gx + cux * x_frac * ul
                        ccy = gy + cuy * x_frac * ul
                        cppx, cppy = -cuy, cux
                        cbcut = self._transect(
                            image, ccx - cppx * half_blind,
                            ccy - cppy * half_blind,
                            ccx + cppx * half_blind,
                            ccy + cppy * half_blind,
                            half_width=smooth_hw, n_samples=nb)
                        if len(cbcut) < 8 or not np.isfinite(cbcut).all():
                            continue
                        cbmid = len(cbcut) // 2
                        cbm = np.zeros(len(cbcut), dtype=bool)
                        cbm[max(0, cbmid - int(mask_w)):
                          min(len(cbcut), cbmid + int(mask_w) + 1)] = True
                        cu = np.where(~cbm)[0]
                        cthr = np.nanpercentile(cbcut[cu], 75)
                        ctop = cu[cbcut[cu] >= cthr]
                        cridge = float(np.mean(ctop))
                        off = abs((cridge - cbmid) * pixscale)
                        c_offs.append(off)
                        c_offs_all.append(off)
                    blind.append({
                        "knot_index": j - 1,
                        "chord_fraction": float(x_frac),
                        "ridge_offset_arcsec": float(ridge_off),
                        "control_median_offset_arcsec": (
                            float(np.median(c_offs)) if c_offs else np.nan
                        ),
                    })
                if c_offs_all:
                    for b in blind:
                        b["empirical_p"] = float(np.mean(
                            [o <= abs(b["ridge_offset_arcsec"])
                             for o in c_offs_all]))

                fil_stats = {
                    "S_filament": float(s_fil) if np.isfinite(s_fil) else None,
                    "sign_p_filament": (
                        float(sign_p_fil) if np.isfinite(sign_p_fil) else None
                    ),
                    "peak_z_filament": (
                        float(fil_peak_z) if np.isfinite(fil_peak_z) else None
                    ),
                    "peak_x_filament": (
                        float(fil_peak_x) if np.isfinite(fil_peak_x) else None
                    ),
                    "n_filament_samples": int(len(fpath)),
                    "mu_filament": float(mu_fb),
                    "mu_filament_ctrl": float(mu_fc),
                    "knot_embeddedness": knot_embed,
                    "n_knot_embed_controls": len(c_contr),
                    "S_filament_g": s_fil_g,
                    "sign_p_filament_g": sign_p_fil_g,
                    "S_filament_dss": s_fil_dss,
                    "sign_p_filament_dss": sign_p_fil_dss,
                    "blind_ridge": blind,
                }
                for i, val in enumerate(fpath):
                    profiles_out.append({
                        "pair_id": pair_id, "transect": "filament",
                        "sample": i, "surface_brightness": val,
                    })
                for k, fp in enumerate(fctrl):
                    for i, val in enumerate(fp):
                        profiles_out.append({
                            "pair_id": pair_id,
                            "transect": f"filament_control_{k}",
                            "sample": i, "surface_brightness": val,
                        })

            # Silhouette decrement statistic where the morphology claims it
            mu_ann = mu_ctrl_ann = s_sil = np.nan
            # Pixel scale in arcsec/pixel: LS cutouts carry a CD
            # matrix (cdelt is ignored then), so derive the scale
            # from whichever WCS form is present.
            if wcs.wcs.has_cd():
                scale = float(np.sqrt(abs(np.linalg.det(wcs.wcs.cd)))) * 3600.0
            else:
                scale = abs(wcs.wcs.cdelt[1]) * 3600.0
            if row["connection_type"] == "silhouette":
                mu_ann, mu_ctrl_ann, s_sil = self._silhouette_stat(
                    image, gx, gy, cx, cy, scale
                )

            for i, val in enumerate(bridge_prof):
                profiles_out.append({
                    "pair_id": pair_id, "transect": "bridge",
                    "sample": i, "surface_brightness": val,
                })
            for k, cp in enumerate(ctrl_profiles):
                for i, val in enumerate(cp):
                    profiles_out.append({
                        "pair_id": pair_id, "transect": f"control_{k}",
                        "sample": i, "surface_brightness": val,
                    })

            records.append({
                "pair_id": pair_id,
                "status": "ok",
                "image": str(img_path.relative_to(self.root)),
                "companion_resolved": True,
                "resolution_method": method,
                "resolved_companion": primary["name"],
                "companion_z": float(primary["z"]),
                "n_companions_matched": len(matches),
                "pa_bridge_deg": float(pa_bridge),
                "pa_sky_deg": pa_sky,
                "catalog_separation_arcsec": float(row["separation_arcsec"]),
                "measured_separation_arcsec": meas_sep,
                "bridge_claimed": bool(bridge_claimed),
                "mu_bridge": float(mu_b),
                "mu_control_mean": float(mu_c),
                "mu_control_std": float(sd_c),
                "S_bridge": float(s_stat) if np.isfinite(s_stat) else None,
                "sign_k": k_seg,
                "sign_n": n_seg,
                "sign_p": float(sign_p) if np.isfinite(sign_p) else None,
                "peak_z": float(peak_z) if np.isfinite(peak_z) else None,
                "peak_x": float(peak_x) if np.isfinite(peak_x) else None,
                "S_silhouette": (
                    float(s_sil) if np.isfinite(s_sil) else None
                ),
                "mu_annulus": (
                    float(mu_ann) if np.isfinite(mu_ann) else None
                ),
                "mu_annulus_ctrl": (
                    float(mu_ctrl_ann) if np.isfinite(mu_ctrl_ann) else None
                ),
                **fil_stats,
            })
            fil_msg = ""
            if fil_stats:
                fil_msg = (
                    f"; filament path S = {fil_stats['S_filament']:.2f} sigma"
                    if fil_stats.get("S_filament") is not None else ""
                )
            print_status(
                f"{pair_id}: S_bridge = {s_stat:.2f} sigma, "
                f"sign p = {sign_p:.3g}, peak z = {peak_z:.2f} "
                f"at x = {peak_x:.2f} ({method}, sep {meas_sep:.1f}\")"
                f"{fil_msg}",
                "TEST",
            )

        if n_images == 0:
            raise RuntimeError(
                "No imaging available for any pair. Run step_02 first; "
                "refusing to fabricate morphology results."
            )

        dfr = pd.DataFrame(records)
        dfp = pd.DataFrame(profiles_out)
        dfr.to_csv(self.data_processed / "bridge_morphology.csv", index=False)
        dfp.to_csv(self.data_processed / "bridge_transects.csv", index=False)
        dfc = pd.DataFrame(companion_rows)
        dfc.to_csv(self.data_processed / "companion_positions.csv", index=False)
        # Filament-knot positions are a required downstream input for
        # the NGC 7603 transect (steps 30-31); write the table whenever
        # any catalogue pair carries knots, and fail loudly if a
        # catalogued knot could not be resolved in any archive.
        knot_catalogued = int(
            catalog["filament_knots_z"].notna().sum()
        ) if "filament_knots_z" in catalog.columns else 0
        if knot_catalogued > 0:
            dfk = pd.DataFrame(knot_rows)
            n_expected = sum(
                len(str(v).split(";"))
                for v in catalog["filament_knots_z"].dropna()
            )
            if len(dfk) != n_expected:
                raise RuntimeError(
                    f"Filament-knot resolution incomplete: "
                    f"{len(dfk)} of {n_expected} catalogued knots "
                    "resolved; refusing to fabricate positions."
                )
            dfk.to_csv(
                self.data_processed / "filament_knot_positions.csv",
                index=False,
            )
            print_status(
                f"Saved {len(dfk)} filament-knot archive positions.",
                "SUCCESS",
            )
        print_status(
            "Saved morphology, transect and companion-position tables.",
            "SUCCESS",
        )

        ok = dfr[dfr["status"] == "ok"]
        n_unresolved = int((dfr["status"] == "companion_unresolved").sum())
        if len(ok) == 0:
            raise RuntimeError(
                "No companion resolved in any archive; refusing to "
                "report morphology on unmeasured axes."
            )
        summary = {
            "n_images": n_images,
            "n_ok": int(len(ok)),
            "n_companion_unresolved": n_unresolved,
            "segment": [SEG_LO, SEG_HI],
            "records": records,
            "s_bridge": ok.reindex(columns=[
                "pair_id", "S_bridge", "sign_p", "peak_z", "peak_x",
                "bridge_claimed", "S_filament", "sign_p_filament",
                "peak_z_filament", "peak_x_filament",
                "S_filament_g", "sign_p_filament_g",
                "S_filament_dss", "sign_p_filament_dss",
                "knot_embeddedness", "n_knot_embed_controls",
                "blind_ridge",
            ]).to_dict(orient="records"),
            "method": (
                "Companion axis resolved through NED/Simbad/Milliquas "
                "archives; bridge-axis surface brightness vs 8 control "
                "axes at the same galactocentric radius over the "
                "inter-object segment "
                f"[{SEG_LO}, {SEG_HI}]; S = (mu_b - mean(mu_c))/std(mu_c). "
                "Where archive-resolved filament knots exist (NGC 7603) "
                "a curved-path transect follows host -> knots -> "
                "companion, with controls produced by rigidly rotating "
                "the same polyline about the nucleus."
            ),
        }
        json_path = self.results / "step_20_bridge_morphology.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")

        # Figure: transect profiles for resolved fields
        try:
            import matplotlib.pyplot as plt
            apply_tep_style()
            fig, ax = plt.subplots(figsize=(9, 6))
            for pid in ok["pair_id"]:
                dsub = dfp[dfp["pair_id"] == pid]
                sub = dsub[dsub["transect"] == "bridge"]
                # NGC 7603: overlay the curved filament path
                fil = dsub[dsub["transect"] == "filament"]
                ctrl = dsub[
                    dsub["transect"].str.match(r"(control_|filament_control_)")
                ]
                x = np.linspace(0, 1, len(sub))
                ax.plot(x, sub["surface_brightness"], lw=1.6, label=f"{pid} axis")
                if len(fil):
                    ax.plot(
                        np.linspace(0, 1, len(fil)), fil["surface_brightness"],
                        lw=1.6, ls="--", label=f"{pid} filament path",
                    )
                ctrl_mean = ctrl.groupby("sample")["surface_brightness"].mean()
                ax.plot(
                    np.linspace(0, 1, len(ctrl_mean)), ctrl_mean.values,
                    ls=":", color="gray", alpha=0.6,
                    label=None if pid != ok["pair_id"].iloc[0] else "control mean",
                )
            ax.axvspan(SEG_LO, SEG_HI, color="gray", alpha=0.08)
            ax.set_xlabel("Normalised transect coordinate (galaxy → companion)")
            ax.set_ylabel("Surface brightness (image units)")
            ax.set_title("Bridge-axis transects vs control axes")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(self.figures / "step_20_bridge_morphology.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
            print_status("Saved figure: step_20_bridge_morphology.png", "SUCCESS")
        except Exception as e:
            self.logger.warning(f"figure generation failed: {e}")

        print_status("Bridge morphology analysis complete.", "SUCCESS")
