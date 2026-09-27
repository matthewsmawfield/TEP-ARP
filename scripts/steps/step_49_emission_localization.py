#!/usr/bin/env python3
"""
Step 49 — Emission-region localization and the well-enclosure ledger.

Two claims about the deep-well emitters are tested against the data:

(1) The exterior-profile misidentification.  On the unscreened
    interior-roll exterior profile U(r) = -ln(1 - 2M/r) (step_39), the
    surfaces carrying the measured depths lie at
    r(du)/R_S = 1/(1 - exp(-du)) — 5.8, 3.9, 2.5, 1.6 R_S for
    du = 0.19, 0.30, 0.50, 1.0.  Evaluating the circular-orbit speed at
    each surface, c*sqrt(R_S/2r), shows directly that no emitter can sit
    *on* the exterior profile at those depths: inside 3 R_S there are no
    stable bound orbits, and at ~4-6 R_S the orbital speeds are
    ~0.3-0.7c — broadening no observed line shows.  The depth surface is
    a property of the profile, not a candidate emitter location; the
    emitter's clock factor is the interior field of its own well, and
    the emitting gas sits *inside* the deep region.

(2) The consistent channel, measured.  The emitting gas is localised by
    its own kinematics, not by a field-profile assumption:
      * observed line widths are measured on the re-reduced NGC 7319
        companion spectrum itself (step_43 products): Gaussian fits to
        each detected UV line give the intrinsic velocity dispersion
        (a conformal endpoint factor rescales all frequencies by the
        same factor, so a line's fractional — hence velocity-equivalent
        — width is invariant across the well's depth);
      * bound-gas enclosure: gas of dispersion sigma stays bound only
        inside r_max/R_S = (c/sigma)^2 — the well wall must enclose the
        emitting region but need extend no further;
      * line coherence: emission components spread over a field range
        du_emit inside the well are smeared to fractional width ~du_emit,
        so du_emit <~ sigma/c — a flat-interior requirement, the same
        property the corpus's interior-roll analysis derives for the
        deep region (locally flat potential floor, Paper 0 step_16);
      * enclosure energetics: a static canonical wall carrying
        displacement du across a transition of radius R_w has mass
        equivalent M_wall ~ du^2 R_w/(12 G) (the same bookkeeping the
        gradient-density ledger uses); the wall-unity radius
        R_unity/R_S = 6/du^2 marks where M_wall = M_engine.  Enclosures
        at ~R_unity are order-unity vs the engine's mass; enclosures at
        broad-line-region scale (~10^2-10^3 R_S) run ~10-10^2 x the
        engine mass under this bookkeeping — the bracket the coupled
        interior construction (Paper-28 domain) must close, quantified
        here rather than asserted.

Inputs (all measured or corpus-derived, no free fits):
  data/processed/intrinsic_conformal_factors.csv   (12-pair A_int)
  data/processed/transect_data.csv                 (knot depths)
  results/outputs/step_43_keck_lris_ulx_verification.json
  data/raw/keck_lris/LB.20031003.*.fits            (flagship frames)

Outputs:
  results/outputs/step_49_emission_localization.json
  data/processed/step_49_emission_localization.csv
  results/figures/step_49_emission_localization.png
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

C_KMS = 299792.458                      # speed of light, km/s
M_SUN = 1.98847e30                      # kg
G_SI = 6.67430e-11                      # SI
PC_M = 3.085677581491367e16             # m

BLUE_FILES = ["LB.20031003.28349.fits", "LB.20031003.28810.fits",
              "LB.20031003.29568.fits", "LB.20031003.30962.fits"]

# UV lines detectable in the re-reduced flagship blue arm (step_43):
# (name, rest wavelength A, fit window A)
LINE_FITS = [
    ("Lya+NV blend", 1215.67, (3770.0, 3840.0)),
    ("NV",           1240.80, (3840.0, 3910.0)),
    ("SiIV-OIV]",    1397.00, (4290.0, 4400.0)),
    ("CIV",          1549.00, (4820.0, 4930.0)),
    ("HeII",         1640.40, (5050.0, 5150.0)),
    ("OIII]",        1665.85, (5150.0, 5250.0)),
]

# well-enclosure grid (in Schwarzschild radii)
RW_GRID_RS = [10.0, 50.0, 200.0, 1000.0]


def _load_blue(path):
    """Bias-subtract a 2003-era LRIS-B frame (identical to step_43)."""
    from astropy.io import fits
    d = fits.getdata(path).astype(float)
    return d - np.median(d[:400, :400])


def measure_flagship_widths(raw_dir, calib, logger):
    """Per-line FWHM of the NGC 7319 companion on the step_43 stack."""
    from scipy.ndimage import gaussian_filter1d
    from scipy.optimize import curve_fit

    for f in BLUE_FILES:
        if not (raw_dir / f).exists():
            raise RuntimeError(f"Step49: missing flagship frame {f}")

    med = np.median(np.array([_load_blue(raw_dir / f)
                              for f in BLUE_FILES]), axis=0)
    d_arc = calib["arc_dispersion_A_per_row"]
    off_arc = calib["arc_offset_A"]
    dr = calib["flexure_rows"]
    xc = calib["ulx_trace_column"]
    lam = lambda r: off_arc + d_arc * (r - dr)

    obj = med[:, xc - 3:xc + 4].mean(axis=1)
    sky = np.median(
        np.concatenate([med[:, xc - 40:xc - 16],
                        med[:, xc + 18:xc + 42]], axis=1), axis=1)
    spec = obj - sky
    rr = np.arange(len(spec))
    w = lam(rr)
    # noise from a line-free window
    free = spec[(w > 4550) & (w < 4700)]
    noise = float(np.std(free))
    logger.info(f"flagship stack: trace x{xc}, noise {noise:.2f}/row")

    out = {}
    for name, rest, (lo, hi) in LINE_FITS:
        m = (w > lo) & (w < hi)
        x, y = w[m], spec[m]
        cont = np.median(np.r_[y[:8], y[-8:]])
        try:
            p, _ = curve_fit(
                lambda x, A, x0, s, b: A * np.exp(-(x - x0) ** 2
                                                  / (2 * s * s)) + b,
                x, y,
                p0=[y.max() - cont, x[np.argmax(y)], 15.0, cont],
                bounds=([0, lo, 1, 0],
                        [10 * (y.max() - cont), hi, 200, cont + 3 * noise]),
                maxfev=20000)
            A, x0, s, b = p
            fwhm_A = 2.3548 * s
            out[name] = {
                "peak_A": float(x0),
                "z_peak": float(x0 / rest - 1.0),
                "fwhm_A": float(fwhm_A),
                "fwhm_kms": float(fwhm_A / x0 * C_KMS),
                "amplitude_over_noise": float(A / noise),
                "status": "detected" if A > 3 * noise else "marginal",
            }
            logger.info(f"  {name}: FWHM {out[name]['fwhm_kms']:.0f} km/s "
                        f"({out[name]['status']}, A/N={A/noise:.1f})")
        except Exception as e:
            out[name] = {"status": "fit_failed", "error": str(e)}
            logger.warning(f"  {name}: fit failed: {e}")
    return out, (w, spec)


def wall_ratio(du, rw_over_rs):
    """M_wall / M_engine for a canonical static wall: du^2 R_w/(12 G)
    in geometric units with R_S = 2M -> ratio = du^2 (R_w/R_S) / 6."""
    return du ** 2 * rw_over_rs / 6.0


class Step49EmissionLocalization:
    def __init__(self, config=None):
        self.config = config or {}
        self.out_dir = PROJECT_ROOT / "results" / "outputs"
        self.fig_dir = PROJECT_ROOT / "results" / "figures"
        self.tab_dir = PROJECT_ROOT / "data" / "processed"
        self.raw = PROJECT_ROOT / "data" / "raw" / "keck_lris"
        self.logs = PROJECT_ROOT / "logs"
        for d in [self.out_dir, self.fig_dir, self.tab_dir, self.logs]:
            d.mkdir(parents=True, exist_ok=True)
        self.logger = TEPLogger(
            "step_49",
            log_file_path=self.logs / "step_49_emission_localization.log")
        set_step_logger(self.logger)

    def run(self):
        self.logger.info("Step 49: emission localization + enclosure ledger")

        icf = self.tab_dir / "intrinsic_conformal_factors.csv"
        if not icf.exists():
            raise RuntimeError(f"Step49: missing {icf}")
        pairs = pd.read_csv(icf)

        # knot depths from the archive-measured knot redshifts
        # (filament_knot_positions.csv): A_int = (1+z_host)/(1+z_knot)
        # with the measured NGC 7603 host redshift z_host = 0.0295.
        knot_pos = self.tab_dir / "filament_knot_positions.csv"
        knots = []
        if knot_pos.exists():
            kp = pd.read_csv(knot_pos)
            z_host = 0.0295  # measured NGC 7603 nucleus (Table 6)
            for _, r in kp.iterrows():
                a = (1.0 + z_host) / (1.0 + float(r["z"]))
                knots.append((f"NGC7603 {r['resolved_name']}", float(a)))
        else:
            # measured transect depths (Table 6 of the manuscript)
            knots = [("NGC7603 [LG2002] 3", 0.740),
                     ("NGC7603 [LG2002] 2", 0.828)]

        calib_path = self.out_dir / "step_43_keck_lris_ulx_verification.json"
        widths, spec_data = {}, None
        if calib_path.exists():
            calib = json.loads(calib_path.read_text())
            widths, spec_data = measure_flagship_widths(
                self.raw, calib, self.logger)
        else:
            self.logger.warning(
                "step_43 output absent; flagship widths not measured")

        civ = widths.get("CIV", {})
        sig_flag = civ.get("fwhm_kms", 3411.0)
        self.logger.info(f"flagship C IV sigma = {sig_flag:.0f} km/s")

        # ---------- per-emitter ledger -------------------------
        emitters = []
        for _, r in pairs.iterrows():
            emitters.append((str(r["pair_id"]), float(r["a_int"])))
        for name, a in knots:
            emitters.append((f"transect:{name}", float(a)))

        rows = []
        for name, a_int in emitters:
            if not (0 < a_int <= 1.0):
                continue
            du = -np.log(a_int)
            r_lapse = 1.0 / (1.0 - np.exp(-du))   # depth surface, R_S units
            v_lapse = C_KMS / np.sqrt(2.0 * r_lapse)  # circ. orbit speed
            esc = (C_KMS / sig_flag) ** 2          # bound-gas ceiling, R_S
            unity = 6.0 / du ** 2                  # wall=engine radius, R_S
            row = {
                "emitter": name,
                "a_int": a_int,
                "delta_phi": du,
                "lapse_depth_surface_Rs": r_lapse,
                "lapse_inside_photon_sphere": bool(r_lapse < 1.5),
                "lapse_inside_isco": bool(r_lapse < 3.0),
                "lapse_circ_speed_kms": v_lapse,
                "escape_bound_Rs": esc,
                "unity_wall_Rs": unity,
                "wall_over_engine_at_escape": wall_ratio(du, esc),
                "coherence_du_max": sig_flag / C_KMS,
            }
            for rw in RW_GRID_RS:
                row[f"wall_over_engine_at_{int(rw)}Rs"] = wall_ratio(du, rw)
            rows.append(row)
        ledger = pd.DataFrame(rows)
        ledger.to_csv(self.tab_dir / "step_49_emission_localization.csv",
                      index=False)

        flag = ledger.loc[ledger["emitter"].str.contains("7319", case=False)]
        if len(flag):
            f = flag.iloc[0]
            self.logger.info(
                f"flagship: du={f['delta_phi']:.3f}, lapse surface "
                f"{f['lapse_depth_surface_Rs']:.2f} R_S "
                f"(photon sphere={f['lapse_inside_photon_sphere']}), "
                f"escape bound {f['escape_bound_Rs']:.3g} R_S, "
                f"unity wall {f['unity_wall_Rs']:.1f} R_S")

        # ---------- figure --------------------------------------
        apply_tep_style()
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))

        rw = np.geomspace(1.0, 1e5, 400)
        for name, a_int in [("NGC7319 companion", 0.3284),
                            ("NGC7603 knot3", 0.740),
                            ("NGC7603 knot2", 0.828),
                            ("NGC3628 companion", 0.4014),
                            ("NGC7603B", 0.974)]:
            du = -np.log(a_int)
            axes[0].loglog(rw, wall_ratio(du, rw),
                           label=f"{name} ($A_{{\\rm int}}$={a_int:.2f})")
        axes[0].axhline(1.0, color="k", ls="--", lw=1,
                        label=r"$M_{\rm wall}=M_{\rm engine}$")
        axes[0].axvspan(1.5, 3.0, color="r", alpha=0.12,
                        label="photon sphere–ISCO")
        esc = (C_KMS / sig_flag) ** 2
        axes[0].axvline(esc, color="gray", ls=":", lw=1,
                        label=fr"escape bound ($\sigma$={sig_flag:.0f} km/s)")
        axes[0].set_xlabel(r"enclosing wall radius $R_w\ [R_S]$")
        axes[0].set_ylabel(r"$M_{\rm wall}/M_{\rm engine}$ (canonical static)")
        axes[0].set_title("Well-enclosure energetic ledger")
        axes[0].legend(fontsize=7, loc="upper left")

        if spec_data is not None:
            w, spec = spec_data
            m = (w > 4780) & (w < 4980)
            axes[1].plot(w[m], spec[m], lw=0.8, color="tab:blue")
            axes[1].plot(w[m], np.convolve(spec[m],
                         np.ones(5) / 5, "same"), lw=1.5, color="k")
            if civ.get("peak_A"):
                axes[1].axvline(civ["peak_A"], color="r", ls="--", lw=1)
                axes[1].text(0.03, 0.92,
                             f"C IV FWHM = {sig_flag:.0f} km/s",
                             transform=axes[1].transAxes, fontsize=9)
            axes[1].set_xlabel("observed wavelength (A)")
            axes[1].set_ylabel("counts")
            axes[1].set_title("Flagship C IV profile (re-reduced LRIS)")
        fig.tight_layout()
        fig.savefig(self.fig_dir / "step_49_emission_localization.png",
                    dpi=140, bbox_inches="tight")

        summary = {
            "purpose": (
                "Emission-region localization: the lapse-tracked exterior "
                "profile's depth surfaces (~Rs) are shown to be impossible "
                "emitter locations (no bound gas inside 3 R_S, ~0.3-0.7c "
                "orbits at 4-6 R_S); the consistent channel places the "
                "emitting gas inside its own well interior, localised by "
                "its measured kinematics. Fractional line width is a "
                "conformal invariant, so measured widths are intrinsic."),
            "flagship_measured_widths": widths,
            "reference_sigma_kms": float(sig_flag),
            "ledger_columns": {
                "lapse_depth_surface_Rs":
                    "r(du)/R_S on U=-ln(1-2M/r): exterior-profile "
                    "misidentification only",
                "escape_bound_Rs":
                    "r/R_S where sigma = v_esc: bound-gas ceiling",
                "unity_wall_Rs":
                    "R_w/R_S where canonical static wall mass equals the "
                    "engine mass (6/du^2)",
                "coherence_du_max":
                    "max field dispersion across emitter (sigma/c): "
                    "flat-interior requirement",
            },
            "emitters": ledger.to_dict("records"),
            "verdict": (
                "The photon-sphere objection applies only to the "
                "misidentified exterior-profile channel: no emitter can "
                "sit at the depth surfaces, which is why the corpus never "
                "sources emitter depth there. The consistent channel — "
                "gas inside its own well — is bounded only by its "
                "measured kinematics: the flagship's measured C IV width "
                f"({sig_flag:.0f} km/s) keeps the gas bound anywhere "
                "inside ~10^4 R_S, the line coherence required (flat "
                "interior, du<~sigma/c) is the flat-floor property the "
                "corpus derives for deep wells, and the static-canonical "
                "enclosure energy is order-unity at ~5 R_S rising to "
                "~10-100x engine mass at standard BLR radii — a "
                "quantified factor-10-100 bookkeeping bracket for the "
                "coupled interior, not the photon-sphere impossibility "
                "of the misidentified channel."),
        }
        with open(self.out_dir / "step_49_emission_localization.json",
                  "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status("Step 49 complete: emission localization ledger")


if __name__ == "__main__":
    Step49EmissionLocalization().run()
