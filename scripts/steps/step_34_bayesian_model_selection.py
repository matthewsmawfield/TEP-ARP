#!/usr/bin/env python3
"""
Step 34: Bayesian Model Selection on the NGC 7603 Transect
==========================================================
Upgrades the transect inference from parameter estimation (step_31)
to Bayesian model comparison.  The "four parameters on four points"
objection is a frequentist degrees-of-freedom argument; the correct
instrument is the Bayesian evidence (marginal likelihood), which
penalises parameter volume automatically through the Occam factor.

A ladder of profile families is fitted to the same four-point
transect by nested sampling (dynesty), and the log-evidence of each
is compared against the nested-wells configuration:

  * const_null:    A_int(x) = 1 — no intrinsic field; all redshift
                   is cosmological/distance.  Zero parameters.
  * linear:        A(x) = 1 - (1 - a_q) x
  * exponential:   A(x) = 1 - (1 - a_q)(1 - e^{-kx})/(1 - e^{-k})
  * tanh:          smooth step centred at x0 with width w
  * yukawa:        phi ~ e^{-m(1-x)}/(1-x+eps), A = exp(-phi)
  * single_well:   one Lorentzian well with free centre
  * nested_2well:  Lorentzian wells at the two measured knot
                   positions only
  * nested_3well:  wells at both knots plus the companion (the TEP
                   configuration of step_31)
  * nested_3well_fwhm: the same model with an informative prior on
                   the shared width derived from the measured
                   angular compactness of the filament knots in the
                   LS-DR10 r-band imaging (this step)

The monotonic families interpolate from A = 1 at the host to a_q at
the companion; they cannot produce interior points lying deeper than
the endpoint, which is the structure the transect measures.  The
evidence quantifies how decisively the data demand that structure.

Priors: uniform on field amplitudes/depths, log-uniform on shape
scales.  A flat-w variant of nested_3well reproduces the step_31
prior for cross-checking.  All runs use a fixed seed; the step is
fully deterministic and requires no network.

Outputs:
    results/outputs/step_34_bayesian_model_selection.json
    data/processed/bayesian_model_selection.csv
    results/figures/step_34_bayesian_model_selection.png
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
from scripts.steps.step_12_scalar_field_profiles import (
    load_ngc7603_well_centers,
)

RNG_SEED = 20260924
NLIVE = 500
DLOGZ = 0.05

SEP_ARCSEC_NGC7603 = 59.0          # catalogued host-companion separation
LS_PIXEL_ARCSEC = 0.262            # LS-DR10 pixel scale
LS_PSF_FWHM_ARCSEC = 1.3           # median LS-DR10 seeing


# ------------------------------------------------------------------
# Model profiles.  Each takes (x, theta) and returns A_int(x).
# Shape parameters arrive already transformed to physical units.
# ------------------------------------------------------------------

def m_const(x, th):
    return np.ones_like(x)


def m_linear(x, th):
    (a_q,) = th
    return 1.0 - (1.0 - a_q) * x


def m_exponential(x, th):
    a_q, k = th
    return 1.0 - (1.0 - a_q) * (1.0 - np.exp(-k * x)) / (1.0 - np.exp(-k))


def m_tanh(x, th):
    a_q, x0, w = th
    f = 0.5 * (1.0 + np.tanh((x - x0) / w))
    f0 = 0.5 * (1.0 + np.tanh((0.0 - x0) / w))
    f1 = 0.5 * (1.0 + np.tanh((1.0 - x0) / w))
    return 1.0 - (1.0 - a_q) * (f - f0) / (f1 - f0)


def m_yukawa(x, th):
    a_q, m = th
    eps = 0.05
    phi = np.exp(-m * (1.0 - x)) / (1.0 - x + eps)
    phi = phi / (1.0 / eps)  # normalised so phi(x = 1) = 1
    return np.exp(np.log(a_q) * phi)


def _lorentz(x, d, c, w):
    return d / (1.0 + ((x - c) / w) ** 2)


def m_single_well(x, th):
    d, x0, w = th
    return np.exp(-_lorentz(x, d, x0, w))


def m_nested_2well(x, th, centers):
    d1, d2, w = th
    phi = _lorentz(x, d1, centers[0], w) + _lorentz(x, d2, centers[1], w)
    return np.exp(-phi)


def m_nested_3well(x, th, centers):
    d1, d2, dq, w = th
    phi = (_lorentz(x, d1, centers[0], w)
           + _lorentz(x, d2, centers[1], w)
           + _lorentz(x, dq, 1.0, w))
    return np.exp(-phi)


# Prior element kinds: ("u", lo, hi) uniform; ("l", lo, hi)
# log-uniform; ("n", mu, sig, lo, hi) truncated normal.
MODELS = [
    {
        "name": "const_null",
        "label": "A = 1 (no intrinsic field)",
        "priors": [],
        "fn": m_const,
    },
    {
        "name": "linear",
        "label": "linear monotonic",
        "priors": [("a_q", "u", 0.3, 1.0)],
        "fn": m_linear,
    },
    {
        "name": "exponential",
        "label": "exponential monotonic",
        "priors": [("a_q", "u", 0.3, 1.0), ("k", "l", 0.05, 100.0)],
        "fn": m_exponential,
    },
    {
        "name": "tanh",
        "label": "tanh step",
        "priors": [("a_q", "u", 0.3, 1.0), ("x0", "u", 0.0, 1.0),
                   ("w", "l", 0.005, 1.0)],
        "fn": m_tanh,
    },
    {
        "name": "yukawa",
        "label": "Yukawa monotonic",
        "priors": [("a_q", "u", 0.3, 1.0), ("m", "l", 0.1, 100.0)],
        "fn": m_yukawa,
    },
    {
        "name": "single_well",
        "label": "single Lorentzian well",
        "priors": [("d", "u", 0.0, 2.0), ("x0", "u", 0.0, 1.0),
                   ("w", "l", 0.002, 1.0)],
        "fn": m_single_well,
    },
    {
        "name": "nested_2well",
        "label": "two knot wells (companion unfitted)",
        "priors": [("d1", "u", 0.0, 2.0), ("d2", "u", 0.0, 2.0),
                   ("w", "l", 0.002, 1.0)],
        "fn": "nested2",
    },
    {
        "name": "nested_3well",
        "label": "three nested wells (TEP)",
        "priors": [("d1", "u", 0.0, 2.0), ("d2", "u", 0.0, 2.0),
                   ("dq", "u", 0.0, 2.0), ("w", "l", 0.002, 1.0)],
        "fn": "nested3",
    },
    {
        "name": "nested_3well_flatw",
        "label": "three nested wells, flat w (step_31 prior)",
        "priors": [("d1", "u", 0.0, 2.0), ("d2", "u", 0.0, 2.0),
                   ("dq", "u", 0.0, 2.0), ("w", "u", 0.002, 1.0)],
        "fn": "nested3",
    },
]


def measure_knot_compactness(data_processed):
    """Measure the radial HWHM of the two NGC 7603 filament knots in
    the LS-DR10 r-band cutout and express it in transect units.

    The knots are emission-line condensations embedded in the
    luminous filament; after local-background subtraction (median
    filter) each shows a compact excess within ~10 px of its
    catalogue position.  The measured HWHM is PSF-limited, so the
    result is an upper bound on the physical knot size — which is
    precisely what an informative prior on the well width requires.
    Fails loudly if the imaging or knot positions are missing.
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    from scipy.ndimage import median_filter

    img_path = (PROJECT_ROOT / "data" / "raw" / "skyview"
                / "NGC7603-NGC7603B_LS-DR10r.fits")
    knot_path = data_processed / "filament_knot_positions.csv"
    for p in (img_path, knot_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Required input not found: {p}. Run steps 02 and 20 first."
            )

    knots = pd.read_csv(knot_path)
    sub = knots[knots["pair_id"] == "NGC7603-NGC7603B"].sort_values("x")
    hdu = fits.open(img_path)[0]
    img = hdu.data.astype(float)
    wcs = WCS(hdu.header)

    rows = []
    for _, k in sub.iterrows():
        px, py = wcs.all_world2pix(float(k["ra_deg"]), float(k["dec_deg"]), 0)
        x0, y0 = int(round(float(px))), int(round(float(py)))
        cut = img[y0 - 30:y0 + 31, x0 - 30:x0 + 31].copy()
        bg = median_filter(cut, size=21)
        resid = cut - bg
        inner = resid[20:41, 20:41]
        iy, ix = np.unravel_index(np.argmax(inner), inner.shape)
        cy, cx = 20 + iy, 20 + ix
        yy, xx = np.mgrid[0:61, 0:61]
        r = np.hypot(xx - cx, yy - cy)
        peak = resid[cy, cx]
        prof = np.array([
            np.nanmean(resid[(r >= i - 0.5) & (r < i + 0.5)])
            for i in range(1, 20)
        ])
        hwhm_px = float(np.interp(0.5 * peak, prof[::-1],
                                  np.arange(1, 20)[::-1]))
        hwhm_arcsec = hwhm_px * LS_PIXEL_ARCSEC
        rows.append({
            "resolved_name": k["resolved_name"],
            "hwhm_px": hwhm_px,
            "hwhm_arcsec": hwhm_arcsec,
            "w_transect": hwhm_arcsec / SEP_ARCSEC_NGC7603,
            "psf_limited": bool(hwhm_arcsec <= LS_PSF_FWHM_ARCSEC / 2.0 * 1.15),
        })
    return rows


def build_prior_transform(priors):
    """Unit-cube -> physical transform for uniform, log-uniform and
    truncated-normal prior elements."""
    from scipy.stats import norm

    def pt(u):
        out = np.empty(len(priors))
        for i, spec in enumerate(priors):
            kind = spec[1]
            if kind == "u":
                _, _, lo, hi = spec
                out[i] = lo + u[i] * (hi - lo)
            elif kind == "l":
                _, _, lo, hi = spec
                out[i] = np.exp(np.log(lo) + u[i] * (np.log(hi) - np.log(lo)))
            elif kind == "n":
                _, _, mu, sig, lo, hi = spec
                flo, fhi = norm.cdf(lo, mu, sig), norm.cdf(hi, mu, sig)
                out[i] = norm.ppf(flo + u[i] * (fhi - flo), mu, sig)
        return out

    return pt


def weighted_quantile(values, weights, qs):
    idx = np.argsort(values)
    v, w = values[idx], weights[idx]
    cw = np.cumsum(w) - 0.5 * w
    cw /= np.sum(w)
    return np.interp(qs, cw, v)


class Step34BayesianModelSelection:
    """Step 34: dynesty evidence comparison of transect families."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.figures = self.root / "results" / "figures"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.figures, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_34",
            log_file_path=self.logs / "step_34_bayesian_model_selection.log",
        )
        set_step_logger(self.logger)

    def _run_model(self, model, x, a, s, centers, lognorm):
        """Return (logZ, logZ_err, ess, chi2_min, posterior dict)."""
        priors = model["priors"]
        fn = model.get("fn")

        def predict(theta):
            if fn == "nested2":
                return m_nested_2well(x, theta, centers)
            if fn == "nested3":
                return m_nested_3well(x, theta, centers)
            return fn(x, theta)

        def loglike(theta):
            pred = predict(theta)
            resid = (a - pred) / s
            if not np.all(np.isfinite(resid)):
                return -np.inf
            return -0.5 * float(np.sum(resid**2)) + lognorm

        ndim = len(priors)
        if ndim == 0:
            # Zero-parameter null: evidence is the likelihood itself.
            return loglike(np.empty(0)), 0.0, np.inf, \
                float(-2.0 * (loglike(np.empty(0)) - lognorm)), {}

        pt = build_prior_transform(priors)
        rng = np.random.default_rng(RNG_SEED)
        try:
            import dynesty
            try:
                sampler = dynesty.NestedSampler(
                    loglike, pt, ndim, nlive=NLIVE,
                    bound="multi", sample="rwalk", rstate=rng,
                )
            except TypeError:
                np.random.seed(RNG_SEED)
                sampler = dynesty.NestedSampler(
                    loglike, pt, ndim, nlive=NLIVE,
                    bound="multi", sample="rwalk",
                )
            sampler.run_nested(dlogz=DLOGZ, print_progress=False)
            res = sampler.results
            logz = float(res.logz[-1])
            logz_err = float(res.logzerr[-1])
            wts = res.importance_weights()
            ess = float(1.0 / np.sum((wts / wts.sum()) ** 2))
            best = res.samples[np.argmax(res.logl)]
            chi2_min = float(-2.0 * (np.max(res.logl) - lognorm))
            post = {}
            for i, spec in enumerate(priors):
                med, lo16, hi84, hi95 = weighted_quantile(
                    res.samples[:, i], wts, [0.5, 0.16, 0.84, 0.95])
                post[spec[0]] = {
                    "median": float(med), "p16": float(lo16),
                    "p84": float(hi84), "p95": float(hi95),
                }
            return logz, logz_err, ess, chi2_min, post
        except ImportError:
            raise RuntimeError(
                "dynesty is required for Bayesian model selection "
                "(step_34); install dynesty to run this step."
            )

    def run(self):
        print_status("Running Bayesian model selection (nested sampling)...",
                     "PROCESS")

        transect_path = self.data_processed / "transect_data.csv"
        if not transect_path.exists():
            raise FileNotFoundError(
                f"Required input not found: {transect_path}. Run step_30 first."
            )
        transect = pd.read_csv(transect_path)
        sub = transect[transect["pair_id"] == "NGC7603-NGC7603B"].sort_values("x")
        x = sub["x"].to_numpy()
        a = sub["a_int"].to_numpy()
        s = sub["sigma_a"].to_numpy()
        centers = load_ngc7603_well_centers(self.data_processed)

        # Gaussian normalisation, common to all models; kept so the
        # absolute logZ values are meaningful rather than relative.
        lognorm = float(-np.sum(np.log(s * np.sqrt(2.0 * np.pi))))

        # --- informative-prior measurement -------------------------
        knot_meas = measure_knot_compactness(self.data_processed)
        for r in knot_meas:
            print_status(
                f"knot {r['resolved_name']}: HWHM = "
                f"{r['hwhm_arcsec']:.2f} arcsec "
                f"(w = {r['w_transect']:.4f} transect units)", "TEST")
        w_meas = float(np.mean([r["w_transect"] for r in knot_meas]))
        # Knots are PSF-limited: the measured HWHM is an upper bound.
        # Centre the prior at the measurement with a width covering
        # genuine resolution uncertainty down to the PSF floor.
        w_sig = max(0.5 * w_meas, 0.004)
        fwhm_model = {
            "name": "nested_3well_fwhm",
            "label": "three nested wells, imaging prior on w",
            "priors": [("d1", "u", 0.0, 2.0), ("d2", "u", 0.0, 2.0),
                       ("dq", "u", 0.0, 2.0),
                       ("w", "n", w_meas, w_sig, 0.002, 1.0)],
            "fn": "nested3",
        }
        models = MODELS + [fwhm_model]

        # --- run the ladder ----------------------------------------
        rows = []
        posts = {}
        for i, model in enumerate(models):
            logz, logz_err, ess, chi2_min, post = self._run_model(
                model, x, a, s, centers, lognorm)
            rows.append({
                "model": model["name"],
                "label": model["label"],
                "n_params": len(model["priors"]),
                "logZ": logz,
                "logZ_err": logz_err,
                "ess": ess,
                "chi2_min": chi2_min,
            })
            posts[model["name"]] = post
            print_status(
                f"{model['name']:>20}: logZ = {logz:9.2f} ± {logz_err:.2f} "
                f"(k={len(model['priors'])}, chi2_min={chi2_min:.1f})",
                "TEST")

        df = pd.DataFrame(rows)
        logz_ref = float(df.loc[df["model"] == "nested_3well", "logZ"].iloc[0])
        best_mono = df[df["model"].isin(
            ["linear", "exponential", "tanh", "yukawa"])].nlargest(1, "logZ")
        best_mono_name = str(best_mono["model"].iloc[0])
        best_mono_logz = float(best_mono["logZ"].iloc[0])
        df["dlogZ_vs_nested3"] = df["logZ"] - logz_ref
        df["log10_bayes_factor_vs_nested3"] = df["dlogZ_vs_nested3"] / np.log(10.0)
        dlz = dict(zip(df["model"], df["dlogZ_vs_nested3"]))
        lbf = dict(zip(df["model"], df["log10_bayes_factor_vs_nested3"]))
        for r in rows:
            r["dlogZ_vs_nested3"] = float(dlz[r["model"]])
            r["log10_bayes_factor_vs_nested3"] = float(lbf[r["model"]])

        csv_path = self.data_processed / "bayesian_model_selection.csv"
        df.to_csv(csv_path, index=False)

        dlogz_best = logz_ref - best_mono_logz
        for _, r in df.iterrows():
            bf = -float(r["log10_bayes_factor_vs_nested3"])
            print_status(
                f"Bayes factor nested_3well vs {r['model']:>20}: "
                f"dlogZ = {-float(r['dlogZ_vs_nested3']):+9.2f} "
                f"(log10 B = {bf:+.1f})",
                "TEST")

        summary = {
            "method": (
                "dynesty static nested sampling, nlive=500, dlogz=0.05; "
                "Gaussian likelihood in A_int with propagated z errors; "
                "uniform priors on amplitudes/depths, log-uniform on "
                "shape scales (flat-w variant reproduces the step_31 "
                "prior), truncated-normal imaging prior in the "
                "nested_3well_fwhm variant. Fixed seed "
                f"{RNG_SEED}; deterministic."
            ),
            "transect_points": int(len(x)),
            "well_centers": list(centers),
            "knot_compactness_measurement": {
                "image": "data/raw/skyview/NGC7603-NGC7603B_LS-DR10r.fits",
                "pixel_scale_arcsec": LS_PIXEL_ARCSEC,
                "psf_fwhm_arcsec_assumed": LS_PSF_FWHM_ARCSEC,
                "knots": knot_meas,
                "w_prior": {
                    "kind": "truncated normal",
                    "mu": w_meas, "sigma": w_sig,
                    "lo": 0.002, "hi": 1.0,
                    "note": (
                        "knots unresolved at LS-DR10 resolution; measured "
                        "HWHM is PSF-limited and therefore an upper bound "
                        "on the physical well width"
                    ),
                },
            },
            "models": rows,
            "posteriors": posts,
            "reference_model": "nested_3well",
            "best_monotonic": {"model": best_mono_name, "logZ": best_mono_logz},
            "dlogZ_nested3_vs_best_monotonic": dlogz_best,
            "log10_bayes_factor_vs_best_monotonic": dlogz_best / np.log(10.0),
            "occam_note": (
                "The evidence integral penalises prior volume: the "
                "four-parameter nested-wells model must overcome an "
                "Occam factor that the two- and three-parameter "
                "monotonic families do not pay. The reported ΔlogZ is "
                "therefore the answer to the 'four parameters on four "
                "points' objection — a saturated chi2 is not evidence, "
                "but a large evidence margin after the parameter-volume "
                "penalty is. The nested_2well and single_well rows "
                "localise which wells the data actually require."
            ),
            "csv": str(csv_path),
        }
        json_path = self.results / "step_34_bayesian_model_selection.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")

        # --- figure -------------------------------------------------
        try:
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.filterwarnings(
                    "ignore", message=".*extended precision.*PINT.*")
                import matplotlib.pyplot as plt
            apply_tep_style()
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

            xx = np.linspace(0, 1.02, 300)
            cmap = {
                "linear": "#9aa5b1", "exponential": "#7a8ba0",
                "tanh": "#6b7f96", "yukawa": "#5c7390",
                "single_well": "#c1913f", "nested_2well": "#a87434",
                "nested_3well": "#241a33", "nested_3well_flatw": "#3d2f52",
                "nested_3well_fwhm": "#56436f", "const_null": "#c3c9d1",
            }
            for _, r in df.iterrows():
                name = r["model"]
                if name == "const_null":
                    ax1.axhline(1.0, color=cmap[name], ls=":", lw=1,
                                label="A = 1 (null)")
                    continue
                post = posts[name]
                model = next(m for m in models if m["name"] == name)
                theta = [post[p[0]]["median"] for p in model["priors"]]
                fn = model["fn"]
                if fn == "nested2":
                    pred = m_nested_2well(xx, theta, centers)
                elif fn == "nested3":
                    pred = m_nested_3well(xx, theta, centers)
                else:
                    pred = fn(xx, theta)
                lw = 2.4 if "nested_3well" in name else 1.2
                ax1.plot(xx, pred, color=cmap.get(name, "#888888"),
                         lw=lw, label=name)
            ax1.errorbar(x, a, yerr=s, fmt="o", color="#8f1d21", ms=7,
                         zorder=10, label="transect")
            ax1.set_xlabel("Normalised bridge coordinate x")
            ax1.set_ylabel(r"$A_{\rm int}(x)$")
            ax1.set_title("Median posterior-predictive profiles")
            ax1.legend(fontsize=7.5, loc="lower left")

            order = df.sort_values("dlogZ_vs_nested3")
            ax2.barh(order["model"], -order["dlogZ_vs_nested3"],
                     color=["#241a33" if m == "nested_3well"
                            else "#798691" for m in order["model"]])
            ax2.set_xlabel(r"$\Delta \ln Z$ vs nested_3well (favours nested wells $\leftarrow$)")
            ax2.set_title("Bayesian evidence (Occam-penalised)")
            fig.tight_layout()
            fig.savefig(self.figures / "step_34_bayesian_model_selection.png",
                        dpi=300, bbox_inches="tight")
            plt.close(fig)
            print_status("Saved figure: step_34_bayesian_model_selection.png",
                         "SUCCESS")
        except Exception as e:
            self.logger.warning(f"figure failed: {e}")

        print_status("Bayesian model selection complete.", "SUCCESS")
