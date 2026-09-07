#!/usr/bin/env python3
"""
Step 12: Scalar Field Profile Models
====================================
Constructs the candidate analytic profiles for the local
proper-time field A_int(x) along the normalized bridge coordinate
x in [0, 1], where x = 0 is the host-galaxy nucleus (A_int = 1)
and x = 1 is the companion position (A_int = A_int(Q)).

TEP requires the local scalar configuration to interpolate
smoothly between the ambient field at the galaxy and the deep
field at the companion.  Three canonical interpolation families
are implemented as forward models to be fitted to the transect
data in steps 30-31:

  * exponential:  A(x) = 1 - (1 - A_Q) * (1 - exp(-k x)) / (1 - exp(-k))
  * yukawa:       phi(x) ~ exp(-m r)/r profile mapped through A = exp(-phi)
  * tanh:         A(x) = 1 - (1 - A_Q) * 0.5 * (1 + tanh((x - x0)/w))
                  normalised to reach A_Q at x = 1
  * nested wells: phi(x) = sum_i d_i / (1 + ((x - x_i)/w)^2),
                  A = exp(-phi) — a superposition of local temporal
                  wells, permitting a non-monotonic transition where
                  interior knots sit deeper than the endpoint

Each monotonic family is parametrised so that endpoint boundary
conditions are exact; the nested-wells family instead assigns each
compact object along the structure its own Lorentzian well.

Outputs:
    results/outputs/step_12_scalar_field_profiles.json
    data/processed/field_profiles.csv
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

N_X = 200


def profile_exponential(x, a_q, k=4.0):
    """Exponential rise of the local field toward the companion."""
    denom = 1.0 - np.exp(-k)
    return 1.0 - (1.0 - a_q) * (1.0 - np.exp(-k * x)) / denom


def profile_tanh(x, a_q, x0=0.5, w=0.15):
    """Smooth step (tanh) transition centred at x0 with width w."""
    f = 0.5 * (1.0 + np.tanh((x - x0) / w))
    f = (f - f[0]) / (f[-1] - f[0])
    return 1.0 - (1.0 - a_q) * f


def profile_yukawa(x, a_q, m=8.0):
    """Yukawa-like scalar profile phi ~ exp(-m(1-x))/(1-x+eps)."""
    eps = 0.05
    phi = np.exp(-m * (1.0 - x)) / (1.0 - x + eps)
    phi = phi / phi.max()  # normalise to 1 at companion
    delta = -np.log(a_q)
    return np.exp(-delta * phi)


def load_ngc7603_well_centers(data_processed):
    """Archive-measured nested-well centres for the NGC 7603 filament.

    Reads the filament-knot position table written by step_20 and
    returns the well centres (knot transect coordinates, ordered along
    the axis, plus the companion at x = 1).  Fails loudly when the
    measured positions are absent — the field-profile inference must
    not silently fall back to assumed knot locations."""
    path = Path(data_processed) / "filament_knot_positions.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Filament-knot positions not found: {path}. "
            "Run step_20 first; the transect uses archive-measured "
            "knot positions, not assumptions."
        )
    knots = pd.read_csv(path)
    sub = knots[knots["pair_id"] == "NGC7603-NGC7603B"].sort_values("x")
    if len(sub) == 0:
        raise RuntimeError(
            f"No NGC 7603 filament knots in {path}; run step_20 first."
        )
    return tuple(float(x) for x in sub["x"]) + (1.0,)


def profile_nested_wells(x, d1, d2, dq, w, centers):
    """Nested proper-time wells: phi(x) is a sum of Lorentzian depressions.

    Each compact object along the structure contributes its own local
    temporal well on top of the shared pair field, so the intrinsic
    conformal factor is

        A_int(x) = exp(-phi(x)),
        phi(x) = sum_i d_i / (1 + ((x - x_i)/w)^2)

    The field transition is thereby non-monotonic: interior knots can sit
    deeper in the temporal well than the companion endpoint, as the
    NGC 7603 filament redshifts require.
    """
    x = np.asarray(x, dtype=float)
    phi = np.zeros_like(x)
    for d, c in zip((d1, d2, dq), centers):
        phi = phi + d / (1.0 + ((x - c) / w) ** 2)
    return np.exp(-phi)


class Step12ScalarFieldProfiles:
    """Step 12: Build candidate A_int(x) profiles per pair."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_12",
            log_file_path=self.logs / "step_12_scalar_field_profiles.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status("Constructing scalar field profile models...", "PROCESS")

        cf_path = self.data_processed / "intrinsic_conformal_factors.csv"
        if not cf_path.exists():
            raise FileNotFoundError(
                f"Conformal factor table not found: {cf_path}. Run step_10 first."
            )
        cf = pd.read_csv(cf_path)

        x = np.linspace(0.0, 1.0, N_X)
        frames = []
        profile_summary = []
        for _, row in cf.iterrows():
            a_q = row["a_int"]
            for name, fn in (
                ("exponential", profile_exponential),
                ("tanh", profile_tanh),
                ("yukawa", profile_yukawa),
            ):
                a_x = fn(x, a_q)
                frames.append(pd.DataFrame({
                    "pair_id": row["pair_id"],
                    "profile": name,
                    "x": x,
                    "a_int_x": a_x,
                    "phi_x": -np.log(a_x),
                }))
                # slope at companion and galaxy ends
                slope_q = float(np.gradient(a_x, x)[-1])
                slope_g = float(np.gradient(a_x, x)[0])
                profile_summary.append({
                    "pair_id": row["pair_id"], "profile": name,
                    "slope_at_galaxy": slope_g, "slope_at_companion": slope_q,
                })

        df = pd.concat(frames, ignore_index=True)
        csv_path = self.data_processed / "field_profiles.csv"
        df.to_csv(csv_path, index=False)
        print_status(f"Saved profile grid: {csv_path} ({len(df)} rows)", "SUCCESS")

        dfs = pd.DataFrame(profile_summary)
        summary = {
            "n_pairs": int(cf.shape[0]),
            "profiles": ["exponential", "tanh", "yukawa"],
            "n_x": N_X,
            "boundary_conditions": "A_int(0)=1 at galaxy nucleus; A_int(1)=A_int(Q) at companion",
            "profile_slopes": profile_summary,
        }
        json_path = self.results / "step_12_scalar_field_profiles.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")
        print_status("Scalar field profiles constructed.", "SUCCESS")
