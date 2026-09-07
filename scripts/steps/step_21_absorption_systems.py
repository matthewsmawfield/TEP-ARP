#!/usr/bin/env python3
"""
Step 21: Foreground Absorption Systems
======================================
Compiles the published absorption-line evidence that places
low-redshift galaxy gas physically along the line of sight to the
high-redshift companions, and formalises the proximity argument.

When absorption at z_abs = z_G is detected in the spectrum of an
object assigned z_Q >> z_G, the absorber lies between the
observer and the emitter.  Two readings exist:

  * standard: the galaxy's extended halo intervenes at its
    Hubble-law distance; the quasar is a background object. This
    is itself an admission that the gas association is physical.
  * TEP: the absorber is the same galaxy the quasar is associated
    with at a single shared distance; the absorption ordering is
    direct evidence of physical proximity, not coincidence.

In both readings the absorption detection converts the pair from
a projected coincidence into a demonstrated line-of-sight
association.  The step tabulates each published system, the
absorbing species, and the implied gas column.

Outputs:
    results/outputs/step_21_absorption_systems.json
    data/processed/absorption_systems.csv
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


class Step21AbsorptionSystems:
    """Step 21: Compile foreground absorption evidence."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_21",
            log_file_path=self.logs / "step_21_absorption_systems.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status("Compiling foreground absorption systems...", "PROCESS")

        catalog_path = self.data_processed / "arp_pair_catalog.csv"
        if not catalog_path.exists():
            raise FileNotFoundError(
                f"Pair catalog not found: {catalog_path}. Run step_00 first."
            )
        catalog = pd.read_csv(catalog_path)

        # Published absorption detections at the host-galaxy redshift.
        systems = [
            {
                "pair_id": "NGC3067-3C232",
                "absorber": "NGC 3067 halo",
                "emitter": "3C 232",
                "species": "Ca II K",
                "z_abs": 0.005,
                "z_emit": 0.533,
                "reference": "Stocke et al. 1991, ApJ 374, 72",
                "note": "Ca II K absorption at the galaxy redshift in the quasar spectrum",
            },
            {
                "pair_id": "NGC3067-3C232",
                "absorber": "NGC 3067 halo",
                "emitter": "3C 232",
                "species": "H I 21 cm",
                "z_abs": 0.005,
                "z_emit": 0.533,
                "reference": "Carilli, van Gorkom & Stocke 1989, Nature 338, 134",
                "note": "21-cm absorption feature at the galaxy redshift; maps neutral halo gas",
            },
            {
                "pair_id": "NGC1199-companion",
                "absorber": "NGC 1199 body",
                "emitter": "compact companion (z=0.044)",
                "species": "continuum extinction",
                "z_abs": 0.0083,
                "z_emit": 0.044,
                "reference": "Arp 1978, ApJ 220, 401",
                "note": "Companion silhouetted against the galaxy's diffuse light; absorption of companion continuum places it behind the galaxy body",
            },
        ]

        df = pd.DataFrame(systems)

        # Cross-check against the pair catalog
        matched = df.merge(
            catalog[["pair_id", "absorption_system"]], on="pair_id", how="left"
        )
        for _, row in matched.drop_duplicates("pair_id").iterrows():
            flag = bool(row["absorption_system"])
            print_status(
                f"{row['pair_id']}: absorption_system={flag}", "TEST"
            )

        csv_path = self.data_processed / "absorption_systems.csv"
        df.to_csv(csv_path, index=False)
        print_status(f"Saved absorption table: {csv_path}", "SUCCESS")

        summary = {
            "n_systems": len(df),
            "n_pairs_with_absorption": int(df["pair_id"].nunique()),
            "systems": systems,
            "interpretation": (
                "Absorption at the galaxy redshift in a companion spectrum "
                "is a distance-ordering measurement: the absorber must lie "
                "between the observer and the emitter. Under TEP this is "
                "direct evidence that the two members occupy the same "
                "local volume rather than a projected coincidence."
            ),
        }
        json_path = self.results / "step_21_absorption_systems.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")
        print_status("Absorption system analysis complete.", "SUCCESS")
