#!/usr/bin/env python3
"""
Step 00: Arp Discordant-Pair Catalog
====================================
Compiles the catalog of Halton Arp's discordant quasar-galaxy pairs
with published redshifts, angular separations, and the physical
connection evidence claimed for each system.

The catalog is seeded with redshift values from the published
literature (Arp 1971; Arp & Sulentic 1979; Sulentic & Arp 1987;
Stocke et al. 1991; Galianni et al. 2005; Lopez-Corredoira &
Gutierrez 2002).  Each entry carries a ``redshift_source`` flag:
entries flagged ``ned_verified`` use values consistent with NED;
entries flagged ``literature_approx`` are carried pending the
NED verification executed in step_01.

Key targets from the research:
  - NGC 4319 / Mrk 205: flagship pair with luminous bridge
  - NGC 7603 / NGC 7603B: connected by filament hosting two knots
  - NGC 3067 / 3C 232: Ca II absorption at the galaxy redshift
  - NGC 1073: three quasars projected on the disk
  - NGC 7319: X-ray quasar at z = 2.11 adjacent to the nucleus
  - NGC 1199: compact companion silhouetted against the galaxy
  - NGC 3628: X-ray quasar Wee 51 at the minor-axis filament tip
  - NGC 3516: chain of ROSAT X-ray quasars across the galaxy
  - NGC 4258: X-ray quasar pair symmetric about the anomalous arms
  - NGC 2639: paired X-ray quasars plus a minor-axis X-ray chain
  - NGC 1097: quasar field correlated with the optical jets
  - NGC 1232: discordant companion galaxy at the arm tip (Arp 41)

Outputs:
    results/outputs/step_00_arp_pair_catalog.json
    data/processed/arp_pair_catalog.csv
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


class Step00ArpPairCatalog:
    """Step 00: Compile the Arp discordant-pair catalog."""

    def __init__(self):
        self.root = PROJECT_ROOT
        self.data_processed = self.root / "data" / "processed"
        self.results = self.root / "results" / "outputs"
        self.logs = self.root / "logs"

        for d in [self.data_processed, self.results, self.logs]:
            d.mkdir(parents=True, exist_ok=True)

        self.logger = TEPLogger(
            "step_00",
            log_file_path=self.logs / "step_00_arp_pair_catalog.log",
        )
        set_step_logger(self.logger)

    def run(self):
        print_status("Compiling Arp discordant-pair catalog...", "PROCESS")

        # Discordant pairs from the published literature.
        # z_gal: host galaxy redshift; z_comp: companion/quasar redshift.
        # separation_arcsec: projected angular separation nucleus-to-companion.
        pairs = [
            {
                "pair_id": "NGC4319-Mrk205",
                "galaxy": "NGC 4319",
                "companion": "Markarian 205",
                "galaxy_type": "SBbc galaxy",
                "companion_type": "Seyfert 1 nucleus",
                "z_gal": 0.00468,
                "z_comp": 0.0708,
                "separation_arcsec": 42.0,
                "connection_type": "luminous_bridge",
                "connection_evidence": (
                    "Optical luminous bridge (Arp 1971); H I 21-cm bridge "
                    "mapped between galaxy and companion (Sulentic & Arp 1987); "
                    "HST WFPC2 imaging confirms connecting filament"
                ),
                "absorption_system": False,
                "redshift_source": "ned_verified",
                "primary_test": "bridge_transect",
                "references": "Arp 1971, ApJ 168, 39; Sulentic & Arp 1987, ApJ 319, 687; Lopez-Corredoira & Gutierrez 2004",
                "notes": "Flagship Arp pair; Mrk 205 projected inside the outer isophotes of NGC 4319",
            },
            {
                "pair_id": "NGC7603-NGC7603B",
                "galaxy": "NGC 7603",
                "companion": "NGC 7603B",
                "galaxy_type": "Seyfert 1 spiral",
                "companion_type": "compact galaxy",
                "z_gal": 0.0295,
                "z_comp": 0.0565,
                "separation_arcsec": 59.0,
                "connection_type": "luminous_filament",
                "connection_evidence": (
                    "Luminous filament connecting the two galaxies (Arp 1970); "
                    "two emission knots embedded in the filament at z=0.243 "
                    "and z=0.391 (Lopez-Corredoira & Gutierrez 2002)"
                ),
                "absorption_system": False,
                "redshift_source": "ned_verified",
                "primary_test": "filament_transect",
                "references": "Arp 1970, Astrofisika 6, 367; Lopez-Corredoira & Gutierrez 2002, A&A 390, L15",
                "notes": "Filament hosts two additional discordant knots, giving a four-point redshift transect",
                "filament_knots_z": "0.243;0.391",
            },
            {
                "pair_id": "NGC3067-3C232",
                "galaxy": "NGC 3067",
                "companion": "3C 232",
                "galaxy_type": "Sb spiral",
                "companion_type": "radio quasar",
                "z_gal": 0.0049,
                "z_comp": 0.533,
                "separation_arcsec": 114.0,
                "connection_type": "absorption_system",
                "connection_evidence": (
                    "Ca II K absorption at z=0.005 in the quasar spectrum at "
                    "the redshift of NGC 3067 (Stocke et al. 1991); 21-cm H I "
                    "absorption at the galaxy redshift (Carilli et al. 1992)"
                ),
                "absorption_system": True,
                "redshift_source": "ned_verified",
                "primary_test": "absorption_proximity",
                "references": "Stocke et al. 1991, ApJ 374, 72; Carilli, van Gorkom & Stocke 1989, Nature 338, 134",
                "notes": "Foreground halo gas of NGC 3067 absorbs the quasar; proves the quasar lies behind the galaxy halo despite a two-orders-of-magnitude redshift difference",
            },
            {
                "pair_id": "NGC1073-QSOs",
                "galaxy": "NGC 1073",
                "companion": "three disk quasars (BSO 1-3)",
                "galaxy_type": "SBc barred spiral",
                "companion_type": "quasars",
                "z_gal": 0.0070,
                "z_comp": 0.5997,
                "z_comp_list": "0.5997;1.4042;1.9452",
                "separation_arcsec": 120.0,
                "connection_type": "disk_projection",
                "connection_evidence": (
                    "Three quasars at z=0.60, 1.40, 1.94 projected on or "
                    "immediately adjacent to the galaxy disk (Arp & Sulentic "
                    "1979); excess radio emission reported in the disk"
                ),
                "absorption_system": False,
                "redshift_source": "ned_verified",
                "primary_test": "chance_alignment",
                "references": "Arp & Sulentic 1979, ApJ 229, 496; Arp 1987, Quasars, Redshifts and Controversies",
                "notes": "Quasar surface density at this brightness is ~10^-2 deg^-2; three within a single low-z disk is vanishingly improbable under chance alignment",
            },
            {
                "pair_id": "NGC7319-QSO",
                "galaxy": "NGC 7319",
                "companion": "QSO (z=2.114)",
                "galaxy_type": "SBbc Seyfert 2 (Stephan's Quintet)",
                "companion_type": "X-ray quasar",
                "z_gal": 0.0225,
                "z_comp": 2.114,
                "separation_arcsec": 8.9,
                "connection_type": "ejection_candidate",
                "connection_evidence": (
                    "Quasar at z=2.114 found 8.9 arcsec from the nucleus of "
                    "NGC 7319 (Galianni et al. 2005: 8 arcsec); strong X-ray "
                    "source (Chandra ULX; Trinchieri et al. 2003); "
                    "radio jet structure extending toward the galaxy"
                ),
                "absorption_system": False,
                "redshift_source": "ned_verified",
                "primary_test": "chance_alignment",
                "references": "Galianni et al. 2005, ApJ 620, 88; Trinchieri et al. 2003, A&A 401, 173",
                "notes": (
                    "Proposed ejection candidate; largest intrinsic redshift "
                    "in the sample. Companion is the ULX/QSO "
                    "[VV2006] J223603.7+335824; separation is the published "
                    "8-9 arcsec (Galianni et al. 2005)"
                ),
                "companion_ned_name": "LQAC 339+033 002",
                "companion_ra_deg": 339.0154,
                "companion_dec_deg": 33.9733,
            },
            {
                "pair_id": "NGC1199-companion",
                "galaxy": "NGC 1199",
                "companion": "compact companion (z=0.044)",
                "galaxy_type": "elliptical",
                "companion_type": "compact object",
                "z_gal": 0.0083,
                "z_comp": 0.044,
                "separation_arcsec": 58.0,
                "connection_type": "silhouette",
                "connection_evidence": (
                    "Compact companion at z=0.044 appears silhouetted in "
                    "absorption against the diffuse light of NGC 1199 at "
                    "z=0.0083 (Arp 1978); the higher-redshift object lies "
                    "in front of the lower-redshift galaxy"
                ),
                "absorption_system": True,
                "redshift_source": "ned_verified",
                "primary_test": "absorption_proximity",
                "references": "Arp 1978, ApJ 220, 401; Arp 1987",
                "notes": (
                    "Silhouette argument: galaxy dust absorbs the companion's "
                    "light, placing it behind the galaxy body yet the redshift "
                    "is five times larger. Companion is 2MASXi J0303366-153740 "
                    "(z=0.0444); separation is archive astrometry (NED), the "
                    "object sits on the southwest edge of the galaxy body"
                ),
                "companion_ned_name": "2MASXi J0303366-153740",
                "companion_ra_deg": 45.90269,
                "companion_dec_deg": -15.62786,
            },
            {
                "pair_id": "NGC3628-XQSO",
                "galaxy": "NGC 3628",
                "companion": "Wee 51 (minor-axis X-ray quasar)",
                "galaxy_type": "Sb edge-on (Leo Triplet)",
                "companion_type": "X-ray quasar",
                "z_gal": 0.0038,
                "z_comp": 2.15,
                "separation_arcsec": 300.0,
                "connection_type": "axis_alignment",
                "connection_evidence": (
                    "X-ray quasar Wee 51 at the tip of the minor-axis "
                    "X-ray filament emerging from the nucleus of NGC 3628 "
                    "(Arp et al. 2002); further quasars align with the "
                    "galaxy's H I plumes; ejection-axis morphology "
                    "consistent with other Arp systems"
                ),
                "absorption_system": False,
                "redshift_source": "literature_approx",
                "primary_test": "chance_alignment",
                "references": "Arp, Burbidge, Chu, Flesch, Patat & Rupprecht 2002, A&A 391, 833",
                "notes": (
                    "Companion is Wee 51 = QSO B1117+137 = [HB89] 1117+137, "
                    "the quasar at the end of the 4.1-arcmin minor-axis "
                    "X-ray filament. Arp et al. (2002) tabulate z=2.15 for "
                    "this object; modern spectroscopic archives (NED, "
                    "Milliquas) carry z=1.498-1.499. Flagged "
                    "literature_approx so step_01 archive verification "
                    "supersedes the literature value."
                ),
                "companion_ned_name": "WEE 51",
                "companion_ra_deg": 170.0494,
                "companion_dec_deg": 13.5230,
            },
            {
                "pair_id": "NGC4258-QSOs",
                "galaxy": "NGC 4258",
                "companion": "X-ray quasar pair (E and W arm sources)",
                "galaxy_type": "SABbc Seyfert 1.9",
                "companion_type": "X-ray quasars",
                "z_gal": 0.0015,
                "z_comp": 0.398,
                "z_comp_list": "0.398;0.653",
                "separation_arcsec": 576.0,
                "connection_type": "axis_alignment",
                "connection_evidence": (
                    "Two compact ROSAT X-ray sources, each ~9 arcmin from "
                    "the nucleus; the line joining them passes within ~5 deg "
                    "of the minor axis and touches the ends of the anomalous "
                    "H-alpha/X-ray arms (Pietsch et al. 1994). Spectroscopy "
                    "shows both are quasars: z=0.398 (W source) and z=0.653 "
                    "(E source) (Burbidge 1995). The accidental-pair "
                    "probability was estimated at <4e-7 (Arp 1997)"
                ),
                "absorption_system": False,
                "redshift_source": "ned_verified",
                "primary_test": "chance_alignment",
                "references": (
                    "Burbidge 1995, A&A 298, L1; Pietsch et al. 1994, "
                    "A&A 284, 386; Burbidge & Burbidge 1997, ApJ 477, L13"
                ),
                "notes": (
                    "The anomalous arms are H-alpha and X-ray features "
                    "interpreted by Arp as ejection channels; the paired "
                    "quasars sit symmetrically along the arm axis. "
                    "Companion members are resolved individually against "
                    "Milliquas/NED in steps 01 and 22"
                ),
            },
            {
                "pair_id": "NGC2639-QSOs",
                "galaxy": "NGC 2639",
                "companion": "paired X-ray quasars and minor-axis chain",
                "galaxy_type": "Sa LINER/Seyfert 2",
                "companion_type": "X-ray quasars",
                "z_gal": 0.0111,
                "z_comp": 0.3048,
                "z_comp_list": "0.3048;0.3232",
                "separation_arcsec": 600.0,
                "connection_type": "axis_alignment",
                "connection_evidence": (
                    "Two bright X-ray quasars paired nearly symmetrically "
                    "across the nucleus along the minor-axis direction, "
                    "z=0.3048 and z=0.3232 (Burbidge 1997); a line of seven "
                    "ROSAT X-ray sources runs along the northeast minor "
                    "axis, four spectroscopically confirmed as QSO/AGN at "
                    "z=0.337-2.63, with FIRST radio sources on the same "
                    "axis (Burbidge, Burbidge, Arp & Zibetti 2004)"
                ),
                "absorption_system": False,
                "redshift_source": "ned_verified",
                "primary_test": "chance_alignment",
                "references": (
                    "Burbidge 1997, ApJ 484, L99; Burbidge, Burbidge, "
                    "Arp & Zibetti 2004, ApJS 153, 159"
                ),
                "notes": (
                    "The z=0.305/0.323 pair is near-symmetric across the "
                    "nucleus; the minor-axis chain extends the ejection-axis "
                    "morphology of NGC 3628 and NGC 3516. Members are "
                    "resolved individually against the archive in steps "
                    "01 and 22"
                ),
            },
            {
                "pair_id": "NGC1097-jetQSOs",
                "galaxy": "NGC 1097",
                "companion": "quasar field correlated with optical jets",
                "galaxy_type": "SBb Seyfert 1",
                "companion_type": "quasars",
                "z_gal": 0.00424,
                "z_comp": 0.52,
                "z_comp_list": "0.52;2.47;2.61;2.63",
                "separation_arcsec": 720.0,
                "connection_type": "axis_alignment",
                "connection_evidence": (
                    "Complete quasar search of the 8.1 sq deg field found "
                    "31 quasars with surface density increasing toward the "
                    "galaxy and peaking between the two strongest northern "
                    "jets (excess factor ~20 for bright quasars; Arp, "
                    "Wolstencroft & He 1984); six quasars group within the "
                    "jet-bounded area, four X-ray detected (Wolstencroft "
                    "et al. 1983). The 2dF catalogue brings the count to "
                    "142 within 1 deg, ~38 in excess of background, with a "
                    "quasar string along the NE minor axis and paired "
                    "redshifts (z=2.47/2.61 opposite z=2.47/2.63; a second "
                    "pair at z=0.52/0.52) (Arp & Carosati 2007)"
                ),
                "absorption_system": False,
                "redshift_source": "ned_verified",
                "primary_test": "chance_alignment",
                "references": (
                    "Arp, Wolstencroft & He 1984, ApJ 285, 44; "
                    "Wolstencroft, Ku, Arp & Scarrott 1983, MNRAS 205, 67; "
                    "Arp & Carosati 2007, arXiv:0706.0143"
                ),
                "notes": (
                    "Jet-correlated quasar field; the claimed association "
                    "is the azimuthal and radial density correlation with "
                    "the optical jets rather than a single small angular "
                    "separation, so its evidential weight is structural "
                    "(as for NGC 3516). Members are resolved individually "
                    "against the archive in steps 01 and 22"
                ),
            },
            {
                "pair_id": "NGC1232-NGC1232A",
                "galaxy": "NGC 1232",
                "companion": "NGC 1232A",
                "galaxy_type": "SABc spiral",
                "companion_type": "companion spiral galaxy",
                "z_gal": 0.00535,
                "z_comp": 0.02201,
                "separation_arcsec": 264.0,
                "connection_type": "luminous_filament",
                "connection_evidence": (
                    "Companion galaxy at the end of a spiral arm (Arp 41; "
                    "the prototype 'companion on arm' morphology); "
                    "measured recession velocity exceeds the host by "
                    "+4776 km/s (Arp 1982), and at maximum "
                    "surface-brightness enhancement the companion image "
                    "joins the arm tip. The companion hosts three "
                    "ultraluminous X-ray sources"
                ),
                "absorption_system": False,
                "redshift_source": "ned_verified",
                "primary_test": "chance_alignment",
                "references": "Arp 1982, ApJ 263, 54; Arp 1966, ApJS 14, 1",
                "notes": (
                    "Discordant companion galaxy (excess-redshift class, "
                    "like NGC 7603B) rather than a quasar pair; the "
                    "companion resolves directly by name in NED"
                ),
            },
            {
                "pair_id": "NGC3516-Qchain",
                "galaxy": "NGC 3516",
                "companion": "X-ray quasar chain (5+ ROSAT QSOs)",
                "galaxy_type": "SB0 Seyfert 1.5",
                "companion_type": "X-ray quasars",
                "z_gal": 0.0088,
                "z_comp": 0.930,
                "z_comp_list": "0.328;0.690;0.930;1.399;2.100",
                "separation_arcsec": 674.5,
                "connection_type": "axis_alignment",
                "connection_evidence": (
                    "Chain of ROSAT X-ray quasars (1WGA sources) at "
                    "z=0.33, 0.69, 0.93, 1.40, 2.10 distributed around "
                    "the galaxy within ~12 arcmin (Arp et al. 2002); "
                    "all five redshifts verified against Milliquas v7.2 "
                    "in this work; a second z=2.10 quasar and two at "
                    "z=2.40 also lie in the field"
                ),
                "absorption_system": False,
                "redshift_source": "ned_verified",
                "primary_test": "chance_alignment",
                "references": (
                    "Arp, Burbidge, Chu, Flesch, Patat & Rupprecht 2002, "
                    "A&A 391, 833; Arp 2001, ApJ 549, 780"
                ),
                "notes": (
                    "X-ray-selected quasar chain; the matching of the "
                    "published chain redshifts to Milliquas v7.2 entries "
                    "is verified by the pipeline's archive resolution, "
                    "and the local confirmed-quasar surface density "
                    "around the host is measured empirically in "
                    "step_22, so the significance is evaluated against "
                    "the real local field rather than an assumed "
                    "global density"
                ),
            },
        ]

        df = pd.DataFrame(pairs)

        # Derived quantities: redshift ratio and Hubble-distance factor.
        df["one_plus_z_ratio"] = (1.0 + df["z_comp"]) / (1.0 + df["z_gal"])
        df["hubble_distance_factor"] = df["z_comp"] / df["z_gal"]

        # Save CSV
        csv_path = self.data_processed / "arp_pair_catalog.csv"
        df.to_csv(csv_path, index=False)
        print_status(f"Saved catalog: {csv_path} ({len(df)} pairs)", "SUCCESS")

        # Summary statistics
        z_ratio_range = (df["one_plus_z_ratio"].min(), df["one_plus_z_ratio"].max())
        print_status(
            f"(1+z) ratio range: {z_ratio_range[0]:.3f} - {z_ratio_range[1]:.2f}",
            "TEST",
        )
        n_absorption = int(df["absorption_system"].sum())
        print_status(f"Pairs with absorption evidence: {n_absorption}", "TEST")
        n_pending = int((df["redshift_source"] == "literature_approx").sum())
        print_status(f"Pairs pending NED verification: {n_pending}", "TEST")

        # Save JSON summary
        summary = {
            "n_pairs": len(df),
            "pairs": df.to_dict(orient="records"),
            "one_plus_z_ratio_range": [float(z_ratio_range[0]), float(z_ratio_range[1])],
            "n_absorption_systems": n_absorption,
            "n_pending_verification": n_pending,
            "primary_tests": df["primary_test"].value_counts().to_dict(),
            "data_policy": (
                "All redshifts are published literature values; NED "
                "verification is executed in step_01. No synthetic data."
            ),
        }

        json_path = self.results / "step_00_arp_pair_catalog.json"
        with open(json_path, "w") as f:
            json.dump(json_safe(summary), f, indent=2)
        print_status(f"Saved JSON: {json_path}", "SUCCESS")

        print_status("Arp pair catalog compiled.", "SUCCESS")
