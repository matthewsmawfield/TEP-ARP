#!/usr/bin/env python3
"""
Step 45: Equilibrium Path Accumulation Test
===========================================

In standard cosmology, light travelling a path corresponding to Delta z > 1
must accumulate intergalactic dust (reddening the continuum) and pierce multiple
foreground galaxy halos (imprinting discrete Mg II and C IV absorption systems).
In the Temporal Equivalence Principle (TEP) framework, an eternal static universe
implies that a true 10 Gpc path will be heavily reddened and populated by foreground
systems. If a quasar is local, it completely bypasses this path and should lack both.

This step dynamically loads the entire verified Arp pair catalog, identifies all 
discordant companions at high redshift, actively queries the SDSS/BOSS servers to 
download their raw spectral data, and measures:
  1. The rest-frame optical spectral index (alpha_nu) to test for dust reddening.
  2. The presence of intervening Mg II doublet (2796, 2803) absorbers in the continuum.

Outputs:
    data/processed/equilibrium_path_test.csv
    results/outputs/step_45_equilibrium_path_test.json
"""

import os
import csv
import json
import numpy as np
from astroquery.sdss import SDSS
from astropy.coordinates import SkyCoord
import astropy.units as u
import warnings

warnings.filterwarnings('ignore')

def load_targets(catalog_path):
    targets = []
    if not os.path.exists(catalog_path):
        print(f"Catalog {catalog_path} not found.")
        return targets
        
    with open(catalog_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                # We only want high-z targets where Mg II test is viable
                z_comp = float(row['Companion_Redshift'])
                if z_comp > 0.5:
                    targets.append({
                        'name': f"{row['System_Name']} Companion",
                        'ra': float(row['Companion_RA']),
                        'dec': float(row['Companion_Dec']),
                        'z': z_comp
                    })
            except (ValueError, KeyError):
                continue
    return targets

def analyze_quasar(ra, dec, z_qso, name):
    print(f"\nAnalyzing {name} (z={z_qso}) at {ra}, {dec}...")
    try:
        c = SkyCoord(ra=ra, dec=dec, unit='deg', frame='icrs')
        spectra = SDSS.query_region(c, spectro=True, radius=10*u.arcsec)
    except Exception as e:
        print(f"  SDSS Query failed: {e}")
        return None

    if not spectra:
        print(f"  No SDSS spectra found for {name}.")
        return None

    try:
        sp_list = SDSS.get_spectra(matches=spectra)
        sp = sp_list[0]
        data = sp[1].data

        wave = 10**data['loglam']
        flux = data['flux']
        ivar = data['ivar']
        and_mask = data['and_mask']

        good = (ivar > 0) & (and_mask == 0)
        wave = wave[good]
        flux = flux[good]
        ivar = ivar[good]

        # Search for Mg II
        mg2_1, mg2_2 = 2796.35, 2803.53
        found_systems = 0
        systems = []
        
        for z_abs in np.arange(0.4, z_qso - 0.05, 0.005):
            obs_1 = mg2_1 * (1 + z_abs)
            obs_2 = mg2_2 * (1 + z_abs)
            
            if obs_1 < wave[0] or obs_2 > wave[-1]:
                continue
                
            mask_cont = (wave > obs_1 - 50) & (wave < obs_2 + 50)
            if np.sum(mask_cont) < 10: continue
                
            local_cont = np.median(flux[mask_cont])
            if local_cont <= 0: continue
                
            mask_1 = np.abs(wave - obs_1) < 2.0
            mask_2 = np.abs(wave - obs_2) < 2.0
            
            if np.sum(mask_1) == 0 or np.sum(mask_2) == 0: continue
                
            flux_1 = np.mean(flux[mask_1])
            flux_2 = np.mean(flux[mask_2])
            
            if flux_1 < local_cont * 0.7 and flux_2 < local_cont * 0.7:
                snr_1 = (local_cont - flux_1) * np.sqrt(np.mean(ivar[mask_1]))
                snr_2 = (local_cont - flux_2) * np.sqrt(np.mean(ivar[mask_2]))
                if snr_1 > 3 and snr_2 > 3:
                    systems.append(float(z_abs))
                    found_systems += 1

        # Reddening calculation
        wave_rest = wave / (1 + z_qso)
        red_mask = ((wave_rest > 1275) & (wave_rest < 1350)) | ((wave_rest > 1425) & (wave_rest < 1450))
        wave_red = wave_rest[red_mask]
        flux_red = flux[red_mask]

        alpha_nu = None
        if len(wave_red) > 10:
            p = np.polyfit(np.log10(wave_red), np.log10(flux_red), 1)
            alpha_lambda = p[0]
            alpha_nu = float(alpha_lambda + 2.0)

        print(f"  --> Found {found_systems} Mg II systems. alpha_nu = {alpha_nu if alpha_nu is not None else 'N/A'}")
        
        return {
            'name': name,
            'z_qso': z_qso,
            'mg2_systems': found_systems,
            'mg2_redshifts': systems,
            'alpha_nu': alpha_nu,
            'is_reddened': bool(alpha_nu > 0.5) if alpha_nu is not None else None,
            'is_local_consistent': bool(found_systems == 0 and alpha_nu < 0) if alpha_nu is not None else None
        }
    except Exception as e:
        print(f"  Error processing spectrum for {name}: {e}")
        return None

def main():
    print("Initializing Step 45: Equilibrium Path Accumulation Test")
    print("==========================================================")
    
    catalog_path = "data/processed/arp_pair_catalog_verified.csv"
    targets = load_targets(catalog_path)
    
    # Add a few known benchmark targets as fallbacks if catalog parsing differs
    benchmark_targets = [
        {"name": "QSO B1117+136 (NGC 3628)", "ra": 169.9508858, "dec": 13.3272136, "z": 2.411},
        {"name": "NGC 7319 Companion", "ra": 339.015416, "dec": 33.973333, "z": 2.114},
    ]
    
    # Merge targets preventing duplicates by RA/Dec
    target_dict = {f"{t['ra']:.3f}_{t['dec']:.3f}": t for t in targets}
    for bt in benchmark_targets:
        key = f"{bt['ra']:.3f}_{bt['dec']:.3f}"
        if key not in target_dict:
            target_dict[key] = bt
            
    all_targets = list(target_dict.values())
    print(f"Loaded {len(all_targets)} high-z companion targets for data extraction.")

    results = []
    for t in all_targets:
        res = analyze_quasar(t['ra'], t['dec'], t['z'], t['name'])
        if res:
            results.append(res)
            
    os.makedirs('results/outputs', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    # Save JSON
    with open('results/outputs/step_45_equilibrium_path_test.json', 'w') as f:
        json.dump({"results": results}, f, indent=4)
        
    # Save CSV
    with open('data/processed/equilibrium_path_test.csv', 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
    print(f"\n[SUCCESS] Completed spectral extraction and analysis of {len(results)} active SDSS targets.")
    print("Results written to results/outputs/step_45_equilibrium_path_test.json and data/processed/equilibrium_path_test.csv")

if __name__ == '__main__':
    main()
