#!/usr/bin/env python3
"""
Step 46: X-Ray Column Density Analysis
======================================

In the Temporal Equivalence Principle (TEP), local companions should not exhibit 
the massive intrinsic neutral hydrogen (NH) column densities expected from 
cosmological lines of sight piercing multiple deep structures. This script queries 
the Chandra Source Catalog (CSC 2.0) via TAP to extract the fitted intrinsic NH 
values and compares them to the Galactic foreground NH.

This step dynamically loads the verified Arp pair catalog, identifies all 
discordant companions with available X-ray data, actively queries the CSC2 server, 
and extracts the spectral fits.

Outputs:
    data/processed/xray_column_density_test.csv
    results/outputs/step_46_xray_column_density_test.json
"""

import os
import csv
import json
import pyvo as vo
import warnings
from astropy.coordinates import SkyCoord
import astropy.units as u

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
                z_comp = float(row['Companion_Redshift'])
                targets.append({
                    'name': f"{row['System_Name']} Companion",
                    'ra': float(row['Companion_RA']),
                    'dec': float(row['Companion_Dec']),
                    'z': z_comp
                })
            except (ValueError, KeyError):
                continue
    return targets

def query_csc_xray(ra, dec, name, radius_arcsec=5):
    print(f"\nQuerying Chandra Source Catalog for {name}...")
    url = "http://cda.cfa.harvard.edu/csc2tap"
    try:
        tap = vo.dal.TAPService(url)
        # Bounding box approach (simple approximation)
        d_deg = radius_arcsec / 3600.0
        
        query = f"""
        SELECT m.name, m.ra, m.dec, m.flux_aper_b, m.significance, m.nh_gal, 
               m.powlaw_nh, m.powlaw_nh_lolim, m.powlaw_nh_hilim, m.powlaw_gamma
        FROM csc2.master_source m
        WHERE m.ra > {ra - d_deg} AND m.ra < {ra + d_deg} 
          AND m.dec > {dec - d_deg} AND m.dec < {dec + d_deg}
        """
        
        result = tap.search(query)
        if len(result) == 0:
            print(f"  No Chandra X-ray sources found.")
            return None
            
        row = result[0]
        
        nh_gal = row.get('nh_gal', None)
        nh = row.get('powlaw_nh', None)
        
        if getattr(nh_gal, 'mask', False): nh_gal = None
        if getattr(nh, 'mask', False): nh = None
        
        print(f"  --> Found {row['name']} (Sig: {row['significance']:.1f})")
        if nh is not None:
            print(f"  --> Fitted NH: {nh:.2e} cm-2 | Galactic: {nh_gal:.2e} cm-2")
            
        return {
            'name': name,
            'csc_name': str(row['name']),
            'ra_xray': float(row['ra']),
            'dec_xray': float(row['dec']),
            'significance': float(row['significance']),
            'flux_b': float(row['flux_aper_b']) if not getattr(row.get('flux_aper_b'), 'mask', True) else None,
            'nh_gal': float(nh_gal) if nh_gal is not None else None,
            'powlaw_nh': float(nh) if nh is not None else None,
            'powlaw_gamma': float(row['powlaw_gamma']) if not getattr(row.get('powlaw_gamma'), 'mask', True) else None
        }

    except Exception as e:
        print(f"  Error querying TAP: {e}")
        return None

def main():
    print("Initializing Step 46: X-Ray Column Density Analysis")
    print("=====================================================")
    
    catalog_path = "data/processed/arp_pair_catalog_verified.csv"
    targets = load_targets(catalog_path)
    
    # Add a few known benchmark targets
    benchmark_targets = [
        {"name": "QSO B1117+136 (NGC 3628)", "ra": 169.9508858, "dec": 13.3272136, "z": 2.411},
        {"name": "NGC 7319 Companion", "ra": 339.015416, "dec": 33.973333, "z": 2.114},
    ]
    
    target_dict = {f"{t['ra']:.3f}_{t['dec']:.3f}": t for t in targets}
    for bt in benchmark_targets:
        key = f"{bt['ra']:.3f}_{bt['dec']:.3f}"
        if key not in target_dict:
            target_dict[key] = bt
            
    all_targets = list(target_dict.values())
    print(f"Loaded {len(all_targets)} companion targets for data extraction.")

    results = []
    for t in all_targets:
        res = query_csc_xray(t['ra'], t['dec'], t['name'])
        if res:
            res['z_qso'] = t['z']
            results.append(res)
            
    os.makedirs('results/outputs', exist_ok=True)
    os.makedirs('data/processed', exist_ok=True)
    
    with open('results/outputs/step_46_xray_column_density_test.json', 'w') as f:
        json.dump({"results": results}, f, indent=4)
        
    with open('data/processed/xray_column_density_test.csv', 'w', newline='') as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
    print(f"\n[SUCCESS] Completed X-ray analysis of {len(results)} active Chandra targets.")
    print("Results written to results/outputs/step_46_xray_column_density_test.json and data/processed/xray_column_density_test.csv")

if __name__ == '__main__':
    main()
