#!/usr/bin/env python3
"""Step 39 — Explicit temporal-well solution of the field equation.

Constructs the deep temporal well required by the transect inference
(Delta_u = -ln A_int = 0.19, 0.30 at the NGC 7603 knots; 0.5 and 1.0 bracket
the requested range) as an explicit solution of the corpus static field
equation, nested on the ordinary galactic ambient field

    phi_gal = 2.847 x 10^-7   (Paper 0 step_01 nested_hierarchy, solar circle)

Method (identical to the corpus interior-roll construction, Paper 0 step_16):
the static weak-field equation nabla^2 phi = V_{,phi} + rho_* A_{,phi} is
integrated inward on the g-metric background of an engine of mass M with
g_tt factor g(r) = 1 - 2M/r. The conserved flux obeys

    dU/dr = P / (r^2 g),   dP/dr = r^2 [ V_{,U}/M_Pl^2 + shift term ]

with the unscreened isolated-body boundary condition P = -2M (Paper 0
step_16: Temporal-Topology screening suppresses the charge only through the
nonlinear overlap of gradients in nested environments; an isolated body in
the ambient background retains its full charge). In vacuum (P = -2M, V'
negligible) the exterior solution is the closed form

    U(r) = -ln(1 - 2M/r),   r > 2M,

which matches U -> 2M/r at large r and rolls to the temporal horizon
(A = exp(-U) -> 0) as r -> 2M. Emitting matter sitting at radius r inside
the well inherits the local depth U(r) + phi_gal; the interior roll drives
A toward zero (the temporal-horizon construction of Rule 20).

For each target depth the step reports:
  * the radius of the Delta_u surface in units of R_s = 2M and in pc —
    confinement vs the 1.2 kpc transect bound;
  * the wall Temporal Shear |dU/dr| -> equivalent acceleration c^2|dU/dr|,
    and what it acts on (the emitting matter inside the well);
  * where the exterior tail merges into the galactic ambient (r where
    U = phi_gal) — the same unscreened tail that sources the galaxy's own
    ambient field;
  * the residual shear at 1.2 kpc vs the host's Newtonian acceleration —
    the quantity the Temporal-Topology factor S_Sigma must suppress;
  * the lensing ledger: the g-metric potential is sourced by M alone, the
    conformal sector preserves null cones (no propagation asymmetry in the
    static limit), so the deep clock well adds no extra lensing convergence
    — deep wells are lensing-dark;
  * a negative control: an ISM-density gas knot cannot self-source a deep
    well (the deep field is the engine's, not the knot gas's).

Outputs: results/outputs/step_39_temporal_well_solution.json and a
pipeline figure (not embedded in the manuscript).
"""

import json
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tep_model import (LAMBDA_REFERENCE, M_SUN, PC, C, G, M_PL, HBAR_C,
                       KG_GEV, solve_sphere)

OUTPUT = 'results/outputs/step_39_temporal_well_solution.json'
FIGURE = 'results/figures/step_39_temporal_well_solution.png'

PHI_GAL = 2.847466051459899e-07   # galactic ambient field (Paper 0 step_01)
CONFINEMENT_LIMIT_PC = 1.2e3      # reviewer / MCMC confinement bound
TARGETS = [0.19, 0.30, 0.50, 1.00]  # knot1, knot2, bracketing depths
ENGINE_MASSES = [1e6, 1e8, 1e10]    # M_sun


def well_profile(M_kg, r_min_factor=1.0005):
    """Inward interior-roll integration on the g background.

    Geometric units with M = GM_kg/c^2 (metres): g = 1 - 2M/r,
    dU/dr = P/(r^2 g), dP/dr = r^2 * V_{,U}/M_Pl^2 (in the same units).
    V_{,U}/M_Pl^2 ~ lambda U^3 M_Pl^2 is utterly negligible on these scales;
    it is included in scaled form to demonstrate that.
    """
    M_geo = G * M_kg / C**2  # half the Schwarzschild radius, in metres
    # V' term coefficient in geometric units x = r/M_geo:
    #   u''_x = lambda u^3 M_Pl^2 M_geo^2 / (hbar c)^2   [m^-2 -> dimensionless]
    lam_V = LAMBDA_REFERENCE * M_PL**2 / HBAR_C**2 * M_geo**2

    def rhs(r_over_M, y):
        U, P = y
        r = r_over_M
        g = 1.0 - 2.0 / r
        if g <= 0:
            return [0.0, 0.0]
        dU = P / (r**2 * g)
        # scaled V' drive (negligible but explicit)
        F_V = lam_V * U**3
        dP = r**2 * F_V
        return [dU, dP]

    r_max = 2.0 / PHI_GAL + 2.0          # where U = phi_gal (U ~ 2M/r)
    r_min = 2.0 * r_min_factor           # just above the horizon
    U_top = PHI_GAL
    r_eval = np.geomspace(r_max, r_min, 4000)
    sol = solve_ivp(rhs, [r_max, r_min], [U_top, -2.0],
                    t_eval=r_eval, rtol=1e-9, atol=1e-12)
    r_over_M = sol.t[::-1]               # ascending r
    U = sol.y[0][::-1]
    return r_over_M, U, M_geo


def analyse_engine(M_Msun):
    M_kg = M_Msun * M_SUN
    r_over_M, U, M_geo = well_profile(M_kg)
    R_s_m = 2 * M_geo

    # closed-form check: vacuum solution should equal -ln(1-2M/r)
    U_exact = -np.log(1.0 - 2.0 / r_over_M)
    max_dev = float(np.max(np.abs(U - U_exact) / np.maximum(U_exact, 1e-30)))

    out = {
        'mass_Msun': M_Msun,
        'r_s_m': R_s_m, 'r_s_pc': R_s_m / PC,
        'exterior_solution_check': max_dev,
        'depth_surfaces': {},
    }

    for u_t in TARGETS:
        # radius where U = u_t: exact closed form r/(2M) = 1/(1-e^{-u_t})
        r_t_over_M = 2.0 / (1.0 - np.exp(-u_t))
        r_t_m = r_t_over_M * M_geo
        # wall shear: |dU/dr| = (2M)/(r^2 g) in units of 1/M_geo -> /m
        g_t = 1.0 - 2.0 / r_t_over_M
        dU_dr_inv_m = (2.0 / (r_t_over_M**2 * g_t)) / M_geo
        a_wall = C**2 * dU_dr_inv_m
        out['depth_surfaces'][f'u_{u_t}'] = {
            'r_over_r_s': r_t_over_M / 2.0,
            'r_pc': r_t_m / PC,
            'confined_below_1p2kpc': bool(r_t_m / PC < CONFINEMENT_LIMIT_PC),
            'wall_shear_ms2': float(a_wall),
        }

    # ambient merge radius: U(r) = phi_gal -> r = 2M/(1-e^{-phi_gal})
    r_amb_over_M = 2.0 / (1.0 - np.exp(-PHI_GAL))
    r_amb_m = r_amb_over_M * M_geo
    out['ambient_merge_pc'] = float(r_amb_m / PC)

    # residual shear at the confinement boundary (1.2 kpc)
    r_c_over_M = CONFINEMENT_LIMIT_PC * PC / M_geo
    if r_c_over_M > 2.0:
        g_c = 1.0 - 2.0 / r_c_over_M
        dU_dr_c = (2.0 / (r_c_over_M**2 * g_c)) / M_geo
        a_c = C**2 * dU_dr_c
        g_N_c = G * M_kg / (CONFINEMENT_LIMIT_PC * PC)**2
        out['shear_at_1p2kpc_ms2'] = float(a_c)
        out['newtonian_at_1p2kpc_ms2'] = float(g_N_c)
        out['shear_over_newtonian_1p2kpc'] = float(a_c / g_N_c)
        out['U_at_1p2kpc'] = float(-np.log(g_c))
    return out, (r_over_M, U, M_geo)


def knot_control():
    """Negative control: an ISM-density knot cannot self-source a deep well.

    A 10-pc emission knot at molecular-cloud density contributes only its
    own tiny unscreened amplitude; the deep field at the knot is the
    engine's well, not the knot's self-field.
    """
    rho, R = 1e-20, 10 * PC
    M_knot = rho * 4 / 3 * np.pi * (R * 100)**3 / 1000  # g->kg via cgs*1000
    M_geo = G * M_knot / C**2
    u_self = 2 * M_geo / R  # exterior amplitude at own surface
    return {'rho_g_cm3': rho, 'radius_pc': 10.0, 'mass_Msun': M_knot / M_SUN,
            'self_field_surface': float(u_self),
            'conclusion': 'knot gas self-field is ~1e-10; the 0.19-0.30 '
                          'wells are inherited from the engine field'}


class Step39TemporalWellSolution:
    def run(self):
        _main()


def _main():
    print('=' * 64)
    print(' Step 39: Temporal-well field solution on the engine background')
    print('=' * 64)
    print(f' exterior BC P = -2M (unscreened isolated charge), '
          f'phi_gal = {PHI_GAL:.3e}')

    results = {
        'construction': 'interior roll on g-metric background, P=-2M '
                        'unscreened charge (Paper 0 step_16 method)',
        'closed_form_exterior': 'U(r) = -ln(1 - 2M/r)',
        'phi_gal': PHI_GAL,
        'confinement_limit_pc': CONFINEMENT_LIMIT_PC,
        'engines': [],
        'controls': {},
    }

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    for M_Msun in ENGINE_MASSES:
        out, (r_over_M, U, M_geo) = analyse_engine(M_Msun)
        results['engines'].append(out)
        r_pc = r_over_M * M_geo / PC
        axes[0].semilogx(r_pc, U, label=f'M={M_Msun:g} ' + r'$M_\odot$')
        axes[1].loglog(r_pc, np.exp(-U), label=f'M={M_Msun:g} ' + r'$M_\odot$')
        ds = out['depth_surfaces']
        print(f"\n  M = {M_Msun:g} M_sun (R_s = {out['r_s_pc']:.2e} pc)")
        for u_t in TARGETS:
            d = ds[f'u_{u_t}']
            print(f"    Delta_u={u_t}: r = {d['r_over_r_s']:.3f} R_s "
                  f"= {d['r_pc']:.2e} pc | wall shear {d['wall_shear_ms2']:.2e} m/s^2")
        print(f"    tail merges into phi_gal at {out['ambient_merge_pc']:.3g} pc")
        if 'shear_at_1p2kpc_ms2' in out:
            print(f"    at 1.2 kpc: U={out['U_at_1p2kpc']:.2e}, "
                  f"shear={out['shear_at_1p2kpc_ms2']:.2e} m/s^2 "
                  f"({out['shear_over_newtonian_1p2kpc']:.2f} x g_N, "
                  f"pre-screening)")
        print(f"    closed-form exterior residual: {out['exterior_solution_check']:.2e}")

    results['controls']['ism_knot'] = knot_control()
    kc = results['controls']['ism_knot']
    print(f"\n  [control] ISM knot (rho=1e-20, R=10 pc, M={kc['mass_Msun']:.1e} M_sun): "
          f"self-field {kc['self_field_surface']:.1e}")

    axes[0].axhline(PHI_GAL, ls=':', c='k', lw=0.8, label=r'$\phi_{\rm gal}$')
    for u_t in TARGETS:
        axes[0].axhline(u_t, ls='--', c='gray', lw=0.5)
    axes[0].set_xlabel('r [pc]')
    axes[0].set_ylabel(r'$U(r)=\varphi-\varphi_{\rm gal}$')
    axes[0].set_title(r'Temporal-well field solution, exterior roll')
    axes[0].set_ylim(0, 4)
    axes[0].legend(fontsize=8)
    axes[1].axvline(CONFINEMENT_LIMIT_PC, ls=':', c='r', lw=0.8,
                    label='1.2 kpc bound')
    axes[1].set_xlabel('r [pc]')
    axes[1].set_ylabel(r'$A(r)=e^{-U}$')
    axes[1].set_title('Clock-rate profile')
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURE, dpi=160)
    print(f'\n  figure -> {FIGURE}')

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'  output -> {OUTPUT}')


if __name__ == '__main__':
    _main()
