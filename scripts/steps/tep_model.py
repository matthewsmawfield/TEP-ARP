"""Canonical TEP weak-field equations and observation operators.

Units: hbar=c=1 in field equations; SI at public geometry interfaces.
phi has mass dimension one, varphi=phi/M_Pl, lambda is dimensionless.
No measured correlation length is used by the radial solver.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad, solve_bvp
from scipy.optimize import brentq
from scipy.linalg import eigh

M_PL = 2.435e18
HBAR_C = 0.1973269804e-15
KG_GEV = 5.60958885e26
G = 6.67430e-11
C = 299792458.0
AU = 149597870700.0
PC = 3.085677581491367e16
M_SUN = 1.98847e30
R_SUN = 6.957e8
M_EARTH = 5.972e24
R_EARTH = 6.371e6
BETA = -1.0
# Density conversion derived from the same mass and length units.
G_CM3_GEV4 = 1000 * KG_GEV * HBAR_C**3
LAMBDA_REFERENCE = 7.526e-71
# Unified master potential (cross-scale closure; the step_33 'uni' gate
# integrates the same function):
#   V(varphi) = (lam/4) varphi^4 exp(-(varphi/VARPHI_S)^4)
#             + V0_PL exp(-(VARPHI_S/varphi)^4)
# The plateau trigger is non-perturbative at varphi = 0 -- every Taylor
# coefficient vanishes -- so the weak-field sector is exactly the quartic
# to all orders (the term underflows to 0.0 at every screening density).
# It activates only at Planck-scale field values varphi ~ VARPHI_S,
# reached by the temporal pileup inside deep wells.
UNIFIED_POTENTIAL = True
VARPHI_S = 10.0        # plateau knee, in units of M_Pl
V0_PL = 0.3            # plateau depth, in units of M_Pl^4 (fiducial)
RESULTS = Path(__file__).resolve().parents[2] / 'results'


def save(name, data):
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def source_parameters(mass_kg, radius_m, lam):
    mass_gev = mass_kg * KG_GEV
    radius_gev = radius_m / HBAR_C
    psi_uns = mass_gev / (4*np.pi*M_PL**2*radius_gev)
    mu = lam * (mass_gev/(4*np.pi*M_PL))**2
    density = 3*mass_kg/(4*np.pi*radius_m**3)/1000
    return psi_uns, mu, density


def _quartic_drive(varphi, lam):
    """dV/dvarphi of the quartic sector in M_Pl^4 units. With
    UNIFIED_POTENTIAL the Gaussian knee factor multiplies in; at
    weak-field values exp(-(varphi/VARPHI_S)^4) rounds to exactly 1."""
    if not UNIFIED_POTENTIAL:
        return lam * varphi**3
    x4 = (varphi / VARPHI_S)**4
    return lam * varphi**3 * np.exp(-x4) * (1.0 - x4)


def _plateau_drive(varphi):
    """d/dvarphi [V0_PL exp(-(VARPHI_S/varphi)^4)] in M_Pl^4 units.
    Essential singularity at the origin: the term underflows to 0.0 for
    |varphi| < VARPHI_S/5.2 (exp(-745) is the double-precision floor), so
    the guard is exact, not an approximation."""
    if abs(varphi) <= VARPHI_S / 5.2:
        return 0.0
    return (4.0 * V0_PL * VARPHI_S**4 / varphi**5
            * np.exp(-(VARPHI_S / varphi)**4))


def _scalar_drive(varphi, lam):
    return _quartic_drive(varphi, lam) + _plateau_drive(varphi)


def equilibrium_varphi(rho_g_cm3, lam):
    if rho_g_cm3 < 0 or lam <= 0:
        raise ValueError('Positive quartic coupling and nonnegative density required')
    if rho_g_cm3 == 0:
        return 0.0
    # lambda M_Pl^4 varphi^3 = rho_* exp(-varphi).
    log_rhs = np.log(rho_g_cm3*G_CM3_GEV4/(lam*M_PL**4))
    log_x = brentq(lambda u: 3*u + np.exp(u) - log_rhs, -1000, 100)
    v = float(np.exp(log_x))
    if UNIFIED_POTENTIAL:
        # The quartic-only root is certified only where the master drive
        # is exactly quartic: both the Gaussian cutoff factor and the
        # essential-singularity plateau must be inert at the root.
        # Fire the full solve whenever either correction is resolved
        # (|corr-1| > 1e-9 <-> varphi >~ 0.05*VARPHI_S/10... i.e. the
        # cutoff region), not merely once the plateau drive is nonzero.
        x4 = (v / VARPHI_S)**4
        correction = np.exp(-x4) * (1.0 - x4) if x4 < 700.0 else 0.0
        if abs(correction - 1.0) > 1e-9 or _plateau_drive(v) != 0.0:
            # Deep-well regime: solve the full master-potential
            # equilibrium directly. If the bounded master drive cannot
            # balance the source, no equilibrium exists -- fail loudly.
            rhs = rho_g_cm3*G_CM3_GEV4/M_PL**4
            f = lambda t: _scalar_drive(t, lam) - rhs*np.exp(-t)
            lo, hi = 1e-30, 10.0*VARPHI_S
            # The plateau tail V_,u ~ 4 V0 u_s^4/u^5 eventually beats
            # rhs e^{-u}, so a root always exists at large u; extend
            # the bracket until the sign changes.
            while f(hi) < 0.0 and hi < 1e4*VARPHI_S:
                hi *= 10.0
            if f(lo) * f(hi) > 0:
                raise RuntimeError(
                    'No unified-potential equilibrium at '
                    f'rho={rho_g_cm3} g/cm^3: master drive cannot '
                    'balance the matter source')
            v = float(brentq(f, lo, hi))
    return v


def mass_squared(varphi, rho_g_cm3, lam):
    """V_eff'' in GeV^2, including the conformal matter contribution."""
    v = np.asarray(varphi, dtype=float)
    if UNIFIED_POTENTIAL:
        x4 = (v / VARPHI_S)**4
        e4 = np.exp(-x4)
        # factored so the weak-field value multiplies by exactly 1.0
        m2 = (3*lam*(M_PL*v)**2 * e4
              * ((1.0 - x4) - (4.0/3.0)*v**4*(2.0 - x4)/VARPHI_S**4))
        plateau_mask = np.abs(v) > VARPHI_S / 5.2
        if np.any(plateau_mask):
            vp = np.where(plateau_mask, v, 1.0)
            g = np.exp(-(VARPHI_S / vp)**4)
            m2 = m2 + np.where(
                plateau_mask,
                V0_PL * M_PL**2 * g
                * (4.0*VARPHI_S**4/vp**6)
                * (4.0*VARPHI_S**4/vp**4 - 5.0),
                0.0)
    else:
        m2 = 3*lam*(M_PL*v)**2
    return m2 + rho_g_cm3*G_CM3_GEV4/M_PL**2*np.exp(-v)


def compton_m(varphi, rho_g_cm3, lam):
    m2 = mass_squared(varphi, rho_g_cm3, lam)
    return float(HBAR_C/np.sqrt(m2)) if m2 > 0 else None


def potential_increment(background, perturbation, lam):
    """Exact V'(background+perturbation)-V'(background), dimensionful fields.
    Under UNIFIED_POTENTIAL the arguments are the dimensionless
    varphi = phi/M_Pl (VARPHI_S is set in those units)."""
    if not UNIFIED_POTENTIAL:
        return lam*perturbation*(3*background**2+3*background*perturbation+perturbation**2)
    return (_scalar_drive(background + perturbation, lam)
            - _scalar_drive(background, lam))


def solve_sphere(mass_kg, radius_m, lam=LAMBDA_REFERENCE, rho_bg=1e-24,
                 x_max=1e5, tol=2e-5, surface_width=.01, density_shape=None,
                 phi_env=0.0, guess=None):
    """Solve a finite-mass overdensity on the quartic equilibrium background.

    u=delta(varphi)/psi_uns; the homogeneous ambient equation is subtracted
    exactly, including its density source. Interior density is smoothed and
    normalized to the specified excess mass. A linear exterior Robin boundary
    replaces the artificial zero-field boundary. Both profile and residuals
    are returned; nonconvergence raises rather than being classified as a pass.

    phi_env: an additional constant background varphi (e.g. the galactic
    ambient field at the source's location) added to the local density
    equilibrium before expanding the quartic potential and matter source.
    The star's own perturbation u still decays to zero at x_max (i.e. the
    total field decays to bg+phi_env, not to zero) -- this is the nested
    "phi_total = phi_env + delta_phi" construction used elsewhere in the
    corpus, ported onto this solver's more robust stiff-quartic handling
    (interior-equilibrium initial guess, scale rescaling, Robin exterior
    BC) so it remains reliable at the same high-mu0 couplings solve_sphere
    is otherwise used for. phi_env=0 (default) exactly reproduces the
    unembedded/isolated calculation.
    """
    if lam <= 0 or rho_bg <= 0 or x_max <= 2:
        raise ValueError('Require lambda>0, rho_bg>0, x_max>2')
    psi_uns, mu, rho_mean = source_parameters(mass_kg, radius_m, lam)
    bg = equilibrium_varphi(rho_bg, lam) + phi_env
    e = bg/psi_uns
    fbg = rho_bg/rho_mean
    shape = density_shape or (lambda x: .5*(1-np.tanh((x-1)/surface_width)))
    norm = 3*quad(lambda x: x*x*shape(x), 0, 2,
                  points=[1], epsabs=1e-11)[0]
    scale = min(1.0, (3/mu)**(1/3))
    mR = radius_m/HBAR_C*np.sqrt(mass_squared(bg,rho_bg,lam))
    def rhs(x,y):
        u = scale*y[0]
        theta = shape(x)/norm
        # Subtract homogeneous V' and rho A' before division by scale.
        potential = mu*u*(3*e*e+3*e*u+u*u)
        matter = -3*np.exp(-bg)*(theta*np.exp(-psi_uns*u)
                     + fbg*np.expm1(-psi_uns*u))
        return np.vstack((y[1], (potential+matter)/scale - 2*y[1]/x))
    xmin=1e-5
    def bc(a,b):
        return np.array([a[1], b[1]+(1/x_max+mR)*b[0]])
    if guess is not None and hasattr(guess,'x'):
        x=guess.x
        y0=guess.y
    else:
        # Interior equilibrium gives a stable amplitude for stiff quartics.
        center=(equilibrium_varphi(rho_mean+rho_bg,lam)-bg)/psi_uns
        center=np.clip(center,-1.5,1.5)
        x=np.unique(np.r_[np.geomspace(xmin,x_max,850),np.linspace(.8,1.2,200)])
        initial=np.where(x<1, center*(1-x*x/3), center*(2/3)/x)
        deriv=np.where(x<1,-2*center*x/3,-center*(2/3)/x**2)
        y0=np.vstack([initial/scale,deriv/scale])
    sol=solve_bvp(rhs,bc,x,y0,tol=tol,max_nodes=200000)
    if not sol.success:
        raise RuntimeError(sol.message)
    def profile(x_eval):
        y=sol.sol(np.asarray(x_eval,dtype=float))
        return scale*y
    return dict(profile=profile, solution=sol, psi_uns=psi_uns, mu=mu,
                background_varphi=bg, background_compton_m=compton_m(bg,rho_bg,lam),
                rho_mean_g_cm3=rho_mean, x_max=x_max,
                max_rms_residual=float(np.max(sol.rms_residuals)))


def diagnostics(solved,x):
    u,du=solved['profile'](x)
    amplitude=float(x*u)
    charge=float(-x*x*du)
    # PPN gamma for a screened source (Burrage & Sakstein 2018, Living Rev.
    # Relativity 21:1, Eq. 3.31): gamma-1 ~ -4*beta^2*(1-M(r_s)/M), i.e.
    # LINEAR in the thin-shell/unscreened mass fraction, which is exactly
    # the source-charge screening factor S = source_charge_ratio used here
    # (0=screened, 1=unscreened). Screening suppresses the exterior field
    # profile once; it does not separately re-suppress the fixed conformal
    # coupling beta that converts that profile into a metric perturbation.
    # A squared-S formula double-counts the screening (it was used in an
    # earlier version of this file and has been corrected here). The Pade
    # form below reduces to this linear result for S<<1 and to the standard
    # unscreened DEF result gamma-1=-2*alpha_DEF^2/(1+alpha_DEF^2) at S=1
    # (alpha_DEF^2=2*beta^2). The Cassini bound |gamma-1|<2.3e-5 translates
    # to S < 2.3e-5/(4*beta^2) ~ 5.75e-6 for beta=-1.
    S = charge
    ppn_gamma_minus_one = float(-4*BETA**2*S / (1 + 2*BETA**2*S))
    return dict(radius_over_source=float(x),source_charge_ratio=charge,
                scalar_potential_ratio=amplitude,
                force_ratio_unscreened_test=2*BETA**2*charge,
                ppn_gamma_minus_one=ppn_gamma_minus_one,
                metric_gamma_minus_one=ppn_gamma_minus_one)


def weighted_residual(signal,design,weights):
    """Actual linear least-squares absorption, in observation space."""
    signal=np.asarray(signal,float);design=np.asarray(design,float)
    weights=np.asarray(weights,float)
    if np.any(weights<=0) or signal.ndim != 1:
        raise ValueError('Positive diagonal weights and a vector signal required')
    sw=np.sqrt(weights)
    parameters=np.linalg.lstsq(sw[:,None]*design,sw*signal,rcond=None)[0]
    return signal-design@parameters,parameters


def graph_resolvent_covariance(laplacian,noise,omega=0.0,damping=1.0,mass2=0.0):
    """C(omega)=G N G^dagger for explicitly supplied operator/noise.

    Matrices use one consistent dimensionless normalization; physical units
    reside in the field/source conversion supplied by an observation model.
    """
    op=np.asarray(laplacian,float)+mass2*np.eye(len(laplacian))
    noise=np.asarray(noise,complex)
    if np.min(np.linalg.eigvalsh(noise)) < -1e-10:
        raise ValueError('Source spectral density must be positive semidefinite')
    g=np.linalg.inv(op-(omega**2+1j*damping*omega)*np.eye(len(op)))
    return g@noise@g.conj().T
