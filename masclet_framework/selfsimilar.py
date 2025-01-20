"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

selfsimilar module
Provides functions to compute self-similar normalization constants for 
    masses-radii-densities, temperatures, pressures, entropies, etc.

Created by David Vallés
"""

import numpy as np
from masclet_framework import units, cosmo_tools

def M_from_R(R, delta=200, norm='m', z=0., units_r='comoving',
             h=0.678, omega_m=0.31):
    """
    Computes mass (in Msun) from radius (in Mpc) for a given overdensity
        delta (either normalised to the critical density ['c'] or to the 
        background density ['m']).

    Args:
        R: radius (in Mpc)
        delta: overdensity (typically, 200, 500, 2500, etc.)
        norm: normalization (whether the overdensity is given in units of 
              the critical density ['c'] or the background density ['m'])
        z: redshift (float)
        units_r: units of the radius (either 'comoving' or 'physical')
        h: dimensionless Hubble constant (float)
        omega_m: matter density parameter, at z=0 (float)
    
    Returns:
        mass (in Msun) corresponding to the given radius and overdensity
    """
    # Check input data is correct
    if norm not in ['c', 'm']:
        raise ValueError("norm must be either 'c' or 'm'")
    if units_r not in ['comoving', 'physical']:
        raise ValueError("units_r must be either 'comoving' or 'physical'")
    if delta < 0:
        raise ValueError("delta must be positive")
    if R < 0:
        raise ValueError("R must be positive")

    omega_lambda = 1 - omega_m # We assume a flat universe with no radiation!

    if norm == 'c':
        mass = 1.06877e14 * (delta/200) * (h/0.678)**2 * cosmo_tools.E(z,omega_m, omega_lambda)**2. * (R/1.)**3
        if units_r == 'comoving':
            mass /= (1+z)**3
    elif norm == 'm':
        mass = 3.31318e13 * (delta/200) * (h/0.678)**2 * (R/1.)**3
        if units_r == 'physical':
            mass *= (1+z)**3

    return mass

def R_from_M(M, delta=200, norm='m', z=0., units_r='comoving',
             h=0.678, omega_m=0.31):
    """
    Computes radius (in Mpc) from mass (in Msun) for a given overdensity
        delta (either normalised to the critical density ['c'] or to the 
        background density ['m']).

    Args:
        M: mass (in Msun)
        delta: overdensity (typically, 200, 500, 2500, etc.)
        norm: normalization (whether the overdensity is given in units of 
              the critical density ['c'] or the background density ['m'])
        z: redshift (float)
        units_r: units of the radius (either 'comoving' or 'physical')
        h: dimensionless Hubble constant (float)
        omega_m: matter density parameter, at z=0 (float)

    Returns:
        radius (in Mpc) corresponding to the given mass and overdensity
    """
    # Check input data is correct
    if norm not in ['c', 'm']:
        raise ValueError("norm must be either 'c' or 'm'")
    if units_r not in ['comoving', 'physical']:
        raise ValueError("units_r must be either 'comoving' or 'physical'")
    if delta < 0:
        raise ValueError("delta must be positive")
    if M < 0:
        raise ValueError("M must be positive")

    omega_lambda = 1 - omega_m # We assume a flat universe with no radiation!

    if norm == 'c':
        R = 0.97807 * (delta/200)**(-1/3) * (h/0.678)**(-2/3) * cosmo_tools.E(z,omega_m, omega_lambda)**(-2/3) * (M/1e14)**(1/3)
        if units_r == 'comoving':
            R *= (1+z)
    elif norm == 'm':
        R = 1.44517 * (delta/200)**(-1/3) * (h/0.678)**(-2/3) * (M/1e14)**(1/3)
        if units_r == 'physical':
            R /= (1+z)

    return R

def T(M, delta=200, norm='m', z=0., h=0.678, omega_m=0.31, mu=0.6, units_out='keV'):
    """ 
    Computes temperature (times kB; in keV; or in K) from mass (in Msun) for a 
        given overdensity delta (either normalised to the critical density 
        ['c'] or to the background density ['m']).

    Args:
        M: mass (in Msun)
        delta: overdensity (typically, 200, 500, 2500, etc.)
        norm: normalization (whether the overdensity is given in units of 
              the critical density ['c'] or the background density ['m'])
        z: redshift (float)
        h: dimensionless Hubble constant (float)
        omega_m: matter density parameter, at z=0 (float)
        mu: mean molecular weight (float)
        units_out: units of the temperature (either 'keV' or 'K')

    Returns:
        temperature (times kB; in keV) corresponding to the given mass and 
        overdensity
    """
    # Check input data is correct
    if norm not in ['c', 'm']:
        raise ValueError("norm must be either 'c' or 'm'")
    if delta < 0:
        raise ValueError("delta must be positive")
    if M < 0:
        raise ValueError("M must be positive")

    omega_lambda = 1 - omega_m # We assume a flat universe with no radiation!

    if norm == 'c':
        T = 1.37725 * (delta/200)**(1/3) * (h/0.678)**(2/3) * cosmo_tools.E(z,omega_m, omega_lambda)**(2/3) *  (mu/0.6) * (M/1e14)**(2/3)
    elif norm == 'm':
        T = 0.93211 * (delta/200)**(1/3) * (h/0.678)**(2/3) * (1+z) * (mu/0.6) * (M/1e14)**(2/3)

    if units_out == 'keV':
        return T 
    elif units_out == 'K':
        return T * (units.keV_to_J / units.kB_isu)
    else:
        raise ValueError("units_out must be either 'keV' or 'K'")

def n(delta=200, norm='m', z=0., h=0.678, omega_m=0.31, mu=0.6, fb=0.155):
    """ 
    Computes baryon number density (in cm^-3) for a given overdensity delta 
        (either normalised to the critical density ['c'] or to the background 
        density ['m']).

    Args:
        delta: overdensity (typically, 200, 500, 2500, etc.)
        norm: normalization (whether the overdensity is given in units of 
              the critical density ['c'] or the background density ['m'])
        z: redshift (float)
        h: dimensionless Hubble constant (float)
        omega_m: matter density parameter, at z=0 (float)
        mu: mean molecular weight (float)
        fb: baryonic mass fraction (float)

    Returns:
        gas number density (in cm^-3) corresponding to the given mass 
        and overdensity
    """
    # Check input data is correct
    if norm not in ['c', 'm']:
        raise ValueError("norm must be either 'c' or 'm'")
    if delta < 0:
        raise ValueError("delta must be positive")

    omega_lambda = 1 - omega_m # We assume a flat universe with no radiation!

    if norm == 'c':
        ne = 2.66720e-4 * (delta/200) * (fb/0.155) * (h/0.678)**2 * (mu/0.6)**(-1) * \
            cosmo_tools.E(z,omega_m, omega_lambda)**2
    elif norm == 'm':
        ne = 8.26818e-5 * (delta/200) * (fb/0.155) * (omega_m/0.31) * (h/0.678)**2 * (mu/0.6)**(-1) * \
            (1+z)**3

    return ne

def ne(delta=200, norm='m', z=0., h=0.678, omega_m=0.31, mu=0.6, fb=0.155):
    """ 
    Computes electron number density (in cm^-3) for a given overdensity delta 
        (either normalised to the critical density ['c'] or to the background 
        density ['m']).

    Args:
        delta: overdensity (typically, 200, 500, 2500, etc.)
        norm: normalization (whether the overdensity is given in units of 
              the critical density ['c'] or the background density ['m'])
        z: redshift (float)
        h: dimensionless Hubble constant (float)
        omega_m: matter density parameter, at z=0 (float)
        mu: mean molecular weight (float)
        fb: baryonic mass fraction (float)

    Returns:
        electron number density (in cm^-3) corresponding to the given mass 
        and overdensity
    """
    # Check input data is correct
    if norm not in ['c', 'm']:
        raise ValueError("norm must be either 'c' or 'm'")
    if delta < 0:
        raise ValueError("delta must be positive")

    omega_lambda = 1 - omega_m # We assume a flat universe with no radiation!

    if norm == 'c':
        ne = 1.38694e-4 * (delta/200) * (fb/0.155) * (h/0.678)**2 * (mu/0.6)**(-1) * \
            ((2+mu)/2.6) * cosmo_tools.E(z,omega_m, omega_lambda)**2
    elif norm == 'm':
        ne = 4.29945e-5 * (delta/200) * (fb/0.155) * (omega_m/0.31) * (h/0.678)**2 * (mu/0.6)**(-1) * \
            ((2+mu)/2.6) * (1+z)**3

    return ne

def K(M, delta=200, norm='m', z=0., h=0.678, omega_m=0.31, mu=0.6, fb=0.155):
    """ 
    Computes gas entropy (in keV cm^2) from mass (in Msun) for a given 
        overdensity delta (either normalised to the critical density ['c'] 
        or to the background density ['m']).

    Args:
        M: mass (in Msun)
        delta: overdensity (typically, 200, 500, 2500, etc.)
        norm: normalization (whether the overdensity is given in units of 
              the critical density ['c'] or the background density ['m'])
        z: redshift (float)
        h: dimensionless Hubble constant (float)
        omega_m: matter density parameter, at z=0 (float)
        mu: mean molecular weight (float)
        fb: baryonic mass fraction (float)
    
    Returns:
        gas entropy (in keV cm^2) corresponding to the given mass and overdensity
    """
    # Check input data is correct
    if norm not in ['c', 'm']:
        raise ValueError("norm must be either 'c' or 'm'")
    if delta < 0:
        raise ValueError("delta must be positive")
    if M < 0:
        raise ValueError("M must be positive")

    omega_lambda = 1 - omega_m # We assume a flat universe with no radiation!

    if norm == 'c':
        K = 332.386 * (delta/200)**(-1/3) * (fb/0.155)**(-2/3) * (h/0.678)**(-2/3) * (mu/0.6)**(5/3) * \
            cosmo_tools.E(z,omega_m, omega_lambda)**(-2/3) * (M/1e14)**(2/3) 
    elif norm == 'm':
        K = 491.127 * (delta/200)**(-1/3) * (fb/0.155)**(-2/3) * (omega_m/0.31)**(-2/3) * (h/0.678)**(-2/3) * \
            (mu/0.6)**(5/3) * (1+z)**(-1) * (M/1e14)**(2/3)

    return K

def Ke(M, delta=200, norm='m', z=0., h=0.678, omega_m=0.31, mu=0.6, fb=0.155):
    """ 
    Computes electron entropy (in keV cm^2) from mass (in Msun) for a given 
        overdensity delta (either normalised to the critical density ['c'] 
        or to the background density ['m']).

    Args:
        M: mass (in Msun)
        delta: overdensity (typically, 200, 500, 2500, etc.)
        norm: normalization (whether the overdensity is given in units of 
              the critical density ['c'] or the background density ['m'])
        z: redshift (float)
        h: dimensionless Hubble constant (float)
        omega_m: matter density parameter, at z=0 (float)
        mu: mean molecular weight (float)
        fb: baryonic mass fraction (float)
    
    Returns:
        electron entropy (in keV cm^2) corresponding to the given mass and overdensity
    """
    # Check input data is correct
    if norm not in ['c', 'm']:
        raise ValueError("norm must be either 'c' or 'm'")
    if delta < 0:
        raise ValueError("delta must be positive")
    if M < 0:
        raise ValueError("M must be positive")

    omega_lambda = 1 - omega_m # We assume a flat universe with no radiation!

    if norm == 'c':
        K = 514.013 * (delta/200)**(-1/3) * (fb/0.155)**(-2/3) * (h/0.678)**(-2/3) * (mu/0.6)**(5/3) * \
            ((2+mu)/2.6)**(-2/3) * cosmo_tools.E(z,omega_m, omega_lambda)**(-2/3) * (M/1e14)**(2/3) 
    elif norm == 'm':
        K = 759.495 * (delta/200)**(-1/3) * (fb/0.155)**(-2/3) * (omega_m/0.31)**(-2/3) * (h/0.678)**(-2/3) * \
            (mu/0.6)**(5/3) * ((2+mu)/2.6)**(-2/3) * (1+z)**(-1) * (M/1e14)**(2/3)

    return K

def P(M, delta=200, norm='m', z=0., h=0.678, omega_m=0.31, mu=0.6, fb=0.155):
    """ 
    Computes gas pressure (in keV cm^-3) from mass (in Msun) for a given
        overdensity delta (either normalised to the critical density ['c'] or 
        to the background density ['m']).

    Args:
        M: mass (in Msun)
        delta: overdensity (typically, 200, 500, 2500, etc.)
        norm: normalization (whether the overdensity is given in units of 
              the critical density ['c'] or the background density ['m'])
        z: redshift (float)
        h: dimensionless Hubble constant (float)
        omega_m: matter density parameter, at z=0 (float)
        mu: mean molecular weight (float)
        fb: baryonic mass fraction (float)

    Returns:
        gas pressure (in keV cm^-3) corresponding to the given mass and overdensity
    """
    # Check input data is correct
    if norm not in ['c', 'm']:
        raise ValueError("norm must be either 'c' or 'm'")
    if delta < 0:
        raise ValueError("delta must be positive")
    if M < 0:
        raise ValueError("M must be positive")

    omega_lambda = 1 - omega_m # We assume a flat universe with no radiation!

    if norm == 'c':
        P = 3.67340e-4 * (delta/200)**(4/3) * (fb/0.155) * (h/0.678)**(8/3) * \
            cosmo_tools.E(z,omega_m, omega_lambda)**(8/3) * (M/1e14)**(2/3)
    elif norm == 'm':
        P = 7.70685e-5 * (delta/200)**(4/3) * (fb/0.155) * (omega_m/0.31)* (h/0.678)**(8/3) * \
            (1+z)**4 * (M/1e14)**(2/3)

    return P

def Pe(M, delta=200, norm='m', z=0., h=0.678, omega_m=0.31, mu=0.6, fb=0.155):
    """ 
    Computes electron pressure (in keV cm^-3) from mass (in Msun) for a given
        overdensity delta (either normalised to the critical density ['c'] or 
        to the background density ['m']).

    Args:
        M: mass (in Msun)
        delta: overdensity (typically, 200, 500, 2500, etc.)
        norm: normalization (whether the overdensity is given in units of 
              the critical density ['c'] or the background density ['m'])
        z: redshift (float)
        h: dimensionless Hubble constant (float)
        omega_m: matter density parameter, at z=0 (float)
        mu: mean molecular weight (float)
        fb: baryonic mass fraction (float)

    Returns:
        electron pressure (in keV cm^-3) corresponding to the given mass and overdensity
    """
    # Check input data is correct
    if norm not in ['c', 'm']:
        raise ValueError("norm must be either 'c' or 'm'")
    if delta < 0:
        raise ValueError("delta must be positive")
    if M < 0:
        raise ValueError("M must be positive")

    omega_lambda = 1 - omega_m # We assume a flat universe with no radiation!

    if norm == 'c':
        P = 1.91017e-4 * (delta/200)**(4/3) * (fb/0.155) * (h/0.678)**(8/3) * \
            ((2+mu)/2.6) * cosmo_tools.E(z,omega_m, omega_lambda)**(8/3) * (M/1e14)**(2/3)
    elif norm == 'm':
        P = 4.00756e-5 * (delta/200)**(4/3) * (fb/0.155) * (omega_m/0.31) * (h/0.678)**(8/3) * \
            ((2+mu)/2.6) * (1+z)**4 * (M/1e14)**(2/3)

    return P
