"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

cosmo_tools module
Provides useful tools to compute time from redshift, evolution of the critical density, etc.

Created by David Vallés
"""

#  Last update on 21/3/20 17:51

import json
import os
import numpy as np
from scipy import integrate
from masclet_framework import units


def write_cosmo_parameters(h, omega_m, omega_lambda, omega_b, filename='cosmo_parameters.json', path=''):
    """
    Creates a JSON file containing the cosmological parameters for a simulation

    Args:
        h: dimensionless Hubble constant
        omega_m: matter density in terms of the critical density, at z=0
        omega_lambda: dark energy density in terms of the critical density, at z=0
        omega_b: baryionic matter density in terms of the critical density, at z=0
        filename: name of the MASCLET parameters file to be saved (str)
        path: path of the file (typically, the codename of the simulation) (str)

    Returns: nothing. A file is created in the specified path

    """
    parameters = {'h': h, 'omega_m': omega_m, 'omega_lambda': omega_lambda,
                  'omega_b': omega_b}

    with open(path + filename, 'w') as json_file:
        json.dump(parameters, json_file)


def read_cosmo_parameters(filename='cosmo_parameters.json', path=''):
    """
    Returns dictionary containing the cosmological parameters of the simulation, that have been previously written
    with the write_cosmo_parameters() function in this same module.

    Args:
        filename: name of the cosmological parameters file (str)
        path: path of the file (typically, the codename of the simulation) (str)

    Returns:
        dictionary containing the parameters (and their names), namely:
        h: dimensionless Hubble constant
        omega_m: matter density parameter, at z=0
        omega_lambda: dark energy density parameter, at z=0
        omega_b: baryionic matter density parameter, at z=0

    """
    filepath = os.path.join(path, filename)
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data

def E(z,omega_m, omega_lambda, omega_r=0):
    """
    Computes the evolution function of the Hubble parameter (H(z) = H_0 E(z)).

    Args:
        z: redshift where to evaluate the function
        omega_m: matter density parameter, at z=0
        omega_lambda: dark energy density parameter, at z=0
        omega_r: radiation density parameter, at z=0. Don't specify if negligible

    Returns:
        Value of the E(z) function for this cosmology
    """
    omega_k = 1 - omega_m - omega_lambda - omega_r
    a = 1/(1+z)
    E = np.sqrt(omega_lambda + omega_k/a**2 + omega_m/a**3 + omega_r/a**4)
    return E



def LCDM_time(z1,z2,omega_m,omega_lambda,h):
    """
    Computes the time between z1 and z2, in years, using a quadrature method

    Args:
        z1: initial redshift
        z2: final redshift
        omega_m: matter density parameter, at z=0
        omega_lambda: dark energy density parameter, at z=0
        h: dimensionless Hubble constant

    Returns:
        Time, in years, from z1 to z2
    """
    if z1<z2:
        raise ValueError('Initial redshift must be larger than final redshift! Negative time will be got.')

    tH = 9784597488

    # nested function for the integrand
    def integrand(z, omega_m, omega_lambda):
        return 1/(E(z, omega_m, omega_lambda)*(1+z))

    t = integrate.quad(integrand, z2, z1, (omega_m, omega_lambda))

    if t[1] > 0.001*t[0]:
        raise ValueError("Error greater than 1 per mil")

    return tH * t[0] / h


def LCDM_time_to_z(t, omega_m, omega_lambda, h, zmin=0., zmax=1000., zini=1.,
                    ztol=1e-4,method='newton'):
    """
    Computes the redshift corresponding to a given time, using a bisection method.

    Args:
        t: time, in years
        omega_m: matter density parameter, at z=0
        omega_lambda: dark energy density parameter, at z=0
        h: dimensionless Hubble constant
        zmin: minimum redshift to consider (bisection)
        zmax: maximum redshift to consider (bisection)
        zinit: initial guess for the redshift (Newton)
        ztol: tolerance for the redshift
        method: method to use for the computation. Options: 'bisection', 'newton'
    
    Returns:
        Redshift corresponding to the given time
    """
    if method == 'bisection':
        # simple bisection method 
        # at all times, z1 < z2 --> t1 > t2
        z1 = zmin 
        z2 = zmax 
        t1 = LCDM_time(np.inf, z1, omega_m, omega_lambda, h)
        t2 = LCDM_time(np.inf, z2, omega_m, omega_lambda, h)
        assert t1 > t and t2 < t

        while z2 - z1 > ztol:
            ztest = (z1+z2)/2
            ttest = LCDM_time(np.inf, ztest, omega_m, omega_lambda, h)
            if ttest < t:
                z2 = ztest
                t2 = ttest
            else:
                z1 = ztest
                t1 = ttest

            assert t1 > t and t2 < t
        
        return (z1+z2)/2
    elif method == 'newton':
        tH = 9784597488 / h

        zbackup = zini + 1000*ztol 
        z = zini 
        while abs(z-zbackup) > ztol:
            zbackup = z
            tz = LCDM_time(np.inf, z, omega_m, omega_lambda, h)
            z = z - (1+z)*E(z, omega_m, omega_lambda) * (t - tz)/tH
        return z
    else:
        raise ValueError("Method not implemented")


def critical_density(h, z=0, omega_m=0, omega_lambda=0):
    """
    Computes the value of the critical density of the universe (for making k=0, Lambda=0) at a given redshift, z.
    Units: solar masses per Mpc^3.

    Args:
        h: dimensionless Hubble parameter, H_0 = 100 h km/s/Mpc
        z: redshift (don't specify if values at the present time required)
        omega_m: matter density parameter, at z=0
        omega_lambda: dark energy density parameter, at z=0

    Returns:
        Critical density, in M_\odot / Mpc^3
    """
    isu_0 = 3 * (100000 * h / units.mpc_to_m) ** 2 / (8*np.pi*units.G_isu)
    if z == 0:
        isu = isu_0
    else:
        isu = isu_0 * E(z, omega_m, omega_lambda) ** 2

    return isu * units.kg_to_sun / units.m_to_mpc**3


def background_density(h, omega_m, z=0):
    """
    Computes the value of the background (matter) density of the universe, at a given redshift z.
    Units: solar masses per Mpc^3.

    Args:
        h: dimensionless Hubble parameter, H_0 = 100 h km/s/Mpc
        z: redshift (don't specify if values at the present time required)
        omega_m: matter density parameter, at z=0

    Returns:
        Bkg density, in M_\odot / Mpc^3
    """
    # rhoB = omega_m * rhoC0 / (1+z)^3
    return omega_m * critical_density(h) * (1+z)**3


def overdensity_BN98(z, omega_m, wrt='background'):
    '''
    Computes the Brian & Norman (1998) overdensity factor for a given redshift.
    Assumes flat LCDM cosmology.

    Args:
        z: redshift
        wrt: 'background' or 'critical' (default: 'background')

    Returns:
        Overdensity factor
    '''
    omega_m_z = omega_m * (1+z)**3 / E(z, omega_m, 1-omega_m)**2
    x = omega_m_z - 1
    delta_c = 18*np.pi**2 + 82*x - 39*x**2
    if wrt == 'critical':
        return delta_c
    elif wrt == 'background':
        return delta_c / omega_m_z
    else:
        raise ValueError("wrt must be 'background' or 'critical'")
