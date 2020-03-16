"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

cosmo_tools module
Provides useful tools to compute time from redshift, evolution of the critical density, etc.

Created by David Vallés
"""

#  Last update on 16/3/20 19:27

import json
import os
import numpy as np
from scipy import integrate


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


def read_cosmo_parameters(filename='masclet_parameters.json', path=''):
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

def LCDM_time(z1,z2,omega_m,omega_lambda,h,npoints=10000):
    """
    Computes the time between z1 and z2, in years.

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

    # Simpson rule requires even number of intervals (odd number of samples)
    if npoints % 2 == 0:
        npoints = npoints + 1

    z = np.linspace(z2,z1,npoints)
    dz = (z1-z2)/npoints
    integrand = 1/((1+z)*E(z,omega_m,omega_lambda))

    t = tH * integrate.simps(integrand, dx=dz) / h

    return t
