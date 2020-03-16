"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

cosmo_tools module
Provides useful tools to compute time from redshift, evolution of the critical density, etc.

Created by David Vallés
"""

#  Last update on 16/3/20 18:28

import json
import os


def write_cosmo_parameters(h, omega_m, omega_lambda, omega_b, filename='cosmo_parameters.json', path=''):
    """
    Creates a JSON file containing the cosmological parameters for a simulation

    Args:
        h: dimensionless Hubble parameter
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
        h: dimensionless Hubble parameter
        omega_m: matter density in terms of the critical density, at z=0
        omega_lambda: dark energy density in terms of the critical density, at z=0
        omega_b: baryionic matter density in terms of the critical density, at z=0

    """
    filepath = os.path.join(path, filename)
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data
