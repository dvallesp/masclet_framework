"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

parameters module
Functions for reading and creating parameters JSON files

Created by David Vallés
"""

#  Last update on 2/9/19 18:01

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

import json
import os


# FUNCTIONS DEFINED IN THIS MODULE


def read_parameters_file(filename='masclet_parameters.json', path=''):
    """
    Returns dictionary containing the MASCLET parameters of the simulation, that have been previously written with the
    write_parameters() function in this same module.

    Args:
        filename: name of the MASCLET parameters file (str)
        path: path of the file (typically, the codename of the simulation) (str)

    Returns:
        dictionary containing the parameters (and their names), namely:
        NMAX, NMAY, NMAZ: number of l=0 cells along each direction (int)
        NPALEV: maximum number of refinement cells per level (int)
        NLEVELS: maximum number of refinement level (int)
        NAMRX (NAMRY, NAMRZ): maximum size of refinement patches (in l-1 cell units) (int)
        SIZE: side of the simulation box in the chosen units (typically Mpc or kpc) (float)

    """
    filepath = os.path.join(path, filename)
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data


def read_parameters(filename='masclet_parameters.json', path='', load_nma=True, load_npalev=True, load_nlevels=True,
                    load_namr=True, load_size=True):
    """
    Returns MASCLET parameters in the old-fashioned way (as a tuple).
    Legacy (can be used, but newer codes should try to switch to directly reading the dictionary with
    read_parameters_file() function).

    Args:
        filename: name of the MASCLET parameters file (str)
        path: path of the file (typically, the codename of the simulation) (str)
        load_nma: whether NMAX, NMAY, NMAZ are read (bool)
        load_npalev: whether NPALEV is read (bool)
        load_nlevels: whether NLEVELS is read (bool)
        load_namr: whether NAMRX, NAMRY, NAMRZ is read (bool)
        load_size: whether SIZE is read (bool)

    Returns:
        tuple containing, in this exact order, the chosen parameters from:
        NMAX, NMAY, NMAZ: number of l=0 cells along each direction (int)
        NPALEV: maximum number of refinement cells per level (int)
        NLEVELS: maximum number of refinement level (int)
        NAMRX (NAMRY, NAMRZ): maximum size of refinement patches (in l-1 cell units) (int)
        SIZE: side of the simulation box in the chosen units (typically Mpc or kpc) (float)

    """
    parameters = read_parameters_file(filename=filename, path=path)
    returnvariables = []
    if load_nma:
        returnvariables.extend([parameters[i] for i in ['NMAX', 'NMAY', 'NMAZ']])
    if load_npalev:
        returnvariables.append(parameters['NPALEV'])
    if load_nlevels:
        returnvariables.append(parameters['NLEVELS'])
    if load_namr:
        returnvariables.extend([parameters[i] for i in ['NAMRX', 'NAMRY', 'NAMRZ']])
    if load_size:
        returnvariables.append(parameters['SIZE'])
    return tuple(returnvariables)


def write_parameters(nmax, nmay, nmaz, npalev, nlevels, narmx, narmy, narmz,
                     size, filename='masclet_parameters.json', path=''):
    """
    Creates a JSON file containing the parameters of a certain simulation

    Args:
        nmax: number of l=0 cells along the X-direction (int)
        nmay: number of l=0 cells along the Y-direction (int)
        nmaz: number of l=0 cells along the Z-direction (int)
        npalev: maximum number of refinement cells per level (int)
        nlevels: maximum number of refinement level (int)
        narmx: maximum X-size of refinement patches (in l-1 cell units) (int)
        narmy: maximum Y-size of refinement patches (in l-1 cell units) (int)
        narmz: maximum Z-size of refinement patches (in l-1 cell units) (int)
        size: side of the simulation box in the chosen units (typically Mpc or kpc) (float)
        filename: name of the MASCLET parameters file to be saved (str)
        path: path of the file (typically, the codename of the simulation) (str)

    Returns: nothing. A file is created in the specified path
    """
    parameters = {'NMAX': nmax, 'NMAY': nmay, 'NMAZ': nmaz,
                  'NPALEV': npalev, 'NLEVELS': nlevels,
                  'NAMRX': narmx, 'NAMRY': narmy, 'NAMRZ': narmz,
                  'SIZE': size}

    with open(path + filename, 'w') as json_file:
        json.dump(parameters, json_file)
