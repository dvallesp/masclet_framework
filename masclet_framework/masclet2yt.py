"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

masclet4yt module
Aims to serve as a user-friendly link between masclet software and the yt
package

Created by David Vallés
"""
#  Last update on 2/9/19 18:01

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

# numpy
import numpy as np

# masclet_framework
from masclet_framework import tools
from masclet_framework import parameters, read_masclet


# FUNCTIONS DEFINED IN THIS MODULE


def load_grids(it, path='', digits=5):
    """
    This function creates a list of dictionaries containing the information requiered for yt's load_amr_grids
    to build the grid structure of a simulations performed by MASCLET.

    Args:
        it: iteration number (int)
        path: path of the grids file in the system (str)
        digits: number of digits the filename is written with (int)

    Returns:
        grid_data: list of dictionaries, each containing the information about one refinement patch (left_edge,
        right_edge, level and dimensions [number of cells])
        bbox: bounds of the simulation box in physical coordinates (typically Mpc or kpc)

    """
    nmax, nmay, nmaz, nlevels, namrx, namry, namrz, size = parameters.read_parameters(load_npalev=False)
    irr, t, nl, mass_dmpart, zeta, npatch, npart, patchnx, patchny, patchnz, patchx, patchy, patchz, pare = \
        read_masclet.read_grids(it, path=path, read_patchposition=False, digits=digits)

    # l=0 (whole box)
    grid_data = [dict(left_edge=[-size / 2, -size / 2, -size / 2],
                      right_edge=[size / 2, size / 2, size / 2],
                      level=0,
                      dimensions=[nmax, nmay, nmaz])]

    levels = tools.create_vector_levels(npatch)
    left = np.array([tools.find_absolute_real_position(i, size, nmax, npatch, patchx, patchy, patchz, pare) for i in
                     range(1, levels.size)])
    left = np.vstack([[-size / 2, -size / 2, -size / 2], left])
    right = left + np.array([patchnx / 2 ** levels, patchny / 2 ** levels, patchnz / 2 ** levels]).transpose()[:, :] \
        * size / nmax

    for i in range(1, levels.size):
        grid_data.append(dict(left_edge=left[i, :].tolist(),
                              right_edge=right[i, :].tolist(),
                              level=levels[i],
                              dimensions=[patchnx[i], patchny[i], patchnz[i]]))

    bbox = np.array([[-size / 2, size / 2], [-size / 2, size / 2], [-size / 2, size / 2]])

    return grid_data, bbox


def load_newfield(grid_data, fieldname, fieldl0, field_refined):
    """
    This function adds a new field to the yt dataset, once created with yt4masclet_load_grids().
    fieldname specifies the name of the field. Then, fieldl0 (base level) and field_refined have
    to be given, as outputted by readclus, readcldm functions.

    Args:
        grid_data: as outputted by yt4masclet_load_grids()
        fieldname: name of the field to be added
        fieldl0: field at the base (l=0) level
        field_refined: field for each refinement patch, as outputted by readclus() or readcldm() functions

    Returns:
        New grid_data object with the additional field included.

    """
    grid_data[0][fieldname] = fieldl0
    for g in range(1, len(grid_data)):
        try:
            grid_data[g][fieldname] = field_refined[g]
        except IndexError:  # field not provided for this patch
            grid_data[g][fieldname] = np.zeros(grid_data[g]['dimensions'])
    return grid_data
