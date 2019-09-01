"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

tools module
Contains several useful functions that other modules might need

Created by David Vallés
"""

#  Last update on 1/9/19 0:53

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

# numpy
import numpy as np

# FUNCTIONS DEFINED IN THIS MODULE

def create_vector_levels(NPATCH):
    """
    Creates a vector containing the level for each patch. Nothing really important, just for ease

    Args:
        NPATCH: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)

    Returns:
        numpy array containing the level for each patch

    """
    vector = [0]
    for l in range(1, NPATCH.size):
        vector.extend([l for i in range(NPATCH[l])])
    return np.array(vector)


def find_absolute_grid_position(ipatch, NPATCH, PATCHX, PATCHY, PATCHZ, PARE):
    """
    Given a patch in a certain iteration, finds the grid (natural) coordinates of its left corner.
    Written as a recursive function this time.

    Args:
        ipatch: number of the patch (in the given iteration)
        NPATCH: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        PATCHX: vector containing the X-position of the leftmost corner of all the patches (numpy vector of integers)
        PATCHY: vector containing the Y-position of the leftmost corner of all the patches (numpy vector of integers)
        PATCHZ: vector containing the Z-position of the leftmost corner of all the patches (numpy vector of integers)
        PARE: vector containing the ipatch of each patch progenitor (numpy vector of integers)

    Returns:
        tuple with the patches left-corner NX, NY, NZ values

    """
    level = create_vector_levels(NPATCH)[ipatch]
    if ipatch < PATCHX.size:
        if level == 1:
            return PATCHX[ipatch], PATCHY[ipatch], PATCHZ[ipatch]
        else:
            parevalues = find_absolute_grid_position(PARE[ipatch], NPATCH, PATCHX, PATCHY, PATCHZ, PARE)
            return ((PATCHX[ipatch] - 1) / 2 ** (level - 1) + parevalues[0],
                    (PATCHY[ipatch] - 1) / 2 ** (level - 1) + parevalues[1],
                    (PATCHZ[ipatch] - 1) / 2 ** (level - 1) + parevalues[2])
    else:
        raise ValueError('Provide a valid patchnumber (ipatch)')


def find_absolute_real_position(ipatch, SIDE, NMAX, NPATCH, PATCHX, PATCHY, PATCHZ, PARE):
    """
    Given a patch in a certain iteration, finds the real coordinates of its left corner with no numerical error.
    This function depends on find_absolute_grid_position, which must be also loaded.

    Args:
        ipatch: number of the patch (in the given iteration) (int)
        SIDE: side of the simulation box in the chosen units (typically Mpc or kpc) (float)
        NMAX: number of cells along each directions at level l=0 (can be loaded from masclet_parameters; NMAX = NMAY =
        = NMAZ is assumed) (int)
        NPATCH: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        PATCHX: vector containing the X-position of the leftmost corner of all the patches (numpy vector of integers)
        PATCHY: vector containing the Y-position of the leftmost corner of all the patches (numpy vector of integers)
        PATCHZ: vector containing the Z-position of the leftmost corner of all the patches (numpy vector of integers)
        PARE: vector containing the ipatch of each patch progenitor (numpy vector of integers)

    Returns:
    numpy array with the patches left corner X, Y and Z values (assuming box centered in 0; same units than SIDE)

    """
    return (np.asarray(
        find_absolute_grid_position(ipatch, NPATCH, PATCHX, PATCHY, PATCHZ, PARE)) - NMAX / 2 - 1) * SIDE / NMAX

