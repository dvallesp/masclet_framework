"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

particles module
Contains several useful functions in order to deal with particles

Created by David Vallés
"""
#  Last update on 27/3/20 17:50

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

# numpy
import numpy as np

from multiprocessing import Pool

from masclet_framework import cosmo_tools, units

import collections

# FUNCTIONS DEFINED IN THIS MODULE


def correct_positive_oripa(oripa, mass, npart):
    """
    Corrects a bug in the simulation's ouput, which caused all oripa to be positive.
    The expected behaviour was that particles which were originally at l=0 retain negative oripa.
    Args:
        oripa: np array contaning the directly read (incorrect) oripa for each particle
        mass: np array containing the mass of each particle
        npart: NPART array, as read from read_grids() (number of dm particles per level)

    Returns:
        corrected oripa array, where particles which originally were at l=0 have negative oripa.

    """
    # particles
    #for i in range(npart[0]):
    #    oripa[i] = - oripa[i]

    dups = collections.defaultdict(list)
    for i, e in enumerate(oripa):
        dups[e].append(i)

    for thisoripa, v in dups.items():
        if len(v) >= 2:
            if mass[v[0]] > mass[v[1]]:
                oripa[v[0]] = - thisoripa
            else:
                oripa[v[1]] = - thisoripa

    return oripa


def shared_particles(x1, y1, z1, oripa1, rx1, ry1, rz1, r1, x2, y2, z2, oripa2, rx2, ry2, rz2, r2):
    """
    Finds the shared particles between two haloes (of DM particles; although it can be used for any particle simulation)

    Args: for i=1,2
        xi: x position of all the particles at iti
        yi: y position of all the particles at iti
        zi: z position of all the particles at iti
        oripai: unique identifier of the particles at iti
        rxi: x position of the center of the halo i
        ryi: y position of the center of the halo i
        rzi: z position of the center of the halo i
        ri: radius of the halo i where the particles are being considered

    Returns:
        A numpy array containing the intersecting oripas
    """
    inside_1 = (x1 - rx1)**2 + (y1 - ry1)**2 + (z1 - rz1)**2 < r1**2
    inside_1 = oripa1[inside_1]

    inside_2 = (x2 - rx2)**2 + (y2 - ry2)**2 + (z2 - rz2)**2 < r2**2
    inside_2 = oripa2[inside_2]

    intersect = np.intersect1d(inside_1, inside_2, return_indices=False, assume_unique=True)

    return intersect


def shared_mass(x1, y1, z1, m1, oripa1, rx1, ry1, rz1, r1, x2, y2, z2, m2, oripa2, rx2, ry2, rz2, r2):
    """
    Computes the shared mass between two haloes (of DM particles; although it can be used for any particle simulation).
    Units: solar masses.

    Args: for i=1,2
        xi: x position of all the particles at iti
        yi: y position of all the particles at iti
        zi: z position of all the particles at iti
        mi: mass of all the particles at iti
        oripai: unique identifier of the particles at iti
        rxi: x position of the center of the halo i
        ryi: y position of the center of the halo i
        rzi: z position of the center of the halo i
        ri: radius of the halo i where the particles are being considered

    Returns:
        The mass in the older halo (smaller iti) also present in the newer one.
    """
    inside_1 = (x1 - rx1) ** 2 + (y1 - ry1) ** 2 + (z1 - rz1) ** 2 < r1 ** 2
    inside_1 = oripa1[inside_1]
    m1 = m1[inside_1]

    inside_2 = (x2 - rx2) ** 2 + (y2 - ry2) ** 2 + (z2 - rz2) ** 2 < r2 ** 2
    inside_2 = oripa2[inside_2]

    _, indices1, _ = np.intersect1d(inside_1, inside_2, return_indices=True, assume_unique=True)

    return m1[indices1].sum() * units.mass_to_sun
