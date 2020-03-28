"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

particles module
Contains several useful functions in order to deal with particles

Created by David Vallés
"""
#  Last update on 28/3/20 1:04

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

# numpy
import numpy as np

from multiprocessing import Pool

from masclet_framework import cosmo_tools, units

import collections
from scipy import optimize

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
    # checks (just in case)
    if np.isnan(r1):
        r1 = 0
    if np.isnan(r2):
        r2 = 0

    inside_1 = (x1 - rx1) ** 2 + (y1 - ry1) ** 2 + (z1 - rz1) ** 2 < r1 ** 2
    inside_1 = oripa1[inside_1]
    m1 = m1[inside_1]

    inside_2 = (x2 - rx2) ** 2 + (y2 - ry2) ** 2 + (z2 - rz2) ** 2 < r2 ** 2
    inside_2 = oripa2[inside_2]

    _, indices1, _ = np.intersect1d(inside_1, inside_2, return_indices=True, assume_unique=True)

    return m1[indices1].sum() * units.mass_to_sun


def find_rDelta_eqn_particles(r, Delta, background_density, clusrx, clusry, clusrz, x, y, z, m, verbose):
    if verbose:
        print('Evaluating at r={:.3f}'.format(r))

    inside_1 = (x - clusrx) ** 2 + (y - clusry) ** 2 + (z - clusrz) ** 2 < r ** 2
    m = m[inside_1].sum() * units.mass_to_sun

    return m - (4*np.pi/3) * r**3 * background_density * Delta


def find_rDelta_particles(Delta, zeta, clusrx, clusry, clusrz, x, y, z, m, h, omega_m, rmin=0.1, rmax=6, rtol=1e-3,
                          verbose=False):
    """
    Finds the value (in Mpc) of r_\Delta, the radius enclosing a mean overdensity (of the DM field, by default) equal
    to Delta times the background density of the universe. By default, it uses the Brent method.

    Args:
        Delta: value of the desired overdensity
        zeta: current redshift
        clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
        x, y, z: vector with the particles' positions
        m: vector with the particles' masses
        h: dimensionless Hubble constant
        omega_m: matter density parameter
        rmin, rmax: starting two points (M(r) - Delta rho_B 4pi/3 r^3) must change sign
        rtol: tolerance in the determination of r
        verbose: if True, prints the patch being opened at a time

    Returns:
        The value of r_\Delta
    """
    background_density = cosmo_tools.background_density(h, omega_m, zeta)
    args = (Delta, background_density, clusrx, clusry, clusrz, x, y, z, m, verbose)
    try:
        rDelta = optimize.brentq(find_rDelta_eqn_particles, rmin, rmax, args=args, xtol=rtol)
    except ValueError:
        if verbose:
            print('Something might be wrong with this one... Might need further checking.')
        return float('nan')
    if verbose:
        print('Converged!')
    return rDelta

