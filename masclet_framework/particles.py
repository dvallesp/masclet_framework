"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

particles module
Contains several useful functions in order to deal with particles

Created by David Vallés
"""
#  Last update on 1/6/20 23:46

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

# numpy
import numpy as np
from tqdm import tqdm

from masclet_framework import cosmo_tools, units, tools

import collections
from scipy import optimize
from scipy.spatial import KDTree

import numba
import multiprocessing as mp

# FUNCTIONS DEFINED IN THIS MODULE


# SECTION: manipulate MASCLET data
def correct_positive_oripa(oripa, mass, use_tqdm=True):
    """
    Corrects a bug in the simulation's ouput, which caused all oripa to be positive.
    The expected behaviour was that particles which were originally at l=0 retain negative oripa.
    Args:
        oripa: np array contaning the directly read (incorrect) oripa for each particle
        mass: np array containing the mass of each particle

    Returns:
        corrected oripa array, where particles which originally were at l=0 have negative oripa.

    """
    dups = collections.defaultdict(list)
    if use_tqdm:
        for i, e in tqdm(enumerate(oripa), total=oripa.size):
            dups[e].append(i)
    else:
        for i, e in enumerate(oripa):
            dups[e].append(i)

    for thisoripa, v in dups.items():
        if len(v) >= 2:
            if mass[v[0]] > mass[v[1]]:
                oripa[v[0]] = - thisoripa
            else:
                oripa[v[1]] = - thisoripa

    return oripa

def oripa_heavier_get_negative(oripa, mass, tol=[0.5,2]):
    """
    We redefine the oripas to be unique and constant without ambiguities.

    Args:
        oripa: np array contaning the directly read (incorrect) oripa for each particle
        mass: np array containing the mass of each particle
        tol: bounds to check the if a particle belongs to the most massive species

    Returns:
        corrected oripa array, with all the most massive particles have negative oripas

    """
    maxmass = mass.max()
    oripa[(tol[0] * maxmass < mass) * (mass < tol[1] * maxmass)] = -abs(
        oripa[(tol[0] * maxmass < mass) * (mass < tol[1] * maxmass)])

    return oripa



# SECTION: compare particles by IDs
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
    inside_1 = (x1 - rx1) ** 2 + (y1 - ry1) ** 2 + (z1 - rz1) ** 2 < r1 ** 2
    inside_1 = oripa1[inside_1]

    inside_2 = (x2 - rx2) ** 2 + (y2 - ry2) ** 2 + (z2 - rz2) ** 2 < r2 ** 2
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


# SECTION: find radius from R_\Delta radii definitions
def find_rDelta_eqn_particles(r, Delta, background_density, zeta, clusrx, clusry, clusrz, x, y, z, m, verbose):
    if verbose:
        print('Evaluating at r={:.3f}'.format(r))

    inside_1 = (x - clusrx) ** 2 + (y - clusry) ** 2 + (z - clusrz) ** 2 < r ** 2
    m = m[inside_1].sum() * units.mass_to_sun

    return m - (4 * np.pi / 3) * (r / (1 + zeta)) ** 3 * background_density * Delta


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
    args = (Delta, background_density, zeta, clusrx, clusry, clusrz, x, y, z, m, verbose)
    try:
        rDelta = optimize.brentq(find_rDelta_eqn_particles, rmin, rmax, args=args, xtol=rtol)
    except ValueError:
        if verbose:
            print('Something might be wrong with this one... Might need further checking.')
        return float('nan')
    if verbose:
        print('Converged!')
    return rDelta


# SECTION: radial profiles
def several_radial_profiles(fields, clusrx, clusry, clusrz, rmin, rmax, nbins, logbins, x, y, z, verbose=False):
    """
    Computes a radial profile of the quantity given in the "field" argument, taking center in (clusrx, clusry, clusrz).
    No weight is applied (all particles treated equally). In order to produce an, e.g., mass weighted profile, consider
    producing the profiles of mass and mass * quantity,

    Args:
        fields: set of particles' quantities whose profiles are wanted to be found
        clusrx, clusry, clusrz: comoving coordinates of the center for the profile
        rmin: starting radius of the profile
        rmax: final radius of the profile
        nbins: number of points for the profile
        logbins: if False, radial shells are spaced linearly. If True, they're spaced logarithmically. Not that, if
                 logbins = True, rmin cannot be 0.
        x, y, z: positions of the particles
        verbose: if True, prints the patch being opened at a time

    Returns:
        Two lists. One of them contains the center of each radial cell. The other contains the value of the field
        averaged across all the cells of the shell.

    """
    # getting the bins
    try:
        assert (rmax > rmin)
    except AssertionError:
        print('You would like to input rmax > rmin...')
        return

    if logbins:
        try:
            assert (rmin > 0)
        except AssertionError:
            print('Cannot use rmin=0 with logarithmic binning...')
            return
        bin_bounds = np.logspace(np.log10(rmin), np.log10(rmax), nbins + 1)

    else:
        bin_bounds = np.linspace(rmin, rmax, nbins + 1)

    bin_centers = (bin_bounds[1:] + bin_bounds[:-1]) / 2

    # first shell / interior sphere
    if rmin > 0:
        particles_outer = (x - clusrx) ** 2 + (y - clusry) ** 2 + (z - clusrz) ** 2 < rmin ** 2
    else:
        particles_outer = np.zeros(x.shape, dtype='bool')

    # subsequent spheres
    profiles = []
    for r_out in bin_bounds[1:]:
        if verbose:
            print('Working at outer radius {} Mpc'.format(r_out))
        particles_inner = particles_outer
        particles_outer = (x - clusrx) ** 2 + (y - clusry) ** 2 + (z - clusrz) ** 2 < r_out ** 2
        shell_mask = particles_inner ^ particles_outer

        profile_thisr = [(field * shell_mask).sum() for field in fields]

        profiles.append(profile_thisr)

    profiles = np.asarray(profiles)
    profiles_split = tuple([profiles[:, i] for i in range(profiles.shape[1])])

    return bin_centers, profiles_split


# SECTION: kinematical quantities: angular momenta, shape tensor, etc.
def angular_momentum_particles(x, y, z, vx, vy, vz, m, inside):
    """
    Computes the angular momentum of a particle distribution

    Args:
        x, y, z: (recentered) position of the particles
        vx, vy, vz: components of the particles velocities
        m: particles' masses
        inside: vector of bool values (whether the particles are inside or outside)

    Returns:
        The angular momentum of the given particle distribution. The three components are returned
        in a tuple.

    """
    Lx = (m * (y * vz - z * vy) * inside).sum()
    Ly = (m * (z * vx - x * vz) * inside).sum()
    Lz = (m * (x * vy - y * vx) * inside).sum()

    return Lx, Ly, Lz


def shape_tensor_particles(x, y, z, m, inside):
    """
    Computes the shape tensor of a particle distribution

    Args:
        x, y, z: (recentered) position of the particles
        m: particles' masses
        inside: vector of bool values (whether the particles are inside or outside)

    Returns:
        Shape tensor as a 3x3 matrix
    """
    mass_tot_inside = (m * inside).sum()

    Sxx = (m * x ** 2 * inside).sum() / mass_tot_inside
    Syy = (m * y ** 2 * inside).sum() / mass_tot_inside
    Szz = (m * z ** 2 * inside).sum() / mass_tot_inside
    Sxy = (m * x * y * inside).sum() / mass_tot_inside
    Sxz = (m * x * z * inside).sum() / mass_tot_inside
    Syz = (m * y * z * inside).sum() / mass_tot_inside

    return np.array([[Sxx, Sxy, Sxz], [Sxy, Syy, Syz], [Sxz, Syz, Szz]])


def ellipsoidal_shape_particles(x, y, z, m, r, tol=1e-3, maxiter=100, preserve='major', verbose=False):
    """
    Finds the shape of a particle distribution (eigenvalues and eigenvectors of the intertia tensor) by using
    the iterative method in Zemp et al (2011).

    Args:
        x, y, z: (recentered!) position of the particles
        m: particles' masses
        r: initial radius (will be kept as the major semi-axis length
        tol: relative error allowed to the quotient between semiaxes
        maxiter: maximum number of allowed iterations
        preserve: which quantity to preserve when changing the axes (could be 'major', 'intermediate', 'minor' or
                    'volume')

    Returns:
        List of semiaxes lengths and list of eigenvectors.

    """
    inside = x ** 2 + y ** 2 + z ** 2 < r ** 2
    shapetensor = shape_tensor_particles(x, y, z, m, inside)
    S_eigenvalues, S_eigenvectors = tools.diagonalize_ascending(shapetensor)
    lambda_xtilde = S_eigenvalues[0]
    lambda_ytilde = S_eigenvalues[1]
    lambda_ztilde = S_eigenvalues[2]
    u_xtilde = S_eigenvectors[0]
    u_ytilde = S_eigenvectors[1]
    u_ztilde = S_eigenvectors[2]
    axisratio1 = np.sqrt(lambda_ytilde / lambda_ztilde)
    axisratio2 = np.sqrt(lambda_xtilde / lambda_ztilde)
    if preserve == 'major':
        semiax_x = axisratio2 * r
        semiax_y = axisratio1 * r
        semiax_z = r
    elif preserve == 'intermediate':
        semiax_x = axisratio2 / axisratio1 * r
        semiax_y = r
        semiax_z = r / axisratio1
    elif preserve == 'minor':
        semiax_x = r
        semiax_y = axisratio1 / axisratio2 * r
        semiax_z = r / axisratio2
    elif preserve == 'volume':
        semiax_z = r / (axisratio1 * axisratio2) ** (1 / 3)
        semiax_x = axisratio2 * semiax_z
        semiax_y = axisratio1 * semiax_z

    # these will keep track of the ppal axes positions as the thing rotates over and over
    ppal_x = u_xtilde
    ppal_y = u_ytilde
    ppal_z = u_ztilde

    if verbose:
        print('Iteration -1: spherical')
        print('New ratios are', axisratio1, axisratio2)
        print('New semiaxes are', semiax_x, semiax_y, semiax_z)
        print('New eigenvectors are ', ppal_x, ppal_y, ppal_z)

    for i in range(maxiter):
        if verbose:
            print('Iteration {}'.format(i))
        # rotating the particle coordinates
        xprev = x
        yprev = y
        zprev = z
        x = xprev * u_xtilde[0] + yprev * u_xtilde[1] + zprev * u_xtilde[2]
        y = xprev * u_ytilde[0] + yprev * u_ytilde[1] + zprev * u_ytilde[2]
        z = xprev * u_ztilde[0] + yprev * u_ztilde[1] + zprev * u_ztilde[2]
        del xprev, yprev, zprev

        # compute the new 'inside' particles, considering the ellipsoidal shape, keeping the major semiaxis length
        # note that the major semiaxis corresponds to the ztilde component (by construction, see diagonalize_ascending)
        inside = (x / semiax_x) ** 2 + (y / semiax_y) ** 2 + (z / semiax_z) ** 2 < 1

        # keep track of the previous axisratios
        axisratio1_prev = axisratio1
        axisratio2_prev = axisratio2

        # diagonalize the new particle selection
        shapetensor = shape_tensor_particles(x, y, z, m, inside)

        if np.isnan(shapetensor).sum() != 0 or np.isinf(shapetensor).sum() != 0:
            print('Shape tensor had {} nans. Operation impossible.'.format(np.isnan(shapetensor).sum() +
                                                                           np.isinf(shapetensor).sum()))
            return None, None
        else:
            S_eigenvalues, S_eigenvectors = tools.diagonalize_ascending(shapetensor)

        lambda_xtilde = S_eigenvalues[0]
        lambda_ytilde = S_eigenvalues[1]
        lambda_ztilde = S_eigenvalues[2]
        u_xtilde = S_eigenvectors[0]
        u_ytilde = S_eigenvectors[1]
        u_ztilde = S_eigenvectors[2]
        axisratio1 = np.sqrt(lambda_ytilde / lambda_ztilde)
        axisratio2 = np.sqrt(lambda_xtilde / lambda_ztilde)
        if preserve == 'major':
            semiax_x = axisratio2 * r
            semiax_y = axisratio1 * r
            semiax_z = r
        elif preserve == 'intermediate':
            semiax_x = axisratio2 / axisratio1 * r
            semiax_y = r
            semiax_z = r / axisratio1
        elif preserve == 'minor':
            semiax_x = r
            semiax_y = axisratio1 / axisratio2 * r
            semiax_z = r / axisratio2
        elif preserve == 'volume':
            semiax_z = r / (axisratio1 * axisratio2) ** (1 / 3)
            semiax_x = axisratio2 * semiax_z
            semiax_y = axisratio1 * semiax_z
        if verbose:
            print('New ratios are', axisratio1, axisratio2)
            print('New semiaxes are', semiax_x, semiax_y, semiax_z)

        # keep track of the newly rotated vectors
        temp_x = u_xtilde[0] * ppal_x + u_xtilde[1] * ppal_y + u_xtilde[2] * ppal_z
        temp_y = u_ytilde[0] * ppal_x + u_ytilde[1] * ppal_y + u_ytilde[2] * ppal_z
        temp_z = u_ztilde[0] * ppal_x + u_ztilde[1] * ppal_y + u_ztilde[2] * ppal_z
        ppal_x = temp_x
        ppal_y = temp_y
        ppal_z = temp_z
        del temp_x, temp_y, temp_z
        if verbose:
            print('New eigenvectors are ', ppal_x, ppal_y, ppal_z)

        # check for tolerance
        if verbose:
            print('Rel. change is {:.6f} and {:.6f}'.format(axisratio1 / axisratio1_prev - 1,
                                                            axisratio2 / axisratio2_prev - 1))
        if abs(axisratio1 / axisratio1_prev - 1) < tol and abs(axisratio2 / axisratio2_prev - 1) < tol:
            if verbose:
                print('Converged!')
            break

    return [semiax_x, semiax_y, semiax_z], [ppal_x, ppal_y, ppal_z]




# SECTION: particle to grid
# Aim: to distribute a particle field onto a grid using an SPH kernel using h as the distance to the k-th nearest neighbour

def locate_particle(xpoint, ypoint, zpoint, dx, dy, dz):
    '''
    This routine locates a particle in the grid, assuming
    that the grid starts at (0, 0, 0).

    Args:
        xpoint: x-coordinate of the point
        ypoint: y-coordinate of the point
        zpoint: z-coordinate of the point
        dx: cell size in the x-direction
        dy: cell size in the y-direction
        dz: cell size in the z-direction

    Returns:
        i: i-index of the cell
        j: j-index of the cell
        k: k-index of the cell
    '''
    i = int(xpoint / dx)
    j = int(ypoint / dy)
    k = int(zpoint / dz)
    return i, j, k


def query_k(xpoint, ypoint, zpoint, tree, k):
    '''
    This routine queries the k-th nearest neighbours to a given point.

    Args:
        xpoint: x-coordinate of the point
        ypoint: y-coordinate of the point
        zpoint: z-coordinate of the point
        tree: KDTree object
        k: k-th nearest neighbor

    Returns:
        dist: distance to the k-th nearest neighbor
    '''
    
    return tree.query([xpoint, ypoint, zpoint], k = [k])[0]


def particle2grid(field, x, y, z, Lx, Ly, Lz, nx, ny, nz, k = 32, ncores = 1):
    '''
    This routine aims to distribute a particle field onto a grid using an
    SPH kernel using h as the distance to the k-th nearest neighbour.

    Args:
        ncores: number of cores to be used
        field: particle field to be distributed
        x: x-coordinate of the particles
        y: y-coordinate of the particles
        z: z-coordinate of the particles
        Lx: length of the domain in the x-direction
        Ly: length of the domain in the y-direction
        Lz: length of the domain in the z-direction
        nx: number of cells in the x-direction
        ny: number of cells in the y-direction
        nz: number of cells in the z-direction
        k: number of nearest neighbour to be used

    Returns:
        hpart: h distance of the particles
        grid_field: particle field distributed into a grid

    Author: Óscar Monllor
    '''

    # First, build KDTree
    data = np.array((x, y, z)).T
    tree = KDTree(data)

    #
    npart = len(x)

    # Compute h distance of the particles using KDTree and multiprocessing
    with mp.get_context("fork").Pool(ncores) as p:
        hpart = np.array(p.starmap(query_k, [(x[ipart], y[ipart], z[ipart], tree, k)
                                             for ipart in range(npart)])).flatten()
        
        
    # Locate particles on the grid
    dx = Lx/nx
    dy = Ly/ny
    dz = Lz/nz

    with mp.get_context("fork").Pool(ncores) as p:
        result = np.array( p.starmap(locate_particle, [(x[ipart], y[ipart], z[ipart], 
                                                         dx, dy, dz)
                                                        for ipart in range(npart)]) )
        
    grid_pos_x = result[:, 0].astype(np.int64)
    grid_pos_y = result[:, 1].astype(np.int64)
    grid_pos_z = result[:, 2].astype(np.int64)


    # Define grid field.
    grid_field = np.zeros((nx, ny, nz))


    ##############################################
    # Distribute the particle field into the grid
    @numba.njit(fastmath = True)
    def SPH_kernel(r, h):
        q = r/h
        if q < 1:
            W = 1 - 1.5 * q**2 + 0.75 * q**3
        elif q < 2:
            W = 0.25 * (2 - q)**3
        else:
            W = 0
        return W
    

    @numba.njit(fastmath = True)
    def put_particles_in_grid(grid_field, grid_pos_x, grid_pos_y, grid_pos_z, x, y, z, field, npart, hpart, dx, dy, dz, nx, ny, nz):

        # Distribute the particle field into the grid
        for ipart in range(npart):
            i = grid_pos_x[ipart]
            j = grid_pos_y[ipart]
            k = grid_pos_z[ipart]

            # extent of the particle kernel
            h = hpart[ipart]
            x_extent = int(h/dx)
            y_extent = int(h/dy)
            z_extent = int(h/dz)

            # for each particle, find the norm
            norm = 0.
            for ix in range(i - x_extent, i + x_extent + 1):
                if ix >= 0 and ix < nx:
                    for jy in range(j - y_extent, j + y_extent + 1):
                        if jy >= 0 and jy < ny:
                            for kz in range(k - z_extent, k + z_extent + 1):
                                if kz >= 0 and kz < nz:
                                    # distance to the particle
                                    r = np.sqrt((ix*dx - x[ipart])**2 + (jy*dy - y[ipart])**2 + (kz*dz - z[ipart])**2)
                                    # kernel value
                                    W = SPH_kernel(r, h)
                                    #
                                    norm += W

            # distribute the field
            for ix in range(i - x_extent, i + x_extent + 1):
                if ix >= 0 and ix < nx:
                    for jy in range(j - y_extent, j + y_extent + 1):
                        if jy >= 0 and jy < ny:
                            for kz in range(k - z_extent, k + z_extent + 1):
                                if kz >= 0 and kz < nz:
                                    # distance to the particle
                                    r = np.sqrt((ix*dx - x[ipart])**2 + (jy*dy - y[ipart])**2 + (kz*dz - z[ipart])**2)
                                    # kernel value
                                    W = SPH_kernel(r, h)
                                    #
                                    grid_field[ix, jy, kz] += field[ipart] * W / norm


        return grid_field
    ##############################################

    return put_particles_in_grid(grid_field, grid_pos_x, grid_pos_y, grid_pos_z, x, y, z,
                            field, npart, hpart, dx, dy, dz, nx, ny, nz)
