"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

tools_xyz module
Contains several useful functions that other modules might need. Here we have the (new) versions, which make more
intensive use of computing x,y,z fields (much faster, but more memory consuming)

Created by David Vallés
"""

#  Last update on 2/9/20 17:06

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

# numpy
import numpy as np

from multiprocessing import Pool

from masclet_framework import cosmo_tools, tools

from scipy import optimize

import numba


# FUNCTIONS DEFINED IN THIS MODULE

# SECTION: compute the position fields
def compute_position_field_onepatch(args):
    """
    Returns a 3 matrices with the dimensions of the patch, containing the position for every cell centre

    Args: tuple containing, in this order:
        nx, ny, nz: extension of the patch (in cells at level n)
        rx, ry, rz: comoving coordinates of the center of the leftmost cell of the patch
        level: refinement level of the given patch
        size: comoving box side (preferred length units)
        nmax: cells at base level

    Returns:
        Matrices as defined
    """
    nx, ny, nz, rx, ry, rz, level, size, nmax, keep = args

    if not keep:
        return 0,0,0

    cellsize = size / nmax / 2 ** level
    first_x = rx - cellsize / 2
    first_y = ry - cellsize / 2
    first_z = rz - cellsize / 2
    patch_x = np.zeros((nx, ny, nz), dtype='f4')
    patch_y = np.zeros((nx, ny, nz), dtype='f4')
    patch_z = np.zeros((nx, ny, nz), dtype='f4')

    for i in range(nx):
        patch_x[i, :, :] = first_x + i * cellsize
    for j in range(ny):
        patch_y[:, j, :] = first_y + j * cellsize
    for k in range(nz):
        patch_z[:, :, k] = first_z + k * cellsize

    return patch_x, patch_y, patch_z


def compute_position_fields(patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, size, nmax, ncores=1,
                            kept_patches=None):
    """
    Returns 3 fields (as usually defined) containing the x, y and z position for each of our cells centres.
    Args:
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        (and Y and Z)
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level
        ncores: number of cores to be used in the computation
        kept_patches: 1d boolean array, True if the patch is kept, False if not. If None, all patches are kept.

    Returns:
        3 fields as described above
    """
    levels = tools.create_vector_levels(npatch)
    if kept_patches is None:
        kept_patches = np.ones(patchnx.size, dtype=bool)

    if ncores == 1 or ncores == 0 or ncores is None:
        cellsrx = []
        cellsry = []
        cellsrz = []
        for ipatch in range(npatch.sum()+1):
            patches = compute_position_field_onepatch((patchnx[ipatch], patchny[ipatch], patchnz[ipatch],
                                                       patchrx[ipatch], patchry[ipatch], patchrz[ipatch],
                                                       levels[ipatch], size, nmax, kept_patches[ipatch]))
            cellsrx.append(patches[0])
            cellsry.append(patches[1])
            cellsrz.append(patches[2])
    else:
        with Pool(ncores) as p:
            positions = p.map(compute_position_field_onepatch,
                              [(patchnx[ipatch], patchny[ipatch], patchnz[ipatch], patchrx[ipatch], patchry[ipatch],
                                patchrz[ipatch], levels[ipatch], size, nmax, kept_patches[ipatch]) 
                                for ipatch in range(len(patchnx))])

        cellsrx = [p[0] for p in positions]
        cellsry = [p[1] for p in positions]
        cellsrz = [p[2] for p in positions]

    return cellsrx, cellsry, cellsrz


# SECTION: grid geometry
def patch_vertices(ipatch, cellsrx, cellsry, cellsrz):
    """
    Returns, for a given patch, the comoving coordinates of its 8 vertices.

    Args:
        ipatch: number of the patch to be considered
        cellsrx, cellsry, cellsrz: position fields

    Returns:
        List containing 8 tuples, each one containing the x, y, z coordinates of the vertex.

    """

    cellsize = cellsrx[ipatch][1, 0, 0] - cellsrx[ipatch][0, 0, 0]

    leftmost_x = cellsrx[ipatch][0, 0, 0] - cellsize / 2
    leftmost_y = cellsry[ipatch][0, 0, 0] - cellsize / 2
    leftmost_z = cellsrz[ipatch][0, 0, 0] - cellsize / 2

    vertices = []

    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = leftmost_x + i * cellsrx[ipatch].shape[0] * cellsize
                y = leftmost_y + j * cellsrx[ipatch].shape[1] * cellsize
                z = leftmost_z + k * cellsrx[ipatch].shape[2] * cellsize

                vertices.append((x, y, z))

    return vertices


def mask_sphere(R, clusrx, clusry, clusrz, cellsrx, cellsry, cellsrz, kept_patches=None):
    """
    Returns a "field", which contains all patches as usual. True means the cell must be considered, False otherwise.
    If a patch has all "False", an array is not ouputted, but a False is, instead.

    Args:
        R: radius of the considered sphere
        clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
        cellsrx, cellsry, cellsrz: position fields
        kept_patches: 1d boolean array, True if the patch is kept, False if not. If None, all patches are kept.

    Returns:
        Field containing the mask as described.
    """
    mask = [(cx - clusrx) ** 2 + (cy - clusry) ** 2 + (cz - clusrz) ** 2 < R ** 2 if ki else False
            for cx, cy, cz, ki in zip(cellsrx, cellsry, cellsrz, kept_patches)]

    return mask


# SECTION: calculations with coordinates: masses, profiles, etc.
def mass_inside(R, clusrx, clusry, clusrz, density, cellsrx, cellsry, cellsrz, npatch, size, nmax):
    """
        Computes the mass inside a radius R sphere centered on (clusrx, clusry, clusrz), from the density field.
        Note that user can either supply:
            - Comoving density and comoving size (most common)
            - Physical density and physical size

        Args:
            R: radius of the considered sphere
            clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
            density: density field, already cleaned from refinements and overlaps (but not masked!)
            cellsrx, cellsry, cellsrz: position fields
            npatch: number of patches in each level, starting in l=0
            size: comoving size of the simulation box
            nmax: cells at base level

        Returns:
            Field containing the mask as described.
    """
    levels = tools.create_vector_levels(npatch)
    cells_volume = (size / nmax / 2 ** levels) ** 3

    if np.isnan(R):
        R = 0

    mask = mask_sphere(R, clusrx, clusry, clusrz, cellsrx, cellsry, cellsrz)

    cellmasses = [d * m * cv for d, m, cv in zip(density, mask, cells_volume)]
    mass = sum([cm.sum() for cm in cellmasses])

    return mass


def radial_profile_vw(field, clusrx, clusry, clusrz, rmin, rmax, nbins, logbins, cellsrx, cellsry,
                      cellsrz, cr0amr, solapst, npatch, size, nmax, up_to_level=1000, verbose=False):
    """
    Computes a (volume-weighted) radial profile of the quantity given in the "field" argument, taking center in
    (clusrx, clusry, clusrz).

    Args:
        field: variable (already cleaned) whose profile wants to be got
        clusrx, clusry, clusrz: comoving coordinates of the center for the profile
        rmin: starting radius of the profile
        rmax: final radius of the profile
        nbins: number of points for the profile
        logbins: if False, radial shells are spaced linearly. If True, they're spaced logarithmically. Not that, if
                 logbins = True, rmin cannot be 0.
        cellsrx, cellsry, cellsrz: position fields
        cr0amr: field containing the refinements of the grid (1: not refined; 0: refined)
        solapst: field containing the overlaps (1: keep; 0: not keep)
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level
        up_to_level: maximum AMR level to be considered for the profile
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
    # profile = np.zeros(bin_centers.shape)
    profile = []

    # finding the volume-weighted mean
    levels = tools.create_vector_levels(npatch)
    cell_volume = (size / nmax / 2 ** levels) ** 3

    field_vw = [f * cv for f, cv in zip(field, cell_volume)]

    if rmin > 0:
        cells_outer = mask_sphere(rmin, clusrx, clusry, clusrz, cellsrx, cellsry, cellsrz)
    else:
        cells_outer = [np.zeros(patch.shape, dtype='bool') for patch in field]

    for r_out in bin_bounds[1:]:
        if verbose:
            print('Working at outer radius {} Mpc'.format(r_out))
        cells_inner = cells_outer
        cells_outer = mask_sphere(r_out, clusrx, clusry, clusrz, cellsrx, cellsry, cellsrz)
        shell_mask = [inner ^ outer for inner, outer in zip(cells_inner, cells_outer)]
        shell_mask = tools.clean_field(shell_mask, cr0amr, solapst, npatch, up_to_level=up_to_level)

        sum_field_vw = sum([(fvw * sm).sum() for fvw, sm in zip(field_vw, shell_mask)])
        sum_vw = sum([(sm * cv).sum() for sm, cv in zip(shell_mask, cell_volume)])

        profile.append(sum_field_vw / sum_vw)

    return bin_centers, np.asarray(profile)


def several_radial_profiles_vw(fields, clusrx, clusry, clusrz, rmin, rmax, nbins, logbins, cellsrx, cellsry,
                               cellsrz, cr0amr, solapst, npatch, size, nmax, up_to_level=1000, verbose=False):
    """
    Computes a (volume-weighted) radial profile of the quantity given in the "field" argument, taking center in
    (clusrx, clusry, clusrz).

    Args:
        fields: set of fields (already cleaned) whose profile wants to be got
        clusrx, clusry, clusrz: comoving coordinates of the center for the profile
        rmin: starting radius of the profile
        rmax: final radius of the profile
        nbins: number of points for the profile
        logbins: if False, radial shells are spaced linearly. If True, they're spaced logarithmically. Not that, if
                 logbins = True, rmin cannot be 0.
        cellsrx, cellsry, cellsrz: position fields
        cr0amr: field containing the refinements of the grid (1: not refined; 0: refined)
        solapst: field containing the overlaps (1: keep; 0: not keep)
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level
        up_to_level: maximum AMR level to be considered for the profile
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

    # finding the volume-weighted mean
    levels = tools.create_vector_levels(npatch)
    cell_volume = (size / nmax / 2 ** levels) ** 3

    fields_vw = [[f * cv for f, cv in zip(field, cell_volume)] for field in fields]

    if rmin > 0:
        cells_outer = mask_sphere(rmin, clusrx, clusry, clusrz, cellsrx, cellsry, cellsrz)
    else:
        cells_outer = [np.zeros(patch.shape, dtype='bool') for patch in fields[0]]

    profiles = []
    for r_out in bin_bounds[1:]:
        if verbose:
            print('Working at outer radius {} Mpc'.format(r_out))
        cells_inner = cells_outer
        cells_outer = mask_sphere(r_out, clusrx, clusry, clusrz, cellsrx, cellsry, cellsrz)
        shell_mask = [inner ^ outer for inner, outer in zip(cells_inner, cells_outer)]
        shell_mask = tools.clean_field(shell_mask, cr0amr, solapst, npatch, up_to_level=up_to_level)
        sum_vw = sum([(sm * cv).sum() for sm, cv in zip(shell_mask, cell_volume)])

        profile_thisr = [(sum([(fvw * sm).sum() for fvw, sm in zip(field_vw, shell_mask)]) / sum_vw) for field_vw in
                         fields_vw]

        profiles.append(profile_thisr)

    profiles = np.asarray(profiles)
    profiles_split = tuple([profiles[:, i] for i in range(profiles.shape[1])])

    return bin_centers, profiles_split


def vol_integral(field, units, a0, zeta, cr0amr, solapst, npatch, patchrx, patchry, patchrz, patchnx, patchny, patchnz, size, nmax, coords, rad, max_refined_level=1000, kept_patches=None, vol=False, verbose=False):
    """
    Given a scalar field and a sphere defined with a center (x,y,z) and a radious together with the patch structure, returns the volumetric integral of the field along the sphere.

    Args:
        - field: scalar field to be integrated
        - units: change of units factor to be multiplied by the final integral if one wants physical units
        - a0: scale factor at the initial time with the units of the simulation
        - zeta: redshift of the simulation snap to calculate the scale factor
        - cr0amr: AMR maximum refinement factor (only the maximally resolved cells are considered)
        - solapst: AMR overlap factor (only the maximally resolved cells are considered)
        - npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        - patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        - patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        - size: comoving size of the simulation box
        - nmax: number of cells in the coarsest resolution level
        - coords: center of the sphere in a numpy array [x,y,z]
        - rad: radius of the sphere
        - max_refined_level: maximum refinement level that wants to be read. Subsequent refinements will be skipped. (int)
        - kept_patches: boolean array to select the patches to be considered in the integration. True if the patch is kept, False if not. If None, all patches are kept.
        - vol: if True, returns the volume of integration in phiysical and comoving units
        - verbose: if True, prints the integral being calculated

    Returns:
        - integral: volumetric integral of the field along the sphere
    
    Author: Marco José Molina Pradillo
    """
    if vol == True:
        ones_array = [np.ones_like(patch) for patch in field]  # Test array
        volume_fi = 0
        volume_co = 0
    
    if kept_patches is None:
        total_npatch = len(field)
        kept_patches = np.ones((total_npatch,), dtype=bool)
        
    vector_levels = tools.create_vector_levels(npatch)
    
    dx = size/nmax
    
    a = a0 / (1 + zeta) # We compute the scale factor
    
    integral = 0
    
    for p in range(len(vector_levels)): # We run across all the patches we are interested in
        
        if kept_patches[p] != 0: # We only want the patches inside the region of interest
        
            patch_res = dx/(2**vector_levels[p])
            
            x0 = patchrx[p] - patch_res/2 #Center of the left-bottom-front cell
            y0 = patchry[p] - patch_res/2
            z0 = patchrz[p] - patch_res/2
            
            x_grid = np.linspace(x0, x0 + patch_res * (patchnx[p] - 1), patchnx[p])
            y_grid = np.linspace(y0, y0 + patch_res * (patchny[p] - 1), patchny[p])
            z_grid = np.linspace(z0, z0 + patch_res * (patchnz[p] - 1), patchnz[p])
            
            X_grid, Y_grid, Z_grid = np.meshgrid(x_grid, y_grid, z_grid, indexing='ij')
            
            # Create a boolean mask where the condition is True
            mask = ((coords[0] - X_grid)**2 + (coords[1] - Y_grid)**2 + (coords[2] - Z_grid)**2) <= rad**2
            
            # Calculate the physical volume of the cell in this simulation patch
            dr3 = (a*patch_res)**3
            
            # Calculate the integral of the scalar quantity over the volume
            
            masked = np.where(mask, field[p], 0)
            
            if vol == True: 
                masked1 = np.where(mask, ones_array[p], 0)
            
            if vector_levels[p] < max_refined_level:
                integral += np.sum(masked*cr0amr[p]*solapst[p])*dr3
                if vol == True:
                    volume_fi += np.sum(masked1*cr0amr[p]*solapst[p])*dr3
                    volume_co += np.sum(masked1*cr0amr[p]*solapst[p])*(patch_res)**3
            elif vector_levels[p] == max_refined_level:
                integral += np.sum(masked*solapst[p])*dr3
                if vol == True:
                    volume_fi += np.sum(masked1*solapst[p])*dr3
                    volume_co += np.sum(masked1*solapst[p])*(patch_res)**3
    
    integral = units * integral
    
    if verbose:
        print('Total integrated field: ' + str(integral))
    
    if vol == True:
        return integral, volume_fi, volume_co
    else:
        return integral
    
# def vol_integral_david(field, units, a0, zeta, cr0amr, solapst, npatch, patchrx, patchry, patchrz, patchnx, patchny, patchnz, size, nmax, coords, rad, max_refined_level=1000, kept_patches=None, vol=False, verbose=False):
#     """
#     Given a scalar field and a sphere defined with a center (x,y,z) and a radious together with the patch structure, returns the volumetric integral of the field along the sphere.

#     Args:
#         - field: scalar field to be integrated
#         - units: change of units factor to be multiplied by the final integral if one wants physical units
#         - a0: scale factor at the initial time with the units of the simulation
#         - zeta: redshift of the simulation snap to calculate the scale factor
#         - cr0amr: AMR maximum refinement factor (only the maximally resolved cells are considered)
#         - solapst: AMR overlap factor (only the maximally resolved cells are considered)
#         - npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
#         - patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
#         - patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
#         - size: comoving size of the simulation box
#         - nmax: number of cells in the coarsest resolution level
#         - coords: center of the sphere in a numpy array [x,y,z]
#         - rad: radius of the sphere
#         - max_refined_level: maximum refinement level that wants to be read. Subsequent refinements will be skipped. (int)
#         - kept_patches: boolean array to select the patches to be considered in the integration. True if the patch is kept, False if not. If None, all patches are kept.
#         - vol: if True, returns the volume of integration in phiysical and comoving units
#         - verbose: if True, prints the integral being calculated

#     Returns:
#         - integral: volumetric integral of the field along the sphere
    
#     Author: Marco José Molina Pradillo
#     """
#         X, Y, Z = tools_xyz.compute_position_fields(grid_patchnx[i+j], grid_patchny[i+j], grid_patchnz[i+j], grid_patchrx[i+j], grid_patchry[i+j], grid_patchrz[i+j], grid_npatch[i+j], size[0], nmax, ncores=1, kept_patches=clus_kp[i+j])

#         integral[i+j] = (1/2)*units*sum([qi[(xi-coords[0])**2 + (yi-coords[1])**2 + (zi-coords[2])**2 <= Rad**2].sum() * ((a0 / (1 + grid_zeta[i+j]))*size[0]/nmax/2**li)**3 if ki else 0 for xi,yi,zi,qi,li,ki in 
#                                             zip(X, Y, Z, tools.clean_field(field[i+j], clus_cr0amr[i+j], clus_solapst[i+j], grid_npatch[i+j]), vector_levels[i+j], clus_kp[i+j])])

#         print(integral[i+j])

        
@numba.njit(fastmath = True)
def patch_contribution(coords,rad,x0,y0,z0,nx,ny,nz,patch_res,cr0amr,solapst,field):
    patch_integral = 0.
    for ix in range(nx):
        for iy in range(ny):
            for iz in range(nz):
                x = x0 + ix*patch_res
                y = y0 + iy*patch_res
                z = z0 + iz*patch_res
                if (x-coords[0])**2 + (y-coords[1])**2 + (z-coords[2])**2 < rad**2:
                    patch_integral += field[ix,iy,iz]*cr0amr[ix,iy,iz]*solapst[ix,iy,iz]*(patch_res)**3

    return patch_integral

def vol_integral_oscar(field, units, a0, zeta, cr0amr, solapst, npatch, patchrx, patchry, patchrz, patchnx, patchny, patchnz, size, nmax, coords, rad, max_refined_level=1000, kept_patches=None):

    vector_levels = tools.create_vector_levels(npatch)
    
    dx = size/nmax
    
    a = a0 / (1 + zeta) # We compute the scale factor
    
    integral = 0
    for p in range(len(vector_levels)): # We run across all the patches we are interested in
        
        if kept_patches[p] != 0: # We only want the patches inside the region of interest
        
            patch_res = dx/(2**vector_levels[p])
            
            x0 = patchrx[p] - patch_res/2 #Center of the left-bottom-front cell
            y0 = patchry[p] - patch_res/2
            z0 = patchrz[p] - patch_res/2
            nx = patchnx[p]
            ny = patchny[p]
            nz = patchnz[p]

            if vector_levels[p] == max_refined_level:
                cr0amr_patch = np.ones((nx, ny, nz))

            else:
                cr0amr_patch = cr0amr[p]

            if p == 0:
                solapst_patch = np.ones((nx, ny, nz))
            else:
                solapst_patch = solapst[p]

            field_patch = field[p]
            patch_integral = patch_contribution(coords,rad,x0,y0,z0,nx,ny,nz,patch_res,cr0amr_patch,solapst_patch,field_patch)
            integral += patch_integral  

    return units*a**3*integral


# SECTION: find radius from R_\Delta radii definitions
def find_rDelta_eqn(r, Delta, background_density, clusrx, clusry, clusrz, density, cellsrx, cellsry, cellsrz, npatch,
                    size, nmax, verbose):
    if verbose:
        print('Evaluating at r={:.3f}'.format(r))
    m = mass_inside(r, clusrx, clusry, clusrz, density, cellsrx, cellsry, cellsrz, npatch, size, nmax)
    return m - (4 * np.pi / 3) * r ** 3 * background_density * Delta


def find_rDelta(Delta, zeta, clusrx, clusry, clusrz, density, cellsrx, cellsry, cellsrz,
                npatch, size, nmax, h, omega_m, rmin=0.1, rmax=6, rtol=1e-3, verbose=False):
    """
    Finds the value (in Mpc) of r_\Delta, the radius enclosing a mean overdensity (of the DM field, by default) equal
    to Delta times the background density of the universe. By default, it uses the Brent method.

    Args:
        Delta: value of the desired overdensity
        zeta: current redshift
        clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
        density: DM density field, already cleaned from refinements and overlaps (but not masked!)
        cellsrx, cellsry, cellsrz: position fields
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level
        h: dimensionless Hubble constant
        omega_m: matter density parameter
        rmin, rmax: starting two points (M(r) - Delta rho_B 4pi/3 r^3) must change sign
        verbose: if True, prints the patch being opened at a time

    Returns:
        The value of r_\Delta
    """
    background_density = cosmo_tools.background_density(h, omega_m, zeta)
    args = (Delta, background_density, clusrx, clusry, clusrz, density, cellsrx, cellsry, cellsrz, npatch,
            size, nmax, verbose)
    try:
        rDelta = optimize.brentq(find_rDelta_eqn, rmin, rmax, args=args, xtol=rtol)
    except ValueError:
        if verbose:
            print('Something might be wrong with this one... Might need further checking.')
        return float('nan')
    if verbose:
        print('Converged!')
    return rDelta


# SECTION: kinematical quantities: angular momenta, shape tensor, etc.
def angular_momentum_cells(cellsrx, cellsry, cellsrz, vx, vy, vz, cellsm, inside):
    """
    Computes the angular momentum of a matter distribution

    Args:
        cellsrx, cellsry, cellsrz: (recentered) position fields of the cells
        vx, vy, vz: velocity fields
        m: mass inside the cells
        inside: field containing True if the cell is inside the considered system, false otherwise. Must be
                cleaned from refinements and overlaps.

    Returns:
        The angular momentum of the given material component (gas). The three components are returned in a tuple.

    """
    Lx = sum([(cellm * (celly * cellvz - cellz * cellvy) * ins).sum() for cellm, celly, cellz, cellvy, cellvz, ins in
              zip(cellsm, cellsry, cellsrz, vy, vz, inside)])
    Ly = sum([(cellm * (cellz * cellvx - cellx * cellvz) * ins).sum() for cellm, cellx, cellz, cellvx, cellvz, ins in
              zip(cellsm, cellsrx, cellsrz, vx, vz, inside)])
    Lz = sum([(cellm * (cellx * cellvy - celly * cellvx) * ins).sum() for cellm, cellx, celly, cellvx, cellvy, ins in
              zip(cellsm, cellsrx, cellsry, vx, vy, inside)])

    return Lx, Ly, Lz


def shape_tensor_cells(cellsrx, cellsry, cellsrz, cellsm, inside):
    """
    Computes the shape tensor of a matter distribution

    Args:
        cellsrx, cellsry, cellsrz: (recentered) position fields of the cells
        cellsm: mass inside the cells
        inside: field containing True if the cell is inside the considered system, false otherwise. Must be
                cleaned from refinements and overlaps.

    Returns:
        Shape tensor as a 3x3 matrix
    """
    mass_tot_inside = sum([(m * ins).sum() for m, ins in zip(cellsm, inside)])

    Sxx = sum([(m * x ** 2 * ins).sum() for m, x, ins in zip(cellsm, cellsrx, inside)]) / mass_tot_inside
    Syy = sum([(m * y ** 2 * ins).sum() for m, y, ins in zip(cellsm, cellsry, inside)]) / mass_tot_inside
    Szz = sum([(m * z ** 2 * ins).sum() for m, z, ins in zip(cellsm, cellsrz, inside)]) / mass_tot_inside
    Sxy = sum([(m * x * y * ins).sum() for m, x, y, ins in zip(cellsm, cellsrx, cellsry, inside)]) / mass_tot_inside
    Sxz = sum([(m * x * z * ins).sum() for m, x, z, ins in zip(cellsm, cellsrx, cellsrz, inside)]) / mass_tot_inside
    Syz = sum([(m * y * z * ins).sum() for m, y, z, ins in zip(cellsm, cellsry, cellsrz, inside)]) / mass_tot_inside

    return np.array([[Sxx, Sxy, Sxz], [Sxy, Syy, Syz], [Sxz, Syz, Szz]])


def ellipsoidal_shape_cells(cellsrx, cellsry, cellsrz, cellsm, r, tol=1e-3, maxiter=100, preserve='major',
                            verbose=False):
    """
    Finds the shape of a matter distribution (eigenvalues and eigenvectors of the intertia tensor) by using
    the iterative method in Zemp et al (2011).

    Args:
        cellsrx, cellsry, cellsrz: (recentered) position fields of the cells
        cellsm: mass inside the cells
        r: initial radius (will be kept as the major semi-axis length
        tol: relative error allowed to the quotient between semiaxes
        maxiter: maximum number of allowed iterations
        preserve: which quantity to preserve when changing the axes (could be 'major', 'intermediate', 'minor' or
                    'volume')

    Returns:
        List of semiaxes lengths and list of eigenvectors.

    """
    inside = [cellrx ** 2 + cellry ** 2 + cellrz ** 2 < r ** 2 for cellrx, cellry, cellrz in zip(cellsrx, cellsry,
                                                                                                 cellsrz)]
    shapetensor = shape_tensor_cells(cellsrx, cellsry, cellsrz, cellsm, inside)
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
        # rotating the cells coordinates
        cellsrxprev = cellsrx
        cellsryprev = cellsry
        cellsrzprev = cellsrz
        cellsrx = [cellrx * u_xtilde[0] + cellry * u_xtilde[1] +
                   cellrz * u_xtilde[2] for cellrx, cellry, cellrz in zip(cellsrxprev, cellsryprev, cellsrzprev)]
        cellsry = [cellrx * u_ytilde[0] + cellry * u_ytilde[1] +
                   cellrz * u_ytilde[2] for cellrx, cellry, cellrz in zip(cellsrxprev, cellsryprev, cellsrzprev)]
        cellsrz = [cellrx * u_ztilde[0] + cellry * u_ztilde[1] +
                   cellrz * u_ztilde[2] for cellrx, cellry, cellrz in zip(cellsrxprev, cellsryprev, cellsrzprev)]
        del cellsrxprev, cellsryprev, cellsrzprev

        # compute the new 'inside' cells, considering the ellipsoidal shape, keeping the major semiaxis length
        # note that the major semiaxis corresponds to the ztilde component (by construction, see diagonalize_ascending)
        inside = [(cellrx / semiax_x) ** 2 + (cellry / semiax_y) ** 2 +
                  (cellrz / semiax_z) ** 2 < 1 for cellrx, cellry, cellrz in zip(cellsrx, cellsry, cellsrz)]

        # keep track of the previous axisratios
        axisratio1_prev = axisratio1
        axisratio2_prev = axisratio2

        # diagonalize the new cell selection
        shapetensor = shape_tensor_cells(cellsrx, cellsry, cellsrz, cellsm, inside)

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


