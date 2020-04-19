"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

tools_xyz module
Contains several useful functions that other modules might need. Here we have the (new) versions, which make more
intensive use of computing x,y,z fields (much faster, but more memory consuming)

Created by David Vallés
"""

#  Last update on 19/4/20 17:58

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

# numpy
import numpy as np

from multiprocessing import Pool

from masclet_framework import cosmo_tools, units, tools

from scipy import optimize


# FUNCTIONS DEFINED IN THIS MODULE

# compute the position fields
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
    nx, ny, nz, rx, ry, rz, level, size, nmax = args
    cellsize = size / nmax / 2 ** level
    first_x = rx - cellsize / 2
    first_y = ry - cellsize / 2
    first_z = rz - cellsize / 2
    patch_x = np.zeros((nx, ny, nz))
    patch_y = np.zeros((nx, ny, nz))
    patch_z = np.zeros((nx, ny, nz))
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                patch_x[i, j, k] = first_x + i * cellsize
                patch_y[i, j, k] = first_y + j * cellsize
                patch_z[i, j, k] = first_z + k * cellsize
    return patch_x, patch_y, patch_z


def compute_position_fields(patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, size, nmax, ncores=1):
    """
    Returns 3 fields (as usually defined) containing the x, y and z position for each of our cells centres.
    Args:
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        (and Y and Z)
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level
        ncores:

    Returns:
        3 fields as described above
    """
    levels = tools.create_vector_levels(npatch)
    with Pool(ncores) as p:
        positions = p.map(compute_position_field_onepatch,
                          [(patchnx[ipatch], patchny[ipatch], patchnz[ipatch], patchrx[ipatch], patchry[ipatch],
                            patchrz[ipatch], levels[ipatch], size, nmax) for ipatch in range(len(patchnx))])

    cellsrx = [p[0] for p in positions]
    cellsry = [p[1] for p in positions]
    cellsrz = [p[2] for p in positions]

    return cellsrx, cellsry, cellsrz


# other functions
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


def mask_sphere(R, clusrx, clusry, clusrz, cellsrx, cellsry, cellsrz):
    """
    Returns a "field", which contains all patches as usual. True means the cell must be considered, False otherwise.
    If a patch has all "False", an array is not ouputted, but a False is, instead.

    Args:
        R: radius of the considered sphere
        clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
        cellsrx, cellsry, cellsrz: position fields

    Returns:
        Field containing the mask as described.
    """
    mask = [(cx - clusrx) ** 2 + (cy - clusry) ** 2 + (cz - clusrz) ** 2 < R ** 2 for cx, cy, cz in
            zip(cellsrx, cellsry, cellsrz)]

    return mask


def clean_field(field, cr0amr, solapst, npatch, up_to_level=1000):
    """
    Receives a field (with its refinement patches) and, using the cr0amr and solapst variables, returns the field
    having "cleaned" for refinements and overlaps. The user can specify the level of refinement required. This last
    level will be cleaned of overlaps, but not of refinements!

    Args:
        field: field to be cleaned
        cr0amr: field containing the refinements of the grid (1: not refined; 0: refined)
        solapst: field containing the overlaps (1: keep; 0: not keep)
        npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        up_to_level: specify if only cleaning until certain level is desired

    Returns:
        "Clean" version of the field, with the same structure.

    """
    levels = tools.create_vector_levels(npatch)
    up_to_level = min(up_to_level, levels.max())

    cleanfield = [field[0] * cr0amr[0]]

    for level in range(1, up_to_level):
        for ipatch in range(sum(npatch[0:level]) + 1, sum(npatch[0:level + 1]) + 1):
            cleanfield.append(field[ipatch] * cr0amr[ipatch] * solapst[ipatch])

    # last level: no refinements
    for ipatch in range(sum(npatch[0:up_to_level]) + 1, sum(npatch[0:up_to_level + 1]) + 1):
        cleanfield.append(field[ipatch] * solapst[ipatch])

    return cleanfield


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
                      cellsrz, npatch, size, nmax, verbose=False):
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
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level
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

        sum_field_vw = sum([(fvw * sm).sum() for fvw, sm in zip(field_vw, shell_mask)])
        sum_vw = sum([(sm * cv).sum() for sm, cv in zip(shell_mask, cell_volume)])

        profile.append(sum_field_vw / sum_vw)

    return bin_centers, np.asarray(profile)


def several_radial_profiles_vw(fields, clusrx, clusry, clusrz, rmin, rmax, nbins, logbins, cellsrx, cellsry,
                               cellsrz, npatch, size, nmax, verbose=False):
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
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level
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
        sum_vw = sum([(sm * cv).sum() for sm, cv in zip(shell_mask, cell_volume)])

        profile_thisr = [(sum([(fvw * sm).sum() for fvw, sm in zip(field_vw, shell_mask)]) / sum_vw) for field_vw in
                         fields_vw]

        profiles.append(profile_thisr)

    profiles = np.asarray(profiles)
    profiles_split = tuple([profiles[:, i] for i in range(profiles.shape[1])])

    return bin_centers, profiles_split


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


def uniform_grid_zoom(field, box_limits, up_to_level, npatch, cr0amr, solapst, cellsrx, cellsry, cellsrz, patchnx,
                      patchny, patchnz, patchrx, patchry, patchrz, size, nmax, verbose=False):
    """
    Builds a uniform grid, zooming on a box specified by box_limits, at level up_to_level, containing the most refined
    data at each region.

    DEPRECATED

    Args:
        field: the ¡uncleaned! field which wants to be represented
        box_limits: a tuple in the form (xmin, xmax, ymin, ymax, zmin, zmax). Limits should correspond to l=0 cell
                    boundaries
        up_to_level: level up to which the fine grid wants to be obtained
        npatch: number of patches in each level, starting in l=0
        cr0amr: field containing the refinements of the grid (1: not refined; 0: refined)
        solapst: field containing the overlaps (1: keep; 0: not keep)
        cellsrx, cellsry, cellsrz: position fields
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        (and Y and Z)
        size: comoving size of the simulation box
        nmax: cells at base level
        verbose: if True, prints the patch being opened at a time

    Returns:
        Uniform grid as described
    """
    # clean the field up to the requiered vertex
    field_clean = tools.clean_field(field, cr0amr, solapst, npatch, up_to_level=up_to_level)

    # geometry
    bxmin = box_limits[0]
    bxmax = box_limits[1]
    bymin = box_limits[2]
    bymax = box_limits[3]
    bzmin = box_limits[4]
    bzmax = box_limits[5]

    uniform_nx = int(round(nmax * (bxmax - bxmin) / size * 2 ** up_to_level))
    uniform_ny = int(round(nmax * (bymax - bymin) / size * 2 ** up_to_level))
    uniform_nz = int(round(nmax * (bzmax - bzmin) / size * 2 ** up_to_level))
    uniform = np.zeros((uniform_nx, uniform_ny, uniform_nz))

    uniform_cellsize = size / nmax / 2 ** up_to_level

    # uniform_verticesrx = np.zeros((uniform_nx, uniform_ny, uniform_nz))
    # uniform_verticesry = np.zeros((uniform_nx, uniform_ny, uniform_nz))
    # uniform_verticesrz = np.zeros((uniform_nx, uniform_ny, uniform_nz))
    uniform_cellsrx = np.zeros(uniform_nx + 1)
    uniform_cellsry = np.zeros(uniform_ny + 1)
    uniform_cellsrz = np.zeros(uniform_nz + 1)
    uniform_verticesrx = np.zeros(uniform_nx + 1)
    uniform_verticesry = np.zeros(uniform_ny + 1)
    uniform_verticesrz = np.zeros(uniform_nz + 1)

    for i in range(uniform_nx + 1):
        uniform_verticesrx[i] = bxmin + i * uniform_cellsize
        uniform_cellsrx[i] = bxmin + (i + 0.5) * uniform_cellsize
    for j in range(uniform_ny + 1):
        uniform_verticesry[j] = bymin + j * uniform_cellsize
        uniform_cellsry[j] = bymin + (j + 0.5) * uniform_cellsize
    for k in range(uniform_nz + 1):
        uniform_verticesrz[k] = bzmin + k * uniform_cellsize
        uniform_cellsrz[k] = bzmin + (k + 0.5) * uniform_cellsize

    levels = tools.create_vector_levels(npatch)
    # up_to_level_patches = npatch[0:up_to_level + 1].sum()
    # relevantpatches = tools.which_patches_inside_box(box_limits, patchnx, patchny, patchnz, patchrx, patchry, patchrz,
    #                                                 npatch, size, nmax)
    # relevantpatches = [i for i in relevantpatches if i <= npatch[0:up_to_level + 1].sum()]
    max_relevantpatches = npatch[0:up_to_level + 1].sum()

    # we can simplify things if we only keep cellsrx, cellsry, cellsrz as 1D arrays
    cells_verticesrx = [
        np.append(cellrx[:, 0, 0] - size / nmax / 2 ** (l + 1), cellrx[-1, 0, 0] + size / nmax / 2 ** (l + 1)) for
        cellrx, l in zip(cellsrx, levels)]
    cells_verticesry = [
        np.append(cellry[0, :, 0] - size / nmax / 2 ** (l + 1), cellry[0, -1, 0] + size / nmax / 2 ** (l + 1)) for
        cellry, l in zip(cellsry, levels)]
    cells_verticesrz = [
        np.append(cellrz[0, 0, :] - size / nmax / 2 ** (l + 1), cellrz[0, 0, -1] + size / nmax / 2 ** (l + 1)) for
        cellrz, l in zip(cellsrz, levels)]

    for i in range(uniform_nx):
        slice_limits = [uniform_verticesrx[i], uniform_verticesrx[i + 1], bymin, bymax, bzmin, bzmax]
        relevantpatches = tools.which_patches_inside_box(slice_limits, patchnx, patchny, patchnz, patchrx, patchry,
                                                         patchrz, npatch, size, nmax)
        relevantpatches = [i for i in relevantpatches if i <= max_relevantpatches]
        if verbose:
            print('Slice {} of {}. {} patches to use.'.format(i, uniform_nx, len(relevantpatches)))
        for j in range(uniform_ny):
            for k in range(uniform_nz):
                for ipatch in relevantpatches:
                    cell_this_x = (cells_verticesrx[ipatch][0:-1] < uniform_cellsrx[i]) * (
                            uniform_cellsrx[i] < cells_verticesrx[ipatch][1:])
                    cell_this_y = (cells_verticesry[ipatch][0:-1] < uniform_cellsry[j]) * (
                            uniform_cellsry[j] < cells_verticesry[ipatch][1:])
                    cell_this_z = (cells_verticesrz[ipatch][0:-1] < uniform_cellsrz[k]) * (
                            uniform_cellsrz[k] < cells_verticesrz[ipatch][1:])
                    cell_this_x = cell_this_x.nonzero()[0]
                    cell_this_y = cell_this_y.nonzero()[0]
                    cell_this_z = cell_this_z.nonzero()[0]
                    if len(cell_this_x) * len(cell_this_y) * len(cell_this_z) > 0:
                        uniform[i, j, k] += field_clean[ipatch][cell_this_x[0], cell_this_y[0], cell_this_z[0]]
    # repensar esto. es muy lento.

    # Esto es MUY FÁCIL DE PARALELIZAR: SLICE A SLICE DEL BUCLE EN I!!!!!! (si va lento)
    return uniform


def uniform_grid_zoom_slice(args):
    i, field_clean, uniform_verticesrx, uniform_verticesry, uniform_verticesrz, bymin, bymax, bzmin, bzmax, \
    patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, size, nmax, max_relevantpatches, uniform_nx, \
    uniform_ny, uniform_nz, cells_verticesrx, cells_verticesry, cells_verticesrz, uniform_cellsrx, \
    uniform_cellsry, uniform_cellsrz, verbose = args

    slice_limits = [uniform_verticesrx[i], uniform_verticesrx[i + 1], bymin, bymax, bzmin, bzmax]
    relevantpatches = tools.which_patches_inside_box(slice_limits, patchnx, patchny, patchnz, patchrx, patchry,
                                                     patchrz, npatch, size, nmax)
    relevantpatches = [ipatch for ipatch in relevantpatches if ipatch <= max_relevantpatches]

    uniform_slice = np.zeros((uniform_ny, uniform_nz))

    if verbose:
        print('Slice {} of {}. {} patches to use.'.format(i, uniform_nx, len(relevantpatches)))
    for j in range(uniform_ny):
        for k in range(uniform_nz):
            for ipatch in relevantpatches:
                cell_this_x = (cells_verticesrx[ipatch][0:-1] < uniform_cellsrx[i]) * (
                        uniform_cellsrx[i] < cells_verticesrx[ipatch][1:])
                cell_this_y = (cells_verticesry[ipatch][0:-1] < uniform_cellsry[j]) * (
                        uniform_cellsry[j] < cells_verticesry[ipatch][1:])
                cell_this_z = (cells_verticesrz[ipatch][0:-1] < uniform_cellsrz[k]) * (
                        uniform_cellsrz[k] < cells_verticesrz[ipatch][1:])
                cell_this_x = cell_this_x.nonzero()[0]
                cell_this_y = cell_this_y.nonzero()[0]
                cell_this_z = cell_this_z.nonzero()[0]
                if len(cell_this_x) * len(cell_this_y) * len(cell_this_z) > 0:
                    uniform_slice[j, k] += field_clean[ipatch][cell_this_x[0], cell_this_y[0], cell_this_z[0]]

    return uniform_slice


def uniform_grid_zoom_parallel(field, box_limits, up_to_level, npatch, cr0amr, solapst, cellsrx, cellsry, cellsrz,
                               patchnx, patchny, patchnz, patchrx, patchry, patchrz, size, nmax, ncores=1,
                               verbose=False):
    """
    Builds a uniform grid, zooming on a box specified by box_limits, at level up_to_level, containing the most refined
    data at each region.

    Parallel version

    DEPRECATED

    Args:
        field: the ¡uncleaned! field which wants to be represented
        box_limits: a tuple in the form (xmin, xmax, ymin, ymax, zmin, zmax). Limits should correspond to l=0 cell
                    boundaries
        up_to_level: level up to which the fine grid wants to be obtained
        npatch: number of patches in each level, starting in l=0
        cr0amr: field containing the refinements of the grid (1: not refined; 0: refined)
        solapst: field containing the overlaps (1: keep; 0: not keep)
        cellsrx, cellsry, cellsrz: position fields
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        (and Y and Z)
        size: comoving size of the simulation box
        nmax: cells at base level
        ncores: number of cores for the parallelization
        verbose: if True, prints the patch being opened at a time

    Returns:
        Uniform grid as described
    """
    # clean the field up to the requiered vertex
    field_clean = tools.clean_field(field, cr0amr, solapst, npatch, up_to_level=up_to_level)

    # geometry
    bxmin = box_limits[0]
    bxmax = box_limits[1]
    bymin = box_limits[2]
    bymax = box_limits[3]
    bzmin = box_limits[4]
    bzmax = box_limits[5]

    uniform_nx = int(round(nmax * (bxmax - bxmin) / size * 2 ** up_to_level))
    uniform_ny = int(round(nmax * (bymax - bymin) / size * 2 ** up_to_level))
    uniform_nz = int(round(nmax * (bzmax - bzmin) / size * 2 ** up_to_level))
    uniform = np.zeros((uniform_nx, uniform_ny, uniform_nz))

    uniform_cellsize = size / nmax / 2 ** up_to_level

    # uniform_verticesrx = np.zeros((uniform_nx, uniform_ny, uniform_nz))
    # uniform_verticesry = np.zeros((uniform_nx, uniform_ny, uniform_nz))
    # uniform_verticesrz = np.zeros((uniform_nx, uniform_ny, uniform_nz))
    uniform_cellsrx = np.zeros(uniform_nx + 1)
    uniform_cellsry = np.zeros(uniform_ny + 1)
    uniform_cellsrz = np.zeros(uniform_nz + 1)
    uniform_verticesrx = np.zeros(uniform_nx + 1)
    uniform_verticesry = np.zeros(uniform_ny + 1)
    uniform_verticesrz = np.zeros(uniform_nz + 1)

    for i in range(uniform_nx + 1):
        uniform_verticesrx[i] = bxmin + i * uniform_cellsize
        uniform_cellsrx[i] = bxmin + (i + 0.5) * uniform_cellsize
    for j in range(uniform_ny + 1):
        uniform_verticesry[j] = bymin + j * uniform_cellsize
        uniform_cellsry[j] = bymin + (j + 0.5) * uniform_cellsize
    for k in range(uniform_nz + 1):
        uniform_verticesrz[k] = bzmin + k * uniform_cellsize
        uniform_cellsrz[k] = bzmin + (k + 0.5) * uniform_cellsize

    levels = tools.create_vector_levels(npatch)
    # up_to_level_patches = npatch[0:up_to_level + 1].sum()
    # relevantpatches = tools.which_patches_inside_box(box_limits, patchnx, patchny, patchnz, patchrx, patchry, patchrz,
    #                                                 npatch, size, nmax)
    # relevantpatches = [i for i in relevantpatches if i <= npatch[0:up_to_level + 1].sum()]
    max_relevantpatches = npatch[0:up_to_level + 1].sum()

    # we can simplify things if we only keep cellsrx, cellsry, cellsrz as 1D arrays
    cells_verticesrx = [
        np.append(cellrx[:, 0, 0] - size / nmax / 2 ** (l + 1), cellrx[-1, 0, 0] + size / nmax / 2 ** (l + 1)) for
        cellrx, l in zip(cellsrx, levels)]
    cells_verticesry = [
        np.append(cellry[0, :, 0] - size / nmax / 2 ** (l + 1), cellry[0, -1, 0] + size / nmax / 2 ** (l + 1)) for
        cellry, l in zip(cellsry, levels)]
    cells_verticesrz = [
        np.append(cellrz[0, 0, :] - size / nmax / 2 ** (l + 1), cellrz[0, 0, -1] + size / nmax / 2 ** (l + 1)) for
        cellrz, l in zip(cellsrz, levels)]

    with Pool(ncores) as p:
        uniform_slices = p.map(uniform_grid_zoom_slice, [(i, field_clean, uniform_verticesrx, uniform_verticesry,
                                                          uniform_verticesrz, bymin, bymax, bzmin, bzmax, patchnx,
                                                          patchny, patchnz, patchrx, patchry, patchrz, npatch, size,
                                                          nmax, max_relevantpatches, uniform_nx, uniform_ny, uniform_nz,
                                                          cells_verticesrx, cells_verticesry, cells_verticesrz,
                                                          uniform_cellsrx, uniform_cellsry,
                                                          uniform_cellsrz, verbose) for i in range(uniform_nx)])

    return np.moveaxis(np.dstack(uniform_slices), [0, 1, 2], [1, 2, 0])


def angular_momentum_particles(x, y, z, vx, vy, vz, m, inside):
    """
    Computes the angular momentum of a list of particles

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
