"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

tools module
Contains several useful functions that other modules might need

Created by David Vallés
"""

#  Last update on 8/4/20 9:57

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

# numpy
import numpy as np

from multiprocessing import Pool

from masclet_framework import cosmo_tools, units

from scipy import optimize


# FUNCTIONS DEFINED IN THIS MODULE


def create_vector_levels(npatch):
    """
    Creates a vector containing the level for each patch. Nothing really important, just for ease

    Args:
        npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)

    Returns:
        numpy array containing the level for each patch

    """
    vector = [[0]] + [[i + 1] * x for (i, x) in enumerate(npatch[1:])]
    vector = [item for sublist in vector for item in sublist]
    return np.array(vector)


def find_absolute_grid_position(ipatch, npatch, patchx, patchy, patchz, pare):
    """
    Given a patch in a certain iteration, finds the grid (natural) coordinates of its left corner.
    Written as a recursive function this time.

    Args:
        ipatch: number of the patch (in the given iteration)
        npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        patchx: vector containing the X-position of the leftmost corner of all the patches (numpy vector of integers)
        patchy: vector containing the Y-position of the leftmost corner of all the patches (numpy vector of integers)
        patchz: vector containing the Z-position of the leftmost corner of all the patches (numpy vector of integers)
        pare: vector containing the ipatch of each patch progenitor (numpy vector of integers)

    Returns:
        tuple with the patches left-corner NX, NY, NZ values

    """
    level = create_vector_levels(npatch)[ipatch]
    if ipatch < patchx.size:
        if level == 1:
            return patchx[ipatch], patchy[ipatch], patchz[ipatch]
        else:
            parevalues = find_absolute_grid_position(pare[ipatch], npatch, patchx, patchy, patchz, pare)
            return (patchx[ipatch] / 2 ** (level - 1) + parevalues[0],
                    patchy[ipatch] / 2 ** (level - 1) + parevalues[1],
                    patchz[ipatch] / 2 ** (level - 1) + parevalues[2])
    else:
        raise ValueError('Provide a valid patchnumber (ipatch)')


def find_absolute_real_position(ipatch, side, nmax, npatch, patchx, patchy, patchz, pare):
    """
    Given a patch in a certain iteration, finds the real coordinates of its left corner with no numerical error.
    This function depends on find_absolute_grid_position, which must be also loaded.

    Args:
        ipatch: number of the patch (in the given iteration) (int)
        side: side of the simulation box in the chosen units (typically Mpc or kpc) (float)
        nmax: number of cells along each directions at level l=0 (can be loaded from masclet_parameters; NMAX = NMAY =
        = NMAZ is assumed) (int)
        npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        patchx: vector containing the X-position of the leftmost corner of all the patches (numpy vector of integers)
        patchy: vector containing the Y-position of the leftmost corner of all the patches (numpy vector of integers)
        patchz: vector containing the Z-position of the leftmost corner of all the patches (numpy vector of integers)
        pare: vector containing the ipatch of each patch progenitor (numpy vector of integers)

    Returns:
    numpy array with the patches left corner X, Y and Z values (assuming box centered in 0; same units than SIDE)

    """
    return (np.asarray(
        find_absolute_grid_position(ipatch, npatch, patchx, patchy, patchz, pare)) - nmax / 2) * side / nmax


def patch_vertices(level, nx, ny, nz, rx, ry, rz, size, nmax):
    """
    Returns, for a given patch, the comoving coordinates of its 8 vertices.

    Args:
        level: refinement level of the given patch
        nx, ny, nz: extension of the patch (in cells at level n)
        rx, ry, rz: comoving coordinates of the center of the leftmost cell of the patch
        size: comoving box side (preferred length units)
        nmax: cells at base level

    Returns:
        List containing 8 tuples, each one containing the x, y, z coordinates of the vertex.

    """

    cellsize = size / nmax / 2 ** level

    leftmost_x = rx - cellsize
    leftmost_y = ry - cellsize
    leftmost_z = rz - cellsize

    vertices = []

    for i in range(2):
        for j in range(2):
            for k in range(2):
                x = leftmost_x + i * nx * cellsize
                y = leftmost_y + j * ny * cellsize
                z = leftmost_z + k * nz * cellsize

                vertices.append((x, y, z))

    return vertices


def patch_is_inside_sphere(R, clusrx, clusry, clusrz, level, nx, ny, nz, rx, ry, rz, size, nmax):
    """

    Args:
        R: radius of the considered sphere
        clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
        level: refinement level of the given patch
        nx, ny, nz: extension of the patch (in cells at level n)
        rx, ry, rz: comoving coordinates of the center of the leftmost cell of the patch
        size: comoving box side (preferred length units)
        nmax: cells at base level

    Returns:
        Returns True if the patch should contain cells within a sphere of radius r of the (clusrx, clusry, clusrz)
        point; False otherwise.

    """
    isinside = False

    vertices = patch_vertices(level, nx, ny, nz, rx, ry, rz, size, nmax)
    xmin = vertices[0][0]
    ymin = vertices[0][1]
    zmin = vertices[0][2]
    xmax = vertices[-1][0]
    ymax = vertices[-1][1]
    zmax = vertices[-1][2]

    if xmin < clusrx < xmax and ymin < clusry < ymax and zmin < clusrz < zmax:
        return True

    cell_l0_size = size / nmax
    max_side = max([nx, ny, nz]) * cell_l0_size / 2 ** level
    upper_bound_squared = R ** 2 + max_side ** 2 / 4
    for vertex in vertices:
        distance_squared = (vertex[0] - clusrx) ** 2 + (vertex[1] - clusry) ** 2 + (vertex[2] - clusrz) ** 2
        if distance_squared <= upper_bound_squared:
            isinside = True
            break

    return isinside


def which_patches_inside_sphere(R, clusrx, clusry, clusrz, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch,
                                size, nmax):
    """
    Finds which of the patches will contain cells within a radius r of a certain point (clusrx, clusry, clusrz) being
    its comoving coordinates.

    Args:
        R: radius of the considered sphere
        clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        (and Y and Z)
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level

    Returns:
        List containing the ipatch of the patches which should contain cells inside the considered radius.

    """
    levels = create_vector_levels(npatch)
    which_ipatch = [0]
    for ipatch in range(1, len(patchnx)):
        if patch_is_inside_sphere(R, clusrx, clusry, clusrz, levels[ipatch], patchnx[ipatch], patchny[ipatch],
                                  patchnz[ipatch],
                                  patchrx[ipatch], patchry[ipatch], patchrz[ipatch], size, nmax):
            which_ipatch.append(ipatch)
    return which_ipatch


def which_cells_inside_sphere(R, clusrx, clusry, clusrz, level, nx, ny, nz, rx, ry, rz, size, nmax):
    """
    Finds, for a given patch, which cells are inside a sphere of radius R centered on (clusrx, clusry, clusrz)

    Args:
        R: radius of the considered sphere
        clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
        level: refinement level of the given patch
        nx, ny, nz: extension of the patch (in cells at level n)
        rx, ry, rz: comoving coordinates of the center of the leftmost cell of the patch, at level l-1
        size: comoving box side (preferred length units)
        nmax: cells at base level

    Returns:
        isinside: numpy bool array, the size of the patch, containing for each cell 1 if inside and 0 otherwise
    """

    isinside = np.zeros((nx, ny, nz), dtype=bool)

    cell_l0_size = size / nmax
    cell_size = cell_l0_size / 2 ** level
    Rsquared = R ** 2

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                # index-1 ==> recall that rx, ry, rz are the center of the leftmost PARENT cell!
                x = rx + (i - 0.5) * cell_size
                y = ry + (j - 0.5) * cell_size
                z = rz + (k - 0.5) * cell_size
                isinside[i, j, k] = ((x - clusrx) ** 2 + (y - clusry) ** 2 + (z - clusrz) ** 2 <= Rsquared)

    return isinside


def mask_sphere(R, clusrx, clusry, clusrz, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, size, nmax,
                which_patches, verbose=False):
    """
    Returns a "field", which contains all patches as usual. True means the cell must be considered, False otherwise.
    If a patch has all "False", an array is not ouputted, but a False is, instead.

    Non-parallel version!

    Args:
        R: radius of the considered sphere
        clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        (and Y and Z)
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level
        which_patches: patches to be open (as found by which_patches_inside_sphere)
        verbose: if True, prints the patch being opened at a time

    Returns:
        Field containing the mask as described.
    """
    levels = create_vector_levels(npatch)

    mask = []

    for ipatch in range(len(patchnx)):
        if verbose:
            print("Masking patch {}".format(ipatch))
        if ipatch not in which_patches:
            mask.append(False)
        else:
            mask.append(which_cells_inside_sphere(R, clusrx, clusry, clusrz, levels[ipatch], patchnx[ipatch],
                                                  patchny[ipatch], patchnz[ipatch], patchrx[ipatch], patchry[ipatch],
                                                  patchrz[ipatch], size, nmax))

    return mask


def mask_patch(args):
    """
    Given a patch and a sphere, returns a matrix containing True (means the cell must be considered), False otherwise.

    Args: tuple containing, in this order:
        R: radius of the considered sphere
        clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
        level: refinement level of the given patch
        nx, ny, nz: extension of the patch (in cells at level n)
        rx, ry, rz: comoving coordinates of the center of the leftmost cell of the patch
        size: comoving size of the simulation box
        nmax: cells at base level
        ipatch: number of the patch
        which_patches: patches to be open (as found by which_patches_inside_sphere)
        verbose: if True, prints the patch being opened at a time

        (R, clusrx, clusry, clusrz, level, nx, ny, nz, rx, ry, rz, size, nmax, ipatch, which_patches,
               verbose=False)
    Returns:
        Masked patch, as described above
    """
    R, clusrx, clusry, clusrz, level, nx, ny, nz, rx, ry, rz, size, nmax, ipatch, which_patches, verbose = args

    if verbose:
        print('Masking patch {}'.format(ipatch))
    if ipatch not in which_patches:
        return False
    else:
        return which_cells_inside_sphere(R, clusrx, clusry, clusrz, level, nx, ny, nz, rx, ry, rz, size, nmax)


def mask_sphere_parallel(R, clusrx, clusry, clusrz, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, size,
                         nmax,
                         which_patches, verbose=False, ncores=1):
    """
    Returns a "field", which contains all patches as usual. True means the cell must be considered, False otherwise.
    If a patch has all "False", an array is not ouputted, but a False is, instead.

    Parallel version.

    Args:
        R: radius of the considered sphere
        clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        (and Y and Z)
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level
        which_patches: patches to be open (as found by which_patches_inside_sphere)
        verbose: if True, prints the patch being opened at a time
        ncores: number of cores to perform the paralellization.

    Returns:
        Field containing the mask as described.
    """
    levels = create_vector_levels(npatch)

    with Pool(ncores) as p:
        mask = p.map(mask_patch, [(R, clusrx, clusry, clusrz, levels[ipatch], patchnx[ipatch], patchny[ipatch],
                                   patchnz[ipatch], patchrx[ipatch], patchry[ipatch], patchrz[ipatch], size, nmax,
                                   ipatch, which_patches, verbose) for ipatch in range(len(patchnx))])

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
    levels = create_vector_levels(npatch)
    up_to_level = min(up_to_level, levels.max())

    cleanfield = []
    if up_to_level == 0:
        cleanfield.append(field[0])
    else:
        cleanfield.append(field[0] * cr0amr[0])  # not overlap in l=0

    for level in range(1, up_to_level):
        for ipatch in range(sum(npatch[0:level]) + 1, sum(npatch[0:level + 1]) + 1):
            cleanfield.append(field[ipatch] * cr0amr[ipatch] * solapst[ipatch])

    # last level: no refinements
    for ipatch in range(sum(npatch[0:up_to_level]) + 1, sum(npatch[0:up_to_level + 1]) + 1):
        cleanfield.append(field[ipatch] * solapst[ipatch])

    return cleanfield


def patch_left_edge_comoving(rx, ry, rz, level, size, nmax):
    """
    Computes comoving coordinates of the left edge of a patch, given its rx, ry, rz.

    Args:
        rx, ry, rz: physical coordinate of the center of the leftmost parent cell (as read from patchrx, patchry, ...)
        level: level of the considered patch
        size: comoving side of the simulation box
        nmax: number of cells per dimension on the base level

    Returns:
        tuple containing the x, y, z physical coordinates of the patch left corner.
    """
    cellsize = size / nmax / 2 ** (level)
    x = rx - cellsize
    y = ry - cellsize
    z = rz - cellsize
    return x, y, z


def patch_left_edge_natural(rx, ry, rz, level, size, nmax):
    """
        Computes natural coordinates (0 being the left corner of the box, nmax being the right corner) of the left edge
        of a patch, given its rx, ry, rz.

        Args:
            rx, ry, rz: physical coordinate of the center of the leftmost parent cell (as read from patchrx, patchry, ...)
            level: level of the considered patch
            size: comoving side of the simulation box
            nmax: number of cells per dimension on the base level

        Returns:
            tuple containing the x, y, z physical coordinates of the patch left corner.
        """
    x, y, z = patch_left_edge_comoving(rx, ry, rz, level, size, nmax)

    xg = x * nmax / size + nmax / 2
    yg = y * nmax / size + nmax / 2
    zg = z * nmax / size + nmax / 2

    # avoid numerical error
    xg = round(xg * 2 ** (level - 1)) / 2 ** (level - 1)
    yg = round(yg * 2 ** (level - 1)) / 2 ** (level - 1)
    zg = round(zg * 2 ** (level - 1)) / 2 ** (level - 1)

    return xg, yg, zg


def uniform_grid(field, up_to_level, npatch, patchnx, patchny, patchnz, patchrx, patchry, patchrz, size, nmax,
                 verbose=False):
    """
    Builds a uniform grid at level up_to_level, containing the most refined data at each region.

    Args:
        field: field to be computed at the uniform grid. Must be already cleaned from refinements and overlaps (check
        clean_field() function).
        up_to_level: level up to which the fine grid wants to be obtained
        npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        (and Y and Z)
        size: comoving side of the simulation box
        nmax: cells at base level
        verbose: if True, prints the patch being opened at a time

    Returns:
        Uniform grid as described

    """
    uniform_size = nmax * 2 ** up_to_level
    uniform = np.zeros((uniform_size, uniform_size, uniform_size))

    reduction = 2 ** up_to_level
    for i in range(uniform_size):
        for j in range(uniform_size):
            for k in range(uniform_size):
                I = int(i / reduction)
                J = int(j / reduction)
                K = int(k / reduction)
                uniform[i, j, k] += field[0][I, J, K]

    levels = create_vector_levels(npatch)
    relevantpatches = npatch[0:up_to_level + 1].sum()

    for ipatch in range(1, relevantpatches + 1):
        if verbose:
            print('Covering patch {}'.format(ipatch))
        reduction = 2 ** (up_to_level - levels[ipatch])
        shift = np.array(patch_left_edge_natural(patchrx[ipatch], patchry[ipatch], patchrz[ipatch],
                                                 levels[ipatch], size, nmax)) * 2 ** up_to_level
        for i in range(patchnx[ipatch] * reduction):
            for j in range(patchny[ipatch] * reduction):
                for k in range(patchnz[ipatch] * reduction):
                    I = int(i / reduction)
                    J = int(j / reduction)
                    K = int(k / reduction)
                    uniform[int(shift[0]) + i, int(shift[1]) + j, int(shift[2]) + k] += field[ipatch][I, J, K]

    return uniform


def mass_inside(R, clusrx, clusry, clusrz, density, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, size,
                nmax, verbose=False, ncores=1):
    """
        Computes the mass inside a radius R sphere centered on (clusrx, clusry, clusrz), from the density field.
        Note that user can either supply:
            - Comoving density and comoving size (most common)
            - Physical density and physical size

        Parallel version.

        Args:
            R: radius of the considered sphere
            clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
            density: density field, already cleaned from refinements and overlaps (but not masked!)
            patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
            patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
            (and Y and Z)
            npatch: number of patches in each level, starting in l=0
            size: comoving size of the simulation box
            nmax: cells at base level
            verbose: if True, prints the patch being opened at a time
            ncores: number of cores to perform the paralellization.

        Returns:
            Field containing the mask as described.
    """
    mass = 0
    levels = create_vector_levels(npatch)
    cell_volume = (size / nmax / 2 ** levels) ** 3

    if verbose:
        print("Finding relevant patches...")
    which_patches = which_patches_inside_sphere(R, clusrx, clusry, clusrz, patchnx, patchny, patchnz, patchrx, patchry,
                                                patchrz, npatch, size, nmax)

    if verbose:
        print("Masking patches... (May take a while)")
    mask = mask_sphere_parallel(R, clusrx, clusry, clusrz, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch,
                                size, nmax, which_patches, verbose, ncores)

    if verbose:
        print("Summing the masses")
    for ipatch in range(len(patchnx)):
        mass += (density[ipatch] * mask[ipatch]).sum() * cell_volume[ipatch]

    return mass


def radial_profile_vw(field, clusrx, clusry, clusrz, rmin, rmax, nbins, logbins, patchnx, patchny, patchnz, patchrx,
                      patchry, patchrz, npatch, size, nmax, verbose=False, ncores=1):
    """
    Computes a (volume-weighted) radial profile of the quantity given in the "field" argument, taking center in
    (clusrx, clusry, clusrz).

    Args:
        field: variables (already cleaned) whose profile wants to be got
        clusrx, clusry, clusrz: comoving coordinates of the center for the profile
        rmin: starting radius of the profile
        rmax: final radius of the profile
        nbins: number of points for the profile
        logbins: if False, radial shells are spaced linearly. If True, they're spaced logarithmically. Not that, if
                 logbins = True, rmin cannot be 0.
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
                                   (and Y and Z)
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level
        verbose: if True, prints the patch being opened at a time
        ncores: number of cores to perform the paralellization.

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
    levels = create_vector_levels(npatch)
    cell_volume = (size / nmax / 2 ** levels) ** 3
    which_patches = which_patches_inside_sphere(rmax, clusrx, clusry, clusrz, patchnx, patchny, patchnz, patchrx,
                                                patchry, patchrz, npatch, size, nmax)
    field_vw = [field[ipatch] * cell_volume[ipatch] for ipatch in range(len(field))]

    if rmin > 0:
        cells_outer = mask_sphere_parallel(rmin, clusrx, clusry, clusrz, patchnx, patchny, patchnz, patchrx, patchry,
                                           patchrz, npatch, size, nmax, which_patches, verbose, ncores)
    else:
        cells_outer = [np.zeros(patch.shape, dtype='bool') for patch in field]

    for r_out in bin_bounds[1:]:
        if verbose:
            print('Working at outer radius {} Mpc'.format(r_out))
        cells_inner = cells_outer
        cells_outer = mask_sphere_parallel(r_out, clusrx, clusry, clusrz, patchnx, patchny, patchnz, patchrx, patchry,
                                           patchrz, npatch, size, nmax, which_patches, False, ncores)
        shell_mask = [cells_inner[ipatch] ^ cells_outer[ipatch] for ipatch in range(len(field))]
        sum_field_vw = sum([(field_vw[ipatch] * shell_mask[ipatch]).sum() for ipatch in range(len(field))])
        sum_vw = sum([(shell_mask[ipatch] * cell_volume[ipatch]).sum() for ipatch in range(len(field))])
        profile.append(sum_field_vw / sum_vw)

    return bin_centers, np.asarray(profile)


def several_radial_profiles_vw(fields, clusrx, clusry, clusrz, rmin, rmax, nbins, logbins, patchnx, patchny, patchnz,
                               patchrx, patchry, patchrz, npatch, size, nmax, verbose=False, ncores=1):
    """
    Computes several (volume-weighted) radial profiles at once, of the quantities given in the "fields" argument, as a
    list of regular fields,taking center in (clusrx, clusry, clusrz).

    Args:
        fields: list of variables (already cleaned) whose profile wants to be got
        clusrx, clusry, clusrz: comoving coordinates of the center for the profile
        rmin: starting radius of the profile
        rmax: final radius of the profile
        nbins: number of points for the profile
        logbins: if False, radial shells are spaced linearly. If True, they're spaced logarithmically. Not that, if
                 logbins = True, rmin cannot be 0.
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
                                   (and Y and Z)
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level
        verbose: if True, prints the patch being opened at a time
        ncores: number of cores to perform the paralellization.

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
    profiles = []

    # finding the volume-weighted mean
    levels = create_vector_levels(npatch)
    cell_volume = (size / nmax / 2 ** levels) ** 3
    which_patches = which_patches_inside_sphere(rmax, clusrx, clusry, clusrz, patchnx, patchny, patchnz, patchrx,
                                                patchry, patchrz, npatch, size, nmax)

    fields_vw = [[field[ipatch] * cell_volume[ipatch] for ipatch in range(len(field))] for field in fields]

    if rmin > 0:
        cells_outer = mask_sphere_parallel(rmin, clusrx, clusry, clusrz, patchnx, patchny, patchnz, patchrx, patchry,
                                           patchrz, npatch, size, nmax, which_patches, False, ncores)
    else:
        cells_outer = [np.zeros(patch.shape, dtype='bool') for patch in fields[0]]

    for r_out in bin_bounds[1:]:
        if verbose:
            print('Working at outer radius {} Mpc'.format(r_out))
        cells_inner = cells_outer
        cells_outer = mask_sphere_parallel(r_out, clusrx, clusry, clusrz, patchnx, patchny, patchnz, patchrx, patchry,
                                           patchrz, npatch, size, nmax, which_patches, False, ncores)
        shell_mask = [cells_inner[ipatch] ^ cells_outer[ipatch] for ipatch in range(len(fields[0]))]

        sum_vw = sum([(shell_mask[ipatch] * cell_volume[ipatch]).sum() for ipatch in range(len(fields[0]))])

        profile_thisr = []
        for field_vw in fields_vw:
            sum_field_vw = sum([(field_vw[ipatch] * shell_mask[ipatch]).sum() for ipatch in range(len(fields[0]))])
            profile_thisr.append(sum_field_vw / sum_vw)

        profiles.append(profile_thisr)

    return bin_centers, np.asarray(profiles).T


def patch_is_inside_box(box_limits, level, nx, ny, nz, rx, ry, rz, size, nmax):
    """
    See "Returns:"

    Args:
        box_limits: a tuple in the form (xmin, xmax, ymin, ymax, zmin, zmax)
        level: refinement level of the given patch
        nx, ny, nz: extension of the patch (in cells at level n)
        rx, ry, rz: comoving coordinates of the center of the leftmost cell of the patch
        size: comoving box side (preferred length units)
        nmax: cells at base level

    Returns:
        Returns True if the patch should contain cells within a sphere of radius r of the (clusrx, clusry, clusrz)
        point; False otherwise.

    """
    vertices = patch_vertices(level, nx, ny, nz, rx, ry, rz, size, nmax)
    pxmin = vertices[0][0]
    pymin = vertices[0][1]
    pzmin = vertices[0][2]
    pxmax = vertices[-1][0]
    pymax = vertices[-1][1]
    pzmax = vertices[-1][2]

    bxmin = box_limits[0]
    bxmax = box_limits[1]
    bymin = box_limits[2]
    bymax = box_limits[3]
    bzmin = box_limits[4]
    bzmax = box_limits[5]

    overlap_x = (pxmin <= bxmax) and (bxmin <= pxmax)
    overlap_y = (pymin <= bymax) and (bymin <= pymax)
    overlap_z = (pzmin <= bzmax) and (bzmin <= pzmax)

    return overlap_x and overlap_y and overlap_z


def which_patches_inside_box(box_limits, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, size, nmax):
    """
    Finds which of the patches will contain cells within a box of defined vertices.

    Args:
        box_limits: a tuple in the form (xmin, xmax, ymin, ymax, zmin, zmax)
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        (and Y and Z)
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level

    Returns:
        List containing the ipatch of the patches which should contain cells inside the considered radius.

    """
    levels = create_vector_levels(npatch)
    which_ipatch = [0]
    for ipatch in range(1, len(patchnx)):
        if patch_is_inside_box(box_limits, levels[ipatch], patchnx[ipatch], patchny[ipatch], patchnz[ipatch],
                               patchrx[ipatch], patchry[ipatch], patchrz[ipatch], size, nmax):
            which_ipatch.append(ipatch)
    return which_ipatch


def uniform_grid_zoom(field, box_limits, up_to_level, npatch, patchnx, patchny, patchnz, patchrx, patchry, patchrz,
                      size, nmax, verbose=False):
    """
    Builds a uniform grid, zooming on a box specified by box_limits, at level up_to_level, containing the most refined
    data at each region.

    Args:
        field: field to be computed at the uniform grid. Must be already cleaned from refinements and overlaps (check
        clean_field() function).
        box_limits: a tuple in the form (xmin, xmax, ymin, ymax, zmin, zmax). Limits should correspond to l=0 cell
                    boundaries
        up_to_level: level up to which the fine grid wants to be obtained
        npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
                                   (and Y and Z)
        size: comoving side of the simulation box
        nmax: cells at base level
        verbose: if True, prints the patch being opened at a time

    Returns:
        Uniform grid as described

    """
    uniform_cellsize = size / nmax / 2 ** up_to_level

    bxmin = box_limits[0]
    bxmax = box_limits[1]
    bymin = box_limits[2]
    bymax = box_limits[3]
    bzmin = box_limits[4]
    bzmax = box_limits[5]

    # BASE GRID
    uniform_size_x = int(round(nmax * (bxmax - bxmin) / size * 2 ** up_to_level))
    uniform_size_y = int(round(nmax * (bymax - bymin) / size * 2 ** up_to_level))
    uniform_size_z = int(round(nmax * (bzmax - bzmin) / size * 2 ** up_to_level))
    uniform = np.zeros((uniform_size_x, uniform_size_y, uniform_size_z))

    reduction = 2 ** up_to_level

    starting_x = (bxmin + size / 2) * nmax / size
    # ending_x = (bxmax + size/2) * nmax / size
    starting_y = (bymin + size / 2) * nmax / size
    # ending_y = (bymax + size / 2) * nmax / size
    starting_z = (bzmin + size / 2) * nmax / size
    # ending_z = (bzmax + size / 2) * nmax / size

    for i in range(uniform_size_x):
        for j in range(uniform_size_y):
            for k in range(uniform_size_z):
                I = int(starting_x + i / reduction)
                J = int(starting_y + j / reduction)
                K = int(starting_z + k / reduction)
                uniform[i, j, k] += field[0][I, J, K]

    # REFINEMENT LEVELS

    levels = create_vector_levels(npatch)
    # up_to_level_patches = npatch[0:up_to_level + 1].sum()
    relevantpatches = which_patches_inside_box(box_limits, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch,
                                               size, nmax)[1:]  # we ignore the 0-th relevant patch (patch 0, base
    # level). By construction, the 0th element is always the l=0 patch.
    relevantpatches = [i for i in relevantpatches if i <= npatch[0:up_to_level + 1].sum()]

    for ipatch in relevantpatches:
        if verbose:
            print('Covering patch {}'.format(ipatch))

        reduction = 2 ** (up_to_level - levels[ipatch])
        ipatch_cellsize = uniform_cellsize * reduction

        vertices = patch_vertices(levels[ipatch], patchnx[ipatch], patchny[ipatch], patchnz[ipatch], patchrx[ipatch],
                                  patchry[ipatch], patchrz[ipatch], size, nmax)
        pxmin = vertices[0][0]
        pymin = vertices[0][1]
        pzmin = vertices[0][2]
        pxmax = vertices[-1][0]
        pymax = vertices[-1][1]
        pzmax = vertices[-1][2]

        # fix left corners
        if pxmin <= bxmin:
            imin = 0
            Imin = (bxmin - pxmin) / ipatch_cellsize
        else:
            imin = int(round((pxmin - bxmin) / uniform_cellsize))
            Imin = 0

        if pymin <= bymin:
            jmin = 0
            Jmin = (bymin - pymin) / ipatch_cellsize
        else:
            jmin = int(round((pymin - bymin) / uniform_cellsize))
            Jmin = 0

        if pzmin <= bzmin:
            kmin = 0
            Kmin = (bzmin - pzmin) / ipatch_cellsize
        else:
            kmin = int(round((pzmin - bzmin) / uniform_cellsize))
            Kmin = 0

        # fix right corners
        if bxmax <= pxmax:
            imax = uniform_size_x
        else:
            # imax = int(round(patchnx[ipatch]*reduction))
            Imax = patchnx[ipatch] - 1
            imax = int(round(imin + (Imax - Imin + 1) * reduction))

        if bymax <= pymax:
            jmax = uniform_size_y
        else:
            # jmax = int(round(patchny[ipatch]*reduction))
            Jmax = patchny[ipatch] - 1
            jmax = int(round(jmin + (Jmax - Jmin + 1) * reduction))

        if bzmax <= pzmax:
            kmax = uniform_size_z
        else:
            # kmax = int(round(patchnz[ipatch]*reduction))
            Kmax = patchnz[ipatch] - 1
            kmax = int(round(kmin + (Kmax - Kmin + 1) * reduction))

        for i in range(imin, imax):
            for j in range(jmin, jmax):
                for k in range(kmin, kmax):
                    I = int(Imin + (i - imin) / reduction)
                    J = int(Jmin + (j - jmin) / reduction)
                    K = int(Kmin + (k - kmin) / reduction)
                    uniform[i, j, k] += field[ipatch][I, J, K]

    return uniform


def find_rDelta_eqn(r, Delta, background_density, clusrx, clusry, clusrz, density, patchnx, patchny, patchnz, patchrx,
                    patchry,
                    patchrz, npatch, size, nmax, verbose, ncores):
    if verbose:
        print('Evaluating at r={:.3f}'.format(r))
    m = mass_inside(r, clusrx, clusry, clusrz, density, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch,
                    size, nmax, verbose=False, ncores=ncores)
    return m - (4 * np.pi / 3) * r ** 3 * background_density * Delta


def find_rDelta(Delta, zeta, clusrx, clusry, clusrz, density, patchnx, patchny, patchnz, patchrx, patchry, patchrz,
                npatch,
                size, nmax, h, omega_m, rmin=0.1, rmax=6, rtol=1e-3, verbose=False, ncores=1):
    """
    Finds the value (in Mpc) of r_\Delta, the radius enclosing a mean overdensity (of the DM field, by default) equal
    to Delta times the background density of the universe. By default, it uses the Brent method.

    Args:
        Delta: value of the desired overdensity
        zeta: current redshift
        clusrx, clusry, clusrz: comoving coordinates of the center of the sphere
        density: DM density field, already cleaned from refinements and overlaps (but not masked!)
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        (and Y and Z)
        npatch: number of patches in each level, starting in l=0
        size: comoving size of the simulation box
        nmax: cells at base level
        h: dimensionless Hubble constant
        omega_m: matter density parameter
        rmin, rmax: starting two points (M(r) - Delta rho_B 4pi/3 r^3) must change sign
        verbose: if True, prints the patch being opened at a time
        ncores: number of cores to perform the paralellization.

    Returns:
        The value of r_\Delta
    """
    background_density = cosmo_tools.background_density(h, omega_m, zeta)
    args = (Delta, background_density, clusrx, clusry, clusrz, density, patchnx, patchny, patchnz, patchrx, patchry,
            patchrz, npatch, size, nmax, verbose, ncores)
    try:
        rDelta = optimize.brentq(find_rDelta_eqn, rmin, rmax, args=args, xtol=rtol)
    except ValueError:
        if verbose:
            print('Something might be wrong with this one... Might need further checking.')
        return float('nan')
    if verbose:
        print('Converged!')
    return rDelta


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
    levels = create_vector_levels(npatch)
    with Pool(ncores) as p:
        positions = p.map(compute_position_field_onepatch, [(patchnx[ipatch], patchny[ipatch], patchnz[ipatch], patchrx[ipatch], patchry[ipatch],
                                   patchrz[ipatch], levels[ipatch], size, nmax) for ipatch in range(len(patchnx))])

    cellsrx = [p[0] for p in positions]
    cellsry = [p[1] for p in positions]
    cellsrz = [p[2] for p in positions]

    return cellsrx, cellsry, cellsrz

