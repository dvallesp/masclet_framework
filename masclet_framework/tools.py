"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

tools module
Contains several useful functions that other modules might need

Created by David Vallés
"""

#  Last update on 18/3/20 12:09

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

# numpy
import numpy as np

from multiprocessing import Pool

# FUNCTIONS DEFINED IN THIS MODULE


def create_vector_levels(npatch):
    """
    Creates a vector containing the level for each patch. Nothing really important, just for ease

    Args:
        npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)

    Returns:
        numpy array containing the level for each patch

    """
    vector = [[0]] + [[i+1]*x for (i, x) in enumerate(npatch[1:])]
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
            return ((patchx[ipatch] - 1) / 2 ** (level - 1) + parevalues[0],
                    (patchy[ipatch] - 1) / 2 ** (level - 1) + parevalues[1],
                    (patchz[ipatch] - 1) / 2 ** (level - 1) + parevalues[2])
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
        find_absolute_grid_position(ipatch, npatch, patchx, patchy, patchz, pare)) - nmax / 2 - 1) * side / nmax


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

    cellsize = size/nmax/2**level

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

                vertices.append((x,y,z))

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
    vertices = patch_vertices(level, nx, ny, nz, rx, ry, rz, size, nmax)

    isinside = False
    cell_l0_size = size/nmax

    max_side = max([nx, ny, nz]) * cell_l0_size / 2**level
    upper_bound_squared = R**2 + max_side**2/4
    for vertex in vertices:
        distance_squared = (vertex[0]-clusrx)**2 + (vertex[1]-clusry)**2 + (vertex[2]-clusrz)**2
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
    for ipatch in range(1,len(patchnx)):
        if patch_is_inside_sphere(R, clusrx, clusry, clusrz, levels[ipatch], patchnx[ipatch], patchny[ipatch], patchnz[ipatch],
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
        rx, ry, rz: comoving coordinates of the center of the leftmost cell of the patch
        size: comoving box side (preferred length units)
        nmax: cells at base level:

    Returns:
        isinside: numpy bool array, the size of the patch, containing for each cell 1 if inside and 0 otherwise
    """

    isinside = np.zeros((nx,ny,nz),dtype=bool)

    cell_l0_size = size / nmax
    cell_size = cell_l0_size / 2**level
    Rsquared = R**2

    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                x = rx + i*cell_size
                y = ry + j*cell_size
                z = rz + k*cell_size
                isinside[i,j,k] = ((x-clusrx)**2 + (y-clusry)**2 + (z-clusrz)**2 <= Rsquared)

    return isinside


def mask_sphere(R, clusrx, clusry, clusrz, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, size, nmax,
                which_patches, verbose=False):
    """
    Returns a "field", which contains all patches as usual. True means the cell must be considered, False otherwise.
    If a patch has all "False", an array is not ouputted, but a False is, instead.

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


def mask_sphere_parallel(R, clusrx, clusry, clusrz, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, size, nmax,
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


def clean_field(field, cr0amr, solapst, npatch, up_to_level = 1000):
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

    field[0] = field[0]*cr0amr[0] # not overlap in l=0

    for level in range(1, up_to_level):
        for ipatch in range(sum(npatch[0:level]) + 1, sum(npatch[0:level + 1]) + 1):
            field[ipatch] = field[ipatch] * cr0amr[ipatch] * solapst[ipatch]

    # last level: no refinements
    for ipatch in range(sum(npatch[0:up_to_level]) + 1, sum(npatch[0:up_to_level + 1]) + 1):
        field[ipatch] = field[ipatch] * solapst[ipatch]

    return field
