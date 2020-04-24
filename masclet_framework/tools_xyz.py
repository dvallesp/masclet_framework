"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

tools_xyz module
Contains several useful functions that other modules might need. Here we have the (new) versions, which make more
intensive use of computing x,y,z fields (much faster, but more memory consuming)

Created by David Vallés
"""

#  Last update on 24/4/20 23:06

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

# numpy
import numpy as np

from multiprocessing import Pool

from masclet_framework import cosmo_tools, tools

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
                      cellsrz, cr0amr, solapst, npatch, size, nmax, verbose=False):
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
        shell_mask = tools.clean_field(shell_mask, cr0amr, solapst, npatch)

        sum_field_vw = sum([(fvw * sm).sum() for fvw, sm in zip(field_vw, shell_mask)])
        sum_vw = sum([(sm * cv).sum() for sm, cv in zip(shell_mask, cell_volume)])

        profile.append(sum_field_vw / sum_vw)

    return bin_centers, np.asarray(profile)


def several_radial_profiles_vw(fields, clusrx, clusry, clusrz, rmin, rmax, nbins, logbins, cellsrx, cellsry,
                               cellsrz, cr0amr, solapst, npatch, size, nmax, verbose=False):
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
        shell_mask = tools.clean_field(shell_mask, cr0amr, solapst, npatch)
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
        try:
            S_eigenvalues, S_eigenvectors = tools.diagonalize_ascending(shapetensor)
        except np.linalg.LinAlgError:
            print('Shape tensor had {} nans. Operation impossible.'.format(np.isnan(shapetensor).sum()))
            return None, None

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




