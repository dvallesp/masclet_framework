"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

particle2grid module
Contains the particle2grid routine to distribute a particle field onto a grid using an SPH kernel

Created by Óscar Monllor
"""

import numpy as np
import numba
import multiprocessing as mp
from scipy.spatial import KDTree
        
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
        grid_field: particle field distributed into a grid
        hpart: h distance of the particles
        
    Author: Óscar Monllor
    '''
    #Ensure Field is float64
    field = field.astype(np.float64)

    # First, build KDTree
    data = np.array((x, y, z)).T
    tree = KDTree(data)

    #
    npart = np.int64(len(x))

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
    @numba.njit([numba.float64(numba.float64, numba.float64)], fastmath = True)
    def SPH_kernel(r, h):
        q = r/h
        if q < 1:
            W = 1 - 1.5 * q**2* (1 - 0.5 * q)
        elif q < 2:
            W = 0.25 * (2 - q)**3
        else:
            W = 0
        return W
    

    @numba.njit([numba.float64[:,:,:](numba.float64[:,:,:], numba.int64[:],   numba.int64[:],   numba.int64[:],
                                      numba.float64[:],     numba.float64[:], numba.float64[:], numba.float64[:], numba.int64,
                                      numba.float64[:],     numba.float64,    numba.float64,    numba.float64,
                                      numba.int64,          numba.int64,      numba.int64)], 
                                      fastmath = True)
    def put_particles_in_grid(grid_field, grid_pos_x, grid_pos_y, grid_pos_z, 
                              x, y, z, field, npart, 
                              hpart, dx, dy, dz, 
                              nx, ny, nz):

        # Distribute the particle field into the grid
        for ipart in range(npart):
            i = grid_pos_x[ipart]
            j = grid_pos_y[ipart]
            k = grid_pos_z[ipart]

            # extent of the particle kernel
            h = hpart[ipart]
            x_extent = int(2*h/dx+ 0.5)
            y_extent = int(2*h/dy+ 0.5)
            z_extent = int(2*h/dz+ 0.5)

            # for each particle, find the norm
            norm = 0.
            for ix in range(i - x_extent, i + x_extent + 1):
                if ix >= 0 and ix < nx:
                    for jy in range(j - y_extent, j + y_extent + 1):
                        if jy >= 0 and jy < ny:
                            for kz in range(k - z_extent, k + z_extent + 1):
                                if kz >= 0 and kz < nz:
                                    # distance to the particle
                                    r = np.sqrt(((ix+0.5)*dx - x[ipart])**2 + ((jy+0.5)*dy - y[ipart])**2 + ((kz+0.5)*dz - z[ipart])**2)
                                    # kernel value
                                    W = SPH_kernel(r, h)
                                    #
                                    norm += W


            # Particle only contributes to its own cell
            if norm == 0.:
                grid_field[ix, jy, kz] += field[ipart]
                continue

            # distribute the field
            for ix in range(i - x_extent, i + x_extent + 1):
                if ix >= 0 and ix < nx:
                    for jy in range(j - y_extent, j + y_extent + 1):
                        if jy >= 0 and jy < ny:
                            for kz in range(k - z_extent, k + z_extent + 1):
                                if kz >= 0 and kz < nz:
                                    # distance to the particle
                                    r = np.sqrt(((ix+0.5)*dx - x[ipart])**2 + ((jy+0.5)*dy - y[ipart])**2 + ((kz+0.5)*dz - z[ipart])**2)
                                    # kernel value
                                    W = SPH_kernel(r, h)
                                    #
                                    grid_field[ix, jy, kz] += field[ipart] * W / norm


        return grid_field
    ##############################################

    #Force np.float64 in x,y,z
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    z = z.astype(np.float64)

    grid_field = put_particles_in_grid(grid_field, grid_pos_x, grid_pos_y, grid_pos_z, x, y, z,
                                       field, npart, hpart, dx, dy, dz, nx, ny, nz)
    return grid_field, hpart