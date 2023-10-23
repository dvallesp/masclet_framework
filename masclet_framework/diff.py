"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

diff module
Provides functions to compute gradients, divergences, curls of scalar and 
vector fields defined on the AMR hierarchy of MASCLET simulations.
Created by David Vallés
"""

import numpy as np
from numba import njit, prange
from masclet_framework import tools

## Patch stuff (arr_ functions)
@njit(fastmath=True)
def arr_diff_x(arr):
    nx = arr.shape[0]
    difference = np.zeros_like(arr)
    difference[1:nx-1,:,:] = (arr[2:nx,:,:] - arr[0:nx-2,:,:])
    difference[0,:,:] = 2*(arr[1,:,:] - arr[0,:,:]) - difference[1,:,:] # Second order extrapolation at the boundary
    difference[nx-1,:,:] = 2*(arr[nx-1,:,:] - arr[nx-2,:,:]) - difference[nx-2,:,:] # Second order extrapolation at the boundary
    return difference

@njit(fastmath=True)
def arr_diff_y(arr):
    ny = arr.shape[1]
    difference = np.zeros_like(arr)
    difference[:,1:ny-1,:] = (arr[:,2:ny,:] - arr[:,0:ny-2,:])
    difference[:,0,:] = 2*(arr[:,1,:] - arr[:,0,:]) - difference[:,1,:] # Second order extrapolation at the boundary
    difference[:,ny-1,:] = 2*(arr[:,ny-1,:] - arr[:,ny-2,:]) - difference[:,ny-2,:] # Second order extrapolation at the boundary
    return difference

@njit(fastmath=True)
def arr_diff_z(arr):
    nz = arr.shape[2]
    difference = np.zeros_like(arr)
    difference[:,:,1:nz-1] = (arr[:,:,2:nz] - arr[:,:,0:nz-2])
    difference[:,:,0] = 2*(arr[:,:,1] - arr[:,:,0]) - difference[:,:,1] # Second order extrapolation at the boundary
    difference[:,:,nz-1] = 2*(arr[:,:,nz-1] - arr[:,:,nz-2]) - difference[:,:,nz-2] # Second order extrapolation at the boundary
    return difference

@njit(fastmath=True)
def arr_gradient(arr, dx):
    den = np.float32(1/dx)
    return den*arr_diff_x(arr), den*arr_diff_y(arr), den*arr_diff_z(arr)

@njit(fastmath=True)
def arr_gradient_magnitude(arr, dx):
    den = np.float32(1/dx)
    return den*np.sqrt(arr_diff_x(arr)**2 + arr_diff_y(arr)**2 + arr_diff_z(arr)**2)

@njit(fastmath=True)
def arr_divergence(arr_x, arr_y, arr_z, dx):
    den = np.float32(1/dx)
    return (arr_diff_x(arr_x) + arr_diff_y(arr_y) + arr_diff_z(arr_z))*den

@njit(fastmath=True)
def arr_curl(arr_x, arr_y, arr_z, dx):
    den = np.float32(1/dx)
    return den*(arr_diff_y(arr_z) - arr_diff_z(arr_y)), den*(arr_diff_z(arr_x) - arr_diff_x(arr_z)), den*(arr_diff_x(arr_y) - arr_diff_y(arr_x))

@njit(fastmath=True)
def arr_curl_magnitude(arr_x, arr_y, arr_z, dx):
    den = np.float32(1/dx)
    return den*np.sqrt((arr_diff_y(arr_z) - arr_diff_z(arr_y))**2 +
                       (arr_diff_z(arr_x) - arr_diff_x(arr_z))**2 + 
                       (arr_diff_x(arr_y) - arr_diff_y(arr_x))**2)

## Fields stuff
def gradient(field, dx, npatch, kept_patches=None):
    '''
    Computes the gradient of a scalar field defined on the AMR hierarchy of
     grids.

    Args:
        field: a list of numpy arrays, each one containing the scalar field
                defined on the corresponding grid of the AMR hierarchy
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.

    Returns:
        grad_x: a list of numpy arrays, each one containing the x-component of
                the gradient of the scalar field defined on the corresponding
                grid of the AMR hierarchy
        grad_y: idem for the y-component
        grad_z: idem for the z-component

     Author: David Vallés
    '''
    levels=tools.create_vector_levels(npatch)
    resolution=dx/2**levels
    grad_x = []
    grad_y = []
    grad_z = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum(), dtype=bool)

    for ipatch in prange(npatch.sum()):
        if kept_patches[ipatch]:
            gx,gy,gz = arr_gradient(field[ipatch], resolution[ipatch])
        else:
            gx,gy,gz = 0,0,0
        grad_x.append(gx)
        grad_y.append(gy)
        grad_z.append(gz)

    return grad_x, grad_y, grad_z

def divergence(field_x, field_y, field_z, dx, npatch, kept_patches=None):
    '''
    Computes the divergence of a vector field defined on the AMR hierarchy of
     grids.

    Args:
        field_x: a list of numpy arrays, each one containing the x-component of 
                the vector field defined on the corresponding grid of the AMR 
                hierarchy
        field_y: idem for the y-component
        field_z: idem for the z-component
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.

    Returns:
        div: a list of numpy arrays, each one containing the divergence of the
                vector field defined on the corresponding grid of the AMR    
                hierarchy 

    Author: David Vallés
    '''
    levels=tools.create_vector_levels(npatch)
    resolution=dx/2**levels
    div = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum(), dtype=bool)

    for ipatch in prange(npatch.sum()):
        if kept_patches[ipatch]:
            div.append(arr_divergence(field_x[ipatch], field_y[ipatch], field_z[ipatch], resolution[ipatch]))
        else:
            div.append(0)

    return div

def curl(field_x, field_y, field_z, dx, npatch, kept_patches=None):
    '''
    Computes the curl of a vector field defined on the AMR hierarchy of
     grids.

    Args:
        field_x: a list of numpy arrays, each one containing the x-component of
                the vector field defined on the corresponding grid of the AMR
                hierarchy
        field_y: idem for the y-component
        field_z: idem for the z-component
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.

    Returns:
        curl_x: a list of numpy arrays, each one containing the x-component of
                the curl of the vector field defined on the corresponding grid  
                of the AMR hierarchy
        curl_y: idem for the y-component
        curl_z: idem for the z-component

    Author: David Vallés
    '''
    levels=tools.create_vector_levels(npatch)
    resolution=dx/2**levels
    curl_x = []
    curl_y = []
    curl_z = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum(), dtype=bool)

    for ipatch in prange(npatch.sum()):
        if kept_patches[ipatch]:
            cx,cy,cz = arr_curl(field_x[ipatch], field_y[ipatch], field_z[ipatch], resolution[ipatch])
        else:
            cx,cy,cz = 0,0,0
        curl_x.append(cx)
        curl_y.append(cy)
        curl_z.append(cz)

    return curl_x, curl_y, curl_z

def curl_magnitude(field_x, field_y, field_z, dx, npatch, kept_patches=None):
    '''
    Computes the magnitude of the curl of a vector field defined on the 
     AMR hierarchy of grids.

    Args:
        field_x: a list of numpy arrays, each one containing the x-component of
                the vector field defined on the corresponding grid of the AMR
                hierarchy
        field_y: idem for the y-component
        field_z: idem for the z-component
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.

    Returns:
        curl_mag: a list of numpy arrays, each one containing the magnitude of
                the curl of the vector field defined on the corresponding grid 
                of the AMR hierarchy

    Author: David Vallés
    '''
    levels=tools.create_vector_levels(npatch)
    resolution=dx/2**levels
    curl_mag = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum(), dtype=bool)

    for ipatch in prange(npatch.sum()):
        if kept_patches[ipatch]:
            curl_mag.append(arr_curl_magnitude(field_x[ipatch], field_y[ipatch], field_z[ipatch], resolution[ipatch]))
        else:
            curl_mag.append(0)

    return curl_mag

def gradient_magnitude(field, dx, npatch, kept_patches=None):
    '''
    Computes the magnitude of the gradient of a scalar field defined on 
     the AMR hierarchy of grids.

    Args:
        field: a list of numpy arrays, each one containing the scalar field
                defined on the corresponding grid of the AMR hierarchy
        dx: the cell size of the coarsest grid
        npatch: the number of patches in each direction
        kept_patches: 1d boolean array, True if the patch is kept, False if not.
                    If None, all patches are kept.

    Returns:
        grad_mag: a list of numpy arrays, each one containing the magnitude of
                the gradient of the scalar field defined on the corresponding
                grid of the AMR hierarchy

     Author: David Vallés
    '''  
    levels=tools.create_vector_levels(npatch)
    resolution=dx/2**levels
    grad_mag = []

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum(), dtype=bool)

    for ipatch in prange(npatch.sum()):
        if kept_patches[ipatch]:
            grad_mag.append(arr_gradient_magnitude(field[ipatch], resolution[ipatch]))
        else:
            grad_mag.append(0)

    return grad_mag