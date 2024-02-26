"""
HALMA-MASCLET FRAMEWORK

Created on Mon Mar 27 2022

@author: ÓSCAR MONLLOR BERBEGAL
"""

import time
import numba
from numba.pycc import CC
import numpy as np
import sys
from multiprocessing import Pool

sys.path.append('/home/marcomol/trabajo/scripts/masclet_framework_marco')
from masclet_framework import tools


# DESCRIPTION
# This function interpolates the AMR field to a uniform grid
# It is faster to clean the patches until l = level and not consider patches with l>level

# PENDING
# - Use also cell neighbors when l = level, not only the closest cell for solving the centering problem

""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
" FROM AMR TO UNIFORM GRID WITH NUMBA COMPILATION "
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

############################################################################################################
## NUMBA COMPILATION OF PATCH_INTERPOLATION FUNCTION
############################################################################################################
#[::1]     IS C ORDER
#[:,:,::1] IS C ORDER
#[::1,:,:] IS F ORDER
#SEE HOW patch_field is float32 and F ORDER !!!!!!!!!!!!!!!!!!!!!!!!
module_name = 'amr2uniform_CC' 
cc = CC(module_name) #MODULE NAME 
signature = numba.float64[:,:,::1](numba.int64, numba.int64, numba.int64, numba.int64, numba.int64, numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.float64[::1], numba.float64[::1], 
                                 numba.int64, numba.int64, numba.int64, numba.float64, numba.float64, numba.float64, numba.float64, numba.float32[:,:,::1])

@numba.njit(signature)
@cc.export(exported_name='patch_interpolation', sig = signature)
def patch_interpolation(level, l, nx, ny, nz, grid_faces_x, grid_faces_y, grid_faces_z, grid_centers_x, grid_centers_y, grid_centers_z, patch_nx, patch_ny, patch_nz, rx, ry, rz, patch_res, patch_field):

    ####################################################################################################################
    # WHAT IS DONE HERE:
    # If level == l, we are in the finest level, so we just copy the values to the closest cell
    # If level > l, we are in a coarser level, so we copy the values to the uniform cells CONTAINED in not solaped patch cells
    # If level < l, RAISE AN ERROR, its faster to clean until l = level and not consider patches with l>level
    ####################################################################################################################

    #Create the uniform grid
    field_uniform = np.zeros((nx, ny, nz), dtype = np.float64)
    
    #Find the closest cell in the uniform grid
    x0 = rx - patch_res/2 #Center of the left-bottom-front cell
    y0 = ry - patch_res/2
    z0 = rz - patch_res/2
    if level == l:
        #We are in the finest level, so we just copy the values to the closest cell
        #Check if the cell is inside the grid and if it is not solaped with a finer patch
        for i in range(patch_nx):
            x = x0 + i*patch_res #cell center
            if not(x < grid_faces_x[0] or x > grid_faces_x[-1]):
                for j in range(patch_ny):
                    y = y0 + j*patch_res
                    if not(y < grid_faces_y[0] or y > grid_faces_y[-1]):
                        for k in range(patch_nz):
                            z = z0 + k*patch_res
                            if not(z < grid_faces_z[0] or z > grid_faces_z[-1]) and patch_field[i, j, k] != 0.:
                                #FIND THE CLOSEST CELL
                                i_uniform = np.argmin(np.abs(grid_centers_x - x))
                                j_uniform = np.argmin(np.abs(grid_centers_y - y))
                                k_uniform = np.argmin(np.abs(grid_centers_z - z))
                                field_uniform[i_uniform, j_uniform, k_uniform] = patch_field[i, j, k]

    elif level > l:
        #We are in a coarser level, so we copy the values to the uniform cells CONTAINED in not solaped patch cells
        #Check if the cell is inside the grid (THIS TIME WITH CELL FACES) and if it is not solaped with a finer patch
        for i in range(patch_nx):
            x_left = x0 + i*patch_res - patch_res/2 #cell left face
            x_right = x0 + i*patch_res + patch_res/2 #cell right face
            if not(x_left < grid_faces_x[0] and x_right > grid_faces_x[-1]):
                for j in range(patch_ny):
                    y_left = y0 + j*patch_res - patch_res/2
                    y_right = y0 + j*patch_res + patch_res/2
                    if not(y_left < grid_faces_y[0] and y_right > grid_faces_y[-1]):
                        for k in range(patch_nz):
                            z_left = z0 + k*patch_res - patch_res/2
                            z_right = z0 + k*patch_res + patch_res/2
                            if not(z_left < grid_faces_z[0] and z_right > grid_faces_z[-1]) and patch_field[i, j, k] != 0.:
                                #FROM WHERE TO WHERE ASSIGN THE VALUE
                                i_uniform_left = np.argmin(np.abs(grid_faces_x - x_left))
                                i_uniform_right = min(np.argmin(np.abs(grid_faces_x - x_right)), nx-1)
                                j_uniform_left = np.argmin(np.abs(grid_faces_y - y_left))
                                j_uniform_right = min(np.argmin(np.abs(grid_faces_y - y_right)), ny-1)
                                k_uniform_left = np.argmin(np.abs(grid_faces_z - z_left))
                                k_uniform_right =  min(np.argmin(np.abs(grid_faces_z - z_right)), nz-1)
                                field_uniform[i_uniform_left:i_uniform_right, j_uniform_left:j_uniform_right, k_uniform_left:k_uniform_right] = patch_field[i, j, k]

    if l > level:
        #RAISE AN ERROR, its faster to clean until l = level and not consider patches with l>level
        raise ValueError('l > level, its faster to clean until l = level and not consider patches with l>level')
    
    return field_uniform
############################################################################################################
############################################################################################################



#COMPILE THE FUNCTION IF MAIN
if __name__ == "__main__":
    cc.compile()

#IMPORT AHEAD OF TIME COMPILED FUNCTION
if not(__name__ == '__main__'):
    from amr2uniform_CC import patch_interpolation as patch_interpolation_CC

def argparser_patch_interpolation(args):
    return patch_interpolation_CC(*args)



############################################################################################################
## FUNCTION THAT CALLS THE NUMBA COMPILED FUNCTION
############################################################################################################
def main(box, level, ncoarse, L, npatch, patchnx, patchny, patchnz, patchrx, patchry, patchrz, field, verbose = False):
    """
    Args:
        level: should be the same as the one used in the cleaning of the field 
    
    """
    # if just_grid = True, it returns only the uniform grid
    just_grid = False
    if type(field) is not list:
        ValueError('Field must be a list of fields')
    else:
        if len(field) != 1 and len(field) != 3:
            ValueError('Field must be a scalar or a vector')
        else:
            if field[0] == 0.:
                just_grid = True


    #Calculate which patches contribute
    if verbose:
        print('Calculating which patches contribute...')
    #See how I extend the box to have find patches in order to information at the boundaries
    which_patches = tools.which_patches_inside_box(box, patchnx, patchny, patchnz, patchrx, patchry, patchrz, npatch, L, ncoarse)

    patch_level = tools.create_vector_levels(npatch)

    if verbose:
        print('max level found:',np.max(patch_level[which_patches]))
    #Define uniform grid
    res = (L/ncoarse)/2**level
    xlims = [box[0], box[1] + (res - (box[1]-box[0])%res)] #Number of cells between the box boundaries must be an integer, thus we modify the right boundary
    ylims = [box[2], box[3] + (res - (box[3]-box[2])%res)]
    zlims = [box[4], box[5] + (res - (box[5]-box[4])%res)]
    nx = int((xlims[1]-xlims[0])/res)
    ny = int((ylims[1]-ylims[0])/res)
    nz = int((zlims[1]-zlims[0])/res)

    grid_faces_x = np.linspace(xlims[0], xlims[1], nx+1)
    grid_faces_y = np.linspace(ylims[0], ylims[1], ny+1)
    grid_faces_z = np.linspace(zlims[0], zlims[1], nz+1)
    grid_centers_x = (grid_faces_x[1:] + grid_faces_x[:-1])/2
    grid_centers_y = (grid_faces_y[1:] + grid_faces_y[:-1])/2
    grid_centers_z = (grid_faces_z[1:] + grid_faces_z[:-1])/2

    print('Maximum level found:', np.max(patch_level[which_patches]))
    if not just_grid:
        #ARGS TO PARSE TO THE PARALLELIZATION OF THE INTERPOLATION
        args = []
        if verbose:
            print('Interpolating...')

        #for each patch, calculate the interpolation
        for ipatch, patch in enumerate(which_patches):  
            if ipatch%1 == 0 and verbose:
                print('Interpolating patch {}/{}'.format(ipatch, len(which_patches)))
            l = patch_level[patch]
            if l <= level:
                patch_res = (L/ncoarse)/2**l
                patch_rx = patchrx[patch]
                patch_ry = patchry[patch]
                patch_rz = patchrz[patch]
                patch_nx = patchnx[patch]
                patch_ny = patchny[patch]
                patch_nz = patchnz[patch]

                # print('---> patch', patch)
                # print(patch_res)
                # print(patch_rx, patch_ry, patch_rz)
                # print(patch_nx, patch_ny, patch_nz)

                if len(field) == 1: #assume that the field is a scalar
                    #FROM FORTRAN ORDER TO C ORDER with np.ascontiguousarray
                    patch_field = np.ascontiguousarray( field[0][patch] ) 
                if len(field) == 3: #assume that the field is a vector, then we take the norm
                    patch_field = np.ascontiguousarray( (field[0][patch]**2 + field[1][patch]**2 + field[2][patch]**2)**0.5 )

                args.append((level, l, nx, ny, nz, grid_faces_x, grid_faces_y, grid_faces_z, grid_centers_x, grid_centers_y, grid_centers_z, 
                            patch_nx, patch_ny, patch_nz, patch_rx, patch_ry, patch_rz, patch_res, patch_field))


        # #SPLIT THE PATCHES CONTRIBUTING BETWEEN THE WORKERS
        # #PARALLEL
        # if ncores > 1:
        #     t0 = time.time()
        #     with Pool(ncores) as p:
        #         field_uniform_list = list(p.map(argparser_patch_interpolation, args))
        #     t1 = time.time()
        #     #REDUCTION OF THE FIELD: +
        #     field_uniform = np.zeros(field_uniform_list[0].shape)
        #     for field in field_uniform_list:
        #         field_uniform += field

        #     t2 = time.time()
        #     print('Time to interpolate ( parallel ', ncores,'):', t2-t0)
        
        #SERIAL
        field_uniform = np.zeros((nx, ny, nz))
        t0 = time.time()
        for arg in args:
            field_uniform += patch_interpolation(*arg)
        tf = time.time()
        print('Time to interpolate ( serial ):', tf-t0)
                
        if verbose:
            print('Done')

    print('Uniform grid created')
    return field_uniform, grid_centers_x, grid_centers_y, grid_centers_z

############################################################################################################
############################################################################################################
