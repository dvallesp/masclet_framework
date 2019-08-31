#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

tools module
Contains several useful functions that other modules might need

v0.1.0, 30/06/2019
David Vallés, 2019
"""

#numpy
import numpy as np

def create_vector_levels(NPATCH):
    """
    Creates a vector containing the level for each patch. Nothing really
    important, just for ease
    """
    vector = [0]
    for l in range(1,NPATCH.size):
        vector.extend([l for i in range(NPATCH[l])])
    return np.array(vector)


def find_absolute_grid_position(ipatch, NPATCH, PATCHX, PATCHY, PATCHZ, PARE):
    '''
    Given a patch in a certain iteration, finds the grid (natural) coordinates 
    of its left corner. Written as a recursive function this time
    
    
    PARAMS:
    - ipatch: number of the patch (in the given iteration)
    - NPATCH: vector containing how many patches exist for each level
    - PATCHX (...Y, Z): vector containing all the patches leftmost corner 
    position (in the previous level) (as np arrays)
    - PARE: vector containing the ipatch of each patch progenitor
    
    RETURNS:
    - tuple with the patches first cell, left corner NX, NY, NZ values
    '''
    level = create_vector_levels(NPATCH)[ipatch]
    if ipatch < PATCHX.size:
        if level == 1:
            return PATCHX[ipatch], PATCHY[ipatch], PATCHZ[ipatch]
        else:
            parevalues = find_absolute_grid_position(PARE[ipatch], NPATCH, PATCHX, PATCHY, PATCHZ, PARE)
            return ((PATCHX[ipatch]-1)/2**(level-1) + parevalues[0], (PATCHY[ipatch]-1)/2**(level-1) + parevalues[1], (PATCHZ[ipatch]-1)/2**(level-1) + parevalues[2])
    else:
        raise ValueError('Provide a valid patchnumber (ipatch)')
        
        
def find_absolute_real_position(ipatch, SIDE, NMAX, NPATCH, PATCHX, PATCHY, PATCHZ, PARE):
    '''
    Given a patch in a certain iteration, finds the real coordinates of its 
    left corner. This function depends on find_absolute_grid_position, which 
    must be also loaded
    
    PARAMS:
    - ipatch: number of the patch (in the given iteration)
    - NPATCH: vector containing how many patches exist for each level
    - PATCHX (...Y, Z): vector containing all the patches leftmost corner 
    position (in the previous level) (as np arrays)
    - PARE: vector containing the ipatch of each patch progenitor
    - SIDE: box side length, in the desired units (typically Mpc or kpc)
    - NMAX: number of cells along each directions at level l=0 (can be loaded 
    from masclet_parameters)
    
    Caution! This functions assumes NMAX = NMAY = NMAZ
    
    RETURNS:
    - numpy array with the patches first cell, left corner X,Y and Z values 
    (assuming box centered in 0; same units than) 
    '''
    return (np.asarray(find_absolute_grid_position(ipatch,NPATCH,PATCHX,PATCHY,PATCHZ,PARE)) - NMAX/2 - 1)*SIDE/NMAX
