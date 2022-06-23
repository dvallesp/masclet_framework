"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

stats module
Contains useful routines for dealing with some statistical analyses

Created by David Vallés
"""

#  Last update on 23/06/2022 11:02

import numpy as np
import astropy.stats

def biweight_statistic(array, nmaxiter=100, tol=1e-4):
    '''
    Return the biweight location estimator (computed iteratively),
     and the beweight scale (sqrt of the mid-variance) computed with
     respect to this location estimation.

    Args:
        array: a numpy array (or a list) containing the data to study (np.array)
        nmaxiter: the maximum number of iterations to determine the centre location (int)
        tol: stop the iteration when the centre location changes by less than this
              fractional change (float)

    Returns:
        M: the location robust estimator
        std: the scale (dispersion) robust estimator
    '''
    M=astropy.stats.biweight.biweight_location(array)
    for it in range(nmaxiter):
        Mprev = M
        M = astropy.stats.biweight.biweight_location(array, M=Mprev)
        err = np.abs(M-Mprev)/np.abs(Mprev)
        if err < tol:
            break
    #print(it, err)
    
    std = astropy.stats.biweight.biweight_scale(array, M=M, modify_sample_size=True)
    return M, std