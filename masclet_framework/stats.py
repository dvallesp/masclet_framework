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
from scipy.stats import spearmanr
from scipy.interpolate import UnivariateSpline

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


def correlation(x, y, mode='spearman'):
    '''
    Compute the correlation between x and y.

    Note: 'pearson' correlation assumes linearity. 
          'spearman' correlation assumes monotonicity.

    Args:
        x: first array (1d np.array)
        y: second array (1d np.array)
        mode: 'spearman' or 'pearson' (str)

    Returns:
        correlation: the correlation value (float)
    '''
    if mode == 'spearman':
        return spearmanr(x, y).correlation
    elif mode == 'pearson':
        return np.corrcoef(x, y)[0,1]
    else:
        raise ValueError('Unknown mode')


def partial_correlation(x, y, z, mode='spearman'):
    ''' 
    Compute the partial correlation between x and y given z, using the formula:

    p_{xy.z} = (p_{xy} - p_{xz} p_{yz}) / sqrt((1-p_{xz}^2)(1-p_{yz}^2))

    where p_{xy} is the correlation between x and y, and p_{xz} and p_{yz} are the correlations
    between x and z, and y and z, respectively. This comes from fitting linear models x(z) and y(z)
    and computing the correlation between the residuals.

    The correlation can be computed in two modes: 'spearman' or 'pearson'.

    Args:
        x: first array (1d np.array)
        y: second array (1d np.array)
        z: third array (1d np.array)
        mode: 'spearman' or 'pearson' (str)

    Returns:
        correlation: the partial correlation value (float)
    '''
    rho_xy = correlation(x, y, mode=mode)
    rho_xz = correlation(x, z, mode=mode)
    rho_yz = correlation(y, z, mode=mode)
    return (rho_xy - rho_xz*rho_yz) / np.sqrt((1-rho_xz**2)*(1-rho_yz**2))


def nonparametric_fit(x, y):
    """ 
    Fit a nonparametric smoothing spline to the data (x, y) using scipy's UnivariateSpline.
    The smoothing parameter s is determined automatically, by estimating the standard deviation 
    of the ordinates in a window around each point.

    Args:
        x: the x values (1d np.array)
        y: the y values (1d np.array)

    Returns:
        interpolant: the interpolant function (UnivariateSpline)
    """
    # Order dots (requirement of scipy UnivariateSpline)
    order = np.argsort(x)

    # Determine the smoothing parameter s 
    width = x.size//100
    if width < 5:
        width = 5
    if width > 100:
        width = 100
    if width > x.size:
        width = x.size

    w = np.zeros_like(x)
    for i in range(x.size):
        i1 = max(0, i-width)
        i2 = min(x.size, i+width)
        w[i] = np.std(y[order][i1:i2])

    w = 1/w
    s = len(w)
    w[:] = sum(w)/len(w)

    interpolant = UnivariateSpline(x[order], y[order], k=3, s=s, w=w)
    return interpolant

def parametric_conditional_correlation(x, y, z, mode='spearman'):
    """
    Compute the conditional correlation between x and y given z, using a nonparametric fit
    to estimate the residuals.

    In detail, the function fits a nonparametric smoothing spline to the data (x, y) and computes
    the residuals. Then, it computes the correlation between the residuals and z. The correlation
    can be computed in two modes: 'spearman' or 'pearson'.

    ****
    Remark: unlike partial_correlation, this function does not assume linearity or monotonicity.
    ****

    Args:
        x: first array (1d np.array)
        y: second array (1d np.array)
        z: third array (1d np.array), control variable
        mode: 'spearman' or 'pearson' (str)

    Returns:
        correlation: the conditional correlation value (float)
    """
    interpolant = nonparametric_fit(x, y)

    # Compute the residuals
    residuals = y - interpolant(x)

    # Compute the correlation between the residuals and z
    return correlation(residuals, z, mode=mode)


def conditional_correlation(x, y, z, mode='spearman', linear=True):
    """
    Compute the conditional correlation between x and y given z, using a linear model or a 
    nonparametric fit to estimate the residuals. For the nonparametric fit, see parametric_conditional_correlation
    for details.

    The correlation can be computed in two modes: 'spearman' or 'pearson'.

    Args:
        x: first array (1d np.array)
        y: second array (1d np.array)
        z: third array (1d np.array), control variable
        mode: 'spearman' or 'pearson' (str)
        linear: if True, use a linear model to estimate the residuals. Otherwise, use a nonparametric fit.

    Returns:
        correlation: the conditional correlation value (float)
    """
    if linear:
        rho_XY = correlation(x, y, mode=mode)
        rho_XZ = correlation(x, z, mode=mode)
        rho_YZ = correlation(y, z, mode=mode)
        return (rho_XY - rho_XZ*rho_YZ) / np.sqrt((1-rho_XZ**2)*(1-rho_YZ**2))

    else:
        return parametric_conditional_correlation(x, y, z, mode=mode)
        