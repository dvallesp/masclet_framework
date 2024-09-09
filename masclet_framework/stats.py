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


def nonparametric_fit(x, y, sfactor=1., verbose=True):
    """ 
    Fit a nonparametric smoothing spline to the data (x, y) using scipy's UnivariateSpline.
    The smoothing parameter s is determined automatically, by estimating the standard deviation 
    of the ordinates in a window around each point.

    Args:
        x: the x values (1d np.array)
        y: the y values (1d np.array)
        sfactor: a scaling factor for the smoothing parameter. (float; optional, default=1. [no scaling])
                 if None, it is optimized to minimize the scatter in the residuals while preventing overfitting.
        verbose: if True and sfactor is None, print the optimal sfactor

    Returns:
        interpolant: the interpolant function (UnivariateSpline)
    """
    if sfactor is not None:
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

        # Scale the smoothing parameter by sfactor
        s *= sfactor

        interpolant = UnivariateSpline(x[order], y[order], k=3, s=s, w=w)
        return interpolant
    else: # Optimize the smoothing parameter
        sfacs = []
        targets = []
        sc1s = []
        sc2s = []

        xxx = np.linspace(min(x), max(x), 50)
        # Note: changin the number of points in xxx will directly affect the scatter_of_fit 
        # and change the results. I keep it to 50 since it is a reasonable scale for typical 
        # 2d plots.
        for sfactor in np.arange(0.1, 3, 0.1):
            interpolant = nonparametric_fit(x, y, sfactor=sfactor)
            yyy = interpolant(xxx)

            # prevent overfitting
            residuals = y - interpolant(x)
            scatter_around_fit = np.mean(residuals**2)**0.5 
            scatter_of_fit = np.mean(np.diff(yyy)**2)**0.5

            # we want minimal scatter_around_fit and minimal scatter_of_fit, thus minimal target
            target = scatter_around_fit + scatter_of_fit
            #print('{:.1f} --> {:.3f} ({:.3f} {:.3f})'.format(sfactor, target, scatter_around_fit, scatter_of_fit))
            sfacs.append(sfactor)
            targets.append(target)
            sc1s.append(scatter_around_fit)
            sc2s.append(scatter_of_fit)

        # Find the optimal sfactor, parabolic interpolation to find the minimum sfactor
        idx = np.argmin(targets)
        if idx == 0 or idx == len(targets)-1:
            sfactor = sfacs[idx]
        else:
            # without parabolic 
            sfactor = sfacs[idx]
            interpolant = nonparametric_fit(x, y, sfactor=sfactor)
            yyy = interpolant(xxx)
            residuals = y - interpolant(x)
            scatter_around_fit = np.mean(residuals**2)**0.5
            scatter_of_fit = np.mean(np.diff(yyy)**2)**0.5
            target0 = scatter_around_fit + scatter_of_fit

            # with parabolic
            x1 = sfacs[idx-1]
            x2 = sfacs[idx]
            x3 = sfacs[idx+1]
            y1 = targets[idx-1]
            y2 = targets[idx]
            y3 = targets[idx+1]
            
            sfactor = x3*(y2-y1) + x1*(y3-y2) + x2*(y1-y3)
            sfactor *= (x3**2*(y2-y1) + x1**2*(y3-y2) + x2**2*(y1-y3))
            sfactor /= 2*((x1-x2)*(x1-x3)*(x2-x3))**2

            interpolant = nonparametric_fit(x, y, sfactor=sfactor)
            yyy = interpolant(xxx)
            residuals = y - interpolant(x)
            scatter_around_fit = np.mean(residuals**2)**0.5
            scatter_of_fit = np.mean(np.diff(yyy)**2)**0.5
            target1 = scatter_around_fit + scatter_of_fit

            if target1 > target0:
                sfactor = sfacs[idx]

            if verbose:
                print('Optimal sfactor:', sfactor)
            
        return nonparametric_fit(x, y, sfactor=sfactor)





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
        