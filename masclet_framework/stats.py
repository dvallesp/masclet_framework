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
import emcee
from masclet_framework import graphics
from multiprocessing import Pool

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
            
            sfactor = x3**2*(y1-y2) + x1**2*(y2-y3) + x2**2*(y3-y1)
            sfactor /= 2*(x3*(y1-y2) + x1*(y2-y3) + x2*(y3-y1))

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


def nonparametric_conditional_correlation(x, y, z, mode='spearman', sfactor=None):
    """
    Compute the conditional correlation between x and y given z, using a nonparametric fit
    to estimate the residuals.

    In detail, the function fits a nonparametric smoothing spline to the data (z, y) and computes
    the residuals. Then, it computes the correlation between the residuals and x. The correlation
    can be computed in two modes: 'spearman' or 'pearson'.

    ****
    Remark: unlike partial_correlation, this function does not assume linearity or monotonicity.
    ****

    Args:
        x: first array (1d np.array)
        y: second array (1d np.array)
        z: third array (1d np.array), control variable
        mode: 'spearman' or 'pearson' (str)
        sfactor: a scaling factor for the smoothing parameter. (float; optional, default=None [optimize])

    Returns:
        correlation: the conditional correlation value (float)
    """
    interpolant = nonparametric_fit(z, y, sfactor=sfactor)

    # Compute the residuals
    residuals = y - interpolant(z)

    # Compute the correlation between the residuals and z
    return correlation(residuals, x, mode=mode)


def conditional_correlation(x, y, z, mode='spearman', linear=True, sfactor=None):
    """
    Compute the conditional correlation between x and y given z, using a linear model or a 
    nonparametric fit to estimate the residuals. For the nonparametric fit, see nonparametric_conditional_correlation
    for details.

    The correlation can be computed in two modes: 'spearman' or 'pearson'.

    Args:
        x: first array (1d np.array)
        y: second array (1d np.array)
        z: third array (1d np.array), control variable
        mode: 'spearman' or 'pearson' (str)
        linear: if True, use a linear model to estimate the residuals. Otherwise, use a nonparametric fit.
        sfactor: a scaling factor for the smoothing parameter. (float; optional, default=None [optimize])

    Returns:
        correlation: the conditional correlation value (float)
    """
    if linear:
        rho_XY = correlation(x, y, mode=mode)
        rho_XZ = correlation(x, z, mode=mode)
        rho_YZ = correlation(y, z, mode=mode)
        return (rho_XY - rho_XZ*rho_YZ) / np.sqrt((1-rho_XZ**2)*(1-rho_YZ**2))

    else:
        return nonparametric_conditional_correlation(x, y, z, mode=mode, sfactor=sfactor)
        

def MCMC_fit(f, initial, x, y, yerr, loglikelihood=None, logprior=None, nwalkers=100, nsteps=1000, burnin=100,
             ncores=1):
    """
    Performs a MCMC fit to the function f using the emcee package.

    Args:
        f: the model function to fit (callable). 
            The function should have the form f(theta, x=x).
                theta is a list of parameters to fit.
                x is the independent variable.
            The function should return the model prediction for x given theta.

        loglikelihood: the log-likelihood function (callable).
            The function should have the form loglikelihood(theta, x, y, yerr).
                theta is a list of parameters to fit.
                x is the independent variable.
                y is the dependent variable.
                yerr is the uncertainty in y.
            The function should return the log-likelihood of the model given the data.
            If None, a chi2 log-likelihood is used.

        x: the independent variable (np.array)

        y: the dependent variable (np.array)

        yerr: the uncertainty in y (np.array)

        logprior: the log-prior function (callable).
            The function should have the form logprior(theta).
                theta is a list of parameters to fit.
            The function should return the log-prior of the parameters.
            If None, a flat prior is used.

            Alternatively, logprior can be a list of 2-element lists, given the prior range for each parameter.
            In this case, a flat prior is used, but only for the specified range.

        nwalkers: the number of walkers in the MCMC chain (int)

        nsteps: the number of steps in the MCMC chain (int)

        burnin: the number of burn-in steps (int)

        ncores: the number of cores to use in the MCMC computation (int)
            !! WARNING: parallel not working as expected. Use with caution!!!!

    Returns:
        sampler: the emcee sampler object, containing the MCMC chain
        pos: the final position of the walkers (np.array)
        prob: the final probability of the walkers (np.array)
        state: the final state of the walkers (dict)
    """

    if loglikelihood is None:
        def loglikelihood(theta, x, y, yerr):
            # Chi2 log-likelihood
            model = f(theta, x=x)
            return -0.5*np.sum((y-model)**2/yerr**2)
        
    if logprior is None:
        def logprior(theta):
            return 0.0
    elif isinstance(logprior, list):
        logpriorlist = [a for a in logprior]
        def logprior(theta):
            for i, t in enumerate(theta):
                if t < logpriorlist[i][0] or t > logpriorlist[i][1]:
                    return -np.inf
            return 0.0

    def logprob(theta, x, y, yerr):
        lp = logprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + loglikelihood(theta, x, y, yerr)
    
    ndim = len(initial)
    if type(initial) is list:
        initial = np.array(initial)
    p0 = [initial * (1 + 1e-3*np.random.randn(ndim)) for i in range(nwalkers)]

    # Burn-in
    if ncores==1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, args=(x, y, yerr))
        
        pos, prob, state = sampler.run_mcmc(p0, burnin)
        sampler.reset()

        # Perform the MCMC
        pos, prob, state = sampler.run_mcmc(pos, nsteps)
    else:
        with Pool(ncores) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, args=(x, y, yerr), pool=pool)

            pos, prob, state = sampler.run_mcmc(p0, burnin)
            sampler.reset()

            # Perform the MCMC
            pos, prob, state = sampler.run_mcmc(pos, nsteps)
    
    return sampler, pos, prob, state


def MCMC_get_samples(sampler, thin=1):
    """
    Given the sampler, return the MCMC samples.

    Args:
        sampler: the emcee sampler object, containing the MCMC chain
        thin: the thinning factor (int). If thin=1, no thinning is performed.
              If thin>1, only every thin-th sample is kept.
              If thin=-1, the thinning factor is automatically determined by the autocorrelation time.

    Returns:
        samples: the MCMC samples (np.array)
    """
    if thin == -1:
        tau = MCMC_get_autocorrelationtime(sampler).mean()
        thin = max(1,int(0.5*tau))
    return sampler.get_chain(thin=thin, flat=True)


def MCMC_get_autocorrelationtime(sampler):
    """
    Given the sampler, return the autocorrelation time.

    Args:
        sampler: the emcee sampler object, containing the MCMC chain

    Returns:
        tau: the autocorrelation time (float)
    """
    return sampler.get_autocorr_time()


def MCMC_parameter_estimation(sampler, parameter_type='mean', uncertainty_type='std'):
    """
    Given the sampler, compute the parameter estimation and uncertainties.

    Args:
        sampler: the emcee sampler object, containing the MCMC chain
        parameter_type: the way to compute the parameter. Options are 'mean' or 'mode'.
        uncertainty_type: the type of uncertainty to compute. 
            'std' for standard deviation, [q1, q2] for quantiles q1 and q2 (per 1.)

    Returns:
        theta: the parameter estimation (np.array)
        (low, high): the lower and upper thresholds (tuple of np.arrays)
    """
    samples = MCMC_get_samples(sampler)
    if parameter_type == 'mean':
        theta = np.mean(samples, axis=0)
    elif parameter_type == 'mode':
        logprob = sampler.get_log_prob(flat=True)
        idx = np.argmax(logprob)
        theta = samples[idx]
    else:
        raise ValueError('Unknown parameter type')
    
    if uncertainty_type == 'std':
        sigma = np.std(samples, axis=0)
        low, high = theta - sigma, theta + sigma
    elif isinstance(uncertainty_type, list):
        q1, q2 = uncertainty_type
        low, high = np.quantile(samples, [q1, q2], axis=0)
    else:
        raise ValueError('Unknown uncertainty type')

    return theta, (low, high)

def MCMC_fit_R2(sampler, x, y, f, parameter_type='mean', take_logs=False):
    """
    Given the sampler, compute the R2 value of the fit.

    Args:
        sampler: the emcee sampler object, containing the MCMC chain
        x: the independent variable (np.array)
        y: the dependent variable (np.array)
        f: the model function to fit (callable).
        parameter_type: the way to compute the parameter. Options are 'mean' or 'mode'.
        take_logs: if True, take the logarithm of the model and data before computing the R2 value.

    Returns:
        R2: the R2 value of the fit (float)
    """
    theta, (low, high) = MCMC_parameter_estimation(sampler, parameter_type=parameter_type)
    model = f(theta, x=x)

    if take_logs:
        model2 = np.log10(model)
        y2 = np.log10(y)
    else:
        model2 = model
        y2 = y

    SS_res = np.sum((y2-model2)**2)
    SS_tot = np.sum((y2-np.mean(y2))**2)

    return 1 - SS_res/SS_tot


def MCMC_cornerplot(sampler, nsamples_plot=1000, thin=1, varnames=None, units=None, logscale=None, figsize=None, labelsize=12, ticksize=10, title=None,
                    s=3, color='blue', kde=True, cmap_kde='Blues', filled_kde=True, annotate_best=True, best_color='red', annotate_best_uncertainty=False,
                    parameter_type='mean', uncertainty_type='std', kde_plot_outliers=True, kde_quantiles=None, axes_limits=None):
    
    """
    Generates a corner plot of the MCMC chain.

    Args:
        sampler: the emcee sampler object, containing the MCMC chain
        nsamples_plot: the number of samples to plot (int)
        thin: the thinning factor (int). If thin=1, no thinning is performed. If thin>1, only every thin-th sample is kept. If thin=-1, the thinning factor is automatically determined by the autocorrelation time.
        varnames: list of m strings with the names of the variables. If not specified, it will place no labels.
        units: list of m strings with the units of the variables. If not specified, it will place no units.
        logscale: list of m booleans, True if the variable is in log scale, False if not. If not specified, it will place no log scale.
        figsize: tuple with the dimensions of the figure
        labelsize: size of the labels
        ticksize: size of the ticks
        title: title of the plot
        s: size of the scatter points
        color: color of the histograms and scatter plots
        kde: whether to plot the KDE or not
        cmap_kde: colormap for the KDE plot
        kde_plot_outliers: whether to plot the outliers in the KDE plot or not
        kde_quantiles: if None, it will just plot the KDE. If a list of quantiles is given, it will plot the
            filled contour plot of the KDE with the specified quantiles (and not the KDE itself).
        filled_kde: whether to fill the KDE plot or not
        annotate_best: whether to annotate the best fit values or not
        best_color: color of the annotation of the best fit values
        kde_plot_outliers: whether to plot the outliers in the KDE plot or not
        kde_quantiles: if None, it will just plot the KDE. If a list of quantiles is given, it will plot the
         filled contour plot of the KDE with the specified quantiles (and not the KDE itself).
        axes_limits: list of m tuples with the limits of the axes. 
            If not specified, it will use the data limits.
            Optionally, if a single float (0<f<1), it will discard this fraction from the extremes of the data.

    Returns:
        figure object and grid of axes with the corner plot
    """
    samples = MCMC_get_samples(sampler, thin=thin)
    if nsamples_plot < samples.shape[0]:
        idx = np.random.choice(samples.shape[0], nsamples_plot, replace=False)
        samples = samples[idx, :]
    
    fig, axes = graphics.cornerplot(samples, varnames=varnames, units=units, figsize=figsize, labelsize=labelsize, ticksize=ticksize, title=title,
                                s=s, color=color, kde=kde, cmap_kde=cmap_kde, filled_kde=filled_kde, kde_plot_outliers=kde_plot_outliers, kde_quantiles=kde_quantiles,
                                axes_limits=axes_limits)

    if annotate_best:
        theta, (low, high) = MCMC_parameter_estimation(sampler, parameter_type=parameter_type, uncertainty_type=uncertainty_type)
        for i in range(samples.shape[1]):
            axes[i,i].axvline(theta[i], color=best_color)
            if annotate_best_uncertainty:
                axes[i,i].axvspan(low[i], high[i], color=best_color, alpha=0.2)
        
        for i in range(samples.shape[1]):
            for j in range(i):
                axes[i,j].scatter(theta[j], theta[i], color=best_color, s=10, marker='x')

    return fig, axes

