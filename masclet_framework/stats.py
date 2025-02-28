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
from scipy.stats import spearmanr, t
from scipy.interpolate import UnivariateSpline
import emcee
from masclet_framework import graphics
from multiprocessing import Pool
from scipy.stats import gaussian_kde
import statsmodels.api as sm
from sklearn.preprocessing import PolynomialFeatures
from scipy import odr
import uncertainties as unc
import uncertainties.unumpy as unp
import logging

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
        

def MCMC_fit(f, initial, x, y, yerr, xerr=None, loglikelihood=None, logprior=None, nwalkers=100, nsteps=1000, burnin=100,
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

        xerr: the uncertainty in x (np.array). If None, no uncertainty in x is considered.

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
        def loglikelihood(theta, x, y, yerr, xerr):
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

    def logprob(theta, x, y, yerr, xerr):
        lp = logprior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + loglikelihood(theta, x, y, yerr, xerr)
    
    ndim = len(initial)
    if type(initial) is list:
        initial = np.array(initial)
    p0 = [initial * (1 + 1e-3*np.random.randn(ndim)) for i in range(nwalkers)]

    # Burn-in
    if ncores==1:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, args=(x, y, yerr, xerr))
        
        pos, prob, state = sampler.run_mcmc(p0, burnin)
        sampler.reset()

        # Perform the MCMC
        pos, prob, state = sampler.run_mcmc(pos, nsteps)
    else:
        with Pool(ncores) as pool:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, logprob, args=(x, y, yerr, xerr), pool=pool)

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


def MCMC_parameter_estimation(sampler, parameter_type='mean', uncertainty_type='std', marginalize_over_variables=None):
    """
    Given the sampler, compute the parameter estimation and uncertainties.

    Args:
        sampler: the emcee sampler object, containing the MCMC chain
        parameter_type: the way to compute the parameter. Options are 'mean' or 'mode'.
        uncertainty_type: the type of uncertainty to compute. 
            'std' for standard deviation, [q1, q2] for quantiles q1 and q2 (per 1.)
        marginalize_over_variables: list of variables (indices) to marginalize over. 
            If None, no marginalization is performed.
            NOTE: this is not working properly yet.

    Returns:
        theta: the parameter estimation (np.array)
        (low, high): the lower and upper thresholds (tuple of np.arrays)
    """
    samples = MCMC_get_samples(sampler)

    if marginalize_over_variables is not None:
        samples = np.delete(samples, marginalize_over_variables, axis=1)

    if parameter_type == 'mean':
        theta = np.mean(samples, axis=0)
    elif parameter_type == 'mode':
        logprob = sampler.get_log_prob(flat=True)
        idx = np.argmax(logprob)

        # with KDE 
        #kde = gaussian_kde(samples.T)
        #idx = np.argmax(kde(samples.T))

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
                                axes_limits=axes_limits, logscale=logscale)

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


def polyfit(x, y, ey, pval_thr=0.05, verbose=False, max_degree=None, xisqred_thr=None, 
            return_model=False, fix_degree=None):
    """
    Fit a polynomial to the data (x, y) using statsmodels WLS.
    It fits the polynomial of the lowest degree that satisfies the p-value and chi2 criteria.
        I.e., the order of the polynomial is increased until the p-value of the last coefficient is below pval_thr,
        (indicating that the coefficient is not significant) or the chi2 is below xisqred_thr (indicating a good fit 
        and the onset of overfitting).

    Args:
        x: the x values (1d np.array)
        y: the y values (1d np.array)
        ey: the uncertainty in y (1d np.array)
        pval_thr: the p-value threshold for the coefficients (float)
        verbose: if True, print the results of the fit (bool)
        max_degree: the maximum degree of the polynomial to fit (int)
        xisqred_thr: the chi2 threshold for the fit (float)
        return_model: if True, return the statsmodels model object (bool)
        fix_degree: if not None, fix the degree of the polynomial to this value (int)
    
    Returns:
        params: the parameters of the polynomial fit (np.array)
        model: the statsmodels model object (optional)
    """

    one_less=False
    two_less=False
    try_extra=False
    
    w = 1/np.array(ey)**2
    
    if fix_degree is None:
        if max_degree is None:
            max_degree=x.size-2

        for deg in range(max_degree+1):
            polynomial_features= PolynomialFeatures(degree=deg)
            xp = polynomial_features.fit_transform(x.reshape(-1, 1))

            model = sm.WLS(y, xp, weights=w).fit()
            ypred = model.predict(xp)
            xisqred = ((ypred-y)**2/ey**2).sum() / (y.size-deg-1)

            if verbose:
                print(deg, model.pvalues[-1], xisqred)

            if xisqred_thr is not None:  
                if xisqred<xisqred_thr:
                    if one_less:
                        two_less=True
                        one_less=False
                    else:
                        two_less=False
                        one_less=True
                    break

            if model.pvalues[-1] > pval_thr:
                if try_extra:
                    two_less=True
                    one_less=False
                    break
                else:
                    one_less=True
                    try_extra=True
            else:
                try_extra=False
                one_less=False #### added 18jul 2022
                two_less=False

        if one_less:
            deg-=1
        if two_less:
            deg-=2

        if deg<0:
            deg=0
    else:
        deg=fix_degree
    
    polynomial_features= PolynomialFeatures(degree=deg)
    xp = polynomial_features.fit_transform(x.reshape(-1, 1))

    model = sm.WLS(y, xp, weights=w).fit()
    
    if return_model:
        return model.params, model
    else:
        return model.params


def get_CIs_model(model, x, CI=0.32):
    """ 
    Compute the confidence intervals of the model at the given x values.

    Args:
        model: the statsmodels model object
        x: the x values (1d np.array)
        CI: the confidence interval (float)

    Returns:
        CIs: the confidence intervals (np.array)
    """

    degree=len(model.params)-1
    polynomial_features= PolynomialFeatures(degree=degree)
    xp = polynomial_features.fit_transform(x.reshape(-1, 1))
    return model.get_prediction(xp).conf_int(alpha=CI)


def fit_with_odr(x, y, xerr, yerr, f, initial_guess, xplot=None):
    """ 
    Fit a model to the data (x, y) using the orthogonal distance regression method.

    Args:
        x: the x values (1d np.array)
        y: the y values (1d np.array)
        xerr: the uncertainty in x (1d np.array)
        yerr: the uncertainty in y (1d np.array)
        f: the model function to fit (callable). 
            The function should have the form f(params, x).
                params is a list of parameters to fit.
                x is the independent variable.
            The function should return the model prediction for x given params.
        initial_guess: the initial guess for the parameters (list)
        xplot: the x values to evaluate the model and obtain confidence intervals

    Returns:
        dictionary with the fit parameters and the fit plot data (x, y, yerr [symmetric])
    """
    model = odr.Model(f)
    data = odr.RealData(x, y, sx=xerr, sy=yerr)
    odr_instance = odr.ODR(data, model, beta0=initial_guess)
    out = odr_instance.run()
    
    params = unc.correlated_values(out.beta, out.cov_beta)
    nominal_values = unp.nominal_values(params)
    std_devs = unp.std_devs(params)

    if xplot is None:
        xplot = np.linspace(x.min(), x.max(), 100)
    yplot = f(params, xplot)
    yplot_nominal = unp.nominal_values(yplot)
    yplot_std_dev = unp.std_devs(yplot)
    
    return {'fit_params': {'values': nominal_values,
                          'std_devs': std_devs},
            'fit_plot': {'x': xplot, 'y': yplot_nominal, 'yerr': yplot_std_dev},
            'model': out}

def polyfit_odr(x, y, xerr, yerr, max_degree=None, xisqred_thr=1.0, pval_thr=0.05,
                 fix_degree=None, xplot=None, verbose=False):
    """
    Fit a polynomial to the data (x, y) using orthogonal distance regression.
    It fits the polynomial of the lowest degree that satisfies the chi2 criteria.
        I.e., the order of the polynomial is increased until the chi2 is below 
        xisqred_thr (indicating a good fit and the onset of overfitting), or the 
        p-value of the last coefficient is above pval_thr (indicating that the
        coefficient is not significant).

    NOTE: the way the coefficients are computed is the opposite to polyval.
        That's to say, there the coefficient [0] is the highest degree, and here is the lowest.
        In this (our) way, the i-th coefficient is the coefficient of the i-th degree.
        To evaluate with polyval, the coefficients should be reversed: np.polyval(coefs[::-1], x)

    Args:
        x: the x values (1d np.array)
        y: the y values (1d np.array)
        xerr: the uncertainty in x (1d np.array)
        yerr: the uncertainty in y (1d np.array)
        max_degree: the maximum degree of the polynomial to fit (int)
        xisqred_thr: the chi2 threshold for the fit (float)
        pval_thr: the p-value threshold for the highest order coefficients (float)
        fix_degree: if not None, fix the degree of the polynomial to this value (int)
        xplot: the x values to evaluate the model and obtain confidence intervals
        verbose: if True, print the results of the fit (bool)

    Returns:
        dictionary with the fit parameters and the fit plot data (x, y, yerr [symmetric])
    """

    # p-value things
    one_less=False
    two_less=False
    try_extra=False

    # chi2 things 
    attempt=0

    if max_degree is None:
        max_degree=x.size-2

    if fix_degree is not None:
        max_degree=fix_degree

    chisq = {}
    models = {}
    pvals = {}

    # degree 0
    deg = 0 
    def f(B, x):
        return np.polyval(B[::-1], x)
    initial = np.mean(y)
    model = fit_with_odr(x, y, xerr, yerr, f, [initial], xplot=xplot)
    chisq[deg] = model['model'].res_var
    models[deg] = model

    tval = model['model'].beta / model['model'].sd_beta
    dof = x.size - deg - 1
    pval = 2 * (1 - t.cdf(np.abs(tval), dof))
    pvals[deg] = pval[-1] # higher order term

    # pvals[deg] = 

    if verbose:
        print('---')
        print(deg, chisq[deg])
        print(models[deg]['fit_params']['values'])
        print(models[deg]['fit_params']['std_devs'])
        print(pvals[deg])

    for deg in range(1,max_degree+1):
        def f(params, x):
            return np.polyval(params[::-1], x)
        initial = list(models[deg-1]['fit_params']['values']) + [0.]
        model = fit_with_odr(x, y, xerr, yerr, f, initial, xplot=xplot)
        chisq[deg] = model['model'].res_var
        models[deg] = model

        tval = model['model'].beta / model['model'].sd_beta
        dof = x.size - deg - 1
        pval = 2 * (1 - t.cdf(np.abs(tval), dof))
        pvals[deg] = pval[-1] # higher order term

        if verbose:
            print('---')
            print(deg, chisq[deg])
            print(models[deg]['fit_params']['values'])
            print(models[deg]['fit_params']['std_devs'])
            print(pvals[deg])

        if xisqred_thr is not None:  
            if chisq[deg]<xisqred_thr:
                #if one_less:
                #    two_less=True
                #    one_less=False
                #else:
                #    two_less=False
                #    one_less=True
                #if fix_degree is None:
                #    break
                if attempt==0:
                    attempt=1
                else:
                    attempt=2 
                    break

        if pvals[deg] > pval_thr:
            if try_extra:
                two_less=True
                one_less=False
                if fix_degree is None:
                    break
            else:
                one_less=True
                try_extra=True
        else:
            try_extra=False
            one_less=False #### added 18jul 2022
            two_less=False

    if verbose:
        print('-->', deg, one_less, two_less, attempt)

    if one_less:
        deg-=1
    elif two_less:
        deg-=2
    elif attempt > 0:
        deg-=(attempt-1)


    if deg<0:
        deg=0

    if fix_degree is not None:
        deg=fix_degree
  
    if verbose:
        print('---')
        print('Best degree:', deg)
        print('---')

    return models[deg]


def weighted_median(x, w, axes=None, interpolate=True):
    """
    Compute the weighted median of the data.

    Args:
        x: the data (n-d np.array)
        w: the weights (n-d np.array)
        axes: the axes along which to compute the median (int or list of ints)
            If None, the median is computed over the flattened array.
            If list of ints, the median is computed over the specified axes flattened array.
        interpolate: if True, interpolate the weighted median between the two closest

    Returns:
        the weighted median (np.array)
    """
    if axes is None:
        xflat = x.flatten()
        wflat = w.flatten()
    elif isinstance(axes, list) or isinstance(axes, tuple) or isinstance(axes, np.ndarray) or isinstance(axes, int):
        if isinstance(axes, int):
            axes = [axes]

        original_shape = np.array(x.shape)
        
        # to flatten the array we put the axes to flatten at the beginning
        all_axes = list(range(x.ndim))
        non_flatten_axes = [a for a in all_axes if a not in axes]
        new_order = axes + non_flatten_axes

        # We move the axes in the order we want to flatten
        xflat = np.moveaxis(x, new_order, range(len(new_order)))
        wflat = np.moveaxis(w, new_order, range(len(new_order))) 

        # Compute the flatten shape 
        flattened_size = np.prod(original_shape[axes])
        new_shape = (flattened_size,) + tuple(original_shape[a] for a in non_flatten_axes)

        # Flatten the arrays
        xflat = xflat.reshape(new_shape)
        wflat = wflat.reshape(new_shape)
    else:
        raise ValueError('Unknown axes specification')

    # Sort the data
    idx = np.argsort(xflat, axis=0)
    # sort 1d vector (along 0-th axis) according to the corresponding idx 
    xflat = np.take_along_axis(xflat, idx, axis=0)
    wflat = np.take_along_axis(wflat, idx, axis=0)

    # Compute the cumulative sum of the weights
    wflat = np.cumsum(wflat, axis=0)
    half = wflat[-1] / 2
    wflat = np.concatenate((np.zeros((1,) + wflat.shape[1:]), wflat), axis=0)
    wflat = 0.5*(wflat[1:] + wflat[:-1])
    
    # Find the first index where wflat > half, along each of the elements of the axes other than the first
    idx = np.argmax(wflat > half, axis=0)
    # If all are False, idx will be 0, so we need to check this case
    idx = np.where(idx > 0, idx, wflat.shape[0]-1)


    if not interpolate:
        return np.take_along_axis(xflat, idx[None, :], axis=0).squeeze()
    else:
        # Interpolate between the two closest
        idx1 = np.maximum(idx-1, 0)
        idx2 = np.minimum(idx, wflat.shape[0]-1)
        x1 = np.take_along_axis(xflat, idx1[None, :], axis=0).squeeze()
        x2 = np.take_along_axis(xflat, idx2[None, :], axis=0).squeeze()
        w1 = np.take_along_axis(wflat, idx1[None, :], axis=0).squeeze()
        w2 = np.take_along_axis(wflat, idx2[None, :], axis=0).squeeze()
        half = wflat[-1] / 2
        w1 = half - w1
        w2 = w2 - half

        if (w1+w2).min() == 0:
            logging.warning('The weighted median is not well defined at least in one of the array elements. '+ 
                                'Returning the average of the two closest values. Proceed with caution.')
            return 0.5*(x1+x2)

        return (x1*w2 + x2*w1) / (w1 + w2)