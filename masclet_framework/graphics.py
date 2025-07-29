"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

graphics module
Contains functions that can directly compute projections along one of the axis,
or generate colormap plots. Makes use of matplotlib.

Created by David Vallés
"""

#  Last update on 16/3/20 18:08

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

# numpy
import numpy as np
# matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
from matplotlib.gridspec import GridSpec
import seaborn as sns
import matplotlib.ticker as mticker
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import ListedColormap
import math
from scipy.stats import gaussian_kde
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
from numba import jit, njit, prange, get_num_threads, set_num_threads
from numba.typed import List


# MASCLET FRAMEWORK MODULES
from masclet_framework.profiles import locate_point
from masclet_framework import tools


# FUNCTIONS DEFINED IN THIS MODULE


def colormap2d(z, x=np.array([]), y=np.array([]), xlabel='', ylabel='', cbarlabel='', title='', cbartitle='',
               are_xy_given=True, are_meshgrid=False, cbarmin=None, cbarmax=None, xmin=None, xmax=None, ymin=None,
               ymax=None, cmap='PuBu_r', axisfont=18, ticksfont=18, titlefont=18, logz=False, manipulate_ticks=False,
               nticks=10):
    """
    Plots z against a xy-plane uniform grid in colorscale

    Args: (only z is mandatory)
        z: function to be plotted (as a numpy matrix)
        x: first cartesian coordinate (either as a vector or as a meshgrid)
        y: second cartesian coordinate (either as a vector or as a meshgrid)
        xlabel: label for the x-axis (str)
        ylabel: label for the y-axis (str)
        cbarlabel: label for the z-axis (colorbar) (str)
        title: title for the plot
        cbartitle: title for the colorbar
        areXYgiven: if False, it won't take x and y into account. Defaults to True (bool)
        areMeshgrid: if True, x and y are expected to be meshgrids. If False, x and y are expected to be vectors.
        Defaults to False. (bool)
        cbarmin: minimum value of the colorbar scale
        cbarmax: maximum value of the colorbar scale
        xmin: minimum value of the x-axis
        xmax: maximum value of the x-axis
        ymin: minimum value of the y-axis
        ymax: maximum value of the y-axis
        cmap: colorbar theme
        axisfont: fontsize of the x and y axis titles
        ticksfont: fontsize of the axis ticks
        titlefont: fontsize of the title
        logz: specifies if z is scaled logarithmically
        manipulateTicks: if True, one can choose how many ticks will the colorbar have. Defaults to False (bool)
        nticks: number of ticks for the colorbar (if manipulateTicks is set to true)

    Returns:
        axis object with the plot

    """

    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    if are_meshgrid:
        xmesh = x
        ymesh = y
        if xmesh.shape[0] != z.shape[0]+1 or xmesh.shape[1] != z.shape[1]+1:
            print('Error, x and y mesh dimensions should be 1 greater than z dimensions')
    else:
        if are_xy_given:
            if x.shape[0] == z.shape[0] and y.shape[0] == z.shape[1]:
                x = np.linspace(x[0], x[x.shape[0]-1], x.shape[0]+1)
                y = np.linspace(y[0], y[y.shape[0]-1], y.shape[0]+1)
            elif x.shape[0] != z.shape[0]+1 and y.shape[0] != z.shape[1]+1:
                print('Error: x and y are badly specified')
                raise ValueError()
            xmesh, ymesh = np.meshgrid(x, y)
        else:
            zsize = z.shape
            xmesh, ymesh = np.meshgrid(np.array(range(zsize[0] + 1)), np.array(range(zsize[1] + 1)))

    if cbarmin is None:
        cbarmin = z.min()
    if cbarmax is None:
        cbarmax = z.max()
    if xmin is None:
        xmin = xmesh.min()
    if xmax is None:
        xmax = xmesh.max()
    if ymin is None:
        ymin = ymesh.min()
    if ymax is None:
        ymax = ymesh.max()

    if logz:
        pcm = ax.pcolormesh(xmesh, ymesh, np.transpose(z), norm=colors.LogNorm(vmin=cbarmin, vmax=cbarmax), cmap=cmap)
    else:
        pcm = ax.pcolormesh(xmesh, ymesh, np.transpose(z), norm=colors.Normalize(vmin=cbarmin, vmax=cbarmax),
                            cmap=cmap)

    ax.set(xlim=(xmin, xmax), ylim=(ymin, ymax))

    plt.xticks(fontsize=ticksfont)
    plt.yticks(fontsize=ticksfont)
    plt.xlabel(xlabel, fontsize=axisfont)
    plt.ylabel(ylabel, fontsize=axisfont)
    plt.title(title, fontsize=titlefont)

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.ax.tick_params(labelsize=ticksfont)
    cbar.set_label(cbarlabel, fontsize=ticksfont)
    cbar.ax.set_title(cbartitle, fontsize=axisfont)

    if manipulate_ticks:  # not working properly right now! we have to format the number of significant figures
        cbar.set_ticks(np.linspace(cbarmin, cbarmax, nticks))
        cbar.set_ticklabels(np.linspace(cbarmin, cbarmax, nticks))
        cbar.update_ticks()

    return ax


def colormap2d_bicolored(proj_hue, proj_lightness, xg=None, yg=None, huevar_min=None, huevar_max=None, lightvar_min=None, lightvar_max=None, huevar_log=True, lightvar_log=True, 
                        cmap='coolwarm', nhue=100, nlight=100, xlabel='', ylabel='', huelabel='', lightlabel='', title='', remove_xy_ticks=False, axisfont=18, ticksfont=16, titlefont=18, 
                        xticks=None, yticks=None, hueticks=None, lightticks=None, figsize=(10,12)):
    """ 
    Plots a 2D colormap with two variables: one for hue and one for lightness (e.g., temperature and density).

    Args:
        proj_hue: 2D numpy array with the hue values
        proj_lightness: 2D numpy array with the lightness values
        xg: 1D numpy array with the x-axis values
        yg: 1D numpy array with the y-axis values
        huevar_min: minimum value of the hue variable to be represented. Values below this will be clipped.
        huevar_max: maximum value of the hue variable to be represented. Values above this will be clipped.
        lightvar_min: minimum value of the lightness variable to be represented. Values below this will be clipped.
        lightvar_max: maximum value of the lightness variable to be represented. Values above this will be clipped.
        huevar_log: if True, the hue variable will be plotted in log scale
        lightvar_log: if True, the lightness variable will be plotted in log scale
        cmap: colormap for the hue variable. Can be a string with the name of the colormap or a matplotlib colormap object.
        nhue: number of colors for the hue variable colormap
        nlight: number of colors for the lightness variable colormap
        xlabel: label for the x-axis (spatial)
        ylabel: label for the y-axis (spatial)
        huelabel: label for the hue colorbar 
        lightlabel: label for the lightness colorbar
        title: title of the plot
        remove_xy_ticks: if True, it will remove the x and y ticks from the plot
        axisfont: fontsize of the axis labels
        ticksfont: fontsize of the ticks
        titlefont: fontsize of the title
        xticks: list of x-axis ticks. If None, it will use the default ticks
        yticks: list of y-axis ticks. If None, it will use the default ticks
        hueticks: list of hue colorbar ticks. If None, it will use the default ticks
        lightticks: list of lightness colorbar ticks. If None, it will use the default ticks
        figsize: size of the figure

    Returns:
        figure object and list of axis objects with the plot
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(6, 5, figure=fig)

    ax = fig.add_subplot(gs[0:5, 0:5])
    ax.set_aspect(1)
    axcol = fig.add_subplot(gs[5, 1:4])

    if huevar_log:
        huevar = np.log10(proj_hue)
    else:
        huevar = proj_hue.copy()
    
    if lightvar_log:
        lightvar = np.log10(proj_lightness)
    else:
        lightvar = proj_lightness.copy()

    if xg is None:
        xg = np.arange(proj_hue.shape[0])
    if yg is None:
        yg = np.arange(proj_hue.shape[1])

    if huevar_min is None:
        huevar_min = huevar.min()
    if huevar_max is None:
        huevar_max = huevar.max()
    if lightvar_min is None:
        lightvar_min = lightvar.min()
    if lightvar_max is None:
        lightvar_max = lightvar.max()

    if type(cmap) is str:
        cmap = cm.get_cmap(cmap)

    hue_scale = np.linspace(huevar_min, huevar_max, nhue)
    light_scale = np.linspace(lightvar_min, lightvar_max, nlight)

    huevar = np.clip((huevar - huevar_min) / (huevar_max-huevar_min), 0, 1)
    lightvar = np.clip((lightvar - lightvar_min) / (lightvar_max-lightvar_min), 0, 1)

    hue = cmap(huevar)[:,:,:3]
    light = lightvar

    colored_map = hue * light[..., None]

    ax.imshow(colored_map, origin='lower', extent=[xg.min(), xg.max(), yg.min(), yg.max()])

    # set labels
    ax.set_xlabel(xlabel, fontsize=axisfont)
    ax.set_ylabel(ylabel, fontsize=axisfont)
    ax.set_title(title, fontsize=titlefont)
    ax.tick_params(axis='both', which='major', labelsize=ticksfont)

    if remove_xy_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)


    # plot the 2d colorbar below 
    hue_cvals = np.linspace(huevar_min, huevar_max, nhue)
    light_cvals = np.linspace(lightvar_min, lightvar_max, nlight)

    hue_cvar = np.clip((hue_cvals - huevar_min) / (huevar_max-huevar_min), 0, 1)
    light_cvar = np.clip((light_cvals - lightvar_min) / (lightvar_max-lightvar_min), 0, 1)
    hue_cvar, light_cvar = np.meshgrid(hue_cvar, light_cvar)

    hue_c = cm.get_cmap(cmap)(hue_cvar)[:,:,:3]
    light_c = light_cvar

    colored_cmap = hue_c * light_c[..., None]

    axcol.imshow(colored_cmap, origin='lower', extent=[huevar_min, huevar_max, lightvar_min, lightvar_max])
    axcol.set_aspect('auto')

    if hueticks is not None:
        axcol.set_xticks(hueticks)
    if lightticks is not None:
        axcol.set_yticks(lightticks)

    # set labels
    axcol.set_xlabel(huelabel, fontsize=axisfont)
    axcol.set_ylabel(lightlabel, fontsize=axisfont)
    axcol.tick_params(axis='both', which='major', labelsize=ticksfont)

    # prevent axes colliding 
    fig.tight_layout()

    return fig, [ax, axcol]


def colormap2d_RGB(proj_R, proj_G, proj_B, 
                   xg=None, yg=None, 
                   Rlog=True, Rmin=None, Rmax=None, Rscale=1., Rcolor=[1.,0.,0.], sigma_R=0., Rlabel='', Rticks=None,
                   Glog=True, Gmin=None, Gmax=None, Gscale=1., Gcolor=[0.,1.,0.], sigma_G=0., Glabel='', Gticks=None,
                   Blog=True, Bmin=None, Bmax=None, Bscale=1., Bcolor=[0.,0.,1.], sigma_B=0., Blabel='', Bticks=None,
                   xlabel='', ylabel='', title='', remove_xy_ticks=False, axisfont=18, ticksfont=16, titlefont=18, 
                   xticks=None, yticks=None, figsize=(10,14)):
    """
    Plots a 2D colormap with three variables as R, G and B channels (or other three color channels).

    Args:
        proj_R, proj_G, proj_B: thre 2D numpy arrays with the R, G and B channel variables 
        xg, yg: 1D numpy arrays with the x, and y-axis values. If None, it will use pixel coordinates.
        Rlog, Glog, Blog: if True, the R, G and B variables will be plotted in log scale, respectively
        Rmin, Gmin, Bmin: minimum value of the R, G and B variables to be represented. Values below this will be clipped.
            These values correspond to black in the colormap.
        Rmax, Gmax, Bmax: maximum value of the R, G and B variables to be represented. Values above this will be clipped.
            These values correspond to Rcolor, Gcolor and Bcolor in the colormap, times Rscale, Gscale and Bscale.
        Rscale, Gscale, Bscale: scaling factor for the R, G and B variables, so that the color does not saturate.
        Rcolor, Gcolor, Bcolor: color of the R, G and B variables. Defaults to red, green and blue.
        sigma_R, sigma_G, sigma_B: width of the 2d gaussian filter for the R, G and B variables. Defaults to 0 (no smoothing)
        Rlabel, Glabel, Blabel: label for the R, G and B colorbars
        Rticks, Gticks, Bticks: list of ticks for the R, G and B colorbars. If None, it will use the default ticks
        xlabel: label for the x-axis (spatial)
        ylabel: label for the y-axis (spatial)
        title: title of the plot
        remove_xy_ticks: if True, it will remove the x and y ticks from the plot
        axisfont: fontsize of the axis labels
        ticksfont: fontsize of the ticks
        titlefont: fontsize of the title
        xticks: list of x-axis ticks. If None, it will use the default ticks
        yticks: list of y-axis ticks. If None, it will use the default ticks
        figsize: size of the figure
        
    Returns:
        figure object and list of axis objects with the plot
    """
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(21, 15, figure=fig)

    ax = fig.add_subplot(gs[0:15, 0:15])
    ax.set_aspect(1)
    axcbar_R = fig.add_subplot(gs[16, 2:13])
    axcbar_G = fig.add_subplot(gs[18, 2:13])
    axcbar_B = fig.add_subplot(gs[20, 2:13])

    if xg is None:
        xg = np.arange(proj_R.shape[0])
    if yg is None:
        yg = np.arange(proj_R.shape[1])

    if Rlog:
        proj_R = np.log10(proj_R)
    if Glog:
        proj_G = np.log10(proj_G)
    if Blog:
        proj_B = np.log10(proj_B)

    if Rmin is None:
        Rmin = proj_R.min()
    if Rmax is None:
        Rmax = proj_R.max()
    if Gmin is None:
        Gmin = proj_G.min()
    if Gmax is None:
        Gmax = proj_G.max()
    if Bmin is None:
        Bmin = proj_B.min()
    if Bmax is None:
        Bmax = proj_B.max()

    proj_R = np.clip((proj_R - Rmin) / (Rmax-Rmin), 0, 1) * Rscale
    proj_G = np.clip((proj_G - Gmin) / (Gmax-Gmin), 0, 1) * Gscale
    proj_B = np.clip((proj_B - Bmin) / (Bmax-Bmin), 0, 1) * Bscale

    if sigma_R > 0:
        proj_R = gaussian_filter(proj_R, sigma=sigma_R)
    if sigma_G > 0:
        proj_G = gaussian_filter(proj_G, sigma=sigma_G)
    if sigma_B > 0:
        proj_B = gaussian_filter(proj_B, sigma=sigma_B)

    proj_RGBA = np.zeros((*proj_R.shape, 4))
    proj_RGBA[:,:,3]=1. # alpha channel

    proj_RGBA[:,:,0] = proj_R * Rcolor[0] + proj_G * Gcolor[0] + proj_B * Bcolor[0]
    proj_RGBA[:,:,1] = proj_R * Rcolor[1] + proj_G * Gcolor[1] + proj_B * Bcolor[1]
    proj_RGBA[:,:,2] = proj_R * Rcolor[2] + proj_G * Gcolor[2] + proj_B * Bcolor[2]

    proj_RGBA = np.clip(proj_RGBA, 0, 1)

    ax.imshow(proj_RGBA, origin='lower', extent=[xg.min(), xg.max(), yg.min(), yg.max()])
    ax.set_aspect('auto')

    # set labels
    ax.set_xlabel(xlabel, fontsize=axisfont)
    ax.set_ylabel(ylabel, fontsize=axisfont)
    ax.set_title(title, fontsize=titlefont)
    ax.tick_params(axis='both', which='major', labelsize=ticksfont)

    if remove_xy_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
    else:
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)

    # plot the 3 1d colorbars below

    # R colorbar
    array = np.linspace(Rmin, Rmax, 100)


    image = np.zeros((1, 100, 4))
    image[0,:,0] = np.clip((array - Rmin) / (Rmax-Rmin), 0, 1) * Rscale * Rcolor[0]
    image[0,:,1] = np.clip((array - Rmin) / (Rmax-Rmin), 0, 1) * Rscale * Rcolor[1]
    image[0,:,2] = np.clip((array - Rmin) / (Rmax-Rmin), 0, 1) * Rscale * Rcolor[2]
    image[0,:,3] = 1.

    axcbar_R.imshow(image, origin='lower', extent=[Rmin, Rmax, 0, 1])
    axcbar_R.set_aspect('auto')

    if Rticks is not None:
        axcbar_R.set_xticks(Rticks)

    axcbar_R.set_xlabel(Rlabel, fontsize=axisfont)
    axcbar_R.set_yticks([])
    axcbar_R.tick_params(axis='both', which='major', labelsize=ticksfont)

    # G colorbar
    array = np.linspace(Gmin, Gmax, 100)


    image = np.zeros((1, 100, 4))
    image[0,:,0] = np.clip((array - Gmin) / (Gmax-Gmin), 0, 1) * Gscale * Gcolor[0]
    image[0,:,1] = np.clip((array - Gmin) / (Gmax-Gmin), 0, 1) * Gscale * Gcolor[1]
    image[0,:,2] = np.clip((array - Gmin) / (Gmax-Gmin), 0, 1) * Gscale * Gcolor[2]
    image[0,:,3] = 1.

    axcbar_G.imshow(image, origin='lower', extent=[Gmin, Gmax, 0, 1])
    axcbar_G.set_aspect('auto')

    if Gticks is not None:
        axcbar_G.set_xticks(Gticks)

    axcbar_G.set_xlabel(Glabel, fontsize=axisfont)
    axcbar_G.set_yticks([])
    axcbar_G.tick_params(axis='both', which='major', labelsize=ticksfont)

    # B colorbar
    array = np.linspace(Bmin, Bmax, 100)

    image = np.zeros((1, 100, 4))
    image[0,:,0] = np.clip((array - Bmin) / (Bmax-Bmin), 0, 1) * Bscale * Bcolor[0]
    image[0,:,1] = np.clip((array - Bmin) / (Bmax-Bmin), 0, 1) * Bscale * Bcolor[1]
    image[0,:,2] = np.clip((array - Bmin) / (Bmax-Bmin), 0, 1) * Bscale * Bcolor[2]
    image[0,:,3] = 1.

    axcbar_B.imshow(image, origin='lower', extent=[Bmin, Bmax, 0, 1])
    axcbar_B.set_aspect('auto')

    if Bticks is not None:
        axcbar_B.set_xticks(Bticks)
    
    axcbar_B.set_xlabel(Blabel, fontsize=axisfont)
    axcbar_B.set_yticks([])
    axcbar_B.tick_params(axis='both', which='major', labelsize=ticksfont)

    # prevent axes colliding
    fig.tight_layout()
    fig.subplots_adjust(hspace=1)

    return fig, [ax, axcbar_R, axcbar_G, axcbar_B]






def compute_projection(matrix, axis=2, slicepositions=None):
    """
    Computes the projection of a 3d numpy matrix along a principal axis.
    Args:
        matrix: the original 3d matrix
        axis: axis the projection will be computed with respect to. Defaults to 2 (z-axis)
        slicepositions: pair of index positions. If specified, the projection will only take into account the selected
        part of the matrix. (list of two ints)

    Returns:
        2d numpy array with the computed projection

    """
    if slicepositions is None:
        slicepositions = [0, 0]
    slicemin, slicemax = tuple(slicepositions)
    if slicemin == slicemax:
        return matrix.mean(axis)
    elif slicemin < slicemax:
        if axis == 0:
            matrix = matrix[slicemin:slicemax + 1, :, :]
        elif axis == 1:
            matrix = matrix[:, slicemin:slicemax + 1, :]
        elif axis == 2:
            matrix = matrix[:, :, slicemin:slicemax + 1]
        else:
            raise ValueError('axis should be 0, 1 or 2')
        return matrix.mean(axis)
    else:
        raise ValueError('slicemin should be smaller than slicemax')
    

def format_uncertainty(median, lower_err, upper_err):
    """
    Formats the median and the asymmetric errors in the form:
    median^{+upper_err}_{-lower_err} with consistent precision and scientific notation when necessary.

    Args:
        median (float): The central value (median).
        lower_err (float): The lower error (difference between the median and lower bound).
        upper_err (float): The upper error (difference between the median and upper bound).

    Returns:
        str: The formatted string in the form of median^{+upper_err}_{-lower_err}, with scientific notation if necessary.
    """
    # Find the largest error and determine its precision (2 significant figures)
    max_err = max(lower_err, upper_err)
    err_order = math.floor(math.log10(abs(max_err))) if max_err != 0 else 0
    precision = -err_order + 1  # 2 significant figures for the errors

    # Round the median and the errors to the same precision
    rounded_median = round(median, precision)
    rounded_lower = round(lower_err, precision)
    rounded_upper = round(upper_err, precision)

    # Determine if scientific notation is needed based on the order of the median
    median_order = math.floor(math.log10(abs(rounded_median))) if rounded_median != 0 else 0

    # If the median is large/small, use scientific notation
    if abs(median_order) >= 3:
        factor = 10 ** median_order
        rounded_median /= factor
        rounded_lower /= factor
        rounded_upper /= factor

        precision = max(0, precision + median_order)

        formatted = f"$({rounded_median:.{precision}f}^{{+{rounded_upper:.{precision}f}}}_{{-{rounded_lower:.{precision}f}}}) \\times 10^{{{median_order}}}$"
    else:
        precision = max(0, precision)
        formatted = f"${rounded_median:.{precision}f}^{{+{rounded_upper:.{precision}f}}}_{{-{rounded_lower:.{precision}f}}}$"


    return formatted

def format_interval(mean, lower, upper, significant_figures_error=2):
    """
    Formats the mean and the asymmetric errors in the form:
    mean^{+upper-mean}_{-(mean-lower)} with consistent precision and scientific notation when necessary.

    Args:
        mean (float): The central value.
        lower (float): The lower bound.
        upper (float): The upper bound.
        significant_figures_error (int): The number of significant figures for the errors. Defaults to 2.

    Returns:
        str: The formatted string in the form of mean^{+upper-mean}_{-(mean-lower)}, with scientific notation if necessary.
    """
    absmean = np.abs(mean)
    superior = np.abs(upper - mean)
    inferior = np.abs(mean - lower)
    if 1e-2 < absmean < 1e3: # non-scientific notation
        precision = - int(np.floor(np.log10(min(superior, inferior))))
        precision += (significant_figures_error - 1)

        mean = '{:.{prec}f}'.format(mean, prec=precision)
        superior = '{:.{prec}f}'.format(superior, prec=precision)
        inferior = '{:.{prec}f}'.format(inferior, prec=precision)
        return '$'+mean+'^{+'+superior+'}_{-'+inferior+'}$'
    else:
        oom = int(np.floor(np.log10(absmean)))
        
        mean /= 10**oom
        lower /= 10**oom
        upper /= 10**oom

        mantissa = format_interval(mean, lower, upper, significant_figures_error=significant_figures_error)[1:-1]

        return '$('+mantissa+') \\times 10^{'+str(oom)+'}$'


def cornerplot(dataset, varnames=None, units=None, logscale=None, figsize=None, labelsize=12, ticksize=10,
               title=None, s=3, color='blue', kde=True, cmap_kde='Blues', filled_kde=True,
               kde_plot_outliers=True, kde_quantiles=None, axes_limits=None):
    """
    Generates a corner plot of the dataset. For scatter plots, it uses KDE to estimate the density of the points.

    Args:
        dataset: n x m numpy array, where n is the number of samples and m the number of variables.
        varnames: list of m strings with the names of the variables. If not specified, it will place no labels.
        units: list of m strings with the units of the variables. If not specified, it will place no units.
        logscale: list of m booleans, indicating if the variables should be plotted in logscale
        figsize: tuple with the dimensions of the figure
        labelsize: size of the labels
        ticksize: size of the ticks
        title: title of the plot
        s: size of the scatter points
        color: color of the histograms and scatter plots
        kde: whether to plot the KDE or not
        cmap_kde: colormap for the KDE plot
        filled_kde: whether to fill the KDE plot or not
        kde_plot_outliers: whether to plot the outliers in the KDE plot or not
        kde_quantiles: if None, it will just plot the KDE. If a list of quantiles is given, it will plot the
         filled contour plot of the KDE with the specified quantiles (and not the KDE itself).
        axes_limits: list of m tuples with the limits of the axes. 
            If not specified, it will use the data limits.
            Optionally, if a single float (0<f<1), it will discard this fraction from the extremes of the data.
    
    Returns:
        figure object and grid of axes with the corner plot
    """
    m = dataset.shape[1]
    n = dataset.shape[0]

    if varnames is None:
        varnames = [''] * m
    else:
        if len(varnames) != m:
            raise ValueError('varnames should have the same length as the number of variables')
    if units is None:
        units = [''] * m
    else:
        if len(units) != m:
            raise ValueError('units should have the same length as the number of variables')
    if logscale is None:
        logscale = [False] * m
    else:
        if len(logscale) != m:
            raise ValueError('logscale should have the same length as the number of variables')
    if figsize is None:
        figsize = (2.5 * m, 2.5 * m)

    if isinstance(axes_limits, list):
        if len(axes_limits) != m:
            raise ValueError('axes_limits should have the same length as the number of variables')
    elif isinstance(axes_limits, float):
        axes_limits = [np.percentile(dataset[:, i], [100*axes_limits/2, 100*(1-axes_limits/2)]) for i in range(m)]
    else:
        axes_limits = []
        for i in range(m):
            vmin = np.min(dataset[:, i])
            vmax = np.max(dataset[:, i])
            ext = 0.1 * (vmax - vmin) / 2
            axes_limits.append((vmin - ext, vmax + ext))
    
    fig, axes = plt.subplots(m, m, figsize=figsize)

    for i in range(m):
        for j in range(m):
            if i == j:
                sns.histplot(dataset[:, i], kde=True, ax=axes[i, j], log_scale=logscale[i], color=color)

                axes[i, j].tick_params(axis='both', which='major', labelsize=ticksize)
                axes[i, j].set_yticks([])
                axes[i, j].set_ylabel('', fontsize=labelsize)
                if i != m - 1:
                    axes[i, j].tick_params(axis='x', which='both', bottom=True, labelbottom=False)
                if logscale[i]:
                    axes[i, j].set_xscale('log')
                    axes[i, j].xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
                    axes[i, j].xaxis.get_offset_text().set_size(labelsize)  # Adjust offset text size if needed
                    axes[i, j].xaxis.get_offset_text().set_position((1.0, 0))  # Position the offset to the right
                    axes[i, j].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                if i == m - 1:
                    axes[i, j].set_xlabel(varnames[i], fontsize=ticksize)

                # string with median and 16-84 percentiles as upper and lower errors (written with super and subindices)
                # if there is scientific notation, it has to be written as (10.1 + 2.3 - 1.2) x 10^3
                median = np.median(dataset[:, i])
                upper = np.percentile(dataset[:, i], 84) - median
                lower = median - np.percentile(dataset[:, i], 16)

                axes[i,j].set_title(varnames[i] + ' = ' + format_uncertainty(median, lower, upper) + '$\;$' + units[i], 
                                    fontsize=labelsize)

            elif i > j:
                if not kde:
                    sns.scatterplot(x=dataset[:, j], y=dataset[:, i], ax=axes[i, j], s=s, color=color)
                else: # kde plot + scatter for outliers
                    thr_kde = 0.2
                    dens_func = gaussian_kde(np.vstack([dataset[:, j], dataset[:, i]]))
                    dens = dens_func(np.vstack([dataset[:, j], dataset[:, i]]))
                    
                    if kde_quantiles is None:
                        try:
                            sns.kdeplot(x=dataset[:, j], y=dataset[:, i], ax=axes[i, j], cmap=cmap_kde, fill=filled_kde, thresh=thr_kde, levels=100, gridsize=100)
                        except ValueError: 
                            error = True 
                            fac = 1
                            while error:
                                try:
                                    fac *= 2
                                    sns.kdeplot(x=dataset[:, j], y=dataset[:, i], ax=axes[i, j], cmap=cmap_kde, fill=filled_kde, thresh=thr_kde, levels=100//fac, gridsize=100)
                                    error = False 
                                except ValueError:
                                    continue
                        #raise ValueError('Not implemented')

                        if kde_plot_outliers:
                            mask = dens < np.percentile(dens, 100*thr_kde)
                            sns.scatterplot(x=dataset[mask, j], y=dataset[mask, i], ax=axes[i, j], s=s, color=color)
                    else:
                        # I already have dens, therefore:
                        # I have to find the percentiles of the density
                        kde_quantiles = [0] + kde_quantiles
                        kde_quantiles = np.sort(kde_quantiles)[::-1]

                        xgrid = np.linspace(axes_limits[j][0], axes_limits[j][1], 200)
                        ygrid = np.linspace(axes_limits[i][0], axes_limits[i][1], 200)
                        xi, yi = np.meshgrid(xgrid, ygrid)
                        zi = dens_func(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)

                        levels = [np.percentile(dens, 100*(1-quant)) for quant in kde_quantiles]

                        # contour with external borders
                        discrete_cmap = ListedColormap([cm.get_cmap(cmap_kde)(i) for i in np.linspace(0, 1, len(levels))])
                        axes[i, j].contourf(xi, yi, zi, levels=levels, cmap=discrete_cmap, alpha=0.5, vmin=min(levels), vmax=max(levels))
                        axes[i,j].contour(xi, yi, zi, levels=levels, colors='k', alpha=0.5, linewidths=1)

                        if kde_plot_outliers:
                            mask = dens < np.percentile(dens, 100*(1-max(kde_quantiles)))
                            sns.scatterplot(x=dataset[mask, j], y=dataset[mask, i], ax=axes[i, j], s=s, color=color)



                if i==m-1:
                    axes[i, j].set_xlabel(varnames[j], fontsize=labelsize)
                if j==0:
                    axes[i, j].set_ylabel(varnames[i], fontsize=labelsize)
                
                axes[i, j].tick_params(axis='both', which='major', labelsize=ticksize)
                
                if i != m - 1:
                    axes[i, j].tick_params(axis='x', which='both', bottom=True, labelbottom=False)
                if j != 0:
                    axes[i, j].tick_params(axis='y', which='both', left=True, labelleft=False)
                
                if logscale[i]:
                    #print('logscale y', i, j)
                    axes[i, j].set_yscale('log')
                    axes[i, j].yaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True, useOffset=True))
                    axes[i, j].yaxis.get_offset_text().set_size(ticksize)
                    axes[i, j].yaxis.get_offset_text().set_position((0, 1.0))
                    axes[i, j].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

                    axes[i, j].yaxis.set_minor_locator(MaxNLocator(5))

                    # put minor ticks only if there are no major ticks
                    formatter = mticker.ScalarFormatter(useMathText=True, useOffset=True)
                    formatter.set_powerlimits((0, 0))
                    axes[i, j].yaxis.set_major_formatter(formatter)
                    axes[i, j].yaxis.set_minor_formatter(formatter)
                if logscale[j]:
                    #print('logscale x', i, j)
                    axes[i, j].set_xscale('log')
                    axes[i, j].xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True, useOffset=True))
                    axes[i, j].xaxis.get_offset_text().set_size(ticksize)  # Adjust offset text size if needed
                    axes[i, j].xaxis.get_offset_text().set_position((1.0, 0))  # Position the offset to the right
                    axes[i, j].ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
                    
                    axes[i, j].xaxis.set_minor_locator(MaxNLocator(5))  # Limit number of ticks

                    formatter = mticker.ScalarFormatter(useMathText=True, useOffset=True)
                    formatter.set_powerlimits((0, 0))  # Force scientific notation
                    axes[i, j].xaxis.set_major_formatter(formatter)
                    axes[i, j].xaxis.set_minor_formatter(formatter) 
            else:
                axes[i, j].axis('off')

    # set axes limits
    for i in range(m):
        for j in range(m):
            if i > j:
                axes[i, j].set_xlim(axes_limits[j])
                axes[i, j].set_ylim(axes_limits[i])
            if i == j:
                axes[i, j].set_xlim(axes_limits[j])

    if title is not None:
        fig.suptitle(title, fontsize=labelsize)

    fig.tight_layout()
    fig.bbox_inches = 'tight'

    return fig, axes
    
def slice_map(field, normal_vector, north_vector, 
              xc, yc, zc, 
              npatch,patchnx,patchny,patchnz,patchrx,patchry,patchrz,size,nmax,nl,
              widthN=None, widthE=None, res=None, resN=None, resE=None, nN=None, nE=None,
              kept_patches=None,
              interpolate=True,
              return_grid=False, return_grid_3d=False,
              use_tqdm=True):
    """ 
    Slice a 3D AMR field along a plane defined by a normal vector and a point (center of the slice).

    Args:
        field: field to be computed at the uniform grid, in the usual MASCLET style (list of 3d np.arrays). MUST BE UNCLEANED!!
        normal_vector (3-element list or 1d np.array): The normal vector to the plane. Points towards the observer.
        north_vector (3-element list or 1d np.array): The vector pointing north in the plane.
        xc, yc, zc (float): The coordinates of the center of the slice.
        npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
                                   (and Y and Z)
        size: comoving side of the simulation box
        nmax: cells at base level
        nl: number of AMR levels (max AMR level to be considered)

        Two of the following three must be specified:
            - widthN, widthE (float): The width of the slice in the north and east directions, respectively. Must be in the same units as the coordinates.
            - res (float), or resN and resE: The resolution of the slice. Must be in the same units as the coordinates.
            - nN, nE (int): The number of points in the north and east directions, respectively.

        kept_patches: list of patches that are read. If None, all patches are assumed to be present
        interpolate (bool): If True, the slice will be interpolated. If False, the slice will be a plane.
        return_grid (bool): If True, the function will return the grid of the slice (2d: east and north directions).
        return_grid_3d (bool): If True, the function will return the simulation coordinates of the slice pixels (3d: x, y, z).
        use_tqdm (bool): If True, a progress bar will be shown.

    Returns:
        2D numpy array with the computed projection.
        If return_grid is True, it will return two 1D numpy arrays with the coordinates of the grid (E and N directions).
    """

    # Vectors to np array
    if type(normal_vector) is list:
        normal_vector = np.array(normal_vector)
    if type(north_vector) is list:
        north_vector = np.array(north_vector)

    # normalize vectors
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    north_vector = north_vector / np.linalg.norm(north_vector)

    if np.dot(normal_vector, north_vector) != 0:
        raise ValueError("The normal and north vectors must be orthogonal")
    east_vector = np.cross(normal_vector, north_vector)

    if res is not None:
        resN = res 
        resE = res

    # Check enough information is given
    if (widthN is None) + (resN is None) + (nN is None) != 1:
        raise ValueError("Exactly two of widthN, resN (or res) and nN must be given")

    if widthN is None:
        widthN = res * nN
    elif resN is None:
        resN = widthN / nN
    else:
        nN = int(widthN // resN)

    if (widthE is None) + (resE is None) + (nE is None) != 1:
        raise ValueError("Exactly two of widthE, resE (or res) and nE must be given")

    if widthE is None:
        widthE = res * nE
    elif resE is None:
        resE = widthE / nE
    else:
        nE = int(widthE // resE)

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum()+1, dtype=bool)

    levels = tools.create_vector_levels(npatch)
    #print(type(levels), levels.shape)
    #print('NCORES=', get_num_threads())

    # Compute the grid
    # Notation: x, y, z are the coordinates in the simulation box, while N, E are the coordinates in the slice
    #@njit(parallel=True, fastmath=True)
    def parallelize(nN, nE, xc, yc, zc, resN, resE, north_vector, east_vector, size, nmax, levels, kept_patches, field):
        xgrid = np.zeros((nN, nE))
        ygrid = np.zeros((nN, nE))
        zgrid = np.zeros((nN, nE))

        imid = int(nN // 2)
        jmid = int(nE // 2)
        
        for i in prange(nN):
            for j in range(nE):
                xgrid[i, j] = xc + (i - imid) * resN * north_vector[0] + (j - jmid) * resE * east_vector[0]
                ygrid[i, j] = yc + (i - imid) * resN * north_vector[1] + (j - jmid) * resE * east_vector[1]
                zgrid[i, j] = zc + (i - imid) * resN * north_vector[2] + (j - jmid) * resE * east_vector[2]


        # Compute the projection
        proj = np.zeros((nN, nE), dtype=field[0].dtype)
        res_worst = max(resN, resE)
        #lmax = np.clip(np.log2((size/nmax)/res_worst).astype('int32'),0,nl)#+1
        lmax = int(np.log2((size/nmax)/res_worst))
        nl = levels.max()
        if lmax>nl:
            lmax = nl
        if lmax<0:
            lmax = 0
        cellsizes = size/nmax/2**levels#.astype('f4')
        

        #for i in tqdm(range(nN), disable=not use_tqdm):
        for i in prange(nN):
            for j in range(nE):
                xij = xgrid[i, j]
                yij = ygrid[i, j]
                zij = zgrid[i, j]

                ip2, ix2, jy2, kz2 = locate_point(xij,yij,zij,npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,lmax,buf=0,no_interp=True)

                interpolate_this = interpolate 
                if interpolate_this:
                    ip, ix, jy, kz = locate_point(xij,yij,zij,npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,lmax,buf=0.5)
                    if levels[ip2] > levels[ip]:
                        interpolate_this = False

                if not interpolate_this:
                    if not kept_patches[ip2]:
                        raise ValueError("Patch not read!!!")

                    n1,n2,n3 = patchnx[ip2],patchny[ip2],patchnz[ip2]
                    #if ix2 < 0 or ix2 >= n1 or jy2 < 0 or jy2 >= n2 or kz2 < 0 or kz2 >= n3:
                    #    dxpa = cellsizes[ip2]
                    #    print("outside the patch!", ip2, n1,n2,n3, ix2, jy2, kz2, 'x',xij,patchrx[ip2]-dxpa,patchrx[ip2]+(n1-1)*dxpa, 'y',yij,patchry[ip2]-dxpa,patchry[ip2]+(n2-1)*dxpa, 'z',zij,patchrz[ip2]-dxpa,patchrz[ip2]+(n3-1)*dxpa)
                    #    raise ValueError("outside the patch!")
                    if ix2 == n1: 
                        ix2 = n1 - 1
                    if jy2 == n2:
                        jy2 = n2 - 1
                    if kz2 == n3:
                        kz2 = n3 - 1
                    if ix2 < 0 or ix2 >= n1 or jy2 < 0 or jy2 >= n2 or kz2 < 0 or kz2 >= n3:
                        dxpa = cellsizes[ip2]
                        print("outside the patch!", ip2, n1,n2,n3, ix2, jy2, kz2, 'x',xij,patchrx[ip2]-dxpa,patchrx[ip2]+(n1-1)*dxpa, 'y',yij,patchry[ip2]-dxpa,patchry[ip2]+(n2-1)*dxpa, 'z',zij,patchrz[ip2]-dxpa,patchrz[ip2]+(n3-1)*dxpa)
                        raise ValueError("outside the patch!")
                    proj[i, j] = field[ip2][ix2, jy2, kz2]
                else:
                    if not kept_patches[ip]:
                        raise ValueError("Patch not read!!!")
                    #assert isinstance(ip, int)
                    dxip = cellsizes[ip]
                    dxx=(xij-(patchrx[ip]+(ix-0.5)*dxip))/dxip
                    dyy=(yij-(patchry[ip]+(jy-0.5)*dxip))/dxip
                    dzz=(zij-(patchrz[ip]+(kz-0.5)*dxip))/dxip

                    if ip==0 and (ix<=0 or jy<=0 or kz<=0 or ix>=nmax-1 or jy>=nmax-1 or kz>=nmax-1):
                        proj[i,j] = field[ip][(ix  )%nmax,(jy  )%nmax,(kz  )%nmax] *(1-dxx)*(1-dyy)*(1-dzz) \
                                + field[ip][(ix  )%nmax,(jy  )%nmax,(kz+1)%nmax] *(1-dxx)*(1-dyy)*  dzz   \
                                + field[ip][(ix  )%nmax,(jy+1)%nmax,(kz  )%nmax] *(1-dxx)*  dyy  *(1-dzz) \
                                + field[ip][(ix  )%nmax,(jy+1)%nmax,(kz+1)%nmax] *(1-dxx)*  dyy  *  dzz   \
                                + field[ip][(ix+1)%nmax,(jy  )%nmax,(kz  )%nmax] *  dxx  *(1-dyy)*(1-dzz) \
                                + field[ip][(ix+1)%nmax,(jy  )%nmax,(kz+1)%nmax] *  dxx  *(1-dyy)*  dzz   \
                                + field[ip][(ix+1)%nmax,(jy+1)%nmax,(kz  )%nmax] *  dxx  *  dyy  *(1-dzz) \
                                + field[ip][(ix+1)%nmax,(jy+1)%nmax,(kz+1)%nmax] *  dxx  *  dyy  *  dzz  
                    else:
                        proj[i,j] = field[ip][ix  , jy  , kz  ] *(1-dxx)*(1-dyy)*(1-dzz) \
                                + field[ip][ix  , jy  , kz+1] *(1-dxx)*(1-dyy)*  dzz   \
                                + field[ip][ix  , jy+1, kz  ] *(1-dxx)*  dyy  *(1-dzz) \
                                + field[ip][ix  , jy+1, kz+1] *(1-dxx)*  dyy  *  dzz   \
                                + field[ip][ix+1, jy  , kz  ] *  dxx  *(1-dyy)*(1-dzz) \
                                + field[ip][ix+1, jy  , kz+1] *  dxx  *(1-dyy)*  dzz   \
                                + field[ip][ix+1, jy+1, kz  ] *  dxx  *  dyy  *(1-dzz) \
                                + field[ip][ix+1, jy+1, kz+1] *  dxx  *  dyy  *  dzz  

        # reorient the projection array: east is the first index increasing, north is the second index increasing
        proj = np.transpose(np.flipud(proj))
        xgrid = np.transpose(np.flipud(xgrid))
        ygrid = np.transpose(np.flipud(ygrid))
        zgrid = np.transpose(np.flipud(zgrid))

        return proj, xgrid, ygrid, zgrid

    proj, xgrid, ygrid, zgrid = parallelize(nN, nE, xc, yc, zc, resN, resE, north_vector, east_vector, size, nmax, levels, kept_patches, 
                                            List([f if ki else np.zeros((2,2,2), dtype=field[0].dtype, order='F') for f,ki in zip(field, kept_patches)]))


    # mirror proj 
    proj = np.flip(proj)

    return_vars = [proj]


    if return_grid:
        Ngrid = np.zeros(nN)
        Egrid = np.zeros(nE)

        imid = int(nN // 2)
        jmid = int(nE // 2)

        for i in range(nN):
            Ngrid[i] = (i - imid) * resN
        for j in range(nE):
            Egrid[j] = (j - jmid) * resE

        return_vars.append(Egrid)
        return_vars.append(Ngrid)

    if return_grid_3d:
        return_vars.append(xgrid)
        return_vars.append(ygrid)
        return_vars.append(zgrid)

    return tuple(return_vars)


from masclet_framework import tools 
from masclet_framework.profiles import locate_point
from numba import njit, prange
from numba.typed import List

def projection_map(field, normal_vector, north_vector, 
              xc, yc, zc, projection_region,
              npatch,patchnx,patchny,patchnz,patchrx,patchry,patchrz,size,nmax,nl,
              weight_field=None,
              widthN=None, widthE=None, res=None, resN=None, resE=None, nN=None, nE=None,
              width_normal=None, res_normal=None, n_normal=None,
              kept_patches=None,
              interpolate=True,
              return_grid=False, return_grid_3d=False,
              use_tqdm=True):
    """ 
    Obtain a projection of a 3D AMR field along a line of sight defined by a normal vector, with a centre and a north vector.

    Args:
        field: field to be computed at the uniform grid, in the usual MASCLET style (list of 3d np.arrays). MUST BE UNCLEANED!!
        normal_vector (3-element list or 1d np.array): The normal vector to the plane. Points towards the observer.
        north_vector (3-element list or 1d np.array): The vector pointing north in the plane.
        xc, yc, zc (float): The coordinates of the center of the slice.
        projection_region: The region to be projected. Can be:
            - a sphere: ("sphere", radius)
            - a box/cylinder: ("cylinder"). The projection span is set then by width_normal

        npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
                                   (and Y and Z)
        size: comoving side of the simulation box
        nmax: cells at base level
        nl: number of AMR levels (max AMR level to be considered)

        weight_field: field to be used as weight in the projection. If None, the projection will be unweighted, i.e. 
                        the quantity will be averaged over the line of sight: \int field dl / \int dl.
                    - If a field is given, then the quantity will be weighted as \int field * weight_field dl / \int weight_field dl

        Two of the following three must be specified:
            - widthN, widthE (float): The width of the slice in the north and east directions, respectively. Must be in the same units as the coordinates.
            - res (float), or resN and resE: The resolution of the slice. Must be in the same units as the coordinates.
            - nN, nE (int): The number of points in the north and east directions, respectively.
        
        width_normal (float): The width of the projection region in the normal direction. Must be in the same units as the coordinates.
            This is only used if projection_region is "cylinder".
        
        One of the following two must be specified:
            - res_normal (float): The resolution of the projection region in the normal direction. Must be in the same units as the coordinates.
            - n_normal (int): The number of points in the normal direction

        kept_patches: list of patches that are read. If None, all patches are assumed to be present
        interpolate (bool): If True, the slice will be interpolated. If False, the slice will be a plane.
        return_grid (bool): If True, the function will return the grid of the slice (2d: east and north directions).
        return_grid_3d (bool): If True, the function will return the simulation coordinates of the slice pixels (3d: x, y, z).
        use_tqdm (bool): If True, a progress bar will be shown.

    Returns:
        2D numpy array with the computed projection.
        If return_grid is True, it will return two 1D numpy arrays with the coordinates of the grid (E and N directions).
    """
    if return_grid_3d:
        return NotImplementedError("return_grid_3d not implemented yet, probably never will")

    # Check the projection region:
    projection_volume = projection_region[0]
    is_sphere=False
    if projection_volume == "sphere":
        is_sphere=True
        radius = projection_region[1]
        width_normal = 2*radius
        radius2 = radius**2
    elif projection_volume == "cylinder":
        if width_normal is None:
            raise ValueError("The width_normal must be given for a cylinder projection region")
    else:
        raise ValueError("The projection region must be either 'sphere' or 'cylinder'")

    if res_normal is not None:
        n_normal = int(width_normal // res_normal)
    elif n_normal is not None:
        res_normal = width_normal / n_normal
    else:
        raise ValueError("Exactly one of res_normal and n_normal must be given")

    # Check the weight field: 
    if weight_field is None:
        weight_field = [np.ones_like(f) if isinstance(f, np.ndarray) else 0 for f in field]

    # Vectors to np array
    if type(normal_vector) is list:
        normal_vector = np.array(normal_vector)
    if type(north_vector) is list:
        north_vector = np.array(north_vector)

    # normalize vectors
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    north_vector = north_vector / np.linalg.norm(north_vector)

    if np.dot(normal_vector, north_vector) != 0:
        raise ValueError("The normal and north vectors must be orthogonal")
    east_vector = np.cross(normal_vector, north_vector)

    if res is not None:
        resN = res 
        resE = res

    # Check enough information is given
    if (widthN is None) + (resN is None) + (nN is None) != 1:
        raise ValueError("Exactly two of widthN, resN (or res) and nN must be given")

    if widthN is None:
        widthN = res * nN
    elif resN is None:
        resN = widthN / nN
    else:
        nN = int(widthN // resN)

    if (widthE is None) + (resE is None) + (nE is None) != 1:
        raise ValueError("Exactly two of widthE, resE (or res) and nE must be given")

    if widthE is None:
        widthE = res * nE
    elif resE is None:
        resE = widthE / nE
    else:
        nE = int(widthE // resE)

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum()+1, dtype=bool)

    levels = tools.create_vector_levels(npatch)

    # Compute the grid
    # Notation: x, y, z are the coordinates in the simulation box, while N, E are the coordinates in the slice
    @njit(parallel=True, fastmath=True)
    def parallelize(nN, nE, xc, yc, zc, resN, resE, north_vector, east_vector, size, nmax, levels, kept_patches, field, wfield,
                    normal_vector, res_normal):
        # weight the field without affecting the original field in the outer scope
        field = [f * w for f, w in zip(field, wfield)]

        imid = int(nN // 2)
        jmid = int(nE // 2)
        kmid = int(n_normal // 2)

        # Compute the projection
        proj = np.zeros((nN, nE), dtype=field[0].dtype)
        proj_weight = np.zeros((nN, nE), dtype=wfield[0].dtype)
        res_worst = max(resN, resE, res_normal)
        #lmax = np.clip(np.log2((size/nmax)/res_worst).astype('int32'),0,nl)#+1
        lmax = int(np.log2((size/nmax)/res_worst))
        #print(lmax)
        nl = levels.max()
        if lmax>nl:
            lmax = nl
        if lmax<0:
            lmax = 0
        cellsizes = size/nmax/2**levels
        

        for i in prange(nN):
            for j in range(nE):
                for k in range(n_normal):
                    xij = xc + (i - imid) * resN * north_vector[0] + (j - jmid) * resE * east_vector[0] + (k - kmid) * res_normal * normal_vector[0]
                    yij = yc + (i - imid) * resN * north_vector[1] + (j - jmid) * resE * east_vector[1] + (k - kmid) * res_normal * normal_vector[1]
                    zij = zc + (i - imid) * resN * north_vector[2] + (j - jmid) * resE * east_vector[2] + (k - kmid) * res_normal * normal_vector[2]

                    if is_sphere:
                        if (xij-xc)**2 + (yij-yc)**2 + (zij-zc)**2 > radius2:
                            continue

                    ip2, ix2, jy2, kz2 = locate_point(xij,yij,zij,npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,lmax,buf=0,no_interp=True)

                    interpolate_this = interpolate 
                    if interpolate_this:
                        ip, ix, jy, kz = locate_point(xij,yij,zij,npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,lmax,buf=0.5)
                        if levels[ip2] > levels[ip]:
                            interpolate_this = False

                    if not interpolate_this:
                        if not kept_patches[ip]:
                            raise ValueError("Patch not read!!!")

                        proj[i, j] += field[ip2][ix2, jy2, kz2] 
                        proj_weight[i, j] += wfield[ip2][ix2, jy2, kz2]
                    else:
                        if not kept_patches[ip]:
                            raise ValueError("Patch not read!!!")

                        dx_l = cellsizes[ip]
                        dxx=(xij-(patchrx[ip]+(ix-0.5)*dx_l))/dx_l
                        dyy=(yij-(patchry[ip]+(jy-0.5)*dx_l))/dx_l
                        dzz=(zij-(patchrz[ip]+(kz-0.5)*dx_l))/dx_l

                        if ip==0 and (ix<=0 or jy<=0 or kz<=0 or ix>=nmax-1 or jy>=nmax-1 or kz>=nmax-1):
                            proj[i,j] += field[ip][(ix  )%nmax,(jy  )%nmax,(kz  )%nmax] *(1-dxx)*(1-dyy)*(1-dzz) \
                                    + field[ip][(ix  )%nmax,(jy  )%nmax,(kz+1)%nmax] *(1-dxx)*(1-dyy)*  dzz   \
                                    + field[ip][(ix  )%nmax,(jy+1)%nmax,(kz  )%nmax] *(1-dxx)*  dyy  *(1-dzz) \
                                    + field[ip][(ix  )%nmax,(jy+1)%nmax,(kz+1)%nmax] *(1-dxx)*  dyy  *  dzz   \
                                    + field[ip][(ix+1)%nmax,(jy  )%nmax,(kz  )%nmax] *  dxx  *(1-dyy)*(1-dzz) \
                                    + field[ip][(ix+1)%nmax,(jy  )%nmax,(kz+1)%nmax] *  dxx  *(1-dyy)*  dzz   \
                                    + field[ip][(ix+1)%nmax,(jy+1)%nmax,(kz  )%nmax] *  dxx  *  dyy  *(1-dzz) \
                                    + field[ip][(ix+1)%nmax,(jy+1)%nmax,(kz+1)%nmax] *  dxx  *  dyy  *  dzz  
                            proj_weight[i,j] += wfield[ip][(ix  )%nmax,(jy  )%nmax,(kz  )%nmax] *(1-dxx)*(1-dyy)*(1-dzz) \
                                              + wfield[ip][(ix  )%nmax,(jy  )%nmax,(kz+1)%nmax] *(1-dxx)*(1-dyy)*  dzz   \
                                              + wfield[ip][(ix  )%nmax,(jy+1)%nmax,(kz  )%nmax] *(1-dxx)*  dyy  *(1-dzz) \
                                              + wfield[ip][(ix  )%nmax,(jy+1)%nmax,(kz+1)%nmax] *(1-dxx)*  dyy  *  dzz   \
                                              + wfield[ip][(ix+1)%nmax,(jy  )%nmax,(kz  )%nmax] *  dxx  *(1-dyy)*(1-dzz) \
                                              + wfield[ip][(ix+1)%nmax,(jy  )%nmax,(kz+1)%nmax] *  dxx  *(1-dyy)*  dzz   \
                                              + wfield[ip][(ix+1)%nmax,(jy+1)%nmax,(kz  )%nmax] *  dxx  *  dyy  *(1-dzz) \
                                              + wfield[ip][(ix+1)%nmax,(jy+1)%nmax,(kz+1)%nmax] *  dxx  *  dyy  *  dzz  
                        else:
                            proj[i,j] += field[ip][ix  , jy  , kz  ] *(1-dxx)*(1-dyy)*(1-dzz) \
                                    + field[ip][ix  , jy  , kz+1] *(1-dxx)*(1-dyy)*  dzz   \
                                    + field[ip][ix  , jy+1, kz  ] *(1-dxx)*  dyy  *(1-dzz) \
                                    + field[ip][ix  , jy+1, kz+1] *(1-dxx)*  dyy  *  dzz   \
                                    + field[ip][ix+1, jy  , kz  ] *  dxx  *(1-dyy)*(1-dzz) \
                                    + field[ip][ix+1, jy  , kz+1] *  dxx  *(1-dyy)*  dzz   \
                                    + field[ip][ix+1, jy+1, kz  ] *  dxx  *  dyy  *(1-dzz) \
                                    + field[ip][ix+1, jy+1, kz+1] *  dxx  *  dyy  *  dzz  
                            proj_weight[i,j] += wfield[ip][ix  , jy  , kz  ] *(1-dxx)*(1-dyy)*(1-dzz) \
                                              + wfield[ip][ix  , jy  , kz+1] *(1-dxx)*(1-dyy)*  dzz   \
                                              + wfield[ip][ix  , jy+1, kz  ] *(1-dxx)*  dyy  *(1-dzz) \
                                              + wfield[ip][ix  , jy+1, kz+1] *(1-dxx)*  dyy  *  dzz   \
                                              + wfield[ip][ix+1, jy  , kz  ] *  dxx  *(1-dyy)*(1-dzz) \
                                              + wfield[ip][ix+1, jy  , kz+1] *  dxx  *(1-dyy)*  dzz   \
                                              + wfield[ip][ix+1, jy+1, kz  ] *  dxx  *  dyy  *(1-dzz) \
                                              + wfield[ip][ix+1, jy+1, kz+1] *  dxx  *  dyy  *  dzz  

        #proj_weight[proj_weight==0] = 1
        proj /= proj_weight
        
        # reorient the projection array: east is the first index increasing, north is the second index increasing
        proj = np.transpose(np.flipud(proj))

        return proj

    proj = parallelize(nN, nE, xc, yc, zc, resN, resE, north_vector, east_vector, size, nmax, levels, kept_patches, 
                                            List([f if ki else np.zeros((2,2,2), dtype=field[0].dtype, order='F') for f,ki in zip(field, kept_patches)]),
                                            List([f if ki else np.zeros((2,2,2), dtype=field[0].dtype, order='F') for f,ki in zip(weight_field, kept_patches)]),
                                            normal_vector, res_normal)


    # mirror proj 
    proj = np.flip(proj)

    return_vars = [proj]


    if return_grid:
        Ngrid = np.zeros(nN)
        Egrid = np.zeros(nE)

        imid = int(nN // 2)
        jmid = int(nE // 2)

        for i in range(nN):
            Ngrid[i] = (i - imid) * resN
        for j in range(nE):
            Egrid[j] = (j - jmid) * resE

        return_vars.append(Egrid)
        return_vars.append(Ngrid)

    return tuple(return_vars)



def projection_map_polars(field, normal_vector,
              xc, yc, zc, projection_region,
              npatch,patchnx,patchny,patchnz,patchrx,patchry,patchrz,size,nmax,nl,
              weight_field=None,
              binsr=None, rmin=None, rmax=None, dex_rbins=None, delta_rbins=None,
              binsphi=None, Nphi=None, east_vector=None,
              width_normal=None, res_normal=None, n_normal=None,
              kept_patches=None,
              interpolate=True,
              return_grid=False, return_grid_3d=False,
              use_tqdm=True):
    """ 
    Obtain a projection of a 3D AMR field along a line of sight defined by a normal vector, with a centre.
    Unlike projection_map, this function will return a polar projection, with the radial bins defined by the user.
    This is especially useful, for instance, if one wants to get logarithmically spaced radial bins, where a cartesian
    projection would be inefficient.

    Args:
        field: field to be computed at the uniform grid, in the usual MASCLET style (list of 3d np.arrays). MUST BE UNCLEANED!!
        normal_vector (3-element list or 1d np.array): The normal vector to the plane. Points towards the observer.
        xc, yc, zc (float): The coordinates of the center of the slice.
        projection_region: The region to be projected. Can be:
            - a sphere: ("sphere", radius)
            - a box/cylinder: ("cylinder", width_normal). The projection span is set then by width_normal.
                The parameter here overrides the width_normal parameter below.

        npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
                                   (and Y and Z)
        size: comoving side of the simulation box
        nmax: cells at base level
        nl: number of AMR levels (max AMR level to be considered)

        weight_field: field to be used as weight in the projection. If None, the projection will be unweighted, i.e. 
                        the quantity will be averaged over the line of sight: \int field dl / \int dl.
                    - If a field is given, then the quantity will be weighted as \int field * weight_field dl / \int weight_field dl

        There are several ways to specify the radial bins:
            - binsr: numpy vector specifying the radial bin edges
            - rmin, rmax, dex_rbins: minimum and maximum radius, and logarithmic bin size
            - rmin, rmax, delta_rbins: minimum and maximum radius, and linear bin size

        Likewise, there are several ways to specify the angular bins:
            - binsphi: numpy vector specifying the angular bin edges
            - Nphi: number of angular bins

        east_vector: The vector pointing east in the plane. If None, it will be chosen arbitrarily.
        
        width_normal (float): The width of the projection region in the normal direction. Must be in the same units as the coordinates.
            This is only used if projection_region is "cylinder".
        
        One of the following two must be specified:
            - res_normal (float): The resolution of the projection region in the normal direction. Must be in the same units as the coordinates.
            - n_normal (int): The number of points in the normal direction

        kept_patches: list of patches that are read. If None, all patches are assumed to be present
        interpolate (bool): If True, the slice will be interpolated. If False, the slice will be a plane.
        return_grid (bool): If True, the function will return the grid of the slice (2d: east and north directions).
        return_grid_3d (bool): If True, the function will return the simulation coordinates of the slice pixels (3d: x, y, z).
        use_tqdm (bool): If True, a progress bar will be shown.

    Returns:
        2D numpy array with the computed projection.
        If return_grid is True, it will return two 1D numpy arrays with the coordinates of the grid (phi and r directions).
    """
    if return_grid_3d:
        return NotImplementedError("return_grid_3d not implemented yet, probably never will")

    # Check the projection region:
    projection_volume = projection_region[0]
    is_sphere=False

    if projection_volume == "sphere":
        is_sphere=True
        radius = projection_region[1]
        width_normal = 2*radius
        radius2 = radius**2
    elif projection_volume == "cylinder":
        width_normal = projection_region[1]
    else:
        raise ValueError("The projection region must be either 'sphere' or 'cylinder'")

    if res_normal is not None:
        n_normal = int(width_normal // res_normal)
    elif n_normal is not None:
        res_normal = width_normal / n_normal
    else:
        raise ValueError("Exactly one of res_normal and n_normal must be given")

    # Check the weight field: 
    if weight_field is None:
        weight_field = [np.ones_like(f) if isinstance(f, np.ndarray) else 0 for f in field]

    # Normal vector to np array
    if type(normal_vector) is list:
        normal_vector = np.array(normal_vector)
    # normalize vector
    normal_vector = normal_vector / np.linalg.norm(normal_vector)


    # Check r bins are properly specified
    if type(binsr) is np.ndarray or type(binsr) is list:
        nR=len(binsr)
        if type(binsr) is list:
            binsr=np.array(binsr)
    elif (rmin is not None) and (rmax is not None) and ((dex_rbins is not None) or (delta_rbins is not None)) and ((dex_rbins is None) or (delta_rbins is None)):
        if dex_rbins is not None:
            nR=int(np.log10(rmax/rmin)/dex_rbins/2)*2+1 # guarantee it is odd
            binsr = np.logspace(np.log10(rmin),np.log10(rmax),nR)
        else:
            nR=int((rmax-rmin)/delta_rbins/2)*2+1 # guarantee it is odd
            binsr = np.linspace(rmin,rmax,nR)
    else:
        raise ValueError('Wrong specification of binsr') 

    # Check phi bins are properly specified
    if type(binsphi) is np.ndarray or type(binsphi) is list:
        nPhi=len(binsphi)
        if type(binsphi) is list:
            binsphi=np.array(binsphi)
    elif Nphi is not None:
        nPhi=Nphi
        binsphi=np.linspace(0,2*np.pi,nPhi+1)
    else:
        raise ValueError('Wrong specification of binsphi')

    # Vectors to np array
    if type(east_vector) is list:
        east_vector = np.array(east_vector)
        if np.dot(normal_vector, east_vector) != 0:
            raise ValueError("The normal and east vectors must be orthogonal")
    elif east_vector is None:
        # we have to make it up, from a vector orthogonal to the normal vector
        east_vector = np.cross(normal_vector, np.array([1,0,0]))
        if np.linalg.norm(east_vector) == 0:
            east_vector = np.cross(normal_vector, np.array([0,1,0]))
        east_vector = east_vector / np.linalg.norm(east_vector)

    north_vector = np.cross(east_vector, normal_vector)

    if kept_patches is None:
        kept_patches = np.ones(npatch.sum()+1, dtype=bool)

    levels = tools.create_vector_levels(npatch)

    # Compute the grid
    # Notation: x, y, z are the coordinates in the simulation box, while N, E are the coordinates in the slice
    @njit(parallel=True, fastmath=True)
    def parallelize(xc, yc, zc, binsr, binsphi, north_vector, east_vector, size, nmax, levels, kept_patches, field, wfield,
                    normal_vector, res_normal):
        # weight the field without affecting the original field in the outer scope
        field = [f * w for f, w in zip(field, wfield)]

        nR = len(binsr)
        nPhi = len(binsphi)

        kmid = int(n_normal // 2)

        # Compute the projection
        proj = np.zeros((nPhi, nR), dtype=field[0].dtype)
        proj_weight = np.zeros((nPhi, nR), dtype=wfield[0].dtype)
        

        drrr = np.concatenate((np.array([binsr[1]-binsr[0]]), np.diff(binsr)))
        #drrr = np.clip(drrr, res_normal, np.inf) # resolution is at best res_normal
        drr[drr < res_normal] = res_normal
        nl = levels.max()
        lev_integral = np.log2((size/nmax)/drrr).astype(np.int32)
        lev_integral[lev_integral>nl] = nl
        lev_integral[lev_integral<0] = 0
        cellsizes = size/nmax/2**levels
        
        for i in prange(nPhi):
            phi = binsphi[i]
            for j in range(nR):
                r = binsr[j]
                lmax = lev_integral[j]

                xijplane = xc + r * np.cos(phi) * east_vector[0] + r * np.sin(phi) * north_vector[0]
                yijplane = yc + r * np.cos(phi) * east_vector[1] + r * np.sin(phi) * north_vector[1]
                zijplane = zc + r * np.cos(phi) * east_vector[2] + r * np.sin(phi) * north_vector[2]

                for k in range(n_normal):
                    xij = xijplane + (k - kmid) * res_normal * normal_vector[0]
                    yij = yijplane + (k - kmid) * res_normal * normal_vector[1]
                    zij = zijplane + (k - kmid) * res_normal * normal_vector[2]

                    if is_sphere:
                        if (xij-xc)**2 + (yij-yc)**2 + (zij-zc)**2 > radius2:
                            continue

                    ip2, ix2, jy2, kz2 = locate_point(xij,yij,zij,npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,lmax,buf=0,no_interp=True)

                    interpolate_this = interpolate 
                    if interpolate_this:
                        ip, ix, jy, kz = locate_point(xij,yij,zij,npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,lmax,buf=0.5)
                        if levels[ip2] > levels[ip]:
                            interpolate_this = False

                    if not interpolate:
                        if not kept_patches[ip]:
                            raise ValueError("Patch not read!!!")

                        proj[i, j] += field[ip2][ix2, jy2, kz2]
                        proj_weight[i, j] += wfield[ip2][ix2, jy2, kz2]
                    else:
                        if not kept_patches[ip]:
                            raise ValueError("Patch not read!!!")

                        dx_l = cellsizes[ip]
                        dxx=(xij-(patchrx[ip]+(ix-0.5)*dx_l))/dx_l
                        dyy=(yij-(patchry[ip]+(jy-0.5)*dx_l))/dx_l
                        dzz=(zij-(patchrz[ip]+(kz-0.5)*dx_l))/dx_l

                        if ip==0 and (ix<=0 or jy<=0 or kz<=0 or ix>=nmax-1 or jy>=nmax-1 or kz>=nmax-1):
                            proj[i,j] += field[ip][(ix  )%nmax,(jy  )%nmax,(kz  )%nmax] *(1-dxx)*(1-dyy)*(1-dzz) \
                                    + field[ip][(ix  )%nmax,(jy  )%nmax,(kz+1)%nmax] *(1-dxx)*(1-dyy)*  dzz   \
                                    + field[ip][(ix  )%nmax,(jy+1)%nmax,(kz  )%nmax] *(1-dxx)*  dyy  *(1-dzz) \
                                    + field[ip][(ix  )%nmax,(jy+1)%nmax,(kz+1)%nmax] *(1-dxx)*  dyy  *  dzz   \
                                    + field[ip][(ix+1)%nmax,(jy  )%nmax,(kz  )%nmax] *  dxx  *(1-dyy)*(1-dzz) \
                                    + field[ip][(ix+1)%nmax,(jy  )%nmax,(kz+1)%nmax] *  dxx  *(1-dyy)*  dzz   \
                                    + field[ip][(ix+1)%nmax,(jy+1)%nmax,(kz  )%nmax] *  dxx  *  dyy  *(1-dzz) \
                                    + field[ip][(ix+1)%nmax,(jy+1)%nmax,(kz+1)%nmax] *  dxx  *  dyy  *  dzz  
                            proj_weight[i,j] += wfield[ip][(ix  )%nmax,(jy  )%nmax,(kz  )%nmax] *(1-dxx)*(1-dyy)*(1-dzz) \
                                              + wfield[ip][(ix  )%nmax,(jy  )%nmax,(kz+1)%nmax] *(1-dxx)*(1-dyy)*  dzz   \
                                              + wfield[ip][(ix  )%nmax,(jy+1)%nmax,(kz  )%nmax] *(1-dxx)*  dyy  *(1-dzz) \
                                              + wfield[ip][(ix  )%nmax,(jy+1)%nmax,(kz+1)%nmax] *(1-dxx)*  dyy  *  dzz   \
                                              + wfield[ip][(ix+1)%nmax,(jy  )%nmax,(kz  )%nmax] *  dxx  *(1-dyy)*(1-dzz) \
                                              + wfield[ip][(ix+1)%nmax,(jy  )%nmax,(kz+1)%nmax] *  dxx  *(1-dyy)*  dzz   \
                                              + wfield[ip][(ix+1)%nmax,(jy+1)%nmax,(kz  )%nmax] *  dxx  *  dyy  *(1-dzz) \
                                              + wfield[ip][(ix+1)%nmax,(jy+1)%nmax,(kz+1)%nmax] *  dxx  *  dyy  *  dzz  
                        else:
                            proj[i,j] += field[ip][ix  , jy  , kz  ] *(1-dxx)*(1-dyy)*(1-dzz) \
                                    + field[ip][ix  , jy  , kz+1] *(1-dxx)*(1-dyy)*  dzz   \
                                    + field[ip][ix  , jy+1, kz  ] *(1-dxx)*  dyy  *(1-dzz) \
                                    + field[ip][ix  , jy+1, kz+1] *(1-dxx)*  dyy  *  dzz   \
                                    + field[ip][ix+1, jy  , kz  ] *  dxx  *(1-dyy)*(1-dzz) \
                                    + field[ip][ix+1, jy  , kz+1] *  dxx  *(1-dyy)*  dzz   \
                                    + field[ip][ix+1, jy+1, kz  ] *  dxx  *  dyy  *(1-dzz) \
                                    + field[ip][ix+1, jy+1, kz+1] *  dxx  *  dyy  *  dzz  
                            proj_weight[i,j] += wfield[ip][ix  , jy  , kz  ] *(1-dxx)*(1-dyy)*(1-dzz) \
                                              + wfield[ip][ix  , jy  , kz+1] *(1-dxx)*(1-dyy)*  dzz   \
                                              + wfield[ip][ix  , jy+1, kz  ] *(1-dxx)*  dyy  *(1-dzz) \
                                              + wfield[ip][ix  , jy+1, kz+1] *(1-dxx)*  dyy  *  dzz   \
                                              + wfield[ip][ix+1, jy  , kz  ] *  dxx  *(1-dyy)*(1-dzz) \
                                              + wfield[ip][ix+1, jy  , kz+1] *  dxx  *(1-dyy)*  dzz   \
                                              + wfield[ip][ix+1, jy+1, kz  ] *  dxx  *  dyy  *(1-dzz) \
                                              + wfield[ip][ix+1, jy+1, kz+1] *  dxx  *  dyy  *  dzz  

        #proj_weight[proj_weight==0] = 1
        proj /= proj_weight
        
        # reorient the projection array: east is the first index increasing, north is the second index increasing
        proj = np.transpose(np.flipud(proj))

        return proj

    proj = parallelize(xc, yc, zc, binsr, binsphi, north_vector, east_vector, size, nmax, levels, kept_patches, 
                                            List([f if ki else np.zeros((2,2,2), dtype=field[0].dtype, order='F') for f,ki in zip(field, kept_patches)]),
                                            List([f if ki else np.zeros((2,2,2), dtype=field[0].dtype, order='F') for f,ki in zip(weight_field, kept_patches)]),
                                            normal_vector, res_normal)


    # mirror proj 
    #proj = np.flip(proj)

    return_vars = [proj]


    if return_grid:
        return_vars.append(binsphi)
        return_vars.append(binsr)

    return tuple(return_vars)
