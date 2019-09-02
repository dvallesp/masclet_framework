"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

graphics module
Contains functions that can directly compute projections along one of the axis,
or generate colormap plots. Makes use of matplotlib.

Created by David Vallés
"""

#  Last update on 2/9/19 18:01

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

# numpy
import numpy as np
# matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors


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
    else:
        if are_xy_given:
            assert (x.shape[0] == z.shape[0] and y.shape[0] == z.shape[1])
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
        pcm = ax.pcolor(xmesh, ymesh, np.transpose(z), norm=colors.LogNorm(vmin=cbarmin, vmax=cbarmax), cmap=cmap)
    else:
        pcm = ax.pcolor(xmesh, ymesh, np.transpose(z), norm=colors.Normalize(vmin=cbarmin, vmax=cbarmax), cmap=cmap)

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
