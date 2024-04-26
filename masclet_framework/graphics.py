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
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from masclet_framework.read_masclet import read_grids
from masclet_framework.tools import create_vector_levels, which_patches_inside_box


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
    



def plot_AMR_grid(it, path='', L = 40., nmax = 128, 
                  figsize = (5,4), dpi = 300, ax = None,
                  box = None, proyection = 'xy', cm = 'inferno',
                  linewidth_max = 2, min_level = 0, max_level = 9,
                  plot_3d = False, plot_halma = False, halma_catalogue = None,
                  factor_R = 10.0, contained = False):
    """
    Plots the AMR grid at iteration it for the whole box
    or a slice of the box
    Args:
        it: iteration number
        path: path to grid data
        L: box size
        nmax: number of cells in coarse grid
        figsize: size of the figure
        dpi: dots per inch
        ax: axis object to plot the grid, if None, a new figure is created
        box: subregion of the computational domain to be plotted
             If None, the whole domain is plotted. 
        proyection: proyection plane. Defaults to 'xy'
        cm: colormap to be used. Defaults to 'inferno'
        linewidth_max: linewidth of the patches at the maximum level
        min_level: minimum level to be plotted
        max_level: maximum level to be plotted
        plot_3d: if True, the plot will be in 3d. Defaults to False
        plot_halma: if True, pyHALMA haloes will be plotted. Defaults to False
        halma_catalogue: pyHALMA catalogue object (with 'arrays' output format). 
                          Defaults to None
        factor_R: factor to multiply the halma radii. Defaults to 10.0
        contained: if True, only patches contained in the box are plotted. Defaults to False

    Returns:
        axis object with the plot

    Author: Óscar Monllor
    """
    
    # Read grid data
    grids = read_grids(it, path, path)
    npatch = grids[5]
    patchnx = grids[7]
    patchny = grids[8]
    patchnz = grids[9]
    patchrx = grids[13]
    patchry = grids[14]
    patchrz = grids[15]
    plevel = create_vector_levels(npatch)

    # Find patches inside box
    if box is None:
        patches_inside = np.arange(0, npatch.sum() + 1, 1)
    else:
        if contained:
            patches_inside = which_patches_inside_box(box, patchnx, patchny, patchnz, 
                                                    patchrx, patchry, patchrz, npatch, L, nmax,
                                                    contained)
        else:
            patches_inside = which_patches_inside_box(box, patchnx, patchny, patchnz, 
                                                    patchrx, patchry, patchrz, npatch, L, nmax)
    
    # Plot patches
    if box is None:
        x0 = -L/2
        x1 =  L/2
        y0 = -L/2
        y1 =  L/2
        z0 = -L/2
        z1 =  L/2

    else:
        x0 = box[0]
        x1 = box[1]
        y0 = box[2]
        y1 = box[3]
        z0 = box[4]
        z1 = box[5]

    if plot_3d:
    
        new_fig = False

        # PLOT 3D
        if ax is None:
            fig = plt.figure(figsize=figsize, dpi=dpi)
            ax = fig.add_subplot(111, projection='3d')
            ax.set_xlabel('X (cMpc)')
            ax.set_ylabel('Y (cMpc)')
            ax.set_zlabel('Z (cMpc)')
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_zlim(z0, z1)
            new_fig = True


        # Plotting patches
        dx = L/nmax
        ncolors =  1 + max_level - min_level

        #Sample ncolors from the colormap
        cmap = plt.get_cmap('inferno')
        colors = cmap(np.linspace(0.2, 0.8, ncolors))
        for ipatch in patches_inside:
            level = plevel[ipatch]
            if level < min_level or level > max_level:
                continue

            #Adjust linewidth and color according to max_level and min_level
            color_level = level - min_level
            linewidth = linewidth_max / (level - min_level + 1)**2

            # patch resolution
            dx_patch = dx / 2**level
            # extension
            nx = patchnx[ipatch]
            ny = patchny[ipatch]
            nz = patchnz[ipatch]
            # low edge of the patch
            x_low = patchrx[ipatch] - dx_patch
            y_low = patchry[ipatch] - dx_patch
            z_low = patchrz[ipatch] - dx_patch
            # high edge of the patch
            x_high = patchrx[ipatch] + (nx - 1) * dx_patch
            y_high = patchry[ipatch] + (ny - 1) * dx_patch
            z_high = patchrz[ipatch] + (nz - 1) * dx_patch

            p1 = [x_low, y_low, z_low]
            p2 = [x_high, y_low, z_low]
            p3 = [x_high, y_high, z_low]
            p4 = [x_low, y_high, z_low]
            p5 = [x_low, y_low, z_high]
            p6 = [x_high, y_low, z_high]
            p7 = [x_high, y_high, z_high]
            p8 = [x_low, y_high, z_high]

            verts = [[p1, p2, p3, p4], 
                    [p5, p6, p7, p8], 
                    [p1, p2, p6, p5], 
                    [p4, p3, p7, p8], 
                    [p1, p4, p8, p5], 
                    [p2, p3, p7, p6]]
            
            ax.add_collection3d(Poly3DCollection(verts, linewidths=linewidth, edgecolors=colors[color_level], alpha = 0.))

        if plot_halma:
            xHal = halma_catalogue['xpeak']*1e-3
            yHal = halma_catalogue['ypeak']*1e-3
            zHal = halma_catalogue['zpeak']*1e-3
            RaHal = 10*halma_catalogue['R']*1e-3

            # Plot 3D spheres
            for i in range(len(xHal)):
                u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
                x = RaHal[i]*np.cos(u)*np.sin(v) + xHal[i]
                y = RaHal[i]*np.sin(u)*np.sin(v) + yHal[i]
                z = RaHal[i]*np.cos(v) + zHal[i]
                ax.plot_surface(x, y, z, color='r', alpha=1)

    else:

        # PLOT 2D
        new_fig = False

        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
            new_fig = True

        assert proyection in ['xy', 'xz', 'yz'], 'proyection should be xy, xz or yz'

        if proyection == 'xy':
            ax.set_xlabel('X (cMpc)')
            ax.set_ylabel('Y (cMpc)')
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
        elif proyection == 'xz':
            ax.set_xlabel('X (cMpc)')
            ax.set_ylabel('Z (cMpc)')
            ax.set_xlim(x0, x1)
            ax.set_ylim(z0, z1)
        elif proyection == 'yz':
            ax.set_xlabel('Y (cMpc)')
            ax.set_ylabel('Z (cMpc)')
            ax.set_xlim(y0, y1)
            ax.set_ylim(z0, z1)

        # Plotting patches
        dx = L/nmax
        ncolors =  1 + max_level - min_level

        #Sample ncolors from the colormap
        cmap = plt.get_cmap(cm)
        colors = cmap(np.linspace(0.2, 0.8, ncolors))
        for ipatch in patches_inside:
            level = plevel[ipatch]
            if level < min_level or level > max_level:
                continue

            #Adjust linewidth and color according to max_level and min_level
            color_level = level - min_level
            linewidth = linewidth_max / (level - min_level + 1)**2

            # patch resolution
            dx_patch = dx / 2**level
            # extension
            nx = patchnx[ipatch]
            ny = patchny[ipatch]
            nz = patchnz[ipatch]
            # low edge of the patch
            x_low = patchrx[ipatch] - dx_patch
            y_low = patchry[ipatch] - dx_patch
            z_low = patchrz[ipatch] - dx_patch
            # high edge of the patch
            x_high = patchrx[ipatch] + (nx - 1) * dx_patch
            y_high = patchry[ipatch] + (ny - 1) * dx_patch
            z_high = patchrz[ipatch] + (nz - 1) * dx_patch

            if proyection == 'xy':
                ax.plot([x_low, x_high], [y_low, y_low], linewidth=linewidth, color=colors[color_level])
                ax.plot([x_low, x_high], [y_high, y_high], linewidth=linewidth, color=colors[color_level])
                ax.plot([x_low, x_low], [y_low, y_high], linewidth=linewidth, color=colors[color_level])
                ax.plot([x_high, x_high], [y_low, y_high], linewidth=linewidth, color=colors[color_level])

            elif proyection == 'xz':
                ax.plot([x_low, x_high], [z_low, z_low], linewidth=linewidth, color=colors[color_level])
                ax.plot([x_low, x_high], [z_high, z_high], linewidth=linewidth, color=colors[color_level])
                ax.plot([x_low, x_low], [z_low, z_high], linewidth=linewidth, color=colors[color_level])
                ax.plot([x_high, x_high], [z_low, z_high], linewidth=linewidth, color=colors[color_level])

            elif proyection == 'yz':
                ax.plot([y_low, y_high], [z_low, z_low], linewidth=linewidth, color=colors[color_level])
                ax.plot([y_low, y_high], [z_high, z_high], linewidth=linewidth, color=colors[color_level])
                ax.plot([y_low, y_low], [z_low, z_high], linewidth=linewidth, color=colors[color_level])
                ax.plot([y_high, y_high], [z_low, z_high], linewidth=linewidth, color=colors[color_level])
        
        if plot_halma:
            xHal = halma_catalogue['xpeak']*1e-3
            yHal = halma_catalogue['ypeak']*1e-3
            zHal = halma_catalogue['zpeak']*1e-3
            RaHal = factor_R*halma_catalogue['R']*1e-3

            # Plot 2D circles
            for i in range(len(xHal)):
                circle = plt.Circle((xHal[i], yHal[i]), RaHal[i], color='r', fill=False)
                ax.add_artist(circle)

    if new_fig:
        return fig, ax
    
    else:
        return ax
