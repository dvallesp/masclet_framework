#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

graphics module
Contains functions that can directly compute projections along one of the axis,
or generate colormap plots. Makes use of matplotlib.

v0.1.0, 30/06/2019
David Vallés, 2019
"""

#numpy
import numpy as np
#matplotlib 
import matplotlib.pyplot as plt
import matplotlib.colors as colors


def colormap2d(Z,x=np.array([]),y=np.array([]),xlabel='',ylabel='',cbarlabel='', 
               title='', cbartitle='', areXYgiven = True, areMeshgrid = False,
               cbarmin=None,cbarmax=None, xmin=None, xmax=None, ymin=None, 
               ymax = None, cmap='PuBu_r', axisfont=18, ticksfont=18,
               titlefont=18, logz = False, manipulateTicks = False, nticks=10):
    '''
    Plots Z against xy plane in colorscale (l=0 only [for now])
    
    PARAMETER CONFIGURATIONS:
    xy are corner positions
    If no x, y are provided and areXYgiven is set to False, integer values are 
    assigned
    If areMeshgrid True, x and y are assumed to be meshgrids. If False, they 
    are assumed to be vectors.
    cbarmin, cbarmax: set if desired. if not, it will be set to max and min of 
    Z
    idem with xmin, xmax, ymin, ymax
    cmap: colormap (see Matplotlib documentation)
    axisfonts, ticksfont, titlefont (in px)
    title, xlabel, ylabel, cbartitle, cbarlabel...
    nticks: number of ticks in the colorbar scale; only works if 
    manipulateTicks set to True
    
    RETURNS:
    axes object for the plot
    '''
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    if not areMeshgrid:
        if areXYgiven:
            assert(x.shape[0]==Z.shape[0] and y.shape[0]==Z.shape[1])
            X,Y = np.meshgrid(x, y)
        else:
            Zsize = Z.shape
            X,Y = np.meshgrid(np.array(range(Zsize[0]+1)), np.array(range(Zsize[1]+1)))
    if cbarmin == None:
        cbarmin = Z.min()
    if cbarmax == None:
        cbarmax = Z.max()
    if xmin == None:
        xmin = X.min()
    if xmax == None:
        xmax = X.max()
    if ymin == None:
        ymin = Y.min()
    if ymax == None:
        ymax = Y.max()
    
    if logz:
        pcm = ax.pcolor(X, Y, np.transpose(Z), norm=colors.LogNorm(vmin=cbarmin, vmax=cbarmax), cmap=cmap)
    else:
        pcm = ax.pcolor(X, Y, np.transpose(Z), norm=colors.Normalize(vmin=cbarmin, vmax=cbarmax),cmap=cmap)
    
    plt.xticks(fontsize=ticksfont)
    plt.yticks(fontsize=ticksfont)
    plt.xlabel(xlabel, fontsize=axisfont)
    plt.ylabel(ylabel, fontsize=axisfont)
    plt.title(title, fontsize=titlefont)

    cbar = fig.colorbar(pcm, ax=ax)
    cbar.ax.tick_params(labelsize=ticksfont)
    cbar.set_label(cbarlabel, fontsize=ticksfont)
    cbar.ax.set_title(cbartitle, fontsize=axisfont)
    
    if manipulateTicks: #not working properly right now! we have to format the number of significant figures
        cbar.set_ticks(np.linspace(cbarmin,cbarmax,nticks))
        cbar.set_ticklabels(np.linspace(cbarmin,cbarmax,nticks))
        cbar.update_ticks()
    
    return ax


def compute_projection(matrix, axis=2, slicepositions = [0,0]):
    '''
    Computes the projection of 3d matrix along the specified axis (default: 3rd
    axis), between the cell positions specified in slicepositions (if it is 
    [0,0], the whole box is projected), both included
    '''
    slicemin, slicemax = tuple(slicepositions)
    if slicemin == slicemax:
        return matrix.mean(axis)
    elif slicemin < slicemax:
        if axis == 0:
            matrix = matrix[slicemin:slicemax+1,:,:]
        elif axis == 1:
            matrix = matrix[:,slicemin:slicemax+1,:]
        elif axis == 2:
            matrix = matrix[:,:,slicemin:slicemax+1]
        else:
            raise ValueError('axis should be 0, 1 or 2')
        return matrix.mean(axis)
    else:
        raise ValueError('slicemin should be smaller than slicemax')


