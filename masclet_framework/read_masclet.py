#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

read_masclet module
Provides the necessary functions for reading MASCLET files and loading them in
memory

v0.1.0, 30/06/2019
David Vallés, 2019

"""

### GENERAL PURPOSE AND SPECIFIC LIBRARIES

#numpy
import numpy as np
#scipy
from scipy.io import FortranFile

#masclet_framework
from masclet_framework import parameters


def filename(it, tipo, digits=5):
    '''
    Reads the iteration number (it) and type ('g' for grids, 'b' for gas y 'd' 
    for dm) and generates a name. The optional parameter digits selects 
    how many digits the it number is written with.
    '''
    nombres = {'g' : "grids", 'b' : 'clus', 'd' : 'cldm'}
    try:
        if np.floor(np.log10(it)) < digits:
            return nombres[tipo] + str(it).zfill(digits)
        else:
            raise ValueError("Digits should be greater to handle that it number")
    except KeyError:
        print('Insert a correct type: g, b o d ')        


def read_grids(it,path='',digits=5,readgeneral = True, readpatchnum = True, readdmpartnum = True, readpatchcellextension = True,
             readpatchcellposition = True, readpatchposition = True, readpatchparent = True, nparray = True):
    """
    reads grids files, containing the information needed for building the AMR 
    structure
    
    PARAMETERS:
        it: iteration number
        path: path of the folder containing the grids file. Don't assign it if 
        it is in the same folder as the script
        digits: no. of digits of iteration number
        optional parameters: exclude part of the information (avoid loading it)
        nparray: reads everything as numpy arrays (default: True)
    
    RETURNS:
        -only if readgeneral set to True
        IRR: iteration number
        T: time
        NL: num of refinement levels
        MAP: mass of DM particles
        ZETA: redshift
        
        -only if readpatchnum set to True
        NPATCH: num de patches en cada nivel, empezando en l=0
        
        -only if readdmpartnum set to True
        NPART: num de dm particles en cada nivel, empezando en l=0
        
        -only if readpatchcellextension set to True
        PATCHNX: x-extension of each patch (in level l cells) (and Y and Z)
        
        -only if readpatchcellposition set to True
        PATCHX: x-position of each patch (left-bottom-front corner; in level 
        l-1 cells) (and Y and Z)
        
        -only if readpatchposition set to True
        PATCHRX: physical position of the center of each patch first ¡l-1! cell 
        (and Y and Z)
        
        -only if readpatchparent set to True
        PARE: which (l-1)-cell is left-bottom-front corner of each patch in
    """
    
    grids = open(path+filename(it,'g',digits),'r')
    
    #first, we load some general parameters
    IRR,T,NL,MAP,_ = tuple(float(i) for i in grids.readline().split())
    IRR = int(IRR)
    assert(it == IRR)
    NL = int(NL)
    ZETA = float(grids.readline().split()[0])
    #l=0
    IR, NDXYZ, _ = tuple(float(i) for i in grids.readline().split())
    IR = int(IR)
    NDXYZ = int(NDXYZ)

    #vectors where the data will be stored
    NPATCH = [0] #number of patches per level, starting with l=0
    NPART = [NDXYZ] #number of dm particles per level, starting with l=0
    PATCHNX = [0]
    PATCHNY = [0]
    PATCHNZ = [0]
    PATCHX = [0]
    PATCHY = [0]
    PATCHZ = [0]
    PATCHRX = [0]
    PATCHRY = [0]
    PATCHRZ = [0]
    PARE = [0]

    for IR in range(1,NL+1):
        level, npatchtemp, nparttemp, _ = tuple(int(i) for i in grids.readline().split())
        NPATCH.append(npatchtemp)
        NPART.append(nparttemp)

        #ignoring a blank line
        grids.readline()

        #loading all values
        for i in range(sum(NPATCH[0:IR])+1,sum(NPATCH[0:IR+1])+1):
            nx,ny,nz = tuple(int(i) for i in grids.readline().split())
            x,y,z = tuple(int(i) for i in grids.readline().split())
            rx,ry,rz = tuple(float(i) for i in grids.readline().split())
            pare = int(grids.readline())
            PATCHNX.append(nx)
            PATCHNY.append(ny)
            PATCHNZ.append(nz)
            PATCHX.append(x)
            PATCHY.append(y)
            PATCHZ.append(z)
            PATCHRX.append(rx)
            PATCHRY.append(ry)
            PATCHRZ.append(rz)
            PARE.append(pare)
    
    #converts everything into numpy arrays if nparray set to True
    if nparray:
        NPATCH = np.array(NPATCH)
        NPART = np.array(NPART)
        PATCHNX = np.array(PATCHNX)
        PATCHNY = np.array(PATCHNY)
        PATCHNZ = np.array(PATCHNZ)
        PATCHX = np.array(PATCHX)
        PATCHY = np.array(PATCHY)
        PATCHZ = np.array(PATCHZ)
        PATCHRX = np.array(PATCHRX)
        PATCHRY = np.array(PATCHRY)
        PATCHRZ = np.array(PATCHRZ)
        PARE = np.array(PARE)
    
    grids.close()
    
    returnvariables = []
    
    if readgeneral:
        returnvariables.extend([IRR, T, NL, MAP, ZETA])
    if readpatchnum:
        returnvariables.append(NPATCH)
    if readdmpartnum:
        returnvariables.append(NPART)
    if readpatchcellextension:
        returnvariables.extend([PATCHNX, PATCHNY, PATCHNZ])
    if readpatchcellposition:
        returnvariables.extend([PATCHX, PATCHY, PATCHZ])
    if readpatchposition:
        returnvariables.extend([PATCHRX, PATCHRY, PATCHRZ])
    if readpatchparent: 
        returnvariables.append(PARE)
    
    return tuple(returnvariables)


def read_clus(it,path='',digits=5,maxRefinedLevel=1000,outputDelta=True, outputV = True, outputPres = True, outputPot = True, outputOpot = False, outputTemp = True, outputMetalicity = True, outputCr0amr = True, outputSolapst = True, verbose = False, fullverbose = False):
    """
    Reads gas (clus) file
    
    PARAMETERS:
        it: iteration number
        path: path of the folder containing the grids file. Don't assign it if it is in the same folder as the script
        digits: no. of digits of iteration number
        maxRefinedLevel: maximum level of refinement that is wanted to be read (if not assigned, it will read all of them)
        optional parameters: choose which part of the information will be loaded
        
    
    RETURNS:
        - variables set to True at l=0, as a NMAX x NMAY x NMAZ np array
        - refined variables set to True up to level maxRefinedLevel (or maximum level, NLEVELS), as a np 1D array 
        containing a numpy array per patch
    """
    
    NMAX, NMAY, NMAZ, NLEVELS = parameters.read_parameters(loadNMA = True, loadNPALEV = False, loadNLEVELS = True, loadNAMR = False, loadSIZE=False)
    NPATCH,PATCHNX,PATCHNY,PATCHNZ = read_grids(it, path=path, readgeneral=False, readpatchnum=True, readdmpartnum = False, readpatchcellextension = True,
             readpatchcellposition = False, readpatchposition = False, readpatchparent = False)
    f = FortranFile(path+filename(it,'b',digits),'r')
    # endian not specified (small by default)
    it_clus, time, z = tuple(f.read_reals(dtype='i4, f4, f4')[0]) 

    #'F' in reshape means Fortran-order (first index changing fastest)
    if fullverbose:
        print('l=0 delta')
    delta = np.reshape(f.read_reals(dtype='f4'),(NMAX,NMAY,NMAZ),'F')
    if fullverbose:
        print('l=0 vx')
    vx = np.reshape(f.read_reals(dtype='f4'),(NMAX,NMAY,NMAZ),'F')
    if fullverbose:
        print('l=0 vy')
    vy = np.reshape(f.read_reals(dtype='f4'),(NMAX,NMAY,NMAZ),'F')
    if fullverbose:
        print('l=0 vz')
    vz = np.reshape(f.read_reals(dtype='f4'),(NMAX,NMAY,NMAZ),'F')
    if fullverbose:
        print('l=0 pres')
    pres = np.reshape(f.read_reals(dtype='f4'),(NMAX,NMAY,NMAZ),'F')
    if fullverbose:
        print('l=0 pot')
    pot = np.reshape(f.read_reals(dtype='f4'),(NMAX,NMAY,NMAZ),'F')
    if fullverbose:
        print('l=0 opot')
    opot = np.reshape(f.read_reals(dtype='f4'),(NMAX,NMAY,NMAZ),'F')
    if fullverbose:
        print('l=0 temp')
    temp = np.reshape(f.read_reals(dtype='f4'),(NMAX,NMAY,NMAZ),'F')
    if fullverbose:
        print('l=0 metalicity')
    metalicity = np.reshape(f.read_reals(dtype='f4'),(NMAX,NMAY,NMAZ),'F')
    if fullverbose:
        print('l=0 cr0amr')
    cr0amr = np.reshape(f.read_ints(),(NMAX,NMAY,NMAZ),'F') # 1 si no refinado, 0 si refinado

    #faster computation if we keep appending to a python list than if we append to a numpy array
    delta_refined = [0]
    vx_refined = [0]
    vy_refined = [0]
    vz_refined = [0]
    pres_refined = [0]
    pot_refined = [0]
    opot_refined = [0]
    temp_refined = [0]
    metalicity_refined = [0]
    cr0amr_refined = [0]
    solapst_refined = [0]

    # 11 variables por patch 
    #TO DO: check which variables we want to output (it's only ~ 1000*11 checks, not that bad)
    for l in range(1,min(NLEVELS+1,maxRefinedLevel+1)):
        if verbose:
            print('Reading level {}.'.format(l))
            print('{} patches.'.format(NPATCH[l]))
        for ipatch in range(NPATCH[0:l].sum()+1,NPATCH[0:l+1].sum()+1):
            if fullverbose:
                print('Reading patch {}'.format(ipatch))
            delta_refined.append(np.reshape(f.read_reals(dtype='f4'),(PATCHNX[ipatch],PATCHNY[ipatch],PATCHNZ[ipatch]),'F'))
            vx_refined.append(np.reshape(f.read_reals(dtype='f4'),(PATCHNX[ipatch],PATCHNY[ipatch],PATCHNZ[ipatch]),'F'))
            vy_refined.append(np.reshape(f.read_reals(dtype='f4'),(PATCHNX[ipatch],PATCHNY[ipatch],PATCHNZ[ipatch]),'F'))
            vz_refined.append(np.reshape(f.read_reals(dtype='f4'),(PATCHNX[ipatch],PATCHNY[ipatch],PATCHNZ[ipatch]),'F'))
            pres_refined.append(np.reshape(f.read_reals(dtype='f4'),(PATCHNX[ipatch],PATCHNY[ipatch],PATCHNZ[ipatch]),'F'))
            pot_refined.append(np.reshape(f.read_reals(dtype='f4'),(PATCHNX[ipatch],PATCHNY[ipatch],PATCHNZ[ipatch]),'F'))
            opot_refined.append(np.reshape(f.read_reals(dtype='f4'),(PATCHNX[ipatch],PATCHNY[ipatch],PATCHNZ[ipatch]),'F'))
            temp_refined.append(np.reshape(f.read_reals(dtype='f4'),(PATCHNX[ipatch],PATCHNY[ipatch],PATCHNZ[ipatch]),'F'))
            metalicity_refined.append(np.reshape(f.read_reals(dtype='f4'),(PATCHNX[ipatch],PATCHNY[ipatch],PATCHNZ[ipatch]),'F'))
            cr0amr_refined.append(np.reshape(f.read_ints(dtype='i4'),(PATCHNX[ipatch],PATCHNY[ipatch],PATCHNZ[ipatch]),'F')) #1 si no refinado, 0 si refinado
            solapst_refined.append(np.reshape(f.read_ints(dtype='i4'),(PATCHNX[ipatch],PATCHNY[ipatch],PATCHNZ[ipatch]),'F')) #1 si no solapado, 0 si solapado

    f.close()
    
    returnvariables = []
    if outputDelta:
        returnvariables.append(delta)
    if outputV:
        returnvariables.extend([vx, vy, vz])
    if outputPres:
        returnvariables.append(pres)
    if outputPot:
        returnvariables.append(pot)
    if outputOpot:
        returnvariables.append(opot)
    if outputTemp:
        returnvariables.append(temp)
    if outputMetalicity:
        returnvariables.append(metalicity)
    if outputCr0amr:
        returnvariables.append(cr0amr)
    if maxRefinedLevel >= 1:
        if outputDelta:
            returnvariables.append(delta_refined)
        if outputV:
            returnvariables.extend([vx_refined, vy_refined, vz_refined])
        if outputPres:
            returnvariables.append(pres_refined)
        if outputPot:
            returnvariables.append(pot_refined)
        if outputOpot:
            returnvariables.append(opot_refined)
        if outputTemp:
            returnvariables.append(temp_refined)
        if outputMetalicity:
            returnvariables.append(metalicity_refined)
        if outputCr0amr:
            returnvariables.append(cr0amr_refined)
        if outputSolapst:
            returnvariables.append(solapst_refined)

    return tuple(returnvariables)