"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

read_masclet module
Provides the necessary functions for reading MASCLET files and loading them in
memory

Created by David Vallés
"""

#  Last update on 17/9/20 14:22

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

import os,sys
# numpy
import numpy as np
# scipy (will be removed)
from scipy.io import FortranFile
# cython_fortran_file
from cython_fortran_file import FortranFile as FF
import struct

# tqdm
import importlib.util
if importlib.util.find_spec('tqdm') is None:
    def tqdm(x, desc=None): return x
else:
    from tqdm import tqdm

# masclet_framework
from masclet_framework import parameters
from masclet_framework import tools


# FUNCTIONS DEFINED IN THIS MODULE

def filename(it, filetype, digits=5):
    """
    Generates filenames for MASCLET output files

    Args:
        it: iteration number (int)
        filetype: 'g' for grids file; 'b' for gas file (baryonic); 'd' for dark matter (dm) (str)
        digits: number of digits the filename is written with (int)

    Returns: filename (str)

    """
    names = {'g': "grids", 'b': 'clus', 'd': 'cldm', 's': 'clst', 'v': 'velocity', 'm': 'MachNum_', 'f': 'filtlen_'}
    try:
        if it == 0:
            return names[filetype] + str(it).zfill(digits)
        elif np.floor(np.log10(it)) < digits:
            return names[filetype] + str(it).zfill(digits)
        else:
            raise ValueError("Digits should be greater to handle that iteration number")
    except KeyError:
        print('Insert a correct type: g, b, d, s, v, f or m')


def read_grids(it, path='', parameters_path='', digits=5, read_general=True, read_patchnum=True, read_dmpartnum=True,
               read_patchcellextension=True, read_patchcellposition=True, read_patchposition=True,
               read_patchparent=True, nparray=True):
    """
    reads grids files, containing the information needed for building the AMR structure

    Args:
        it: iteration number (int)
        path: path of the grids file in the system (str)
        parameters_path: path of the json parameters file of the simulation (str)
        digits: number of digits the filename is written with (int)
        read_general: whether irr, T, NL, MAP and ZETA are returned (bool)
        read_patchnum: whether NPATCH is returned (bool)
        read_dmpartnum: whether NPART is returned (bool)
        read_patchcellextension: whether PATCHNX, PATCHNY, PATCHNZ are returned (bool)
        read_patchcellposition: whether PATCHX, PATCHY, PATCHZ are returned (bool)
        read_patchposition: whether PATCHRX, PATCHRY, PATCHRZ are returned (bool)
        read_patchparent: whether PARENT is returned (bool)
        nparray: if True (default), all variables are returned as numpy arrays (bool)

    Returns: (in order)

        -only if readgeneral set to True
        irr: iteration number
        t: time
        nl: num of refinement levels
        mass_dmpart: mass of DM particles
        zeta: redshift

        -only if readpatchnum set to True
        npatch: number of patches in each level, starting in l=0

        -only if readdmpartnum set to True
        npart: number of dm particles in each leve, starting in l=0

        -only if readpatchcellextension set to True
        patchnx (...): x-extension of each patch (in level l cells) (and Y and Z)

        -only if readpatchcellposition set to True
        patchx (...): x-position of each patch (left-bottom-front corner; in level
        l-1 cells) (and Y and Z)
        CAUTION!!! IN THE OUTPUT, FIRST CELL IS 1. HERE, WE SET IT TO BE 0. THUS, PATCHNX's READ HERE WILL BE LOWER IN
        A UNIT FROM THE ONE WRITTEN IN THE FILE.

        -only if readpatchposition set to True
        patchrx (...): physical position of the center of each patch first ¡l-1! cell
        (and Y and Z)

        -only if readpatchparent set to True
        pare: which (l-1)-cell is left-bottom-front corner of each patch in

    """
    nmax, nmay, nmaz, size = parameters.read_parameters(load_nma=True, load_npalev=False, load_nlevels=False,
                                                        load_namr=False, load_size=True, path=parameters_path)
    rx = - size / 2 + size / nmax

    grids = open(os.path.join(path, filename(it, 'g', digits)), 'r')

    # first, we load some general parameters
    irr, t, nl, mass_dmpart, _ = tuple(float(i) for i in grids.readline().split())
    irr = int(irr)
    # assert (it == irr)
    nl = int(nl)
    zeta = float(grids.readline().split()[0])
    # l=0
    _, ndxyz, _ = tuple(float(i) for i in grids.readline().split())[0:3]
    ndxyz = int(ndxyz)

    # vectors where the data will be stored
    npatch = [0]  # number of patches per level, starting with l=0
    npart = [ndxyz]  # number of dm particles per level, starting with l=0
    patchnx = [nmax]
    patchny = [nmay]
    patchnz = [nmaz]
    patchx = [0]
    patchy = [0]
    patchz = [0]
    patchrx = [rx]
    patchry = [rx]
    patchrz = [rx]
    pare = [0]

    for ir in range(1, nl + 1):
        level, npatchtemp, nparttemp = tuple(int(i) for i in grids.readline().split())[0:3]
        npatch.append(npatchtemp)
        npart.append(nparttemp)

        # ignoring a blank line
        grids.readline()

        # loading all values
        for i in range(sum(npatch[0:ir]) + 1, sum(npatch[0:ir + 1]) + 1):
            this_nx, this_ny, this_nz = tuple(int(i) for i in grids.readline().split())
            this_x, this_y, this_z = tuple(int(i) for i in grids.readline().split())
            this_rx, this_ry, this_rz = tuple(float(i) for i in grids.readline().split())
            this_pare = int(grids.readline())
            patchnx.append(this_nx)
            patchny.append(this_ny)
            patchnz.append(this_nz)
            patchx.append(this_x - 1)
            patchy.append(this_y - 1)
            patchz.append(this_z - 1)
            patchrx.append(this_rx)
            patchry.append(this_ry)
            patchrz.append(this_rz)
            pare.append(this_pare)

    # converts everything into numpy arrays if nparray set to True
    if nparray:
        npatch = np.array(npatch)
        npart = np.array(npart)
        patchnx = np.array(patchnx)
        patchny = np.array(patchny)
        patchnz = np.array(patchnz)
        patchx = np.array(patchx)
        patchy = np.array(patchy)
        patchz = np.array(patchz)
        patchrx = np.array(patchrx)
        patchry = np.array(patchry)
        patchrz = np.array(patchrz)
        pare = np.array(pare)

    grids.close()

    returnvariables = []

    if read_general:
        returnvariables.extend([irr, t, nl, mass_dmpart, zeta])
    if read_patchnum:
        returnvariables.append(npatch)
    if read_dmpartnum:
        returnvariables.append(npart)
    if read_patchcellextension:
        returnvariables.extend([patchnx, patchny, patchnz])
    if read_patchcellposition:
        returnvariables.extend([patchx, patchy, patchz])
    if read_patchposition:
        returnvariables.extend([patchrx, patchry, patchrz])
    if read_patchparent:
        returnvariables.append(pare)

    return tuple(returnvariables)


def read_clus(it, path='', parameters_path='', digits=5, max_refined_level=1000, output_delta=True, output_v=True,
              output_pres=True, output_pot=True, output_opot=False, output_temp=True, output_metalicity=True,
              output_cr0amr=True, output_solapst=True, is_mascletB=False, output_B=False, is_cooling=True,
              verbose=False, read_region=None):
    """
    Reads the gas (baryonic, clus) file

    Args:
        it: iteration number (int)
        path: path of the grids file in the system (str)
        parameters_path: path of the json parameters file of the simulation
        digits: number of digits the filename is written with (int)
        max_refined_level: maximum refinement level that wants to be read. Subsequent refinements will be skipped. (int)
        output_delta: whether delta (density contrast) is returned (bool)
        output_v: whether velocities (vx, vy, vz) are returned (bool)
        output_pres: whether pressure is returned (bool)
        output_pot: whether gravitational potential is returned (bool)
        output_opot: whether gravitational potential in the previous iteration is returned (bool)
        output_temp: whether temperature is returned (bool)
        output_metalicity: whether metalicity is returned (bool)
        output_cr0amr: whether "refined variable" (1 if not refined, 0 if refined) is returned (bool)
        output_solapst: whether "solapst variable" (1 if the cell is kept, 0 otherwise) is returned (bool)
        is_mascletB: whether the outputs correspond to masclet-B (contains magnetic fields) (bool)
        output_B: whether magnetic field is returned; only if is_mascletB = True (bool)
        is_cooling: whether there is cooling (an thus T and metalicity are written) or not (bool)
        verbose: whether a message is printed when each refinement level is started (bool)
        read_region: whether to select a subregion (see region specification below), or keep all the simulation data 
                     (None). If a region wants to be selected, there are the following possibilities:
                        - ("sphere", cx, cy, cz, R) for a sphere of radius R centered in (cx, cy, cz)
                        - ("box", x1, x2, y1, y2, z1, z2) for a box with corners (x1, y1, z1) and (x2, y2, z2)
                        - ("box_cw", xc, yc, zc, width) for a box centered in (xc, yc, zc) with width "width"

    Returns:
        Chosen quantities, as a list of arrays (one for each patch, starting with l=0 and subsequently);
        in the order specified by the order of the parameters in this definition.
        
        If read_region is not None, only the patches inside the region are read, and in the positions of the 
        arrays corresponding to the patches outside the region, a single scalar value of zero is written. Also
        in this case, after all the returned variables, a 1d array of booleans is returned, with the same length
        as the number of patches, indicating which patches are inside the region (True) and which are not (False).
    """
    #if not verbose:
    #    def tqdm(x): return x

    if output_B and (not is_mascletB):
        print('Error: cannot output magnetic field if the simulation has not.')
        print('Terminating')
        return
    
    if output_temp and (not is_cooling):
        print('Error: cannot output temperature if cooling was not allowed in the simulation.')
        print('Terminating')
        return
    
    if output_metalicity and (not is_cooling):
        print('Error: cannot output metalicity if cooling was not allowed in the simulation.')
        print('Terminating')
        return

    nmax, nmay, nmaz, nlevels, size = parameters.read_parameters(load_nma=True, load_npalev=False, load_nlevels=True,
                                                                 load_namr=False, load_size=True, path=parameters_path)
    npatch, patchnx, patchny, patchnz, \
            patchrx, patchry, patchrz = read_grids(it, path=path, parameters_path=parameters_path, read_general=False,
                                                   read_patchnum=True, read_dmpartnum=False,
                                                   read_patchcellextension=True, read_patchcellposition=False,
                                                   read_patchposition=True, read_patchparent=False)

    if read_region is None:
        keep_patches = np.ones(patchnx.size, dtype='bool')
    else:
        keep_patches = np.zeros(patchnx.size, dtype='bool')
        region_type = read_region[0]
        if region_type == 'sphere':
            cx, cy, cz, R = read_region[1:]
            which = tools.which_patches_inside_sphere(R, cx, cy, cz, patchnx, patchny, patchnz, patchrx, patchry, 
                                                      patchrz, npatch, size, nmax)
            keep_patches[which] = True
        elif region_type == 'box' or region_type == 'box_cw':
            if region_type == 'box':
                x1, x2, y1, y2, z1, z2 = read_region[1:]
            else:
                xc, yc, zc, width = read_region[1:]
                x1 = xc - width/2
                x2 = xc + width/2
                y1 = yc - width/2
                y2 = yc + width/2
                z1 = zc - width/2
                z2 = zc + width/2
            which = tools.which_patches_inside_box((x1, x2, y1, y2, z1, z2), patchnx, patchny, patchnz, patchrx, patchry,
                                                   patchrz,npatch,size,nmax)
            keep_patches[which] = True
        else:
            raise ValueError('Unknown region type. Please specify one of "sphere", "box" or "box_cw"')

    with FF(os.path.join(path, filename(it, 'b', digits))) as f:
        # read header
        it_clus = f.read_vector('i')[0]
        # assert(it == it_clus)
        f.seek(0)  # this is a little bit ugly but whatever
        time, z = tuple(f.read_vector('f')[1:3])

        # l=0
        if verbose:
            print('Reading base grid...')
        if output_delta:
            delta = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
        else:
            f.skip()

        if output_v:
            vx = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            vy = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            vz = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
        else:
            f.skip(3)

        if output_pres:
            pres = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
        else:
            f.skip()

        if output_pot:
            pot = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
        else:
            f.skip()

        if output_opot:
            opot = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
        else:
            f.skip()

        if is_cooling:
            if output_temp:
                temp = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            else:
                f.skip()

            if output_metalicity:
                metalicity = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            else:
                f.skip()

        if output_cr0amr:
            cr0amr = [np.reshape(f.read_vector('i'), (nmax, nmay, nmaz), 'F').astype('bool')]
        else:
            f.skip()

        if output_solapst:
            solapst = [1]

        if is_mascletB:
            if output_B:
                Bx = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
                By = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
                Bz = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            else:
                f.skip(3)


        # refinement levels
        for l in range(1, min(nlevels + 1, max_refined_level + 1)):
            if verbose:
                print('Reading level {}.'.format(l))
                print('{} patches.'.format(npatch[l]))
            for ipatch in tqdm(range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1), desc='Level {:}'.format(l)):
                #if verbose:
                #    print('Reading patch {}'.format(ipatch))
                if output_delta and keep_patches[ipatch]:
                    delta.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                            'F'))
                else:
                    f.skip()
                    if output_delta:
                        delta.append(0)

                if output_v and keep_patches[ipatch]:
                    vx.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    vy.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    vz.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                else:
                    f.skip(3)
                    if output_v:
                        vx.append(0)
                        vy.append(0)
                        vz.append(0)

                if output_pres and keep_patches[ipatch]:
                    pres.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                           'F'))
                else:
                    f.skip()
                    if output_pres:
                        pres.append(0)

                if output_pot and keep_patches[ipatch]:
                    pot.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                else:
                    f.skip()
                    if output_pot:
                        pot.append(0)

                if output_opot and keep_patches[ipatch]:
                    opot.append(
                        np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                else:
                    f.skip()
                    if output_opot:
                        opot.append(0)

                if is_cooling:
                    if output_temp and keep_patches[ipatch]:
                        temp.append(
                            np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    else:
                        f.skip()
                        if output_temp:
                            temp.append(0)

                    if output_metalicity and keep_patches[ipatch]:
                        metalicity.append(
                            np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    else:
                        f.skip()
                        if output_metalicity:
                            metalicity.append(0)

                if output_cr0amr and keep_patches[ipatch]:
                    cr0amr.append(np.reshape(f.read_vector('i'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                             'F').astype('bool'))
                else:
                    f.skip()
                    if output_cr0amr:
                        cr0amr.append(0)

                if output_solapst and keep_patches[ipatch]:
                    solapst.append(np.reshape(f.read_vector('i'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                              'F').astype('bool'))
                else:
                    f.skip()
                    if output_solapst:
                        solapst.append(0)

                if is_mascletB:
                    if output_B and keep_patches[ipatch]:
                        Bx.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                             'F'))
                        By.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                             'F'))
                        Bz.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                             'F'))
                    else:
                        f.skip(3)
                        if output_B:
                            Bx.append(0)
                            By.append(0)
                            Bz.append(0)

    returnvariables = []
    if output_delta:
        returnvariables.append(delta)
    if output_v:
        returnvariables.extend([vx, vy, vz])
    if output_pres:
        returnvariables.append(pres)
    if output_pot:
        returnvariables.append(pot)
    if output_opot:
        returnvariables.append(opot)
    if output_temp:
        returnvariables.append(temp)
    if output_metalicity:
        returnvariables.append(metalicity)
    if output_cr0amr:
        returnvariables.append(cr0amr)
    if output_solapst:
        returnvariables.append(solapst)
    if output_B:
        returnvariables.extend([Bx,By,Bz])

    if read_region is not None:
        returnvariables.append(keep_patches)

    return tuple(returnvariables)


def lowlevel_read_clus(it, path='', parameters_path='', digits=5, max_refined_level=1000, output_delta=True, output_v=True,
              output_pres=True, output_pot=True, output_opot=False, output_temp=True, output_metalicity=True,
              output_cr0amr=True, output_solapst=True, is_mascletB=False, output_B=False, is_cooling=True,
              verbose=False, read_region=None):
    """
    Reads the gas (baryonic, clus) file

    Args:
        it: iteration number (int)
        path: path of the grids file in the system (str)
        parameters_path: path of the json parameters file of the simulation
        digits: number of digits the filename is written with (int)
        max_refined_level: maximum refinement level that wants to be read. Subsequent refinements will be skipped. (int)
        output_delta: whether delta (density contrast) is returned (bool)
        output_v: whether velocities (vx, vy, vz) are returned (bool)
        output_pres: whether pressure is returned (bool)
        output_pot: whether gravitational potential is returned (bool)
        output_opot: whether gravitational potential in the previous iteration is returned (bool)
        output_temp: whether temperature is returned (bool)
        output_metalicity: whether metalicity is returned (bool)
        output_cr0amr: whether "refined variable" (1 if not refined, 0 if refined) is returned (bool)
        output_solapst: whether "solapst variable" (1 if the cell is kept, 0 otherwise) is returned (bool)
        is_mascletB: whether the outputs correspond to masclet-B (contains magnetic fields) (bool)
        output_B: whether magnetic field is returned; only if is_mascletB = True (bool)
        is_cooling: whether there is cooling (an thus T and metalicity are written) or not (bool)
        verbose: whether a message is printed when each refinement level is started (bool)
        read_region: whether to select a subregion (see region specification below), or keep all the simulation data 
                     (None). If a region wants to be selected, there are the following possibilities:
                        - ("sphere", cx, cy, cz, R) for a sphere of radius R centered in (cx, cy, cz)
                        - ("box", x1, x2, y1, y2, z1, z2) for a box with corners (x1, y1, z1) and (x2, y2, z2)
                        - ("box_cw", xc, yc, zc, width) for a box centered in (xc, yc, zc) with width "width"

    Returns:
        Chosen quantities, as a list of arrays (one for each patch, starting with l=0 and subsequently);
        in the order specified by the order of the parameters in this definition.
        
        If read_region is not None, only the patches inside the region are read, and in the positions of the 
        arrays corresponding to the patches outside the region, a single scalar value of zero is written. Also
        in this case, after all the returned variables, a 1d array of booleans is returned, with the same length
        as the number of patches, indicating which patches are inside the region (True) and which are not (False).
    """
    #if not verbose:
    #    def tqdm(x): return x

    if output_B and (not is_mascletB):
        print('Error: cannot output magnetic field if the simulation has not.')
        print('Terminating')
        return

    nmax, nmay, nmaz, nlevels, size = parameters.read_parameters(load_nma=True, load_npalev=False, load_nlevels=True,
                                                                 load_namr=False, load_size=True, path=parameters_path)
    npatch, patchnx, patchny, patchnz, \
            patchrx, patchry, patchrz = read_grids(it, path=path, parameters_path=parameters_path, read_general=False,
                                                   read_patchnum=True, read_dmpartnum=False,
                                                   read_patchcellextension=True, read_patchcellposition=False,
                                                   read_patchposition=True, read_patchparent=False)

    nfields=9
    if is_cooling: 
        nfields+=2
    if is_mascletB: 
        nfields+=3

    if read_region is None:
        keep_patches = np.ones(patchnx.size, dtype='bool')
    else:
        keep_patches = np.zeros(patchnx.size, dtype='bool')
        region_type = read_region[0]
        if region_type == 'sphere':
            cx, cy, cz, R = read_region[1:]
            which = tools.which_patches_inside_sphere(R, cx, cy, cz, patchnx, patchny, patchnz, patchrx, patchry, 
                                                      patchrz, npatch, size, nmax)
            keep_patches[which] = True
        elif region_type == 'box' or region_type == 'box_cw':
            if region_type == 'box':
                x1, x2, y1, y2, z1, z2 = read_region[1:]
            else:
                xc, yc, zc, width = read_region[1:]
                x1 = xc - width/2
                x2 = xc + width/2
                y1 = yc - width/2
                y2 = yc + width/2
                z1 = zc - width/2
                z2 = zc + width/2
            which = tools.which_patches_inside_box((x1, x2, y1, y2, z1, z2), patchnx, patchny, patchnz, patchrx, patchry,
                                                   patchrz,npatch,size,nmax)
            keep_patches[which] = True
        else:
            raise ValueError('Unknown region type. Please specify one of "sphere", "box" or "box_cw"')

    with open(os.path.join(path, filename(it, 'b', digits)), 'rb') as f:
        # read header
        header=read_record(f, dtype='f4')
        time, z = header[1:3]

        # l=0
        if verbose:
            print('Reading base grid...')
        if output_delta:
            delta = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
        else:
            skip_record(f)

        if output_v:
            vx = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            vy = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            vz = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
        else:
            for irec in range(3):
                skip_record(f)

        if output_pres:
            pres = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
        else:
            skip_record(f)

        if output_pot:
            pot = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
        else:
            skip_record(f)

        if output_opot:
            opot = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
        else:
            skip_record(f)

        if is_cooling:
            if output_temp:
                temp = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            else:
                skip_record(f)

            if output_metalicity:
                metalicity = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            else:
                skip_record(f)

        if output_cr0amr:
            cr0amr = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F').astype('bool')]
        else:
            skip_record(f)

        if output_solapst:
            solapst = [1]

        if is_mascletB:
            if output_B:
                Bx = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
                By = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
                Bz = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            else:
                for irec in range(3):
                    skip_record(f)


        # refinement levels
        for l in range(1, min(nlevels + 1, max_refined_level + 1)):
            if verbose:
                print('Reading level {}.'.format(l))
                print('{} patches.'.format(npatch[l]))
            for ipatch in tqdm(range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1), desc='Level {:}'.format(l)):
                #if verbose:
                #    print('Reading patch {}'.format(ipatch))
                if not keep_patches[ipatch]:
                    length_field=patchnx[ipatch]*patchny[ipatch]*patchnz[ipatch]*4+8
                    f.seek(nfields*length_field,1)

                    if output_delta:
                        delta.append(0)
                    if output_v:
                        vx.append(0)
                        vy.append(0)
                        vz.append(0)
                    if output_pres:
                        pres.append(0)
                    if output_pot:
                        pot.append(0)
                    if output_opot:
                        opot.append(0)
                    if is_cooling:
                        if output_temp:
                            temp.append(0)
                        if output_metalicity:
                            metalicity.append(0)
                    if output_cr0amr:
                        cr0amr.append(0)
                    if output_solapst:
                        solapst.append(0)
                    if is_mascletB:
                        if output_B:
                            Bx.append(0)
                            By.append(0)
                            Bz.append(0)

                    continue

                if output_delta and keep_patches[ipatch]:
                    delta.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                            'F'))
                else:
                    skip_record(f)
                    if output_delta:
                        delta.append(0)

                if output_v and keep_patches[ipatch]:
                    vx.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    vy.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    vz.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                else:
                    for irec in range(3):
                        skip_record(f)
                    if output_v:
                        vx.append(0)
                        vy.append(0)
                        vz.append(0)

                if output_pres and keep_patches[ipatch]:
                    pres.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                           'F'))
                else:
                    skip_record(f)
                    if output_pres:
                        pres.append(0)

                if output_pot and keep_patches[ipatch]:
                    pot.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                else:
                    skip_record(f)
                    if output_pot:
                        pot.append(0)

                if output_opot and keep_patches[ipatch]:
                    opot.append(
                        np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                else:
                    skip_record(f)
                    if output_opot:
                        opot.append(0)

                if is_cooling:
                    if output_temp and keep_patches[ipatch]:
                        temp.append(
                            np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    else:
                        skip_record(f)
                        if output_temp:
                            temp.append(0)

                    if output_metalicity and keep_patches[ipatch]:
                        metalicity.append(
                            np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    else:
                        skip_record(f)
                        if output_metalicity:
                            metalicity.append(0)

                if output_cr0amr and keep_patches[ipatch]:
                    cr0amr.append(np.reshape(read_record(f, dtype='i4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                             'F').astype('bool'))
                else:
                    skip_record(f)
                    if output_cr0amr:
                        cr0amr.append(0)

                if output_solapst and keep_patches[ipatch]:
                    solapst.append(np.reshape(read_record(f, dtype='i4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                              'F').astype('bool'))
                else:
                    skip_record(f)
                    if output_solapst:
                        solapst.append(0)

                if is_mascletB:
                    if output_B and keep_patches[ipatch]:
                        Bx.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                             'F'))
                        By.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                             'F'))
                        Bz.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                             'F'))
                    else:
                        for irec in range(3):
                            skip_record(f)
                        if output_B:
                            Bx.append(0)
                            By.append(0)
                            Bz.append(0)

    returnvariables = []
    if output_delta:
        returnvariables.append(delta)
    if output_v:
        returnvariables.extend([vx, vy, vz])
    if output_pres:
        returnvariables.append(pres)
    if output_pot:
        returnvariables.append(pot)
    if output_opot:
        returnvariables.append(opot)
    if output_temp:
        returnvariables.append(temp)
    if output_metalicity:
        returnvariables.append(metalicity)
    if output_cr0amr:
        returnvariables.append(cr0amr)
    if output_solapst:
        returnvariables.append(solapst)
    if output_B:
        returnvariables.extend([Bx,By,Bz])

    if read_region is not None:
        returnvariables.append(keep_patches)

    return tuple(returnvariables)


def read_cldm(it, path='', parameters_path='', digits=5, max_refined_level=1000, output_deltadm=True,
              output_position=True, output_velocity=True, output_mass=True, output_id=False, verbose=False):
    """
    Reads the dark matter (cldm) file.

    Args:
        it: iteration number (int)
        path: path of the output files in the system (str)
        parameters_path: path of the json parameters file of the simulation
        digits: number of digits the filename is written with (int)
        max_refined_level: maximum refinement level that wants to be read. Subsequent refinements will be skipped. (int)
        output_deltadm: whether deltadm (dark matter density contrast) is returned (bool)
        output_position: whether particles' positions are returned (bool)
        output_velocity: whether particles' velocities are returned (bool)
        output_mass: whether particles' masses are returned (bool)
        output_id: whether particles' ids are returned (bool)
        verbose: whether a message is printed when each refinement level is started (bool)

    Returns:
        Chosen quantities, in the order specified by the order of the parameters in this definition.
        delta_dm is returned as a list of numpy matrices. The 0-th element corresponds to l=0. The i-th element
        corresponds to the i-th patch.
        The rest of quantities are outputted as numpy vectors, the i-th element corresponding to the i-th DM particle.


    """
    #if not verbose:
    #    def tqdm(x): return x
    nmax, nmay, nmaz, nlevels = parameters.read_parameters(load_nma=True, load_npalev=False, load_nlevels=True,
                                                           load_namr=False, load_size=False, path=parameters_path)
    npatch, npart, patchnx, patchny, patchnz = read_grids(it, path=path, read_general=False, read_patchnum=True,
                                                          read_dmpartnum=True, read_patchcellextension=True,
                                                          read_patchcellposition=False, read_patchposition=False,
                                                          read_patchparent=False, parameters_path=parameters_path)
    
    if npart.max()>5e8:
        print('Warning! Fortran unformatted file contains split records')
        print('... Resorting to the low-level reader')
        print('... This may take a while')
        return lowlevel_read_cldm(it, path=path, parameters_path=parameters_path, digits=digits, 
                                  max_refined_level=max_refined_level, output_deltadm=output_deltadm,
                                  output_position=output_position, output_velocity=output_velocity, 
                                  output_mass=output_mass, output_id=output_id, verbose=verbose)

    with FF(os.path.join(path, filename(it, 'd', digits))) as f:

        # read header
        it_cldm = f.read_vector('i')[0]

        f.seek(0)  # this is a little bit ugly but whatever
        time, mdmpart, z = tuple(f.read_vector('f')[1:4])

        # l=0
        if verbose:
            print('Reading base grid...')

        if output_deltadm:
            delta_dm = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
        else:
            f.skip()

        if output_position:
            dmpart_x = f.read_vector('f')
            dmpart_y = f.read_vector('f')
            dmpart_z = f.read_vector('f')
        else:
            f.skip(3)

        if output_velocity:
            dmpart_vx = f.read_vector('f')
            dmpart_vy = f.read_vector('f')
            dmpart_vz = f.read_vector('f')
        else:
            f.skip(3)

        if output_id:
            dmpart_id = f.read_vector('i')
        else:
            f.skip()

        if output_mass:
            dmpart_mass = (mdmpart * np.ones(npart[0])).astype('f4')

        # refinement levels
        for l in range(1, min(nlevels + 1, max_refined_level + 1)):
            if verbose:
                print('Reading level {}.'.format(l))
                print('{} patches. {} particles.'.format(npatch[l], npart[l]))
            for ipatch in tqdm(range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1), desc='Level {:}'.format(l)):
                if output_deltadm:
                    delta_dm.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                               'F'))
                else:
                    f.skip()
            if output_position:
                dmpart_x = np.append(dmpart_x, f.read_vector('f'))
                dmpart_y = np.append(dmpart_y, f.read_vector('f'))
                dmpart_z = np.append(dmpart_z, f.read_vector('f'))
            else:
                f.skip(3)

            if output_velocity:
                dmpart_vx = np.append(dmpart_vx, f.read_vector('f'))
                dmpart_vy = np.append(dmpart_vy, f.read_vector('f'))
                dmpart_vz = np.append(dmpart_vz, f.read_vector('f'))
            else:
                f.skip(3)

            if output_mass:
                dmpart_mass = np.append(dmpart_mass, f.read_vector('f'))
            else:
                f.skip()

            if output_id:
                dmpart_id = np.append(dmpart_id, f.read_vector('i'))
            else:
                f.skip()

    returnvariables = []

    if output_deltadm:
        returnvariables.append(delta_dm)
    if output_position:
        returnvariables.extend([dmpart_x, dmpart_y, dmpart_z])
    if output_velocity:
        returnvariables.extend([dmpart_vx, dmpart_vy, dmpart_vz])
    if output_mass:
        returnvariables.append(dmpart_mass)
    if output_id:
        returnvariables.append(dmpart_id)

    return tuple(returnvariables)

def read_record(f, dtype='f4'):
    # Following the description of the standard, found in:
    #  https://stackoverflow.com/questions/15608421/inconsistent-record-marker-while-reading-fortran-unformatted-file
    # This method works always, even when there are split records (because they are longer than 2^31 bytes)!!!!
    head=struct.unpack('i',f.read(4))
    recl=head[0]
    
    if recl>0:
        numval=recl//np.dtype(dtype).itemsize
        data=np.fromfile(f,dtype=dtype,count=numval)
        endrec=struct.unpack('i',f.read(4))[0] 
        #assert recl==endrec
    else:
        the_bytes=f.read(abs(recl))
        endrec=struct.unpack('i',f.read(4))[0] 
        while recl<0:
            head=struct.unpack('i',f.read(4))
            recl=head[0]
            the_bytes=the_bytes+f.read(abs(recl))
            endrec=struct.unpack('i',f.read(4))[0] 
            
        if dtype=='f4':
            dtype2='f'
            len_data=4
        elif dtype=='i4':
            dtype2='i'
            len_data=4
        elif dtype=='f8':
            dtype2='d'
            len_data=4
        elif dtype=='i8':
            dtype2='q'
            len_data=4
        else:
            print('Unknown data type!!!!')
            raise ValueError
        dtype2='{:}'.format(len(the_bytes)//len_data)+dtype2
        #print(dtype2)
        
        data=struct.unpack(dtype2,the_bytes)
    
    if len(data)==0:
        return np.array([], dtype=dtype)

    return data

def skip_record(f):
    head=struct.unpack('i',f.read(4))
    recl=head[0]

    if recl>0:
        f.seek(recl,1)
        endrec=struct.unpack('i',f.read(4))[0] 
        #assert recl==endrec
    else:
        f.seek(abs(recl),1)
        endrec=struct.unpack('i',f.read(4))[0] 
        while recl<0:
            head=struct.unpack('i',f.read(4))
            recl=head[0]
            f.seek(abs(recl),1)
            endrec=struct.unpack('i',f.read(4))[0] 

    return


def lowlevel_read_cldm(it, path='', parameters_path='', digits=5, max_refined_level=1000, output_deltadm=True,
                       output_position=True, output_velocity=True, output_mass=True, output_id=False, verbose=False,
                       read_region=None):
    """
    Reads the dark matter (cldm) file.
    This is a low level implementation, useful when there are more than 2^29-1 particles at a given refinement level.

    Args:
        it: iteration number (int)
        path: path of the output files in the system (str)
        parameters_path: path of the json parameters file of the simulation
        digits: number of digits the filename is written with (int)
        max_refined_level: maximum refinement level that wants to be read. Subsequent refinements will be skipped. (int)
        output_deltadm: whether deltadm (dark matter density contrast) is returned (bool)
        output_position: whether particles' positions are returned (bool)
        output_velocity: whether particles' velocities are returned (bool)
        output_mass: whether particles' masses are returned (bool)
        output_id: whether particles' ids are returned (bool)
        verbose: whether a message is printed when each refinement level is started (bool)
        read_region: whether to select a subregion (see region specification below), or keep all the simulation data 
        (None). If a region wants to be selected, there are the following possibilities:
        - ("sphere", cx, cy, cz, R) for a sphere of radius R centered in (cx, cy, cz)
        - ("box", x1, x2, y1, y2, z1, z2) for a box with corners (x1, y1, z1) and (x2, y2, z2)
        - ("box_cw", xc, yc, zc, width) for a box centered in (xc, yc, zc) with width "width"

    Returns:
        Chosen quantities, in the order specified by the order of the parameters in this definition.
        delta_dm is returned as a list of numpy matrices. The 0-th element corresponds to l=0. The i-th element
        corresponds to the i-th patch.
        The rest of quantities are outputted as numpy vectors, the i-th element corresponding to the i-th DM particle.

        If read_region is not None, only the patches inside the region are read, and in the positions of the 
        arrays corresponding to the patches outside the region, a single scalar value of zero is written. Also
        in this case, after all the returned variables, a 1d array of booleans is returned, with the same length
        as the number of patches, indicating which patches are inside the region (True) and which are not (False).
        Regarding particles, all are read, but the ones outside the region are not returned.



    """
    #if not verbose:
    #    def tqdm(x): return x

    force_read_positions = False
    if (read_region is not None) and (output_position or output_velocity or output_mass or output_id):
        force_read_positions = True

    nmax, nmay, nmaz, nlevels, size = parameters.read_parameters(load_nma=True, load_npalev=False, load_nlevels=True,
                                                                 load_namr=False, load_size=True, path=parameters_path)
    npatch, npart, patchnx, patchny, patchnz, \
                   patchrx, patchry, patchrz = read_grids(it, path=path, read_general=False, read_patchnum=True,
                                                          read_dmpartnum=True, read_patchcellextension=True,
                                                          read_patchcellposition=False, read_patchposition=True,
                                                          read_patchparent=False, parameters_path=parameters_path)

    if read_region is None:
        keep_patches = np.ones(patchnx.size, dtype='bool')
    else:
        keep_patches = np.zeros(patchnx.size, dtype='bool')
        region_type = read_region[0]
        if region_type == 'sphere':
            cx, cy, cz, R = read_region[1:]
            which = tools.which_patches_inside_sphere(R, cx, cy, cz, patchnx, patchny, patchnz, patchrx, patchry, 
                                                      patchrz, npatch, size, nmax)
            keep_patches[which] = True
        elif region_type == 'box' or region_type == 'box_cw':
            if region_type == 'box':
                x1, x2, y1, y2, z1, z2 = read_region[1:]
            else:
                xc, yc, zc, width = read_region[1:]
                x1 = xc - width/2
                x2 = xc + width/2
                y1 = yc - width/2
                y2 = yc + width/2
                z1 = zc - width/2
                z2 = zc + width/2
            which = tools.which_patches_inside_box((x1, x2, y1, y2, z1, z2), patchnx, patchny, patchnz, patchrx, patchry,
                                                   patchrz,npatch,size,nmax)
            keep_patches[which] = True
        else:
            raise ValueError('Unknown region type. Please specify one of "sphere", "box" or "box_cw"')

    with open(os.path.join(path, filename(it, 'd', digits)), 'rb') as f:
        header=read_record(f, dtype='i4')
        it_cldm=header[0]
    
    with open(os.path.join(path, filename(it, 'd', digits)), 'rb') as f:
        header=read_record(f, dtype='f4')
        time, mdmpart, z = header[1:4]

        # l=0
        if verbose:
            print('Reading base grid...')

        if output_deltadm:
            delta_dm = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
        else:
            skip_record(f)

        if output_position or force_read_positions:
            dmpart_x = read_record(f, dtype='f4')
            dmpart_y = read_record(f, dtype='f4')
            dmpart_z = read_record(f, dtype='f4')
        else:
            for irec in range(3):
                skip_record(f)

        if output_velocity:
            dmpart_vx = read_record(f, dtype='f4')
            dmpart_vy = read_record(f, dtype='f4')
            dmpart_vz = read_record(f, dtype='f4')
        else:
            for irec in range(3):
                skip_record(f)

        if output_id:
            dmpart_id = read_record(f, dtype='i4')
        else:
            skip_record(f)

        if output_mass:
            dmpart_mass = (mdmpart * np.ones(npart[0])).astype('f4')

        # refinement levels
        for l in range(1, min(nlevels + 1, max_refined_level + 1)):
            if verbose:
                print('Reading level {}.'.format(l))
                print('{} patches. {} particles.'.format(npatch[l], npart[l]))

            if output_deltadm:
                for ipatch in tqdm(range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1), desc='Level {:}'.format(l)):
                    if keep_patches[ipatch]:
                        delta_dm.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    else:
                        skip_record(f)
                        delta_dm.append(0)
            else:
                length_field=0
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                    length_field += patchnx[ipatch]*patchny[ipatch]*patchnz[ipatch]*4+8
                f.seek(length_field,1)
            
            if output_position or force_read_positions:
                dmpart_x = np.append(dmpart_x, read_record(f, dtype='f4'))
                dmpart_y = np.append(dmpart_y, read_record(f, dtype='f4'))
                dmpart_z = np.append(dmpart_z, read_record(f, dtype='f4'))
            else:
                for irec in range(3):
                    skip_record(f)

            if output_velocity:
                dmpart_vx = np.append(dmpart_vx, read_record(f, dtype='f4'))
                dmpart_vy = np.append(dmpart_vy, read_record(f, dtype='f4'))
                dmpart_vz = np.append(dmpart_vz, read_record(f, dtype='f4'))
            else:
                for irec in range(3):
                    skip_record(f)

            if output_mass:
                dmpart_mass = np.append(dmpart_mass, read_record(f, dtype='f4'))
            else:
                skip_record(f)

            if output_id:
                dmpart_id = np.append(dmpart_id, read_record(f, dtype='i4'))
            else:
                skip_record(f)

    returnvariables = []

    if force_read_positions:
        if region_type == 'sphere':
            keep_particles = (dmpart_x-cx)**2 + (dmpart_y-cy)**2 + (dmpart_z-cz)**2 < R**2
        elif region_type == 'box' or region_type == 'box_cw':
            keep_particles = (dmpart_x>x1) * (dmpart_x<x2) * (dmpart_y>y1) * (dmpart_y<y2) * (dmpart_z>z1) * (dmpart_z<z2)
        
        if output_position:
            dmpart_x = dmpart_x[keep_particles]
            dmpart_y = dmpart_y[keep_particles]
            dmpart_z = dmpart_z[keep_particles]
        if output_velocity:
            dmpart_vx = dmpart_vx[keep_particles]
            dmpart_vy = dmpart_vy[keep_particles]
            dmpart_vz = dmpart_vz[keep_particles]
        if output_mass:
            dmpart_mass = dmpart_mass[keep_particles]
        if output_id:
            dmpart_id = dmpart_id[keep_particles]

    if output_deltadm:
        returnvariables.append(delta_dm)
    if output_position:
        returnvariables.extend([dmpart_x, dmpart_y, dmpart_z])
    if output_velocity:
        returnvariables.extend([dmpart_vx, dmpart_vy, dmpart_vz])
    if output_mass:
        returnvariables.append(dmpart_mass)
    if output_id:
        returnvariables.append(dmpart_id)

    return tuple(returnvariables)


def read_clst(it, path='', parameters_path='', digits=5, max_refined_level=1000, output_deltastar=True, verbose=False,
              output_position=False, output_velocity=False, output_mass=False, output_time=False,
              output_metalicity=False, output_id=False, output_BH = False, are_BH = True):
    """
    Reads the stellar (clst) file.

    Args:
        it: iteration number (int)
        path: path of the output files in the system (str)
        parameters_path: path of the json parameters file of the simulation
        digits: number of digits the filename is written with (int)
        max_refined_level: maximum refinement level that wants to be read. Subsequent refinements will be skipped. (int)
        output_deltastar: whether deltadm (dark matter density contrast) is returned (bool)
        output_position: whether particles' positions are returned (bool)
        output_velocity: whether particles' velocities are returned (bool)
        output_mass: whether particles' masses are returned (bool)
        output_time: whether particles' birth times are returned (bool)
        output_metalicity: whether particles' metalicities are returned (bool)
        output_id: whether particles' ids are returned (bool)
        verbose: whether a message is printed when each refinement level is started (bool)
        are_BH: if True, it is assumed that BH data is appended at the end of the stellar data
        output_BH: All BH data is returned --> x, y, z, vx, vy, vz, mass, birth time, id


    Returns:
        Chosen quantities, in the order specified by the order of the parameters in this definition.
        delta_star is returned as a list of numpy matrices. The 0-th element corresponds to l=0. The i-th element
        corresponds to the i-th patch.
        The rest of quantities are outputted as numpy vectors, the i-th element corresponding to the i-th stellar
        particle.


    """
    nmax, nmay, nmaz, nlevels = parameters.read_parameters(load_nma=True, load_npalev=False, load_nlevels=True,
                                                           load_namr=False, load_size=False, path=parameters_path)
    npatch, npart, patchnx, patchny, patchnz = read_grids(it, path=path, read_general=False, read_patchnum=True,
                                                          read_dmpartnum=True, read_patchcellextension=True,
                                                          read_patchcellposition=False, read_patchposition=False,
                                                          read_patchparent=False, parameters_path=parameters_path)

    with FF(os.path.join(path, filename(it, 's', digits))) as f:

        # read header
        it_clst = f.read_vector('i')[0]

        f.seek(0)  # this is a little bit ugly but whatever
        time, z = tuple(f.read_vector('f')[1:3])

        # l=0
        if verbose:
            print('Reading base grid...')

        if output_deltastar:
            delta_star = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
        else:
            f.skip()

        if output_position:
            stpart_x = f.read_vector('f')
            stpart_y = f.read_vector('f')
            stpart_z = f.read_vector('f')
        else:
            f.skip(3)

        if output_velocity:
            stpart_vx = f.read_vector('f')
            stpart_vy = f.read_vector('f')
            stpart_vz = f.read_vector('f')
        else:
            f.skip(3)

        if output_mass:
            stpart_mass = f.read_vector('f')
        else:
            f.skip()

        if output_time:
            stpart_time = f.read_vector('f')
        else:
            f.skip()

        if output_metalicity:
            stpart_metalicity = f.read_vector('f')
        else:
            f.skip()

        if output_id:
            stpart_id = np.zeros(stpart_x.size, dtype='i4')
        
        if are_BH and output_BH:
            bhpart_x = np.zeros(stpart_x.size)
            bhpart_y = np.zeros(stpart_x.size)
            bhpart_z = np.zeros(stpart_x.size)
            bhpart_vx = np.zeros(stpart_x.size)
            bhpart_vy = np.zeros(stpart_x.size)
            bhpart_vz = np.zeros(stpart_x.size)
            bhpart_mass = np.zeros(stpart_x.size)
            bhpart_time = np.zeros(stpart_x.size)
            bhpart_id = np.zeros(stpart_x.size).astype(np.int32)


        # refinement levels
        for l in range(1, min(nlevels + 1, max_refined_level + 1)):
            if verbose:
                print('Reading level {}.'.format(l))
                print('{} patches'.format(npatch[l]))
            for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                if output_deltastar:
                    delta_star.append(
                        np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                   'F'))
                else:
                    f.skip()

            if output_position:
                stpart_x = np.append(stpart_x, f.read_vector('f'))
                stpart_y = np.append(stpart_y, f.read_vector('f'))
                stpart_z = np.append(stpart_z, f.read_vector('f'))
            else:
                f.skip(3)

            if output_velocity:
                stpart_vx = np.append(stpart_vx, f.read_vector('f'))
                stpart_vy = np.append(stpart_vy, f.read_vector('f'))
                stpart_vz = np.append(stpart_vz, f.read_vector('f'))
            else:
                f.skip(3)

            if output_mass:
                stpart_mass = np.append(stpart_mass, f.read_vector('f'))
            else:
                f.skip()

            if output_time:
                stpart_time = np.append(stpart_time, f.read_vector('f'))
            else:
                f.skip()

            if output_metalicity:
                stpart_metalicity = np.append(stpart_metalicity, f.read_vector('f'))
            else:
                f.skip()

            if output_id:
                stpart_id = np.append(stpart_id, f.read_vector('i'))
            else:
                f.skip()
            
            if are_BH:
                if output_BH:
                    bhpart_x = np.append(bhpart_x, f.read_vector('f'))
                    bhpart_y = np.append(bhpart_y, f.read_vector('f'))
                    bhpart_z = np.append(bhpart_z, f.read_vector('f'))
                    bhpart_vx = np.append(bhpart_vx, f.read_vector('f'))
                    bhpart_vy = np.append(bhpart_vy, f.read_vector('f'))
                    bhpart_vz = np.append(bhpart_vz, f.read_vector('f'))
                    bhpart_mass = np.append(bhpart_mass, f.read_vector('f'))
                    bhpart_time = np.append(bhpart_time, f.read_vector('f'))
                    bhpart_id = np.append(bhpart_id, f.read_vector('i'))
                else:
                    f.skip(9)

    returnvariables = []

    if output_deltastar:
        returnvariables.append(delta_star)
    if output_position:
        returnvariables.extend([stpart_x, stpart_y, stpart_z])
    if output_velocity:
        returnvariables.extend([stpart_vx, stpart_vy, stpart_vz])
    if output_mass:
        returnvariables.append(stpart_mass)
    if output_time:
        returnvariables.append(stpart_time)
    if output_metalicity:
        returnvariables.append(stpart_metalicity)
    if output_id:
        returnvariables.append(stpart_id)
    if are_BH and output_BH:
        returnvariables.extend([bhpart_x, bhpart_y, bhpart_z, bhpart_vx, bhpart_vy, bhpart_vz, bhpart_mass, bhpart_time, bhpart_id])

    return tuple(returnvariables)

def lowlevel_read_clst(it, path='', parameters_path='', digits=5, max_refined_level=1000, output_deltastar=True, 
                       verbose=False, output_position=False, output_velocity=False, output_mass=False, 
                       output_time=False, output_metalicity=False, output_id=False, output_BH = False, are_BH = True,
                       read_region=None):
    """
    Reads the stellar (clst) file.

    Args:
        it: iteration number (int)
        path: path of the output files in the system (str)
        parameters_path: path of the json parameters file of the simulation
        digits: number of digits the filename is written with (int)
        max_refined_level: maximum refinement level that wants to be read. Subsequent refinements will be skipped. (int)
        output_deltastar: whether deltadm (dark matter density contrast) is returned (bool)
        output_position: whether particles' positions are returned (bool)
        output_velocity: whether particles' velocities are returned (bool)
        output_mass: whether particles' masses are returned (bool)
        output_time: whether particles' birth times are returned (bool)
        output_metalicity: whether particles' metalicities are returned (bool)
        output_id: whether particles' ids are returned (bool)
        verbose: whether a message is printed when each refinement level is started (bool)
        are_BH: if True, it is assumed that BH data is appended at the end of the stellar data
        output_BH: All BH data is returned --> x, y, z, vx, vy, vz, mass, birth time, id
        read_region: whether to select a subregion (see region specification below), or keep all the simulation data
        (None). If a region wants to be selected, there are the following possibilities:
            - ("sphere", cx, cy, cz, R) for a sphere of radius R centered in (cx, cy, cz)
            - ("box", x1, x2, y1, y2, z1, z2) for a box with corners (x1, y1, z1) and (x2, y2, z2)
            - ("box_cw", xc, yc, zc, width) for a box centered in (xc, yc, zc) with width "width"


    Returns:
        Chosen quantities, in the order specified by the order of the parameters in this definition.
        delta_star is returned as a list of numpy matrices. The 0-th element corresponds to l=0. The i-th element
        corresponds to the i-th patch.
        The rest of quantities are outputted as numpy vectors, the i-th element corresponding to the i-th stellar
        particle.

        If read_region is not None, only the patches inside the region are read, and in the positions of the 
        arrays corresponding to the patches outside the region, a single scalar value of zero is written. Also
        in this case, after all the returned variables, a 1d array of booleans is returned, with the same length
        as the number of patches, indicating which patches are inside the region (True) and which are not (False).
        Regarding particles, all are read, but the ones outside the region are not returned.



    """
    force_read_positions = False
    if (read_region is not None) and (output_position or output_velocity or output_mass or output_id or 
                                      output_metalicity or output_time or output_BH):
        force_read_positions = True

    nmax, nmay, nmaz, nlevels, size = parameters.read_parameters(load_nma=True, load_npalev=False, load_nlevels=True,
                                                                 load_namr=False, load_size=True, path=parameters_path)
    npatch, npart, patchnx, patchny, patchnz, \
                   patchrx, patchry, patchrz = read_grids(it, path=path, read_general=False, read_patchnum=True,
                                                          read_dmpartnum=True, read_patchcellextension=True,
                                                          read_patchcellposition=False, read_patchposition=True,
                                                          read_patchparent=False, parameters_path=parameters_path)

    if read_region is None:
        keep_patches = np.ones(patchnx.size, dtype='bool')
    else:
        keep_patches = np.zeros(patchnx.size, dtype='bool')
        region_type = read_region[0]
        if region_type == 'sphere':
            cx, cy, cz, R = read_region[1:]
            which = tools.which_patches_inside_sphere(R, cx, cy, cz, patchnx, patchny, patchnz, patchrx, patchry, 
                                                      patchrz, npatch, size, nmax)
            keep_patches[which] = True
        elif region_type == 'box' or region_type == 'box_cw':
            if region_type == 'box':
                x1, x2, y1, y2, z1, z2 = read_region[1:]
            else:
                xc, yc, zc, width = read_region[1:]
                x1 = xc - width/2
                x2 = xc + width/2
                y1 = yc - width/2
                y2 = yc + width/2
                z1 = zc - width/2
                z2 = zc + width/2
            which = tools.which_patches_inside_box((x1, x2, y1, y2, z1, z2), patchnx, patchny, patchnz, patchrx, patchry,
                                                   patchrz,npatch,size,nmax)
            keep_patches[which] = True
        else:
            raise ValueError('Unknown region type. Please specify one of "sphere", "box" or "box_cw"')

    with open(os.path.join(path, filename(it, 's', digits)), 'rb') as f:
        # read header
        header=read_record(f, dtype='f4')
        time, z = header[1:3]

        # l=0
        if verbose:
            print('Reading base grid...')

        if output_deltastar:
            delta_star = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
        else:
            skip_record(f)

        if output_position:
            stpart_x = read_record(f, dtype='f4')
            stpart_y = read_record(f, dtype='f4')
            stpart_z = read_record(f, dtype='f4')
            if len(stpart_x)==0:
                stpart_x = np.array([], dtype='f4')
                stpart_y = np.array([], dtype='f4')
                stpart_z = np.array([], dtype='f4')
        else:
            for irec in range(3):
                skip_record(f)

        if output_velocity:
            stpart_vx = read_record(f, dtype='f4')
            stpart_vy = read_record(f, dtype='f4')
            stpart_vz = read_record(f, dtype='f4')
            if len(stpart_vx)==0:
                stpart_vx = np.array([], dtype='f4')
                stpart_vy = np.array([], dtype='f4')
                stpart_vz = np.array([], dtype='f4')
        else:
             for irec in range(3):
                skip_record(f)

        if output_mass:
            stpart_mass = read_record(f, dtype='f4')
            if len(stpart_mass)==0:
                stpart_mass = np.array([], dtype='f4')
        else:
            skip_record(f)

        if output_time:
            stpart_time = read_record(f, dtype='f4')
            if len(stpart_time)==0:
                stpart_time = np.array([], dtype='f4')
        else:
            skip_record(f)

        if output_metalicity:
            stpart_metalicity = read_record(f, dtype='f4')
            if len(stpart_metalicity)==0:
                stpart_metalicity = np.array([], dtype='f4')
        else:
            skip_record(f)

        if output_id:
            stpart_id = np.zeros(stpart_x.size, dtype='i4').astype('i4')
        
        if are_BH and output_BH:
            bhpart_x = np.zeros(stpart_x.size)
            bhpart_y = np.zeros(stpart_x.size)
            bhpart_z = np.zeros(stpart_x.size)
            bhpart_vx = np.zeros(stpart_x.size)
            bhpart_vy = np.zeros(stpart_x.size)
            bhpart_vz = np.zeros(stpart_x.size)
            bhpart_mass = np.zeros(stpart_x.size)
            bhpart_time = np.zeros(stpart_x.size)
            bhpart_id = np.zeros(stpart_x.size).astype(np.int32)

        # refinement levels
        for l in range(1, min(nlevels + 1, max_refined_level + 1)):
            if verbose:
                print('Reading level {}.'.format(l))
                print('{} patches'.format(npatch[l]))

            if output_deltastar:
                for ipatch in tqdm(range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1), desc='Level {:}'.format(l)):
                    if keep_patches[ipatch]:
                        delta_star.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    else:
                        skip_record(f)
            else:
                length_field=0
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                    length_field += patchnx[ipatch]*patchny[ipatch]*patchnz[ipatch]*4+8
                f.seek(length_field,1)


            if output_position:
                stpart_x = np.append(stpart_x, read_record(f, dtype='f4'))
                stpart_y = np.append(stpart_y, read_record(f, dtype='f4'))
                stpart_z = np.append(stpart_z, read_record(f, dtype='f4'))
            else:
                for irec in range(3):
                    skip_record(f)

            if output_velocity:
                stpart_vx = np.append(stpart_vx, read_record(f, dtype='f4'))
                stpart_vy = np.append(stpart_vy, read_record(f, dtype='f4'))
                stpart_vz = np.append(stpart_vz, read_record(f, dtype='f4'))
            else:
                for irec in range(3):
                    skip_record(f)

            if output_mass:
                stpart_mass = np.append(stpart_mass, read_record(f, dtype='f4'))
            else:
                skip_record(f)

            if output_time:
                stpart_time = np.append(stpart_time, read_record(f, dtype='f4'))
            else:
                skip_record(f)

            if output_metalicity:
                stpart_metalicity = np.append(stpart_metalicity, read_record(f, dtype='f4'))
            else:
                skip_record(f)

            if output_id:
                stpart_id = np.append(stpart_id, read_record(f, dtype='i4'))
            else:
                skip_record(f)
            
            if are_BH:
                if output_BH:
                    bhpart_x = np.append(bhpart_x, read_record(f, dtype='f4'))
                    bhpart_y = np.append(bhpart_y, read_record(f, dtype='f4'))
                    bhpart_z = np.append(bhpart_z, read_record(f, dtype='f4'))
                    bhpart_vx = np.append(bhpart_vx, read_record(f, dtype='f4'))
                    bhpart_vy = np.append(bhpart_vy, read_record(f, dtype='f4'))
                    bhpart_vz = np.append(bhpart_vz, read_record(f, dtype='f4'))
                    bhpart_mass = np.append(bhpart_mass, read_record(f, dtype='f4'))
                    bhpart_time = np.append(bhpart_time, read_record(f, dtype='f4'))
                    bhpart_id = np.append(bhpart_id, read_record(f, dtype='i4'))
                else:
                    for irec in range(9):
                        skip_record(f)

    returnvariables = []

    if force_read_positions:
        if region_type == 'sphere':
            keep_particles = (stpart_x-cx)**2 + (stpart_y-cy)**2 + (stpart_z-cz)**2 < R**2
            if output_BH:
                keep_particles_BH = (bhpart_x-cx)**2 + (bhpart_y-cy)**2 + (bhpart_z-cz)**2 < R**2
        elif region_type == 'box' or region_type == 'box_cw':
            keep_particles = (stpart_x>x1) * (stpart_x<x2) * (stpart_y>y1) * (stpart_y<y2) * (stpart_z>z1) * (stpart_z<z2)
            if output_BH:
                keep_particles_BH = (bhpart_x>x1) * (bhpart_x<x2) * (bhpart_y>y1) * (bhpart_y<y2) * (bhpart_z>z1) * (bhpart_z<z2)
        
        if output_position:
            stpart_x = stpart_x[keep_particles]
            stpart_y = stpart_y[keep_particles]
            stpart_z = stpart_z[keep_particles]
        if output_velocity:
            stpart_vx = stpart_vx[keep_particles]
            stpart_vy = stpart_vy[keep_particles]
            stpart_vz = stpart_vz[keep_particles]
        if output_mass:
            stpart_mass = stpart_mass[keep_particles]
        if output_time:
            stpart_time = stpart_time[keep_particles]
        if output_metalicity:
            stpart_metalicity = stpart_metalicity[keep_particles]
        if output_id:
            stpart_id = stpart_id[keep_particles]
        if output_BH:
            bhpart_x = bhpart_x[keep_particles_BH]
            bhpart_y = bhpart_y[keep_particles_BH]
            bhpart_z = bhpart_z[keep_particles_BH]
            bhpart_vx = bhpart_vx[keep_particles_BH]
            bhpart_vy = bhpart_vy[keep_particles_BH]
            bhpart_vz = bhpart_vz[keep_particles_BH]
            bhpart_mass = bhpart_mass[keep_particles_BH]
            bhpart_time = bhpart_time[keep_particles_BH]
            bhpart_id = bhpart_id[keep_particles_BH]

    if output_deltastar:
        returnvariables.append(delta_star)
    if output_position:
        returnvariables.extend([stpart_x, stpart_y, stpart_z])
    if output_velocity:
        returnvariables.extend([stpart_vx, stpart_vy, stpart_vz])
    if output_mass:
        returnvariables.append(stpart_mass)
    if output_time:
        returnvariables.append(stpart_time)
    if output_metalicity:
        returnvariables.append(stpart_metalicity)
    if output_id:
        returnvariables.append(stpart_id)
    if are_BH and output_BH:
        returnvariables.extend([bhpart_x, bhpart_y, bhpart_z, bhpart_vx, bhpart_vy, bhpart_vz, bhpart_mass, bhpart_time, bhpart_id])

    return tuple(returnvariables)


def read_npz_field(filename, path=''):
    """
    Reads a field written using the numpy savez function.
    E.g., read solapst variable compute with my ./Projects/TFM/Compute_solapst code

    Args:
        filename: name of the npz file
        path: path to the npz file

    Returns:
        A list of numpy arrays, containing the field

    """

    filename = os.path.join(path, filename)

    field = []
    with np.load(filename) as f:
        for arrayname in f:
            field.append(f[arrayname])

    return field

def read_vortex(it, path='', grids_path='', parameters_path='', digits=5, are_divrot=True, are_potentials=True,
                are_velocities=True, is_total_velocity = False, is_filtered=False, is_header=True, verbose=False):
    
    """
    Reads the vortex (Helmholtz-Hodge decomposition) files, velocity##### and filten#####

    Args:
        it: iteration number (int)
        path: path of the grids file in the system (str)
        parameters_path: path of the json parameters file of the simulation
        digits: number of digits the filename is written with (int)
        max_refined_level: maximum refinement level that wants to be read. Subsequent refinements will be skipped. (int)
        are_divrot: whehther velocity divergences and rotationals are written in the file
        are_potentials: whether (scalar and vector) potentials are written in the file
        are_velocities: whether ([total], compressional and rotational) velocities are written in the file
        is_filtered: whether the multiscale filtering file (filtlen) is written and the scale lenght and turbulent velocity field are to be read
        is_solapst: whether the overlap variable computed using the error estimate is written in the file

    Returns:
        Chosen quantities, as a list of arrays (one for each patch, starting with l=0 and subsequently);
        in the order specified by the order of the parameters in this definition.
    """

    nmax, nmay, nmaz, nlevels = parameters.read_parameters(load_nma=True, load_npalev=False, load_nlevels=True,
                                                           load_namr=False, load_size=False, path=parameters_path)

    npatch, patchnx, patchny, patchnz = read_grids(it, path=grids_path, parameters_path=parameters_path, read_general=False,
                                                   read_patchnum=True, read_dmpartnum=False,
                                                   read_patchcellextension=True, read_patchcellposition=False,
                                                   read_patchposition=False, read_patchparent=False)

    returnvariables = []
    with FF(os.path.join(path, filename(it, 'v', digits))) as f:
        # read header
        if is_header:
            it_clus = f.read_vector('i')[0]
            # assert(it == it_clus)
            f.seek(0)  # this is a little bit ugly but whatever
            time, z = tuple(f.read_vector('f')[1:3])

        if are_divrot:
            # divergence
            if verbose:
                print('Reading divergence...')
            div = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            for l in range(1, nlevels + 1):
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                    div.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))

            # rotational
            if verbose:
                print('Reading rotational...')
            rotx = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            roty = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            rotz = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            for l in range(1, nlevels + 1):
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                    rotx.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    roty.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    rotz.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))

            returnvariables.extend([div, rotx, roty, rotz])

        
        # scalar
        if verbose:
            print('Reading scalar potential...')
        if are_potentials:
            scalarpot = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            for l in range(1, nlevels + 1):
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                        scalarpot.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch],
                                                                        patchnz[ipatch]), 'F'))

        # vector
        if verbose:
            print('Reading vector potential...')

        if are_potentials:
            vecpotx = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            vecpoty = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            vecpotz = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            for l in range(1, nlevels + 1):
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                        vecpotx.append(
                            np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        vecpoty.append(
                            np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        vecpotz.append(
                            np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        
            returnvariables.extend([scalarpot, vecpotx, vecpoty, vecpotz])


        if are_velocities:
            # total
            if is_total_velocity:
                if verbose:
                    print('Reading total velocity...')
                vx = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
                vy = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
                vz = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
                for l in range(1, nlevels + 1):
                    for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                        vx.append(
                            np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        vy.append(
                            np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        vz.append(
                            np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))

            # compressive
            if verbose:
                print('Reading compressive velocity...')
            velcompx = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            velcompy = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            velcompz = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            for l in range(1, nlevels + 1):
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                    velcompx.append(
                        np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    velcompy.append(
                        np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    velcompz.append(
                        np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
            # rotational
            if verbose:
                print('Reading rotational velocity...')
            velrotx = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            velroty = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            velrotz = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            for l in range(1, nlevels + 1):
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                    velrotx.append(np.reshape(f.read_vector('f'),
                                              (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    velroty.append(np.reshape(f.read_vector('f'),
                                              (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    velrotz.append(np.reshape(f.read_vector('f'),
                                              (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))

            if is_total_velocity:        
                returnvariables.extend([vx, vy, vz, velcompx, velcompy, velcompz, velrotx, velroty, velrotz])
            else:
                returnvariables.extend([velcompx, velcompy, velcompz, velrotx, velroty, velrotz])

    if is_filtered:

        with FF(os.path.join(path, filename(it, 'f', digits))) as f:
        # filtlen files
        
            if verbose:
                print('Reading filter lenght and turbulent velocity...')

            L = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            vx = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            vy = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            vz = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            
            for l in range(1, nlevels + 1):
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                    L.append(
                        np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    vx.append(
                        np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    vy.append(
                        np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    vz.append(
                        np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))

            returnvariables.extend([L, vx, vy, vz])

    return tuple(returnvariables)

def lowlevel_read_vortex(it, path='', grids_path='', parameters_path='', digits=5, are_divrot=True, are_potentials=True,
                are_velocities=True, is_total_velocity = False, is_filtered=False, is_header=True, verbose=False,
                read_region=None):
    
    """
    Reads the vortex (Helmholtz-Hodge decomposition) files, velocity##### and filten#####

    Args:
        it: iteration number (int)
        path: path of the grids file in the system (str)
        parameters_path: path of the json parameters file of the simulation
        digits: number of digits the filename is written with (int)
        max_refined_level: maximum refinement level that wants to be read. Subsequent refinements will be skipped. (int)
        are_divrot: whehther velocity divergences and rotationals are written in the file
        are_potentials: whether (scalar and vector) potentials are written in the file
        are_velocities: whether ([total], compressional and rotational) velocities are written in the file
        is_filtered: whether the multiscale filtering file (filtlen) is written and the scale lenght and turbulent velocity field are to be read
        is_solapst: whether the overlap variable computed using the error estimate is written in the file
        read_region: whether to select a subregion (see region specification below), or keep all the simulation data 
                (None). If a region wants to be selected, there are the following possibilities:
                - ("sphere", cx, cy, cz, R) for a sphere of radius R centered in (cx, cy, cz)
                - ("box", x1, x2, y1, y2, z1, z2) for a box with corners (x1, y1, z1) and (x2, y2, z2)
                - ("box_cw", xc, yc, zc, width) for a box centered in (xc, yc, zc) with width "width"


    Returns:
        Chosen quantities, as a list of arrays (one for each patch, starting with l=0 and subsequently);
        in the order specified by the order of the parameters in this definition.

        If read_region is not None, only the patches inside the region are read, and in the positions of the 
        arrays corresponding to the patches outside the region, a single scalar value of zero is written. Also
        in this case, after all the returned variables, a 1d array of booleans is returned, with the same length
        as the number of patches, indicating which patches are inside the region (True) and which are not (False).
    """

    nmax, nmay, nmaz, nlevels, size = parameters.read_parameters(load_nma=True, load_npalev=False, load_nlevels=True,
                                                                 load_namr=False, load_size=True, path=parameters_path)
    npatch, patchnx, patchny, patchnz, \
            patchrx, patchry, patchrz = read_grids(it, path=grids_path, parameters_path=parameters_path, read_general=False,
                                                   read_patchnum=True, read_dmpartnum=False,
                                                   read_patchcellextension=True, read_patchcellposition=False,
                                                   read_patchposition=True, read_patchparent=False)

    if read_region is None:
        keep_patches = np.ones(patchnx.size, dtype='bool')
    else:
        keep_patches = np.zeros(patchnx.size, dtype='bool')
        region_type = read_region[0]
        if region_type == 'sphere':
            cx, cy, cz, R = read_region[1:]
            which = tools.which_patches_inside_sphere(R, cx, cy, cz, patchnx, patchny, patchnz, patchrx, patchry, 
                                                      patchrz, npatch, size, nmax)
            keep_patches[which] = True
        elif region_type == 'box' or region_type == 'box_cw':
            if region_type == 'box':
                x1, x2, y1, y2, z1, z2 = read_region[1:]
            else:
                xc, yc, zc, width = read_region[1:]
                x1 = xc - width/2
                x2 = xc + width/2
                y1 = yc - width/2
                y2 = yc + width/2
                z1 = zc - width/2
                z2 = zc + width/2
            which = tools.which_patches_inside_box((x1, x2, y1, y2, z1, z2), patchnx, patchny, patchnz, patchrx, patchry,
                                                   patchrz,npatch,size,nmax)
            keep_patches[which] = True
        else:
            raise ValueError('Unknown region type. Please specify one of "sphere", "box" or "box_cw"')

    returnvariables = []
    with open(os.path.join(path, filename(it, 'v', digits)), 'rb') as f:
        # read header
        if is_header:
            header=read_record(f, dtype='f4')
            time, z = header[1:3]

        if are_divrot:
            # divergence
            if verbose:
                print('Reading divergence...')
            div = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            for l in range(1, nlevels + 1):
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                    if keep_patches[ipatch]:
                        div.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    else:
                        skip_record(f)
                        div.append(0)


            # rotational
            if verbose:
                print('Reading rotational...')
            rotx = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            roty = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            rotz = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            for l in range(1, nlevels + 1):
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                    if keep_patches[ipatch]:
                        rotx.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        roty.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        rotz.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    else:
                        length_field=patchnx[ipatch]*patchny[ipatch]*patchnz[ipatch]*4+8
                        length_field*=3
                        f.seek(length_field,1)
                        rotx.append(0)
                        roty.append(0)
                        rotz.append(0)

            returnvariables.extend([div, rotx, roty, rotz])

        
        # scalar
        if verbose:
            print('Reading scalar potential...')
        if are_potentials:
            scalarpot = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            for l in range(1, nlevels + 1):
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                    if keep_patches[ipatch]:
                        scalarpot.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    else:
                        skip_record(f)
                        scalarpot.append(0)

        # vector
        if verbose:
            print('Reading vector potential...')

        if are_potentials:
            vecpotx = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            vecpoty = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            vecpotz = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            for l in range(1, nlevels + 1):
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                    if keep_patches[ipatch]:
                        vecpotx.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        vecpoty.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        vecpotz.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    else:
                        length_field=patchnx[ipatch]*patchny[ipatch]*patchnz[ipatch]*4+8
                        length_field*=3
                        f.seek(length_field,1)
                        vecpotx.append(0)
                        vecpoty.append(0)
                        vecpotz.append(0)

            returnvariables.extend([scalarpot, vecpotx, vecpoty, vecpotz])


        if are_velocities:
            # total
            if is_total_velocity:
                if verbose:
                    print('Reading total velocity...')
                vx = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
                vy = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
                vz = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
                for l in range(1, nlevels + 1):
                    for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                        if keep_patches[ipatch]:
                            vx.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                            vy.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                            vz.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        else:
                            length_field=patchnx[ipatch]*patchny[ipatch]*patchnz[ipatch]*4+8
                            length_field*=3
                            f.seek(length_field,1)
                            vx.append(0)
                            vy.append(0)
                            vz.append(0)

            # compressive
            if verbose:
                print('Reading compressive velocity...')
            velcompx = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            velcompy = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            velcompz = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            for l in range(1, nlevels + 1):
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                    if keep_patches[ipatch]:
                        velcompx.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        velcompy.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        velcompz.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    else:
                        length_field=patchnx[ipatch]*patchny[ipatch]*patchnz[ipatch]*4+8
                        length_field*=3
                        f.seek(length_field,1)
                        velcompx.append(0)
                        velcompy.append(0)
                        velcompz.append(0)

            # rotational
            if verbose:
                print('Reading rotational velocity...')
            velrotx = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            velroty = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            velrotz = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            for l in range(1, nlevels + 1):
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                    if keep_patches[ipatch]:
                        velrotx.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        velroty.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        velrotz.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    else:
                        length_field=patchnx[ipatch]*patchny[ipatch]*patchnz[ipatch]*4+8
                        length_field*=3
                        f.seek(length_field,1)
                        velrotx.append(0)
                        velroty.append(0)
                        velrotz.append(0)

            if is_total_velocity:        
                returnvariables.extend([vx, vy, vz, velcompx, velcompy, velcompz, velrotx, velroty, velrotz])
            else:
                returnvariables.extend([velcompx, velcompy, velcompz, velrotx, velroty, velrotz])

    if is_filtered:

        with open(os.path.join(path, filename(it, 'f', digits)), 'rb') as f:
        # filtlen files
        
            if verbose:
                print('Reading filter lenght and turbulent velocity...')

            L = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            vx = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            vy = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            vz = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
            
            for l in range(1, nlevels + 1):
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                    if keep_patches[ipatch]:
                        L.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        vx.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        vy.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                        vz.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    else:
                        length_field=patchnx[ipatch]*patchny[ipatch]*patchnz[ipatch]*4+8
                        length_field*=4
                        f.seek(length_field,1)

            returnvariables.extend([L, vx, vy, vz])

    return tuple(returnvariables)


def read_mach(it, path='', grids_path='', parameters_path='', digits=5, verbose=False):
    nmax, nmay, nmaz, nlevels = parameters.read_parameters(load_nma=True, load_npalev=False, load_nlevels=True,
                                                           load_namr=False, load_size=False, path=parameters_path)
    npatch, patchnx, patchny, patchnz = read_grids(it, path=grids_path, parameters_path=parameters_path,
                                                   read_general=False,
                                                   read_patchnum=True, read_dmpartnum=False,
                                                   read_patchcellextension=True, read_patchcellposition=False,
                                                   read_patchposition=False, read_patchparent=False)

    with FF(os.path.join(path, filename(it, 'm', digits))) as f:
        if verbose:
            print('Reading mach number...')
        M = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
        for l in range(1, nlevels + 1):
            for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                M.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))

    return tuple([M])

def lowlevel_read_mach(it, path='', grids_path='', parameters_path='', digits=5, verbose=False,
                       max_refined_level=1000, read_region=None):
    """
    Reads the MachNum_XXXXX file, output by the shock finder.
    This is the low-level implementation, and incorporates the region selector.

    Args:
        it: iteration number (int)
        path: path of the MachNum file in the system (str)
        grids_path: path of the grids file in the system (str)
        parameters_path: path of the json parameters file of the simulation
        digits: number of digits the filename is written with (int)
        max_refined_level: maximum refinement level that wants to be read. Subsequent refinements will be skipped. (int)
        read_region: whether to select a subregion (see region specification below), or keep all the simulation data
            (None). If a region wants to be selected, there are the following possibilities:
            - ("sphere", cx, cy, cz, R) for a sphere of radius R centered in (cx, cy, cz)
            - ("box", x1, x2, y1, y2, z1, z2) for a box with corners (x1, y1, z1) and (x2, y2, z2)
            - ("box_cw", xc, yc, zc, width) for a box centered in (xc, yc, zc) with width "width"

    Returns:
        The Mach number field.
    """
    nmax, nmay, nmaz, nlevels, size = parameters.read_parameters(load_nma=True, load_npalev=False, load_nlevels=True,
                                                           load_namr=False, load_size=True, path=parameters_path)
    npatch, patchnx, patchny, patchnz, \
            patchrx, patchry, patchrz = read_grids(it, path=grids_path, parameters_path=parameters_path, read_general=False,
                                                   read_patchnum=True, read_dmpartnum=False,
                                                   read_patchcellextension=True, read_patchcellposition=False,
                                                   read_patchposition=True, read_patchparent=False)

    if read_region is None:
        keep_patches = np.ones(patchnx.size, dtype='bool')
    else:
        keep_patches = np.zeros(patchnx.size, dtype='bool')
        region_type = read_region[0]
        if region_type == 'sphere':
            cx, cy, cz, R = read_region[1:]
            which = tools.which_patches_inside_sphere(R, cx, cy, cz, patchnx, patchny, patchnz, patchrx, patchry, 
                                                      patchrz, npatch, size, nmax)
            keep_patches[which] = True
        elif region_type == 'box' or region_type == 'box_cw':
            if region_type == 'box':
                x1, x2, y1, y2, z1, z2 = read_region[1:]
            else:
                xc, yc, zc, width = read_region[1:]
                x1 = xc - width/2
                x2 = xc + width/2
                y1 = yc - width/2
                y2 = yc + width/2
                z1 = zc - width/2
                z2 = zc + width/2
            which = tools.which_patches_inside_box((x1, x2, y1, y2, z1, z2), patchnx, patchny, patchnz, patchrx, patchry,
                                                   patchrz,npatch,size,nmax)
            keep_patches[which] = True
        else:
            raise ValueError('Unknown region type. Please specify one of "sphere", "box" or "box_cw"')                             

    with open(os.path.join(path, filename(it, 'm', digits)), 'rb') as f:
        if verbose:
            print('Reading mach number...')
        M = [np.reshape(read_record(f, dtype='f4'), (nmax, nmay, nmaz), 'F')]
        for l in range(1, nlevels + 1):
            for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                if keep_patches[ipatch]:
                    M.append(np.reshape(read_record(f, dtype='f4'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                else:
                    skip_record(f)
                    M.append([0])

    return tuple([M])
