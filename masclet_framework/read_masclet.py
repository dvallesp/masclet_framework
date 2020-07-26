"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

read_masclet module
Provides the necessary functions for reading MASCLET files and loading them in
memory

Created by David Vallés
"""

#  Last update on 26/7/20 11:31

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE

import os
# numpy
import numpy as np
# scipy (will be removed)
from scipy.io import FortranFile
# cython_fortran_file
from cython_fortran_file import FortranFile as FF

# masclet_framework
from masclet_framework import parameters


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
    names = {'g': "grids", 'b': 'clus', 'd': 'cldm', 's': 'clst', 'v': 'velocity'}
    try:
        if np.floor(np.log10(it)) < digits:
            return names[filetype] + str(it).zfill(digits)
        else:
            raise ValueError("Digits should be greater to handle that iteration number")
    except KeyError:
        print('Insert a correct type: g, b, d, s or v')


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
    _, ndxyz, _ = tuple(float(i) for i in grids.readline().split())
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
        level, npatchtemp, nparttemp, _ = tuple(int(i) for i in grids.readline().split())
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
              output_cr0amr=True, output_solapst=True, verbose=False):
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
        verbose: whether a message is printed when each refinement level is started (bool)
        fullverbose: whether a message is printed for each patch (recommended for debugging issues) (bool)

    Returns:
        Chosen quantities, as a list of arrays (one for each patch, starting with l=0 and subsequently);
        in the order specified by the order of the parameters in this definition.
    """

    nmax, nmay, nmaz, nlevels = parameters.read_parameters(load_nma=True, load_npalev=False, load_nlevels=True,
                                                           load_namr=False, load_size=False, path=parameters_path)
    npatch, patchnx, patchny, patchnz = read_grids(it, path=path, parameters_path=parameters_path, read_general=False,
                                                   read_patchnum=True, read_dmpartnum=False,
                                                   read_patchcellextension=True, read_patchcellposition=False,
                                                   read_patchposition=False, read_patchparent=False)
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
            solapst = [0]

        # refinement levels
        for l in range(1, min(nlevels + 1, max_refined_level + 1)):
            if verbose:
                print('Reading level {}.'.format(l))
                print('{} patches.'.format(npatch[l]))
            for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                if verbose:
                    print('Reading patch {}'.format(ipatch))

                if output_delta:
                    delta.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                            'F'))
                else:
                    f.skip()

                if output_v:
                    vx.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    vy.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    vz.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                else:
                    f.skip(3)

                if output_pres:
                    pres.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                           'F'))
                else:
                    f.skip()

                if output_pot:
                    pot.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                else:
                    f.skip()

                if output_opot:
                    opot.append(
                        np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                else:
                    f.skip()

                if output_temp:
                    temp.append(
                        np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                else:
                    f.skip()

                if output_metalicity:
                    metalicity.append(
                        np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                else:
                    f.skip()

                if output_cr0amr:
                    cr0amr.append(np.reshape(f.read_vector('i'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                             'F').astype('bool'))
                else:
                    f.skip()

                if output_solapst:
                    solapst.append(np.reshape(f.read_vector('i'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]),
                                              'F').astype('bool'))
                else:
                    f.skip()

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
    nmax, nmay, nmaz, nlevels = parameters.read_parameters(load_nma=True, load_npalev=False, load_nlevels=True,
                                                           load_namr=False, load_size=False, path=parameters_path)
    npatch, npart, patchnx, patchny, patchnz = read_grids(it, path=path, read_general=False, read_patchnum=True,
                                                          read_dmpartnum=True, read_patchcellextension=True,
                                                          read_patchcellposition=False, read_patchposition=False,
                                                          read_patchparent=False, parameters_path=parameters_path)

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
            dmpart_mass = mdmpart * np.ones(npart[0])

        # refinement levels
        for l in range(1, min(nlevels + 1, max_refined_level + 1)):
            if verbose:
                print('Reading level {}.'.format(l))
                print('{} patches. {} particles.'.format(npatch[l], npart[l]))
            for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
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


def read_clst(it, path='', parameters_path='', digits=5, max_refined_level=1000, output_deltastar=True, verbose=False,
              output_position=False, output_velocity=False, output_mass=False, output_temp=False,
              output_metalicity=False, output_id=False):
    """
    Reads the stellar (clst) file.
    For now, it only reads the delta.

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
        output_temp: whether particles' temperatures are returned (bool)
        output_metalicity: whether particles' metalicities are returned (bool)
        output_id: whether particles' ids are returned (bool)
        verbose: whether a message is printed when each refinement level is started (bool)

    Returns:
        Chosen quantities, in the order specified by the order of the parameters in this definition.
        delta_dm is returned as a list of numpy matrices. The 0-th element corresponds to l=0. The i-th element
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

        if output_temp:
            stpart_temp = f.read_vector('f')
        else:
            f.skip()

        if output_metalicity:
            stpart_metalicity = f.read_vector('f')
        else:
            f.skip()

        if output_id:
            stpart_id = np.zeros(stpart_x.size)

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

            if output_temp:
                stpart_temp = np.append(stpart_temp, f.read_vector('f'))
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

    returnvariables = []

    if output_deltastar:
        returnvariables.append(delta_star)
    if output_position:
        returnvariables.extend([stpart_x, stpart_y, stpart_z])
    if output_velocity:
        returnvariables.extend([stpart_vx, stpart_vy, stpart_vz])
    if output_mass:
        returnvariables.append(stpart_mass)
    if output_temp:
        returnvariables.append(stpart_temp)
    if output_metalicity:
        returnvariables.append(stpart_metalicity)
    if output_id:
        returnvariables.append(stpart_id)

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
                are_velocities=True, verbose=False):
    """
    Reads the vortex (Helmholtz-Hodge decomposition) files

    Args:
        it: iteration number (int)
        path: path of the grids file in the system (str)
        parameters_path: path of the json parameters file of the simulation
        digits: number of digits the filename is written with (int)
        max_refined_level: maximum refinement level that wants to be read. Subsequent refinements will be skipped. (int)
        are_divrot: whehther velocity divergences and rotationals are written in the file
        are_potentials: whether (scalar and vector) potentials are written in the file
        are_velocities: whether (total, compressional and rotational) velocities are written in the file

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
    with FF(os.path.join(path, filename(it, 'v', digits))) as f:
        # read header
        it_clus = f.read_vector('i')[0]
        # assert(it == it_clus)
        f.seek(0)  # this is a little bit ugly but whatever
        time, z = tuple(f.read_vector('f')[1:3])

        returnvariables = []

        if are_divrot:
            # divergence
            if verbose:
                print('Reading divergence...')
            div = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
            for l in range(1, nlevels + 1):
                for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                    div.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch], patchnz[ipatch]), 'F'))
                    # for some arbitrary reason, we have not written these files in fortran order...

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

            if are_potentials:
                # scalar
                if verbose:
                    print('Reading scalar potential...')
                scalarpot = [np.reshape(f.read_vector('f'), (nmax, nmay, nmaz), 'F')]
                for l in range(1, nlevels + 1):
                    for ipatch in range(npatch[0:l].sum() + 1, npatch[0:l + 1].sum() + 1):
                        scalarpot.append(np.reshape(f.read_vector('f'), (patchnx[ipatch], patchny[ipatch],
                                                                         patchnz[ipatch]), 'F'))

                # vector
                if verbose:
                    print('Reading vector potential...')
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

                    returnvariables.extend([vx, vy, vz, velcompx, velcompy, velcompz, velrotx, velroty, velrotz])

    return tuple(returnvariables)
