"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

read_halma module
Provides a function to read the HALMA stellar halo catologue and a function to read
the particle data of each halo in each iteration

Created by Óscar Monllor and David Vallés
"""

#  Last update on 03/11/22 10:14

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE
import numpy as np
from cython_fortran_file import FortranFile as FF



# FUNCTIONS DEFINED IN THIS MODULE

def read_stellar_catalogue(it, path='', name='halma_halo_stars_rp.res', legacy = False, 
                        output_format = 'dictionaries', output_redshift = False, min_mass = None):
                        
    """
    Reads halma_halo_stars_rp.res containing the stellar halo catalogue.

    Args:
        path: path of "halma_halo_stars_rp.res"
        name: name of the catalogue file, default is "halma_halo_stars_rp.res"
        legacy: if legacy (bool), returns return1, else returns return2

        (APLLIED ONLY IF legacy = False)
        it: MASCLET iteration
        output_format: 'dictionaries' or 'arrays
        output_redshift: bool, if True, redshift is also returned
        min_mass: minimum mass for a halo to be in the returned array or dictionary. If "None"
                  it is skipped.

    Return1: 
             total_halo_data[iteration][halo][dictionary]: list (for all iterations) of lists (for all halos) of dictionaries with halo data
             total_iteration_data[iteration][dictionary]: list (for all iterations) of dictionaries with the iteration data

    Return2: 
             list of dictionaries (one per halo) or dictionary of arrays, for iteration 'it'.
    """
    halma_catalogue = open(path+name, 'r')

    # Total number of iterations
    num_iter = int(halma_catalogue.readline())

    # Arrays with catalogue info
    total_iteration_data = []
    total_halo_data = []
    for it_halma in range(num_iter):
        halma_catalogue.readline()
        iteration_data = {}
        data_line = np.array(halma_catalogue.readline().split()).astype(np.float64())
        iteration_data['nhal'] = int(data_line[0])
        iteration_data['nparhal'] = int(data_line[1])
        iteration_data['it_halma'] = int(data_line[2])
        iteration_data['it_masclet'] = int(data_line[3])
        iteration_data['t'] = data_line[4]
        iteration_data['z'] = data_line[5]
        halma_catalogue.readline()
        halma_catalogue.readline()
        halma_catalogue.readline()
        halma_catalogue.readline()

        num_halos = iteration_data['nhal']
        haloes=[]
        for ih in range(num_halos):
            halo = {}
            data_line = np.array(halma_catalogue.readline().split()).astype(np.float64())
            halo['id'] = int(data_line[0])
            halo['partNum'] = int(data_line[1])
            halo['M'] = data_line[2]
            halo['Mv'] = data_line[3]
            halo['Mgas'] = data_line[4]
            halo['fcold'] = data_line[5]
            halo['Mhotgas'] = data_line[6]
            halo['Mcoldgas'] = data_line[7]
            halo['Msfr'] = data_line[8]
            halo['Rmax'] = data_line[9]
            halo['R'] = data_line[10]
            halo['R_1d'] = data_line[11]
            halo['R_1dx'] = data_line[12]
            halo['R_1dy'] = data_line[13]
            halo['R_1dz'] = data_line[14]
            halo['sigma_v'] = data_line[15]
            halo['sigma_v_1d'] = data_line[16]
            halo['sigma_v_1dx'] = data_line[17]
            halo['sigma_v_1dy'] = data_line[18]
            halo['sigma_v_1dz'] = data_line[19]
            halo['L'] = data_line[20]
            halo['xcm'] = data_line[21]
            halo['ycm'] = data_line[22]
            halo['zcm'] = data_line[23]
            halo['vx'] = data_line[24]
            halo['vy'] = data_line[25]
            halo['vz'] = data_line[26]
            halo['father1'] = int(data_line[27])
            halo['father2'] = int(data_line[28])
            halo['nmerg'] = int(data_line[29])
            halo['mergType'] = int(data_line[30])
            halo['age_m'] = data_line[31]
            halo['age'] = data_line[32]
            halo['Z_m'] = data_line[33]
            halo['Z'] = data_line[34]
            #CLASSIC HALMA catalogue is 35 column long
            #DATA OF NEW HALO FINDER
            if len(data_line>35):
                halo['Vsigma'] = data_line[35]
                halo['lambda'] = data_line[36]
                halo['kin_morph'] = data_line[37]
                halo['v_TF'] = data_line[38]
                halo['a'] = data_line[39]
                halo['b'] = data_line[40]
                halo['c'] = data_line[41]
                halo['sersic'] = data_line[42]
                #CALIPSO
                halo['lum_u'] = data_line[43]
                halo['lum_g'] = data_line[44]
                halo['lum_r'] = data_line[45]
                halo['lum_i'] = data_line[46]
                halo['sb_u'] = data_line[47]
                halo['sb_g'] = data_line[48]
                halo['sb_r'] = data_line[49]
                halo['sb_i'] = data_line[50]
                halo['ur_color'] = data_line[51]
                halo['gr_color'] = data_line[52]
                halo['sersic_lum'] = data_line[53]
            haloes.append(halo)
        
        total_iteration_data.append(iteration_data)
        total_halo_data.append(haloes)

    halma_catalogue.close()

    ####################################
    ####################################

    if legacy:
        return total_iteration_data, total_halo_data

    else:
        #finding HALMA iteration index
        for it_halma in range(num_iter):
            if total_iteration_data[it_halma]['it_masclet'] == it:
                break

        
        haloes = total_halo_data[it_halma]
        zeta = total_iteration_data[it_halma]['z']

        if min_mass is not None:
            haloes=[halo for halo in haloes if halo['M']>min_mass]

        if output_format=='dictionaries':
            if output_redshift:
                return haloes, zeta
            else:
                return haloes
        elif output_format=='arrays':
            if output_redshift:
                return {k: np.array([h[k] for h in haloes]) for k in haloes[0].keys()}, zeta
            else:
                return {k: np.array([h[k] for h in haloes]) for k in haloes[0].keys()}



def read_halo_particles(it, haloes=None, old = False, path = '', name='halma_halo_stars_rp.res', 
                        path_binary = None, output_dictionary=True):
    """
    Reads the halma binary catalogue containing the information of every halo particle.

    Args:
        it: MASCLET ITERATION
        halos: haloes which HALMA haloes are to be analyzed (list o array).
            If None (default), this outputs the particles of all the haloes.
        path: path of HALMA catalogue 
        name: name of HALMA catalogue
        path_binary: path of HALMA stellar binary catalogue (halotree).
            If None (default), it is assumed to be the same as path (above).
        output_dictionary: if True, outputs are a dictionary whose keys are halo
            ID. If False, outputs are a list in the same order as haloes/catalogue.

    Returns: list (lenght of haloes) of ndarrays containing the particle information -->
             --> example: output[halo_index]['x'][particle_index]
    """

    if path_binary is None:
        path_binary=path

    haloes_dict = read_stellar_catalogue(it, path=path, name=name, old = old, legacy = False)
    
    if haloes is None:
        haloes = np.arange(len(haloes_dict))
    else:
        if type(haloes) is not list:
            haloes = [haloes]
        haloes = np.array(haloes) - 1 #correction of indices (0 to n-1) and numpy array
        
    num_haloes = len(haloes)

    string_it = '{:05d}'.format(it)

    output = [[] for halo in range(num_haloes)]

    f_float = FF(path_binary+'halotree'+string_it)
    f_int = FF(path_binary+'halotree'+string_it)
    f_int.skip(1) # read header
    f_float.skip(1) 
    low_old = 0
    halo_old = 0
    npart_old = 0
    ih = 0 #index in haloes
    for halo in haloes:

        npart = int(haloes_dict[halo]['partNum'])
        particles = np.zeros(npart, dtype={'names':('x', 'y', 'z', 'vx', 'vy', 'vz', 'mass', 'age', 'met', 'donde', 'id'),
                          'formats':('f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'f8', 'i4', 'i4')})

        low = 0
        for h in range(halo):
            low += int(haloes_dict[h]['partNum'])

        jump_particle = low - (low_old+npart_old) #number of particles lines to jump
        jump_merger = halo - halo_old #number of merger lines to jump

        f_int.skip(jump_particle) #nos saltamos las particulas que no pertenecen a este halo
        f_float.skip(jump_particle)
        f_int.skip(jump_merger) #saltamos la linea correspondiente a MERGER para los halos anteriores
        f_float.skip(jump_merger) #saltamos la linea correspondiente a MERGER para los halos anteriores
        for p in range(npart):        
                line_int = f_int.read_vector('i4')
                line_float = f_float.read_vector('f')
                particles['x'][p] = line_float[0]
                particles['y'][p] = line_float[1]
                particles['z'][p] = line_float[2]
                particles['vx'][p] = line_float[3]
                particles['vy'][p] = line_float[4]
                particles['vz'][p] = line_float[5]
                particles['mass'][p] = line_float[6]
                particles['age'][p] = line_float[7]
                particles['met'][p] = line_float[8]
                particles['donde'][p] = line_int[9]
                particles['id'][p] = line_int[10]

        output[ih] = particles

        low_old = low
        halo_old = halo
        npart_old = npart
        ih += 1
    
    f_float.close()
    f_int.close()

    if output_dictionary:
        output2={}
        for ih in range(len(haloes)):
            haloid = haloes_dict[ih]['id']
            output2[haloid] = output[ih]
        return output2
    else:
        return output



def read_particles_npy(it, old = False, path = '', name='halma_halo_stars_rp.res', path_binary = None):
    """
    Reads the halma/masclet_pyfof npy binary catalogue containing which particles are in haloes

    Args:
        it: MASCLET ITERATION
        path: path of HALMA/pyfof catalogue 
        name: name of HALMA/pyfof catalogue
        path_binary: path of HALMA stellar binary catalogue (halotree).
            If None (default), it is assumed to be the same as path (above).

    Returns: list (lenght number of haloes) of 1d numpy arrays (of lenght number of particles of each halo) 
            containing particle indices for fast read_masclet numpy arrays indexing

    """

    #catalogue data
    haloes_dict = read_stellar_catalogue(it, path=path, name=name, old = old, legacy = False)
    nhal = len(haloes_dict)
    
    #particle indices of each halo
    string_it = f'{it:05d}'
    array_groups = np.load(path_binary+'/halotree'+string_it+'.npy')

    ##number of particles per halo
    npart = np.zeros(nhal, dtype=np.int32)
    for ih in range(nhal):
        npart[ih] = haloes_dict[ih]['partNum']

    npart_sum = np.zeros(nhal, dtype=np.int32)
    ## acumulate npart:
    for ih in range(nhal):  
        npart_sum[ih] = np.sum(npart[:ih+1])

    #split the all_particles_in_haloes array into nhal arrays, one for each halo
    groups = np.split(array_groups, npart_sum)

    return groups