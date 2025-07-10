"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

read_halma module
Provides a function to read the pyHALMA stellar halo catologue and a function to read
the particle data of each halo in each iteration

Created by Óscar Monllor and David Vallés
"""

#  Last update on 15/02/2024

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE
import numpy as np
import os

# FUNCTIONS DEFINED IN THIS MODULE
def read_stellar_catalogue(it, path='', output_format = 'dictionaries', output_redshift = False, 
                           min_mass = None, ret_it_data = False, box = None):
    """
    Reads the stellar_haloes files, containing the stellar haloes catalogue of the pyHALMA halo finder.
    
    Args:
        - it: iteration number (int)
        - path: path of the stellar_haloes (str)
        - output_format: 'dictionaries' or 'arrays'
        - output_redshift: whether to output the redshift of the snapshot, 
                           after the halo catalogue (bool)
        - min_mass: minimum mass of the haloes to be output (float)
        - ret_it_data: whether to return the iteration data (bool)
        - box: if None, all haloes considered, if (x_min, x_max, y_min, y_max, z_min, z_max) 
                only haloes within the box are considered (in Mpc)

    Returns:
        - if output_format='dictionaries': list of dictionaries, one per halo
        - if output_format='arrays': dictionary of arrays, one array per halo property
    """
    
    #Check output format
    if output_format not in ['dictionaries', 'arrays']:
        raise ValueError('output_format must be "dictionaries" or "arrays"')

    with open(os.path.join(path, 'stellar_haloes{:05d}'.format(it)), 'r') as halma_catalogue:

        # Iteration data
        halma_catalogue.readline()
        it_halma, it_masclet, zeta, cosmo_time, nhal, nparhal = halma_catalogue.readline().split()
        it_halma = int(it_halma)
        it_masclet = int(it_masclet)
        zeta = float(zeta)
        cosmo_time = float(cosmo_time)
        nhal = int(nhal)
        nparhal = int(nparhal)
        halma_catalogue.readline()

        it_data = {'it_halma': it_halma, 'it_masclet': it_masclet, 
                   'zeta': zeta, 'cosmo_time': cosmo_time, 'nhal': nhal,
                   'nparhal': nparhal}

        # Halo data
        halma_catalogue.readline()
        halma_catalogue.readline()
        halma_catalogue.readline()
        halma_catalogue.readline()
        
        haloes=[]
        for ih in range(nhal):
            halo = {}
            data_line = np.array(halma_catalogue.readline().split()).astype('f4')
            halo['id'] = int(data_line[0])
            halo['partNum'] = int(data_line[1])
            halo['M'] = data_line[2]
            halo['xcm'] = data_line[3]
            halo['ycm'] = data_line[4]
            halo['zcm'] = data_line[5]
            halo['xpeak'] = data_line[6]
            halo['ypeak'] = data_line[7]
            halo['zpeak'] = data_line[8]
            halo['xbound'] = data_line[9]
            halo['ybound'] = data_line[10]
            halo['zbound'] = data_line[11]
            halo['id_bound'] = int(data_line[12])
            halo['Mgas'] = data_line[13]
            halo['fcold'] = data_line[14]
            halo['Mhotgas'] = data_line[15]
            halo['Mcoldgas'] = data_line[16]
            halo['Msfr'] = data_line[17]
            halo['Min'] = data_line[18]
            halo['Rmax'] = data_line[19]
            halo['R'] = data_line[20]
            halo['R_1d'] = data_line[21]
            halo['R_1dx'] = data_line[22]
            halo['R_1dy'] = data_line[23]
            halo['R_1dz'] = data_line[24]
            halo['sigma_v'] = data_line[25]
            halo['sigma_v_1d'] = data_line[26]
            halo['sigma_v_1dx'] = data_line[27]
            halo['sigma_v_1dy'] = data_line[28]
            halo['sigma_v_1dz'] = data_line[29]
            halo['L'] = data_line[30]
            halo['Lx'] = data_line[31]
            halo['Ly'] = data_line[32]
            halo['Lz'] = data_line[33]
            halo['vx'] = data_line[34]
            halo['vy'] = data_line[35]
            halo['vz'] = data_line[36]
            halo['father1'] = int(data_line[37])
            halo['father2'] = int(data_line[38])
            halo['nmerg'] = int(data_line[39])
            halo['mergType'] = int(data_line[40])
            halo['age_m'] = data_line[41]
            halo['age'] = data_line[42]
            halo['Z_m'] = data_line[43]
            halo['Z'] = data_line[44]
            halo['Vsigma'] = data_line[45]
            halo['lambda'] = data_line[46]
            halo['kin_morph'] = data_line[47]
            halo['v_TF'] = data_line[48]
            halo['a'] = data_line[49]
            halo['b'] = data_line[50]
            halo['c'] = data_line[51]
            halo['sersic'] = data_line[52]
            halo['lum_u'] = data_line[53]
            halo['lum_g'] = data_line[54]
            halo['lum_r'] = data_line[55]
            halo['lum_i'] = data_line[56]
            halo['sb_u'] = data_line[57]
            halo['sb_g'] = data_line[58]
            halo['sb_r'] = data_line[59]
            halo['sb_i'] = data_line[60]
            halo['ur_color'] = data_line[61]
            halo['gr_color'] = data_line[62]
            halo['bh_mass'] = data_line[63]
            halo['asohf_ID'] = int(data_line[64])
            halo['asohf_mass'] = data_line[65]
            halo['asohf_Rvir'] = data_line[66]
            halo['darkmatter_mass'] = data_line[67]
            haloes.append(halo)

    ####################################
    ####################################
    output = []
    
    if min_mass is not None:
        haloes=[halo for halo in haloes if halo['M']>min_mass]

    if box is not None:
        x_min, x_max, y_min, y_max, z_min, z_max = box
        haloes = [halo for halo in haloes if (x_min <= halo['xcm']*1e-3 <= x_max and
                                              y_min <= halo['ycm']*1e-3 <= y_max and
                                              z_min <= halo['zcm']*1e-3 <= z_max)]

    if output_format=='dictionaries':
        output.append(haloes)
    elif output_format=='arrays':
        output.append({k: np.array([h[k] for h in haloes]) for k in haloes[0].keys()})

    if output_redshift:
        output.append(zeta)

    if ret_it_data:
        output.append(it_data)

    output = tuple(output)

    if len(output)==1:
        output = output[0]

    return output

def read_particles_npy(it, path = '', min_mass = None):
    """
    Reads the halma/masclet_pyfof npy binary catalogue containing which particles are in haloes

    Args:
        it: MASCLET ITERATION
        path: path of HALMA catalogue 
        min_mass: minimum mass of the haloes to be output (float)

    Returns: list (lenght number of haloes) of 1d numpy arrays (of lenght number of particles of each halo) 
            containing particle indices for fast read_masclet numpy arrays indexing

    """

    #catalogue data
    haloes_dict = read_stellar_catalogue(it, path=path, output_format='dictionaries')
    nhal = len(haloes_dict)
    
    #particle indices of each halo
    array_groups = np.load(os.path.join(path, 'stellar_particles{:05d}.npy'.format(it)))

    #number of particles per halo
    npart = np.zeros(nhal, dtype=np.int32)
    for ih in range(nhal):
        npart[ih] = haloes_dict[ih]['partNum']

    npart_sum = np.zeros(nhal, dtype=np.int32)
    # acumulate npart:
    for ih in range(nhal):  
        npart_sum[ih] = np.sum(npart[:ih+1])

    #split the all_particles_in_haloes array into nhal arrays, one for each halo
    groups = np.split(array_groups, npart_sum)

    #Clean empty arrays
    groups = [group for group in groups if len(group)>0]

    #Filter by mass
    if min_mass is not None:
        groups = [group for group, halo in zip(groups, haloes_dict) if halo['M']>min_mass]
        
    return groups