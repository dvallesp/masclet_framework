"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

read_masclet module
Provides a function to read the HALMA stellar halo catologue and a function to read
the particle data of each halo in each iteration

Created by David Vallés
"""

#  Last update on 03/11/22 10:14

# GENERAL PURPOSE AND SPECIFIC LIBRARIES USED IN THIS MODULE
import os, sys, numpy as np
from cython_fortran_file import FortranFile as FF



# FUNCTIONS DEFINED IN THIS MODULE

def read_stellar_catalogue(path, name, old, legacy, it, 
                    output_format, output_redshift, min_mass):
    """
    Reads halma_halo_stars_rp.res containing the stellar halo catalogue.

    Args:
        path: path of "halma_halo_stars_rp.res"
        name: name of the catalogue file, typically "halma_halo_stars_rp.res"
        old: If old (bool), it assumes halma in 2017 version, which requires old_catalog_with_SFR_merger.npy file, 
            created with SFR_mergertype_old_catalog.py. Default is False, assuming halma27 (2022 version)
        legacy: if legacy (bool), returns return1, else returns return2

        (APLLIED ONLY IF legacy = False)
        it: MASCLET iteration
        output_format: 'dictionaries' or 'arrays
        output_redshift: bool, if True, redshift is also returned
        min_mass: minimum mass for a halo to be in the returned array or dictionary. If "None"
                  it is skipped.

    Return1: 
             total_halo_data[iteration][halo, data]: list of 2D-arrays containing the halo data for every iteration
             total_iteration_data[iteration, data]: 2D-array containing the iteration data

    Return2: 
             list of dictionaries (one per halo) or dictionary of arrays, for iteration 'it'.
    """
    halma_catalogue = open(path+name, 'r')

    # Total number of iterations
    num_iter = int(halma_catalogue.readline())

    # Arrays with catalogue info
    total_iteration_data = np.empty((num_iter, 6))
    total_halo_data = [0]*num_iter 


    if not old:
        # HALMA 2022
        for i in range(num_iter):
            halma_catalogue.readline()
            iteration_data = np.array(halma_catalogue.readline().split()).astype(np.float64())
            num_halos = int(iteration_data[0])
            halma_catalogue.readline()
            halma_catalogue.readline()
            halma_catalogue.readline()
            halma_catalogue.readline()
            halo_data = np.empty((num_halos, 35))
            for j in range(num_halos):
                data_line = np.array(halma_catalogue.readline().split()).astype(np.float64())
                halo_data[j] = data_line 
            
            total_iteration_data[i,:] = iteration_data[:]
            total_halo_data[i] = halo_data

    else:
        #HALMA 2017
        for i in range(num_iter):
            halma_catalogue.readline()
            iteration_data = np.array(halma_catalogue.readline().split()).astype(np.float64())
            num_halos = int(iteration_data[0])
            halma_catalogue.readline()
            halma_catalogue.readline()
            halma_catalogue.readline()
            for j in range(num_halos):
                data_line = np.array(halma_catalogue.readline().split()).astype(np.float64())

            total_iteration_data[i,:] = iteration_data[:]

        total_halo_data = np.load(path+'old_catalog_with_SFR_merger.npy', allow_pickle=True)

    halma_catalogue.close()

    ####################################
    ####################################

    if legacy:
        return total_iteration_data, total_halo_data

    else:
        #finding it HALMA index
        for it_halma in range(num_iter):
            if total_iteration_data[it_halma, 3] == it:
                break

        zeta = total_iteration_data[it_halma, 5]
        num_halos = int(total_iteration_data[it_halma, 0])
        it_halo_data = total_halo_data[it_halma] 
        haloes=[]
        for ih in range(num_halos):
            halo = {}
            halo['id'] = it_halo_data[ih, 0]
            halo['partNum'] = it_halo_data[ih, 1]
            halo['M'] = it_halo_data[ih, 2]
            halo['Mv'] = it_halo_data[ih, 3]
            halo['Mgas'] = it_halo_data[ih, 4]
            halo['fcold'] = it_halo_data[ih, 5]
            halo['Mhotgas'] = it_halo_data[ih, 6]
            halo['Mcoldgas'] = it_halo_data[ih, 7]
            halo['Msfr'] = it_halo_data[ih, 8]
            halo['Rmax'] = it_halo_data[ih, 9]
            halo['R'] = it_halo_data[ih, 10]
            halo['R_1d'] = it_halo_data[ih, 11]
            halo['R_1dx'] = it_halo_data[ih, 12]
            halo['R_1dy'] = it_halo_data[ih, 13]
            halo['R_1dz'] = it_halo_data[ih, 14]
            halo['sigma_v'] = it_halo_data[ih, 15]
            halo['sigma_v_1d'] = it_halo_data[ih, 16]
            halo['sigma_v_1dx'] = it_halo_data[ih, 17]
            halo['sigma_v_1dy'] = it_halo_data[ih, 18]
            halo['sigma_v_1dz'] = it_halo_data[ih, 19]
            halo['L'] = it_halo_data[ih, 20]
            halo['xcm'] = it_halo_data[ih, 21]
            halo['ycm'] = it_halo_data[ih, 22]
            halo['zcm'] = it_halo_data[ih, 23]
            halo['vx'] = it_halo_data[ih, 24]
            halo['vy'] = it_halo_data[ih, 25]
            halo['vz'] = it_halo_data[ih, 26]
            halo['father1'] = it_halo_data[ih, 27]
            halo['father2'] = it_halo_data[ih, 28]
            halo['nmerg'] = it_halo_data[ih, 29]
            halo['mergType'] = it_halo_data[ih, 30]
            halo['age_m'] = it_halo_data[ih, 31]
            halo['age'] = it_halo_data[ih, 32]
            halo['Z_m'] = it_halo_data[ih, 33]
            halo['Z'] = it_halo_data[ih, 34]
            haloes.append(halo)

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



def read_halo_particles(it, path, total_halo_data, total_iteration_data, halo):
    """
    Reads the halma binary catalogue containing the information of every halo particle.

    Args:
        it: MASCLET ITERATION
        path: path of the binary catalogue (halotree)
        total_halo_data: list of 2D-arrays  containing the halo data
        total_iteration_Data: 2D-array containing the iteration data
        halo: which halo is the one to be analyzed (If we want info about halo 3 in HALMA, 
                                                    halo = 2 should be given, that is, from 0 
                                                    to last_halo-1)

    Returns: list of arrays of lenght the number of particles of the halo, containing the particle 
             information
    """
    #finding it HALMA index
    num_iter = len(total_halo_data)
    for it_halma in range(num_iter):
        if total_iteration_data[it_halma, 3] == it:
            break

    string_it = '{:05d}'.format(it)

    npart = int(total_halo_data[it_halma][halo, 1])
    stpart_x = np.zeros((npart))
    stpart_y = np.zeros((npart))
    stpart_z = np.zeros((npart))
    stpart_vx = np.zeros((npart))
    stpart_vy = np.zeros((npart))
    stpart_vz = np.zeros((npart))
    stpart_mass = np.zeros((npart))
    stpart_age = np.zeros((npart))
    stpart_met = np.zeros((npart))
    stpart_donde = np.zeros((npart))
    stpart_id = np.zeros((npart))

    low = 0
    for h in range(0, halo):
        low += int(total_halo_data[it_halma][h, 1])

    f_float = FF(path+'halotree'+string_it)
    f_int = FF(path+'halotree'+string_it)
       
    f_int.skip(1) # read header
    f_float.skip(1) 
    f_int.skip(low) #nos saltamos las particulas que no pertenecen a este halo
    f_float.skip(low)
    f_int.skip(halo) #saltamos la linea correspondiente a MERGER para los halos anteriores
    f_float.skip(halo) #saltamos la linea correspondiente a MERGER para los halos anteriores
    for p in range(npart):        
            line_int = f_int.read_vector('i4')
            line_float = f_float.read_vector('f')
            stpart_x[p] = line_float[0]
            stpart_y[p] = line_float[1]
            stpart_z[p] = line_float[2]
            stpart_vx[p] = line_float[3]
            stpart_vy[p] = line_float[4]
            stpart_vz[p] = line_float[5]
            stpart_mass[p] = line_float[6]
            stpart_age[p] = line_float[7]
            stpart_met[p] = line_float[8]
            stpart_donde[p] = line_int[9]
            stpart_id[p] = line_int[10]

    return [stpart_x, stpart_y, stpart_z, stpart_vx, stpart_vy, stpart_vz, 
            stpart_mass, stpart_age, stpart_met, stpart_donde, stpart_id]



