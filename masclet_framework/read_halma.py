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

def read_stellar_catalogue(path='', name='', old = False):
    """
    Reads halma_halo_stars_rp.res containing the stellar halo catalogue.

    Args:
        path: path of "halma_halo_stars_rp.res"
        name: name of the catalogue file, typically "halma_halo_stars_rp.res"
        old: If old (bool), it assumes halma in 2017 version, which requires old_catalog_with_SFR_merger.npy file, 
            created with SFR_mergertype_old_catalog.py. Default is False, assuming halma27 (2022 version)

    Returns: num_iter: number of iterations (int)
             total_halo_data[iteration][halo, data]: list of 2D-arrays  containing the halo data
             total_iteration_data[iteration, data]: 2D-array containing the iteration data
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

    return num_iter, total_iteration_data, total_halo_data

def read_halo_particles(it, path, total_halo_data, total_iteration_data, halo):
    """
    Reads the halma binary catalogue containing the information of every halo particle.

    Args:
        it: iteration (iterations go from 0 to halma_final_iteration-1)
        path: path of the binary catalogue (halotree)
        total_halo_data: list of 2D-arrays  containing the halo data
        total_iteration_Data: 2D-array containing the iteration data
        halo: which halo is the one to be analyzed (If we want info about halo 3 in HALMA, 
                                                    halo = 2 should be given, that is, from 0 
                                                    to last_halo-1)

    Returns: list of arrays of lenght the number of particles of the halo, containing the particle 
             information
    """


    it_masclet = int(total_iteration_data[it, 3])
    string_it = str(it_masclet)
    while len(string_it) < 5:
        string_it = '0'+string_it

    npart = int(total_halo_data[it][halo, 1])
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
        low += int(total_halo_data[it][h, 1])

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



