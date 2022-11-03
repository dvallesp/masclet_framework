import os, sys, numpy as np
from cython_fortran_file import FortranFile as FF


"""""""""""""""""""""""""""""""""""""""""""""
"""" TOTAL ITERATION DATA ORDERING """""
"""""""""""""""""""""""""""""""""""""""""""""
""" 
N HALOS,      N PARTICULAS,     ITERACIÓN,     ITERACIÓN DE MASCLET,      TIEMPO,     REDSHIFT
"""""""""""""""""""""""""""""""""""""""""""""

"""""""""""""""""""""""""""""""""""""""""""""
"""" TOTAL HALO DATA ORDERING """""
"""""""""""""""""""""""""""""""""""""""""""""
"""""""""""""""""""""""""""""""""""""""""""""
*---------------------------------------------------------------------------
*     FICHERO 21 , formato:
*     col 1: halo
*     col 2: numero de particulas 
*     col 3: masa estelar del halo
*     col 4: masa estelar visible del halo
*     col 5: masa gas del halo
*     col 6: fraccion de la masa de gas de halo en gas frio < 5.e4 K
*     col 7: masa de gas caliente  NO ligado en el halo
*     col 8: masa de gas frio NO ligado en el halo
*     col 9: masa de estrellas formadas desde el output previo
*     col 10: radio(kpc) estimado como la particula ligada mas lejana
*     col 11: radio(kpc) de mitad masa
*     col 12: radio(kpc) de mitad masa visible proyectado 1D promedio
*     col 13,14,15: radio(kpc) de mitad masa visible proyectado en los tres ejes
*     col 16: dispersion vel.(km/s) dentro de el radio de mitad masa 
*     col 17: dispersion vel.(km/s) proyectada dentro de el radio de mitad masa visible 1D promedio
*     col 18,19,20: dispersion vel.(km/s) proyectada dentro de el radio de mitad masa visible 1D en cada eje
*     col 21: momento angular especifico en unidades de kpc km/s
*     col 22, 23, 24: coor. centro de massas (x,y,z)
*     col 25, 26, 27: velocidad. centro de massas (x,y,z) en km/s
*     col 28,29: halo del instante anterior que fue el principal progenitor,segundo halo
*     col 30: numero de mergers, sufridos
*     col 31: tipo de merger *     TIPOMER = 0 no merger
                             *             = 1 major merger 1 < m1/m2 < 3
                             *             = 2 minor merger 3 < m1/m2 < 10
                             *             = 3 super minor merger "acreation" 10 < m1/m2 < inf
                             *             = -1 the halo breaks apart
*     col 32: edad media pesada en masa
*     col 33: edad media
*     col 34: metalicidad media pesada en masa
*     col 35: metalicidad media
*--------------------------------------------------------------------------
"""""""""""""""""""""""""""""""""""""""""""""



def read_stellar_catalogue(path='', name='', old = False):
    '''
    Reads halma_halo_stars_rp.res containing the stellar halo catalogue.
    It returns number of iterations (num_iter), 
    total_halo_data[iteration][halo, data] (list of arrays 2d),
    and total_iteration_data[iteration, data] (array 2d)
    If old, it assumes halma in 2017 version, which requires old_catalog_with_SFR_merger.npy file, 
    created with SFR_mergertype_old_catalog.py
    '''
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

        total_halo_data = np.load('old_catalog_with_SFR_merger.npy', allow_pickle=True)

    halma_catalogue.close()

    return num_iter, total_iteration_data, total_halo_data

def read_halo_particles(it, path, total_halo_data, total_iteration_data, halo):
    """
    Reads the halma binary catalogue. Information of every halo particle, give HALMA catalogue.
    Be careful! If we want info about halo 3 in HALMA, 
    halo = 2 should be given (from 0 to n-1 in Python, instead of 1 to n)
    The same aplies to "it", iterations go from 0 to halma_final_iteration-1.
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

    return (stpart_x, stpart_y, stpart_z, stpart_vx, stpart_vy, stpart_vz, 
            stpart_mass, stpart_age, stpart_met, stpart_donde, stpart_id)



