"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

stars module
Provides functions to calculate basic cosmological relations related to stars, 
such as the stellar mass function and the cosmic star formation rate density.


Created by Óscar Monllor
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from masclet_framework import read_halma, read_masclet, units



# FUNCTIONS DEFINED IN THIS MODULE
def stellar_mass_function(it, path='', L = 40.):
    '''
    Plots the stellar mass function of the simulation at a given iteration, given
    the pyHALMA halo finder output.

    Args:
        it: MASCLET iteration
        path: path to PyHALMA output
        L: simulation box size in cMpc

    Returns:
        ax: matplotlib axis object

    Author: Óscar Monllor
    '''

    haloes, zeta = read_halma.read_stellar_catalogue(it = it, path=path, output_redshift=True)

    ################################
    # Fit from Ilbert et al A&A 2013  
    ################################
    def smf_fit(M, z):

        if -0.1<=z<0.5:   	
            M_star = 10**(10.88) 
            phi1 = 1.68 * 1e-3
            phi2 = 0.77 * 1e-3
            alpha1 = -0.69
            alpha2 = -1.42

        elif 0.5<=z<0.8:
            M_star = 10**(11.03)
            phi1 = 1.22 * 1e-3
            phi2 = 0.16 * 1e-3
            alpha1 = -1.0
            alpha2 = -1.64
            
    	
        elif 0.8<=z<1.1:
            M_star = 10**(10.87)
            phi1 = 2.03 * 1e-3
            phi2 = 0.29 * 1e-3
            alpha1 = -0.52
            alpha2 = -1.62
    	
        elif 1.1<=z<1.5:
            M_star = 10**(10.71)
            phi1 = 1.35 * 1e-3
            phi2 = 0.67 * 1e-3
            alpha1 = -0.08
            alpha2 = -1.46
    	
        elif 1.5<=z<2.0:
            M_star = 10**(10.74)
            phi1 = 0.33 * 1e-3
            phi2 = 0.77 * 1e-3
            alpha1 = -0.24
            alpha2 = -1.6
    	
        elif 2.0<=z<2.5:
            M_star = 10**(10.74)
            phi1 = 0.62 * 1e-3
            phi2 = 0.15 * 1e-3
            alpha1 = -0.22
            alpha2 = -1.6
    	
        elif 2.5<=z<3.0:
            M_star = 10**(10.76)
            phi1 = 0.26 * 1e-3
            phi2 = 0.14 * 1e-3
            alpha1 = -0.15
            alpha2 = -1.6
    	
        elif z >= 3.0:
            M_star = 10**(10.74)
            phi1 = 0.03 * 1e-3
            phi2 = 0.09 * 1e-3
            alpha1 = 0.95
            alpha2 = -1.6

        phi = np.log(10) * np.exp(-10**(np.log10(M) - np.log10(M_star))) *              \
    			         ( phi1*(10**(np.log10(M) - np.log10(M_star)))**(alpha1+1) +    \
    			           phi2*(10**(np.log10(M) - np.log10(M_star)))**(alpha2+1) )    
        return phi

    #Define Figure
    figsize = (5,4)
    fig, ax = plt.subplots(1,1, figsize=figsize, dpi = 300)
    ax.set_xlabel('$M_{\star}$ [$M_{\odot}$]', fontsize = 'large')
    ax.set_ylabel('$\phi$ [Mpc]$^{-3}$ [dex]$^{-1}$', fontsize = 'large')
    
    #Bins
    M_min = 7 
    M_max = 12

    Nbins = 20
    dlogM = (M_max - M_min)/Nbins
    Mbins_edges = np.linspace(M_min, M_max, Nbins+1)
    Mbins = (Mbins_edges[1:] + Mbins_edges[:-1])/2

    #Mass function
    phi_intervals = np.zeros((Nbins))
    for halo in haloes:
        index = np.argmin(np.abs(np.log10(halo['M']) - Mbins))
        phi_intervals[index] += 1

    #Poissonian errors (2 sigma)
    err_phi = 2*np.sqrt(phi_intervals)

    #Normalize by volume and bin width
    phi_intervals /= (L**3 * dlogM)
    err_phi /= (L**3 * dlogM)

    ###### Plot
    #Limits
    ax.set_ylim((1e-6, 1))
    ax.set_xlim((10**M_min, 10**M_max))
    
    #ticks modifier
    ax.minorticks_on()
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    ax.tick_params(which='both', direction='in', labelsize = 'large')
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)

    #Plot the mass function
    ax.plot(10**Mbins[phi_intervals > 0], phi_intervals[phi_intervals > 0], color = 'blue', alpha = 0.8, label = f'z = {zeta:.1f}')

    #Plot the fit
    M_fit = np.logspace(M_min, M_max,  num = 100)
    phi = smf_fit(M_fit, zeta)
    ax.plot(M_fit, phi, color = 'black', label = 'Ilbert et al A&A 2013')

    #Poissonian errors, also for the fit
    err_fit = 2*np.sqrt(phi*L**3 * dlogM) / (L**3 * dlogM)
    ax.fill_between(M_fit, phi - err_fit, phi + err_fit, color = 'black', alpha = 0.15)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend()
    fig.tight_layout()

    return ax




def cosmic_SFR_density(init_it, final_it, step, path='', L = 40.):
    '''
    Plots the cosmic star formation rate density of the simulation at the specified iterations
    Args:
        init_it: initial MASCLET iteration
        final_it: final MASCLET iteration
        step: step between iterations
        path: path to MASCLET output
        L: simulation box size in cMpc

    Returns:
        ax: matplotlib axis object
    
    Author: Óscar Monllor
    '''

    SFRdens = np.zeros(int(final_it/step +1))
    z_array = np.zeros(int(final_it/step +1))

    for i, it in enumerate(range(init_it, final_it, step)):
        masclet_st_data = read_masclet.read_clst(it, path = path, parameters_path = path, 
                                                        digits=5, max_refined_level=1000, 
                                                        output_deltastar=False, verbose=False, output_position=False, 
                                                        output_velocity=False, output_mass=True, output_time=True,
                                                        output_metalicity=True, output_id=False, are_BH = True,
                                                        output_BH=False)

        grid_data_it = read_masclet.read_grids(it, path = path, parameters_path = path, 
                                               digits=5, read_general=True)
        cosmo_time = grid_data_it[1]
        z_array[i] = grid_data_it[4]
        
        if i > 0:
            grid_data_it_bef = read_masclet.read_grids(it-step, path = path, parameters_path = path, 
                                                       digits=5, read_general=True)
            
            cosmo_time_bef = grid_data_it_bef[1]

        else:
            cosmo_time_bef = 0

        dt = abs(cosmo_time - cosmo_time_bef)


        st_mass = masclet_st_data[0]*units.mass_to_sun 
        st_age = masclet_st_data[1]
        time_condition = st_age > (cosmo_time-1.1*dt)

        #SFR density in M_sun/yr/cMpc^3
        SFRdens[i] = np.sum(st_mass[time_condition]) / (dt * L**3) / units.time_to_yr 


    #################################
    # Data from Madau & Dickinson 2014
    # and Oesch et al 2014
    #################################
        
    # z data
    z_madau = [0]*27
    z_oesch = [0]*2

    z_madau[0] = [0.01, 0.1]
    z_madau[1] = [0.2, 0.4]
    z_madau[2] = [0.4, 0.6]
    z_madau[3] = [0.6, 0.8]
    z_madau[4] = [0.8, 1.2]
    z_madau[5] = [0.05, 0.05]
    z_madau[6] = [0.05, 0.2]
    z_madau[7] = [0.2, 0.4]
    z_madau[8] = [0.4, 0.6]
    z_madau[9] = [0.6, 0.8]
    z_madau[10] = [0.8, 1.0]
    z_madau[11] = [1.0, 1.2]
    z_madau[12] = [1.2, 1.7]
    z_madau[13] = [1.7, 2.5]
    z_madau[14] = [2.5, 3.5]
    z_madau[15] = [3.5, 4.5]
    z_madau[16] = [0.92, 1.33]
    z_madau[17] = [1.62, 1.88]
    z_madau[18] = [2.08, 2.37]
    z_madau[19] = [1.9, 2.7]
    z_madau[20] = [2.7, 3.4]
    z_madau[21] = [3.8, 3.8]
    z_madau[22] = [4.9, 4.9]
    z_madau[23] = [5.9, 5.9]
    z_madau[24] = [7.0, 7.0]
    z_madau[25] = [7.9, 7.9]
    z_madau[26] = [7.0, 7.0]

    z_oesch[0] = [9.0, 9.0]
    z_oesch[1] = [10.0, 10.0]

    # SFRD data
    SFRD_madau = [0]*27
    SFRD_oesch = [0]*2

    SFRD_madau[0] = [-1.82, 0.09, 0.02]
    SFRD_madau[1] = [-1.50, 0.05, 0.05]
    SFRD_madau[2] = [-1.39, 0.15, 0.08]
    SFRD_madau[3] = [-1.20, 0.31, 0.13]
    SFRD_madau[4] = [-1.25, 0.31, 0.13]
    SFRD_madau[5] = [-1.77, 0.08, 0.09]
    SFRD_madau[6] = [-1.75, 0.18, 0.18]
    SFRD_madau[7] = [-1.55, 0.12, 0.12]
    SFRD_madau[8] = [-1.44, 0.10, 0.10]
    SFRD_madau[9] = [-1.24, 0.10, 0.10]
    SFRD_madau[10] = [-0.99, 0.09, 0.08]
    SFRD_madau[11] = [-0.94, 0.09, 0.09]
    SFRD_madau[12] = [-0.95, 0.15, 0.08]
    SFRD_madau[13] = [-0.75, 0.49, 0.09]
    SFRD_madau[14] = [-1.04, 0.26, 0.15]
    SFRD_madau[15] = [-1.69, 0.22, 0.32]
    SFRD_madau[16] = [-1.02, 0.08, 0.08]
    SFRD_madau[17] = [-0.75, 0.12, 0.12]
    SFRD_madau[18] = [-0.87, 0.09, 0.09]
    SFRD_madau[19] = [-0.75, 0.09, 0.11]
    SFRD_madau[20] = [-0.97, 0.11, 0.15]
    SFRD_madau[21] = [-1.29, 0.05, 0.05]
    SFRD_madau[22] = [-1.42, 0.06, 0.06]
    SFRD_madau[23] = [-1.65, 0.08, 0.08]
    SFRD_madau[24] = [-1.79, 0.10, 0.10]
    SFRD_madau[25] = [-2.09, 0.11, 0.11]
    SFRD_madau[26] = [-2.00, 0.10, 0.11]

    SFRD_oesch[0] = [-2.86, 0.19, 0.21]
    SFRD_oesch[1] = [-3.7, 0.7, 0.9]

    #to numpy.array
    SFRD_madau = np.array(SFRD_madau)
    err_madau = np.zeros((len(SFRD_madau), 2))

    z_madau = np.array(z_madau)
    z_center = np.zeros(len(z_madau))
    z_err = np.zeros((len(SFRD_madau), 2))

    for i in range(len(z_madau)):
        z_center[i] = (z_madau[i][0] + z_madau[i][1])/2
        err_madau[i,0] = SFRD_madau[i, 1]
        err_madau[i,1] = SFRD_madau[i, 2]
        z_err[i,0] = abs(z_madau[i,0]-z_center[i])
        z_err[i,1] = abs(z_madau[i,1]-z_center[i])

    SFRD_oesch = np.array(SFRD_oesch)
    err_oesch = np.zeros((len(SFRD_oesch),2))
    z_oesch = np.array(z_oesch)
    z_center_oesch = np.zeros(len(z_oesch))
    z_err_oesch = np.zeros((len(SFRD_oesch),2))
    for i in range(len(z_oesch)):
        z_center_oesch[i] = (z_oesch[i][0] + z_oesch[i][1])/2
        err_oesch[i,0] = SFRD_oesch[i, 1]
        err_oesch[i,1] = SFRD_oesch[i, 2]
        z_err_oesch[i,0] = abs(z_oesch[i,0]-z_center_oesch[i])
        z_err_oesch[i,1] = abs(z_oesch[i,1]-z_center_oesch[i])

    ##################################
    # Madau & Dickinson 2014 Fit
    ##################################

    def rho(z):
        return 0.015*(1+z)**2.7 / (1 + ((1+z)/2.9)**5.6)

    z_fit = np.linspace(0, 10, 100)
    rho_fit = rho(z_fit)

    #######################
    # PLOT
    #############
    figsize=(5,4)
    fig, ax = plt.subplots(1,1, figsize=figsize, dpi = 300)
    ax.set_ylim(-6, -0.5)
    ax.set_xlim(2, 11.5)
    ax.set_xlabel('$z$', fontsize = 15)
    ax.set_ylabel(r'$ \log(\dot{\rho}_{SFR}$ [$M_{\odot}$  cMpc$^{-3}$ yr$^{-1}$])', fontsize = 15)
    ax.minorticks_on()
    #ticks modifier
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    #inward ticks
    ax.tick_params(which='both', direction='in', labelsize = 'large')
    ax.tick_params(which='major', length=6)
    ax.tick_params(which='minor', length=3)
    ax.plot(z_array, np.log10(SFRdens), color = 'deepskyblue', label = 'SFRD', linewidth = 3, alpha = 0.8)
    ax.plot(z_fit, np.log10(rho_fit), color = 'black', label = 'Madau & Dickinson 2014', linestyle= 'dotted', linewidth = 2)
    ax.errorbar(z_center, SFRD_madau.T[0], xerr = z_err.T, yerr = err_madau.T, fmt = '.', color = 'black', linewidth = 1, capsize = 3)
    ax.errorbar(z_center_oesch, SFRD_oesch.T[0], xerr = z_err_oesch.T, yerr = err_oesch.T, fmt = 's', color = 'fuchsia', linewidth = 1,label = 'Oesch+2014', capsize = 3)
    ax.legend(fontsize = 'small', loc = 'best')
    fig.tight_layout()

    return ax