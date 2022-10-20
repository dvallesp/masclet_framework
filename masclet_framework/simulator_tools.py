"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

simulator_tools module
Provides handy tools for debugging or assessing the progress of simulations.
Created by David Vallés
"""

from masclet_framework import units

def dt_to_velocity(dt, cellsize, CFL, z=0.0):
    '''
    From the timestep (particles/gas) computes the maximum speed of a particle/
     gas cell (including sound speed --> Courant).

    Args:
        - dt: timestep in MASCLET units
        - cellsize: coarse grid cellsize (cMpc)
        - CFL: safety factor for timesteps
        - z: redshift

    Returns:
        - speed in units of c
    '''
    dt *= units.time_to_yr/1e6
    v = CFL*cellsize/((1+z)*dt) # speed in Mpc/Myr
    v /= units.c_MpcMyr
    return v
