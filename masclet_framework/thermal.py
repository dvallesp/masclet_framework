"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

thermal module
Provides useful functions to compute and work with thermodynamical quantities
Created by David Vallés
"""

from masclet_framework import units

def entropy(mcells=None, Vcells=None, Tcells=None, rhocells=None, mu=0.6):
    '''
    Computes the entropy,
    
    K = kB T n_e^{-2/3},
    
    assuming neutrality (n_p = n_e). Either (mcells, Vcells), or (rhocells) must 
        be supplied.
        
    Args:
        - mcells: mass of each cell in Msun
        - Vcells: volume of each cell in Mpc^3
        - Tcells: temperature of the cell in K
        - rhocells: density of the cell in Msun/Mpc^3
        - mu (optional, default 0.6): mean molecular weight
        
    Returns:
        - entropy: entropy of the cell in keV cm^2
    
    '''
    
    if (mcells is not None) and (Vcells is not None):
        if rhocells is None:
            rhocells = mcells/Vcells
        else:
            print('Error! Give mcells and Vcells; or rhocells; but not both')
            return None
    elif rhocells is not None:
        pass
    else:
        print('Error! Give either (mcells, Vcells), or (rhocells); but not neither!')
        return None
    
    constant = units.kB_isu * units.mp_isu**(2/3) * mu**(2/3) / (1-mu)**(2/3)
    unit_conversion = units.J_to_keV * units.kg_to_sun**(2/3) * units.mpc_to_cm**2
    
    return (constant*unit_conversion) * (Tcells / rhocells**(2/3))