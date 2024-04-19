"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

thermal module
Provides useful functions to compute and work with thermodynamical quantities
Created by David Vallés
"""

from masclet_framework import units, cosmo_tools

def molecular_weight(T, P, delta, h=0.678, omega_m=0.31, kept_patches=None, electrons=False): 
    ''' 
    Computes the local mean melecular weight, 

    mp mu = rho kB T / P

    The computation is performed on the AMR grid.

    Args:
        - T: temperature field in K (mandatory)
        - P: pressure field, in MASCLET units
        - delta: overdensity field
        - h: dimensionless Hubble parameter, H_0 = 100 h km/s/Mpc
        - omega_m: matter density parameter, at z=0
        - kept_patches: 1d boolean array, True if the patch is kept, False if not.
            If None, all patches are kept.
        - electrons (bool): if True, returns the molecular weight per electron, 
            mu_e = 5 mu / (2 + mu) (default: False)
    '''
    if kept_patches is None:
        kept_patches = np.ones(len(T), dtype=bool)
    
    consta_rho = (cosmo_tools.background_density(h, omega_m, z=0.) * (units.sun_to_kg / units.mpc_to_m**3)) # kg/m^3
    consta_T = units.kB_isu # J / K = kg m^2 s^-2 / K
    consta_num = consta_rho * consta_T # kg^2 m^-1 s^-2 / K
    consta_den = units.mp_isu * units.pressure_to_isu # kg^2 m^-1 s^-2
    consta = consta_num / consta_den # 1/K 

    mu = [consta * (1+di) * Ti / Pi if ki else 0. for Ti, Pi, di, ki in zip(T, P, delta, kept_patches)]

    if electrons:
        mu = [5*mi / (2+mi) if ki else 0. for mi, ki in zip(mu, kept_patches)]

    return mu


def ionisation_fraction(T, P, delta, h=0.678, omega_m=0.31, kept_patches=None):
    ''' 
    Computes the local ionisation fraction, 

    chi = (16 - 13 mu) / (14 mu).

    The computation is performed on the AMR grid.

    Args:
        - T: temperature field in K (mandatory)
        - P: pressure field, in MASCLET units
        - delta: overdensity field
        - h: dimensionless Hubble parameter, H_0 = 100 h km/s/Mpc
        - omega_m: matter density parameter, at z=0
        - kept_patches: 1d boolean array, True if the patch is kept, False if not.
            If None, all patches are kept.
    
    Returns:
        - chi: ionisation fraction
    '''
    if kept_patches is None:
        kept_patches = np.ones(len(T), dtype=bool)

    mu = molecular_weight(T, P, delta, h=h, omega_m=omega_m, kept_patches=kept_patches, electrons=False)

    chi = [(16-13*mi) / (14*mi) if ki else 0. for mi, ki in zip(mu, kept_patches)]

    return chi

def entropy(T, P=None, delta=None, mu=None, z=0.0, h=0.678, omega_m=0.31, mode='local', kept_patches=None):
    '''
    Computes the gas entropy, 

    K = kB T n^{-2/3},

    either assuming constant molecular mass (mu) (mode='global') by n = rho / mu mp, 
    or local mu (mode='local') by K = (kB T)^(5/3) / P^(2/3).

    The computation is performed on the AMR grid. 

    Args:
        - T: temperature field in K (mandatory)
        - P: pressure field, in MASCLET units
        - delta: overdensity field
        - mu: mean molecular weight (scalar)
        - z: redshift. z=0. to use the comoving particle number density.
        - h: dimensionless Hubble parameter, H_0 = 100 h km/s/Mpc
        - omega_m: matter density parameter, at z=0
        - mode: 'global' or 'local' 
            - if 'local', one must supply T and P 
            - if 'global', one must supply T, delta, mu, and z
        - kept_patches: 1d boolean array, True if the patch is kept, False if not.
            If None, all patches are kept.

    Returns: 
        - K: gas entropy field in keV cm^2
    ''' 
    if kept_patches is None:
        kept_patches = np.ones(len(T), dtype=bool)

    if mode=='local':
        if (T is None) or (P is None):
            raise ValueError('Error! T and P must be supplied in local mode')
        consta_T = units.kB_isu * units.J_to_keV # keV
        consta_P = units.pressure_to_keVcm3
        consta = consta_T**(5/3) / consta_P**(2/3) # keV cm^2
        K = [consta * Ti**(5/3) / Pi**(2/3) if ki else 0. for Ti, Pi, ki in zip(T, P, kept_patches)]
    elif mode=='global':
        if (T is None) or (delta is None) or (mu is None) or (z is None):
            raise ValueError('Error! T, delta, mu, and z must be supplied in global mode')
        consta_T = units.kB_isu * units.J_to_keV # keV
        consta_rho = cosmo_tools.background_density(h, omega_m, z=z) * (units.sun_to_g / units.mpc_to_cm**3) # g/cm^3
        consta_n = consta_rho / (mu * units.mp_cgs) # cm^-3
        consta = consta_T / consta_n**(2/3) # keV cm^2
        K = [consta * Ti / (1+di)**(2/3) if ki else 0. for Ti, di, ki in zip(T, delta, kept_patches)]
    else:
        raise ValueError('Error! mode must be either "local" or "global"')

    return K


def particle_number_density(delta, T=None, P=None, mu=None, z=0.0, h=0.678, omega_m=0.31, mode='local', kept_patches=None):
    ''' 
    Computes the particle number density,

    n = rho / mu mp,
    
    where mu is the mean molecular weight, either constant (mu) or local (see molecular_weight function).

    The computation is performed on the AMR grid.

    Args:
        - delta: overdensity field (mandatory)
        - T: temperature field in K
        - P: pressure field, in MASCLET units
        - mu: mean molecular weight (scalar)
        - z: redshift. z=0. to use the comoving particle number density.
        - h: dimensionless Hubble parameter, H_0 = 100 h km/s/Mpc
        - omega_m: matter density parameter, at z=0
        - mode: 'global' or 'local' 
            - if 'local', one must supply delta, T, and P
            - if 'global', one must supply delta, mu
        - kept_patches: 1d boolean array, True if the patch is kept, False if not.
            If None, all patches are kept.

    Returns:
        - n: particle number density field in cm^-3
    '''
    if kept_patches is None:
        kept_patches = np.ones(len(delta), dtype=bool)

    if mode=='local':
        if (T is None) or (P is None):
            raise ValueError('Error! T and P must be supplied in local mode')
        consta_rho = (cosmo_tools.background_density(h, omega_m, z=z) * (units.sun_to_kg / units.mpc_to_cm**3)) # kg / cm^3
        mu = molecular_weight(T, P, delta, h=h, omega_m=omega_m, kept_patches=kept_patches)
        n = [consta_rho * (1+di) / (mi * units.mp_isu) if ki else 0. for di, mi, ki in zip(delta, mu, kept_patches)] # cm^-3
    elif mode=='global':
        if mu is None:
            raise ValueError('Error! mu must be supplied in global mode')
        consta_rho = (cosmo_tools.background_density(h, omega_m, z=z) * (units.sun_to_kg / units.mpc_to_cm**3)) # kg / cm^3
        n = [consta_rho * (1+di) / (mu * units.mp_isu) if ki else 0. for di, ki in zip(delta, kept_patches)] # cm^-3
    else:
        raise ValueError('Error! mode must be either "local" or "global"')

    return n


def electron_number_density(delta, T=None, P=None, mu=None, z=0.0, h=0.678, omega_m=0.31, mode='local', kept_patches=None):
    '''
    Computes the electron number density, 

    n_e = n_p = rho / mu_e mp,

    where mu_e = 5mu / (2+mu) is the mean molecular weight of the electrons, either constant (given through mu) 
        or local (see molecular_weight function).

    The computation is performed on the AMR grid.

    Args:
        - delta: overdensity field (mandatory)
        - T: temperature field in K
        - P: pressure field, in MASCLET units
        - mu: mean molecular weight (scalar)
        - z: redshift. z=0. to use the comoving particle number density.
        - h: dimensionless Hubble parameter, H_0 = 100 h km/s/Mpc
        - omega_m: matter density parameter, at z=0
        - mode: 'global' or 'local' 
            - if 'local', one must supply delta, T, and P
            - if 'global', one must supply delta, mu
        - kept_patches: 1d boolean array, True if the patch is kept, False if not.
            If None, all patches are kept.

    Returns:
        - n_e: electron number density field in cm^-3
    '''
    if kept_patches is None:
        kept_patches = np.ones(len(delta), dtype=bool)
    
    n = particle_number_density(delta, T=T, P=P, mu=mu, z=z, h=h, omega_m=omega_m, mode=mode, kept_patches=kept_patches)

    if mode=='local':
        if (T is None) or (P is None):
            raise ValueError('Error! T and P must be supplied in local mode')

        mu = molecular_weight(T, P, delta, h=h, omega_m=omega_m, kept_patches=kept_patches)

        n = [ni * (2+mi)/5 if ki else 0. for ni, mi, ki in zip(n, mu, kept_patches)]
    elif mode=='global':
        if mu is None:
            raise ValueError('Error! mu must be supplied in global mode')

        n = [ni * (2+mu)/5 if ki else 0. for ni, ki in zip(n, kept_patches)]

    return n


        

def entropy_electrons(T, delta, P=None, mu=None, z=0.0, h=0.678, omega_m=0.31, mode='local', kept_patches=None):
    ''' 
    Computes the electron entropy, 

    K_e = kB T n_e^{-2/3},

    either assuming constant molecular mass (mu) (mode='global') by n = rho / mu mp,
    or local mu (mode='local') by K = (kB T)^(5/3) / P^(2/3).

    The computation is performed on the AMR grid.

    Args:
        - T: temperature field in K (mandatory)
        - delta: overdensity field (mandatory)
        - P: pressure field, in MASCLET units
        - mu: mean molecular weight (scalar)
        - z: redshift. z=0. to use the comoving particle number density.
        - h: dimensionless Hubble parameter, H_0 = 100 h km/s/Mpc
        - omega_m: matter density parameter, at z=0
        - mode: 'global' or 'local' 
            - if 'local', one must supply T, delta and P 
            - if 'global', one must supply T, delta, mu, and z
        - kept_patches: 1d boolean array, True if the patch is kept, False if not.
            If None, all patches are kept.

    Returns:
        - K_e: electron entropy field in keV cm^2
    ''' 
    if kept_patches is None:
        kept_patches = np.ones(len(T), dtype=bool)

    K = entropy(T=T, P=P, delta=delta, mu=mu, z=z, h=h, omega_m=omega_m, mode=mode, kept_patches=kept_patches) 

    if mode=='global':
        if mu is None:
            raise ValueError('Error! mu must be supplied in global mode')
        consta = ((2+mu)/5)**(-2/3)
        Ke = [Ki * consta if ki else 0. for Ki, ki in zip(K, kept_patches)]
    elif mode=='local':
        if P is None:
            raise ValueError('Error! P must be supplied in local mode')
        mu = molecular_weight(T, P, delta, h=h, omega_m=omega_m, kept_patches=kept_patches)
        Ke = [Ki * ((2+mi)/5)**(-2/3) if ki else 0. for Ki, mi, ki in zip(K, mu, kept_patches)]
    else:
        raise ValueError('Error! mode must be either "local" or "global"')

    return Ke

       
def electron_pressure(T, delta, P=None, mu=None, z=0.0, h=0.678, omega_m=0.31, mode='local', kept_patches=None):
    ''' 
    Computes the electron pressure,

    P_e = n_e k_B T,

    where n_e is the electron number density, computed either by assuming a constant (mu) or local (electron_number_density 
     function) mean molecular weight.

    The computation is performed on the AMR grid.

    Args:
        - T: temperature field in K (mandatory)
        - delta: overdensity field (mandatory)
        - P: pressure field, in MASCLET units
        - mu: mean molecular weight (scalar)
        - z: redshift. z=0. to use the comoving particle number density.
        - h: dimensionless Hubble parameter, H_0 = 100 h km/s/Mpc
        - omega_m: matter density parameter, at z=0
        - mode: 'global' or 'local'
            - if 'local', one must supply P 
            - if 'global', one must supply mu
        - kept_patches: 1d boolean array, True if the patch is kept, False if not.
            If None, all patches are kept.

    Returns:
        - P_e: electron pressure field in keV cm^-3
    '''

    if kept_patches is None:
        kept_patches = np.ones(len(T), dtype=bool)

    if mode=='local':
        if P is None:
            raise ValueError('Error! P must be supplied in local mode')
        ne = electron_number_density(delta, T=T, P=P, mu=mu, z=z, h=h, omega_m=omega_m, mode='local', kept_patches=kept_patches) # cm^-3
        consta = units.kB_isu * units.J_to_keV # keV
        Pe = [nei * Ti * consta if ki else 0. for nei, Ti, ki in zip(ne, T, kept_patches)]
    elif mode=='global':
        if mu is None:
            raise ValueError('Error! mu must be supplied in global mode')
        ne = electron_number_density(delta, T=None, P=None, mu=mu, z=z, h=h, omega_m=omega_m, mode='global', kept_patches=kept_patches)
        consta = units.kB_isu * units.J_to_keV # keV
        Pe = [nei * Ti * consta if ki else 0. for nei, Ti, ki in zip(ne, T, kept_patches)]
    else:
        raise ValueError('Error! mode must be either "local" or "global"')

    return Pe


def entropy_cells(mcells=None, Vcells=None, Tcells=None, rhocells=None, mu=0.6):
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