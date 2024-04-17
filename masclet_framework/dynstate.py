"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

dynstate module
Contains functions to determine the dynamical state of DM haloes, using the parametrisation described by Vallés-Pérez et al. (2023) MNRAS 519(4) 6111-6125.

Created by David Vallés
"""
import numpy as np
from masclet_framework import units

def dynamical_state_thresholds(z=0., mode='VP23'):
    '''
    Returns the thresholds for the dynamical state indicators 
     at a given redshift z.

    Args: 
        - z: redshift (default is 0.)
        - mode: dynamical state parametrisation (default is 'VP23'):
            - 'VP23': Vallés-Pérez et al. (2023) MNRAS 519(4) 6111-6125.
            - 'VP24': Vallés-Pérez et al. (2024) PhD Thesis. This is equivalent to the 'VP23' mode, but including substructure fraction.

    Returns:
        - thresholds (dict): dictionary with the thresholds for the dynamical state indicators
    '''
    thresholds = {}

    if mode=='VP23':
        thresholds['centre_offset'] = 0.0849*z**0
        thresholds['virial_ratio'] = 1.3383*z**0 + 0.197*z - 0.0276*z**2
        thresholds['mean_vr'] = 0.0718*z**0 + 0.0056*z
        thresholds['sparsity_200c500c'] = 1.491*z**0 + 0.064*z - 0.031*z**2 + 0.0060*z**3
        thresholds['ellipticity'] = 0.2696*z**0
    elif mode=='VP24':
        thresholds['centre_offset'] = 0.0716*z**0 + 0.0323*z - 0.0146*z**2 + 0.00192*z**3
        thresholds['virial_ratio'] = 1.3411*z**0 + 0.162*z - 0.043*z**2 + 0.0038*z**3
        thresholds['mean_vr'] = 0.0858*z**0
        thresholds['sparsity_200c500c'] = 1.550*z**0 + 0.0558*z
        thresholds['ellipticity'] = 0.2798*z**0
        thresholds['substructure_fraction'] = 0.0178*z**0 - 0.0080*z + 0.00092*z**2
    else:
        print('ERROR: mode not recognised. Please, choose between VP23 and VP24.')
        return None
    
    return thresholds
    

def dynamical_state_weights(z=0., mode='VP23', min_thr_weights=0.05):
    '''
    Returns the weights for the dynamical state indicators 
     at a given redshift z.

    Args: 
        - z: redshift (default is 0.)
        - mode: dynamical state parametrisation (default is 'VP23'):
            - 'VP23': Vallés-Pérez et al. (2023) MNRAS 519(4) 6111-6125.
            - 'VP24': Vallés-Pérez et al. (2024) PhD Thesis. This is equivalent 
                      to the 'VP23' mode, but including substructure fraction.

    Returns:
        - weights (dict): dictionary with the thresholds for the dynamical state indicators
    '''
    if min_thr_weights < 0. or min_thr_weights > 1.:
        raise ValueError('ERROR: min_thr_weights must be between 0 and 1.')

    weights = {}

    if mode=='VP23':
        weights['centre_offset'] = 0.1679*z**0 + 0.0423*z
        weights['virial_ratio'] = 0.1965*z**0 - 0.1037*z + 0.0134*z**2
        weights['mean_vr'] = 0.1370*z**0 + 0.0364*z
        weights['sparsity_200c500c'] = 0.2327*z**0 + 0.051*z - 0.0153*z**2
        weights['ellipticity'] = 0.2603*z**0 - 0.0181*z
    elif mode=='VP24':
        weights['centre_offset'] = 0.1781*z**0 + 0.0297*z
        weights['virial_ratio'] = 0.1722*z**0 - 0.1510*z + 0.0259*z**2
        weights['mean_vr'] = 0.093*z**0 + 0.051*z
        weights['sparsity_200c500c'] = 0.2034*z**0
        weights['ellipticity'] = 0.175*z**0 + 0.057*z - 0.054*z**2 + 0.0074*z**3
        weights['substructure_fraction'] = 0.1492*z**0 + 0.1052*z - 0.0297*z**2
    else:
        print('ERROR: mode not recognised. Please, choose between VP23 and VP24.')
        return None
    
    # Correction for non-monotonic behaviour of virial ratio with redshift
    if type(z) is not np.ndarray:
        if z > 4.:
            weights['virial_ratio'] = 0.
    else: # z is an array
        weights['virial_ratio'][z > 4.] = 0.
    # Also for ellipticity in mode VP24
    if mode=='VP24':
        if type(z) is not np.ndarray:
            if z > 4.:
                weights['ellipticity'] = 0.
        else: # z is an array
            weights['ellipticity'][z > 4.] = 0.
    
    # Normalise weights and make zero those below min_thr_weights
    if type(z) is not np.ndarray:
        sum_weights = sum(weights.values())
        for key in weights.keys():
            weights[key] /= sum_weights
        
        itera=True 
        while itera:
            itera=False
            for key in weights.keys():
                if 0 < weights[key] < min_thr_weights:
                    itera=True
                    weights[key] = 0.
            if itera:
                sum_weights = sum(weights.values())
                for key in weights.keys():
                    weights[key] /= sum_weights
    else: # z is an array
        sum_weights = np.sum(list(weights.values()), axis=0)
        for key in weights.keys():
            weights[key] /= sum_weights
        
        itera=True
        while itera:
            itera=False
            for key in weights.keys():
                if np.any((weights[key] < min_thr_weights) * (weights[key] > 0.)):
                    itera=True
                    weights[key][weights[key] < min_thr_weights] = 0.
            if itera:
                sum_weights = np.sum(list(weights.values()), axis=0)
                for key in weights.keys():
                    weights[key] /= sum_weights
    
    return weights


def dynamical_state_classification(z=0., centre_offset=None, virial_ratio=None, mean_vr=None, 
                                   sparsity_200c500c=None, ellipticity=None, substructure_fraction=None,
                                   mode='VP23', num_indicators_for_unrelaxed=1, return_full_summary=False):
    '''
    Classifies the dynamical state of a DM halo based on the values of the dynamical state indicators.

    Args:
        - z: redshift
        - centre_offset: centre offset, normalised to virial radius
        - virial_ratio: virial ratio
        - mean_vr: mean radial velocity in absolute value, normalised to the circular velocity at the virial radius
        - sparsity_200c500c: sparsity between R200c and R500c
        - ellipticity: ellipticity
        - substructure_fraction: substructure mass fraction
        - mode: dynamical state parametrisation (default is 'VP23'):
            - 'VP23': Vallés-Pérez et al. (2023) MNRAS 519(4) 6111-6125.
            - 'VP24': Vallés-Pérez et al. (2024) PhD Thesis. This is equivalent 
                      to the 'VP23' mode, but including substructure fraction.
        - num_indicators_for_unrelaxed: number of indicators that must be above the threshold for the halo 
           to be ruled out of the totally relaxed class
        - return_full_summary: if True, returns a dictionary with the dynamical state combined indicator (xi)
                               and the values of the individual dynamical state indicators. Else, it only returns xi.

    Returns:
        - dynamical_state (str): dynamical state classification (either 'RELAXED', 'UNRELAXED' 
                                                                 or 'MARGINALLY_RELAXED')
        - if return_full_summary is True:
            - summary (dict): summary of the classification, including the dynamical state combined indicator (xi)
                              and the values of the individual dynamical state indicators
          else:
            - xi (float): dynamical state combined indicator (relaxedness parameter)
    '''
    # Check that at least one indicator is given
    num_indicators = (centre_offset is not None) + (virial_ratio is not None) + (mean_vr is not None) + \
                     (sparsity_200c500c is not None) + (ellipticity is not None) + ((substructure_fraction is not None) and mode=='VP24')
    if num_indicators == 0:
        raise ValueError('ERROR: at least one dynamical state indicator must be given.')
    if num_indicators < num_indicators_for_unrelaxed:
        raise ValueError('ERROR: num_indicators_for_unrelaxed must be less or equal to the number of indicators given.')
    
    # Get thresholds and weights
    thresholds = dynamical_state_thresholds(z=z, mode=mode)
    weights = dynamical_state_weights(z=z, mode=mode)
    if centre_offset is None:
        weights['centre_offset'] = 0.
    if virial_ratio is None:
        weights['virial_ratio'] = 0.
    if mean_vr is None:
        weights['mean_vr'] = 0.
    if sparsity_200c500c is None:
        weights['sparsity_200c500c'] = 0.
    if ellipticity is None:
        weights['ellipticity'] = 0.
    if substructure_fraction is None or mode=='VP23':
        weights['substructure_fraction'] = 0.
    # Normalize the weights after this 
    sum_weights = sum(weights.values())
    for key in weights.keys():
        weights[key] /= sum_weights

    # Indicator values normalised to their thresholds 
    xi_centre_offset = centre_offset / thresholds['centre_offset'] if centre_offset is not None else 0. 
    xi_virial_ratio = (virial_ratio-1.) / (thresholds['virial_ratio']-1.) if virial_ratio is not None else 0.
    xi_mean_vr = mean_vr / thresholds['mean_vr'] if mean_vr is not None else 0.
    xi_sparsity_200c500c = (sparsity_200c500c-1.) / (thresholds['sparsity_200c500c']-1.) if sparsity_200c500c is not None else 0.
    xi_ellipticity = ellipticity / thresholds['ellipticity'] if ellipticity is not None else 0.
    xi_substructure_fraction = substructure_fraction / thresholds['substructure_fraction'] if (substructure_fraction is not None and weights['substructure_fraction']>0.) else 0.

    # Dynamical state combined indicator
    xi =  (weights['centre_offset'] * xi_centre_offset**2 + \
           weights['virial_ratio'] * xi_virial_ratio**2 + \
           weights['mean_vr'] * xi_mean_vr**2 + \
           weights['sparsity_200c500c'] * xi_sparsity_200c500c**2 + \
           weights['ellipticity'] * xi_ellipticity**2 + \
           weights['substructure_fraction'] * xi_substructure_fraction**2)**(-0.5)

    # Determine if it is totally relaxed 
    num_unrelaxed = (xi_centre_offset is not None and xi_centre_offset > 1.) + \
                    (xi_virial_ratio is not None and xi_virial_ratio > 1.) + \
                    (xi_mean_vr is not None and xi_mean_vr > 1.) + \
                    (xi_sparsity_200c500c is not None and xi_sparsity_200c500c > 1.) + \
                    (xi_ellipticity is not None and xi_ellipticity > 1.) + \
                    (xi_substructure_fraction is not None and xi_substructure_fraction > 1.)
    if num_unrelaxed < num_indicators_for_unrelaxed:
        classification = 'RELAXED'
    else:
        if xi >= 1.:
            classification = 'MARGINALLY RELAXED'
        else:
            classification = 'UNRELAXED'
    summary = {'xi': xi,
               'centre_offset': {'value': centre_offset, 'threshold': thresholds['centre_offset'], 'weight': weights['centre_offset']},
               'virial_ratio': {'value': virial_ratio, 'threshold': thresholds['virial_ratio'], 'weight': weights['virial_ratio']},
               'mean_vr': {'value': mean_vr, 'threshold': thresholds['mean_vr'], 'weight': weights['mean_vr']},
               'sparsity_200c500c': {'value': sparsity_200c500c, 'threshold': thresholds['sparsity_200c500c'], 'weight': weights['sparsity_200c500c']},
               'ellipticity': {'value': ellipticity, 'threshold': thresholds['ellipticity'], 'weight': weights['ellipticity']}
               }
    if mode=='VP24':
        summary['substructure_fraction'] = {'value': substructure_fraction, 'threshold': thresholds['substructure_fraction'], 'weight': weights['substructure_fraction']}
 
    if return_full_summary:
        return classification, summary
    else:
        return classification, xi


def dynamical_state_classification_halo(halo, z=0., mode='VP23', num_indicators_for_unrelaxed=1,
                                        ignore_centre_offset=False, ignore_virial_ratio=False, ignore_mean_vr=False,
                                        ignore_sparsity_200c500c=False, ignore_ellipticity=False, ignore_substructure_fraction=False,
                                        verbose=True, return_full_summary=False):
    '''
    Classifies the dynamical state of a DM halo based on the values of the dynamical state indicators.
    The halo is input as a dictionary in the format of ASOHF outputs (see read_ASOHF).

    Args:
        - halo: dictionary with the halo properties
        - z: redshift
        - mode: dynamical state parametrisation (default is 'VP23'):
            - 'VP23': Vallés-Pérez et al. (2023) MNRAS 519(4) 6111-6125.
            - 'VP24': Vallés-Pérez et al. (2024) PhD Thesis. This is equivalent 
                      to the 'VP23' mode, but including substructure fraction.
        - num_indicators_for_unrelaxed: number of indicators that must be above the threshold for the halo 
           to be ruled out of the totally relaxed class
        - ignore_centre_offset: if True, the centre offset indicator is ignored
        - ignore_virial_ratio: if True, the virial ratio indicator is ignored
        - ignore_mean_vr: if True, the mean radial velocity indicator is ignored
        - ignore_sparsity_200c500c: if True, the sparsity indicator is ignored
        - ignore_ellipticity: if True, the ellipticity indicator is ignored
        - ignore_substructure_fraction: if True, the substructure fraction indicator is ignored
        - verbose: if True (default), prints warnings if some indicators are undefined
        - return_full_summary: if True, returns a dictionary with the dynamical state combined indicator (xi)
                               and the values of the individual dynamical state indicators. Else, it only returns xi.

    Returns:
        - dynamical_state (str): dynamical state classification (either 'RELAXED', 'UNRELAXED' 
                                                                 or 'MARGINALLY_RELAXED')
        - if return_full_summary is True:
            - summary (dict): summary of the classification, including the dynamical state combined indicator (xi)
                              and the values of the individual dynamical state indicators
          else:
            - xi (float): dynamical state combined indicator (relaxedness parameter)
    '''
    # Check that at least one indicator is given
    num_indicators = (not ignore_centre_offset) + (not ignore_virial_ratio) + (not ignore_mean_vr) + \
                     (not ignore_sparsity_200c500c) + (not ignore_ellipticity) + (not ignore_substructure_fraction)
    if num_indicators == 0:
        raise ValueError('ERROR: at least one dynamical state indicator must be given.')
    
    # Compute the indicators 
    xc, yc, zc = halo['x'], halo['y'], halo['z']
    xcm, ycm, zcm = halo['xcm'], halo['ycm'], halo['zcm']
    rvir, mvir = halo['Rvir'], halo['M']
    m200c = halo['M200c']
    m500c = halo['M500c']
    epot, ekin = halo['Epot'], halo['Ekin']
    mean_vr = halo['mean_vr']

    centre_offset = np.sqrt((xc-xcm)**2 + (yc-ycm)**2 + (zc-zcm)**2) / rvir

    virial_ratio = 2.*ekin / abs(epot) if ekin > 0. else None
    if virial_ratio is None and verbose:
        print('WARNING: virial ratio is not defined for halo', halo['id'])
    
    vcirc = units.G_isu * mvir*units.sun_to_kg / (rvir*units.mpc_to_m)
    vcirc = np.sqrt(vcirc)/1000 # in km/s
    mean_vr = abs(mean_vr) / vcirc

    sparsity_200c500c = m200c / m500c if m500c>0 else None 
    if sparsity_200c500c is None and verbose:
        print('WARNING: sparsity_200c500c is not defined for halo', halo['id'])

    ellipticity = 1 - halo['minorSemiaxis'] / halo['majorSemiaxis']

    substructure_fraction = halo['fsub']
    
    return dynamical_state_classification(z=z, centre_offset=centre_offset, virial_ratio=virial_ratio, mean_vr=mean_vr,
                                          sparsity_200c500c=sparsity_200c500c, ellipticity=ellipticity, substructure_fraction=substructure_fraction,
                                          mode=mode, num_indicators_for_unrelaxed=num_indicators_for_unrelaxed, return_full_summary=return_full_summary)




