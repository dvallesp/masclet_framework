"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

spectral module
Provides useful tools to compute power spectra, energy power spectra, etc.
Created by David Vallés
"""


import numpy as np
from scipy import fft, stats


def power_spectrum_scalar_field(data, dx=1., ncores=1, do_zero_pad=False):
    '''
    This function computes the power spectrum, P(k), of a 3D cubic scalar field.

    Args:
        - data: the 3D array containing the input field
        - dx: uniform spacing of the grid in the desired input units
        - ncores: number of workers for parallel computation of the FFT
        - do_zero_pad: if True, the FFTs are computed using 0-padding, doubling the domain

    Returns:
        - kvals: the spatial frequencies on which the power spectra has been
            evaluated, in the inverse of the units of dx
        - Pk: the power spectrum at the kvals spatial frequency points

    '''

    # Step 1. Compute the FFT, its amplitude square, and normalise it
    #fft_data = np.fft.fftn(data, s=data.shape)#.shape
    
    ### SPECIAL TREATMENT OF ZERO-PADDING
    if not do_zero_pad:
        shape = data.shape
    else:
        shape = [2*s for s in data.shape]
    ### END SPECIAL TREATMENT OF ZERO-PADDING
        
    fft_data = fft.fftn(data, s=shape, workers=ncores)
    fourier_amplitudes = (np.abs(fft_data)**2).flatten() / data.size**2 * (data.shape[0]*dx)**3
    nx,ny,nz = data.shape

    # Step 2. Obtain the frequencies
    frequencies_x = np.fft.fftfreq(shape[0], d=dx)
    frequencies_y = np.fft.fftfreq(shape[1], d=dx)
    frequencies_z = np.fft.fftfreq(shape[2], d=dx)
    a,b,c=np.meshgrid(frequencies_x, frequencies_y, frequencies_z, indexing='ij')
    knrm=np.sqrt(a**2+b**2+c**2).flatten()

    # Step 3. Assuming isotropy, obtain the P(k) (1-dimensional power spectrum)
    
    ### SPECIAL TREATMENT OF ZERO-PADDING
    if not do_zero_pad:
        delta_f = frequencies_x[1]
    else:
        delta_f = 2.0*frequencies_x[1]
    ### END SPECIAL TREATMENT OF ZERO-PADDING
    
    kbins = np.arange(frequencies_x[1]/2, np.abs(frequencies_x).max(), delta_f)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    Pk, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)

    return kvals, Pk


def power_spectrum_vector_field(data_x, data_y, data_z, dx=1., ncores=1, do_zero_pad=False):
    '''
    This function computes the power spectrum, P(k), of a 3D cubic vector field.

    Args:
        - data_x, data_y, data_z: the 3D arrays containing the 3 components of the vector field
        - dx: uniform spacing of the grid in the desired input units
        - ncores: number of workers for parallel computation of the FFT
        - do_zero_pad: if True, the FFTs are computed using 0-padding, doubling the domain

    Returns:
        - kvals: the spatial frequencies on which the power spectra has been
            evaluated, in the inverse of the units of dx
        - Pk: the power spectrum at the kvals spatial frequency points

    '''

    kvals, Pk_x = power_spectrum_scalar_field(data_x, dx=dx, ncores=ncores, do_zero_pad=do_zero_pad)
    kvals, Pk_y = power_spectrum_scalar_field(data_y, dx=dx, ncores=ncores, do_zero_pad=do_zero_pad)
    kvals, Pk_z = power_spectrum_scalar_field(data_z, dx=dx, ncores=ncores, do_zero_pad=do_zero_pad)

    Pk = Pk_x+Pk_y+Pk_z

    return kvals, Pk


def energy_spectrum_scalar_field(data, dx=1., ncores=1, do_zero_pad=False):
    '''
    This function computes the energy power spectrum, E(k), of a 3D cubic scalar field.

    This is defined from the P(k) as:
        E(k) = 2 \pi k^2 P(k)

    And satisfies:
        \int_0^\infty E(k) dk = 1/2 <data^2>

    Args:
        - data: the 3D array containing the input field
        - dx: uniform spacing of the grid in the desired input units
        - ncores: number of workers for parallel computation of the FFT
        - do_zero_pad: if True, the FFTs are computed using 0-padding, doubling the domain

    Returns:
        - kvals: the spatial frequencies on which the power spectra has been
            evaluated, in the inverse of the units of dx
        - Ek: the energy power spectrum at the kvals spatial frequency points

    '''

    kvals, pk = power_spectrum_scalar_field(data, dx=dx, ncores=ncores, do_zero_pad=do_zero_pad)
    Ek = pk * (2*np.pi*kvals**2)

    return kvals, Ek


def energy_spectrum_vector_field(data_x, data_y, data_z, dx=1., ncores=1, do_zero_pad=False):
    '''
    This function computes the energy power spectrum, E(k), of a 3D cubic vector field.

    This is defined from the P(k) as:
        E(k) = 2 \pi k^2 P(k)

    And satisfies:
        \int_0^\infty E(k) dk = 1/2 <\vec data^2>

    Args:
        - data_x, data_y, data_z: the 3D arrays containing the 3 components of the vector field
        - dx: uniform spacing of the grid in the desired input units
        - ncores: number of workers for parallel computation of the FFT
        - do_zero_pad: if True, the FFTs are computed using 0-padding, doubling the domain

    Returns:
        - kvals: the spatial frequencies on which the power spectra has been
            evaluated, in the inverse of the units of dx
        - Ek: the energy power spectrum at the kvals spatial frequency points

    '''
    kvals, pk = power_spectrum_vector_field(data_x, data_y, data_z, dx=dx, ncores=ncores, do_zero_pad=do_zero_pad)
    Ek = pk * (2*np.pi*kvals**2)
    return kvals, Ek

def power_spectrum_vector_field_Helmholtz(data_x, data_y, data_z, dx=1., ncores=1, do_zero_pad=False):
    '''
    This function computes the power spectrum, P(k), of a 3D cubic vector field,
     returning separately the compressive and the solenoidal one.

    Args:
        - data: the 3D array containing the input field
        - dx: uniform spacing of the grid in the desired input units
        - ncores: number of workers for parallel computation of the FFT
        - do_zero_pad: if True, the FFTs are computed using 0-padding, doubling the domain

    Returns:
        - kvals: the spatial frequencies on which the power spectra has been
            evaluated, in the inverse of the units of dx
        - Pk: the power spectrum of the input field at the kvals spatial frequency 
            points
        - Pkcomp: the compressive power spectrum
        - Pksol: the solenoidal power spectrum

    '''

    # Step 1. Compute the FFT, its amplitude square, and normalise it
    #fft_data = np.fft.fftn(data, s=data.shape)#.shape
    
    ### SPECIAL TREATMENT OF ZERO-PADDING
    if do_zero_pad is False:
        shape = data.shape
    else:
        shape = [2*s for s in data.shape]
    ### END SPECIAL TREATMENT OF ZERO-PADDING
    
    nx,ny,nz = data_x.shape
    
    ft_x = fft.fftn(data_x, s=shape, workers=ncores)
    ft_y = fft.fftn(data_y, s=shape, workers=ncores)
    ft_z = fft.fftn(data_z, s=shape, workers=ncores)

    # Step 2. Obtain the frequencies
    frequencies_x = np.fft.fftfreq(shape[0], d=dx)
    frequencies_y = np.fft.fftfreq(shape[1], d=dx)
    frequencies_z = np.fft.fftfreq(shape[2], d=dx)
    kx,ky,kz=np.meshgrid(frequencies_x, frequencies_y, frequencies_z, indexing='ij')
    knrm=np.sqrt(kx**2 + ky**2 + kz**2)#.flatten()
    identity=np.ones(knrm.shape)
    
    #kx = np.nan_to_num(kx/knrm)
    #ky = np.nan_to_num(ky/knrm)
    #kz = np.nan_to_num(kz/knrm)
    knrm[knrm==0.]=-1.
    kx = kx/knrm
    ky = ky/knrm
    kz = kz/knrm
    knrm[knrm==-1.]=0.
    
    # Step 3. Do the projections. Compressive and solenoidal are orthogonal in Fourier space
    #ft_x_sol = (identity - kx**2/knrm**2) * ft_x               - kx*ky/knrm**2  * ft_y               - kx*kz/knrm**2  * ft_z
    #ft_y_sol =           - ky*kx/knrm**2  * ft_x  +  (identity - ky**2/knrm**2) * ft_y               - ky*kz/knrm**2  * ft_z
    #ft_z_sol =           - kz*kx/knrm**2  * ft_x               - kz*ky/knrm**2  * ft_y  +  (identity - kz**2/knrm**2) * ft_z
    ft_x_sol = (identity - kx*kx) * ft_x               - kx*ky  * ft_y               - kx*kz  * ft_z
    ft_y_sol =           - ky*kx  * ft_x  +  (identity - ky*ky) * ft_y               - ky*kz  * ft_z
    ft_z_sol =           - kz*kx  * ft_x               - kz*ky  * ft_y  +  (identity - kz*kz) * ft_z
    
    ft_x_comp = ft_x - ft_x_sol
    ft_y_comp = ft_y - ft_y_sol
    ft_z_comp = ft_z - ft_z_sol
    
    # Step 4. Compute the corresponding amplitudes squared
    fourier_amplitudes = (np.abs(ft_x)**2 + np.abs(ft_y)**2 + np.abs(ft_z)**2).flatten() / data_x.size**2 * (data_x.shape[0]*dx)**3
    fourier_amplitudes_comp = (np.abs(ft_x_comp)**2 + np.abs(ft_y_comp)**2 + np.abs(ft_z_comp)**2).flatten() / data_x.size**2 * (data_x.shape[0]*dx)**3
    fourier_amplitudes_sol = (np.abs(ft_x_sol)**2 + np.abs(ft_y_sol)**2 + np.abs(ft_z_sol)**2).flatten() / data_x.size**2 * (data_x.shape[0]*dx)**3
    knrm = knrm.flatten()

    # Step 5. Assuming isotropy, obtain the P(k) (1-dimensional power spectrum)
    
    ### SPECIAL TREATMENT OF ZERO-PADDING
    if not do_zero_pad:
        delta_f = frequencies_x[1]
    else:
        delta_f = 2.0*frequencies_x[1]
    ### END SPECIAL TREATMENT OF ZERO-PADDING
    
    kbins = np.arange(frequencies_x[1]/2, np.abs(frequencies_x).max(), delta_f)
    kvals = 0.5 * (kbins[1:] + kbins[:-1])
    
    Pk, _, _ = stats.binned_statistic(knrm, fourier_amplitudes,
                                         statistic = "mean",
                                         bins = kbins)
    
    Pkcomp, _, _ = stats.binned_statistic(knrm, fourier_amplitudes_comp,
                                         statistic = "mean",
                                         bins = kbins)
    
    Pksol, _, _ = stats.binned_statistic(knrm, fourier_amplitudes_sol,
                                         statistic = "mean",
                                         bins = kbins)

    return kvals, Pk, Pkcomp, Pksol


def energy_spectrum_vector_field_Helmholtz(data_x, data_y, data_z, dx=1., ncores=1, do_zero_pad=False):
    '''
    This function computes the energy power spectrum, E(k), of a 3D cubic vector field,
     returning separately the total, the compressive and the solenoidal one.

    This is defined from the P(k) as:
        E(k) = 2 \pi k^2 P(k)

    And satisfies:
        \int_0^\infty E(k) dk = 1/2 <\vec data^2>

    Args:
        - data: the 3D array containing the input field
        - dx: uniform spacing of the grid in the desired input units
        - ncores: number of workers for parallel computation of the FFT
        - do_zero_pad: if True, the FFTs are computed using 0-padding, doubling the domain

    Returns:
        - kvals: the spatial frequencies on which the power spectra has been
            evaluated, in the inverse of the units of dx
        - Ek: the energy power spectrum of the input field at the kvals spatial 
            frequency points
        - Ekcomp: the compressive energy power spectrum
        - Eksol: the solenoidal energy power spectrum

    '''
    kvals, pk, pkcomp, pksol = power_spectrum_vector_field_Helmholtz(data_x, data_y, data_z, 
                                                                     dx=dx, ncores=ncores, do_zero_pad=do_zero_pad)
    Ek = pk * (2*np.pi*kvals**2)
    Ekcomp = pkcomp * (2*np.pi*kvals**2)
    Eksol = pksol * (2*np.pi*kvals**2)
    
    return kvals, Ek, Ekcomp, Eksol