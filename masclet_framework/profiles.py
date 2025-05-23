"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

profiles module
Several functions to compute directional profiles of gridded data.
Created by David Vallés
"""

from numba import jit
import numpy as np
from masclet_framework import tools, stats
try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, *args, **kwargs):
        return iterator
from scipy.spatial import KDTree
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import interp1d
from scipy.stats import gaussian_kde
from astropy.stats import biweight_location

@jit(nopython=True, fastmath=True)
def locate_point(x,y,z,npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,nl,buf=1,
                 no_interp=False):
    """
    Given a point (x,y,z) and the patch structure, returns the patch and cell
    where the point is located.

    Args:
        - x,y,z: coordinates of the point
        - npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        - patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        - patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        - size: comoving size of the simulation box
        - nmax: cells at base level
        - nl: maximum AMR level to be considered
        - buf: number of cells to be ignored in the border of each patch
        - no_interp: if True, the point is located in the cell where it is. If False (default), it is 
                     assigned to its left-bottom-back corner cell, so as to do CIC interpolation.

    Returns:
        - f_patch: patch where the point is located
        - ix,jy,kz: cell where the point is located
    """
    f_patch=0
    lev_patch=0
    for l in range(nl,0,-1):
        #print(l)
        low1=npatch[0:l].sum()+1
        low2=npatch[0:l+1].sum()
        dxpa=size/nmax/2**l
        for ipatch in range(low1,low2+1):
            x1=patchrx[ipatch]+(buf-1)*dxpa
            x2=patchrx[ipatch]+(patchnx[ipatch]-1-buf)*dxpa
            if x1>x or x2<x:
                continue

            y1=patchry[ipatch]+(buf-1)*dxpa
            y2=patchry[ipatch]+(patchny[ipatch]-1-buf)*dxpa
            #print(y1,y2)
            if y1>y or y2<y:
                continue
            z1=patchrz[ipatch]+(buf-1)*dxpa
            z2=patchrz[ipatch]+(patchnz[ipatch]-1-buf)*dxpa
            if z1>z or z2<z:
                continue
            
            f_patch=ipatch
            lev_patch=l
            #print('a')
            break
        
        if f_patch>0:
            break
    
    dxpa=size/nmax/2**lev_patch
    x1=patchrx[f_patch]-dxpa
    y1=patchry[f_patch]-dxpa
    z1=patchrz[f_patch]-dxpa

    if no_interp:
        ix=int((x-x1)/dxpa)
        jy=int((y-y1)/dxpa)
        kz=int((z-z1)/dxpa)
    else:  
        ix=int((x-x1)/dxpa-.5)
        jy=int((y-y1)/dxpa-.5)
        kz=int((z-z1)/dxpa-.5)
    
    return f_patch,ix,jy,kz

def dir_profile(field, cx,cy,cz,
                npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,
                binsphi=None, binscostheta=None, binsr=None, rmin=None, rmax=None, dex_rbins=None, delta_rbins=None,
                maxl=None,
                interpolate=True,
                use_tqdm=False):
    """
    Computes a directional profile of a given field, centered in a given point (cx,cy,cz).
    There are several ways to specify the directions along which the profile is computed:
        - binsphi, binscostheta: numpy vectors specifying the direcitons in the spherical angles phi, cos(theta).
        - Nphi, Ncostheta: number of bins in phi and cos(theta), equally spaced between -pi and pi, and -1 and 1, respectively.
    Likewise, there are several ways to specify the radial bins:
        - binsr: numpy vector specifying the radial bin edges
        - rmin, rmax, dex_rbins: minimum and maximum radius, and logarithmic bin size
        - rmin, rmax, delta_rbins: minimum and maximum radius, and linear bin size
    The profile can be computed by nearest neighbour interpolation (interpolate=False) or by averaging the values of the cells in each bin (interpolate=True).

    Args:
        - field: field to be profiled
        - cx,cy,cz: coordinates of the center of the profile
        - npatch: number of patches in each level, starting in l=0 (numpy vector of NLEVELS integers)
        - patchrx, patchry, patchrz: physical position of the center of each patch first ¡l-1! cell
        - patchnx, patchny, patchnz: x-extension of each patch (in level l cells) (and Y and Z)
        - size: comoving size of the simulation box
        - nmax: cells at base level
        One and only one of these pairs of arguments must be specified:
            - binsphi, binscostheta: numpy vectors specifying the direcitons in the spherical angles phi, cos(theta).
            - Nphi, Ncostheta: number of bins in phi and cos(theta), equally spaced between -pi and pi, and -1 and 1, respectively.
        One and only one of these sets of arguments must be specified:
            - binsr: numpy vector specifying the radial bins
            - rmin, rmax, dex_rbins: minimum and maximum radius, and logarithmic bin size
            - rmin, rmax, delta_rbins: minimum and maximum radius, and linear bin size
        - maxl: maximum level to be considered. If None, all levels are considered.
        - interpolate: if True, the profile is computed by averaging the values of the cells in each bin. If False, the profile is computed by nearest neighbour interpolation.
        - use_tqdm (default: False): whether a tqdm progressbar should be displayed.

    Returns:
        - dirprof: directional profile of the field
        - rrr: radial bins
        - vec_costheta: cos(theta) bins
        - vec_phi: phi bins
    """
    # Check phi bins are properly specified
    if type(binsphi) is np.ndarray or type(binsphi) is list:
        Nphi=len(binsphi)
        vec_phi=binsphi
        if type(binsphi) is list:
            vec_phi=np.array(vec_phi)
    elif type(binsphi) is int:
        Nphi=binsphi
        vec_phi=np.linspace(-np.pi + np.pi/Nphi,np.pi -np.pi/Nphi,Nphi)
    else:
        raise ValueError('Wrong specification of binsphi')
        
    # Check theta bins are properly specified
    if type(binscostheta) is np.ndarray or type(binscostheta) is list:
        Ncostheta=len(binscostheta)
        vec_costheta=binscostheta
        if type(binscostheta) is list:
            vec_costheta=np.array(vec_costheta)
    elif type(binscostheta) is int:
        Ncostheta=binscostheta
        vec_costheta=np.linspace(-1+1/Ncostheta,1-1/Ncostheta,Ncostheta)
    else:
        raise ValueError('Wrong specification of binscostheta')  
        
    # Check r bins are properly specified
    if type(binsr) is np.ndarray or type(binsr) is list:
        num_bins=len(binsr)
        rrr=binsr
        if type(binsr) is list:
            rrr=np.array(rrr)
    elif (rmin is not None) and (rmax is not None) and ((dex_rbins is not None) or (delta_rbins is not None)) and ((dex_rbins is None) or (delta_rbins is None)):
        if dex_rbins is not None:
            num_bins=int(np.log10(rmax/rmin)/dex_rbins/2)*2+1 # guarantee it is odd
            rrr = np.logspace(np.log10(rmin),np.log10(rmax),num_bins)
        else:
            num_bins=int((rmax-rmin)/delta_rbins/2)*2+1 # guarantee it is odd
            rrr = np.linspace(rmin,rmax,num_bins)
    else:
        raise ValueError('Wrong specification of binsr') 
    

    levels=tools.create_vector_levels(npatch)
    nl=levels.max()

    if maxl is not None:
        nl = min(nl,maxl)
        
    drrr=np.concatenate([[rrr[1]-rrr[0]], np.diff(rrr)])
    lev_integral = np.clip(np.log2((size/nmax)/drrr).astype('int32'),0,nl)#+1
    del drrr
    
    dir_profiles = np.zeros((Ncostheta,Nphi,num_bins))

    halfsize=size/2
    mhalfsize=-halfsize

    if interpolate:
        for itheta,costheta in tqdm(enumerate(vec_costheta),total=len(vec_costheta),disable=not use_tqdm):
            for jphi,phi in enumerate(vec_phi):
                xxx=cx+rrr*np.sqrt(1-costheta**2)*np.cos(phi)
                yyy=cy+rrr*np.sqrt(1-costheta**2)*np.sin(phi)
                zzz=cz+rrr*costheta

                # Periodic boundary conditions
                xxx[xxx>halfsize]=xxx[xxx>halfsize]-size
                xxx[xxx<mhalfsize]=xxx[xxx<mhalfsize]+size
                yyy[yyy>halfsize]=yyy[yyy>halfsize]-size
                yyy[yyy<mhalfsize]=yyy[yyy<mhalfsize]+size
                zzz[zzz>halfsize]=zzz[zzz>halfsize]-size
                zzz[zzz<mhalfsize]=zzz[zzz<mhalfsize]+size

                for kbin,(xi,yi,zi,li) in enumerate(zip(xxx,yyy,zzz,lev_integral)):
                    ip,i,j,k=locate_point(xi,yi,zi,npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,li)
                    ll=levels[ip]
                    dxx=(xi-(patchrx[ip]+(i-0.5)*(size/nmax/2**ll)))/(size/nmax/2**ll)
                    dyy=(yi-(patchry[ip]+(j-0.5)*(size/nmax/2**ll)))/(size/nmax/2**ll)
                    dzz=(zi-(patchrz[ip]+(k-0.5)*(size/nmax/2**ll)))/(size/nmax/2**ll)
                    #assert 0 <= dxx <= 1
                    #assert 0 <= dyy <= 1
                    #assert 0 <= dzz <= 1
                    if ip==0 and (i<=0 or j<=0 or k<=0 or i>=nmax-1 or j>=nmax-1 or k>=nmax-1):
                        dir_profiles[itheta,jphi,kbin]=field[ip][(i  )%nmax,(j  )%nmax,(k  )%nmax] *(1-dxx)*(1-dyy)*(1-dzz) \
                                                     + field[ip][(i  )%nmax,(j  )%nmax,(k+1)%nmax] *(1-dxx)*(1-dyy)*  dzz   \
                                                     + field[ip][(i  )%nmax,(j+1)%nmax,(k  )%nmax] *(1-dxx)*  dyy  *(1-dzz) \
                                                     + field[ip][(i  )%nmax,(j+1)%nmax,(k+1)%nmax] *(1-dxx)*  dyy  *  dzz   \
                                                     + field[ip][(i+1)%nmax,(j  )%nmax,(k  )%nmax] *  dxx  *(1-dyy)*(1-dzz) \
                                                     + field[ip][(i+1)%nmax,(j  )%nmax,(k+1)%nmax] *  dxx  *(1-dyy)*  dzz   \
                                                     + field[ip][(i+1)%nmax,(j+1)%nmax,(k  )%nmax] *  dxx  *  dyy  *(1-dzz) \
                                                     + field[ip][(i+1)%nmax,(j+1)%nmax,(k+1)%nmax] *  dxx  *  dyy  *  dzz  
                    else:
                        dir_profiles[itheta,jphi,kbin]=field[ip][i,j,k]      *(1-dxx)*(1-dyy)*(1-dzz) \
                                                     + field[ip][i,j,k+1]    *(1-dxx)*(1-dyy)*  dzz   \
                                                     + field[ip][i,j+1,k]    *(1-dxx)*  dyy  *(1-dzz) \
                                                     + field[ip][i,j+1,k+1]  *(1-dxx)*  dyy  *  dzz   \
                                                     + field[ip][i+1,j,k]    *  dxx  *(1-dyy)*(1-dzz) \
                                                     + field[ip][i+1,j,k+1]  *  dxx  *(1-dyy)*  dzz   \
                                                     + field[ip][i+1,j+1,k]  *  dxx  *  dyy  *(1-dzz) \
                                                     + field[ip][i+1,j+1,k+1]*  dxx  *  dyy  *  dzz  
    else:
        for itheta,costheta in tqdm(enumerate(vec_costheta),total=len(vec_costheta),disable=not use_tqdm):
            for jphi,phi in enumerate(vec_phi):
                xxx=cx+rrr*np.sqrt(1-costheta**2)*np.cos(phi)
                yyy=cy+rrr*np.sqrt(1-costheta**2)*np.sin(phi)
                zzz=cz+rrr*costheta

                for kbin,(xi,yi,zi,li) in enumerate(zip(xxx,yyy,zzz,lev_integral)):
                    ip,i,j,k=locate_point(xi,yi,zi,npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,li,buf=0,no_interp=True)
                    if ip != 0:
                        dir_profiles[itheta,jphi,kbin]=field[ip][i,j,k]
                    else:
                        dir_profiles[itheta,jphi,kbin]=field[ip][i%nmax,j%nmax,k%nmax]
                    
    return dir_profiles, rrr, vec_costheta, vec_phi


def radial_profile(field,cx,cy,cz,
                   npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,
                   weight_field=None,
                   binsr=None,dex_rbins=None,delta_rbins=None,rmin=None,rmax=None,
                   maxl=None,
                   interpolate=True, average="mean", Ncostheta=20, Nphi=20, use_tqdm=False):
    """
    Computes the radially-average profile of a field around a given center (cx,cy,cz).
    The bins can be specified in three ways:
        - binsr: numpy vector specifying the radial bin edges
        - rmin, rmax, dex_rbins: minimum and maximum radius, and logarithmic bin size
        - rmin, rmax, delta_rbins: minimum and maximum radius, and linear bin size
    The profile can be computed by nearest neighbour interpolation (interpolate=False) or by averaging the values of the cells in each bin (interpolate=True).
    In order to compute the radially-averaged profile, directional profiles are computed and then combined using arithmetic average, geometric average or median.

    Parameters:
        - field: field to compute the profile of
        - cx,cy,cz: center of the profile
        - npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax: patch information
        - weight_field (optional): weighting field to use to normalize the profile
        One and only one of these sets of arguments must be specified:
            - binsr: numpy vector specifying the radial bins
            - rmin, rmax, dex_rbins: minimum and maximum radius, and logarithmic bin size
            - rmin, rmax, delta_rbins: minimum and maximum radius, and linear bin size
        - interpolate: whether to interpolate the field values or not
        - average: type of average to use to combine the directional profiles. Can be "mean", "median" or "geometric"
        - Ncostheta: number of bins in the cos(theta) direction
        - Nphi: number of bins in the phi direction
        - use_tqdm: whether to use tqdm to display a progress bar or not
        - maxl: maximum level to be considered. If None, all levels are considered.
    Returns:
        - profile: radially-averaged profile
        - rrr: radial bins
    """

    dir_profiles, rrr, vec_costheta, vec_phi = dir_profile(field,cx,cy,cz,
                                                           npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,
                                                           binsr=binsr,dex_rbins=dex_rbins,delta_rbins=delta_rbins,rmin=rmin,rmax=rmax,
                                                           interpolate=interpolate, 
                                                           maxl=maxl,
                                                           binscostheta=Ncostheta, binsphi=Nphi, use_tqdm=use_tqdm)

    if weight_field is not None:
        dir_profiles_weight, rrr, vec_costheta, vec_phi = dir_profile(weight_field,cx,cy,cz,
                                                            npatch,patchrx,patchry,patchrz,patchnx,patchny,patchnz,size,nmax,
                                                            binsr=binsr,dex_rbins=dex_rbins,delta_rbins=delta_rbins,rmin=rmin,rmax=rmax,
                                                            interpolate=interpolate, 
                                                            maxl=maxl,
                                                            binscostheta=Ncostheta, binsphi=Nphi, use_tqdm=use_tqdm)

    # Combine directional profiles
    if weight_field is None:
        if average=="mean":
            profile=np.mean(dir_profiles,axis=(0,1))
        elif average=="median":
            profile=np.median(dir_profiles,axis=(0,1))
        elif average=="geometric":
            profile=np.exp(np.mean(np.log(dir_profiles),axis=(0,1)))
        elif type(average) is list or type(average) is np.ndarray: # list of percentiles
            assert all([0<=p<=100 for p in average]), "Percentiles must be between 0 and 100"
            profile={'perc'+str(p): np.percentile(dir_profiles, p, axis=(0,1)) for p in average}
        else:
            raise ValueError('Wrong specification of average')
    else: # weight_field is not None --> weighted average
        if average=="mean":
            profile = np.mean(dir_profiles * dir_profiles_weight, axis=(0,1)) / np.mean(dir_profiles_weight, axis=(0,1))
        elif average=="median":
            profile = stats.weighted_median(dir_profiles, dir_profiles_weight, axes=(0,1))
        elif average=="geometric":
            profile = np.exp(np.mean(np.log(dir_profiles * dir_profiles_weight), axis=(0,1)) / np.mean(np.log(dir_profiles_weight), axis=(0,1)))
        elif type(average) is list or type(average) is np.ndarray: # list of percentiles
            assert all([0<=p<=100 for p in average]), "Percentiles must be between 0 and 100"
            profile={'perc'+str(p): stats.weighted_percentile(dir_profiles, dir_profiles_weight, p, axes=(0,1)) for p in average}
        else:
            raise ValueError('Wrong specification of average')

    return profile, rrr

def build_kdtree_displaced(xpart,ypart,zpart, size):
    """ 
    Returns a KDTree object for the particle distribution, displaced by size/2 in each direction,
    so that the periodic boundary conditions are taken into account.

    Pararameters:
        - xpart,ypart,zpart: particle positions
        - size: size of the simulation box
    
    Returns:
        - tree: KDTree object
    """

    data=np.array([xpart,ypart,zpart]).T+size/2
    data[data>=size]=data[data>=size]-size

    tree=KDTree(data,boxsize=np.array([size,size,size]))

    return tree

def dir_profile_particles(particles_field, cx,cy,cz,size, tree=None,
                binsphi=None, binscostheta=None, binsr=None, rmin=None, rmax=None, dex_rbins=None, delta_rbins=None,
                num_neigh=64, force_resol=None, normalize='volume', weight=None,
                use_tqdm=False):
    """
    Computes the directional profile of a given field defined by a set of particles around a given center (cx,cy,cz). 
    To produce smooth profiles, around each point we use the largest of the following kernels:
        - The radius of the sphere enclosing at least, num_neigh particles. 
        - Force resolution, if specified. 
        - Bin spacing.
    There are several ways to specify the directions along which the profile is computed:
        - binsphi, binscostheta: numpy vectors specifying the direcitons in the spherical angles phi, cos(theta).
        - Nphi, Ncostheta: number of bins in phi and cos(theta), equally spaced between -pi and pi, and -1 and 1, respectively.
    Likewise, there are several ways to specify the radial bins:
        - binsr: numpy vector specifying the radial bin edges
        - rmin, rmax, dex_rbins: minimum and maximum radius, and logarithmic bin size
        - rmin, rmax, delta_rbins: minimum and maximum radius, and linear bin size

    Parameters:
        - particles_field: particle field to compute the profile of. Must be a one-dimensional numpy array.
            Warning! Particles ought to be sorted as the particles positions used to build the tree with build_kdtree_displaced.
        - cx,cy,cz: center of the profile
        - size: size of the simulation box
        - tree: KDTree object for the particle distribution. If None, it is built using build_kdtree_displaced.
        One and only one of these pairs of arguments must be specified:
            - binsphi, binscostheta: numpy vectors specifying the direcitons in the spherical angles phi, cos(theta).
            - Nphi, Ncostheta: number of bins in phi and cos(theta), equally spaced between -pi and pi, and -1 and 1, respectively.
        One and only one of these sets of arguments must be specified:
            - binsr: numpy vector specifying the radial bins
            - rmin, rmax, dex_rbins: minimum and maximum radius, and logarithmic bin size
            - rmin, rmax, delta_rbins: minimum and maximum radius, and linear bin size
        - num_neigh: the value of profile around any point uses, at least, num_neigh particles.
        - force_resol: force resolution. If specified, the value of profile around any point uses, at least, the particles within a 
            sphere of radius force_resol.
        - normalize: whether to normalize the profile by the volume of the bins ("volume") by the number of particles ("number") or
            by a weighting field ("weight"). If "weight" is specified, the argument "weight" must be specified as well.
        - weight: weighting field to use to normalize the profile. Must be a one-dimensional numpy array, the same size as particles_field.
            Only used if normalize="weight".
        - use_tqdm: whether to use tqdm to show a progress bar or not. Default: False.

    Returns:
        - dir_profiles: directional profiles
        - rrr: radial bins
        - vec_costheta: cos(theta) bins
        - vec_phi: phi bins
    """
    # Check phi bins are properly specified
    if type(binsphi) is np.ndarray or type(binsphi) is list:
        Nphi=len(binsphi)
        vec_phi=binsphi
        if type(binsphi) is list:
            vec_phi=np.array(vec_phi)
    elif type(binsphi) is int:
        Nphi=binsphi
        vec_phi=np.linspace(-np.pi + np.pi/Nphi,np.pi -np.pi/Nphi,Nphi)
    else:
        raise ValueError('Wrong specification of binsphi')
        
    # Check theta bins are properly specified
    if type(binscostheta) is np.ndarray or type(binscostheta) is list:
        Ncostheta=len(binscostheta)
        vec_costheta=binscostheta
        if type(binscostheta) is list:
            vec_costheta=np.array(vec_costheta)
    elif type(binscostheta) is int:
        Ncostheta=binscostheta
        vec_costheta=np.linspace(-1+1/Ncostheta,1-1/Ncostheta,Ncostheta)
    else:
        raise ValueError('Wrong specification of binscostheta')  
        
    # Check r bins are properly specified
    if type(binsr) is np.ndarray or type(binsr) is list:
        num_bins=len(binsr)
        rrr=binsr
        if type(binsr) is list:
            rrr=np.array(rrr)
    elif (rmin is not None) and (rmax is not None) and ((dex_rbins is not None) or (delta_rbins is not None)) and ((dex_rbins is None) or (delta_rbins is None)):
        if dex_rbins is not None:
            num_bins=int(np.log10(rmax/rmin)/dex_rbins/2)*2+1 # guarantee it is odd
            rrr = np.logspace(np.log10(rmin),np.log10(rmax),num_bins)
        else:
            num_bins=int((rmax-rmin)/delta_rbins/2)*2+1 # guarantee it is odd
            rrr = np.linspace(rmin,rmax,num_bins)
    else:
        raise ValueError('Wrong specification of binsr') 

    # Check normalization is properly specified
    if normalize not in ['volume','number','weight']:
        raise ValueError('Wrong specification of normalize')
    elif normalize is 'weight' and (type(weight) is not np.ndarray or weight.size != particles_field.size):
        raise ValueError('Wrong specification of weight')

    # Check tree is properly specified
    if type(tree) is not KDTree:
        raise ValueError('Wrong specification of tree')

    if force_resol is None:
        force_resol = 0. # Irrelevant for computations

    # Initialize profile
    dir_profiles = np.zeros((Ncostheta,Nphi,num_bins))
    norma = np.zeros((Ncostheta,Nphi,num_bins))

    drrr=np.concatenate([[rrr[1]-rrr[0]], np.diff(rrr)])
    halfsize=size/2
    mhalfsize=-halfsize

    # Compute profile
    for itheta,costheta in tqdm(enumerate(vec_costheta),total=len(vec_costheta),disable=not use_tqdm):
        for jphi,phi in enumerate(vec_phi):
            xxx=cx+rrr*np.sqrt(1-costheta**2)*np.cos(phi)
            yyy=cy+rrr*np.sqrt(1-costheta**2)*np.sin(phi)
            zzz=cz+rrr*costheta

            # Periodic boundary conditions
            xxx[xxx>halfsize]=xxx[xxx>halfsize]-size
            xxx[xxx<mhalfsize]=xxx[xxx<mhalfsize]+size
            yyy[yyy>halfsize]=yyy[yyy>halfsize]-size
            yyy[yyy<mhalfsize]=yyy[yyy<mhalfsize]+size
            zzz[zzz>halfsize]=zzz[zzz>halfsize]-size
            zzz[zzz<mhalfsize]=zzz[zzz<mhalfsize]+size
            
            for kbin,(x,y,z) in enumerate(zip(xxx,yyy,zzz)):
                radius=max([drrr[kbin], force_resol])
                particles=tree.query_ball_point((x+halfsize,y+halfsize,z+halfsize),radius)
                if len(particles)<num_neigh:
                    particles=tree.query((x+halfsize,y+halfsize,z+halfsize),num_neigh)
                    radius=particles[0][num_neigh-1]
                    particles=particles[1]
                
                if normalize=='volume':
                    dir_profiles[itheta,jphi,kbin]=particles_field[particles].sum()
                    norma[itheta,jphi,kbin]=4*np.pi/3*radius**3
                elif normalize=='number':
                    dir_profiles[itheta,jphi,kbin]=particles_field[particles].sum()
                    norma[itheta,jphi,kbin]=len(particles)
                elif normalize=='weight':
                    dir_profiles[itheta,jphi,kbin]=(weight[particles]*particles_field[particles]).sum()
                    norma[itheta,jphi,kbin]=weight[particles].sum()
    
    dir_profiles /= norma 

    return dir_profiles, rrr, vec_costheta, vec_phi

def radial_profile_particles(particles_field, cx,cy,cz,size, tree=None,
                binsr=None, rmin=None, rmax=None, dex_rbins=None, delta_rbins=None,
                num_neigh=64, force_resol=None, normalize='volume', weight=None,
                use_tqdm=False, average="mean", Ncostheta=20, Nphi=20):
    """
    Computes the radially-average profile of a field defined by a set of particles around a given center (cx,cy,cz). 
    The bins can be specified in three ways:
        - binsr: numpy vector specifying the radial bin edges
        - rmin, rmax, dex_rbins: minimum and maximum radius, and logarithmic bin size
        - rmin, rmax, delta_rbins: minimum and maximum radius, and linear bin size
    The profile can be computed by nearest neighbour interpolation (interpolate=False) or by averaging the values of the cells in each bin (interpolate=True).
    In order to compute the radially-averaged profile, directional profiles are computed and then combined using arithmetic average, geometric average or median.

    Parameters:
        - particles_field: particle field to compute the profile of. Must be a one-dimensional numpy array.
            Warning! Particles ought to be sorted as the particles positions used to build the tree with build_kdtree_displaced.
        - cx,cy,cz: center of the profile
        - size: size of the simulation box
        - tree: KDTree object for the particle distribution. If None, it is built using build_kdtree_displaced.
        One and only one of these sets of arguments must be specified:
            - binsr: numpy vector specifying the radial bins
            - rmin, rmax, dex_rbins: minimum and maximum radius, and logarithmic bin size
            - rmin, rmax, delta_rbins: minimum and maximum radius, and linear bin size
        - num_neigh: the value of profile around any point uses, at least, num_neigh particles.
        - force_resol: force resolution. If specified, the value of profile around any point uses, at least, the particles within a 
            sphere of radius force_resol.
        - normalize: whether to normalize the profile by the volume of the bins ("volume") by the number of particles ("number") or
            by a weighting field ("weight"). If "weight" is specified, the argument "weight" must be specified as well.
        - weight: weighting field to use to normalize the profile. Must be a one-dimensional numpy array, the same size as particles_field.
            Only used if normalize="weight".
        - use_tqdm: whether to use tqdm to show a progress bar or not. Default: False.
        - average: type of average to use to combine the directional profiles. Must be "mean", "median" or "geometric".
        - Ncostheta: number of bins in the cos(theta) direction.
        - Nphi: number of bins in the phi direction.
    Returns:
        - profile: radially-averaged profile
        - rrr: radial bins
    """

    dir_profiles, rrr, vec_costheta, vec_phi = dir_profile_particles(particles_field, cx, cy, cz, size, tree=tree,
                binsr=binsr, rmin=rmin, rmax=rmax, dex_rbins=dex_rbins, delta_rbins=delta_rbins,
                num_neigh=num_neigh, force_resol=force_resol, normalize=normalize, weight=weight,
                use_tqdm=use_tqdm, binsphi=Nphi, binscostheta=Ncostheta)

    # Combine directional profiles
    if average=="mean":
        profile=np.mean(dir_profiles,axis=(0,1))
    elif average=="median":
        profile=np.median(dir_profiles,axis=(0,1))
    elif average=="geometric":
        profile=np.exp(np.mean(np.log(dir_profiles),axis=(0,1)))
    elif type(average) is list or type(average) is np.ndarray: # list of percentiles
        assert all([0<=p<=100 for p in average]), "Percentiles must be between 0 and 100"
        profile={'perc'+str(p): np.percentile(dir_profiles, p, axis=(0,1)) for p in average}
    else:
        raise ValueError('Wrong specification of average')

    return profile, rrr


def integrate_profile_volume(rprof, prof, rmin=None, rmax=None, cumulative=False):
    '''
    Compute the volume integral of the profile 'prof', defined on the 
     radial grid 'rprof', from rmin to rmax.

    Args:
        - rprof: radial grid
        - prof: profile to integrate
        - rmin: minimum radius
        - rmax: maximum radius
        - cumulative: if True, return the cumulative integral as a function of 
            the input radial grid, only within rmin and rmax.
    
    Returns:
        - integral of the profile
    '''
    if rmin is None:
        rmin = rprof[0]
    if rmax is None:
        rmax = rprof[-1]

    mask = (rprof >= rmin) & (rprof <= rmax)

    if not cumulative: # Total integral
        return (4*np.pi) * np.trapz(prof[mask] * rprof[mask]**2, x=rprof[mask])
    else: # Cumulative integral
        return (4*np.pi) * cumulative_trapezoid(prof[mask] * rprof[mask]**2, x=rprof[mask], initial=0)
        

def average_from_profile(rprof, prof, prof_weight=None, rmin=None, rmax=None, cumulative=False):
    '''
    Compute the weighted volume integral of the profile 'prof', defined on the 
     radial grid 'rprof', from rmin to rmax. That is to say, it returns:

        \int_{rmin}^{rmax} prof(r) * prof_weight(r) * 4*pi*r^2 dr 
        ---------------------------------------------------------
              \int_{rmin}^{rmax} prof_weight(r) * 4*pi*r^2 dr

    Args:
        - rprof: radial grid
        - prof: profile to integrate
        - prof_weight: weight profile. If not specified, the weight is 1 (volume-weighted)
        - rmin: minimum radius
        - rmax: maximum radius
        - cumulative: if True, return the cumulative integral as a function of 
            the input radial grid, only within rmin and rmax.
    
    Returns:
        - weighted integral of the profile
    '''
    if rmin is None:
        rmin = rprof[0]
    if rmax is None:
        rmax = rprof[-1]
    if prof_weight is None:
        prof_weight = np.ones_like(prof)

    num = integrate_profile_volume(rprof, prof * prof_weight, rmin=rmin, rmax=rmax, cumulative=cumulative)
    den = integrate_profile_volume(rprof, prof_weight, rmin=rmin, rmax=rmax, cumulative=cumulative)

    if cumulative:
        if num[0]==0. and den[0]==0.:
            den[0] = 1.
            num[0] = num[1]/den[1] #?

    return num / den


def stack_profiles(list_r, list_profiles, stacking_method='median', rmin=None, rmax=None, nbins=None, logbins=True, logloginterp=True, logstack=False,
                   return_list_profiles=False):
    """ 
    Stack profiles using a specified method (mean, median, biweight, mode), taking 
    care of resampling the profiles to a common grid.

    Args:
        - list_r: list of radial grids
        - list_profiles: list of profiles to stack
        - stacking_method: method to stack the profiles. Can be 'mean', 'median', 'biweight' or 'mode'.
            Also, a list of percentiles can be specified.
        - rmin: minimum radius (default: maximum of the minimums of the input radial grids)
        - rmax: maximum radius (default: minimum of the maximums of the input radial grids)
        - nbins: number of bins in the common grid (default: length of the input radial grids, where all
                 are assumed to be the same)
        - logbins: whether the common grid should be logarithmic or not
        - logloginterp: whether to interpolate in log-log space or not
        - logstack: whether to stack in log space or not
        - return_list_profiles: instead of stacking, return the list of resampled profiles

    Returns:
        - r: common radial grid
        - stacked_profile: stacked profile
            If return_list_profiles is True, instead of the stacked profile, the list of resampled profiles is returned.
            If stacking_method is a list of percentiles, a dictionary with the percentiles is returned.
    """ 

    # interpolate profiles to a common grid
    if rmin is None:
        rmin = max([min(r) for r in list_r])
    if rmax is None:
        rmax = min([max(r) for r in list_r])
    if nbins is None:
        nbins = len(list_r[0])

    if logbins:
        r = np.logspace(np.log10(rmin), np.log10(rmax), nbins)
    else:
        r = np.linspace(rmin, rmax, nbins)

    #print(rmin, rmax, r)

    for i in range(len(list_profiles)):
        rprof = list_r[i]
        prof = list_profiles[i]
        if logloginterp:
            rprof = np.log10(rprof)
            prof = np.log10(prof)
        interpolant = interp1d(rprof, prof, bounds_error=False, fill_value=(prof[0], prof[-1]))
        
        if logloginterp:
            prof_interp = 10**interpolant(np.log10(r))
        else:
            prof_interp = interpolant(r)

        list_profiles[i] = prof_interp

    if return_list_profiles:
        return r, list_profiles

    #print(list_profiles)
    if logstack:
        list_profiles = [np.log10(p) for p in list_profiles]

    output_is_dict = False
    if stacking_method == 'mean':
        stacked_profile = np.mean(list_profiles, axis=0)
    elif stacking_method == 'median':
        stacked_profile = np.median(list_profiles, axis=0)
    elif stacking_method == 'biweight':
        stacked_profile = biweight_location(list_profiles, axis=0)
    elif stacking_method == 'mode':
        stacked_profile = np.zeros_like(list_profiles[0])
        for i in range(len(list_profiles[0])):
            vals = [x[i] for x in list_profiles]
            testvals = np.linspace(min(vals), max(vals), 1000)
            kde = gaussian_kde(vals)
            stacked_profile[i] = testvals[np.argmax(kde(testvals))]
    elif type(stacking_method) is list or type(stacking_method) is np.ndarray:
        assert all([0<=p<=100 for p in stacking_method]), "Percentiles must be between 0 and 100"
        stacked_profile = {'perc'+str(p): np.percentile(list_profiles, p, axis=0) for p in stacking_method}
        output_is_dict = True

    if logstack:
        stacked_profile = 10**stacked_profile
        if output_is_dict:
            stacked_profile = {k: 10**v for k,v in stacked_profile.items()}
        
    return r, stacked_profile


def project_profile_extensive(prof3d, r3d, R2d=None):
    """ 
    Project an extensive (additive) profile to a 2D plane, by integrating the profile along the line of sight.

    prof2d(R2d) = \int_{r3d} prof3d(r3d) * 2 * r3d * dr3d / sqrt(r3d^2 - R2d^2)

    Args:
        - prof3d: 3D profile to project
        - r3d: 3D radial grid
        - R2d: 2D radial grid. If None, the 2D grid is assumed to be the same as the 3D grid.

    Returns:
        - prof2d: 2D projected profile
    """
    if R2d is None:
        R2d = r3d.copy()

    prof2d = np.zeros_like(R2d, dtype=prof3d.dtype)

    # Compute the radial bin widths (dr)
    delta_r = np.concatenate([[r3d[1]-r3d[0]], (r3d[1:]-r3d[:-1])/2, [r3d[-1]-r3d[-2]]]) 

    # Upper summation limit
    Jmax = prof3d.size
    r3dmax = r3d.max()

    for i in range(prof2d.size):
        Ri = R2d[i]
        if Ri >= r3dmax:
            prof2d[i] = 0
            continue
        Jmin = np.where(r3d > Ri)[0][0]
        prof2d[i] = 2 * (prof3d[Jmin:Jmax] * r3d[Jmin:Jmax] * delta_r[Jmin:Jmax] / np.sqrt(r3d[Jmin:Jmax]**2 - Ri**2)).sum()

    return prof2d

def project_profile_intensive(prof3d, r3d, R2d, weight3d=None):
    """
    Project an intensive (non-additive) profile to a 2D plane, by integrating the profile along the line of sight using a weighting field.

    prof2d(R2d) = \int_{r3d} prof3d(r3d) * weight3d(r3d) * 2 * r3d * dr3d / sqrt(r3d^2 - R2d^2) / \int_{r3d} weight3d(r3d) * 2 * r3d * dr3d / sqrt(r3d^2 - R2d^2)

    Args:
        - prof3d: 3D profile to project
        - r3d: 3D radial grid
        - R2d: 2D radial grid
        - weight3d: weighting field. If None, the integration is unweighted (length-weighted).

    Returns:
        - prof2d: 2D projected profile
    """

    if weight3d is None:
        weight3d = np.ones_like(prof3d)

    num = project_profile_extensive(prof3d * weight3d, r3d, R2d)
    den = project_profile_extensive(weight3d, r3d, R2d)

    return num / den