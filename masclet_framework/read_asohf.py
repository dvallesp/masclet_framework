import os, sys, numpy as np
from cython_fortran_file import FortranFile as FF

def read_families(it, path='', output_format='dictionaries', output_redshift=False,
                  min_mass=None, exclude_subhaloes=False, read_region=None, keep_boundary_contributions=False):
    '''
    Reads the families files, containing the halo catalogue.
    Can be outputted as a list of dictionaries, one per halo
     (output_format='dictionaries') or as a dictionary of 
     arrays (output_format='arrays').

    Args:
        - it: iteration number (int)
        - path: path of the families file (str)
        - output_format: 'dictionaries' or 'arrays'
        - output_redshift: whether to output the redshift of the snapshot, 
            after the halo catalogue (bool)
        - min_mass: minimum mass of the haloes to be output (float)
        - exclude_subhaloes: whether to exclude subhaloes (bool)
        - read_region: whether to select a subregion (see region specification below), or keep all the simulation data 
            (None). If a region wants to be selected, there are the following possibilities:
            - ("sphere", cx, cy, cz, R) for a sphere of radius R centered in (cx, cy, cz)
            - ("box", x1, x2, y1, y2, z1, z2) for a box with corners (x1, y1, z1) and (x2, y2, z2)
            - ("box_cw", xc, yc, zc, width) for a box centered in (xc, yc, zc) with width "width"
        - keep_boundary_contributions: only if read_region is used, whether to keep haloes that might be incomplete (True)
            or not (False).
    '''
    with open(os.path.join(path, 'families{:05d}'.format(it)), 'r') as f:
        _=f.readline()
        _,_,_,zeta = f.readline().split()
        zeta=float(zeta)
        for i in range(5):
            _=f.readline()
        haloes=[]
        for l in f:
            l=l.split()
            halo={}
            halo['id']=int(l[0])
            halo['substructureOf']=int(l[1])
            halo['x']=float(l[2])
            halo['y']=float(l[3])
            halo['z']=float(l[4])
            halo['Mvir']=float(l[5])
            halo['Rvir']=float(l[6])
            if halo['substructureOf']==-1:
                halo['M']=halo['Mvir']
                halo['R']=halo['Rvir']
            else:
                halo['M']=float(l[7])
                halo['R']=float(l[8])
            halo['partNum']=int(l[9])
            halo['mostBoundPart']=int(l[10])
            halo['xcm']=float(l[11])
            halo['ycm']=float(l[12])
            halo['zcm']=float(l[13])
            halo['majorSemiaxis']=float(l[14])
            halo['intermediateSemiaxis']=float(l[15])
            halo['minorSemiaxis']=float(l[16])
            halo['Ixx']=float(l[17])
            halo['Ixy']=float(l[18])
            halo['Ixz']=float(l[19])
            halo['Iyy']=float(l[20])
            halo['Iyz']=float(l[21])
            halo['Izz']=float(l[22])
            halo['Lx']=float(l[23])
            halo['Ly']=float(l[24])
            halo['Lz']=float(l[25])
            halo['sigma_v']=float(l[26])
            halo['vx']=float(l[27])
            halo['vy']=float(l[28])
            halo['vz']=float(l[29]) 
            halo['max_v']=float(l[30])
            halo['mean_vr']=float(l[31])
            halo['Ekin']=float(l[32])
            halo['Epot']=float(l[33])
            halo['Vcmax']=float(l[34])
            halo['Mcmax']=float(l[35])
            halo['Rcmax']=float(l[36])
            halo['R200m']=float(l[37])
            halo['M200m']=float(l[38])
            halo['R200c']=float(l[39])
            halo['M200c']=float(l[40])
            halo['R500m']=float(l[41])
            halo['M500m']=float(l[42])
            halo['R500c']=float(l[43])
            halo['M500c']=float(l[44])
            halo['R2500m']=float(l[45])
            halo['M2500m']=float(l[46])
            halo['R2500c']=float(l[47])
            halo['M2500c']=float(l[48])
            halo['fsub']=float(l[49])
            halo['Nsubs']=int(l[50])

            haloes.append(halo)
    
    if exclude_subhaloes:
        haloes=[halo for halo in haloes if halo['substructureOf']==-1]
    if min_mass is not None:
        haloes=[halo for halo in haloes if halo['M']>min_mass]

    if read_region is not None:
        kept_haloes = {}
        haloes_temp = [h for h in haloes]
        haloes = []

        region_type = read_region[0]
        if region_type == 'sphere':
            cx, cy, cz, R = read_region[1:]
            
            for h in haloes_temp:
                R_reg = R + h['R'] if keep_boundary_contributions else R - h['R']
                if (h['x']-cx)**2 + (h['y']-cy)**2 + (h['z']-cz)**2 < R_reg**2:
                    haloes.append(h)
                    kept_haloes[h['id']] = True
                else:
                    kept_haloes[h['id']] = False

        elif region_type == 'box' or region_type == 'box_cw':
            if region_type == 'box':
                x1, x2, y1, y2, z1, z2 = read_region[1:]
            else:
                xc, yc, zc, width = read_region[1:]
                x1 = xc - width/2
                x2 = xc + width/2
                y1 = yc - width/2
                y2 = yc + width/2
                z1 = zc - width/2
                z2 = zc + width/2

            for h in haloes_temp:
                xh1 = x1 - h['R'] if keep_boundary_contributions else x1 + h['R']
                xh2 = x2 + h['R'] if keep_boundary_contributions else x2 - h['R']
                yh1 = y1 - h['R'] if keep_boundary_contributions else y1 + h['R']
                yh2 = y2 + h['R'] if keep_boundary_contributions else y2 - h['R']
                zh1 = z1 - h['R'] if keep_boundary_contributions else z1 + h['R']
                zh2 = z2 + h['R'] if keep_boundary_contributions else z2 - h['R']

                if xh1 < h['x'] < xh2 and yh1 < h['y'] < yh2 and zh1 < h['z'] < zh2:
                    haloes.append(h)
                    kept_haloes[h['id']] = True
                else:
                    kept_haloes[h['id']] = False

        else:
            raise ValueError('Unknown region type. Please specify one of "sphere", "box" or "box_cw"')

    return_variables = []
    if output_format=='dictionaries':
        return_variables.append(haloes)
    elif output_format=='arrays':
        return_variables.append({k: np.array([h[k] for h in haloes]) for k in haloes[0].keys()})
    else:
        raise ValueError('Error! output_format argument should be either dictionaries or arrays')

    if output_redshift:
        return_variables.append(zeta)

    if read_region is not None:
        return_variables.append(kept_haloes)

    if len(return_variables) == 1:
        return return_variables[0]
    else:
        return tuple(return_variables)


def read_stellar_haloes(it, path='', output_format='dictionaries', read_region=None, keep_boundary_contributions=False):
    '''
    Reads the stellar haloes files, containing the halo catalogue.
    Can be outputted as a list of dictionaries, one per halo
     (output_format='dictionaries') or as a dictionary of 
     arrays (output_format='arrays').

    Args:
        - it: iteration number (int)
        - path: path of the stellar_haloes file (str)
        - output_format: 'dictionaries' or 'arrays'
        - read_region: whether to select a subregion (see region specification below), or keep all the simulation data 
            (None). If a region wants to be selected, there are the following possibilities:
            - ("sphere", cx, cy, cz, R) for a sphere of radius R centered in (cx, cy, cz)
            - ("box", x1, x2, y1, y2, z1, z2) for a box with corners (x1, y1, z1) and (x2, y2, z2)
            - ("box_cw", xc, yc, zc, width) for a box centered in (xc, yc, zc) with width "width"
        - keep_boundary_contributions: only if read_region is used, whether to keep haloes that might be incomplete (True)
    '''
    with open(os.path.join(path, 'stellar_haloes{:05d}'.format(it)), 'r') as f:
        for i in range(7):
            f.readline()
        haloes=[]
        for l in f:
            l=l.split()
            halo={}
            halo['id']=int(l[0])
            halo['DMid']=int(l[1])
            halo['xDM']=float(l[2])
            halo['yDM']=float(l[3])
            halo['zDM']=float(l[4])
            halo['x']=float(l[5])
            halo['y']=float(l[6])
            halo['z']=float(l[7]) 
            halo['Mhalf']=float(l[8])
            halo['Rhalf']=float(l[9])
            halo['numpart']=int(l[10])
            halo['xcm']=float(l[11])
            halo['ycm']=float(l[12])
            halo['zcm']=float(l[13])
            halo['majorSemiaxis']=float(l[14])
            halo['intermediateSemiaxis']=float(l[15])
            halo['minorSemiaxis']=float(l[16])
            halo['Ixx']=float(l[17])
            halo['Ixy']=float(l[18])
            halo['Ixz']=float(l[19])
            halo['Iyy']=float(l[20])
            halo['Iyz']=float(l[21])
            halo['Izz']=float(l[22])
            halo['Lx']=float(l[23])
            halo['Ly']=float(l[24])
            halo['Lz']=float(l[25])
            halo['sigma_v']=float(l[26])
            halo['vx']=float(l[27])
            halo['vy']=float(l[28])
            halo['vz']=float(l[29]) 

            haloes.append(halo)

    if read_region is not None:
        kept_haloes = {}
        haloes_temp = [h for h in haloes]
        haloes = []

        region_type = read_region[0]
        if region_type == 'sphere':
            cx, cy, cz, R = read_region[1:]
            
            for h in haloes_temp:
                R_reg = R + h['Rhalf'] if keep_boundary_contributions else R - h['Rhalf']
                if (h['x']-cx)**2 + (h['y']-cy)**2 + (h['z']-cz)**2 < R_reg**2:
                    haloes.append(h)
                    kept_haloes[h['id']] = True
                else:
                    kept_haloes[h['id']] = False

        elif region_type == 'box' or region_type == 'box_cw':
            if region_type == 'box':
                x1, x2, y1, y2, z1, z2 = read_region[1:]
            else:
                xc, yc, zc, width = read_region[1:]
                x1 = xc - width/2
                x2 = xc + width/2
                y1 = yc - width/2
                y2 = yc + width/2
                z1 = zc - width/2
                z2 = zc + width/2

            for h in haloes_temp:
                xh1 = x1 - h['Rhalf'] if keep_boundary_contributions else x1 + h['Rhalf']
                xh2 = x2 + h['Rhalf'] if keep_boundary_contributions else x2 - h['Rhalf']
                yh1 = y1 - h['Rhalf'] if keep_boundary_contributions else y1 + h['Rhalf']
                yh2 = y2 + h['Rhalf'] if keep_boundary_contributions else y2 - h['Rhalf']
                zh1 = z1 - h['Rhalf'] if keep_boundary_contributions else z1 + h['Rhalf']
                zh2 = z2 + h['Rhalf'] if keep_boundary_contributions else z2 - h['Rhalf']

                if xh1 < h['x'] < xh2 and yh1 < h['y'] < yh2 and zh1 < h['z'] < zh2:
                    haloes.append(h)
                    kept_haloes[h['id']] = True
                else:
                    kept_haloes[h['id']] = False

        else:
            raise ValueError('Unknown region type. Please specify one of "sphere", "box" or "box_cw"')
    
    return_variables = []
    if output_format=='dictionaries':
        return_variables.append(haloes)
    elif output_format=='arrays':
        return_variables.append({k: np.array([h[k] for h in haloes]) for k in haloes[0].keys()})

    if read_region is not None:
        return_variables.append(kept_haloes)

    if len(return_variables) == 1:
        return return_variables[0]
    else:
        return tuple(return_variables)


def read_particles(it, path='', parttype='DM', sort='oripa', kept_haloes=None):
    '''
    Reads the particles list. Set parttype='DM' for DM particles,
    'stellar' for stellar particles.
    - sort: 'oripa' (sort particles by increasing ID) or 'r' (sort by 
       increasing radius).

    If the partial reader has been used, the kept_haloes argument 
     can be used to select which haloes to read the particles of. 
     It is outputted by the read_families or read_stellar_haloes 
        functions.
    '''
    if parttype=='DM':
        filename='particles'
    elif parttype=='stellar':
        filename='stellar_particles'

    particles_oripa={}
    particles_lut={}
    with FF(os.path.join(path, filename+'{:05d}'.format(it)), 'r') as f:
        nhaloes=f.read_vector('i4')[0]
        for i in range(nhaloes):
            nclus,i1,i2=f.read_vector('i4')
            particles_lut[nclus]=[i1-1,i2-1]
        f.read_vector('i4')
        particles=f.read_vector('i4')

    if kept_haloes is None:
        kept_haloes = {hid: True for hid in particles_lut.keys()}
    
    if sort=='oripa':
        for k,(i1,i2) in particles_lut.items():
            if kept_haloes[k]:
                particles_oripa[k]=np.sort(particles[i1:i2+1])
    elif sort=='r':
        for k,(i1,i2) in particles_lut.items():
            if kept_haloes[k]:
                particles_oripa[k]=particles[i1:i2+1]
    else:
        raise ValueError('Error! sort argument should be either oripa or r')

    return particles_oripa

