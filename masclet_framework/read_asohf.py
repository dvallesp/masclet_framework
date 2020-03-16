"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

read_asohf module
Provides the necessary functions to read ASOHF outputs

Created by David Vallés
"""

#  Last update on 16/3/20 23:04

import masclet_framework as masclet
import numpy as np
import os


def filename(it,digits=5):
    """
    Generates filenames for ASOHF output files

    Args:
        it: iteration number (int)
        digits: number of digits the filename is written with (int)

    Returns:
        filename (str)

    """
    if np.floor(np.log10(it)) < digits:
        return 'families' + str(it).zfill(digits)
    else:
        raise ValueError("Digits should be greater to handle that iteration number")


def read_families(it, path='', digits=5, exclude_subhaloes=False, minmass=0):
    """
    Reads the ASOHF familiesXXXXX files, containing the information about each cluster

    Args:
        it: iteration number (int)
        path: path of the families file in the system (str)
        digits: number of digits the filename is written with (int)
        exclude_subhaloes: if True, will only output haloes which are not subhaloes of any structure. Defaults to False.
        minmass: if specified (in Msun), only haloes of mass > minmass will be read

    Returns:
        List of dictionaries, each one containing the information of one halo.
    """
    with open(os.path.join(path, filename(it, digits))) as f:
        f.readline()
        it_asohf, nnclus, konta2, zeta = f.readline().split()
        it_asohf = int(it_asohf)
        assert (it == it_asohf)
        nnclus = int(nnclus)  # esto que es?????
        konta2 = int(konta2)
        zeta = float(zeta)
        f.readline()

        haloes = []

        for line in f:
            line = line.split()
            halo = {
                "id": int(line[0]),
                "rx": float(line[1]),
                "ry": float(line[2]),
                "rz": float(line[3]),
                "mass": float(line[4]),
                "radius": float(line[5]),
                "dm_part_count": int(line[6]),
                "substructure_of": int(line[7]),
                "nlevhal": int(line[8]),
                "subhaloes": int(line[9]),
                "eigenval1": float(line[10]),
                "eigenval2": float(line[11]),
                "eigenval3": float(line[12]),
                "vcm2": float(line[13]),
                "concentra": float(line[14]),
                "angularm": float(line[15]),
                "vcmax": float(line[16]),
                "mcmax": float(line[17]),
                "rcmax": float(line[18]),
                "m200": float(line[19]),
                "r200": float(line[20]),
                "vx": float(line[21]),
                "vy": float(line[22]),
                "vz": float(line[23])
            }
            if ((not exclude_subhaloes) or (halo["substructure_of"] == -1)) and halo["mass"] > minmass:
                haloes.append(halo)

    return haloes
