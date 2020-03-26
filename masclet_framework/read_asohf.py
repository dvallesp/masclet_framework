"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

read_asohf module
Provides the necessary functions to read ASOHF outputs

Created by David Vallés
"""

#  Last update on 26/3/20 11:06

import masclet_framework as masclet
import numpy as np
import os


def filename(it, filetype, digits=5):
    """
    Generates filenames for ASOHF output files

    Args:
        it: iteration number (int)
        filetype: 'f' for families file; 'm' for the merger tree (str)
        digits: number of digits the filename is written with (int)

    Returns:
        filename (str)

    """
    names = {'f': "families", 'm': 'merger_t'}
    if np.floor(np.log10(it)) < digits:
        return names[filetype] + str(it).zfill(digits)
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
    with open(os.path.join(path, filename(it, 'f', digits))) as f:
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


def read_merger_tree(it, path='', digits=5):
    """
    Reads the ASOHF merger_tXXXXX files, containing the information about the merger tree of the DM haloes

    Args:
        it: iteration number (int)
        path: path of the families file in the system (str)
        digits: number of digits the filename is written with (int)

    Returns:
        A list of dictionaries, each entry of the list being a cluster in the iteration it+EVERY. For each of these
        clusters, besides its mass, position, etc., one finds a field "parents". The value of this key is a list of
        dictionaries, each one containing one of the parent clusters (in the iteration it).
    """
    mtree = []
    with open(os.path.join(path, filename(it, 'm', digits))) as f:
        f.readline()  # first string
        _, _, _, real_haloes = f.readline().split()
        real_haloes = int(real_haloes)
        f.readline()  # previous iteration
        for cluster in range(1, real_haloes + 1):
            f.readline()  # ---- new clus ---- string
            # parent values
            _, haloid, mass, ndad, nr = f.readline().split()
            haloid = int(haloid)
            mass = float(mass)
            ndad = int(ndad)
            nr = float(nr)
            rx, ry, rz, level = f.readline().split()
            rx = float(rx)
            ry = float(ry)
            rz = float(rz)
            level = int(level)

            parents = []
            for dad in range(1, ndad + 1):
                id_dad, ratio, mass_dad, nr_dad = f.readline().split()
                id_dad = int(id_dad)
                ratio = float(ratio)
                mass_dad = float(mass_dad)
                nr_dad = float(nr_dad)
                rx_dad, ry_dad, rz_dad, level_dad = f.readline().split()
                rx_dad = float(rx_dad)
                ry_dad = float(ry_dad)
                rz_dad = float(rz_dad)
                level_dad = int(level_dad)
                parents.append({'id': id_dad, 'ratio': ratio, 'mass': mass_dad, 'nr': nr_dad,
                                'rx': rx_dad, 'ry': ry_dad, 'rz': rz_dad, 'level': level_dad})

            mtree.append({'id': haloid, 'mass': mass, 'ndad': ndad, 'nr': nr,
                          'rx': rx, 'ry': ry, 'rz': rz, 'level': level, 'parents': parents})

    return mtree


def read_merger_tree_essentials(it, path='', digits=5):
    """
    Reads the ASOHF merger_tXXXXX files, containing the information about the merger tree of the DM haloes, and returns
    a "essentials" version: only the IDs of parents and ratios

    Args:
        it: iteration number (int)
        path: path of the families file in the system (str)
        digits: number of digits the filename is written with (int)

    Returns:
        A dictionary, each entry of which being a cluster in the iteration it+EVERY. For each of these
        clusters, one finds a list "parents" (containing the IDs of the parents), a list "ratios" (containing the
        ratios of parents' mass which have gone to the son cluster), and a list "masses" (containing the parents masses)
    """
    mtree = read_merger_tree(it, path=path, digits=digits)

    essentials = {}

    for cluster in mtree:
        id = cluster["id"]
        #ndad = cluster["ndad"]
        parents = []
        ratios = []
        masses = []
        for parent in cluster["parents"]:
            parents.append(parent["id"])
            ratios.append(parent["ratio"])
            masses.append(parent["mass"])
        essentials[id] = {'mass': cluster["mass"], 'parents': parents, 'ratios': ratios, 'masses': masses}

    return essentials


def read_merger_tree_reduced(it, path='', digits=5):
    """
        Reads the ASOHF merger_tXXXXX files, containing the information about the merger tree of the DM haloes, and
        returns the reduced merger tree: for each cluster, the progenitor which has given the most mass is given

        Args:
            it: iteration number (int)
            path: path of the families file in the system (str)
            digits: number of digits the filename is written with (int)

        Returns:
            A dictionary, each entry of which being a cluster in the iteration it+EVERY. For each of these
            clusters, the ID of the parent and the ratio of mass given.
        """
    essentials = read_merger_tree_essentials(it, path=path, digits=digits)

    reduced = {}

    for clusterid in essentials:
        cluster = essentials[clusterid]
        if len(cluster["parents"]) > 0:
            givenmass = [cluster["ratios"][i]*cluster["masses"][i] for i in range(len(cluster["parents"]))]
            whichparent_pos = givenmass.index(max(givenmass))
            parent = cluster["parents"][whichparent_pos]
            ratio = cluster["ratios"][whichparent_pos]
        else:
            parent = None
            ratio = None
        reduced[clusterid] = {'parent': parent, 'ratio': ratio}

    return reduced
