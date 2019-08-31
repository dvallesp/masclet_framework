#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

parameters module
Functions for reading and creating parameters JSON files

v0.1.0, 30/06/2019
David Vallés, 2019
"""

import json
import os

def read_parameters_file(filename='masclet_parameters.json',path=''):
    '''
    Returns dictionary containing the MASCLET parameters of the simulation,
    that have been previously written with the write_parameters() function in
    this same module.

    PARAMETERS:
    Filename and path of the json file

    RETURNS a dictionary containing:
    NMAX (NMAY, NMAZ): number of cells along each direction
    NPALEV: maximum number of refinement cells per level
    NLEVELS: maximum number of refinement level
    NAMRX (NAMRY, NAMRZ): maximum size of refinement level (in l-1 cell units)
    SIZE: size (in the preferred length unit) of the side of the box
    '''
    filepath = os.path.join(path, filename)
    with open(filepath) as json_file:
        data = json.load(json_file)
    return data

def read_parameters(filename='masclet_parameters.json',path='',loadNMA = True,
                    loadNPALEV = True, loadNLEVELS = True, loadNAMR = True, 
                    loadSIZE = True):
    '''
    Returns MASCLET parameters in the old-fashion way. Legacy (can be used, 
    but newer codes should try to switch to directly reading the dictionary)
    '''
    parameters = read_parameters_file(filename=filename,path=path)
    returnvariables = []
    if loadNMA:
        returnvariables.extend([parameters[i] for i in ['NMAX','NMAY','NMAZ']])
    if loadNPALEV:
        returnvariables.append(parameters['NPALEV'])
    if loadNLEVELS:
        returnvariables.append(parameters['NLEVELS'])
    if loadNAMR:
        returnvariables.extend([parameters[i] for i in ['NAMRX','NAMRY','NAMRZ']])
    if loadSIZE:
        returnvariables.append(parameters['SIZE'])
    return tuple(returnvariables)

def write_parameters(NMAX,NMAY,NMAZ,NPALEV,NLEVELS,NAMRX,NAMRY,NAMRZ,
                     SIZE,filename='masclet_parameters.json',path=''):
    '''
    Creates a JSON file containing the parameters of a certain simulation
    
    PARAMETERS:
    NMAX (NMAY, NMAZ): number of cells along each direction
    NPALEV: maximum number of refinement cells per level
    NLEVELS: maximum number of refinement level
    NAMRX (NAMRY, NAMRZ): maximum size of refinement level (in l-1 cell units)
    SIZE: size (in the preferred length unit) of the side of the box
    filename and path specify where the file will be created
    
    RETURNS:
    Nothing. A file is created in the specified path.
    '''
    parameters = {'NMAX':NMAX, 'NMAY':NMAY, 'NMAZ':NMAZ, 
                  'NPALEV':NPALEV, 'NLEVELS':NLEVELS,
                  'NAMRX':NAMRX, 'NAMRY':NAMRY, 'NAMRZ':NAMRZ,
                  'SIZE':SIZE}
    
    with open(path+filename, 'w') as json_file:  
        json.dump(parameters, json_file)
