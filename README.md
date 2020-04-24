# MASCLET Framework

This library implements some functions which can be used to handle the outputs from the cosmological code MASCLET (Quilis, 2004). Do not hesitate to contact the creator of the library (David Vallés) for any troubles you might encounter or suggestions you might have regarding this library.

In the following lines, we add some brief information about each of the modules of the library. 

## Installation
The modules can be imported independently,

```python
from masclet_framework import read_masclet
```

or all at once,

```python
import masclet_framework as masclet
```

in which case the modules can be accessed by masclet.read_masclet, for instance.

Our recommendation is to clone this repository and move the folder masclet_framework to the top level of your src folder (or equivalent). Then, you can import the library just by adding the src folder to your sys.path:

```python
import sys
sys.path.append('/path/to/the/folder/containing/masclet_framework')
```

## Modules description

### read_masclet
Contains the necessary functions to read MASCLET output files (gas, dark matter and stars files), as well as some other utilities.

### parameters
Contains functions in order to write a json file with the parameters of the simulation and read them in other functions / in your own scripts.

### tools
Contains several functions which can become handy in the daily work with these simulations. From cleaning the fields from overlaps and refinements to build uniform grids from the AMR structure of MASCLET.

### tools_xyz
Contains functions to compute the coordinates of each cell in the simulation, which can be extremely useful to build radial profiles and compute enclosed masses. Functions for these precise purposes are also included. Several other calculations, like computing angular momenta, shape tensors, etc. can also be performed with this library.

### particles
Contains functions to deal with particle species (either dark matter or stars), which could indeed be easily applicable to other simulation codes. 

### units
Supplies easy unit conversion from MASCLET internal units to cgs, ISU and astrophysical units.

### cosmo_tools
Includes several essential cosmological computations (like cosmological time from redshift, background and critical densities, etc. Also includes a function to write and read a cosmological parameters file for a simulation, analogously to what can be done with the parameters module.

### graphics
Includes several functions for straightforward representations, like computing projections from uniform grids and plotting colormaps (using matplotlib).

### masclet2yt
Provides a framework from MASCLET outputs to the yt-package, in order to load the grid structure and fields into it. Although this function has not been thoroughly tested, this can be quite useful to perform more advanced representations (volume renders, for example). 

### read_asohf
Includes functions for reading the outputs from the halo finder ASOHF (Planelles & Quilis, 2010).
