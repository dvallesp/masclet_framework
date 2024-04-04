"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

Created by David Vallés
"""

#  Last update on 3/4/20 10:29

import masclet_framework.parameters
import masclet_framework.read_masclet
import masclet_framework.tools
import masclet_framework.graphics
import masclet_framework.masclet2yt
import masclet_framework.units
import masclet_framework.cosmo_tools
import masclet_framework.read_halma
import masclet_framework.read_asohf
import masclet_framework.read_old_asohf
import masclet_framework.particles
import masclet_framework.tools_xyz
import masclet_framework.spectral
import masclet_framework.thermal
import masclet_framework.stats
import masclet_framework.simulator_tools
import masclet_framework.profiles
import masclet_framework.selfsimilar
import masclet_framework.diff
import masclet_framework.particle2grid
import masclet_framework.calcAMR
import masclet_framework.stars
import masclet_framework.dynstate
