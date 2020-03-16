"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

units module
Contains the basic units of the MASCLET code

Created by David Vallés
"""

#  Last update on 16/3/20 19:14

### FUNDAMENTAL UNITS; MASCLET TO PHYSICAL UNITS

mass_to_g = 1.816e52
mass_to_kg = 1.816e49
mass_to_sun = 9.1717e18

length_to_cm = 3.38807679684064e+25
length_to_m = 3.3881e+23
length_to_ly = 3.581e+7
length_to_mpc = 10.98

time_to_s = 1.129e15
time_to_yr = 35.8e6

### FUNDAMENTAL UNITS: PHYSICAL TO MASCLET UNITS
g_to_mass = 1/mass_to_g
kg_to_mass = 1/mass_to_kg
sun_to_mass = 1/mass_to_sun

cm_to_length = 1/length_to_cm
m_to_length = 1/length_to_m
ly_to_length = 1/length_to_ly
mpc_to_length = 1/length_to_mpc

s_to_time = 1/time_to_s
yr_to_time = 1/time_to_yr


### DERIVED UNITS OF COMMON USE: PHYSICAL TO MASCLET UNITS
cgs_to_density = 2.1416e24


### DERIVED UNITS OF COMMON USE: MASCLET TO PHYSICAL UNITS
density_to_cgs = 1/cgs_to_density
