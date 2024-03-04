"""
MASCLET framework

Provides several functions to read MASCLET outputs and perform basic operations.
Also, serves as a bridge between MASCLET data and the yt package (v 3.5.1).

units module
Contains the basic units of the MASCLET code

Created by David Vallés
"""

#  Last update on 21/3/20 18:04

### FUNDAMENTAL UNITS; MASCLET TO PHYSICAL UNITS

mass_to_g = 1.816e52
mass_to_kg = 1.816e49
mass_to_sun = 9.1717e18

length_to_cm = 3.3881e+25
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

### OTHER USEFUL RELATIONS

kg_to_g = 1000
g_to_kg = 1/kg_to_g
sun_to_kg = 1.98848e30
kg_to_sun = 1/sun_to_kg
sun_to_g = sun_to_kg * kg_to_g
g_to_sun = 1/sun_to_g

m_to_cm = 100
cm_to_m = 1/m_to_cm
mpc_to_m = 3.085678e22
m_to_mpc = 1/mpc_to_m
mpc_to_cm = mpc_to_m * m_to_cm
cm_to_mpc = 1/mpc_to_cm

yr_to_s = 31556736
s_to_yr = 1/yr_to_s

eV_to_J = 1.6022e-19
keV_to_J = 1e3 * eV_to_J
J_to_eV = 1/eV_to_J
J_to_keV = 1/keV_to_J

### DERIVED UNITS OF COMMON USE: PHYSICAL TO MASCLET UNITS
cgs_to_density = 2.1416e24
sunMpc3_to_density = sun_to_g / mpc_to_cm**3 * cgs_to_density

### DERIVED UNITS OF COMMON USE: MASCLET TO PHYSICAL UNITS
density_to_cgs = 1/cgs_to_density
density_to_sunMpc3 = 1/sunMpc3_to_density


### CONSTANTS
G_isu = 6.67430e-11
G_cgs = 6.67430e-8
G_masclet = G_cgs * cm_to_length**3 / g_to_mass / s_to_time**2

kB_isu = 1.380649e-23
kB_cgs = 1.380649e-16
#kB_masclet =

mp_isu = 1.67262192e-27
mp_cgs = mp_isu*1e3
mp_masclet = mp_isu * kg_to_mass

c_masclet = 1.
c_MpcMyr = 0.306594745

e_isu = 1.60218e-19

### MAGNETIC FIELD
magneticfield_to_gauss = 1/13.75
magneticfield_to_microgauss = 1e6 * magneticfield_to_gauss
gauss_to_magneticfield = 1/magneticfield_to_gauss
microgauss_to_magneticfield = 1 / magneticfield_to_microgauss
