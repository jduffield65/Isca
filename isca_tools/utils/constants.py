from scipy.constants import Stefan_Boltzmann
import numpy as np

### The constants below are from Holton 2004 Appendix A

# Gravity at sea level (Isca value). Units: ms^{-2}
g = 9.8

# Gas constant for dry air (Isca value). Units: JK^{-1}kg^{-1}
R = 287.04

# Gas constant of water (Isca value). Units: JK^{-1}kg^{-1}
R_v = 461.5

# Ratio of molecular weight of water to that of dry air. Dimensionless.
epsilon = R/R_v

# Specific heat of water at constant pressure. Units: JK^{-1}kg^{-1}
c_p_water = 4184

# density of water. Units: kgm^{-3}
rho_water = 1000

# Molecular weight of water. Units: kgkmol^{-1} or equally gmol^{-1}
m_v = 18.016

# Molecular weight of dry air. Units: kgkmol^{-1} or equally gmol^{-1}
m_d = 18.016 / epsilon

# Latent heat of vaporization (or condensation) at 0 Celsius. Units: Jkg^{-1}
L_v = 2.5e6

# Latent heat of sublimation. Units: Jkg^{-1}
L_sub = 2.834E6

# Latent heat of fusion. Units: Jkg^{-1}
L_fus = L_sub - L_v

# kappa is related to the ratio between heat capacity at constant volume to that at constant pressure through:
# kappa = 1 - c_v/c_p (Isca value)
kappa = 2/7

# Specific heat of dry air at constant pressure (Isca value). Units: JK^{-1}kg^{-1}
c_p = R / kappa

# Dry lapse rate, -dT/dz. Units: Km^{-1}
lapse_dry = g / c_p

# Add this to temperature in Celsius to give temperature in Kelvin
temp_kelvin_to_celsius = 273.15

# Radius of the earth in meters
radius_earth = 6.3781e6

# Rotation of earth in radians/s. I.e. the Omega factor in the coriolis parameter: 2 Omega sin(theta)
rot_earth = 2*np.pi/(24*60**2)
