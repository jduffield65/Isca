### The constants below are from Holton 2004 Appendix A

# Gravity at sea level. Units: ms^{-2}
g = 9.81

# Ratio of molecular weight of water to that of dry air. Dimensionless.
# Comes from Appendix D2 of Holton 2004
epsilon = 0.622

# Gas constant for dry air. Units: JK^{-1}kg^{-1}
R = 287

# Gas constant of water. Units: JK^{-1}kg^{-1}
R_v = R / epsilon

# Specific heat of dry air at constant pressure. Units: JK^{-1}kg^{-1}
c_p = 1004

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

# Dry lapse rate, -dT/dz. Units: Km^{-1}
lapse_dry = g / c_p

# kappa is related to the ratio between heat capacity at constant volume to that at constant pressure through:
# kappa = 1 - c_v/c_p
kappa = R/c_p

# Add this to temperature in Celsius to give temperature in Kelvin
temp_kelvin_to_celsius = 273.15

# Radius of the earth in meters
radius_earth = 6.3781e6
