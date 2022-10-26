# Convection

The namelist for configuring convection is either [`qe_moist_convection_nml`](qe_moist_convection.md),
[`betts_miller_nml`](betts_miller.md) or [`ras_nml`](ras.md) depending on the choice of the
[`convection_scheme`](../main/idealized_moist_physics.md#convection_scheme) option.

The relevant diagnostics are given through the [`atmosphere`](../main/idealized_moist_physics.md#convection) 
module name in the `idealized_moist_phys` module.

If [`convection_scheme`](../main/idealized_moist_physics.md#convection_scheme) is [`ras_nml`](ras.md), 
then there are some additional diagnostics that can be specified with the [`ras`](ras.md#diagnostics) module name.

