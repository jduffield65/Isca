import unittest
import os
import numpy as np
from ..lapse_integral import integral_lapse_dlnp_hydrostatic, const_lapse_2_layer_fitting_xr
from typing import Optional
from ..lapse_theory import _get_var_at_plev, interp_hybrid_to_pressure, get_var_at_plev
from ...convection.base import lcl_metpy
from ...utils.constants import g, R
import xarray as xr

def generate_random_profiles(n_plev: int, temp_upper: float = 250, temp_lower: float = 300,
                             p_upper: float = 300*100, p_lower: float = 1000*100,
                             seed: Optional[int] = None) -> tuple[np.ndarray, np.ndarray]:
    """Generate random temperature and pressure profiles.

    Args:
        n_plev (int): Number of vertical levels.
        temp_upper: Temperature at `p_upper` furthest from the surface.
        temp_lower: Temperature at `p_lower` closest to the surface.
        p_upper: Pressure furthest from the surface.
        p_lower: Pressure closest to the surface.
        seed: Optional random seed for reproducibility.

    Returns:
        temp: Temperature profile ascending from ~250 K to ~300 K.
        pres: Pressure profile descending from ~100000 Pa to ~30000 Pa.
    """
    rng = np.random.default_rng(seed)

    # Temperature: ascending from ~250 to ~300 with small random noise
    temp = np.linspace(temp_upper, temp_lower, n_plev) + rng.normal(0, 2, n_plev)

    # Pressure: descending from ~100000 to ~30000 with small random noise
    pres = np.linspace(p_upper, p_lower, n_plev) + rng.normal(0, 500, n_plev)
    return temp, pres


class TestLapseIntegral(unittest.TestCase):
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    tol = 1e-3
    n_plev = 30
    n_repeat = 5        # number of times to run each of the below functions

    def test_integral_lapse_dlnp_hydrostatic1(self):
        for i in range(self.n_repeat):
            with self.subTest(run=i):
                # Test integral of single lapse rate function with analytic solution
                temp_lev, p_lev = generate_random_profiles(self.n_plev)
                p_low = np.random.uniform(p_lev[int(self.n_plev/2)], p_lev[-1])
                p_upper = np.random.uniform(p_lev[0], p_low)
                temp_low = _get_var_at_plev(temp_lev, p_lev, p_low, method='log')
                temp_upper = _get_var_at_plev(temp_lev, p_lev, p_upper, method='log')
                ans_func = integral_lapse_dlnp_hydrostatic(temp_lev, p_lev, p_low, p_upper, temp_low, temp_upper) * 1000
                ans_analytic = g/R * np.log(temp_upper/temp_low) * 1000     # Multiply by 1000 to convert to K/km units
                diff = ans_func - ans_analytic
                self.assertTrue(np.abs(diff).max() <= self.tol)

    def test_integral_lapse_dlnp_hydrostatic2(self):
        for i in range(self.n_repeat):
            with self.subTest(run=i):
                # Test integral of difference between two profiles function with analytic solution
                temp_lev, p_lev = generate_random_profiles(self.n_plev)
                p_low = np.random.uniform(p_lev[int(self.n_plev/2)], p_lev[-1])
                p_upper = np.random.uniform(p_lev[0], p_low)
                temp_low = _get_var_at_plev(temp_lev, p_lev, p_low, method='log')
                temp_upper = _get_var_at_plev(temp_lev, p_lev, p_upper, method='log')

                temp_lev_ref = generate_random_profiles(self.n_plev, temp_upper=220, temp_lower=330)[0]
                temp_low_ref = _get_var_at_plev(temp_lev_ref, p_lev, p_low, method='log')
                temp_upper_ref = _get_var_at_plev(temp_lev_ref, p_lev, p_upper, method='log')

                ans_func = integral_lapse_dlnp_hydrostatic(temp_lev, p_lev, p_low, p_upper, temp_low, temp_upper,
                                                           temp_lev_ref, temp_low_ref, temp_upper_ref) * 1000
                ans_analytic = g/R * (np.log(temp_upper/temp_low) - np.log(temp_upper_ref/temp_low_ref)) * 1000     # Multiply by 1000 to convert to K/km units
                diff = ans_func - ans_analytic
                self.assertTrue(np.abs(diff).max() <= self.tol)

    def test_integral_lapse_dlnp_hydrostatic3(self):
        for i in range(self.n_repeat):
            with self.subTest(run=i):
                # Test integral of absolute difference between two profiles function with analytic solution
                temp_lev, p_lev = generate_random_profiles(self.n_plev)
                p_low = np.random.uniform(p_lev[int(self.n_plev/2)], p_lev[-1])
                p_upper = np.random.uniform(p_lev[0], p_low)
                temp_low = _get_var_at_plev(temp_lev, p_lev, p_low, method='log')
                temp_upper = _get_var_at_plev(temp_lev, p_lev, p_upper, method='log')
                # Compute k, A such that T(p) = Ap^k
                k = np.log(temp_upper/temp_low) / np.log(p_upper/p_low)
                A = temp_low / p_low**k

                # Choose temp_upper_ref such that k is different
                temp_shift_const = np.random.uniform(-10, 10)
                temp_low_ref = temp_low       # keep same temp_low
                temp_upper_ref = temp_upper + temp_shift_const
                k_ref = np.log(temp_upper_ref / temp_low_ref) / np.log(p_upper / p_low)
                A_ref = temp_low / p_low ** k_ref

                # Regenerate temp_lev with k and A. This is so lapse rate difference at any given level is always same sign
                # This then allows for analytic solution
                temp_lev = A * p_lev**k
                temp_lev_ref = A_ref * p_lev**k_ref

                ans_func = integral_lapse_dlnp_hydrostatic(temp_lev, p_lev, p_low, p_upper, temp_low, temp_upper,
                                                           temp_lev_ref, temp_low_ref, temp_upper_ref, take_abs=True) * 1000
                ans_analytic = g/R * np.abs(np.log(temp_upper/temp_low) - np.log(temp_upper_ref/temp_low_ref)) * 1000     # Multiply by 100 to convert to K/km units
                diff = ans_func - ans_analytic
                self.assertTrue(np.abs(diff).max() <= self.tol)


class TestFitting(unittest.TestCase):
    folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'data')
    tol = 1e-3
    n_repeat = 5        # number of times to run each of the below functions
    n_lev_above_integral = 3
    p_ft = 400 * 100

    def get_ds(self):
        ds = xr.load_dataset(os.path.join(self.folder, 'ds1.nc'))
        ds['p_lcl'], ds['T_lcl_parcel'] = lcl_metpy(ds.TREFHT, ds.QREFHT, ds.PREFHT)
        ds['T_lcl_env'] = get_var_at_plev(ds.T, ds.P, ds.p_lcl)
        ds['T_ft'] = interp_hybrid_to_pressure(ds.T, ds.PS, ds.hyam, ds.hybm, ds.P0, np.atleast_1d(self.p_ft), lev_dim='lev')
        ds['T_ft'] = ds['T_ft'].isel(plev=0, drop=True).load()
        return ds

    def test_const_lapse(self):
        ds = self.get_ds()
        var = const_lapse_2_layer_fitting_xr(ds.T, ds.P, ds.TREFHT, ds.PREFHT, ds.T_lcl_env, ds.p_lcl, ds.T_ft,
                                             self.p_ft, n_lev_above_upper2_integral=self.n_lev_above_integral)
        # Sanity check that error below LCL is small in regions where near dry adiabat in this region
        large_lapse_below_lcl = var[0].isel(layer=0) > 8.5
        frac_error = np.abs(var[2]/var[1])
        max_error_below_lcl = float(frac_error.isel(layer=0).where(large_lapse_below_lcl).max())
        self.assertTrue(max_error_below_lcl <= 0.3)

        # Confirm that without `n_lev_above_upper2_integral`, bulk lapse and below LCL fitting is the same, but above is different
        var2 = const_lapse_2_layer_fitting_xr(ds.T, ds.P, ds.TREFHT, ds.PREFHT, ds.T_lcl_env, ds.p_lcl, ds.T_ft,
                                             self.p_ft, n_lev_above_upper2_integral=0)
        self.assertEqual(list(var[0].values.flatten()), list(var2[0].values.flatten()))
        for i in range(1, 3):
            self.assertEqual(list(var[i].isel(layer=0).values.flatten()), list(var2[i].isel(layer=0).values.flatten()))
            self.assertNotEqual(list(var[i].isel(layer=1).values.flatten()), list(var2[i].isel(layer=1).values.flatten()))

    def test_const_lapse_optimal(self):
        # Test constant lapse fitting but considering multiple LCL pressures for each location. Then chose best one
        ds = self.get_ds()
        n_lcl_mod = 11
        lcl_mod = np.linspace(-0.05, 0.05, n_lcl_mod)
        p_lcl_use = [(10 ** (np.log10(ds.p_lcl) + lcl_mod[i])) for i in range(n_lcl_mod)]
        ds['p_lcl'] = xr.concat(p_lcl_use, dim=xr.DataArray(lcl_mod, name='lcl_mod', dims='lcl_mod'))
        ds['T_lcl_env'] = get_var_at_plev(ds.T, ds.P, ds.p_lcl)
        var = const_lapse_2_layer_fitting_xr(ds.T, ds.P, ds.TREFHT, ds.PREFHT, ds.T_lcl_env, ds.p_lcl, ds.T_ft,
                                             self.p_ft, n_lev_above_upper2_integral=self.n_lev_above_integral)
        var_error = var[2].sum(dim='layer')
        ind_best = var_error.argmin(dim='lcl_mod')
        var_integral = var[1].sum(dim='layer')
        # Ensure that integral value is independent of split level
        self.assertTrue((var_integral.max(dim='lcl_mod') - var_integral.min(dim='lcl_mod')).max() <= self.tol)

        # Ensure that errors are different for different split levels
        self.assertTrue(var_error.std(dim='lcl_mod').mean() >= self.tol)
