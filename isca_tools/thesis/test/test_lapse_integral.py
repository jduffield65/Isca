import unittest
import os
import numpy as np
from ..lapse_integral import integral_lapse_dlnp_hydrostatic
from typing import Optional
from ..lapse_theory import _get_var_at_plev
from ...utils.constants import g, R

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