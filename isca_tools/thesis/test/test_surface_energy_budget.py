import unittest
import numpy as np
from typing import Optional
from ..surface_energy_budget import get_temp_fourier_analytic, get_temp_fourier_analytic2
from ...utils.constants import c_p, rho_water
from ...utils.fourier import fourier_series
from ...utils.radiation import get_heat_capacity


def generate_swdn_sfc(time: np.ndarray, harmonic1_amp_approx: float = 100,
                      harmonic2_amp_approx: float = 50,
                      noise_amp_approx: float = 50,
                      amp_random_frac: float = 0.1,
                      seed: Optional[int] = None) -> np.ndarray:
    """Generate a random shortwave radiation at the surface, typical of the Northern Hemisphere.
    Will be based on two harmonic fourier series, with noise added.

    Args:
        time: int `[n_time]`
            The time e.g., 0 to 360 for 360-day calendar indicating days of the year.
        harmonic1_amp_approx: Amplitude of first harmonic will be drawn by normal distribution centered
            on this. Units: Wm-2.
        harmonic2_amp_approx: Amplitude of second harmonic will be drawn by normal distribution centered
            on this. Units: Wm-2.
        noise_amp_approx: Amplitude of noise added to fourier swdn will be drawn by normal distribution centered
            on this. Units: Wm-2.
        amp_random_frac: Standard deviation of normal distributions drawn will be this fraction of the amplitude.
        seed: Optional random seed for reproducibility.

    Returns:
        swdn: float `[n_time]`
            Shortwave radiation at the surface at each day of the year. Units: Wm-2.
    """
    rng = np.random.default_rng(seed)
    harmonic1_amp = rng.normal(harmonic1_amp_approx, amp_random_frac * harmonic1_amp_approx)
    harmonic2_amp = rng.normal(harmonic2_amp_approx, amp_random_frac * harmonic2_amp_approx)
    noise_amp = rng.normal(noise_amp_approx, amp_random_frac * noise_amp_approx)
    # -harmonic1_amp as northern hemisphere
    swdn = fourier_series(time, [0, -harmonic1_amp, harmonic2_amp], [0, 0])
    swdn = swdn + np.random.random(time.size) * noise_amp
    swdn = swdn - swdn.min()
    return swdn


class TestFourierAnalytic(unittest.TestCase):
    # Make sure both methods to compute fourier solution to surface energy budget give same answer
    tol = 1e-3
    n_time = 360
    time = np.arange(n_time)
    n_repeat = 5  # number of times to run each of the below functions
    depth_max = 30  # Max mixed layer depth in m.
    # Mean and stdev of distribution to draw params from
    lambda_approx = [4, 0.4]
    lambda_phase_approx = [0, 2]
    lambda_sq_approx = [0, 0.03]
    lambda_cos_approx = [0, 3]
    lambda_sin_approx = [0, 3]
    seed = None

    def test_get_temp_fourier_analytic(self):
        for i in range(self.n_repeat):
            with self.subTest(run=i):
                # Test integral of single lapse rate function with analytic solution
                rng = np.random.default_rng(seed=self.seed)
                swdn_sfc = generate_swdn_sfc(time=self.time, seed=self.seed)
                depth = rng.random() * self.depth_max
                heat_capacity = get_heat_capacity(c_p, rho_water, depth)
                lambda_const = rng.normal(self.lambda_approx[0], self.lambda_approx[1])
                lambda_phase = rng.normal(self.lambda_phase_approx[0], self.lambda_phase_approx[1])
                lambda_sq = rng.normal(self.lambda_sq_approx[0], self.lambda_sq_approx[1])
                lambda_cos = rng.normal(self.lambda_cos_approx[0], self.lambda_cos_approx[1])
                lambda_sin = rng.normal(self.lambda_cos_approx[0], self.lambda_cos_approx[1])

                # Test single and two harmonics - not allowed lambda_sq, lambda_cos, lambda_sin for single
                for n_harmonics in [2]:
                    # Must have no sw phase in sol1 because always the case in sol2.
                    sol1 = get_temp_fourier_analytic(self.time, swdn_sfc, heat_capacity, lambda_const, lambda_phase,
                                                     lambda_sq=0 if n_harmonics == 1 else lambda_sq,
                                                     lambda_cos=0 if n_harmonics == 1 else lambda_cos,
                                                     lambda_sin=0 if n_harmonics == 1 else lambda_sin,
                                                     n_harmonics_sw=n_harmonics, include_sw_phase=False)
                    sol2 = get_temp_fourier_analytic2(self.time, swdn_sfc, heat_capacity, lambda_const, lambda_phase,
                                                      lambda_sq=0 if n_harmonics == 1 else lambda_sq,
                                                      lambda_cos=0 if n_harmonics == 1 else lambda_cos,
                                                      lambda_sin=0 if n_harmonics == 1 else lambda_sin,
                                                      n_harmonics=n_harmonics)
                    for i in range(len(sol1)):
                        diff = sol1[i] - sol2[i]
                        if np.abs(diff).max() > self.tol:
                            hi = 5
                        self.assertTrue(np.abs(diff).max() <= self.tol)
