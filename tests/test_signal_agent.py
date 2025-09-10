import unittest
import numpy as np
from src.signal_agent import calculate_equilibrium_volatility, generate_signal

class TestSignalAgent(unittest.TestCase):

    def test_calculate_equilibrium_volatility(self):
        vol = calculate_equilibrium_volatility(theta=5, sigma=0.1)
        self.assertAlmostEqual(vol, 0.1 / np.sqrt(10), places=5)

        # Test with theta <= 0
        vol_inf = calculate_equilibrium_volatility(theta=0, sigma=0.1)
        self.assertTrue(np.isinf(vol_inf))

    def test_generate_signal(self):
        signal = generate_signal(current_residual=0.5, mu=0.1, equilibrium_volatility=0.2)
        self.assertAlmostEqual(signal, - (0.5 - 0.1) / 0.2, places=5)

        # Test with zero volatility
        signal_zero = generate_signal(current_residual=0.5, mu=0.1, equilibrium_volatility=0)
        self.assertEqual(signal_zero, 0.0)

if __name__ == '__main__':
    unittest.main()
