import unittest
import pandas as pd
import numpy as np
from src.modeling_agent import get_beta, fit_ou_process, is_mean_reverting

class TestModelingAgent(unittest.TestCase):

    def test_get_beta(self):
        x = pd.Series([1, 2, 3, 4, 5])
        y = pd.Series([2, 4, 6, 8, 10])
        beta, residuals = get_beta(x, y)
        self.assertAlmostEqual(beta, 2.0, places=5)
        self.assertAlmostEqual(residuals.sum(), 0.0, places=5)

    def test_fit_ou_process(self):
        # Create a synthetic mean-reverting series
        np.random.seed(42)
        t = np.linspace(0, 1, 252)
        x = 0.5 * np.exp(-5 * t) + 0.1 * np.random.randn(252)
        residuals = pd.Series(x)

        theta, mu, sigma = fit_ou_process(residuals)

        self.assertGreater(theta, 0)
        self.assertTrue(is_mean_reverting(theta))

    def test_is_mean_reverting(self):
        self.assertTrue(is_mean_reverting(10))
        self.assertFalse(is_mean_reverting(-5))

if __name__ == '__main__':
    unittest.main()
