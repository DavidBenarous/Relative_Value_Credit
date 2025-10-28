import unittest
import pandas as pd
import numpy as np
from src.modeling_agent import (
    get_beta,
    fit_ou_process,
    is_mean_reverting,
    run_modeling_pipeline,
)


class TestModelingAgent(unittest.TestCase):
    def setUp(self):
        # Create synthetic time series data for testing
        dates = pd.to_datetime(pd.date_range(end="2025-10-28", periods=504, freq="B"))
        self.series_x = pd.Series(np.random.randn(504), index=dates)
        self.series_y = pd.Series(
            2 * self.series_x.values + np.random.randn(504), index=dates
        )
        self.residuals = self.series_y - 2 * self.series_x

    def test_get_beta(self):
        beta, residuals = get_beta(self.series_x, self.series_y, lookback_window="1Y")
        self.assertIsInstance(beta, float)
        self.assertEqual(len(residuals), 262)

    def test_fit_ou_process(self):
        theta, mu, sigma = fit_ou_process(self.residuals, lookback_window="26W")
        self.assertIsInstance(theta, float)
        self.assertIsInstance(mu, float)
        self.assertIsInstance(sigma, float)

    def test_is_mean_reverting(self):
        self.assertTrue(is_mean_reverting(10))
        self.assertFalse(is_mean_reverting(-5))

    def test_run_modeling_pipeline(self):
        beta, residuals, theta, mu, sigma = run_modeling_pipeline(
            self.series_x, self.series_y, regression_lookback="2Y", ou_lookback="52W"
        )
        self.assertIsInstance(beta, float)
        self.assertIsInstance(residuals, pd.Series)
        self.assertIsInstance(theta, float)
        self.assertIsInstance(mu, float)
        self.assertIsInstance(sigma, float)
        self.assertGreater(len(residuals), 0)


if __name__ == "__main__":
    unittest.main()
