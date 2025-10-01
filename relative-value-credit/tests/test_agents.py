import unittest
from src.agents.data_agent import DataAgent
from src.agents.modeling_agent import ModelingAgent
from src.agents.signal_agent import SignalGenerationAgent
from src.agents.backtest_agent import BacktestingAgent

class TestAgents(unittest.TestCase):

    def setUp(self):
        self.data_agent = DataAgent()
        self.modeling_agent = ModelingAgent()
        self.signal_agent = SignalGenerationAgent()
        self.backtest_agent = BacktestingAgent()

    def test_data_agent_retrieval(self):
        tickers = ['CDX.NA.IG.5Y']
        data = self.data_agent.retrieve_data(tickers)
        self.assertIsNotNone(data)
        self.assertGreater(len(data), 0)

    def test_modeling_agent_regression(self):
        sample_data = {
            'CDX.NA.IG.5Y': [100, 101, 102, 103, 104]
        }
        beta = self.modeling_agent.perform_regression(sample_data)
        self.assertIsNotNone(beta)

    def test_signal_generation_agent(self):
        residual = 0.5
        mu = 1.0
        sigma_eq = 0.2
        signal = self.signal_agent.generate_signal(residual, mu, sigma_eq)
        self.assertIsNotNone(signal)

    def test_backtesting_agent_performance(self):
        signals = [0.1, -0.2, 0.3]
        performance = self.backtest_agent.evaluate_performance(signals)
        self.assertIsNotNone(performance)
        self.assertIn('Sharpe Ratio', performance)

if __name__ == '__main__':
    unittest.main()