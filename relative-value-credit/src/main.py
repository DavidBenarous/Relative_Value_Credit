from agents.data_agent import DataAgent
from agents.modeling_agent import ModelingAgent
from agents.signal_agent import SignalGenerationAgent
from agents.backtest_agent import BacktestingAgent

def main():
    # Initialize agents
    data_agent = DataAgent()
    modeling_agent = ModelingAgent()
    signal_generation_agent = SignalGenerationAgent()
    backtesting_agent = BacktestingAgent()

    # Step 1: Data retrieval and preprocessing
    cleaned_data = data_agent.retrieve_and_clean_data()

    # Step 2: Statistical modeling
    regression_results = modeling_agent.perform_regression(cleaned_data)
    ou_parameters = modeling_agent.fit_ou_process(regression_results)

    # Step 3: Signal generation
    signals = signal_generation_agent.generate_signals(ou_parameters)

    # Step 4: Backtesting
    backtest_results = backtesting_agent.run_backtest(signals)

    # Output results
    print("Backtest Results:", backtest_results)

if __name__ == "__main__":
    main()