# Relative Value Credit Analysis

This project implements a framework for Relative Value Credit analysis, focusing on the statistical modeling and backtesting of credit instruments. The framework is designed to identify mean-reverting behavior in credit spreads and generate actionable trading signals.

## Project Structure

- **data/**: Contains historical data files for credit indices.
  - `CDX.NA.IG.5Y.csv`: Sample historical data for the CDX NA IG 5Y credit index.

- **src/**: Contains the source code for the project.
  - **agents/**: Implements various agents responsible for data processing, modeling, signal generation, and backtesting.
    - `data_agent.py`: Defines the `DataAgent` class for sourcing and preprocessing financial market data.
    - `modeling_agent.py`: Defines the `ModelingAgent` class for statistical analysis and mean-reversion modeling.
    - `signal_agent.py`: Defines the `SignalGenerationAgent` class for generating trading signals.
    - `backtest_agent.py`: Defines the `BacktestingAgent` class for evaluating trading signals and performance metrics.
  - **models/**: Contains the implementation of statistical models.
    - `ou.py`: Implements the Ornstein-Uhlenbeck process for mean-reversion modeling.
  - **utils/**: Includes utility functions for input/output operations.
    - `io.py`: Contains functions for reading CSV files and saving results.
  - `main.py`: The entry point for the application, orchestrating the interaction between different agents.

- **notebooks/**: Contains Jupyter notebooks for exploratory data analysis and visualization.
  - `exploratory.ipynb`: Notebook for analyzing and visualizing financial data.

- **tests/**: Contains unit tests for the project.
  - `test_agents.py`: Unit tests for the various agent classes.

- **requirements.txt**: Lists the Python dependencies required for the project.

- **pyproject.toml**: Project configuration and dependency management.

- **.gitignore**: Specifies files and directories to be ignored by Git.

## Setup Instructions

1. Clone the repository:
   ```
   git clone <repository-url>
   cd relative-value-credit
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the main application:
   ```
   python src/main.py
   ```

## Usage

The framework can be used to analyze credit instruments by following these steps:

1. Load historical data using the `DataAgent`.
2. Perform statistical analysis and model mean-reversion using the `ModelingAgent`.
3. Generate trading signals with the `SignalGenerationAgent`.
4. Backtest the trading strategy using the `BacktestingAgent`.

## Contributing

Contributions are welcome! Please submit a pull request or open an issue for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.