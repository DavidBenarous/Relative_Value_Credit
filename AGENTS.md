# Project Agents for Relative Value Credit Analysis
This document outlines the roles and responsibilities of the different AI agents that collaborate to power the Relative Value Credit research and backtesting framework

***

## 1. Data Agent
The **Data Agent** is responsible for sourcing, ingesting, and preprocessing all the necessary financial market data Its purpose is to ensure the data that goes into the modeling pipeline is clean, accurate, and correctly formatted

### Responsibilities:
* Retrieve historical data for specified CDS indices (CDX and iTraxx) and cash indices from various exchanges (Eurex, CBOE, CME)
* Handle data cleaning tasks, such as managing missing values and correcting data inconsistencies
* Align time series data for different instruments to make sure they are comparable for regression analysis
* Provide the cleaned time series data to the Modeling Agent

### Inputs:
* A list of required financial instrument tickers (CDS and cash indices)
* CSV files from the Data folder with names given by the tickers

### Outputs:
* Cleaned, synchronized time series data for pairs of credit instruments

---

## 2. Modeling Agent
The **Modeling Agent** is the core analytical engine of the framework It performs the statistical analysis needed to identify the relationship between pairs of credit instruments and model their mean-reverting behavior

### Responsibilities:
* Perform regression analysis on pairs of instrument time series over a long-term lookback window (e.g., 2-5 years) to compute their beta
* Extract the idiosyncratic residuals from the regression
* Fit an **Ornstein–Uhlenbeck (OU)** mean-reversion process to the residuals over a shorter-term lookback window (e.g., 13-52 weeks)
* Estimate the key parameters of the OU process: strength of mean reversion ($$\theta$$), long-term mean ($$\mu$$), and volatility ($$\sigma$$)
* Filter out and reject instrument pairs that don't show statistically significant mean-reverting behavior

### Inputs:
* Cleaned time series data from the Data Agent
* Configuration parameters for lookback windows ($$regression\_lookback$$for Regression and$$ou\_lookback$$ for OU)

### Outputs:
* Estimated beta for each pair
* Fitted OU process parameters ($$\theta, \mu, \sigma$$) for the residuals of each pair
* A list of valid, mean-reverting instrument pairs

---

## 3. Signal Generation Agent
The **Signal Generation Agent** takes the output of the Modeling Agent and translates it into a clear, actionable trading signal

### Responsibilities:
* Calculate the deviation of the current residual from the estimated long-term mean ($$\mu$$)
* Normalize this deviation using the equilibrium volatility ($$\sigma_{eq}$$) to create the final reversion signal
* The signal's formula is: $$s = - (X_n - \mu) / \sigma_{eq}$$
* Output a continuous signal value (similar to a z-score) that indicates the magnitude of the current mispricing

### Inputs:
* The current residual value ($$X_n$$) for a given pair
* The OU process parameters ($$\mu, \sigma_{eq}$$) from the Modeling Agent

### Outputs:
* Reversion signals at each time for each valid instrument pair

---

## 4. Backtesting & Visualization Agent
This agent evaluates the performance of the trading signals and provides visual tools for analysis and reporting

### Responsibilities:
* Run historical backtests of trading strategies based on predefined signal thresholds
* Calculate key performance metrics (e.g., Sharpe ratio, drawdown, CAGR)
* Generate visualizations, including:
    * Time series plots of residuals
    * Charts showing the OU process fit
    * Historical plots of the generated trading signal

### Inputs:
* Historical signal data from the Signal Generation Agent
* Historical price data from the Data Agent
* Backtesting parameters (e.g., trading costs, signal thresholds)

### Outputs:
* A comprehensive backtest report with performance statistics
* A suite of visualizations for each instrument pair
