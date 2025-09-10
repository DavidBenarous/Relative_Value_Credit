# Relative Value Credit

**Relative Value Credit** is a research and backtesting framework designed to identify and capitalize on **relative value opportunities** in credit markets. The framework focuses on the relationships between credit instruments, specifically **CDS indices** and **cash indices**, and uses statistical **mean reversion models** to create trading signals.

---

## Project Overview

The project's primary goal is to detect mispricings between credit indices by:
* Measuring their **beta relationships**.
* Extracting **idiosyncratic residuals** from regressions.
* Fitting an **Ornstein–Uhlenbeck (OU) process** to the residuals to model mean reversion.
* Generating signals based on the estimated mean-reversion strength.

The final signal functions similarly to a **z-score**, indicating how far the current spread relationship is from its equilibrium and providing a forecast for one-step-ahead forward returns.

---

## Instruments and Data

### CDS Indices
* **CDX:**
    * Markit CDX.NA.IG 5Y
    * Markit CDX.NA.IG 10Y
    * Markit CDX.NA.HY 5Y
* **iTraxx:**
    * iTraxx Europe 5Y & 10Y
    * iTraxx Europe Senior Financial 5Y
    * iTraxx Europe Subordinated Financial 5Y
    * iTraxx Crossover 5Y

### Cash Indices
| Future Ticker | Exchange | Bloomberg Ticker | Currency | Market Segment | Underlying Ticker | Underlying Name |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **FECX** | Eurex | LXY | EUR | Investment Grade | RECMTREU | MSCI Euro Corp SRB |
| **FUIG** | Eurex | BBE | USD | Investment Grade | LUACTRUU | Bloomberg US Corp |
| **IBIG** | CBOE | IHB | USD | Investment Grade | IBOXIG | iBoxx Shares & IG Corp |
| **FEHY** | Eurex | AHW | EUR | High Yield | BEHLTREU | Bloomberg Liquidity Screened Euro HY |
| **IBHY** | CBOE | IBY | USD | High Yield | IBOXIBHY | iBoxx Shares & HY Corp |
| **BYH** | CME | HYB | USD | High Yield | LHLTRUU | Bloomberg VLI HY |
| **DHB** | CME | DHB | USD | IG - Hedged | I3287US | Bloomberg US Corp Dur Hedged |
| **DHY** | CME | DHY | USD | HY - Hedged | H9131US | Bloomberg VLI HY Dur Hedged |

---

## Methodology

1.  **Regression & Beta Estimation**
    * Compute the **beta** between two time series over a long lookback window (2–5 years).
    * Extract residuals to represent **idiosyncratic returns**.

2.  **Mean-Reversion Model (OU Process)**
    * Fit an **OU process** on the residuals over a short lookback window (13–52 weeks).
    * Estimate the following parameters:
        * **$θ$**: Strength of mean reversion
        * **$μ$**: Long-term mean
        * **$σ$**: Volatility of residuals
    * Reject pairs with non-mean-reverting behavior ($b ≤ 0$ or $b ≥ 1$).

3.  **Signal Construction**
    * Compute deviation from equilibrium using the formula: $s = - (X_n - μ) / σ_{eq}$.
    * The signal's magnitude is proportional to the **normalized distance from the mean**.

---

## Parameters

| Parameter | Description | Typical Values |
| :--- | :--- | :--- |
| Regression Lookback | Window length for beta estimation | 2Y / 3Y / 5Y |
| OU Lookback | Window length for OU fitting on residuals | 13W / 26W / 52W / 2Y |

No additional hyperparameters are required beyond these two choices, making the approach **robust and reproducible**.

---

## Output

* **Reversion Signal:** A continuous measure similar to a z-score.
* **Backtests:** Performance statistics based on signal thresholds.
* **Visualization:** Time series of residuals, OU fit, and signal history.

---

## Implementation
 The framework is structured into four distinct modules, each representing an 'agent' with a specific responsibility:

1.  **Data Agent (`src/data_agent.py`)**: Handles loading, cleaning, and synchronizing time series data for financial instruments.

2.  **Modeling Agent (`src/modeling_agent.py`)**: Performs linear regression to calculate beta and residuals, and then fits an Ornstein-Uhlenbeck (OU) mean-reversion process to the residuals.

3.  **Signal Generation Agent (`src/signal_agent.py`)**: Generates trading signals based on the parameters of the fitted OU process.

4.  **Backtesting & Visualization Agent (`src/backtest_agent.py`)**: Provides functionality to run historical backtests and generate visualizations of the analysis, including residuals, OU fit, and trading signals.

A main orchestration script (`main.py`) is included to run the entire pipeline from data ingestion to backtesting. Unit tests have been added for each agent to ensure the correctness of the implementation. A `requirements.txt` file is also provided to manage project dependencies.
