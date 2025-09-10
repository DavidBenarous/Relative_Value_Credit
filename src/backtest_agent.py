import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def run_backtest(prices: pd.Series, signals: pd.Series, signal_threshold: float) -> dict:
    """
    Runs a simple backtest of a trading strategy.

    Args:
        prices: The time series of prices.
        signals: The time series of trading signals.
        signal_threshold: The threshold to trigger a trade.

    Returns:
        A dictionary with backtest performance metrics.
    """
    positions = pd.Series(index=signals.index, data=0.0)
    positions[signals > signal_threshold] = -1  # Sell signal
    positions[signals < -signal_threshold] = 1   # Buy signal
    positions = positions.ffill().fillna(0)

    returns = prices.pct_change()
    strategy_returns = (positions.shift(1) * returns).fillna(0)

    # Calculate metrics
    cagr = (1 + strategy_returns.mean()) ** 252 - 1
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)

    cumulative_returns = (1 + strategy_returns).cumprod()
    peak = cumulative_returns.cummax()
    drawdown = (cumulative_returns - peak) / peak
    max_drawdown = drawdown.min()

    return {
        "cagr": cagr,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown": max_drawdown,
    }

def plot_residuals(residuals: pd.Series, filename: str):
    """
    Plots the time series of residuals.
    """
    plt.figure(figsize=(10, 6))
    residuals.plot()
    plt.title("Idiosyncratic Residuals")
    plt.xlabel("Date")
    plt.ylabel("Residual Value")
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_ou_fit(residuals: pd.Series, mu: float, filename: str):
    """
    Plots the residuals and the fitted OU process mean.
    """
    plt.figure(figsize=(10, 6))
    residuals.plot(label="Residuals")
    plt.axhline(y=mu, color='r', linestyle='--', label=f"Long-Term Mean (μ={mu:.4f})")
    plt.title("Ornstein-Uhlenbeck Process Fit")
    plt.xlabel("Date")
    plt.ylabel("Residual Value")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()

def plot_signal(signals: pd.Series, filename: str):
    """
    Plots the historical trading signal.
    """
    plt.figure(figsize=(10, 6))
    signals.plot()
    plt.axhline(y=1.5, color='r', linestyle='--', label="Sell Threshold")
    plt.axhline(y=-1.5, color='g', linestyle='--', label="Buy Threshold")
    plt.title("Trading Signal")
    plt.xlabel("Date")
    plt.ylabel("Signal (z-score)")
    plt.legend()
    plt.grid(True)
    plt.savefig(filename)
    plt.close()
