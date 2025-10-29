import numpy as np  # import numpy library and give it alias np for numerical operations
import pandas as pd  # import pandas library and give it alias pd for data structures like Series/DataFrame
import matplotlib.pyplot as plt  # import matplotlib's pyplot module as plt for plotting

def run_backtest(prices: pd.Series, signals: pd.Series, signal_threshold: float) -> dict:  # define function run_backtest with typed parameters and dict return
    """
    Runs a simple backtest of a trading strategy.

    Args:
        prices: The time series of prices.
        signals: The time series of trading signals.
        signal_threshold: The threshold to trigger a trade.

    Returns:
        A dictionary with backtest performance metrics.
    """
    positions = pd.Series(index=signals.index, data=0.0)  # create a Series of positions initialized to 0.0 with the same index as signals
    positions[signals > signal_threshold] = -1  # set positions to -1 (short/sell) where signal is greater than the threshold
    positions[signals < -signal_threshold] = 1   # set positions to +1 (long/buy) where signal is less than negative threshold
    positions = positions.ffill().fillna(0)  # forward-fill positions to carry forward last position and fill any leading NaNs with 0

    returns = prices.pct_change()  # compute percentage returns of the price series (NaN for first entry)
    strategy_returns = (positions.shift(1) * returns).fillna(0)  # apply lagged positions to returns to avoid look-ahead, fill NaN with 0

    # Calculate metrics
    cagr = (1 + strategy_returns.mean()) ** 252 - 1  # approximate annualized return using mean daily return compounded over 252 trading days
    sharpe_ratio = strategy_returns.mean() / strategy_returns.std() * np.sqrt(252)  # compute annualized Sharpe ratio (mean/std * sqrt(252))

    cumulative_returns = (1 + strategy_returns).cumprod()  # compute cumulative compounded returns over time
    peak = cumulative_returns.cummax()  # compute the running maximum (peak) of cumulative returns for drawdown calculation
    drawdown = (cumulative_returns - peak) / peak  # compute drawdown as percent drop from the running peak
    max_drawdown = drawdown.min()  # the maximum drawdown is the minimum value of the drawdown series (most negative)

    return {
        "cagr": cagr,  # return the calculated CAGR
        "sharpe_ratio": sharpe_ratio,  # return the calculated Sharpe ratio
        "max_drawdown": max_drawdown,  # return the maximum drawdown
    }

def plot_residuals(residuals: pd.Series):  # removed filename parameter
    """
    Plots the time series of residuals.
    """
    plt.figure(figsize=(10, 6))  # create a new matplotlib figure with specified size
    residuals.plot()  # plot the residuals Series using pandas' plotting wrapper
    plt.title("Idiosyncratic Residuals")  # set the plot title
    plt.xlabel("Date")  # label the x-axis
    plt.ylabel("Residual Value")  # label the y-axis
    plt.grid(True)  # enable grid on the plot
    plt.show()  # display the plot instead of saving
    plt.close()  # close the figure to free memory

def plot_ou_fit(residuals: pd.Series, mu: float):  # removed filename parameter
    """
    Plots the residuals and the fitted OU process mean.
    """
    plt.figure(figsize=(10, 6))  # create a new matplotlib figure with specified size
    residuals.plot(label="Residuals")  # plot the residuals Series and set a label for the legend
    plt.axhline(y=mu, color='r', linestyle='--', label=f"Long-Term Mean (μ={mu:.4f})")  # draw a horizontal line at the OU long-term mean mu
    plt.title("Ornstein-Uhlenbeck Process Fit")  # set the plot title
    plt.xlabel("Date")  # label the x-axis
    plt.ylabel("Residual Value")  # label the y-axis
    plt.legend()  # show the legend
    plt.grid(True)  # enable grid on the plot
    plt.show()  # display the plot instead of saving
    plt.close()  # close the figure to free memory

def plot_signal(signals: pd.Series):  # removed filename parameter
    """
    Plots the historical trading signal.
    """
    plt.figure(figsize=(10, 6))  # create a new matplotlib figure with specified size
    signals.plot()  # plot the signals Series using pandas' plotting wrapper
    plt.axhline(y=1.5, color='r', linestyle='--', label="Sell Threshold")  # draw a horizontal line indicating sell threshold at 1.5
    plt.axhline(y=-1.5, color='g', linestyle='--', label="Buy Threshold")  # draw a horizontal line indicating buy threshold at -1.5
    plt.title("Trading Signal")  # set the plot title
    plt.xlabel("Date")  # label the x-axis
    plt.ylabel("Signal (z-score)")  # label the y-axis
    plt.legend()  # show the legend
    plt.grid(True)  # enable grid on the plot
    plt.show()  # display the plot instead of saving
    plt.close()  # close the figure to free memory
