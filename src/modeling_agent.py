import numpy as np
import pandas as pd


def _slice_data_by_lookback(series: pd.Series, lookback_window: str) -> pd.Series:
    """Slices a time series based on a lookback window string."""
    end_date = series.index.max()
    offset_unit = lookback_window[-1].upper()
    if offset_unit in ("Y", "M"):
        offset_val = int(lookback_window[:-1])
        if offset_unit == "Y":
            offset = pd.DateOffset(years=offset_val)
        else:
            offset = pd.DateOffset(months=offset_val)
        start_date = end_date - offset
    else:
        start_date = end_date - pd.to_timedelta(lookback_window)
    return series[start_date:end_date]


def get_beta(
    series_x: pd.Series, series_y: pd.Series, lookback_window: str
) -> tuple[float, pd.Series]:
    """
    Calculates the beta and residuals of two time series using linear regression.

    Args:
        series_x: The independent variable time series.
        series_y: The dependent variable time series.
        lookback_window: The lookback window for the regression (e.g., "2Y").

    Returns:
        A tuple containing the beta (slope) and the residuals.
    """
    series_x = _slice_data_by_lookback(series_x, lookback_window)
    series_y = _slice_data_by_lookback(series_y, lookback_window)
    x = series_x.values
    y = series_y.values

    # Add a constant for the intercept
    X = np.vstack([x, np.ones(len(x))]).T

    # Using numpy's least squares
    beta, intercept = np.linalg.lstsq(X, y, rcond=None)[0]

    residuals = y - (beta * x + intercept)

    return beta, pd.Series(residuals, index=series_x.index)


def fit_ou_process(
    residuals: pd.Series, lookback_window: str
) -> tuple[float, float, float]:
    """
    Fits an Ornstein-Uhlenbeck process to a time series of residuals.

    Args:
        residuals: The time series of residuals.
        lookback_window: The lookback window for the fitting (e.g., "26W").

    Returns:
        A tuple containing the OU parameters (theta, mu, sigma).
    """
    residuals = _slice_data_by_lookback(residuals, lookback_window)
    x = residuals.values

    # Discretized OU process regression
    # dX = theta * (mu - X) * dt + sigma * dW
    # X_t+1 - X_t = theta * (mu - X_t) * dt + ...
    # X_t+1 = X_t + theta*mu*dt - theta*X_t*dt
    # X_t+1 = (1 - theta*dt) * X_t + theta*mu*dt
    # y = beta * x + alpha

    dt = 1 / 252  # Assuming daily data

    y = x[1:]
    X = np.vstack([x[:-1], np.ones(len(x) - 1)]).T

    beta, alpha = np.linalg.lstsq(X, y, rcond=None)[0]

    theta = -np.log(beta) / dt
    mu = alpha / (1 - beta)

    # Sigma from the residuals of this regression
    residuals_ou = y - (beta * x[:-1] + alpha)
    sigma = np.std(residuals_ou) / np.sqrt(dt)

    return theta, mu, sigma


def is_mean_reverting(theta: float) -> bool:
    """
    Checks if a pair is mean-reverting based on the OU parameter theta.
    A positive theta indicates mean reversion.

    Args:
        theta: The strength of mean reversion (theta from OU process).

    Returns:
        True if the pair is mean-reverting, False otherwise.
    """
    return theta > 0


def run_modeling_pipeline(
    series_x: pd.Series,
    series_y: pd.Series,
    regression_lookback: str,
    ou_lookback: str,
) -> tuple[float, pd.Series, float, float, float]:
    """
    Runs the full modeling pipeline for a pair of time series.

    Args:
        series_x: The independent variable time series.
        series_y: The dependent variable time series.
        regression_lookback: The lookback window for regression (e.g., "2Y").
        ou_lookback: The lookback window for OU process fitting (e.g., "26W").

    Returns:
        A tuple containing beta, residuals, and OU parameters (theta, mu, sigma).
    """
    beta, residuals = get_beta(series_x, series_y, regression_lookback)
    theta, mu, sigma = fit_ou_process(residuals, ou_lookback)
    return beta, residuals, theta, mu, sigma
