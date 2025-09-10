import numpy as np
import pandas as pd

def get_beta(series_x: pd.Series, series_y: pd.Series) -> tuple[float, pd.Series]:
    """
    Calculates the beta and residuals of two time series using linear regression.

    Args:
        series_x: The independent variable time series.
        series_y: The dependent variable time series.

    Returns:
        A tuple containing the beta (slope) and the residuals.
    """
    x = series_x.values
    y = series_y.values

    # Add a constant for the intercept
    X = np.vstack([x, np.ones(len(x))]).T

    # Using numpy's least squares
    beta, intercept = np.linalg.lstsq(X, y, rcond=None)[0]

    residuals = y - (beta * x + intercept)

    return beta, pd.Series(residuals, index=series_x.index)

def fit_ou_process(residuals: pd.Series) -> tuple[float, float, float]:
    """
    Fits an Ornstein-Uhlenbeck process to a time series of residuals.

    Args:
        residuals: The time series of residuals.

    Returns:
        A tuple containing the OU parameters (theta, mu, sigma).
    """
    x = residuals.values

    # Discretized OU process regression
    # dX = theta * (mu - X) * dt + sigma * dW
    # X_t+1 - X_t = theta * (mu - X_t) * dt + ...
    # X_t+1 = X_t + theta*mu*dt - theta*X_t*dt
    # X_t+1 = (1 - theta*dt) * X_t + theta*mu*dt
    # y = beta * x + alpha

    dt = 1 / 252  # Assuming daily data

    y = x[1:]
    X = np.vstack([x[:-1], np.ones(len(x)-1)]).T

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
