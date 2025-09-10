import numpy as np

def calculate_equilibrium_volatility(theta: float, sigma: float) -> float:
    """
    Calculates the equilibrium volatility of the OU process.

    Args:
        theta: The strength of mean reversion.
        sigma: The volatility of the process.

    Returns:
        The equilibrium volatility.
    """
    if theta <= 0:
        return np.inf
    return sigma / np.sqrt(2 * theta)

def generate_signal(current_residual: float, mu: float, equilibrium_volatility: float) -> float:
    """
    Generates the reversion signal.

    Args:
        current_residual: The current value of the residual.
        mu: The long-term mean of the OU process.
        equilibrium_volatility: The equilibrium volatility of the OU process.

    Returns:
        The reversion signal.
    """
    if equilibrium_volatility == 0 or np.isinf(equilibrium_volatility):
        return 0.0
    return - (current_residual - mu) / equilibrium_volatility
