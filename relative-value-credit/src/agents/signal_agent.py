class SignalGenerationAgent:
    def __init__(self, mu, sigma_eq):
        self.mu = mu
        self.sigma_eq = sigma_eq

    def generate_signal(self, current_residual):
        """
        Calculate the reversion signal based on the current residual.
        
        Parameters:
        current_residual (float): The current residual value for the instrument pair.

        Returns:
        float: The calculated reversion signal.
        """
        return - (current_residual - self.mu) / self.sigma_eq

    def generate_signals(self, residuals):
        """
        Generate reversion signals for a list of current residuals.

        Parameters:
        residuals (list of float): The current residual values for the instrument pairs.

        Returns:
        list of float: The calculated reversion signals.
        """
        return [self.generate_signal(residual) for residual in residuals]