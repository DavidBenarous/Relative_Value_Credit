class OrnsteinUhlenbeck:
    def __init__(self, theta, mu, sigma):
        self.theta = theta  # Speed of mean reversion
        self.mu = mu        # Long-term mean
        self.sigma = sigma  # Volatility

    def simulate(self, x0, dt, n):
        import numpy as np
        
        # Initialize the array to hold the simulated values
        x = np.zeros(n)
        x[0] = x0
        
        for i in range(1, n):
            # Apply the Ornstein-Uhlenbeck process formula
            dx = self.theta * (self.mu - x[i-1]) * dt + self.sigma * np.sqrt(dt) * np.random.normal()
            x[i] = x[i-1] + dx
            
        return x

    def mean_reversion_signal(self, current_value):
        return (self.mu - current_value) / self.sigma