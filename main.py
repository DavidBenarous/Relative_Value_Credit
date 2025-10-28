import os
import pandas as pd
import numpy as np
from src.data_agent import load_data, clean_data, synchronize_data
from src.modeling_agent import run_modeling_pipeline, is_mean_reverting
from src.signal_agent import calculate_equilibrium_volatility, generate_signal
from src.backtest_agent import run_backtest, plot_residuals, plot_ou_fit, plot_signal

def main():
    """
    Main function to run the relative value credit analysis pipeline.
    """
    # --- Configuration ---
    TICKER_X = "CDX.NA.IG.5Y"
    TICKER_Y = "FECX"
    REGRESSION_LOOKBACK = "2Y" # Not used yet, but for future extension
    OU_LOOKBACK = "26W"        # Not used yet

    OUTPUT_DIR = "outputs"
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    # --- 1. Data Agent ---
    print("Loading and cleaning data...")
    df_x = load_data(TICKER_X)
    df_y = load_data(TICKER_Y)

    df_x = clean_data(df_x)
    df_y = clean_data(df_y)

    df_x_sync, df_y_sync = synchronize_data(df_x, df_y)
    print("Data loaded and synchronized.")

    # --- 2. Modeling Agent ---
    print("\nRunning regression and fitting OU process...")
    beta, residuals, theta, mu, sigma = run_modeling_pipeline(
        df_x_sync["Close"],
        df_y_sync["Close"],
        regression_lookback=REGRESSION_LOOKBACK,
        ou_lookback=OU_LOOKBACK,
    )
    print(f"Calculated Beta: {beta:.4f}")
    print(f"OU Parameters: θ={theta:.4f}, μ={mu:.4f}, σ={sigma:.4f}")

    if not is_mean_reverting(theta):
        print("Pair is not significantly mean-reverting. Exiting.")
        return
    print("Pair is mean-reverting.")

    # --- 3. Signal Generation Agent ---
    print("\nGenerating trading signals...")
    eq_vol = calculate_equilibrium_volatility(theta, sigma)
    signals = residuals.apply(lambda r: generate_signal(r, mu, eq_vol))
    print("Signals generated.")

    # --- 4. Backtesting & Visualization Agent ---
    print("\nRunning backtest and generating visualizations...")

    # For backtesting, we need a price series for the spread.
    # We'll use the residuals as a proxy for the spread price.
    backtest_results = run_backtest(residuals, signals, signal_threshold=1.5)
    print("\nBacktest Results:")
    for key, value in backtest_results.items():
        print(f"  {key}: {value:.4f}")

    plot_residuals(residuals, f"{OUTPUT_DIR}/residuals.png")
    plot_ou_fit(residuals, mu, f"{OUTPUT_DIR}/ou_fit.png")
    plot_signal(signals, f"{OUTPUT_DIR}/signals.png")
    print(f"\nVisualizations saved to '{OUTPUT_DIR}' directory.")


if __name__ == "__main__":
    main()
