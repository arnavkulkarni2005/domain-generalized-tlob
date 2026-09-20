# preprocess.py
import numpy as np
import pandas as pd

def z_score_orderbook(data, mean_size=None, mean_prices=None, std_size=None, std_prices=None):
    """ 
    Applies Z-score. Evens (0::2) are Prices (distances), Odds (1::2) are Sizes (notionals).
    """
    print("Normalizing features...")
    if (mean_size is None) or (std_size is None):
        mean_size = data.iloc[:, 1::2].stack().mean()
        std_size = data.iloc[:, 1::2].stack().std()

    if (mean_prices is None) or (std_prices is None):
        mean_prices = data.iloc[:, 0::2].stack().mean()
        std_prices = data.iloc[:, 0::2].stack().std()

    price_cols = data.columns[0::2]
    size_cols = data.columns[1::2]

    for col in size_cols:
        data[col] = data[col].astype("float64")
        data[col] = (data[col] - mean_size) / std_size

    for col in price_cols:
        data[col] = data[col].astype("float64")
        data[col] = (data[col] - mean_prices) / std_prices

    if data.isnull().values.any():
        raise ValueError("Data contains null values after normalization.")

    return data, mean_size, mean_prices, std_size, std_prices


def label_from_midpoint(midpoints, smoothing_len, horizon):
    """
    Calculates 0 (Up), 1 (Stationary), 2 (Down) labels using the true midpoint.
    Adapted for clock-time horizons.
    """
    print(f"Generating labels (Horizon: {horizon}, Smoothing: {smoothing_len})...")
    
    if horizon < smoothing_len:
        smoothing_len = horizon

    # Slide window over 1D midpoints array
    windowed_midpoints = np.lib.stride_tricks.sliding_window_view(midpoints, window_shape=smoothing_len)
    
    previous_mids = windowed_midpoints[:-horizon]
    future_mids = windowed_midpoints[horizon:]
    
    # Smooth via mean
    prev_mean = np.mean(previous_mids, axis=1)
    future_mean = np.mean(future_mids, axis=1)
    
    # Calculate percentage return
    percentage_change = (future_mean - prev_mean) / prev_mean
    
    # Dynamic alpha based on asset volatility
    alpha = np.abs(percentage_change).mean() / 2
    print(f"Calculated Dynamic Alpha threshold: {alpha:.6f}")
    
    # Labeling logic
    # 0 = UP (> alpha), 2 = DOWN (< -alpha), 1 = STATIONARY
    labels = np.where(percentage_change < -alpha, 2, 
             np.where(percentage_change > alpha, 0, 1))
    
    counts = np.unique(labels, return_counts=True)
    print(f"Label Distribution (0:Up, 1:Stat, 2:Down): {counts[1]}")
    return labels