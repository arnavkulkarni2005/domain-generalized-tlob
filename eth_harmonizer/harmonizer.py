# harmonizer.py
import pandas as pd

def harmonize_eth_to_fi2010(raw_csv_path: str, num_levels: int):
    """
    Translates Kaggle ETH schema into a strictly alternating FI-2010 matrix.
    Column order: [Ask_P1, Ask_S1, Bid_P1, Bid_S1, Ask_P2, Ask_S2, Bid_P2, Bid_S2...]
    """
    print(f"Loading raw dataset from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    
    harmonized_data = {}
    ordered_cols = []
    
    # Extract only distance (Price) and notional (Size) for the top N levels
    for i in range(num_levels):
        ask_p = f'Ask_Price_{i+1}'
        ask_s = f'Ask_Size_{i+1}'
        bid_p = f'Bid_Price_{i+1}'
        bid_s = f'Bid_Size_{i+1}'
        
        # Map Kaggle columns to standard names
        harmonized_data[ask_p] = df[f'asks_distance_{i}']
        harmonized_data[ask_s] = df[f'asks_notional_{i}']
        harmonized_data[bid_p] = df[f'bids_distance_{i}']
        harmonized_data[bid_s] = df[f'bids_notional_{i}']
        
        # Enforce the strict interleaving expected by your existing codebase
        ordered_cols.extend([ask_p, ask_s, bid_p, bid_s])
        
    unified_df = pd.DataFrame(harmonized_data)[ordered_cols]
    
    # We must extract the true midpoint separately. Because the model uses 
    # 'distance' for generalization, we need the true nominal price for labels.
    midpoints = df['midpoint'].values
    
    print(f"Harmonized Feature Matrix Shape: {unified_df.shape}")
    return unified_df, midpoints