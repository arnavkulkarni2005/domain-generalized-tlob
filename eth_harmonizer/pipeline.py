# pipeline.py
import numpy as np
import config
from harmonizer import harmonize_eth_to_fi2010
from preprocess import z_score_orderbook, label_from_midpoint

def main():
    # 1. Parse and Harmonize
    features_df, raw_midpoints = harmonize_eth_to_fi2010(
        raw_csv_path=config.RAW_ETH_PATH, 
        num_levels=config.NUM_LEVELS
    )
    
    # 2. Normalize the Feature Matrix
    # (In a real pipeline, you would save these mean/std values to apply to a test set)
    features_df_norm, m_size, m_price, s_size, s_price = z_score_orderbook(features_df)
    
    # 3. Generate Labels
    labels = label_from_midpoint(
        midpoints=raw_midpoints, 
        smoothing_len=config.ETH_SMOOTHING, 
        horizon=config.ETH_HORIZON
    )
    
    # 4. Temporal Alignment
    # The labeling function drops the last (horizon + smoothing_len - 1) rows. 
    # We must trim the feature matrix to match the labels exactly.
    trim_length = len(raw_midpoints) - len(labels)
    features_trimmed = features_df_norm.values[:-trim_length]
    
    assert len(features_trimmed) == len(labels), "Feature and Label arrays must be equal length!"
    
    # 5. Save Final Tensors
    np.save(config.OUTPUT_FEATURES_PATH, features_trimmed)
    np.save(config.OUTPUT_LABELS_PATH, labels)
    
    print("\n--- Pipeline Complete ---")
    print(f"Final Features Tensor: {features_trimmed.shape} saved to {config.OUTPUT_FEATURES_PATH}")
    print(f"Final Labels Tensor:   {labels.shape} saved to {config.OUTPUT_LABELS_PATH}")

if __name__ == "__main__":
    main()