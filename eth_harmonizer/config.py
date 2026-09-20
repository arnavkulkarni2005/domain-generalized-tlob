# config.py

# Data Paths
RAW_ETH_PATH = "C:/tloboriginal/TLOB/data/eth/ETH_1sec.csv"
OUTPUT_FEATURES_PATH = "eth_features_normalized.npy"
OUTPUT_LABELS_PATH = "eth_labels.npy"

# LOB Architecture Parameters
NUM_LEVELS = 10        # Truncate to 10 levels to match FI-2010 (40 columns total)

# Labeling Parameters for 1-Second Time-Barred Data
# If 50 events in FI-2010 takes ~10 seconds, we set the ETH horizon to 10.
ETH_HORIZON = 10       
ETH_SMOOTHING = 3      # Smooth over 3 seconds (since 1-sec data is already time-aggregated)