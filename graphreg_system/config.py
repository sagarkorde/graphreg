import os
import torch

# System Hardware Resources (HP Omen i9 13th Gen, 64GB RAM, RTX 4060 8GB)
MAX_RAM_GB = 50.0
MAX_RAM_BYTES = int(MAX_RAM_GB * 1024 * 1024 * 1024)

# GPU Acceleration
HAS_GPU = torch.cuda.is_available()
DEVICE = "cuda" if HAS_GPU else "cpu"
GPU_NAME = torch.cuda.get_device_name(0) if HAS_GPU else "CPU Only"

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CUSTOM_DATASET_PATH = os.path.join(BASE_DIR, "datasets", "custom", "Dataset.parquet")
ELLIPTIC_DIR = os.path.join(BASE_DIR, "datasets", "elliptic")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "graphreg_system", "checkpoints")
FIGURES_DIR = os.path.join(BASE_DIR, "graphreg_system", "figures")
LOGS_DIR = os.path.join(BASE_DIR, "graphreg_system", "logs")

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Feature Definitions
FEATURE_COLS = [
    'size', 'vsize', 'weight', 'input_count', 'output_count', 
    'total_input_value', 'total_output_value', 'fee',
    'input_output_ratio', 'value_difference', 
    'fee_rate_sat_per_byte', 'fee_rate_sat_per_vbyte',
    'input_address_count', 'output_address_count', 'total_addresses',
    'input_script_count', 'output_script_count', 'address_reuse',
    'has_op_return', 'rbf_enabled', 'avg_input_value', 'avg_output_value',
    'value_concentration_ratio'
]

PATTERN_CLASSES = {
    0: 'Peer-to-Peer',
    1: 'Consolidation',
    2: 'Distribution',
    3: 'Batch Payment',
    4: 'CoinJoin-like',
    5: 'General/Other'
}
