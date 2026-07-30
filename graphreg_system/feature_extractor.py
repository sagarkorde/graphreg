import time
import pandas as pd
import numpy as np
from graphreg_system.config import FEATURE_COLS, PATTERN_CLASSES
from graphreg_system.logger import logger
from graphreg_system.checkpoint import checkpoint_mgr

class FeatureExtractorModule:
    def __init__(self):
        self.feature_cols = FEATURE_COLS

    def assign_pattern_class(self, row):
        if row.get('is_coinjoin_like', False): return 4
        elif row.get('is_batch_payment', False): return 3
        elif row.get('is_distribution', False): return 2
        elif row.get('is_consolidation', False): return 1
        elif row.get('is_peer_to_peer', False): return 0
        else: return 5

    def extract_features(self, df):
        step_name = "feature_extraction_completed"
        if checkpoint_mgr.is_step_completed(step_name):
            X = checkpoint_mgr.load_dataframe("custom_features_X")
            y = checkpoint_mgr.load_dataframe("custom_features_y")['pattern_class']
            if X is not None and y is not None:
                return X, y, df

        t0 = time.time()
        logger.info(f"Extracting 23 topological & monetary features across {len(df):,} transactions...")

        X = df[self.feature_cols].copy().fillna(0)
        
        logger.info("Computing ground truth structural transaction pattern classes...")
        pattern_classes = df.apply(self.assign_pattern_class, axis=1)
        df['pattern_class'] = pattern_classes
        y = pattern_classes

        t_elapsed = time.time() - t0
        latency_us = (t_elapsed / len(df)) * 1e6

        logger.success(f"Extracted {X.shape[1]} features for {len(X):,} transactions in {t_elapsed:.3f}s ({latency_us:.2f} µs/tx).")
        logger.info(f"Class distribution: {dict(y.value_counts())}")

        # Save checkpoint
        checkpoint_mgr.save_dataframe("custom_features_X", X)
        checkpoint_mgr.save_dataframe("custom_features_y", pd.DataFrame({'pattern_class': y}))
        checkpoint_mgr.mark_step_completed(step_name, {
            "num_features": X.shape[1],
            "num_samples": len(X),
            "extraction_time_s": t_elapsed,
            "latency_us_per_tx": latency_us
        })

        return X, y, df

feature_extractor = FeatureExtractorModule()
