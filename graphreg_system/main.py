"""
GraphReg System Master Entry Point
Executes the full resumable pipeline, tracking timestamps, memory usage, and GPU acceleration.
"""

import time

from graphreg_system.config import (
    MAX_RAM_GB, HAS_GPU, GPU_NAME, DEVICE,
    CUSTOM_DATASET_PATH, ELLIPTIC_DIR
)
from graphreg_system.logger import logger
from graphreg_system.data_ingestion import ingestion_engine
from graphreg_system.feature_extractor import feature_extractor
from graphreg_system.regularity_solver import regularity_solver
from graphreg_system.ml_engine import ml_engine
from graphreg_system.elliptic_benchmark import elliptic_benchmark

def run_pipeline():
    t_start = time.time()

    logger.info("================================================================================")
    logger.info("  GRAPHREG ENGINE: SYSTEM IMPLEMENTATION & EXPERIMENTAL PIPELINE               ")
    logger.info("================================================================================")
    logger.info(f"Target Hardware Configuration:")
    logger.info(f" - Max Memory Allocation: {MAX_RAM_GB} GB RAM")
    logger.info(f" - GPU Acceleration: {GPU_NAME} ({'CUDA Active' if HAS_GPU else 'CPU Only'})")
    logger.info(f" - Target Primary Dataset: {CUSTOM_DATASET_PATH}")
    logger.info(f" - Benchmark Dataset: {ELLIPTIC_DIR}")
    logger.info("================================================================================")

    # Step 1: Resumable Data Ingestion
    logger.info(">>> [STEP 1/5] Ingesting & Sampling Mainnet Ledger Data...")
    df_raw = ingestion_engine.stream_and_sample_dataset(sample_target=200000)

    # Step 2: Resumable Feature Extraction
    logger.info(">>> [STEP 2/5] Extracting 23-Dimensional Topological & Monetary Features...")
    X, y, df_meta = feature_extractor.extract_features(df_raw)

    # Step 3: Resumable Sparse Regularity Partitioning
    logger.info(">>> [STEP 3/5] Solving Sparse Regularity Lemma Partition...")
    reg_results = regularity_solver.solve_regularity_partition(df_meta)

    # Step 4: Resumable Machine Learning Training & Anomaly Inference
    logger.info(">>> [STEP 4/5] Training & Evaluating Mainnet Machine Learning Engines...")
    ml_results = ml_engine.train_and_evaluate_custom(X, y, df_meta)

    # Step 5: Resumable Elliptic Benchmark Evaluation
    logger.info(">>> [STEP 5/5] Running Benchmark Evaluation on Elliptic Dataset...")
    elliptic_results = elliptic_benchmark.evaluate_elliptic()

    t_total = time.time() - t_start
    logger.info("================================================================================")
    logger.success(f"GRAPHREG PIPELINE EXECUTED SUCCESSFULLY IN {t_total:.2f} SECONDS.")
    logger.info("================================================================================")

    return {
        "regularity": reg_results,
        "ml": ml_results,
        "elliptic": elliptic_results,
    }

if __name__ == "__main__":
    run_pipeline()
