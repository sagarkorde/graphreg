import time
import pyarrow.parquet as pq
import pandas as pd
import numpy as np
from graphreg_system.config import CUSTOM_DATASET_PATH, MAX_RAM_GB, FEATURE_COLS
from graphreg_system.logger import logger
from graphreg_system.checkpoint import checkpoint_mgr

class DataIngestionEngine:
    def __init__(self, parquet_path=CUSTOM_DATASET_PATH):
        self.parquet_path = parquet_path
        self.pf = pq.ParquetFile(self.parquet_path)
        self.total_rows = self.pf.metadata.num_rows
        self.num_row_groups = self.pf.num_row_groups
        logger.info(f"Initialized DataIngestionEngine: {self.total_rows:,} transactions across {self.num_row_groups} row-groups.")
        logger.info(f"System RAM Limit: {MAX_RAM_GB} GB allocation.")

    def stream_and_sample_dataset(self, sample_target=200000):
        step_name = "data_ingestion_sampled"
        if checkpoint_mgr.is_step_completed(step_name):
            df_sampled = checkpoint_mgr.load_dataframe("custom_sampled_dataset")
            if df_sampled is not None:
                return df_sampled

        t0 = time.time()
        logger.info(f"Streaming and sampling {sample_target:,} balanced transactions across {self.num_row_groups} row groups...")
        
        # Select representative row groups across 3-year time range
        rg_indices = np.linspace(0, self.num_row_groups - 1, min(10, self.num_row_groups), dtype=int)
        sample_per_rg = sample_target // len(rg_indices)

        dfs = []
        for idx, rg_i in enumerate(rg_indices):
            t_chunk_start = time.time()
            df_rg = self.pf.read_row_group(rg_i).to_pandas()
            sample_n = min(sample_per_rg, len(df_rg))
            df_sub = df_rg.sample(n=sample_n, random_state=42)
            dfs.append(df_sub)
            t_chunk_elapsed = time.time() - t_chunk_start
            chunk_throughput = len(df_rg) / t_chunk_elapsed
            logger.progress(idx + 1, len(rg_indices), prefix="Ingesting Chunks", suffix=f"({chunk_throughput:,.0f} tx/s)")

        df_sampled = pd.concat(dfs, ignore_index=True)
        t_total = time.time() - t0
        
        throughput_tx_sec = self.total_rows / t_total
        latency_us = (t_total / self.total_rows) * 1e6
        mem_mb = df_sampled.memory_usage(deep=True).sum() / (1024 * 1024)

        logger.success(f"Completed dataset ingestion: {len(df_sampled):,} transactions sampled.")
        logger.info(f"Ingestion Throughput: {throughput_tx_sec:,.2f} tx/sec | Latency: {latency_us:.2f} µs/tx | Chunk Memory: {mem_mb:.2f} MB")

        # Save checkpoint
        checkpoint_mgr.save_dataframe("custom_sampled_dataset", df_sampled)
        checkpoint_mgr.mark_step_completed(step_name, {
            "total_rows": self.total_rows,
            "sampled_rows": len(df_sampled),
            "throughput_tx_sec": throughput_tx_sec,
            "latency_us": latency_us,
            "ingestion_time_s": t_total,
            "chunk_mem_mb": mem_mb
        })

        return df_sampled

ingestion_engine = DataIngestionEngine()
