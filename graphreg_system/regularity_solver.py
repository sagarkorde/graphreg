import time
import numpy as np
import pandas as pd
from graphreg_system.logger import logger
from graphreg_system.checkpoint import checkpoint_mgr

class SparseRegularitySolver:
    def __init__(self, epsilon=0.05, k_clusters=10):
        self.epsilon = epsilon
        self.k_clusters = k_clusters

    def solve_regularity_partition(self, df):
        step_name = "regularity_partition_solved"
        if checkpoint_mgr.is_step_completed(step_name):
            reg_results = checkpoint_mgr.load_object("regularity_results")
            if reg_results is not None:
                return reg_results

        t0 = time.time()
        logger.info(f"Solving Sparse Regularity Lemma partition (epsilon={self.epsilon}, k={self.k_clusters} clusters)...")

        # Extract input/output count features as node degree representations
        in_degrees = df['input_count'].values
        out_degrees = df['output_count'].values
        total_tx = len(df)

        # Quantile-based regular vertex partitioning V_1, ..., V_k
        in_quantiles = np.quantile(in_degrees, np.linspace(0, 1, self.k_clusters + 1))
        
        # Bipartite density matrix D (k x k)
        density_matrix = np.zeros((self.k_clusters, self.k_clusters))
        sparse_scaling = max(1e-6, np.mean(in_degrees + out_degrees) / total_tx)

        for i in range(self.k_clusters):
            mask_i = (in_degrees >= in_quantiles[i]) & (in_degrees <= in_quantiles[i+1])
            count_i = np.sum(mask_i)
            for j in range(self.k_clusters):
                mask_j = (out_degrees >= in_quantiles[j]) & (out_degrees <= in_quantiles[j+1])
                count_j = np.sum(mask_j)
                if count_i > 0 and count_j > 0:
                    # Normalized bipartite density d_sparse(V_i, V_j)
                    edges_ij = np.sum(out_degrees[mask_i])
                    density_matrix[i, j] = edges_ij / (sparse_scaling * count_i * count_j)

        # Separate regular pairs (quasi-random) vs irregular pair shifts (forensic anomalies)
        mean_density = np.mean(density_matrix)
        irregular_mask = np.abs(density_matrix - mean_density) > self.epsilon * mean_density
        num_irregular = np.sum(irregular_mask)

        t_elapsed = time.time() - t0

        results = {
            "epsilon": self.epsilon,
            "k_clusters": self.k_clusters,
            "density_matrix": density_matrix.tolist(),
            "mean_density": float(mean_density),
            "num_irregular_pairs": int(num_irregular),
            "regular_pair_pct": float((1 - num_irregular / (self.k_clusters ** 2)) * 100),
            "solver_time_s": float(t_elapsed)
        }

        logger.success(f"Regularity Partition Solved in {t_elapsed:.3f}s: {results['regular_pair_pct']:.2f}% regular pairs, {num_irregular} irregular pair shifts.")

        # Save checkpoint
        checkpoint_mgr.save_object("regularity_results", results)
        checkpoint_mgr.mark_step_completed(step_name, results)

        return results

regularity_solver = SparseRegularitySolver()
