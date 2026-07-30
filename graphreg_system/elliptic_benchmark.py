import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import xgboost as xgb

from graphreg_system.config import ELLIPTIC_DIR, FIGURES_DIR, HAS_GPU
from graphreg_system.logger import logger
from graphreg_system.checkpoint import checkpoint_mgr

class EllipticBenchmarkModule:
    def __init__(self, elliptic_dir=ELLIPTIC_DIR):
        self.elliptic_dir = elliptic_dir

    def evaluate_elliptic(self):
        step_name = "elliptic_benchmark_completed"
        if checkpoint_mgr.is_step_completed(step_name):
            results = checkpoint_mgr.load_object("elliptic_results")
            if results is not None:
                return results

        t0 = time.time()
        logger.info("Loading and evaluating Elliptic Benchmark Dataset (203,769 transactions)...")

        classes_path = os.path.join(self.elliptic_dir, "elliptic_txs_classes.csv")
        features_path = os.path.join(self.elliptic_dir, "elliptic_txs_features.csv")

        df_classes = pd.read_csv(classes_path)
        df_feats = pd.read_csv(features_path, header=None)

        num_cols = df_feats.shape[1]
        feat_names = ['txId', 'time_step'] + [f'feat_{i}' for i in range(num_cols - 2)]
        df_feats.columns = feat_names

        df_elliptic = pd.merge(df_feats, df_classes, on='txId')
        df_labeled = df_elliptic[df_elliptic['class'].isin(['1', '2'])].copy()
        df_labeled['target'] = (df_labeled['class'] == '1').astype(int)

        logger.info(f"Elliptic Labeled Dataset: {len(df_labeled):,} transactions (Class 1 Illicit: {np.sum(df_labeled['target']):,}, Class 2 Licit: {len(df_labeled) - np.sum(df_labeled['target']):,})")

        # Temporal Train/Test Split (Train: timesteps 1..34, Test: 35..49)
        train_mask = df_labeled['time_step'] <= 34
        test_mask = df_labeled['time_step'] > 34

        X_cols = [f'feat_{i}' for i in range(num_cols - 2)]
        X_train, X_test = df_labeled.loc[train_mask, X_cols], df_labeled.loc[test_mask, X_cols]
        y_train, y_test = df_labeled.loc[train_mask, 'target'], df_labeled.loc[test_mask, 'target']

        logger.info(f"Elliptic Temporal Split: Train shape {X_train.shape}, Test shape {X_test.shape}")

        xgb_params = {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.1, "random_state": 42, "n_jobs": -1}
        if HAS_GPU:
            xgb_params["tree_method"] = "hist"
            xgb_params["device"] = "cuda"

        models = {
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
            "Random Forest Engine": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "XGBoost Engine": xgb.XGBClassifier(**xgb_params),
            "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
        }

        results = {}

        for name, model in models.items():
            t_m0 = time.time()
            model.fit(X_train, y_train)
            t_train = time.time() - t_m0

            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else 0.0

            results[name] = {
                "Accuracy": float(acc),
                "Precision (Illicit)": float(prec),
                "Recall (Illicit)": float(rec),
                "F1-Score (Illicit)": float(f1),
                "ROC-AUC": float(auc),
                "Train_Time_s": float(t_train)
            }
            logger.success(f"Elliptic [{name}] -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}, AUC: {auc:.4f} ({t_train:.2f}s)")

        # Fig 2: Model Performance Comparison
        self._plot_comparison(results)

        t_elapsed = time.time() - t0
        logger.success(f"Completed Elliptic Benchmark evaluation in {t_elapsed:.3f}s.")

        # Save Checkpoint
        checkpoint_mgr.save_object("elliptic_results", results)
        checkpoint_mgr.mark_step_completed(step_name, results)

        return results

    def _plot_comparison(self, results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

        df_c = pd.DataFrame(checkpoint_mgr.load_object("ml_custom_results")).T.drop("top_features", errors='ignore').drop("CoinJoin Anomaly (XGB)", errors='ignore')
        df_c[['Accuracy', 'Precision', 'Recall', 'F1-Score']].plot(kind='bar', ax=ax1, colormap='viridis', edgecolor='black')
        ax1.set_title('(a) Mainnet Multi-Class Pattern F1-Scores', fontweight='bold')
        ax1.set_ylim(0.5, 1.02)
        ax1.legend(loc='lower right', fontsize=8)

        df_e = pd.DataFrame(results).T
        df_e[['Precision (Illicit)', 'Recall (Illicit)', 'F1-Score (Illicit)', 'ROC-AUC']].plot(kind='bar', ax=ax2, colormap='plasma', edgecolor='black')
        ax2.set_title('(b) Elliptic Benchmark Illicit Detection Metrics', fontweight='bold')
        ax2.set_ylim(0.3, 1.02)
        ax2.legend(loc='lower right', fontsize=8)

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig2_model_performance_comparison.png'), dpi=300)
        plt.close()

elliptic_benchmark = EllipticBenchmarkModule()
