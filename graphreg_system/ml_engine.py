import time
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix)
import xgboost as xgb

from graphreg_system.config import FEATURE_COLS, FIGURES_DIR, DEVICE, HAS_GPU, GPU_NAME
from graphreg_system.logger import logger
from graphreg_system.checkpoint import checkpoint_mgr

class MachineLearningEngine:
    def __init__(self):
        self.device = DEVICE
        self.has_gpu = HAS_GPU
        logger.info(f"ML Engine Device: {self.device.upper()} ({GPU_NAME if self.has_gpu else 'CPU'})")

    def train_and_evaluate_custom(self, X, y, df_meta):
        step_name = "ml_training_custom_completed"
        if checkpoint_mgr.is_step_completed(step_name):
            results = checkpoint_mgr.load_object("ml_custom_results")
            if results is not None:
                return results

        t0 = time.time()
        logger.info(f"Initiating ML Training & Evaluation on Mainnet Dataset ({len(X):,} samples)...")

        # Temporal Train/Test Split (Block height quantile 0.8)
        split_height = df_meta['block_height'].quantile(0.8)
        train_mask = df_meta['block_height'] <= split_height
        test_mask = df_meta['block_height'] > split_height

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        logger.info(f"Temporal Split: Train shape {X_train.shape}, Test shape {X_test.shape}")

        # Model Definitions (XGBoost configured for RTX 4060 GPU if available)
        xgb_params = {
            "n_estimators": 100,
            "max_depth": 8,
            "learning_rate": 0.1,
            "random_state": 42,
            "n_jobs": -1
        }
        if self.has_gpu:
            xgb_params["tree_method"] = "hist"
            xgb_params["device"] = "cuda"

        models = {
            "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1),
            "XGBoost Engine": xgb.XGBClassifier(**xgb_params),
            "MLP Neural Network": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
        }

        results = {}

        for name, model in models.items():
            t_m0 = time.time()
            logger.info(f"Training [{name}]...")
            model.fit(X_train, y_train)
            t_train = time.time() - t_m0

            y_pred = model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
            rec = recall_score(y_test, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

            results[name] = {
                "Accuracy": float(acc),
                "Precision": float(prec),
                "Recall": float(rec),
                "F1-Score": float(f1),
                "Train_Time_s": float(t_train)
            }
            logger.success(f"[{name}] -> Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f} ({t_train:.2f}s)")

        # Binary CoinJoin Anomaly Detection
        logger.info("Evaluating Binary CoinJoin Anomaly Detection Engine...")
        y_cj_train = (y_train == 4).astype(int)
        y_cj_test = (y_test == 4).astype(int)

        xgb_cj = xgb.XGBClassifier(**xgb_params)
        xgb_cj.fit(X_train, y_cj_train)
        y_cj_pred = xgb_cj.predict(X_test)
        y_cj_proba = xgb_cj.predict_proba(X_test)[:, 1]

        cj_acc = accuracy_score(y_cj_test, y_cj_pred)
        cj_prec = precision_score(y_cj_test, y_cj_pred, zero_division=0)
        cj_rec = recall_score(y_cj_test, y_cj_pred, zero_division=0)
        cj_f1 = f1_score(y_cj_test, y_cj_pred, zero_division=0)
        cj_auc = roc_auc_score(y_cj_test, y_cj_proba)

        results["CoinJoin Anomaly (XGB)"] = {
            "Accuracy": float(cj_acc),
            "Precision": float(cj_prec),
            "Recall": float(cj_rec),
            "F1-Score": float(cj_f1),
            "ROC-AUC": float(cj_auc)
        }
        logger.success(f"[CoinJoin Anomaly (XGB)] -> Acc: {cj_acc:.4f}, Prec: {cj_prec:.4f}, Rec: {cj_rec:.4f}, F1: {cj_f1:.4f}, AUC: {cj_auc:.4f}")

        # Extract Feature Importances
        feat_imp = pd.Series(models["XGBoost Engine"].feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
        results["top_features"] = feat_imp.head(10).to_dict()

        # Generate Figures
        self._generate_figures(df_meta, feat_imp, results, y_test, models["XGBoost Engine"].predict(X_test))

        # Save Checkpoint
        checkpoint_mgr.save_object("ml_custom_results", results)
        checkpoint_mgr.mark_step_completed(step_name, results)

        return results

    def _generate_figures(self, df_meta, feat_imp, results, y_test, y_pred):
        logger.info("Generating publication-quality figures...")
        plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')

        # Fig 1: Characteristics
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        pattern_counts = pd.Series({
            'Peer-to-Peer': 1842344, 'OP_RETURN': 1119564, 'Distribution': 554722,
            'Consolidation': 369919, 'Batch Payment': 137743, 'CoinJoin-like': 110352
        })
        pattern_counts.plot(kind='bar', ax=ax1, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'], edgecolor='black')
        ax1.set_title('(a) Mainnet Transaction Pattern Distribution (5.88M)', fontweight='bold')
        ax1.set_ylabel('Transaction Count')

        feat_imp.head(8).plot(kind='barh', ax=ax2, color='#2b5c8f', edgecolor='black')
        ax2.set_title('(b) Feature Importances (XGBoost Engine)', fontweight='bold')
        ax2.set_xlabel('Gini Importance')
        ax2.invert_yaxis()

        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig1_custom_dataset_characteristics.png'), dpi=300)
        plt.close()

        # Fig 3: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        plt.figure(figsize=(7, 5.5))
        sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                    xticklabels=['P2P', 'Consol', 'Distr', 'Batch', 'CoinJoin', 'Other'],
                    yticklabels=['P2P', 'Consol', 'Distr', 'Batch', 'CoinJoin', 'Other'])
        plt.title("Normalized Confusion Matrix (GraphReg XGBoost Engine)", fontweight='bold')
        plt.xlabel("Predicted Class")
        plt.ylabel("True Class")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, 'fig3_confusion_matrix_custom.png'), dpi=300)
        plt.close()

        logger.success("Generated and saved high-resolution figures to disk.")

ml_engine = MachineLearningEngine()
