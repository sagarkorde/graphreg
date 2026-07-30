import os
import json
import pickle
import pandas as pd
from graphreg_system.config import CHECKPOINT_DIR
from graphreg_system.logger import logger

class CheckpointManager:
    def __init__(self, checkpoint_dir=CHECKPOINT_DIR):
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.state_file = os.path.join(self.checkpoint_dir, "system_state.json")
        self.state = self._load_state()

    def _load_state(self):
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    logger.info(f"Loaded existing system state with {len(state.get('completed_steps', []))} completed steps.")
                    return state
            except Exception as e:
                logger.warning(f"Could not load state file ({e}). Starting fresh.")
        return {"completed_steps": [], "completed_row_groups": [], "metrics": {}}

    def save_state(self):
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    def is_step_completed(self, step_name):
        return step_name in self.state.get("completed_steps", [])

    def mark_step_completed(self, step_name, metadata=None):
        if step_name not in self.state["completed_steps"]:
            self.state["completed_steps"].append(step_name)
        if metadata:
            self.state["metrics"][step_name] = metadata
        self.save_state()
        logger.success(f"Checkpoint saved for step: '{step_name}'.")

    def save_dataframe(self, key, df):
        file_path = os.path.join(self.checkpoint_dir, f"{key}.parquet")
        df.to_parquet(file_path, index=False)
        logger.info(f"Saved dataframe checkpoint '{key}' ({len(df):,} rows) to disk.")

    def load_dataframe(self, key):
        file_path = os.path.join(self.checkpoint_dir, f"{key}.parquet")
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            logger.info(f"Loaded dataframe checkpoint '{key}' ({len(df):,} rows) from disk.")
            return df
        return None

    def save_object(self, key, obj):
        file_path = os.path.join(self.checkpoint_dir, f"{key}.pkl")
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Saved object checkpoint '{key}' to disk.")

    def load_object(self, key):
        file_path = os.path.join(self.checkpoint_dir, f"{key}.pkl")
        if os.path.exists(file_path):
            with open(file_path, "rb") as f:
                obj = pickle.load(f)
            logger.info(f"Loaded object checkpoint '{key}' from disk.")
            return obj
        return None

checkpoint_mgr = CheckpointManager()
