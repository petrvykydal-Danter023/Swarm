import os
import numpy as np
import logging
from typing import Dict, Any, List
try:
    from safetensors.numpy import save_file, load_file
except ImportError:
    import pickle # Fallback

class DemoStorage:
    """
    Handles storage of demonstration data.
    Abstraction layer to allow swapping backend (pickle, safetensors, parquet).
    """
    def __init__(self, config: Any):
        self.config = config
        self.storage_path = config.get("storage", {}).get("path", "data/demos/")
        self.format = config.get("storage", {}).get("format", "safetensors")
        self.logger = logging.getLogger("DemoStorage")
        
        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

    def save_trajectory(self, trajectory: List[Dict[str, Any]], filename: str):
        """
        Save a single trajectory.
        """
        # Convert list of dicts to dict of stacked arrays for safetensors
        # Assuming trajectory is list of OracleDemo objects or dicts
        
        # Simple implementation: flatten
        data_dict = {}
        if not trajectory:
            return

        # MVP: just pickle for complex objects, safetensors for pure arrays
        # For now, let's use pickle for simplicity to ensure "everything works" without strict schema
        # In production this would use safetensors flattener
        
        full_path = os.path.join(self.storage_path, filename)
        
        if self.format == "safetensors":
             # Placeholder for proper implementation
             # save_file(data_dict, full_path)
             # Use pickle for MVP to support nested dicts which safetensors struggles with natively without flattening
             pass
        
        # Fallback/MVP
        import pickle
        with open(full_path + ".pkl", "wb") as f:
            pickle.dump(trajectory, f)
            
        self.logger.debug(f"Saved trajectory to {full_path}")

    def load_trajectory(self, filename: str) -> List[Dict[str, Any]]:
        full_path = os.path.join(self.storage_path, filename)
        import pickle
        with open(full_path + ".pkl", "rb") as f:
            return pickle.load(f)
