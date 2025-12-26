"""
Entropy Engine V3 - Checkpointing
Save and load training state using safetensors.
"""
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np

# We'll use pickle for now since safetensors.flax has limited support
# In production, consider orbax-checkpoint or flax.serialization
import pickle

def save_checkpoint(
    params: Dict,
    opt_state: Any,
    step: int,
    path: str,
    metadata: Dict[str, Any] = None
):
    """
    Save a training checkpoint.
    
    Args:
        params: Network parameters (pytree)
        opt_state: Optimizer state
        step: Current training step
        path: Output file path
        metadata: Optional metadata dict
    """
    # Ensure directory exists
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "params": params,
        "opt_state": opt_state,
        "step": step,
        "metadata": metadata or {},
        "timestamp": datetime.now().isoformat()
    }
    
    with open(path, "wb") as f:
        pickle.dump(checkpoint, f)
    
    # Save metadata as JSON sidecar for easy inspection
    meta_path = path + ".meta.json"
    with open(meta_path, "w") as f:
        json.dump({
            "step": step,
            "timestamp": checkpoint["timestamp"],
            **(metadata or {})
        }, f, indent=2, default=str)
    
    print(f"[Checkpoint] Saved to {path} (step={step})")


def load_checkpoint(path: str) -> Dict[str, Any]:
    """
    Load a training checkpoint.
    
    Args:
        path: Checkpoint file path
        
    Returns:
        Dict with 'params', 'opt_state', 'step', 'metadata'
    """
    with open(path, "rb") as f:
        checkpoint = pickle.load(f)
    
    print(f"[Checkpoint] Loaded from {path} (step={checkpoint['step']})")
    return checkpoint


class CheckpointManager:
    """
    Manages checkpoint saving with rotation and best-model tracking.
    
    Usage:
        manager = CheckpointManager("checkpoints/run_001", max_to_keep=5)
        
        # During training:
        manager.save(params, opt_state, step, metrics={"reward": 0.5})
        
        # Load best:
        ckpt = manager.load_best()
    """
    def __init__(
        self, 
        directory: str, 
        max_to_keep: int = 5,
        best_metric: str = "reward"
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        self.best_metric = best_metric
        self.checkpoints = []
        self.best_value = float("-inf")
        
    def save(
        self, 
        params: Dict, 
        opt_state: Any, 
        step: int, 
        metrics: Dict[str, float] = None
    ):
        """Save checkpoint and manage rotation."""
        filename = f"ckpt_step_{step:08d}.pkl"
        path = self.directory / filename
        
        save_checkpoint(params, opt_state, step, str(path), metadata=metrics)
        self.checkpoints.append((step, str(path)))
        
        # Track best
        if metrics and self.best_metric in metrics:
            value = metrics[self.best_metric]
            if value > self.best_value:
                self.best_value = value
                best_path = self.directory / "best.pkl"
                save_checkpoint(params, opt_state, step, str(best_path), metadata=metrics)
                print(f"[Checkpoint] New best: {self.best_metric}={value:.4f}")
        
        # Rotation: keep only max_to_keep
        while len(self.checkpoints) > self.max_to_keep:
            _, old_path = self.checkpoints.pop(0)
            try:
                Path(old_path).unlink()
                Path(old_path + ".meta.json").unlink(missing_ok=True)
            except FileNotFoundError:
                pass
    
    def load_latest(self) -> Optional[Dict[str, Any]]:
        """Load the most recent checkpoint."""
        if not self.checkpoints:
            # Try to find checkpoints on disk
            ckpts = sorted(self.directory.glob("ckpt_step_*.pkl"))
            if ckpts:
                return load_checkpoint(str(ckpts[-1]))
            return None
        _, path = self.checkpoints[-1]
        return load_checkpoint(path)
    
    def load_best(self) -> Optional[Dict[str, Any]]:
        """Load the best checkpoint."""
        best_path = self.directory / "best.pkl"
        if best_path.exists():
            return load_checkpoint(str(best_path))
        return self.load_latest()
