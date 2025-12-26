"""
Entropy Engine V3 - WandB Logging
Weights & Biases integration for experiment tracking.
"""
import wandb
from typing import Dict, Any, Optional
from pathlib import Path

def setup_wandb(
    project: str = "entropy-v3",
    name: str = None,
    config: Dict = None,
    mode: str = "online",
    tags: list = None,
    group: str = None
) -> wandb.run:
    """
    Initialize WandB run.
    
    Args:
        project: WandB project name
        name: Run name (auto-generated if None)
        config: Configuration dict to log
        mode: "online", "offline", or "disabled"
        tags: List of tags for filtering
        group: Group name for related runs
        
    Returns:
        WandB run object
    """
    run = wandb.init(
        project=project,
        name=name,
        config=config,
        mode=mode,
        tags=tags or [],
        group=group,
        reinit=True
    )
    print(f"[WandB] Run initialized: {run.name} ({run.url})")
    return run


def log_metrics(metrics: Dict[str, float], step: int = None, commit: bool = True):
    """
    Log metrics to WandB.
    
    Args:
        metrics: Dict of metric_name -> value
        step: Optional step number
        commit: If True, flush immediately
    """
    wandb.log(metrics, step=step, commit=commit)


def log_episode(
    episode: int,
    reward_mean: float,
    reward_std: float,
    episode_length: float,
    goal_reached_rate: float,
    fps: float = None,
    extra: Dict = None
):
    """
    Log a complete episode summary.
    """
    metrics = {
        "train/episode": episode,
        "train/reward_mean": reward_mean,
        "train/reward_std": reward_std,
        "train/episode_length": episode_length,
        "env/goal_reached_rate": goal_reached_rate,
    }
    if fps:
        metrics["system/fps"] = fps
    if extra:
        metrics.update(extra)
    
    wandb.log(metrics)


def log_losses(
    step: int,
    policy_loss: float,
    value_loss: float,
    entropy: float,
    total_loss: float = None,
    comm_loss: float = None
):
    """Log training losses."""
    metrics = {
        "loss/policy": policy_loss,
        "loss/value": value_loss,
        "loss/entropy": entropy,
    }
    if total_loss is not None:
        metrics["loss/total"] = total_loss
    if comm_loss is not None:
        metrics["loss/communication"] = comm_loss
    
    wandb.log(metrics, step=step)


def log_artifact(
    path: str,
    name: str,
    artifact_type: str = "model",
    metadata: Dict = None
):
    """
    Log a file as a WandB artifact.
    
    Args:
        path: Path to file
        name: Artifact name
        artifact_type: "model", "dataset", "result", etc.
        metadata: Optional metadata
    """
    artifact = wandb.Artifact(name, type=artifact_type, metadata=metadata)
    artifact.add_file(path)
    wandb.log_artifact(artifact)
    print(f"[WandB] Artifact logged: {name}")


class WandBCallback:
    """
    Callback class for easy integration with training loops.
    
    Usage:
        callback = WandBCallback(log_interval=10)
        
        for step in range(num_steps):
            metrics = train_step()
            callback.on_step(step, metrics)
    """
    def __init__(
        self,
        log_interval: int = 10,
        save_interval: int = 1000,
        checkpoint_dir: str = None
    ):
        self.log_interval = log_interval
        self.save_interval = save_interval
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
    def on_step(self, step: int, metrics: Dict[str, float]):
        """Called after each training step."""
        if step % self.log_interval == 0:
            wandb.log(metrics, step=step)
    
    def on_episode_end(
        self, 
        episode: int, 
        reward: float, 
        length: int,
        goal_rate: float
    ):
        """Called after each episode."""
        log_episode(
            episode=episode,
            reward_mean=reward,
            reward_std=0.0,  # Would need history for std
            episode_length=length,
            goal_reached_rate=goal_rate
        )
    
    def on_save(self, step: int, checkpoint_path: str, metrics: Dict = None):
        """Called when a checkpoint is saved."""
        if wandb.run:
            log_artifact(
                path=checkpoint_path,
                name=f"checkpoint-{step}",
                artifact_type="model",
                metadata={"step": step, **(metrics or {})}
            )

def finish():
    """Finish WandB run."""
    wandb.finish()
