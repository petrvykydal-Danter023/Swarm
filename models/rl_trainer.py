"""
POLYMORPH Engine - RL Trainer

Training orchestration for multi-agent swarm using Stable Baselines3 PPO.
Supports vectorized environments and generates visualization GIFs.
"""

import os
import uuid
from pathlib import Path
from typing import Any

import imageio
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from universal_env import UniversalSwarmEnv, make_env


class SwarmRewardCallback(BaseCallback):
    """Callback to track mean reward during training."""
    
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.current_episode_reward = 0.0
        self.episodes_count = 0
    
    def _on_step(self) -> bool:
        # Track rewards from infos
        if "infos" in self.locals:
            for info in self.locals["infos"]:
                if "mean_reward" in info:
                    self.current_episode_reward += info["mean_reward"]
        
        # Check for episode end
        if "dones" in self.locals:
            for i, done in enumerate(self.locals["dones"]):
                if done:
                    self.episode_rewards.append(self.current_episode_reward)
                    self.episodes_count += 1
                    self.current_episode_reward = 0.0
        
        return True


def flatten_observations(obs: np.ndarray) -> np.ndarray:
    """
    Flatten multi-agent observations to work with single policy.
    
    For vectorized envs, obs shape is (num_envs, num_agents, obs_dim).
    We flatten to (num_envs * num_agents, obs_dim) for the policy,
    then reshape actions back.
    """
    if len(obs.shape) == 3:
        # (num_envs, num_agents, obs_dim) -> (num_envs, num_agents * obs_dim)
        num_envs, num_agents, obs_dim = obs.shape
        return obs.reshape(num_envs, num_agents * obs_dim)
    return obs


def unflatten_observations(flat_obs: np.ndarray, num_agents: int) -> np.ndarray:
    """Unflatten observations back to (num_envs, num_agents, obs_dim)."""
    if len(flat_obs.shape) == 2:
        num_envs, flat_dim = flat_obs.shape
        obs_dim = flat_dim // num_agents
        return flat_obs.reshape(num_envs, num_agents, obs_dim)
    return flat_obs


class FlattenedSwarmEnv(UniversalSwarmEnv):
    """
    Wrapper that flattens multi-agent observations/actions for SB3 compatibility.
    
    SB3 expects (obs_dim,) observations and (action_dim,) actions.
    This wrapper flattens (num_agents, obs_dim) -> (num_agents * obs_dim,)
    and similarly for actions.
    """
    
    def __init__(self, config: dict, render_mode: str = "rgb_array"):
        super().__init__(config, render_mode)
        
        # Get original space dimensions - use dynamic obs_dim from sensor config
        if self.observation_type == "spatial":
            self.flat_obs_dim = self.num_agents * self.obs_dim  # Dynamic from sensors!
        else:  # grid
            orig_obs_shape = (self.num_agents, 5, 5)
            self.flat_obs_dim = self.num_agents * 5 * 5
        
        # Flatten observation space
        from gymnasium import spaces
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.flat_obs_dim,), dtype=np.float32
        )
        
        # Flatten action space
        if self.action_space_type == "continuous":
            self.flat_action_dim = self.num_agents * 2
            self.action_space = spaces.Box(
                low=-1.0, high=1.0, shape=(self.flat_action_dim,), dtype=np.float32
            )
        else:  # discrete - use MultiDiscrete flattened
            # For discrete, we use a single categorical that encodes all agent actions
            # This is a simplification - in practice, you might want to use MultiDiscrete
            self.action_space = spaces.MultiDiscrete([6] * self.num_agents)
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs.flatten(), info
    
    def step(self, actions):
        # Reshape actions back to (num_agents, action_dim)
        if self.action_space_type == "continuous":
            actions = np.array(actions).reshape(self.num_agents, 2)
        else:
            actions = np.array(actions).flatten()
        
        obs, rewards, terminated, truncated, info = super().step(actions)
        
        # Return mean reward for the swarm (SB3 expects scalar)
        mean_reward = float(np.mean(rewards))
        info["agent_rewards"] = rewards.tolist()
        
        return obs.flatten(), mean_reward, terminated, truncated, info


def make_flattened_env(config: dict):
    """Factory function for creating FlattenedSwarmEnv instances."""
    def _init():
        return FlattenedSwarmEnv(config)
    return _init


def train_task(config: dict) -> dict:
    """
    Train a swarm using PPO based on the provided configuration.
    
    Args:
        config: Configuration dictionary containing:
            - task_name: str
            - observation_type: "spatial" | "grid"
            - action_space_type: "continuous" | "discrete"
            - env_params: dict with world parameters
            - reward_code: str with Python reward function
            - training_params: dict with PPO hyperparameters
    
    Returns:
        dict with:
            - status: "done" | "error"
            - video_path: path to generated GIF
            - metrics: training metrics
    """
    
    task_name = config.get("task_name", "unnamed")
    training_params = config.get("training_params", {})
    
    # Extract training parameters with defaults
    total_timesteps = training_params.get("total_timesteps", 10000)
    learning_rate = training_params.get("learning_rate", 3e-4)
    gamma = training_params.get("gamma", 0.99)
    batch_size = training_params.get("batch_size", 64)
    n_envs = training_params.get("n_envs", 4)
    
    # Model persistence parameters for transfer learning
    save_model_path = training_params.get("save_model_path", None)
    load_pretrained_path = training_params.get("load_pretrained_path", None)
    
    print(f"Starting training for task: {task_name}")
    print(f"  Total timesteps: {total_timesteps}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Gamma: {gamma}")
    print(f"  Batch size: {batch_size}")
    print(f"  Parallel environments: {n_envs}")
    if load_pretrained_path:
        print(f"  📦 Loading pretrained model from: {load_pretrained_path}")
    if save_model_path:
        print(f"  💾 Will save model to: {save_model_path}")
    
    # GPU/Device configuration
    device = training_params.get("device", "auto")
    if device == "auto":
        # Auto-detect: prefer CUDA if available
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            print(f"  🚀 GPU detected: {torch.cuda.get_device_name(0)}")
        else:
            device = "cpu"
            print(f"  💻 Using CPU (no GPU detected)")
    else:
        print(f"  🔧 Device: {device}")
    
    try:
        # Create vectorized environment using SB3's DummyVecEnv
        env = DummyVecEnv([make_flattened_env(config) for _ in range(n_envs)])
        
        # Create or load PPO model
        if load_pretrained_path and Path(load_pretrained_path).exists():
            # Fine-tuning: Load existing model and set new env
            print(f"Loading pretrained model...")
            model = PPO.load(load_pretrained_path, env=env, device=device)
            # Update hyperparameters for fine-tuning
            model.learning_rate = learning_rate
            model.gamma = gamma
            model.batch_size = batch_size
            print(f"Pretrained model loaded! Fine-tuning with new parameters.")
        else:
            # Train from scratch
            if load_pretrained_path:
                print(f"Warning: Pretrained model not found at {load_pretrained_path}, training from scratch.")
            model = PPO(
                "MlpPolicy",
                env,
                learning_rate=learning_rate,
                gamma=gamma,
                batch_size=batch_size,
                n_steps=2048,
                ent_coef=0.01,
                verbose=1,
                device=device,  # GPU or CPU
            )
        
        # Create callback for tracking
        callback = SwarmRewardCallback()
        
        # Train
        print("Training...")
        model.learn(total_timesteps=total_timesteps, callback=callback)
        print("Training complete!")
        
        # Save model if path specified
        if save_model_path:
            # Create models directory if needed
            save_path = Path(save_model_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            model.save(str(save_path))
            print(f"💾 Model saved to: {save_path}")
        
        # Close training env
        env.close()
        
        # Evaluation and GIF generation
        print("Generating evaluation video...")
        eval_env = FlattenedSwarmEnv(config)
        
        frames = []
        obs, info = eval_env.reset()
        max_eval_steps = config.get("training_params", {}).get("max_episode_steps", 500)
        total_reward = 0.0
        
        for step in range(max_eval_steps):
            # Get action from trained model
            action, _ = model.predict(obs, deterministic=True)
            
            # Step environment
            obs, reward, terminated, truncated, info = eval_env.step(action)
            total_reward += reward
            
            # Render and save frame
            frame = eval_env.render()
            frames.append(frame)
            
            if terminated or truncated:
                break
        
        eval_env.close()
        
        # Save GIF
        videos_dir = Path(__file__).parent.parent / "videos"
        videos_dir.mkdir(exist_ok=True)
        
        video_filename = f"{task_name}_{uuid.uuid4().hex[:8]}.gif"
        video_path = videos_dir / video_filename
        
        # Save as GIF with imageio
        imageio.mimsave(str(video_path), frames, fps=30, loop=0)
        print(f"Video saved to: {video_path}")
        
        # Calculate metrics
        mean_episode_reward = np.mean(callback.episode_rewards) if callback.episode_rewards else 0.0
        
        return {
            "status": "done",
            "video_path": str(video_path),
            "model_path": str(save_model_path) if save_model_path else None,
            "metrics": {
                "mean_reward": float(mean_episode_reward),
                "eval_reward": float(total_reward),
                "eval_steps": len(frames),
                "episodes": callback.episodes_count,
                "total_timesteps": total_timesteps,
            }
        }
        
    except Exception as e:
        import traceback
        error_msg = f"Training error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        return {
            "status": "error",
            "error": error_msg,
            "video_path": None,
            "metrics": {}
        }


if __name__ == "__main__":
    # Test with a simple config
    test_config = {
        "task_name": "test_task",
        "description": "Test training",
        "observation_type": "spatial",
        "action_space_type": "continuous",
        "env_params": {
            "world_width": 50,
            "world_height": 50,
            "num_agents": 5,
            "physics": {
                "gravity_y": 0.0,
                "friction": 0.1,
                "time_step": 0.1
            },
            "special_objects": [
                {"type": "goal", "x": 45, "y": 45}
            ]
        },
        "reward_code": """
# Simple distance-based reward
goal = env_state['goals'][0] if env_state['goals'] else None
if goal:
    dx = goal['x'] - agent['x']
    dy = goal['y'] - agent['y']
    distance = math.sqrt(dx*dx + dy*dy)
    reward = -distance / 100.0  # Negative distance as reward
    if distance < 5:
        reward = 10.0  # Bonus for reaching goal
""",
        "training_params": {
            "algo": "PPO",
            "total_timesteps": 1000,
            "learning_rate": 3e-4,
            "gamma": 0.99,
            "batch_size": 64,
            "max_episode_steps": 200
        }
    }
    
    result = train_task(test_config)
    print(f"Result: {result}")
