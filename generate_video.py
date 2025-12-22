import sys
import os
import math
import numpy as np
import imageio
from pathlib import Path
from stable_baselines3 import PPO
from universal_env import UniversalSwarmEnv
from models.swarm_wrapper import SwarmVecEnv

# Config from test_bridge.py (Top-Down Move Test)
config = {
    "task_name": "topdown_move_test",
    "description": "Top-down navigation test - no gaps, just obstacles and goal",
    "observation_type": "spatial",
    "action_space_type": "discrete",
    "env_params": {
        "world_width": 100,
        "world_height": 50,
        "num_agents": 5,
        "physics": {
            "gravity_y": 0.0,
            "friction": 0.1,
            "time_step": 0.1
        },
        "spawn_zone": {
            "x1": 5, "x2": 20,
            "y1": 5, "y2": 45
        },
        "enable_communication": True,
        "sensor_noise_std": 0.05,
        "special_objects": [
            {"type": "obstacle", "x": 40, "y": 25, "radius": 5},
            {"type": "obstacle", "x": 60, "y": 15, "radius": 4},
            {"type": "obstacle", "x": 60, "y": 35, "radius": 4},
            {"type": "goal", "x": 90, "y": 25}
        ],
        "sensors": [
            "position", "velocity", "goal_vector", "obstacle_radar",
            "neighbor_vectors", "neighbor_signals", "grabbing_state", "energy"
        ]
    },
    "reward_code": "reward=0.0" 
}

def generate_video():
    model_path = Path("models/saved/topdown_move_test")
    if not model_path.with_suffix(".zip").exists():
        print(f"Error: Model not found at {model_path}")
        # Try finding recent zip in that dir
        saved_dir = Path("models/saved")
        zips = list(saved_dir.glob("*.zip"))
        if zips:
            print(f"Available models: {[z.name for z in zips]}")
        return

    print(f"Loading model from {model_path}...")
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Initializing environment...")
    env = UniversalSwarmEnv(config)
    vec_env = SwarmVecEnv(env)

    obs = vec_env.reset()
    frames = []
    
    print("Simulating...")
    max_steps = 400
    for i in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec_env.step(action)
        
        # Render every frame (no sub-skipping here beyond env's internal)
        frame = vec_env.render(mode="rgb_array")
        frames.append(frame)
        
        # Check if all reached goal?
        # Env treats 'dones' as episode end, but SwarmVecEnv auto-resets on done.
        # We just want to capture one episode or up to max_steps.
        
        if (i+1) % 50 == 0:
            print(f"Step {i+1}/{max_steps}")

    # Save video
    video_path = Path("videos/proof_topdown.gif")
    video_path.parent.mkdir(exist_ok=True)
    print(f"Saving video to {video_path} ({len(frames)} frames)...")
    imageio.mimsave(video_path, frames, duration=0.05, loop=0) # 20 FPS
    print("Done!")

if __name__ == "__main__":
    generate_video()
