
import json
import numpy as np
import imageio
from pathlib import Path
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from topdown_engine import TopDownSwarmEnv
from stable_baselines3 import PPO
from models.swarm_wrapper import SwarmVecEnv

def generate_comparison_video():
    # 1. Load Config
    config_path = "examples/safety_training.json"
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 2. Setup Environments
    # We need to wrap environment for Stable Baselines compatibility (even for untrained check if we want same interface)
    # But for untrained random, we can just use base env.
    env = TopDownSwarmEnv(config)
    
    max_steps = 300
    frames_untrained = []
    frames_trained = []
    
    print("🎥 Recording Untrained Swarm...")
    obs, _ = env.reset()
    for _ in range(max_steps):
        # Random actions
        action = env.action_space.sample()
        _, _, _, _, _ = env.step(action)
        frames_untrained.append(env.render())
    env.close()
        
    print("🎥 Recording Trained Swarm...")
    # For trained, we need the wrapper because PPO expects flattened obs
    # And we need to match the training environment structure.
    # Training uses ParallelSwarmVecEnv or SwarmVecEnv which flattens.
    
    # We use SwarmVecEnv for single env evaluation
    eval_env_inner = TopDownSwarmEnv(config)
    eval_env = SwarmVecEnv(eval_env_inner)
    
    model_path = "models/saved/safety_demo_model"
    # SB3 automatically adds .zip if missing
    
    try:
        model = PPO.load(model_path, env=eval_env)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    obs = eval_env.reset()
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = eval_env.step(action)
        frames_trained.append(eval_env.render())
        if any(dones):
            break
    eval_env.close()

    print("✂️ Stiching videos side-by-side...")
    # Ensure same length
    min_len = min(len(frames_untrained), len(frames_trained))
    final_frames = []
    
    for i in range(min_len):
        f1 = frames_untrained[i]
        f2 = frames_trained[i]
        
        # Concatenate horizontally
        # Check shapes
        if f1.shape != f2.shape:
            # Resize or crop? Assuming same config, same resolution.
            pass
            
        combined = np.concatenate((f1, f2), axis=1)
        
        # Add labels maybe? (Requires PIL/cv2, avoiding extra heavy deps if possible)
        # We can just rely on the visual difference.
        
        final_frames.append(combined)

    output_path = "videos/comparison_before_after.gif"
    Path("videos").mkdir(exist_ok=True)
    imageio.mimsave(output_path, final_frames, fps=30, loop=0)
    print(f"✅ Comparison saved to {output_path}")

if __name__ == "__main__":
    generate_comparison_video()
