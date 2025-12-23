
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

def generate_comparison_video(config_path, model_path, output_path):
    # 1. Load Config
    with open(config_path, "r") as f:
        config = json.load(f)
    
    # 2. Setup Environments
    env = TopDownSwarmEnv(config)
    
    max_steps = 400
    frames_untrained = []
    frames_trained = []
    
    print("🎥 Recording Untrained Swarm (Random Actions)...")
    obs, _ = env.reset()
    for _ in range(max_steps):
        # Random actions
        action = env.action_space.sample()
        _, _, _, _, _ = env.step(action)
        frames_untrained.append(env.render())
    env.close()
        
    print("🎥 Recording Trained Swarm...")
    eval_env_inner = TopDownSwarmEnv(config)
    eval_env = SwarmVecEnv(eval_env_inner)
    
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

    print("✂️ Stitching videos side-by-side...")
    min_len = min(len(frames_untrained), len(frames_trained))
    final_frames = []
    
    import cv2
    
    for i in range(min_len):
        f1 = frames_untrained[i]
        f2 = frames_trained[i]
        
        # Add labels
        f1_labeled = f1.copy()
        f2_labeled = f2.copy()
        cv2.putText(f1_labeled, "RANDOM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(f2_labeled, "TRAINED", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        combined = np.concatenate((f1_labeled, f2_labeled), axis=1)
        final_frames.append(combined)

    Path("videos").mkdir(exist_ok=True)
    imageio.mimsave(output_path, final_frames, fps=30, loop=0)
    print(f"✅ Comparison saved to {output_path}")

if __name__ == "__main__":
    # Args: config model output
    if len(sys.argv) >= 4:
        generate_comparison_video(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        # Default for payload push test
        generate_comparison_video(
            "examples/payload_push_training.json",
            "models/saved/payload_push_swarm",
            "videos/payload_push_comparison.gif"
        )
