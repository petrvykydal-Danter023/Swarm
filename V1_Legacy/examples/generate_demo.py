
import gymnasium as gym
import numpy as np
import imageio
import json
import os
import sys
import argparse

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from topdown_engine import TopDownSwarmEnv
from stable_baselines3 import PPO

def generate_video(config_path, model_path, output_path):
    with open(config_path, "r") as f:
        config = json.load(f)
        
    print(f"🎥 Recording Demo from {config_path}...")
    
    # Enable rendering
    config["env_params"]["render_mode"] = "rgb_array"
    env = TopDownSwarmEnv(config)
    
    try:
        model = PPO.load(model_path)
    except:
        print("Model not found, running random agent")
        model = None
        
    frames = []
    obs, _ = env.reset()
    
    for _ in range(500): # Longer for search
        if model:
            action, _ = model.predict(obs)
        else:
            action = [env.action_space.sample() for _ in range(env.num_agents)]
            
        obs, rewards, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        
        if terminated or truncated:
            obs, _ = env.reset()
            
    print(f"✅ Saving to {output_path}")
    imageio.mimsave(output_path, frames, fps=30)

if __name__ == "__main__":
    if len(sys.argv) > 1:
        generate_video(sys.argv[1], sys.argv[2], sys.argv[3])
    else:
        # Default to search demo
        generate_video(
            "examples/search_training.json",
            "models/saved/search_demo_model",
            "videos/blind_search_demo.gif"
        )
