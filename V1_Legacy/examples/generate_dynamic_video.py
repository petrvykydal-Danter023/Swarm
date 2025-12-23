
import gymnasium as gym
import numpy as np
import imageio
import json
import os
import sys

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from topdown_engine import TopDownSwarmEnv
from stable_baselines3 import PPO

def generate_video():
    config_path = "examples/dynamic_goal_training.json"
    model_path = "models/saved/dynamic_goal_model"
    video_path = "videos/dynamic_goal_demo.gif"
    
    with open(config_path, "r") as f:
        config = json.load(f)
        
    print("🎥 Recording Dynamic Goal Demo...")
    
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
    
    for _ in range(300):
        if model:
            action, _ = model.predict(obs)
        else:
            action = [env.action_space.sample() for _ in range(env.num_agents)]
            
        obs, rewards, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        
        if terminated or truncated:
            obs, _ = env.reset()
            
    print(f"✅ Saving to {video_path}")
    imageio.mimsave(video_path, frames, fps=30)

if __name__ == "__main__":
    generate_video()
