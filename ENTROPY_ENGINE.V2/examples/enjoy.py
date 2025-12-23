import sys
import os
import time
import pygame
import numpy as np

# Add the module root to path
# Assuming we run this from ENTROPY_ENGINE.V2/examples/ or root, we need to find V2_ROOT
# If file is in examples/, V2_ROOT is parent
V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(V2_ROOT)

from env.entropy_env import EntropyEnv
from training.custom_wrapper import PettingZooToVecEnv
from sb3_contrib import RecurrentPPO

def main():
    print("Initializing Live Inference Environment...")
    pygame.init()
    
    # Live View Settings
    N_AGENTS_SHOW = 10
    RENDER_MODE = "human" # Opens Pygame window
    
    # Create Env
    parallel_env = EntropyEnv(nr_agents=N_AGENTS_SHOW, render_mode=RENDER_MODE) 
    
    # Wrap it like in training (VecEnv) so shapes match
    vec_env = PettingZooToVecEnv(parallel_env)
    
    # Load Model
    # Explicitly using the latest multicore model we found
    model_name = "ppo_multicore_entropy_v2_map1ykof" 
    model_path = os.path.join(V2_ROOT, "models", model_name)
    
    print(f"Loading Model: {model_path} ...")
    try:
        model = RecurrentPPO.load(model_path, env=vec_env)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    print("Model Loaded! Starting Loop... (Press CTRL+C or Close Window to stop)")
    
    obs = vec_env.reset()
    
    # LSTM states
    lstm_states = None
    
    # For consistent frame rate
    clock = pygame.time.Clock()
    episode_rewards = 0
    
    running = True
    while running:
        # Handle Pygame Events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Predict Action
        # deterministic=True -> Agent uses best action (no exploration noise)
        action, lstm_states = model.predict(obs, state=lstm_states, deterministic=True)
        
        # Step Env
        obs, rewards, dones, infos = vec_env.step(action)
        
        episode_rewards += np.sum(rewards)
        
        # VecEnv automatically resets if all agents done, 
        # but here we have infinite horizon usually (respawns)
        
        # Render happens inside step() for "human" mode in our EntropyEnv, 
        # but we need to ensure tick limit
        clock.tick(60) # Limit to 60 FPS for smooth viewing
        
        # Optional: Print info occasionally
        # print(f"Rew: {rewards[0]:.2f}", end="\r")

    print("Closing...")
    vec_env.close()

if __name__ == "__main__":
    main()
