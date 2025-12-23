import sys
import os
import time

# Add the module root to path
# This file is in ENTROPY_ENGINE.V2/training/train_v1.py
# We want to add ENTROPY_ENGINE.V2 to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.entropy_env import EntropyEnv
from training.custom_wrapper import PettingZooToVecEnv
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

def main():
    print("Initializing Environment...")
    # Parallel Env
    parallel_env = EntropyEnv(nr_agents=10)
    
    # Vector Wrapper (Parameter Sharing)
    vec_env = PettingZooToVecEnv(parallel_env)
    
    print(f"Observation Space: {vec_env.observation_space.shape}")
    print(f"Action Space: {vec_env.action_space.shape}")
    
    # Model
    print("Creating PPO Model...")
    model = PPO(
        "MlpPolicy", 
        vec_env, 
        verbose=1, 
        learning_rate=3e-4, 
        n_steps=2048, 
        batch_size=256,
        gamma=0.99,
        tensorboard_log="./tensorboard_logs/"
    )
    
    print("Starting Training...")
    try:
        model.learn(total_timesteps=100_000, progress_bar=True)
    except KeyboardInterrupt:
        print("Training interrupted.")
    
    print("Saving Model...")
    model.save("ppo_entropy_v2")
    
    print("Done.")

if __name__ == "__main__":
    main()
