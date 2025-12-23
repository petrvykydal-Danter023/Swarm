import sys
import os
import time

# Add the module root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.entropy_env import EntropyEnv
from training.custom_wrapper import PettingZooToVecEnv
from training.callbacks import GifRecorderCallback, RichLoggerCallback
from shared.logger import RichLogger
from sb3_contrib import RecurrentPPO
import wandb
from wandb.integration.sb3 import WandbCallback

def main():
    print("Initializing Environment...")
    
    # Init WandB
    config = {
        "policy_type": "MlpLstmPolicy",
        "total_timesteps": 200_000,
        "n_agents": 10,
        "env_name": "EntropyEnv-Navigation",
        "algo": "RecurrentPPO"
    }
    
    run = wandb.init(
        project="entropy-engine-v2",
        config=config,
        sync_tensorboard=True, # Auto-upload sb3 tensorboard logs
        monitor_gym=True,      # Auto-upload videos if Gym monitor is used (we use custom though)
        save_code=True,
    )
    
    # Training Environment
    parallel_env = EntropyEnv(nr_agents=config["n_agents"], render_mode=None) 
    vec_env = PettingZooToVecEnv(parallel_env)
    
    # Evaluation Environment
    eval_env = EntropyEnv(nr_agents=5, render_mode="rgb_array")
    
    # Setup Logger
    rich_logger = RichLogger(config["total_timesteps"])
    
    # Setup Callbacks
    gif_callback = GifRecorderCallback(eval_env, save_path="./videos", name_prefix=f"lstm_{run.id}")
    log_callback = RichLoggerCallback(rich_logger)
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=f"models/{run.id}",
        verbose=2
    )
    
    print(f"Observation Space: {vec_env.observation_space.shape}")
    print(f"Action Space: {vec_env.action_space.shape}")
    
    # Model
    print("Creating RecurrentPPO (LSTM) Model...")
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        vec_env, 
        verbose=1, 
        learning_rate=3e-4, 
        n_steps=2048, 
        batch_size=256,
        gamma=0.99,
        tensorboard_log=f"runs/{run.id}",
        policy_kwargs={
            "lstm_hidden_size": 256,
            "n_lstm_layers": 1,
            "enable_critic_lstm": True
        }
    )
    
    print("Starting Training (LSTM)...")
    try:
        model.learn(
            total_timesteps=config["total_timesteps"], 
            callback=[log_callback, gif_callback, wandb_callback],
            progress_bar=False 
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        eval_env.close()
        vec_env.close()
        run.finish()
    
    print("Saving Model...")
    model.save(f"models/ppo_lstm_entropy_v2_{run.id}")
    
    print("Done.")

if __name__ == "__main__":
    main()
