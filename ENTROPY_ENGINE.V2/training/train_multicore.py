import sys
import os
import time

# Add the module root to path
V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(V2_ROOT)

from env.entropy_env import EntropyEnv
from training.multicore_wrapper import AsyncVectorizedEntropyEnv
from training.callbacks import GifRecorderCallback, RichLoggerCallback
from shared.logger import RichLogger
from sb3_contrib import RecurrentPPO
import wandb
from wandb.integration.sb3 import WandbCallback

def main():
    print("Initializing Multicore Environment...")
    
    # 8 cores * 10 agents = 80 agents learning in parallel
    N_ENVS = 8 
    AGENTS_PER_ENV = 10
    
    # Init WandB
    config = {
        "policy_type": "MlpLstmPolicy",
        "total_timesteps": 1_000_000, # Increased timesteps for faster training
        "n_agents_total": N_ENVS * AGENTS_PER_ENV,
        "n_envs": N_ENVS,
        "env_name": "EntropyEnv-Navigation-Multicore",
        "algo": "RecurrentPPO",
        "device": "cuda" # Ensure GPU is used
    }
    
    run = wandb.init(
        project="entropy-engine-v2",
        dir=os.path.join(V2_ROOT, "wandb"),
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        name=f"multicore_{int(time.time())}"
    )
    
    # Training Environment (Multicore)
    print(f"Spawning {N_ENVS} processes with {AGENTS_PER_ENV} agents each...")
    vec_env = AsyncVectorizedEntropyEnv(
        n_envs=N_ENVS, 
        agents_per_env=AGENTS_PER_ENV
    )
    
    # Evaluation Environment (Single process for rendering)
    eval_env = EntropyEnv(nr_agents=5, render_mode="rgb_array")
    
    # Setup Logger
    rich_logger = RichLogger(config["total_timesteps"])
    
    # Setup Callbacks
    video_path = os.path.join(V2_ROOT, "videos")
    gif_callback = GifRecorderCallback(eval_env, save_path=video_path, name_prefix=f"multicore_{run.id}")
    log_callback = RichLoggerCallback(rich_logger)
    
    models_path = os.path.join(V2_ROOT, "models")
    wandb_callback = WandbCallback(
        gradient_save_freq=1000,
        model_save_path=os.path.join(models_path, run.id),
        verbose=2
    )
    
    print(f"Observation Space: {vec_env.observation_space.shape}")
    print(f"Action Space: {vec_env.action_space.shape}")
    
    # Model
    print("Creating RecurrentPPO (LSTM) Model on GPU...")
    runs_path = os.path.join(V2_ROOT, "runs")
    
    model = RecurrentPPO(
        "MlpLstmPolicy", 
        vec_env, 
        verbose=1, 
        learning_rate=3e-4, 
        n_steps=512, # 512 * 80 agents = 40960 steps per update? No, n_steps is per env.
                     # SB3: n_steps * n_envs = buffer size.
                     # Here n_envs = 80 (agents).
                     # So 512 * 80 = 40,960 transitions per update.
        batch_size=4096, # Huge batch size for GPU efficiency
        gamma=0.99,
        tensorboard_log=os.path.join(runs_path, run.id),
        policy_kwargs={
            "lstm_hidden_size": 256,
            "n_lstm_layers": 1,
            "enable_critic_lstm": True
        }
    )
    
    print("Starting Training (Multicore)...")
    try:
        model.learn(
            total_timesteps=config["total_timesteps"], 
            callback=[log_callback, gif_callback, wandb_callback],
            progress_bar=False 
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        print("Closing environments...")
        eval_env.close()
        vec_env.close()
        run.finish()
    
    print("Saving Model...")
    model.save(os.path.join(models_path, f"ppo_multicore_entropy_v2_{run.id}"))
    
    print("Done.")

if __name__ == "__main__":
    # Support for Windows Multiprocessing
    import multiprocessing
    multiprocessing.freeze_support()
    main()
