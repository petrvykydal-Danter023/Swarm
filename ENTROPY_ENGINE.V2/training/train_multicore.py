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
    
    # Ryzen 5 5600X (6 Cores, 12 Threads)
    # We use 10 processes to leave 2 threads for OS/GPU driver overhead
    N_ENVS = 10 
    AGENTS_PER_ENV = 10
    # Total Agents = 100
    
    # Init WandB
    config = {
        "policy_type": "MlpLstmPolicy",
        "total_timesteps": 2_000_000, # Extended training for larger swarm
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
    
    
    # Model - Load existing for fine-tuning
    print("Loading Existing RecurrentPPO Model for Fine-tuning...")
    runs_path = os.path.join(V2_ROOT, "runs")
    
    # Path to the model we just trained
    prev_model = "ppo_multicore_entropy_v2_xuoj0qxz"
    model_path = os.path.join(models_path, prev_model)
    
    model = RecurrentPPO.load(
        model_path, 
        env=vec_env,
        # Update hyperparameters if needed (e.g. learning rate)
        learning_rate=3e-4,
        # Important: Ensure tensorboard log continues or starts new
        tensorboard_log=os.path.join(runs_path, run.id),
        # We need to set device manually sometimes when loading
        device="cuda"
    )
    
    print(f"Loaded: {prev_model}")
    
    print("Starting Training (Multicore - Fine-tuning)...")
    try:
        model.learn(
            total_timesteps=config["total_timesteps"], 
            callback=[log_callback, gif_callback, wandb_callback],
            progress_bar=False,
            reset_num_timesteps=False # Continue counting steps
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
