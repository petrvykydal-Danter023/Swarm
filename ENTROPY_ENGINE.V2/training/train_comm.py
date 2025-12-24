import sys
import os
import time

# Add the module root to path
V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(V2_ROOT)

from env.entropy_env import EntropyEnv
from training.multicore_wrapper import AsyncVectorizedEntropyEnv
from training.callbacks import GifRecorderCallback, RichLoggerCallback, EntropySchedulingCallback, CurriculumCallback, VocabLoggerCallback


from shared.logger import RichLogger
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import VecNormalize
import wandb
from wandb.integration.sb3 import WandbCallback

def main():
    print("Initializing Communication Training (Phase 1: DIAL)...")
    
    # Configuration
    N_ENVS = 10 
    AGENTS_PER_ENV = 5 # Start with smaller teams for comms
    TOTAL_TIMESTEPS = 2_000_000
    
    config = {
        "policy_type": "MlpLstmPolicy",
        "total_timesteps": TOTAL_TIMESTEPS, 
        "n_agents_total": N_ENVS * AGENTS_PER_ENV,
        "n_envs": N_ENVS,
        "env_name": "Entropy-Communication-Phase1",
        "algo": "RecurrentPPO",
        "device": "cuda",
        "ent_coef_start": 0.02, # Lower entropy to allow grounding to work
        "ent_coef_end": 0.01
    }
    
    run_name = f"comm_phase1_{int(time.time())}"
    print(f"🔹 Starting Run: {run_name}")

    # Initialize WandB
    run = wandb.init(
        project="entropy-communication",
        dir=os.path.join(V2_ROOT, "wandb"),
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
        name=run_name,
        id=run_name
    )
    
    # Training Environment
    print(f"Spawning {N_ENVS} processes...")
    vec_env = AsyncVectorizedEntropyEnv(
        n_envs=N_ENVS, 
        agents_per_env=AGENTS_PER_ENV
    )
    # Normalize Observations and Rewards
    vec_env = VecNormalize(vec_env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # Evaluation Environment
    eval_env = EntropyEnv(nr_agents=AGENTS_PER_ENV, render_mode="rgb_array")
    
    # Logger
    rich_logger = RichLogger(config["total_timesteps"])
    
    # Callbacks
    video_path = os.path.join(V2_ROOT, "videos")
    models_path = os.path.join(V2_ROOT, "models")
    
    callbacks = [
        RichLoggerCallback(rich_logger),
        GifRecorderCallback(eval_env, save_path=video_path, name_prefix=run_name),
        WandbCallback(gradient_save_freq=1000, model_save_path=os.path.join(models_path, run_name), verbose=2),
        EntropySchedulingCallback(
            high_ent=config["ent_coef_start"], 
            low_ent=config["ent_coef_end"], 
            total_timesteps=config["total_timesteps"]
        ),
        CurriculumCallback(
            total_timesteps=config["total_timesteps"]
        ),
        VocabLoggerCallback()
    ]
    
    # Model Initialization (New Model for Phase 1)
    print("Initializing New RecurrentPPO Model...")
    model = RecurrentPPO(
        config["policy_type"],
        vec_env,
        verbose=1,
        tensorboard_log=os.path.join(V2_ROOT, "runs"),
        learning_rate=3e-4,
        ent_coef=config["ent_coef_start"], # Initial value, updated by callback
        device=config["device"]
    )
    
    print(f"Starting Training ({run_name})...")
    try:
        model.learn(
            total_timesteps=config["total_timesteps"], 
            callback=callbacks,
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
    model.save(os.path.join(models_path, f"{run_name}_final"))
    print("Done.")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
