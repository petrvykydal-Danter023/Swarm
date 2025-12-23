"""
Quick test of the new Rich Progress Bar with all features.
"""
import sys
import os

V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(V2_ROOT)

from training.multicore_wrapper import AsyncVectorizedEntropyEnv
from training.callbacks import RichLoggerCallback
from shared.logger import RichLogger
from sb3_contrib import RecurrentPPO

def main():
    print("Testing Enhanced Rich Dashboard...")
    
    N_ENVS = 4
    AGENTS_PER_ENV = 5
    TOTAL_STEPS = 100_000  # Longer to see notifications
    
    config = {
        "total_timesteps": TOTAL_STEPS,
        "n_envs": N_ENVS,
        "agents_per_env": AGENTS_PER_ENV,
        "learning_rate": 3e-4,
        "policy": "MlpLstmPolicy"
    }
    
    vec_env = AsyncVectorizedEntropyEnv(n_envs=N_ENVS, agents_per_env=AGENTS_PER_ENV)
    
    # Create logger with all features
    rich_logger = RichLogger(
        total_timesteps=TOTAL_STEPS,
        run_name="test_dashboard",
        wandb_url="https://wandb.ai/petr-vykydal/entropy-engine-v2/runs/test",
        save_dir=os.path.join(V2_ROOT, "checkpoints"),
        config=config
    )
    log_callback = RichLoggerCallback(rich_logger)
    
    # Load model
    models_path = os.path.join(V2_ROOT, "models")
    model = RecurrentPPO.load(
        os.path.join(models_path, "entropy_v2_1.zip"),
        env=vec_env,
        device="cuda"
    )
    
    print(f"Model loaded. Starting steps: {model.num_timesteps}")
    
    try:
        model.learn(
            total_timesteps=TOTAL_STEPS,
            callback=[log_callback],
            progress_bar=False,
            reset_num_timesteps=False
        )
    except KeyboardInterrupt:
        print("Interrupted!")
    finally:
        vec_env.close()
        
    print("Test Complete!")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
