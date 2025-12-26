"""
Verification Script for Phase 5: Training
Runs a short PPO training loop to verify JIT compilation and execution.
"""
import jax
import time
import traceback
from entropy.training.ppo import PPOConfig, make_train

def verify_training():
    print("Starting Training Verification...")
    
    try:
        # 1. Config
        config = PPOConfig(
            num_agents=10,
            total_timesteps=2048, # Short run
            num_steps=128,
            num_minibatches=2,
            update_epochs=2,
            seed=42
        )
        
        # 2. Make Train Function
        print("Compiling Training Loop (this might take a minute)...")
        start_time = time.time()
        train_fn = make_train(config)
        
        # JIT Compile (by calling it)
        rng = jax.random.PRNGKey(0)
        
        # Run
        runner_state, metrics = train_fn(rng)
        
        # Force computation
        jax.block_until_ready(metrics)
        end_time = time.time()
        
        print(f"Compilation and Execution finished in {end_time - start_time:.2f}s")
        print(f"Metrics (Last Update): {metrics}")
        
    except Exception:
        with open("traceback.txt", "w", encoding="utf-8") as f:
            f.write(traceback.format_exc())
        print("Error occurred. Check traceback.txt")

if __name__ == "__main__":
    verify_training()
