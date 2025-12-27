import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import jax
import jax.numpy as jnp
import time
from entropy.config import ExperimentConfig, IntentConfig, SafetyConfig
from entropy.training.env_wrapper import EntropyGymWrapper

def main():
    print("🚀 Smoke Test: Full Pipeline (Intent + Safety)")
    
    # 1. Configure
    cfg = ExperimentConfig()
    cfg.agent.num_agents = 50
    # Enable Intent Mode
    cfg.intent = IntentConfig(enabled=True)
    # Enable Safety
    cfg.safety = SafetyConfig(enabled=True)
    
    # 2. Initialize Wrapper
    print("Initializing Wrapper...")
    env = EntropyGymWrapper(cfg)
    
    rng = jax.random.PRNGKey(42)
    state, obs = env.reset(rng)
    
    # 3. JIT Compile Warmup
    print("JIT Compiling step()...")
    start_warmup = time.time()
    
    # Action Dim should be larger now
    # 3 dims for Intent (Type, P1, P2) + Rest
    # Direct mode was 2 dims (L, R) + Gate...
    # EnvWrapper calculates action_dim. Let's use it.
    act_dim = env.action_dim
    print(f"Action Dimension: {act_dim}")
    
    fake_actions = jnp.zeros((cfg.agent.num_agents, act_dim))
    
    # Step
    rng, step_rng = jax.random.split(rng)
    _ = env.step(state, fake_actions, step_rng)
    
    end_warmup = time.time()
    print(f"Compilation took: {end_warmup - start_warmup:.4f}s")
    
    # 4. Run Loop
    print("Running 500 steps...")
    start_sim = time.time()
    
    for i in range(500):
        rng, step_rng, act_rng = jax.random.split(rng, 3)
        
        # Random Actions
        # [0]: Type (-1 to 1) -> <0 Velocity, >0 Target
        # [1, 2]: Params (-1 to 1)
        # [3]: Gate
        actions = jax.random.uniform(act_rng, (cfg.agent.num_agents, act_dim), minval=-1.0, maxval=1.0)
        
        state, obs, rewards, done, info = env.step(state, actions, step_rng)
        
        # Block
        state.agent_positions.block_until_ready()
        
    end_sim = time.time()
    duration = end_sim - start_sim
    fps = 500 / duration
    
    print(f"✅ Simulation complete.")
    print(f"Time: {duration:.4f}s")
    print(f"FPS: {fps:.2f} (Full Wrapper Overhead)")
    
    # Check
    if jnp.any(jnp.isnan(state.agent_positions)):
        print("❌ FAILURE: NaNs detected!")
    else:
        print("✅ SUCCESS: System is stable.")

if __name__ == "__main__":
    main()
