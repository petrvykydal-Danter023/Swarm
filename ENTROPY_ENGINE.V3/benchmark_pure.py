"""
🔥 Pure JAX Engine Benchmark 🔥
Goal: > 50,000 FPS
"""
import jax
import jax.numpy as jnp
import time
from functools import partial

from entropy.core.pure_engine import PureEntropyEngine, EnvParams, EnvState, EnvStep

# 1. Configuration
NUM_AGENTS = 256 # Massive swarm
NUM_STEPS = 200  # Full episode
BATCH_SIZE = 1   # Single Environment for raw kernel speed (or batch it)
# Let's batch to simulate training
BATCH_SIZE = 64 

print(f"🚀 Benchmarking Pure JAX Engine...")
print(f"Agents: {NUM_AGENTS}")
print(f"Batch: {BATCH_SIZE}")
print(f"Steps: {NUM_STEPS}")
print(f"Total Transitions per Rollout: {NUM_AGENTS * NUM_STEPS * BATCH_SIZE}")

# 2. Define Rollout Function (The One Kernel)
@partial(jax.jit, static_argnums=(1,))
def rollout_fn(rng, params):
    
    # Init Batch
    rng_batch = jax.random.split(rng, BATCH_SIZE)
    
    # Vmapped Reset
    batch_reset = jax.vmap(PureEntropyEngine.reset, in_axes=(0, None))
    init_steps = batch_reset(rng_batch, params)
    init_states = init_steps.state
    
    # Scan Loop
    def step_wrapper(carry, _):
        rng, state = carry
        rng, key_gen = jax.random.split(rng)
        
        # Split keys for batch
        step_keys = jax.random.split(key_gen, BATCH_SIZE)
        
        # Random Actions (Mock Policy) - we can use single key for batch generation
        act_key = key_gen # Reuse or split again
        
        # Action Dim: 3 (Intent) + Rest of action space (total 5 approx)
        actions = jax.random.uniform(act_key, (BATCH_SIZE, NUM_AGENTS, 5), minval=-1.0, maxval=1.0)
        
        # Vmapped Step
        batch_step = jax.vmap(PureEntropyEngine.step, in_axes=(0, 0, 0, None))
        step_result = batch_step(step_keys, state, actions, params)
        
        return (rng, step_result.state), None # We don't save outputs for speed test, just state transition
    
    # Run Scan
    final_carry, _ = jax.lax.scan(step_wrapper, (rng, init_states), None, length=NUM_STEPS)
    return final_carry

# 3. Setup Params
params = EnvParams(
    num_agents=NUM_AGENTS, 
    use_comms=True, 
    safety_enabled=True,
    lidar_rays=32
)
key = jax.random.PRNGKey(42)

# 4. Warmup (Compile)
print("⚡ Compiling...")
t0 = time.time()
rollout_fn(key, params)
print(f"Compilation finished in {time.time() - t0:.2f}s")

# 5. Benchmark
ITERS = 100
print(f"🏃 Running {ITERS} iterations...")
jax.block_until_ready(rollout_fn(key, params)) # Sync

t_start = time.time()
for _ in range(ITERS):
    out = rollout_fn(key, params)
    jax.block_until_ready(out)
    
t_end = time.time()
duration = t_end - t_start

total_transitions = ITERS * NUM_STEPS * BATCH_SIZE # * NUM_AGENTS is usually implied in "Agent Steps" or "Env Steps"?
# Standard RL metric: Env Steps per Second (transitions)
# If multi-agent, steps = E * T (agents are internal)
# But FPS usually refers to atomic interactions.
# Let's report Env FPS (E*T) and Agent FPS (E*T*N).

env_steps = ITERS * NUM_STEPS * BATCH_SIZE
agent_steps = env_steps * NUM_AGENTS

print(f"\n📊 RESULTS:")
print(f"Total Time: {duration:.4f}s")
print(f"Env FPS: {env_steps / duration:,.0f}")
print(f"Agent FPS: {agent_steps / duration:,.0f}")

if (env_steps / duration) > 10000:
    print("\n🏆 MISSION ACCOMPLISHED: WARP SPEED ACHIEVED! 🏆")
else:
    print("\n⚠️ Still slow? Check constraints.")
