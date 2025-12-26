import jax
import jax.numpy as jnp
import time
from entropy.core.world import create_initial_state
from entropy.core.physics import physics_step

def main():
    print("🔥 Smoke Test: Entropy Physics")
    
    # 1. Initialize
    print("Initializing state...")
    num_agents = 100
    state = create_initial_state(num_agents=num_agents)
    
    # 2. JIT Compile warmup
    print("JIT Compiling...")
    start_warmup = time.time()
    actions = jnp.zeros((num_agents, 2))
    state = physics_step(state, actions) # Trigger compilation
    end_warmup = time.time()
    print(f"Compilation took: {end_warmup - start_warmup:.4f}s")
    
    # 3. Simulation Loop
    print("Running simulation (1000 steps)...")
    rng = jax.random.PRNGKey(0)
    
    start_sim = time.time()
    for _ in range(1000):
        # Random actions
        rng, key = jax.random.split(rng)
        actions = jax.random.uniform(key, (num_agents, 2), minval=-1.0, maxval=1.0)
        state = physics_step(state, actions)
        
        # Block until result is ready (for accurate timing of async dispatch)
        state.agent_positions.block_until_ready()
        
    end_sim = time.time()
    duration = end_sim - start_sim
    fps = 1000 / duration
    
    print(f"✅ Simulation complete.")
    print(f"Time: {duration:.4f}s")
    print(f"FPS: {fps:.2f} (Agents: {num_agents})")
    
    # Check for NaNs
    if jnp.any(jnp.isnan(state.agent_positions)):
        print("❌ FAILURE: NaNs detected in positions!")
    else:
        print("✅ SUCCESS: No NaNs detected")

if __name__ == "__main__":
    main()
