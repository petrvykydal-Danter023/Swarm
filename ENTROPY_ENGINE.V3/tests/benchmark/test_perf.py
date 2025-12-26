import time
import pytest
import jax
import jax.numpy as jnp
from entropy.core.physics import physics_step

@pytest.mark.benchmark
def test_physics_throughput(small_world, rng):
    """Benchmark: measure FPS for 1000 agents."""
    # Scale up world
    n = 1000
    w = small_world.replace(
        agent_positions=jnp.zeros((n, 2)),
        agent_velocities=jnp.zeros((n, 2)),
        agent_angles=jnp.zeros(n), 
        agent_ang_velocities=jnp.zeros(n),
        agent_radii=jnp.ones(n)*10,
        goal_positions=jnp.zeros((n, 2)),
        goal_radii=jnp.ones(n)*15,
        goal_reached=jnp.zeros(n, dtype=bool)
        # Note: arrays need full resizing, this is a bit hacky for a concise test
        # Ideally use proper create_world function
    )
    
    # Just mock arrays for speed
    def step_fn(s, a):
        return physics_step(s, a)
    
    step_jit = jax.jit(step_fn)
    actions = jnp.zeros((n, 2))
    
    # Warmup
    for _ in range(100):
        w = step_jit(w, actions)
    jax.block_until_ready(w.agent_positions)
    
    # Bench
    start = time.perf_counter()
    steps = 1000
    for _ in range(steps):
        w = step_jit(w, actions)
    jax.block_until_ready(w.agent_positions)
    elapsed = time.perf_counter() - start
    
    fps = steps / elapsed
    print(f"\nOptimization Perf: {fps:.0f} FPS (1000 agents)")
    
    # Expect > 1000 FPS on decent hardware (even CPU for simple physics)
    # Be conservative for CI
    assert fps > 100, "Performance regression! <100 FPS"
