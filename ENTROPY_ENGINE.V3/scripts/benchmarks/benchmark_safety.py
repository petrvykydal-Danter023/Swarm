import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import jax
import jax.numpy as jnp
import numpy as np
import time
from entropy.config import ExperimentConfig, IntentConfig, SafetyConfig
from entropy.training.env_wrapper import EntropyGymWrapper

def get_heuristic_action(state, action_dim, mode="direct"):
    """
    Simple heuristic: Move towards goal.
    """
    diff = state.goal_positions - state.agent_positions
    dist = jnp.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
    
    if mode == "intent":
        # Output Intent: [Type=1.0 (Target), RelX, RelY, ...]
        # RelX/RelY are in BODY frame? 
        # Wait, process_intent documentation says: "param1 = RelX... Cíl je relativně k agentovi"
        # AND check test_intent logic: 
        # "Angle err = target_angle ... Target Angle (kde je cíl relativně k pohledu agenta)"
        # So yes, we need to transform world diff to body frame.
        
        # World diff
        dx = diff[:, 0]
        dy = diff[:, 1]
        
        # Rotate by -agent_angle
        angles = state.agent_angles
        cos_a = jnp.cos(-angles)
        sin_a = jnp.sin(-angles)
        
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        # Intent Vector: [Type, P1, P2]
        intent_core = jnp.stack([
            jnp.full((state.num_agents,), 1.0), # Target Mode
            local_x,
            local_y
        ], axis=1)
        
        padding = jnp.zeros((state.num_agents, action_dim - 3))
        return jnp.concatenate([intent_core, padding], axis=1)

    else:
        # Direct Mode: Motors
        # Simple Logic: 
        # 1. Turn to goal
        # 2. Drive
        
        # Angle to goal
        goal_angles = jnp.arctan2(diff[:, 1], diff[:, 0])
        angle_diff = goal_angles - state.agent_angles
        # Normalize -pi to pi
        angle_diff = (angle_diff + jnp.pi) % (2 * jnp.pi) - jnp.pi
        
        # Turn
        turn = jnp.clip(angle_diff * 5.0, -1.0, 1.0)
        # Drive (slow down if turning)
        fwd = jnp.clip(1.0 - jnp.abs(turn), 0.0, 1.0)
        
        left = fwd - turn
        right = fwd + turn
        
        motors = jnp.stack([left, right], axis=1)
        padding = jnp.zeros((state.num_agents, action_dim - 2))
        return jnp.concatenate([motors, padding], axis=1)

def run_benchmark(mode_name, steps=500):
    print(f"\n🏃 Running Benchmark: {mode_name}")
    
    # 1. Configure
    cfg = ExperimentConfig()
    cfg.agent.num_agents = 20
    cfg.sim.max_steps = steps
    
    if mode_name == "Direct (Unsafe)":
        cfg.intent = IntentConfig(enabled=False)
        cfg.safety = SafetyConfig(enabled=False)
        mode_key = "direct"
    elif mode_name == "Direct + Safety":
        cfg.intent = IntentConfig(enabled=False)
        cfg.safety = SafetyConfig(enabled=True, log_metrics=True)
        mode_key = "direct"
    elif mode_name == "Hybrid (Intent + Safety)":
        cfg.intent = IntentConfig(enabled=True)
        cfg.safety = SafetyConfig(enabled=True, log_metrics=True)
        mode_key = "intent"
    else:
        raise ValueError(f"Unknown mode: {mode_name}")
        
    # 2. Init
    env = EntropyGymWrapper(cfg)
    rng = jax.random.PRNGKey(42)
    state, obs = env.reset(rng)
    
    # Metrics
    total_collisions = 0 # Need to extract collision info? metrics only track "interventions"
    # Actually, SAFETY metrics track interventions.
    # To check ACTUAL performance, we want goal reached, avg distance, and if safety was active.
    
    acc_stats = {
        "reward": 0.0,
        "goal_reached": 0,
        "speed_reductions": 0,
        "hard_stops": 0,
        "repulsion_acts": 0
    }
    
    start_t = time.time()
    
    for t in range(steps):
        rng, step_rng = jax.random.split(rng)
        
        # Heuristic Action
        actions = get_heuristic_action(state, env.action_dim, mode=mode_key)
        
        state, obs, rewards, dones, info = env.step(state, actions, step_rng)
        
        # Accumulate
        acc_stats["reward"] += float(jnp.mean(rewards))
        acc_stats["goal_reached"] += int(jnp.sum(info["goal_reached"]))
        
        if "speed_reductions" in info:
            acc_stats["speed_reductions"] += int(jnp.sum(info["speed_reductions"]))
        if "hard_stops" in info:
            acc_stats["hard_stops"] += int(jnp.sum(info["hard_stops"]))
            
        # Repulsion acts typically not in info unless we added it to metrics
        # (It is in SafetyMetrics but maybe I didn't verify if it's summed)
        # speed_reductions is good proxy.
        
        # Handle done? 
        # Env wrapper auto-resets? No, wrapper is pure function usually or handles reset internally?
        # EnvWrapper step does NOT auto-reset in V3 pure-jax style usually, unless wrapped in AutoReset.
        # But here we just run physics. If goal reached, agents stay there or reset logic?
        # Wrapper step doesn't seem to have auto-reset logic in the code I saw.
        # But state.goal_reached is updated.
        
    duration = time.time() - start_t
    print(f"⏱️ Time: {duration:.2f}s | FPS: {steps/duration:.0f}")
    
    return acc_stats

def main():
    print("📊 SAFETY LAYER BENCHMARK 📊")
    print("==============================")
    
    results = {}
    
    modes = [
        "Direct (Unsafe)", 
        "Direct + Safety", 
        "Hybrid (Intent + Safety)"
    ]
    
    for m in modes:
        stats = run_benchmark(m, steps=200) # Short run
        results[m] = stats
        print(f"   Results for {m}:")
        print(f"   - Total Reward: {stats['reward']:.2f}")
        print(f"   - Goal Reaches: {stats['goal_reached']}")
        print(f"   - Speed Reductions: {stats['speed_reductions']}")
        print(f"   - Hard Stops: {stats['hard_stops']}")
        
    print("\n🏆 CONCLUSION:")
    # Simple logic to print summary
    safe_acts = results["Direct + Safety"]["speed_reductions"]
    hybrid_acts = results["Hybrid (Intent + Safety)"]["speed_reductions"]
    
    print(f"Safety Interventions (Direct): {safe_acts}")
    print(f"Safety Interventions (Hybrid): {hybrid_acts}")
    
    if hybrid_acts < safe_acts:
        print("✅ Hybrid architecture required FEWER safety interventions (Intent handling is smoother)!")
    elif hybrid_acts > safe_acts:
        print("ℹ️ Hybrid architecture triggered MORE interventions (PID tuning needed?)")
    else:
        print("Managed similar safety profile.")

if __name__ == "__main__":
    main()
