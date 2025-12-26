
"""
Verification Script for Virtual Pheromones
Tests placement, decay, and observation signal.
"""
import jax
import jax.numpy as jnp
from entropy.config import ExperimentConfig, AgentConfig, CommConfig, SimConfig
from entropy.training.env_wrapper import EntropyGymWrapper

def test_pheromones():
    print("🐜 Testing Virtual Pheromones...")
    
    # 1. Config with Pheromones
    cfg = ExperimentConfig(
        sim=SimConfig(num_envs=1, max_steps=100),
        agent=AgentConfig(
            num_agents=2,
            use_communication=True,
            comm=CommConfig(
                mode="spatial",
                msg_dim=8,
                pheromones_enabled=True,
                pheromone_dim=4,
                pheromone_ttl=10,
                max_pheromones=100,
                max_neighbors=1, # Avoid top_k error needs K <= N-1 (N=2 -> K=1)
                hierarchy_enabled=False
            )
        )
    )
    
    # 2. Init wrapper
    env = EntropyGymWrapper(cfg)
    rng = jax.random.PRNGKey(0)
    
    # 3. Reset
    state, obs = env.reset(rng)
    
    # Obs Dim check
    # Lidar(32) + Vel(2) + Goal(2) = 36
    # Inbox(K=1) -> Msg(8) + Meta(3) + Mask(1) = 12
    # Pheromone(4)
    # Total = 36 + 12 + 4 = 52
    
    print(f"Obs Dim: {env.obs_dim}")
    print(f"Pheromone Dim: {cfg.agent.comm.pheromone_dim}")
    expected_dim = 32 + 4 + 12 + 4
    if hasattr(cfg.agent.comm, 'hierarchy_enabled') and cfg.agent.comm.hierarchy_enabled:
         expected_dim += 3 # Approx
         
    # Only asserting if close, since previous update might have added hierarchy check in wrapper implicitly?
    # No, hierarchy disabled in this config.
    assert env.obs_dim == 52, f"Expected 52, got {env.obs_dim}"
    print("✅ Obs Dim Verified: 52")
    
    # 4. Place Pheromone
    print("STEP 1: Placing Pheromone...")
    actions = jnp.zeros((2, env.action_dim))
    
    # Agent 0 places
    # Action[3] > 2.0 (Channel)
    actions = actions.at[0, 3].set(3.0) 
    actions = actions.at[0, 2].set(10.0) # Gate Open 
    
    # Target: Relative (0,0) -> Current Pos
    actions = actions.at[0, 5].set(0.0) # Dist 0
    
    # Message: [1, 1, 1, 1]
    actions = actions.at[0, 6:10].set(1.0)
    
    # Step
    next_state, obs, _, _, _ = env.step(state, actions, rng)
    
    # Verify State Update
    ptr = next_state.pheromone_write_ptr
    # Ptr should be 1? Or 1+? Init was 0.
    # We used Fori loop, agent 0 placed.
    
    # Check valid mask
    valid = next_state.pheromone_valid
    assert valid[0] == True, "Pheromone 0 not valid"
    
    # Check message
    p_msg = next_state.pheromone_messages[0]
    # Expect [1, 1, 1, 1]
    # Note: If comm_config.msg_dim=8 but p_dim=4, we take first 4.
    # We set actions[6:10] to 1.0. This corresponds to first 4 msg dims.
    assert jnp.allclose(p_msg[:4], 1.0), f"Expected [1,1,1,1], got {p_msg}"
    print("✅ Pheromone Placed in State.")
    
    # 5. Verify Obs (Sensing)
    print("STEP 2: Reading Pheromone signal...")
    # Agent 0 is at same position as pheromone. Dist ~ 0.
    # Obs layout: [Lidar(32), Vel(2), Goal(2), Phero(4), Inbox(12)]
    # Indices: 0-31 Lidar, 32-33 Vel, 34-35 Goal
    # 36-39 Phero
    
    phero_obs = obs[0, 36:40]
    print(f"Agent 0 Phero Obs: {phero_obs}")
    
    # Weight should be roughly 1.0 (some movement might occur)
    # Signal should be [1, 1, 1, 1]
    print(f"Observed Phero Signal: {phero_obs}")
    assert jnp.all(phero_obs > 0.9), f"Expected > 0.9, got {phero_obs}"
    print("✅ Pheromone Observed.")
    
    # 6. Verify Decay
    print("Stepping until Decay...")
    curr_state = next_state
    
    # TTL is 10. Already stepped once (Placement). Remaining ~9.
    # Step 10 times.
    for i in range(12):
        curr_state, _, _, _, _ = env.step(curr_state, jnp.zeros_like(actions), rng)
        ttl = curr_state.pheromone_ttls[0]
        # print(f"Step {i+2}, TTL[0]: {ttl}, Valid: {curr_state.pheromone_valid[0]}")
        
    assert curr_state.pheromone_valid[0] == False, "Pheromone should be expired"
    print("✅ Pheromone Expired correctly.")
    
    print("✅ Virtual Pheromones Verification Complete!")

if __name__ == "__main__":
    test_pheromones()
