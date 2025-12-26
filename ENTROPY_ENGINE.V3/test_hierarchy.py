
"""
Verification Script for Dynamic Hierarchy
Tests Squad Formation, Leader Election, and Hierarchical Routing.
"""
import jax
import jax.numpy as jnp
from entropy.config import ExperimentConfig, AgentConfig, CommConfig, SimConfig
from entropy.training.env_wrapper import EntropyGymWrapper

def test_hierarchy():
    print("👑 Testing Dynamic Hierarchy...")
    
    # 1. Config
    num_agents = 10
    squad_size = 5
    cfg = ExperimentConfig(
        sim=SimConfig(num_envs=1, max_steps=100),
        agent=AgentConfig(
            num_agents=num_agents,
            use_communication=True,
            comm=CommConfig(
                mode="spatial",
                msg_dim=8,
                pheromones_enabled=False,
                hierarchy_enabled=True,
                squad_size=squad_size,
                max_neighbors=4,
                leader_broadcast_only=True
            )
        )
    )
    
    # 2. Init wrapper
    env = EntropyGymWrapper(cfg)
    rng = jax.random.PRNGKey(0)
    
    # 3. Reset
    state, obs = env.reset(rng)
    
    # Expected Obs Dim:
    # Lidar(32) + Vel(2) + Goal(2) = 36
    # Inbox(4 * 12) = 48
    # Hierarchy = num_agents(10) + 1 + 2 = 13 (Assuming max_squads=N)
    # Total = 36 + 48 + 13 = 97
    expect_h_dim = num_agents + 1 + 2
    expected_dim = 36 + 48 + expect_h_dim
    
    print(f"Env Obs Dim: {env.obs_dim}")
    assert env.obs_dim == expected_dim, f"Dim mismatch! Got {env.obs_dim}, expected {expected_dim}"
    print(f"✅ Obs Dim Verified.")
    
    # 4. Step to trigger Election (every 50 steps, or init?)
    # Logic: `is_election_step = (state.timestep % re_elect_interval) == 0`
    # At T=0 this is True. So next state T=1 will have hierarchy.
    
    print("STEP 1: Triggering Election...")
    actions = jnp.zeros((num_agents, env.action_dim))
    state, obs, _, _, _ = env.step(state, actions, rng)
    
    # 5. Verify Squads
    squad_ids = state.agent_squad_ids
    print(f"Squad IDs: {squad_ids}")
    
    unique_squads = jnp.unique(squad_ids)
    print(f"Unique Squads: {unique_squads}")
    assert len(unique_squads) >= 2, "Expected at least 2 squads"
    
    # 6. Verify Leaders
    leaders = state.agent_is_leader
    print(f"Leaders: {leaders}")
    
    num_leaders = jnp.sum(leaders)
    print(f"Num Leaders: {num_leaders}")
    # One per squad
    assert num_leaders == len(unique_squads), f"Expected {len(unique_squads)} leaders, got {num_leaders}"
    
    print("✅ Squads and Leaders Verified.")
    
    # 7. Routing Isolation Test
    non_leader_idx = jnp.where(~leaders)[0][0]
    
    print(f"Testing Non-Leader Broadcast (Agent {non_leader_idx})...")
    
    # Agent tries to Broadcast (Channel=3.0)
    actions = jnp.zeros((num_agents, env.action_dim))
    
    # Non-leader broadcasting
    actions = actions.at[non_leader_idx, 3].set(3.0) 
    actions = actions.at[non_leader_idx, 2].set(10.0) # Gate Open
    actions = actions.at[non_leader_idx, 6:10].set(1.0) # Msg
    
    # Step
    state, obs, _, _, _ = env.step(state, actions, rng)
    
    # Check Inbox of someone in DIFFERENT squad
    my_squad = state.agent_squad_ids[non_leader_idx]
    other_squad_agents = jnp.where(state.agent_squad_ids != my_squad)[0]
    
    if len(other_squad_agents) > 0:
        other_agent = other_squad_agents[0]
        # Inbox of other agent
        # Inbox starts at 36 + H_dim.
        inbox_start = 36 + expect_h_dim
        inbox_vals = obs[other_agent, inbox_start:]
        
        # Check masks (every 12th element, at offset 11)
        # Layout: [Msg(8), Meta(3), Mask(1)] * K
        # Reshape to [K, 12]
        inbox_reshaped = inbox_vals.reshape(4, 12)
        masks = inbox_reshaped[:, -1]
        print(f"Masks: {masks}")
        
        # Expect all masks to be 0 for cross-squad
        assert jnp.all(masks == 0), "Non-Leader leaked message to other squad!"
        print("✅ Isolation Verified.")
    else:
        print("⚠️ Could not find agent in other squad (only 1 squad formed?)")

    print("✅ Dynamic Hierarchy Verification Complete!")

if __name__ == "__main__":
    test_hierarchy()
