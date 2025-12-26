"""
Full pipeline smoke test.
Verifies: env reset -> policy inference -> env step -> reward.
"""
import pytest
import jax
import jax.numpy as jnp

def test_smoke_full_pipeline():
    """Full training pipeline runs without error."""
    print("🔥 Smoke Test: Full Pipeline")
    
    from entropy.core.world import create_initial_state
    from entropy.core.physics import physics_step
    from entropy.training.network import ActorCritic
    
    # 1. Create world
    state = create_initial_state(num_agents=5)
    print(f"  ✅ World created: {state.num_agents} agents")
    
    # 2. Create policy
    obs_dim = 100
    action_dim = 2
    policy = ActorCritic(action_dim=action_dim, width=64)
    
    rng = jax.random.PRNGKey(42)
    dummy_obs = jnp.zeros((1, obs_dim))
    params = policy.init(rng, dummy_obs)
    print("  ✅ Policy created and initialized")

    
    # 3. Run rollout
    total_steps = 100
    for step in range(total_steps):
        # Get observations (simplified: zeros)
        obs = jnp.zeros((state.num_agents, obs_dim))
        
        # Policy inference
        output = policy.apply(params, obs)
        if isinstance(output, tuple):
            actions = output[0]  # action_mean
        else:
            actions = output[:, :action_dim]
        
        # Clip actions
        actions = jnp.clip(actions, -1.0, 1.0)
        
        # Physics step
        state = physics_step(state, actions)
        
    print(f"  ✅ {total_steps} steps completed")
    
    # 4. Check no NaNs
    assert not jnp.any(jnp.isnan(state.agent_positions)), "NaN in positions!"
    print("  ✅ No NaN values")
    
    print("🎉 Full Pipeline Smoke Test PASSED!")

def test_hand_of_god_integration():
    """Hand of God expert + mixer works."""
    print("🔥 Smoke Test: Hand of God")
    
    from entropy.core.world import create_initial_state
    from entropy.training.hand_of_god.expert import (
        DirectNavigator, 
        ActionMixer, 
        AlphaScheduler
    )
    
    state = create_initial_state(num_agents=3)
    # Set some goal positions
    state = state.replace(goal_positions=jnp.array([[100, 100], [200, 200], [300, 300]]))
    
    expert = DirectNavigator()
    mixer = ActionMixer()
    scheduler = AlphaScheduler(initial=1.0, final=0.0, total_steps=1000)
    
    rng = jax.random.PRNGKey(0)
    
    # Get expert actions
    expert_actions = expert.act(state, rng)
    print(f"  ✅ Expert actions shape: {expert_actions.shape}")
    
    # Simulate AI actions
    ai_actions = jnp.zeros((3, 2))
    
    # Mix
    alpha = scheduler.get_alpha(0)
    mixed = mixer.mix(ai_actions, expert_actions, alpha)
    print(f"  ✅ Mixed actions (alpha={alpha}): {mixed.shape}")
    
    # At step 0, alpha=1.0, so mixed should equal expert
    assert jnp.allclose(mixed, expert_actions)
    
    # At step 1000, alpha=0.0, so mixed should equal AI
    alpha_end = scheduler.get_alpha(1000)
    mixed_end = mixer.mix(ai_actions, expert_actions, alpha_end)
    assert jnp.allclose(mixed_end, ai_actions)
    
    print("🎉 Hand of God Smoke Test PASSED!")
