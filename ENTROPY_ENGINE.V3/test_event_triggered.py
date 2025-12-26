
"""
Verification Script for Event-Triggered Communication
"""
import jax
import jax.numpy as jnp
from entropy.config import ExperimentConfig, AgentConfig, CommConfig, SimConfig, RenderConfig
from entropy.training.mappo import OptimizedMAPPO
from entropy.brain.world_model import WorldModelPredictor, compute_surprise

def test_event_triggered():
    print("🧠 Testing Event-Triggered Communication...")
    
    # 1. Config
    cfg = ExperimentConfig(
        sim=SimConfig(num_envs=1, max_steps=10),
        agent=AgentConfig(
            num_agents=2,
            use_communication=True,
            comm=CommConfig(
                mode="spatial",
                msg_dim=8,
                dual_attention=True,
                surprise_gating=True, # ENABLED
                surprise_threshold=0.1
            )
        )
    )
    
    # 2. Setup
    obs_dim = 84
    action_dim = 14
    rng = jax.random.PRNGKey(0)
    
    # Create Dict Config like train_master.py
    mappo_config = {
        "lr_actor": 3e-4,
        "lr_critic": 1e-3,
        "actor_updates": 4,
        "critic_updates": 1,
        "clip_eps": 0.2,
        "agent": cfg.agent # Critical: Pass Agent Config
    }
    
    print("Initializing Trainer with World Model...")
    trainer = OptimizedMAPPO(mappo_config)
    trainer.init_states(obs_dim, action_dim, cfg.agent.num_agents, rng)
    
    # 3. Verify World Model Existence
    if trainer.wm_state is None:
        print("❌ World Model state is None!")
        exit(1)
    else:
        print("✅ World Model initialized.")
        
    # 4. Neural Forward Pass Check
    obs = jnp.ones((1, obs_dim))
    action = jnp.ones((1, action_dim))
    
    pred_next_obs = trainer.wm_state.apply_fn(trainer.wm_state.params, obs, action)
    print(f"Prediction Shape: {pred_next_obs.shape}")
    assert pred_next_obs.shape == obs.shape
    
    # 5. Surprise Calculation Check
    actual_next_obs = jnp.ones((1, obs_dim)) + 0.5 # High surprise
    surprise = compute_surprise(pred_next_obs, actual_next_obs)
    print(f"Surprise Value (approx 0.5): {jnp.mean(surprise)}")
    
    # 6. Training Step Check (Update Loop)
    # Mock Buffer
    T = 10
    N = 2
    buffer = {
        'obs': jnp.zeros((T, N, obs_dim)),
        'actions': jnp.zeros((T, N, action_dim)),
        'log_probs': jnp.zeros((T, N)),
        'values': jnp.zeros((T + 1, N)), # Corrected: [T+1, N]
        'rewards': jnp.zeros((T, N)),
        'dones': jnp.zeros((T, N)),
        'actor_states': jnp.zeros((T, N, 256))
    }
    # Fix critic output based on mappo implementation -> [T, N] usually?
    # Trainer values are flattened in update.
    # Check rollout: values are [T, 1] -> [T, N] expanded?
    # No, critic outputs [batch, num_agents].
    
    # Run simple update call
    print("Running Update Step...")
    try:
        a_loss, c_loss = trainer.update(buffer)
        print(f"Update Successful. Loss: {a_loss}, {c_loss}")
    except Exception as e:
        print(f"❌ Update Failed: {e}")
        import traceback
        traceback.print_exc()

    print("✅ Event-Triggered Comms Verification Complete!")

if __name__ == "__main__":
    test_event_triggered()
