
"""
Verification Script for Dual Channel Attention
"""
import jax
import jax.numpy as jnp
from entropy.config import ExperimentConfig, AgentConfig, CommConfig, PPOConfig, SimConfig, RenderConfig
from entropy.training.mappo import OptimizedMAPPO

def test_dual_channel():
    print("🧠 Testing Dual Channel Attention...")
    
    # 1. Config
    cfg = ExperimentConfig(
        sim=SimConfig(num_envs=1, max_steps=10),
        agent=AgentConfig(
            num_agents=2,
            use_communication=True,
            comm=CommConfig(
                mode="spatial",
                msg_dim=8,
                max_neighbors=4,
                dual_attention=True,
                local_radius=100.0,
                local_heads=2,
                global_heads=2
            )
        ),
        render=RenderConfig(enabled=False)
    )
    
    # 2. Mock Data
    # Calculate Obs Dim
    # Lidar(32) + Vel(2) + Goal(2) = 36
    # Inbox: K * (Msg(8) + Meta(3) + Mask(1)) = 4 * 12 = 48
    # Total = 84
    obs_dim = 36 + 48
    action_dim = 2 + 1 + 1 + 1 + 1 + 8 # 14
    
    print(f"expected obs_dim: {obs_dim}")
    
    # 3. Init Trainer
    rng = jax.random.PRNGKey(0)
    trainer = OptimizedMAPPO(cfg)
    trainer.init_states(obs_dim, action_dim, cfg.agent.num_agents, rng)
    
    # 4. Forward Pass Test
    obs = jnp.ones((1, obs_dim)) # Batch 1
    carry = jnp.zeros((1, 256))
    
    print("Input shapes:", obs.shape, carry.shape)
    
    # Apply
    # We expect separate heads to process
    # This just ensures shapes match and no JAX compilation errors
    new_carry, mean, log_std = trainer.actor_state.apply_fn(trainer.actor_state.params, obs, carry)
    
    print("Output shapes:", mean.shape, log_std.shape)
    print("✅ Dual Channel Actor verified!")

if __name__ == "__main__":
    test_dual_channel()
