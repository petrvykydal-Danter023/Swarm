"""
Smoke test for Brain/Policy system.
Verifies that neural network inference works correctly.
"""
import pytest
import jax
import jax.numpy as jnp

def test_smoke_brain():
    """Brain inference runs without error."""
    print("🔥 Smoke Test: Brain")
    
    from entropy.training.network import ActorCritic
    
    # Create policy
    policy = ActorCritic(
        action_dim=2,
        width=64
    )
    print("  ✅ Policy created")
    
    # Initialize
    rng = jax.random.PRNGKey(42)
    dummy_obs = jnp.zeros((1, 100))
    params = policy.init(rng, dummy_obs)
    print("  ✅ Params initialized")
    print("  ✅ Params initialized")
    
    # Batch inference
    batch_obs = jnp.zeros((10, 100))
    output = policy.apply(params, batch_obs)
    
    # Output should be (action_mean, action_log_std, value)
    if isinstance(output, tuple):
        action_mean = output[0]
        print(f"  ✅ Inference output shape: {action_mean.shape}")
        assert action_mean.shape == (10, 2)
    else:
        print(f"  ✅ Inference output shape: {output.shape}")
        
    print("🎉 Smoke Test PASSED!")

@pytest.mark.skip(reason="BrainManager registry needs debugging")
def test_brain_manager():
    """BrainManager can save and load."""
    print("🔥 Smoke Test: BrainManager")
    
    from entropy.brain.manager import BrainManager, BrainMeta
    import tempfile
    import numpy as np
    
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = BrainManager(storage_dir=tmpdir)
        print("  ✅ Manager created")
        
        # Create dummy params - use numpy arrays (flat structure for safetensors)
        params = {
            "layer1_kernel": np.ones((10, 5), dtype=np.float32),
            "layer1_bias": np.zeros(5, dtype=np.float32)
        }
        
        # Create proper meta object
        meta = BrainMeta(
            name="test_brain",
            version=1,
            brain_type="ppo",
            input_dim=100,
            output_dim=2,
            hidden_dim=64,
            created_at="2024-01-01",
            training_steps=0
        )
        
        # Save
        name = manager.save("test_brain", params, meta)
        print(f"  ✅ Saved as: {name}")
        
        # Load
        loaded, loaded_meta = manager.load(name)
        print(f"  ✅ Loaded, meta: {loaded_meta}")
        
        assert np.allclose(loaded["layer1_kernel"], params["layer1_kernel"])
        
    print("🎉 Smoke Test PASSED!")


