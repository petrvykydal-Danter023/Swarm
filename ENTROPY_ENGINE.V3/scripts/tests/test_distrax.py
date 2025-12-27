import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))
import jax
import jax.numpy as jnp
import distrax
import traceback

def test_distrax():
    print("Testing Distrax...")
    try:
        key = jax.random.PRNGKey(42)
        mean = jnp.zeros((10, 2))
        scale = jnp.ones((10, 2))
        
        pi = distrax.MultivariateNormalDiag(mean, scale)
        print("Distribution created.")
        
        sample = pi.sample(seed=key)
        print(f"Sample shape: {sample.shape}")
        
        log_prob = pi.log_prob(sample)
        print(f"LogProb shape: {log_prob.shape}")
        
        entropy = pi.entropy()
        print(f"Entropy shape: {entropy.shape}")
        
        print("✅ Distrax works!")
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    test_distrax()
