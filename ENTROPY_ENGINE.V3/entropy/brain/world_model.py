
"""
Entropy Engine V3 - World Model Predictor
Crucial for Event-Triggered Communication:
"Surprise" = |Predicted_Obs - Actual_Obs|
"""
import jax.numpy as jnp
from flax import linen as nn

class WorldModelPredictor(nn.Module):
    """
    Predicts the next observation vector based on current state and action.
    Used to compute 'Surprise'.
    """
    hidden_dim: int = 128
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray, action: jnp.ndarray):
        """
        Args:
            obs: [..., ObsDim]
            action: [..., ActionDim]
            
        Returns:
            predicted_next_obs: [..., ObsDim]
        """
        # Concatenate inputs
        x = jnp.concatenate([obs, action], axis=-1)
        
        # Simple MLP Predictor
        x = nn.Dense(self.hidden_dim, name='fc1')(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, name='fc2')(x)
        x = nn.relu(x)
        
        # Output head matching obs dimension
        predicted_next_obs = nn.Dense(obs.shape[-1], name='out')(x)
        
        return predicted_next_obs
    
def compute_surprise(predicted_obs: jnp.ndarray, actual_obs: jnp.ndarray) -> jnp.ndarray:
    """
    Computes surprise as Mean Absolute Error per agent.
    Returns: [..., 1]
    """
    error = jnp.abs(predicted_obs - actual_obs)
    # Average across feature dimension
    surprise = jnp.mean(error, axis=-1, keepdims=True)
    return surprise
