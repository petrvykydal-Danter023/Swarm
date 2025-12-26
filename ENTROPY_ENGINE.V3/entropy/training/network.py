"""
Entropy Engine V3 - Actor-Critic Network
Flax implementation of the PPO policy and value network.
"""
import jax.numpy as jnp
from flax import linen as nn
from typing import Sequence

class ActorCritic(nn.Module):
    """
    Combined Actor-Critic network.
    """
    action_dim: int
    activation: str = "tanh"
    width: int = 256
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: Observation [Batch, ObsDim]
            
        Returns:
            actor_mean: [Batch, ActionDim]
            actor_logstd: [ActionDim] parameter
            value: [Batch, 1]
        """
        # Activation function
        if self.activation == "relu":
            act = nn.relu
        else:
            act = nn.tanh
            
        # Shared Trunk
        # Simple MLP
        trunk = nn.Dense(self.width, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), bias_init=nn.initializers.constant(0.0))(x)
        trunk = act(trunk)
        trunk = nn.Dense(self.width, kernel_init=nn.initializers.orthogonal(np.sqrt(2)), bias_init=nn.initializers.constant(0.0))(trunk)
        trunk = act(trunk)
        
        # Critic Head (Value)
        value = nn.Dense(1, kernel_init=nn.initializers.orthogonal(1), bias_init=nn.initializers.constant(0.0))(trunk)
        
        # Actor Head (Policy Mean)
        actor_mean = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01), bias_init=nn.initializers.constant(0.0))(trunk)
        
        # Actor LogStd (Learnable parameter, separate from input)
        # We output a diagonal Gaussian policy
        actor_logstd = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        
        return actor_mean, actor_logstd, value.squeeze(-1)

import numpy as np # Import at top level actually required for default args in standard python, but here inside scope it'S okay-ish, keeping consistent.
