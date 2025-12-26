"""
Training utilities: Prioritized Sampling, Coverage-aware BC Loss
"""
import numpy as np
from typing import Any, List

class PrioritizedDemoSampler:
    """
    Prioritized sampling based on prediction error.
    Samples with higher error are sampled more frequently.
    """
    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha  # Priority exponent

    def sample(self, demos: List[Any], model: Any, batch_size: int = 256) -> List[Any]:
        """
        Sample demos prioritized by prediction error.
        
        Args:
            demos: List of demonstration tuples (obs, action)
            model: Current policy model
            batch_size: Number of samples to return
        """
        if len(demos) == 0:
            return []
            
        # Mock error computation
        # In real impl: predictions = model(demos); errors = |predictions - targets|
        errors = np.random.random(len(demos))
        
        # Compute priorities
        priorities = errors ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample according to priorities
        indices = np.random.choice(len(demos), size=min(batch_size, len(demos)), p=probs, replace=False)
        return [demos[i] for i in indices]


def coverage_aware_bc_loss(policy: Any, demos: Any, entropy_coef: float = 0.01, label_smoothing: float = 0.1):
    """
    Behavioral Cloning loss with entropy regularization for broader coverage.
    
    This helps RL fine-tuning by keeping policy distribution wide enough.
    
    Args:
        policy: Policy network
        demos: Demonstration batch (observations, expert_actions)
        entropy_coef: Weight for entropy bonus
        label_smoothing: Amount of noise to add to targets
    
    Returns:
        Combined loss value (mock)
    """
    # Mock implementation
    # Real impl would:
    # 1. pred_actions = policy(demos.observations)
    # 2. bc_loss = mse(pred_actions, demos.expert_actions)
    # 3. entropy = -sum(policy.log_prob(pred_actions))
    # 4. smoothed = 0.9*targets + 0.1*noise
    # 5. smooth_loss = mse(pred_actions, smoothed)
    # 6. return bc_loss + 0.5*smooth_loss - entropy_coef*entropy
    
    return 0.1  # Mock loss value
