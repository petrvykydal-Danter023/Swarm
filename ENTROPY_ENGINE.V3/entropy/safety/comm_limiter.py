import jax
import jax.numpy as jnp
from flax import struct
from typing import Tuple
from entropy.core.world import WorldState
from entropy.config import SafetyConfig

@struct.dataclass
class TokenBucketState:
    """Tracks message credits per agent with staggered refill."""
    tokens: jnp.ndarray          # [N] current tokens
    last_refill: jnp.ndarray     # [N] per-agent last refill step (STAGGERED!)

def create_token_bucket(num_agents: int, window: int, rng: jax.Array) -> TokenBucketState:
    """
    Inicializuje Token Bucket se STAGGERED refill offsets.
    Zabraňuje burst komunikaci když všichni refillují naráz.
    """
    # Náhodný offset pro každého agenta: 0 až window-1
    offsets = jax.random.randint(rng, (num_agents,), 0, window)
    
    return TokenBucketState(
        tokens=jnp.full(num_agents, 5, dtype=jnp.float32),  # Start with full tokens
        last_refill=-offsets.astype(jnp.float32)  # Staggered init
    )

def apply_comm_limit(
    state: WorldState,
    actions: jnp.ndarray,
    bucket_state: TokenBucketState,
    config: SafetyConfig
) -> Tuple[jnp.ndarray, TokenBucketState]:
    """
    Agent can only send message if they have tokens.
    Uses per-agent staggered refill to prevent burst.
    """
    N = actions.shape[0]
    
    # Per-agent refill check (staggered!)
    steps_since_refill = state.timestep - bucket_state.last_refill
    should_refill = steps_since_refill >= config.msg_rate_window
    
    # Refill only agents whose window expired
    new_tokens = jnp.where(
        should_refill,
        jnp.full(N, config.msg_rate_limit, dtype=jnp.float32),
        bucket_state.tokens
    )
    new_last_refill = jnp.where(
        should_refill,
        jnp.full(N, state.timestep, dtype=jnp.float32),
        bucket_state.last_refill
    )
    
    # Check if speaking (gate > threshold)
    # Assuming gate is at index 2 (Motor_L, Motor_R, Gate, ...)
    gate_logits = actions[:, 2]
    wants_to_speak = jax.nn.sigmoid(gate_logits) > 0.5
    
    # Can speak only if tokens > 0
    can_speak = new_tokens > 0
    allowed_to_speak = wants_to_speak & can_speak
    
    # Deduct token if speaking
    new_tokens = jnp.where(allowed_to_speak, new_tokens - 1, new_tokens)
    
    # Force gate closed if no tokens
    forced_silent = wants_to_speak & ~can_speak
    safe_actions = actions.at[:, 2].set(
        jnp.where(forced_silent, -10.0, actions[:, 2])  # Force gate low
    )
    
    new_bucket = TokenBucketState(tokens=new_tokens, last_refill=new_last_refill)
    return safe_actions, new_bucket
