"""
Entropy Engine V3 - Communication System
Implements: 03_COMMUNICATION.txt blueprint
"""
import jax
import jax.numpy as jnp
from flax import linen as nn
from flax import struct
from enum import IntEnum
from typing import Optional
import optax

# ================================================================================
# 1. VOCABULARY (Grounded Tokens)
# ================================================================================

class Token(IntEnum):
    """Vocabulary for agent communication. 32 tokens total."""
    SILENCE = 0        # Mlčení
    HELP = 1           # Žádost o pomoc
    DANGER = 2         # Varování
    CARRYING = 3       # Nesu objekt
    NEED_PICKUP = 4    # Potřebuji odvoz
    FOUND_RESOURCE = 5
    FOUND_ENEMY = 6
    FOUND_TARGET = 7   # Cíl
    GOING_TO = 8       # Směřuji k
    FOLLOWING = 9      # Následuji tě
    WAITING = 10       # Čekám
    ACKNOWLEDGE = 11   # Rozumím
    NEGATIVE = 12      # Ne/odmítám
    # 13-31 are EMERGENT (learned meanings)

VOCAB_SIZE = 32
PAYLOAD_DIM = 4
MSG_DIM = VOCAB_SIZE + PAYLOAD_DIM  # 36

# ================================================================================
# 2. MESSAGE DATACLASS
# ================================================================================

@struct.dataclass
class Message:
    """Structured message for communication."""
    sender_id: int
    token: int               # 0-31 (Token enum)
    payload: jnp.ndarray     # [4] - (rel_x, rel_y, urgency, target_id)
    channel: str = "broadcast"  # "broadcast", "targeted", "team"
    target_id: int = -1      # -1 for broadcast
    priority: float = 1.0    # For queue arbitration
    timestamp: int = 0

# ================================================================================
# 3. MESSAGE BUFFER (Temporal Consistency)
# ================================================================================

@struct.dataclass
class MessageBuffer:
    """Ring buffer for message history. JAX-compatible."""
    buffer: jnp.ndarray      # [MAX_HISTORY, N, MSG_DIM]
    timestamps: jnp.ndarray  # [MAX_HISTORY]
    head: int = 0
    
    @classmethod
    def create(cls, max_history: int, num_agents: int, msg_dim: int = MSG_DIM):
        return cls(
            buffer=jnp.zeros((max_history, num_agents, msg_dim)),
            timestamps=jnp.zeros(max_history, dtype=jnp.int32),
            head=0
        )
    
    def push(self, messages: jnp.ndarray, timestep: int) -> "MessageBuffer":
        """Add new messages to buffer. Returns new buffer (immutable)."""
        new_buffer = self.buffer.at[self.head].set(messages)
        new_timestamps = self.timestamps.at[self.head].set(timestep)
        new_head = (self.head + 1) % self.buffer.shape[0]
        return self.replace(buffer=new_buffer, timestamps=new_timestamps, head=new_head)
    
    def get_recent(self, n: int) -> jnp.ndarray:
        """Return last n messages for all agents. Shape: [n, N, MSG_DIM]"""
        max_history = self.buffer.shape[0]
        indices = jnp.arange(self.head - n, self.head) % max_history
        return self.buffer[indices]

# ================================================================================
# 4. TRANSFORMER CONTEXT DECODER
# ================================================================================

class TransformerContextDecoder(nn.Module):
    """
    Processes messages from all agents using Self-Attention.
    Output: Context vector for each agent summarizing communication.
    """
    msg_dim: int = MSG_DIM
    context_dim: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout_rate: float = 0.0
    pool_output: bool = False  # If True, mean-pool to [B, ctx_dim]
    
    @nn.compact
    def __call__(
        self, 
        messages: jnp.ndarray, 
        positions: Optional[jnp.ndarray] = None,
        mask: Optional[jnp.ndarray] = None, 
        training: bool = False
    ):
        """
        Args:
            messages: [B, N, msg_dim] - messages from all agents
            positions: [B, N, 2] - relative positions for positional encoding (optional)
            mask: [B, 1, N, N] - attention mask (True=see, False=hide)
            training: bool - for dropout
            
        Returns:
            context: [B, N, context_dim] or [B, context_dim] if pool_output=True
        """
        # 1. Message Embedding
        x = nn.Dense(self.context_dim, name='msg_embed')(messages)
        x = nn.relu(x)
        
        # 2. Optional Positional Encoding
        if positions is not None:
            pos_embed = self._position_encoding(positions)
            x = x + pos_embed
        
        # 3. Transformer Encoder Layers
        for i in range(self.num_layers):
            # Self-attention
            attn_out = nn.MultiHeadDotProductAttention(
                num_heads=self.num_heads,
                qkv_features=self.context_dim,
                dropout_rate=self.dropout_rate,
                name=f'attention_{i}'
            )(inputs_q=x, inputs_kv=x, mask=mask, deterministic=not training)
            
            # Residual + LayerNorm
            x = nn.LayerNorm(name=f'ln1_{i}')(x + attn_out)
            
            # Feed Forward
            ff = nn.Dense(self.context_dim * 2, name=f'ff1_{i}')(x)
            ff = nn.relu(ff)
            ff = nn.Dense(self.context_dim, name=f'ff2_{i}')(ff)
            
            # Residual + LayerNorm
            x = nn.LayerNorm(name=f'ln2_{i}')(x + ff)
        
        # 4. Output
        if self.pool_output:
            # Mean pool across agents to single context vector
            context = jnp.mean(x, axis=1)  # [B, context_dim]
        else:
            # Per-agent context
            context = x  # [B, N, context_dim]
        
        return context
    
    def _position_encoding(self, positions: jnp.ndarray) -> jnp.ndarray:
        """Simple positional encoding from 2D relative positions."""
        return nn.Dense(self.context_dim, name='pos_enc')(positions)

# ================================================================================
# 5. COMMUNICATION ENCODER
# ================================================================================

class CommunicationEncoder(nn.Module):
    """
    Generates messages from observations.
    Output: Token logits + continuous payload.
    """
    msg_dim: int = MSG_DIM
    vocab_size: int = VOCAB_SIZE
    payload_dim: int = PAYLOAD_DIM
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray, deterministic: bool = True):
        """
        Args:
            obs: [B, N, obs_dim]
            deterministic: If False, sample token via Gumbel-Softmax
            
        Returns:
            message: [B, N, msg_dim] (token logits + payload)
        """
        x = nn.Dense(64)(obs)
        x = nn.relu(x)
        x = nn.Dense(64)(x)
        x = nn.relu(x)
        
        # Token logits
        token_logits = nn.Dense(self.vocab_size, name='token_head')(x)  # [B, N, 32]
        
        # Payload
        payload = nn.Dense(self.payload_dim, name='payload_head')(x)  # [B, N, 4]
        payload = nn.tanh(payload)  # Normalize to [-1, 1]
        
        # Combine
        message = jnp.concatenate([token_logits, payload], axis=-1)  # [B, N, 36]
        
        return message

# ================================================================================
# 6. TRAINING HELPERS
# ================================================================================

def compute_bandwidth_penalty(tokens: jnp.ndarray, silence_token: int = Token.SILENCE, penalty: float = -0.01) -> jnp.ndarray:
    """
    Penalize unnecessary speaking.
    
    Args:
        tokens: [N] or [B, N] - token IDs
        silence_token: Token ID for silence
        penalty: Negative reward per non-silence message
        
    Returns:
        penalties: [N] or [B, N]
    """
    is_speaking = tokens != silence_token
    return is_speaking.astype(jnp.float32) * penalty

def contrastive_grounding_loss(messages: jnp.ndarray, states: jnp.ndarray, temperature: float = 0.1) -> jnp.ndarray:
    """
    InfoNCE-style contrastive loss for grounding.
    Similar states -> similar messages.
    
    Args:
        messages: [N, msg_dim] - encoded messages
        states: [N, state_dim] - agent states/observations
        temperature: Scaling factor
        
    Returns:
        loss: scalar
    """
    # Normalize
    msg_norm = messages / (jnp.linalg.norm(messages, axis=-1, keepdims=True) + 1e-8)
    state_norm = states / (jnp.linalg.norm(states, axis=-1, keepdims=True) + 1e-8)
    
    # Similarity matrix
    logits = jnp.matmul(msg_norm, state_norm.T) / temperature
    
    # Diagonal = positive pairs (same agent)
    labels = jnp.arange(len(messages))
    
    # Cross-entropy loss
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss

# ================================================================================
# 7. ATTENTION VISUALIZATION HELPER
# ================================================================================

def get_attention_weights(decoder: TransformerContextDecoder, params: dict, messages: jnp.ndarray):
    """
    Extract attention weights from decoder for visualization.
    
    Note: This requires modifying the decoder to return intermediates.
    For now, returns placeholder. Production: use nn.Module.capture_intermediates().
    
    Returns:
        attention_weights: [num_layers, num_heads, N, N] (placeholder)
    """
    # TODO: Implement using Flax capture_intermediates API
    # For now, return None as placeholder
    return None
