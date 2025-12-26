"""
Unit tests for Communication system.
"""
import pytest
import jax
import jax.numpy as jnp
from entropy.brain.communication import CommunicationEncoder, TransformerContextDecoder

class TestCommunicationEncoder:
    def test_encoder_output_shape(self):
        """Encoder outputs correct message shape."""
        encoder = CommunicationEncoder(
            obs_dim=100,
            vocab_size=32,
            payload_dim=4,
            hidden_dim=64
        )
        
        # Initialize
        rng = jax.random.PRNGKey(0)
        obs = jnp.zeros((1, 100))
        params = encoder.init(rng, obs)
        
        # Forward
        message = encoder.apply(params, obs)
        
        # Check shape: vocab_size + payload_dim = 36
        assert message.shape == (1, 36)
        
    def test_encoder_batch(self):
        """Encoder handles batched input."""
        encoder = CommunicationEncoder(
            obs_dim=100,
            vocab_size=32,
            payload_dim=4,
            hidden_dim=64
        )
        
        rng = jax.random.PRNGKey(0)
        obs = jnp.zeros((10, 100))  # Batch of 10
        params = encoder.init(rng, obs)
        
        messages = encoder.apply(params, obs)
        assert messages.shape == (10, 36)

class TestContextDecoder:
    def test_decoder_output_shape(self):
        """Decoder outputs context of correct shape."""
        decoder = TransformerContextDecoder(
            msg_dim=36,
            context_dim=64,
            num_heads=4,
            hidden_dim=128
        )
        
        rng = jax.random.PRNGKey(0)
        # 5 agents, each with 36-dim message
        messages = jnp.zeros((5, 36))
        params = decoder.init(rng, messages)
        
        contexts = decoder.apply(params, messages)
        
        # Each agent gets 64-dim context
        assert contexts.shape == (5, 64)
        
    def test_decoder_attention(self):
        """Decoder uses attention (different inputs -> different outputs)."""
        decoder = TransformerContextDecoder(
            msg_dim=36,
            context_dim=64,
            num_heads=4,
            hidden_dim=128
        )
        
        rng = jax.random.PRNGKey(0)
        
        # All zeros
        msg1 = jnp.zeros((3, 36))
        params = decoder.init(rng, msg1)
        ctx1 = decoder.apply(params, msg1)
        
        # One agent has different message
        msg2 = msg1.at[0, :].set(1.0)
        ctx2 = decoder.apply(params, msg2)
        
        # Contexts should differ
        assert not jnp.allclose(ctx1, ctx2)
