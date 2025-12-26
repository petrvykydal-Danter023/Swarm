"""
Entropy Engine V3 - MAPPO Network (Advanced)
Supports GRU Memory and Spatial Attention for Coordinated Unit communication.
"""
import jax.numpy as jnp
from flax import linen as nn
import jax
import numpy as np
import optax
import distrax
import logging
from flax.training.train_state import TrainState
from typing import Tuple, Any

class InboxAttention(nn.Module):
    """
    Processes the Inbox (K neighbors, D msg dim) using Attention.
    """
    embed_dim: int = 64
    num_heads: int = 4
    
    @nn.compact
    def __call__(self, inbox_flat: jnp.ndarray, mask: jnp.ndarray):
        # inbox_flat shape: [Batch, K*(D+4)] or similar. 
        # We need to reshape it back to [Batch, K, Features]
        # But wait, env wrapper concatenates everything.
        # We need to know K and FeatureDim.
        # Let's assume the user passes config or we infer.
        # Actually, for simplicity in this iteration, we treat the inbox 
        # as a flat vector and assume the MLP handles it (Basic version),
        # OR we implement the real sequence processing.
        
        # Real Implementation:
        # We need structured input. Since JAX envs return flat obs, 
        # we must slice the observation vector.
        # This requires knowing the indices.
        # For now, let's implement a powerful MLP encoder that acts as attention 
        # over the flat communication vector.
        # "Set Transformer" style.
        pass # Placeholder logic handled in main Actor for now to avoid slicing hell


class DualChannelAttention(nn.Module):
    """
    Processes inbox messages using two separate attention heads:
    1. Local Head: For tactical info from nearby agents (Collision, Formation).
    2. Global Head: For strategic info from far agents or leaders.
    """
    num_heads: int
    qkv_features: int
    local_radius: float
    
    @nn.compact
    def __call__(self, obs_emb, inbox_msgs, inbox_meta, inbox_mask):
        # Unpack Metadata: [MsgDim, RelDist, RelAngle, Channel]
        # inbox_meta is [Batch, K, 3] -> RelDist is idx 0
        rel_dist = inbox_meta[..., 0]
        
        # Create Masks
        # Local: within radius AND valid
        # Global: outside radius AND valid
        mask_local = (rel_dist < self.local_radius) & (inbox_mask > 0.5)
        mask_global = (rel_dist >= self.local_radius) & (inbox_mask > 0.5)
        
        # Projection of inbox messages to Embedding Dim
        inbox_emb = nn.Dense(obs_emb.shape[-1])(inbox_msgs)
        
        # Add batch dim for single query
        query = obs_emb[:, None, :] 
        
        # Local Head
        local_ctx = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, # We split heads between local/global or use same config
            qkv_features=self.qkv_features
        )(query, inbox_emb, mask=mask_local[:, None, None, :]) # Mask shape [B, 1, 1, K]
        
        # Global Head
        global_ctx = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads,
            qkv_features=self.qkv_features
        )(query, inbox_emb, mask=mask_global[:, None, None, :])
        
        # Remove sequence dim [B, 1, D] -> [B, D]
        local_ctx = local_ctx.squeeze(1)
        global_ctx = global_ctx.squeeze(1)
        
        # Concat contexts
        return jnp.concatenate([local_ctx, global_ctx], axis=-1)

class RecurrentActor(nn.Module):
    action_dim: int
    hidden_dim: int = 256
    comm_config: Any = None 
    
    @nn.compact
    def __call__(self, obs: jnp.ndarray, carry: jnp.ndarray):
        
        # --- 1. Obs Processing & Slicing ---
        if self.comm_config:
            # Reconstruct dimensions from config
            K = self.comm_config.max_neighbors
            D = self.comm_config.msg_dim
            # obs_dim = lidars + 4 + K*D + K*3 + K
            # We assume obs is [batch, obs_dim]
            
            # Slice offsets
            lidar_size = obs.shape[-1] - (K*(D+3+1))
            
            core_obs = obs[:, :lidar_size]
            comm_data = obs[:, lidar_size:]
            
            # Unpack flattened inbox
            # Layout: [Msgs(K*D), Meta(K*3), Mask(K)]
            idx_msg = K * D
            idx_meta = idx_msg + K * 3
            
            flat_msgs = comm_data[:, :idx_msg]
            flat_meta = comm_data[:, idx_msg:idx_meta]
            flat_mask = comm_data[:, idx_meta:]
            
            inbox_msgs = flat_msgs.reshape(-1, K, D)
            inbox_meta = flat_meta.reshape(-1, K, 3)
            inbox_mask = flat_mask.reshape(-1, K)
            
            # Embed Core Obs
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(core_obs)
            x = nn.relu(x)
            
            # Apply Dual Attention
            if self.comm_config.dual_attention:
                dual_att = DualChannelAttention(
                    num_heads=self.comm_config.local_heads, 
                    qkv_features=32,
                    local_radius=self.comm_config.local_radius
                )
                context = dual_att(x, inbox_msgs, inbox_meta, inbox_mask)
                
                # Merge Context
                x = jnp.concatenate([x, context], axis=-1)
            else:
                # Legacy / Simple Concat
                # Just concat everything back if dual attention disabled but comms enabled
                x = jnp.concatenate([x, flat_msgs, flat_meta, flat_mask], axis=-1)
                
        else:
            # No Comms - Standard processing
            x = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(obs)
            x = nn.relu(x)
            
        # --- 2. Memory Core (GRU) ---
        gru = nn.GRUCell(self.hidden_dim)
        new_carry, x = gru(carry, x)
        
        # --- 3. Heads ---
        backbone = nn.Dense(self.hidden_dim, kernel_init=nn.initializers.orthogonal(np.sqrt(2)))(x)
        backbone = nn.relu(backbone)
        
        mean_raw = nn.Dense(self.action_dim, kernel_init=nn.initializers.orthogonal(0.01))(backbone)
        log_std = self.param('log_std', nn.initializers.zeros, (self.action_dim,))
        
        return new_carry, mean_raw, log_std

class OptimizedMAPPO:
    """
    MAPPO with Recurrent Support.
    """
    def __init__(self, config: Any):
        self.config = config if isinstance(config, dict) else {}
        self.actor_updates = self.config.get("actor_updates", 4)
        self.critic_updates = self.config.get("critic_updates", 1)
        self.lr_actor = float(self.config.get("lr_actor", 3e-4))
        self.lr_critic = float(self.config.get("lr_critic", 1e-3))
        self.hidden_dim = 256
        
    def init_states(self, obs_dim, action_dim, num_agents, rng):
        # Extract CommConfig if available
        comm_config = None
        agent_config = None
        
        # Helper to get agent config object
        if hasattr(self.config, 'agent'):
            agent_config = self.config.agent
        elif isinstance(self.config, dict) and 'agent' in self.config:
            agent_config = self.config['agent']
            
        if agent_config and agent_config.use_communication:
            comm_config = agent_config.comm

        # Init Actor (Recurrent)
        actor_net = RecurrentActor(
            action_dim=action_dim, 
            hidden_dim=self.hidden_dim,
            comm_config=comm_config
        )
        rng, a_key = jax.random.split(rng)
        
        # Init with dummy input [1, Obs] and dummy carry [1, Hidden]
        dummy_obs = jnp.zeros((1, obs_dim))
        dummy_carry = jnp.zeros((1, self.hidden_dim))
        
        actor_params = actor_net.init(a_key, dummy_obs, dummy_carry)
        self.actor_state = TrainState.create(
            apply_fn=actor_net.apply, params=actor_params, tx=optax.adam(self.lr_actor)
        )
        
        # Init Critic (Feedforward for now, keep it simple/fast)
        # Critic sees global state.
        critic_net = MAPPOCritic(num_agents=num_agents) # Defined below
        rng, c_key = jax.random.split(rng)
        global_dim = num_agents * obs_dim
        critic_params = critic_net.init(c_key, jnp.zeros((1, global_dim)))
        self.critic_state = TrainState.create(
            apply_fn=critic_net.apply, params=critic_params, tx=optax.adam(self.lr_critic)
        )
        
        # Init World Model (Predictor)
        # Needed for Surprise-Gated Communication
        if agent_config and hasattr(agent_config, 'comm') and agent_config.comm.surprise_gating:
            from entropy.brain.world_model import WorldModelPredictor
            self.world_model = WorldModelPredictor(hidden_dim=self.hidden_dim)
            rng, wm_key = jax.random.split(rng)
            
            # Input: Obs + Action
            # Dummy Action needs to be correct size
            dummy_action = jnp.zeros((1, action_dim))
            
            wm_params = self.world_model.init(wm_key, dummy_obs, dummy_action)
            
            # Simple Adam for World Model
            self.wm_state = TrainState.create(
                apply_fn=self.world_model.apply, params=wm_params, tx=optax.adam(1e-3)
            )
        else:
            self.wm_state = None
        
    def update(self, buffer: Any) -> Tuple[float, float]:
        # Unwrap buffer
        # obs: [T, N, D]
        # carry: [T, N, H] (We need to store carries in buffer during rollout!)
        # Check if buffer has carries. If not, we can't train GRU properly without re-scanning.
        # For PPO Recurrent, we typically engage "Burn-in" or just use stored carries.
        # Assuming buffer has 'actor_states' which are the carries.
        
        obs = jnp.array(buffer['obs'])
        actions = jnp.array(buffer['actions'])
        old_log_probs = jnp.array(buffer['log_probs'])
        values = jnp.array(buffer['values'])
        rewards = jnp.array(buffer['rewards'])
        dones = jnp.array(buffer['dones'])
        carries = jnp.array(buffer['actor_states']) # Crucial: Hidden states from rollout
        
        # GAE
        advantages = self._calculate_gae(rewards, values, dones)
        targets = advantages + values[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Flatten [T, N] -> [Batch]
        T, N = obs.shape[:2]
        batch_size = T * N
        
        flat_obs = obs.reshape(batch_size, -1)
        flat_actions = actions.reshape(batch_size, -1)
        flat_log_probs = old_log_probs.reshape(batch_size)
        flat_adv = advantages.reshape(batch_size)
        flat_carries = carries.reshape(batch_size, -1)
        
        # Critic Update
        global_states = obs.reshape(T, -1) # Flatten all agents for global
        critic_loss = 0.0
        for _ in range(self.critic_updates):
             self.critic_state, l, _ = self._train_critic(
                 self.critic_state, global_states, targets.flatten(), N
             )
             critic_loss += l

        # Actor Update
        actor_loss = 0.0
        for _ in range(self.actor_updates):
            self.actor_state, l, _ = self._train_actor(
                self.actor_state, flat_obs, flat_actions, flat_log_probs, flat_adv, flat_carries
            )
            actor_loss += l
            
        # World Model Update
        if self.wm_state is not None:
             # Train on T-1 transitions
             # Obs[t], Action[t] -> Obs[t+1]
             # We use flattened batch but we need to respect sequence
             # buffer['obs'] is [T, N, D]
             
             obs_t = obs[:-1].reshape(-1, obs.shape[-1])
             act_t = actions[:-1].reshape(-1, actions.shape[-1])
             obs_next = obs[1:].reshape(-1, obs.shape[-1])
             # Check dones[t] to see if transition is valid
             dones_t = dones[:-1].reshape(-1)
             
             self.wm_state, wm_loss, _ = self._train_world_model(
                 self.wm_state, obs_t, act_t, obs_next, dones_t
             )
            
        return actor_loss, critic_loss

    @staticmethod
    def _calculate_gae(rewards, values, dones, gamma=0.99, lam=0.95):
        T = len(rewards)
        advantages = np.zeros_like(rewards)
        last_gae_lam = 0
        for t in reversed(range(T)):
            next_non_terminal = 1.0 - dones[t]
            next_value = values[t+1]
            delta = rewards[t] + gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_gae_lam = delta + gamma * lam * next_non_terminal * last_gae_lam
        return jnp.array(advantages)

    @staticmethod
    @jax.jit
    def _train_critic(state, global_state, targets, num_agents):
        def loss_fn(params):
            values = state.apply_fn(params, global_state)
            loss = jnp.mean((values.reshape(-1) - targets) ** 2)
            return loss
        grads = jax.grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss_fn(state.params), None

    @staticmethod
    @jax.jit
    def _train_actor(state, obs, actions, old_log_probs, advantages, carries):
        def loss_fn(params):
            # Recurrent Forward
            # Note: We are using stored carries (Teacher Forcing style for hidden state)
            # This is technically an approximation if we don't re-rollout the sequence.
            # But for simple PPO with short windows it often suffices.
            # Ideally: Reshape to [T, N, ...], scan with re-init.
            # For "Massive" speed, we use the "Independent Step" approximation:
            # We assume the stored 'carry' is valid input for this step's gradient.
            
            _, mean, log_std = state.apply_fn(params, obs, carries)
            
            dist = distrax.MultivariateNormalDiag(mean, jnp.exp(log_std))
            log_probs = dist.log_prob(actions)
            
            ratio = jnp.exp(log_probs - old_log_probs)
            clip_ratio = jnp.clip(ratio, 0.8, 1.2)
            
            pg_loss = -jnp.minimum(advantages * ratio, advantages * clip_ratio).mean()
            entropy = dist.entropy().mean()
            
            return pg_loss - 0.01 * entropy
            
        grads = jax.grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss_fn(state.params), None
    
    @staticmethod
    @jax.jit
    def _train_world_model(state, obs, actions, next_obs, dones):
        def loss_fn(params):
            pred_next_obs = state.apply_fn(params, obs, actions)
            # Mask out transitions that crossed episode boundary
            mask = 1.0 - dones
            
            # MSE Loss
            sq_err = (pred_next_obs - next_obs) ** 2
            # Average over features [..., D] -> [...]
            mse = jnp.mean(sq_err, axis=-1)
            
            # Apply mask
            masked_loss = mse * mask
            
            # Normalize by valid samples
            loss = jnp.sum(masked_loss) / (jnp.sum(mask) + 1e-6)
            return loss
            
        grads = jax.grad(loss_fn)(state.params)
        state = state.apply_gradients(grads=grads)
        return state, loss_fn(state.params), None

class MAPPOCritic(nn.Module):
    num_agents: int
    hidden_dim: int = 256
    @nn.compact
    def __call__(self, global_state):
        x = nn.Dense(self.hidden_dim)(global_state)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim)(x)
        x = nn.relu(x)
        return nn.Dense(self.num_agents)(x)

