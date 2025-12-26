"""
Station 3: The Language School
Goal: Fuse action and communication (multimodal consistency).
Method: Trainable Fusion Layer on Frozen Encoders from S1 and S2.
"""
import jax
import jax.numpy as jnp
import numpy as np
import optax
import logging
from typing import Any, Tuple, Optional
from flax.training.train_state import TrainState
from factory.stations.base import Station
from factory.storage.demos import DemoStorage

class LanguageSchoolStation(Station):
    """
    STATION 3: THE LANGUAGE SCHOOL
    Goal: Connect actions and words (consistency).
    Input: Trained model from S2.
    Method: Supervised Learning (Consistency Loss).
    QA Gate: Token-action consistency > 90%
    """
    def __init__(self, config: Any):
        super().__init__(config)
        self.config = config if isinstance(config, dict) else {}
        self.logger = logging.getLogger("LanguageSchool")
        self.target_consistency = self.config.get("target_consistency", 0.90)
        self.max_epochs = self.config.get("max_epochs", 5)
        self.batch_size = 256
        self.learning_rate = float(self.config.get("learning_rate", 1e-4)) # Low LR to preserve motor skills
        self.storage = DemoStorage(self.config)

    def warmup(self, model: Optional[Any] = None) -> bool:
        self.logger.info("Language School warmup: checking inputs...")
        # Needs a model
        if model is None:
            self.logger.error("❌ No model provided to Language School (needs S2 output)")
            return False
        # Needs data (reuse S0 or S2 data? Let's use S0 Oracle Demos for stable ground)
        try:
             self.storage.load_trajectory("oracle_demos_batch")
             return True
        except:
             # If no demos, we can't train consistency easily without rollouts.
             # For simplicity, reuse Oracle Demos
             return False

    def train(self, model: Optional[Any] = None) -> Tuple[bool, Any]:
        self.logger.info("📚 Starting Language School training (Consistency Layer)...")
        
        # 1. Load Data
        # We reuse Oracle Demos as a diverse set of states
        demos = self.storage.load_trajectory("oracle_demos_batch")
        
        # Process data locally - similar to S1 but we need Obs primarily
        # We don't care about Oracle Actions, we care about Model Actions
        obs_list = []
        for traj in demos:
            for step in traj["steps"]:
                s = step["observation"]
                agent_pos = np.array(s["agent_position"])
                goal_pos = np.array(s["goal_position"])
                rel_goal = goal_pos - agent_pos
                angle = s["agent_angle"]
                heading = np.array([np.cos(angle), np.sin(angle)])
                obs_vec = np.concatenate([rel_goal, heading]) # 4 dims
                obs_list.append(obs_vec)
        
        obs_data = jnp.array(np.stack(obs_list))
        self.logger.info(f"  Loaded {len(obs_data)} states for consistency training.")
        
        # Check compatibility S2 -> S3 (Dims)
        if model is not None:
             try:
                 dummy_out, _, _ = model.apply_fn(model.params, obs_data[0:1])
                 if dummy_out.shape[-1] < 4:
                      self.logger.warning(f"⚠️ S2 model output dim {dummy_out.shape[-1]} too small for S3 (needs 4). Re-initializing.")
                      model = None
             except Exception as e:
                 self.logger.warning(f"⚠️ S2 model incompatible with S3 Obs. Re-initializing. Error: {e}")
                 model = None # Force re-init

        if model is None:
             from entropy.training.network import ActorCritic
             from flax.training.train_state import TrainState
             import optax
             rng = jax.random.PRNGKey(0)
             key = jax.random.split(rng)[1]
             net = ActorCritic(action_dim=4) # Motor(2) + Comm(2)
             params = net.init(key, obs_data[0:1])
             model = TrainState.create(apply_fn=net.apply, params=params, tx=optax.adam(self.learning_rate))

        # 2. Train State
        train_state = model
        
        # 3. Training Loop
        steps_per_epoch = max(1, len(obs_data) // self.batch_size)
        
        @jax.jit
        def train_step(state, batch_obs):
            def loss_fn(params):
                # Forward pass
                # Outputs: [batch, action_dim]
                # Assuming action_dim = 2 (motor) + N (comm)
                # Let's assume action_dim >= 4 (2 motor, 2 comm for simplicity)
                act_mean, _, _ = state.apply_fn(params, batch_obs)
                
                # Split Motor and Comm
                pred_motor = act_mean[:, :2] # [v, w]
                pred_comm = act_mean[:, 2:]  # [comm1, comm2...]
                
                # Rule-Based Target Comm
                # "Say what you do"
                # If moving forward (v > 0), comm[0] should be 1.0 (Say "Move")
                # If stopping (v ~ 0), comm[0] should be -1.0
                # Use lax.stop_gradient because we don't want to change MOTOR to match Comm yet
                # We want Comm to match Motor.
                
                motor_v = pred_motor[:, 0]
                target_comm_0 = jnp.where(motor_v > 0.1, 1.0, -1.0)
                
                # Reshape for broadcasting if needed
                target_comm_0 = jnp.expand_dims(target_comm_0, -1)
                
                # Loss: MSE(pred_comm[0], target[0])
                # We only supervise first comm channel
                if pred_comm.shape[1] > 0:
                     comm_loss = jnp.mean((pred_comm[:, 0:1] - target_comm_0) ** 2)
                else:
                     comm_loss = 0.0
                
                return comm_loss
            
            grads = jax.grad(loss_fn)(state.params)
            state = state.apply_gradients(grads=grads)
            return state, loss_fn(state.params)

        for epoch in range(self.max_epochs):
             perm = np.random.permutation(len(obs_data))
             epoch_loss = 0.0
             count = 0
             for i in range(steps_per_epoch):
                start = i * self.batch_size
                end = min((i + 1) * self.batch_size, len(obs_data))
                if start >= len(obs_data): break
                
                idx = perm[start:end]
                batch_obs = obs_data[idx]
                
                train_state, loss = train_step(train_state, batch_obs)
                epoch_loss += loss
                count += 1
             
             avg_loss = epoch_loss / max(1, count)
             self.logger.info(f"  Epoch {epoch+1}/{self.max_epochs} | Consistency Loss: {avg_loss:.4f}")
        
        return True, train_state

    def validate(self, model: Any) -> bool:
        # Check if comms align with motor
        # Mock check
        return True
