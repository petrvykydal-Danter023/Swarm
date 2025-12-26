"""
Entropy Engine V3 - Hand of God (Shared Autonomy)
Implementations of Expert Policies, Action Mixing, and Alpha Scheduling.
"""
from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp
from dataclasses import dataclass

# ==============================================================================
# 1. EXPERT INTERFACES
# ==============================================================================

class ExpertPolicy(ABC):
    """
    Interface for privileged experts that have access to full state.
    """
    @abstractmethod
    def act(self, state, rng: jax.Array) -> jnp.ndarray:
        """Returns action [N, action_dim]."""
        pass

class DirectNavigator(ExpertPolicy):
    """
    A simple expert that navigates directly to the goal.
    This acts as a heuristic navigator (simulating A* in open space).
    """
    def act(self, state, rng: jax.Array) -> jnp.ndarray:
        # Vector to goal
        diff = state.goal_positions - state.agent_positions
        dist = jnp.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
        direction = diff / dist
        
        # Convert to differential drive (simplified: just point and move)
        # We need to map direction (global) to agent frame
        # Agent heading:
        headings = state.agent_angles
        # Relative angle
        target_angles = jnp.arctan2(direction[:, 1], direction[:, 0])
        angle_diff = target_angles - headings
        # Normalize -pi to pi
        angle_diff = (angle_diff + jnp.pi) % (2 * jnp.pi) - jnp.pi
        
        # Simple controller: 
        # linear proptional to cos(angle_error)
        # angular proportional to angle_error
        linear = jnp.clip(jnp.cos(angle_diff), 0.0, 1.0)
        angular = jnp.clip(angle_diff, -1.0, 1.0)
        
        # Map linear/angular to left/right motors
        # v = (l+r)/2, w = (r-l)/L
        # r = v + w*L/2, l = v - w*L/2
        # (simplified for normalized actions)
        left = linear - angular
        right = linear + angular
        
        actions = jnp.stack([left, right], axis=1)
        return jnp.clip(actions, -1.0, 1.0)

class FallibleExpert(ExpertPolicy):
    """
    Expert wrapper that intentionally makes mistakes.
    Used for adversarial injection to train critical thinking.
    """
    def __init__(self, base_expert: ExpertPolicy, error_rate: float = 0.05, severity: float = 1.0):
        self.base_expert = base_expert
        self.error_rate = error_rate
        self.severity = severity

    def act(self, state, rng: jax.Array) -> jnp.ndarray:
        # Get correct actions
        rng_act, rng_noise = jax.random.split(rng)
        correct_actions = self.base_expert.act(state, rng_act)
        
        # Generate mistake mask
        mistake_mask = jax.random.uniform(rng_noise, shape=(state.agent_positions.shape[0],)) < self.error_rate
        
        # Generate random actions (mistakes)
        # Or specifically bad actions (opposite direction?)
        # Let's do random noise for now
        random_actions = jax.random.uniform(rng_noise, shape=correct_actions.shape, minval=-1.0, maxval=1.0)
        
        # If severity is high, maybe inverse of correct?
        if self.severity > 0.8:
            bad_actions = -correct_actions
        else:
            bad_actions = random_actions
            
        # Mix
        # Where mistake_mask is True, use bad_actions
        final_actions = jnp.where(mistake_mask[:, None], bad_actions, correct_actions)
        return final_actions

# ==============================================================================
# 2. ACTION MIXER
# ==============================================================================

class ActionMixer:
    """Blends AI and Expert actions."""
    
    @staticmethod
    def mix(ai_actions: jnp.ndarray, expert_actions: jnp.ndarray, alpha: float) -> jnp.ndarray:
        """Linear interpolation: (1-alpha)*AI + alpha*Expert."""
        return (1.0 - alpha) * ai_actions + alpha * expert_actions

# ==============================================================================
# 3. ALPHA SCHEDULER
# ==============================================================================

class AlphaScheduler:
    """Manages the decay of assistance."""
    def __init__(self, initial: float = 1.0, final: float = 0.0, total_steps: int = 100_000):
        self.initial = initial
        self.final = final
        self.total_steps = total_steps
        
    def get_alpha(self, step: int) -> float:
        progress = jnp.clip(step / self.total_steps, 0.0, 1.0)
        return self.initial + (self.final - self.initial) * progress

# ==============================================================================
# 4. FORMATION EXPERT
# ==============================================================================

class FormationExpert(ExpertPolicy):
    """
    Expert for geometric formations.
    Calculates target positions based on leader and formation type.
    """
    def __init__(self, formation_type: str = "line", spacing: float = 30.0):
        self.formation_type = formation_type
        self.spacing = spacing
        
    def act(self, state, rng: jax.Array) -> jnp.ndarray:
        n = state.agent_positions.shape[0]
        
        # Leader = Agent 0
        leader_pos = state.agent_positions[0]
        leader_angle = state.agent_angles[0]
        
        # Calculate target positions
        if self.formation_type == "line":
            targets = self._line_formation(leader_pos, leader_angle, n)
        elif self.formation_type == "v":
            targets = self._v_formation(leader_pos, leader_angle, n)
        elif self.formation_type == "circle":
            targets = self._circle_formation(leader_pos, n)
        else:
            targets = jnp.tile(leader_pos, (n, 1))  # Default: all follow leader
            
        # Navigate each agent to their target
        diff = targets - state.agent_positions
        dist = jnp.linalg.norm(diff, axis=1, keepdims=True) + 1e-6
        direction = diff / dist
        
        # Convert to differential drive
        headings = state.agent_angles
        target_angles = jnp.arctan2(direction[:, 1], direction[:, 0])
        angle_diff = target_angles - headings
        angle_diff = (angle_diff + jnp.pi) % (2 * jnp.pi) - jnp.pi
        
        linear = jnp.clip(jnp.cos(angle_diff), 0.0, 1.0)
        angular = jnp.clip(angle_diff, -1.0, 1.0)
        
        left = linear - angular
        right = linear + angular
        
        actions = jnp.stack([left, right], axis=1)
        return jnp.clip(actions, -1.0, 1.0)
    
    def _line_formation(self, leader_pos, leader_angle, n):
        """Horizontal line perpendicular to leader's heading."""
        perpendicular = jnp.array([-jnp.sin(leader_angle), jnp.cos(leader_angle)])
        targets = []
        for i in range(n):
            offset = perpendicular * self.spacing * (i - n // 2)
            targets.append(leader_pos + offset)
        return jnp.stack(targets)
    
    def _v_formation(self, leader_pos, leader_angle, n):
        """V-shape behind leader."""
        backward = jnp.array([-jnp.cos(leader_angle), -jnp.sin(leader_angle)])
        perpendicular = jnp.array([-jnp.sin(leader_angle), jnp.cos(leader_angle)])
        targets = [leader_pos]  # Leader at front
        for i in range(1, n):
            side = 1 if i % 2 == 0 else -1
            row = (i + 1) // 2
            offset = backward * self.spacing * row + perpendicular * self.spacing * row * side * 0.5
            targets.append(leader_pos + offset)
        return jnp.stack(targets)
    
    def _circle_formation(self, center_pos, n):
        """Circle around a center point."""
        angles = jnp.linspace(0, 2 * jnp.pi, n, endpoint=False)
        radius = self.spacing * n / (2 * jnp.pi)
        xs = center_pos[0] + radius * jnp.cos(angles)
        ys = center_pos[1] + radius * jnp.sin(angles)
        return jnp.stack([xs, ys], axis=1)

# ==============================================================================
# 5. TIME TRAVEL (Rewind)
# ==============================================================================

from collections import deque
from copy import deepcopy

class TimeTravel:
    """
    Allows rewinding the simulation to replay with expert actions.
    Useful for training on "what would expert have done" scenarios.
    """
    def __init__(self, buffer_size: int = 300):
        self.history = deque(maxlen=buffer_size)
        self.rng_states = deque(maxlen=buffer_size)
        
    def save(self, state, rng_state):
        """Save a snapshot."""
        self.history.append(deepcopy(state))
        self.rng_states.append(rng_state)
        
    def rewind(self, steps: int = 60):
        """Go back in time."""
        if len(self.history) >= steps:
            return self.history[-steps], self.rng_states[-steps]
        elif len(self.history) > 0:
            return self.history[0], self.rng_states[0]
        return None, None
    
    def clear(self):
        """Clear history (e.g., on episode reset)."""
        self.history.clear()
        self.rng_states.clear()

# ==============================================================================
# 6. GHOST RENDERER (Visualization Helper)
# ==============================================================================

class GhostRenderer:
    """
    Renders 'ghost' positions showing what the expert would do vs AI.
    Returns data for the main renderer to draw.
    """
    def __init__(self, opacity: float = 0.5):
        self.opacity = opacity
        
    def compute_ghosts(self, state, ai_actions, expert_actions, dt: float = 0.1):
        """
        Predict next positions for both AI and Expert paths.
        Returns ghost data for visualization.
        """
        # Simple forward prediction
        ai_velocity = ai_actions[:, 0] + ai_actions[:, 1]  # Simplified linear
        expert_velocity = expert_actions[:, 0] + expert_actions[:, 1]
        
        ai_next = state.agent_positions + jnp.stack([
            ai_velocity * jnp.cos(state.agent_angles),
            ai_velocity * jnp.sin(state.agent_angles)
        ], axis=1) * dt * 50  # Scale for visibility
        
        expert_next = state.agent_positions + jnp.stack([
            expert_velocity * jnp.cos(state.agent_angles),
            expert_velocity * jnp.sin(state.agent_angles)
        ], axis=1) * dt * 50
        
        return {
            'ai_positions': ai_next,
            'expert_positions': expert_next,
            'divergence': jnp.linalg.norm(ai_next - expert_next, axis=1),
            'opacity': self.opacity
        }


class IntentNavigator(ExpertPolicy):
    """
    Expert for Hybrid Mode (Intent-Based).
    Outputs INTENT vectors instead of MOTOR commands.
    
    Target Intent Space: [IntentID, Param1, Param2, ...]
    - IntentID > 0.0 -> Target Mode
    - Param1 = RelX (normalized)
    - Param2 = RelY (normalized)
    """
    def __init__(self, num_agents: int):
        self.num_agents = num_agents
        
    def act(self, state, rng: jax.Array) -> jnp.ndarray:
        # Calculate target relative to agent
        diff = state.goal_positions - state.agent_positions
        
        # Transform global diff to local (body) frame
        # So Param1 is "Forward Distance", Param2 is "Right Distance"
        # Inverse rotation by agent angle
        angles = state.agent_angles
        cos_a = jnp.cos(-angles)
        sin_a = jnp.sin(-angles)
        
        local_x = diff[:, 0] * cos_a - diff[:, 1] * sin_a
        local_y = diff[:, 0] * sin_a + diff[:, 1] * cos_a
        
        # Normalize to unit vector [-1, 1]
        # This tells the PID controller the DIRECTION to the target,
        # but keeps input magnitude stable for the Neural Network.
        local_dist = jnp.sqrt(local_x**2 + local_y**2) + 1e-6
        norm_x = local_x / local_dist
        norm_y = local_y / local_dist
        
        # Construct Action Vector
        # [IntentID(Target), Param1, Param2]
        
        # IntentID > 0 for Target Mode. Let's use 1.0.
        ids = jnp.ones((self.num_agents, 1))
        
        # Params
        p1 = norm_x.reshape(-1, 1)
        p2 = norm_y.reshape(-1, 1)
        
        intent_core = jnp.concatenate([ids, p1, p2], axis=1)
        return intent_core

