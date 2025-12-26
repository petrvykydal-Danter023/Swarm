import pytest
import jax
import jax.numpy as jnp
from entropy.safety.reflexes import apply_collision_reflex
from entropy.safety.watchdog import apply_watchdog, create_watchdog_state
from entropy.safety.geofence import apply_geofence
from entropy.config import SafetyConfig
from entropy.core.world import create_initial_state

class TestCollisionReflex:
    
    def test_speed_reduction_near_wall(self):
        """Agent blízko zdi by měl být zpomalen."""
        config = SafetyConfig(safety_radius=30.0)
        state = create_initial_state(num_agents=1, arena_size=(800.0, 600.0))
        # Agent u levé zdi (x=10, y=300), moving right
        state = state.replace(agent_positions=jnp.array([[10.0, 300.0]]))
        
        raw_actions = jnp.array([[1.0, 0.0, 0.0, 0.0]])  # Full speed right (idx 0 is L, idx 1 is R in tank drive? Or Speed/Rot?)
        # Let's assume actions are [Motor_L, Motor_R, ...] for default physics (tank drive)
        # 1.0, 1.0 would be full forward. 1.0, 0.0 is turn? behavior depends on physics model.
        # But reflex creates speed_factor and multiplies both motors.
        
        safe_actions = apply_collision_reflex(state, raw_actions, config)
        
        # Speed should be reduced (ratio = 10/30 = 0.33)
        # raw action 1.0 -> 0.33
        assert safe_actions[0, 0] < raw_actions[0, 0]
        # Use approx since it might be slightly different due to min_obstacle calc
        assert safe_actions[0, 0] == pytest.approx(0.333, rel=0.1)
    
    def test_no_reduction_far_from_obstacle(self):
        """Agent daleko od překážky by neměl být ovlivněn."""
        config = SafetyConfig(safety_radius=30.0)
        state = create_initial_state(num_agents=1)
        state = state.replace(agent_positions=jnp.array([[400.0, 300.0]]))
        
        raw_actions = jnp.array([[1.0, 1.0, 0.0, 0.0]])
        safe_actions = apply_collision_reflex(state, raw_actions, config)
        
        assert jnp.allclose(safe_actions[:, :2], raw_actions[:, :2])
    
    def test_repulsion_between_agents(self):
        """Dva blízcí agenti by se měli odpuzovat."""
        config = SafetyConfig(enable_repulsion=True, repulsion_radius=25.0, repulsion_force=0.5)
        state = create_initial_state(num_agents=2)
        # Agents very close on X axis
        # Agent 0 at 400.0, Agent 1 at 410.0
        state = state.replace(agent_positions=jnp.array([
            [400.0, 300.0],
            [410.0, 300.0]
        ]))
        
        # Assign different squads to ensure they repel (Same squad agents do not repel)
        state = state.replace(agent_squad_ids=jnp.array([0, 1]))
        
        raw_actions = jnp.zeros((2, 4))
        safe_actions = apply_collision_reflex(state, raw_actions, config)
        
        # Agent 0 should be pushed left (-X) -> Motor values changed
        # Repulsion vec for agent 0 is 400-410 = -10 (Left)
        # Agent 1 should be pushed right (+X)
        
        # Check sign of change
        assert safe_actions[0, 0] < 0  # Motor L pushed negative/modified
        assert safe_actions[1, 0] > 0
        

class TestWatchdog:
    
    def test_stalemate_detection(self):
        """Agent který se nehýbe by měl být detekován."""
        config = SafetyConfig(
            stalemate_window=10,
            stalemate_min_distance=5.0,
            stalemate_random_duration=5,
            stalemate_random_speed=0.5
        )
        state = create_initial_state(num_agents=1)
        state = state.replace(agent_positions=jnp.array([[100.0, 100.0]]))
        
        watchdog = create_watchdog_state(1)
        # Initialize old position to current position to simulate "not moving" from start
        watchdog = watchdog.replace(position_old=state.agent_positions)
        rng = jax.random.PRNGKey(0)
        
        # Simulate 11 steps (window is 10)
        # We need steps_since_snapshot to reach 10
        actions = jnp.zeros((1, 4))
        
        # Step 0 to 9 (10 steps)
        for _ in range(config.stalemate_window + 1):
             actions, watchdog = apply_watchdog(state, actions, watchdog, config, rng)
             # To simulate time passing we must inc watchdog steps (done inside apply)
             # But apply_watchdog resets steps if triggered.
             
        # After window steps, if moved < 5.0, random walk should trigger
        # We didn't move state, so distance is 0.
        
        # Check if random walk is active
        assert watchdog.random_walk_remaining[0] > 0

class TestGeoFence:
    
    def test_push_from_wall(self):
        """Agent u hranice by měl být tlačen dovnitř."""
        config = SafetyConfig(
            geofence_push_distance=30.0,
            geofence_push_force=1.0
        )
        state = create_initial_state(num_agents=1, arena_size=(800.0, 600.0))
        # Agent at x=15 (close to left wall 0)
        state = state.replace(
            agent_positions=jnp.array([[15.0, 300.0]])
        )
        
        raw_actions = jnp.zeros((1, 4))
        safe_actions = apply_geofence(state, raw_actions, [], config)
        
        # Should be pushed right (+X)
        # Left force is positive
        assert safe_actions[0, 0] > 0
