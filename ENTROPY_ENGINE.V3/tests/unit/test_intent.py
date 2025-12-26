import pytest
import jax.numpy as jnp
from entropy.safety.intent import process_intent
from entropy.config import IntentConfig
from entropy.core.world import create_initial_state

class TestIntentTranslator:
    
    def test_velocity_mode(self):
        """Test translation of Velocity Intent (Type < 0)."""
        config = IntentConfig(enabled=True)
        state = create_initial_state(num_agents=1)
        
        # Intent: [Type=-1.0, V=1.0, Omega=0.0] -> Forward Full Speed
        actions = jnp.array([[-1.0, 1.0, 0.0, 0.0]])
        
        motor_actions = process_intent(state, actions, config)
        
        # Expecting L=1.0, R=1.0 (V + Omega, V - Omega)
        assert motor_actions[0, 0] == pytest.approx(1.0)
        assert motor_actions[0, 1] == pytest.approx(1.0)
        
        # Rotation only: [Type=-1.0, V=0.0, Omega=0.5]
        actions_rot = jnp.array([[-1.0, 0.0, 0.5, 0.0]])
        motor_rot = process_intent(state, actions_rot, config)
        
        # L = V - Omega = -0.5
        # R = V + Omega = 0.5
        assert motor_rot[0, 0] == pytest.approx(-0.5)
        assert motor_rot[0, 1] == pytest.approx(0.5)

    def test_target_mode_forward(self):
        """Test Target Intent (Type > 0) - Target Ahead."""
        config = IntentConfig(
            enabled=True,
            pid_pos_kp=1.0,
            pid_rot_kp=1.0
        )
        state = create_initial_state(num_agents=1)
        
        # Target is at x=10, y=0 (Relative). Agent is at 0,0 angle 0.0
        # Intent: [Type=1.0, RelX=10.0, RelY=0.0]
        actions = jnp.array([[1.0, 10.0, 0.0, 0.0]])
        
        motor_actions = process_intent(state, actions, config)
        
        # Angle err = 0. Dist = 10.
        # Turn = 0.
        # Drive = 10 * Kp = 10 -> Clipped to 1.0
        # Motors: L=1, R=1
        assert motor_actions[0, 0] == pytest.approx(1.0)
        assert motor_actions[0, 1] == pytest.approx(1.0)

    def test_target_mode_turn(self):
        """Test Target Intent - Target to the Left."""
        config = IntentConfig(enabled=True)
        state = create_initial_state(num_agents=1)
        
        # Target at x=0, y=10 (Relative Left)
        # Angle = +90 deg (pi/2)
        actions = jnp.array([[1.0, 0.0, 10.0, 0.0]])
        
        motor_actions = process_intent(state, actions, config)
        
        # Turn should be positive (Left turn -> L < R)
        # Drive might be damped because cos(pi/2) ~ 0
        
        L = motor_actions[0, 0]
        R = motor_actions[0, 1]
        
        assert L < R # Should turn left
