"""
Entropy Engine V3 - Sim-to-Real Bridge
Domain Randomization and Sensor Noise models.
"""
import jax
import jax.numpy as jnp
from typing import Tuple, Dict

def apply_domain_randomization(
    rng: jax.Array, 
    masses: jnp.ndarray,
    friction: float = 1.0,
    motor_strength: float = 1.0
) -> Tuple[jnp.ndarray, float, float]:
    """
    Randomize physics parameters for the episode.
    """
    rng1, rng2, rng3 = jax.random.split(rng, 3)
    
    # Randomize masses (multiply by 0.8-1.2)
    mass_scale = jax.random.uniform(rng1, shape=masses.shape, minval=0.8, maxval=1.2)
    new_masses = masses * mass_scale
    
    # Randomize friction
    friction_scale = jax.random.uniform(rng2, minval=0.5, maxval=1.5)
    new_friction = friction * friction_scale
    
    # Randomize motor strength
    motor_scale = jax.random.uniform(rng3, minval=0.9, maxval=1.1)
    new_motor = motor_strength * motor_scale
    
    return new_masses, new_friction, new_motor

def add_sensor_noise(
    rng: jax.Array, 
    readings: jnp.ndarray, 
    noise_std: float = 0.05,
    dropout_prob: float = 0.01
) -> jnp.ndarray:
    """
    Add Gaussian noise and dropouts to sensor readings.
    """
    rng_noise, rng_drop = jax.random.split(rng)
    
    # Gaussian noise
    noise = jax.random.normal(rng_noise, shape=readings.shape) * noise_std
    noisy = readings + noise
    
    # Clip to valid range
    noisy = jnp.clip(noisy, 0.0, 1.0)
    
    # Dropouts (sensor failure -> returns 0.0 or 1.0? usually max range = 1.0)
    # Lidar usually returns max range on failure/inf
    mask = jax.random.uniform(rng_drop, shape=readings.shape) > dropout_prob
    noisy = jnp.where(mask, noisy, 1.0)
    
    return noisy

def add_actuator_noise(
    rng: jax.Array,
    actions: jnp.ndarray,
    noise_std: float = 0.02
) -> jnp.ndarray:
    """
    Add noise to motor commands.
    """
    noise = jax.random.normal(rng, shape=actions.shape) * noise_std
    noisy_actions = jnp.clip(actions + noise, -1.0, 1.0)
    return noisy_actions
