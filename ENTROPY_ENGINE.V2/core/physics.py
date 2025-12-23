import pymunk
from typing import List, Tuple, Optional, Dict
import math

class PhysicsWorld:
    def __init__(self, dt: float = 1/60.0):
        self.dt = dt
        self.space = pymunk.Space()
        self.space.gravity = (0, 0) # Top-down, no gravity
        self.space.damping = 0.5    # Global damping to simulate friction/air resistance
        
        # Collision Handlers
        # We will define them later as needed
        
    def step(self):
        self.space.step(self.dt)
        
    def add(self, *objects):
        self.space.add(*objects)
        
    def remove(self, *objects):
        self.space.remove(*objects)

class Entity:
    def __init__(self, world: PhysicsWorld, position: Tuple[float, float], angle: float = 0.0):
        self.world = world
        self.body = self._create_body(position, angle)
        self.shape = self._create_shape()
        self.world.add(self.body, self.shape)
        
    def _create_body(self, position, angle) -> pymunk.Body:
        raise NotImplementedError
        
    def _create_shape(self) -> pymunk.Shape:
        raise NotImplementedError
    
    @property
    def position(self):
        return self.body.position
    
    @property
    def angle(self):
        return self.body.angle
    
    @property
    def velocity(self):
        return self.body.velocity
        
    def destroy(self):
        self.world.remove(self.body, self.shape)
