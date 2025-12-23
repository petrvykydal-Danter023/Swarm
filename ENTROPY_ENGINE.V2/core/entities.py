import pymunk
import math
import numpy as np
from typing import List, Tuple, Optional
from .world import PhysicsWorld

# Collision Types
COLLISION_AGENT = 1
COLLISION_WALL = 2
COLLISION_GOAL = 3

class Entity:
    def __init__(self, world: PhysicsWorld, position: Tuple[float, float], angle: float = 0.0):
        self.world = world
        self.body = None
        self.shape = None
        self._init_physics(position, angle)
        self.world.add(self.body, self.shape)
        
    def _init_physics(self, position, angle):
        raise NotImplementedError
        
    @property
    def position(self):
        return self.body.position
    
    @property
    def angle(self):
        return self.body.angle
    
    def destroy(self):
        self.world.remove(self.body, self.shape)

class Wall(Entity):
    def __init__(self, world: PhysicsWorld, p1: Tuple[float, float], p2: Tuple[float, float], thickness: float = 5.0):
        self.p1 = p1
        self.p2 = p2
        self.thickness = thickness
        super().__init__(world, (0,0)) # Position unused for static body
        
    def _init_physics(self, position, angle):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC)
        self.body.position = position # Use the position passed to init
        self.shape = pymunk.Segment(self.body, self.p1, self.p2, self.thickness)
        self.shape.elasticity = 0.5
        self.shape.friction = 0.5
        self.shape.collision_type = COLLISION_WALL

class Agent(Entity):
    def __init__(self, world: PhysicsWorld, position: Tuple[float, float], radius: float = 10.0, color=(0,0,255)):
        self.radius = radius
        self.color = color
        self.lidar_rays = 32
        self.lidar_range = 300.0
        self.sensors = np.zeros(self.lidar_rays, dtype=np.float32)
        self.current_signal = np.zeros(2, dtype=np.float32) # [Tx1, Tx2]
        super().__init__(world, position)
        
    def _init_physics(self, position, angle):
        mass = 1.0
        moment = pymunk.moment_for_circle(mass, 0, self.radius)
        self.body = pymunk.Body(mass, moment)
        self.body.position = position
        self.body.angle = angle
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.elasticity = 0.2
        self.shape.friction = 0.7
        self.shape.collision_type = COLLISION_AGENT
        
    def set_signal(self, s1: float, s2: float):
        self.current_signal[:] = [s1, s2]

    def control(self, left_motor: float, right_motor: float):
        # Differential drive logic
        # Motors apply force at offsets
        force_mult = 1000.0
        
        # Clamp inputs
        left_motor = max(-1.0, min(1.0, left_motor))
        right_motor = max(-1.0, min(1.0, right_motor))
        
        # Apply forces relative to body
        # Left motor is at (0, -radius), pointing forward (x+)
        # Right motor is at (0, radius), pointing forward (x+)
        # Wait, usually X is forward in Pymunk? Let's assume X is forward.
        
        f_left = left_motor * force_mult
        f_right = right_motor * force_mult
        
        # Force vector is along local X axis
        self.body.apply_force_at_local_point((f_left, 0), (0, -self.radius))
        self.body.apply_force_at_local_point((f_right, 0), (0, self.radius))
        
    def update_sensors(self):
        # Raycast Lidar
        start_angle = self.body.angle
        angle_step = (2 * math.pi) / self.lidar_rays
        
        # Filter: Don't hit self (handled by start offset)
        filter_ = pymunk.ShapeFilter() 
        # Actually we need to exclude SELF. 
        # Pymunk raycast filter "exclude" doesn't work easily with specific shapes unless we use categories/masks.
        # For now, we raycast from slightly outside the radius to avoid self-hit.
        
        for i in range(self.lidar_rays):
            angle = start_angle + i * angle_step
            # Start slightly outside to avoid self
            start_pos = self.body.position + pymunk.Vec2d(self.radius + 1.0, 0).rotated(angle - self.body.angle)
            end_pos = self.body.position + pymunk.Vec2d(self.lidar_range, 0).rotated(angle - self.body.angle)
            
            # Simple global raycast
            # We want to hit Walls(2) and other Agents(1).
            # filter=pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS())
            
            res = self.world.space.segment_query_first(start_pos, end_pos, 1.0, pymunk.ShapeFilter())
            
            if res:
                dist = res.alpha # 0 to 1
                self.sensors[i] = dist
            else:
                self.sensors[i] = 1.0 # Max range

class Goal(Entity):
    def __init__(self, world: PhysicsWorld, position: Tuple[float, float], radius: float = 15.0):
        self.radius = radius
        super().__init__(world, position)

    def _init_physics(self, position, angle):
        self.body = pymunk.Body(body_type=pymunk.Body.STATIC) # Static or Sensor?
        self.body.position = position
        
        self.shape = pymunk.Circle(self.body, self.radius)
        self.shape.sensor = True # Goal is a sensor (no physical collision response)
        self.shape.collision_type = COLLISION_GOAL
