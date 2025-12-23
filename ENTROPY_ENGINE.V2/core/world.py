import pymunk

class PhysicsWorld:
    def __init__(self, dt: float = 1/60.0, damping: float = 0.5):
        self.dt = dt
        self.space = pymunk.Space()
        self.space.gravity = (0, 0) 
        self.space.damping = damping
        
    def step(self):
        self.space.step(self.dt)
        
    def add(self, *objects):
        self.space.add(*objects)
        
    def remove(self, *objects):
        self.space.remove(*objects)
