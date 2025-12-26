"""
Entropy Engine V3 - Entity-Component-System
Implements: 02_ECS_ENTITIES.txt blueprint
"""
from dataclasses import dataclass, field
from typing import Dict, Type, Any, Optional, Set
from enum import IntEnum, auto
import jax.numpy as jnp

# ================================================================================
# 1. ENTITY TYPES
# ================================================================================

class EntityType(IntEnum):
    AGENT = auto()
    GOAL = auto()
    OBSTACLE = auto()
    RESOURCE = auto()
    WALL = auto()

# ================================================================================
# 2. COMPONENTS (Data Schemas)
# ================================================================================

@dataclass
class TransformComponent:
    """Position and orientation of an entity."""
    position: jnp.ndarray  # [2] - x, y
    angle: float = 0.0     # radians

@dataclass
class PhysicsComponent:
    """Physical properties for movable entities."""
    velocity: jnp.ndarray       # [2] - vx, vy
    angular_velocity: float = 0.0
    mass: float = 1.0
    radius: float = 10.0        # Collision radius

@dataclass
class RenderComponent:
    """Visual properties for rendering."""
    color: tuple = (0, 0, 255)  # RGB
    shape: str = "circle"       # "circle", "rectangle", "sprite"
    sprite_path: Optional[str] = None

@dataclass
class SensorComponent:
    """Lidar sensors."""
    lidar_rays: int = 32
    lidar_range: float = 300.0
    lidar_readings: Optional[jnp.ndarray] = None  # [lidar_rays]

@dataclass
class CommunicationComponent:
    """Ability to communicate."""
    message: Optional[jnp.ndarray] = None   # [MSG_DIM] - outgoing
    context: Optional[jnp.ndarray] = None   # [CTX_DIM] - decoded from others
    vocabulary_size: int = 32
    payload_dim: int = 4

@dataclass
class InventoryComponent:
    """Ability to carry objects."""
    carrying: int = -1              # Entity ID of carried object (-1 = none)
    max_carry_weight: float = 10.0

@dataclass
class GoalComponent:
    """Goal assigned to an agent."""
    target_position: Optional[jnp.ndarray] = None  # [2]
    target_radius: float = 15.0
    reached: bool = False

@dataclass
class BrainComponent:
    """Reference to AI model."""
    brain_id: str = "default_ppo"   # Key into BrainManager
    observation_dim: int = 36
    action_dim: int = 2

@dataclass
class PickupableComponent:
    """Object that can be picked up."""
    weight: float = 1.0
    value: float = 1.0              # For scoring
    carried_by: int = -1            # Agent ID or -1

@dataclass
class ObstacleComponent:
    """Static obstacle."""
    destructible: bool = False
    health: float = 100.0

@dataclass
class CrateComponent:
    """Pushable crate object."""
    mass: float = 5.0
    friction_static: float = 0.5
    friction_kinetic: float = 0.3
    requires_coop: int = 1      # Number of agents needed to push
    value: float = 10.0         # Score when delivered
    color: tuple = (0.6, 0.4, 0.2)  # Brown

@dataclass
class ZoneComponent:
    """Trigger zone in the environment."""
    zone_type: int = 0          # 0=DropOff, 1=Charging, 2=Hazard
    bounds: tuple = (0.0, 0.0, 100.0, 100.0)  # min_x, min_y, max_x, max_y
    effect_strength: float = 1.0
    team_id: int = -1           # -1 = all teams
    color: tuple = (0.0, 1.0, 0.0, 0.2)  # Semi-transparent green

@dataclass
class TerrainComponent:
    """Terrain properties for a region."""
    friction: float = 0.7       # 0.1=ice, 0.9=carpet
    elevation: float = 0.0      # For slopes


# ================================================================================
# 3. ENTITY REGISTRY
# ================================================================================

@dataclass
class EntityRegistry:
    """
    Central registry for entities and their components.
    Stores data as dicts for flexibility (use WorldState for production SoA).
    """
    next_entity_id: int = 0
    active_entities: Set[int] = field(default_factory=set)
    components: Dict[Type, Dict[int, Any]] = field(default_factory=dict)
    
    def create_entity(self) -> int:
        """Create a new entity and return its ID."""
        entity_id = self.next_entity_id
        self.next_entity_id += 1
        self.active_entities.add(entity_id)
        return entity_id
    
    def destroy_entity(self, entity_id: int):
        """Destroy an entity and all its components."""
        if entity_id in self.active_entities:
            self.active_entities.remove(entity_id)
            for comp_storage in self.components.values():
                comp_storage.pop(entity_id, None)
    
    def add_component(self, entity_id: int, component):
        """Add a component to an entity."""
        comp_type = type(component)
        if comp_type not in self.components:
            self.components[comp_type] = {}
        self.components[comp_type][entity_id] = component
    
    def get_component(self, entity_id: int, comp_type: Type):
        """Get a component of an entity."""
        return self.components.get(comp_type, {}).get(entity_id)
    
    def has_component(self, entity_id: int, comp_type: Type) -> bool:
        """Check if entity has a component."""
        return entity_id in self.components.get(comp_type, {})
    
    def get_entities_with(self, *comp_types: Type) -> Set[int]:
        """Return IDs of entities that have ALL specified components."""
        if not comp_types:
            return self.active_entities.copy()
        
        sets = [set(self.components.get(t, {}).keys()) for t in comp_types]
        return set.intersection(*sets) if sets else set()

# ================================================================================
# 4. FACTORY FUNCTIONS
# ================================================================================

def create_agent(registry: EntityRegistry, position: tuple, brain_id: str = "default") -> int:
    """Factory for creating a complete agent."""
    eid = registry.create_entity()
    
    registry.add_component(eid, TransformComponent(
        position=jnp.array(position),
        angle=0.0
    ))
    registry.add_component(eid, PhysicsComponent(
        velocity=jnp.zeros(2),
        angular_velocity=0.0,
        radius=10.0
    ))
    registry.add_component(eid, SensorComponent(lidar_rays=32))
    registry.add_component(eid, CommunicationComponent(
        message=jnp.zeros(36),
        context=jnp.zeros(64)
    ))
    registry.add_component(eid, InventoryComponent())
    registry.add_component(eid, BrainComponent(brain_id=brain_id))
    registry.add_component(eid, RenderComponent(color=(0, 100, 255)))
    
    return eid

def create_goal(registry: EntityRegistry, position: tuple, target_agent_id: int = -1) -> int:
    """Factory for creating a goal entity."""
    eid = registry.create_entity()
    
    registry.add_component(eid, TransformComponent(
        position=jnp.array(position),
        angle=0.0
    ))
    registry.add_component(eid, GoalComponent(
        target_position=jnp.array(position),
        target_radius=15.0
    ))
    registry.add_component(eid, RenderComponent(color=(0, 255, 0), shape="circle"))
    
    # Link goal to agent if specified
    if target_agent_id >= 0:
        agent_goal = registry.get_component(target_agent_id, GoalComponent)
        if agent_goal is None:
            registry.add_component(target_agent_id, GoalComponent(
                target_position=jnp.array(position),
                target_radius=15.0
            ))
    
    return eid

def create_resource(registry: EntityRegistry, position: tuple, value: float = 1.0) -> int:
    """Factory for creating a pickupable resource."""
    eid = registry.create_entity()
    
    registry.add_component(eid, TransformComponent(
        position=jnp.array(position),
        angle=0.0
    ))
    registry.add_component(eid, PickupableComponent(value=value))
    registry.add_component(eid, RenderComponent(color=(255, 215, 0), shape="circle"))
    
    return eid

def create_obstacle(registry: EntityRegistry, position: tuple, destructible: bool = False) -> int:
    """Factory for creating a static obstacle."""
    eid = registry.create_entity()
    
    registry.add_component(eid, TransformComponent(
        position=jnp.array(position),
        angle=0.0
    ))
    registry.add_component(eid, ObstacleComponent(destructible=destructible))
    registry.add_component(eid, RenderComponent(color=(128, 128, 128), shape="rectangle"))
    
    return eid

# ================================================================================
# 5. SYSTEMS (Logic)
# ================================================================================

class System:
    """Abstract base for ECS systems."""
    def update(self, registry: EntityRegistry, dt: float):
        raise NotImplementedError

class PhysicsSystem(System):
    """Updates positions based on velocities."""
    def update(self, registry: EntityRegistry, dt: float):
        entities = registry.get_entities_with(TransformComponent, PhysicsComponent)
        for eid in entities:
            transform = registry.get_component(eid, TransformComponent)
            physics = registry.get_component(eid, PhysicsComponent)
            
            # Euler integration
            transform.position = transform.position + physics.velocity * dt
            transform.angle = transform.angle + physics.angular_velocity * dt

class SensorSystem(System):
    """Computes lidar readings for all agents."""
    def update(self, registry: EntityRegistry, dt: float):
        # In production, this would call the vectorized compute_lidars()
        # For now, placeholder
        entities = registry.get_entities_with(TransformComponent, SensorComponent)
        for eid in entities:
            sensor = registry.get_component(eid, SensorComponent)
            if sensor.lidar_readings is None:
                sensor.lidar_readings = jnp.ones(sensor.lidar_rays)

class CommunicationSystem(System):
    """Processes message sending and receiving."""
    def update(self, registry: EntityRegistry, dt: float):
        # Collect all messages
        # Decode context for each agent
        # In production, uses TransformerContextDecoder
        entities = registry.get_entities_with(CommunicationComponent)
        messages = []
        for eid in entities:
            comm = registry.get_component(eid, CommunicationComponent)
            if comm.message is not None:
                messages.append(comm.message)
        
        # Placeholder: set context to zeros
        for eid in entities:
            comm = registry.get_component(eid, CommunicationComponent)
            comm.context = jnp.zeros(64)

class RewardSystem(System):
    """Computes rewards for RL agents."""
    def update(self, registry: EntityRegistry, dt: float):
        # Goal distance, collision penalties, communication bonuses
        # Returns dict of {entity_id: reward}
        pass

class GoalCheckSystem(System):
    """Checks if agents have reached their goals."""
    def update(self, registry: EntityRegistry, dt: float):
        entities = registry.get_entities_with(TransformComponent, GoalComponent)
        for eid in entities:
            transform = registry.get_component(eid, TransformComponent)
            goal = registry.get_component(eid, GoalComponent)
            
            if goal.target_position is not None:
                dist = jnp.linalg.norm(transform.position - goal.target_position)
                goal.reached = bool(dist < goal.target_radius)
