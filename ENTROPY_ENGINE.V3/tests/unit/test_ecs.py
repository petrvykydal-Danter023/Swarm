"""
Unit tests for ECS (Entity Component System).
"""
import pytest
import jax.numpy as jnp
from entropy.core.ecs import (
    EntityRegistry, 
    TransformComponent, 
    PhysicsComponent,
    CrateComponent,
    ZoneComponent,
    create_agent,
    create_goal
)

class TestEntityRegistry:
    def test_create_entity(self):
        """Entity creation returns incrementing IDs."""
        registry = EntityRegistry()
        eid1 = registry.create_entity()
        eid2 = registry.create_entity()
        
        assert eid1 == 0
        assert eid2 == 1
        assert eid1 in registry.active_entities
        assert eid2 in registry.active_entities
        
    def test_destroy_entity(self):
        """Destroyed entity is removed from active set."""
        registry = EntityRegistry()
        eid = registry.create_entity()
        registry.destroy_entity(eid)
        
        assert eid not in registry.active_entities
        
    def test_add_component(self):
        """Components can be added and retrieved."""
        registry = EntityRegistry()
        eid = registry.create_entity()
        
        transform = TransformComponent(position=jnp.array([10.0, 20.0]), angle=0.5)
        registry.add_component(eid, transform)
        
        retrieved = registry.get_component(eid, TransformComponent)
        assert retrieved is not None
        assert jnp.allclose(retrieved.position, jnp.array([10.0, 20.0]))
        
    def test_has_component(self):
        """has_component returns correct boolean."""
        registry = EntityRegistry()
        eid = registry.create_entity()
        
        assert not registry.has_component(eid, TransformComponent)
        
        registry.add_component(eid, TransformComponent(position=jnp.zeros(2)))
        assert registry.has_component(eid, TransformComponent)
        
    def test_get_entities_with(self):
        """Can query entities by component type."""
        registry = EntityRegistry()
        e1 = registry.create_entity()
        e2 = registry.create_entity()
        e3 = registry.create_entity()
        
        registry.add_component(e1, TransformComponent(position=jnp.zeros(2)))
        registry.add_component(e1, PhysicsComponent(velocity=jnp.zeros(2)))
        registry.add_component(e2, TransformComponent(position=jnp.zeros(2)))
        # e3 has no components
        
        with_transform = registry.get_entities_with(TransformComponent)
        assert e1 in with_transform
        assert e2 in with_transform
        assert e3 not in with_transform
        
        with_both = registry.get_entities_with(TransformComponent, PhysicsComponent)
        assert e1 in with_both
        assert e2 not in with_both

class TestComponents:
    def test_crate_component_defaults(self):
        """CrateComponent has sensible defaults."""
        crate = CrateComponent()
        assert crate.mass == 5.0
        assert crate.friction_static == 0.5
        assert crate.requires_coop == 1
        
    def test_zone_component_defaults(self):
        """ZoneComponent has sensible defaults."""
        zone = ZoneComponent()
        assert zone.zone_type == 0  # DropOff
        assert zone.team_id == -1   # All teams

class TestFactories:
    def test_create_agent(self):
        """create_agent adds all required components."""
        registry = EntityRegistry()
        agent_id = create_agent(registry, position=(100, 200))
        
        assert registry.has_component(agent_id, TransformComponent)
        assert registry.has_component(agent_id, PhysicsComponent)
        
        transform = registry.get_component(agent_id, TransformComponent)
        assert jnp.allclose(transform.position, jnp.array([100, 200]))
        
    def test_create_goal(self):
        """create_goal creates a goal entity."""
        registry = EntityRegistry()
        goal_id = create_goal(registry, position=(300, 400))
        
        assert goal_id in registry.active_entities
