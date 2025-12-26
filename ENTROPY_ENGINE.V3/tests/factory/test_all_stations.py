"""
Comprehensive tests for all Factory stations.
"""
import unittest
import logging

# Suppress logging during tests
logging.disable(logging.CRITICAL)

from factory.stations.s0_oracle import OracleStation
from factory.stations.s1_kindergarten import KindergartenStation
from factory.stations.s2_gym import GymStation, DAggerTrainer
from factory.stations.s3_language_school import LanguageSchoolStation
from factory.stations.s4_team_building import TeamBuildingStation, OptimizedMAPPO
from factory.stations.s5_war_room import WarRoomStation
from factory.stations.s55_domain_randomization import DomainRandomizationStation, DomainRandomizer
from factory.analysis.skepticism import AdaptiveSkepticismDial
from factory.training.utils import PrioritizedDemoSampler, coverage_aware_bc_loss
from factory.manager import FactoryManager


class TestStation0Oracle(unittest.TestCase):
    def setUp(self):
        self.station = OracleStation({})

    def test_warmup(self):
        self.assertTrue(self.station.warmup())

    def test_train_returns_path(self):
        success, path = self.station.train()
        self.assertTrue(success)
        self.assertIn("oracle_demos", path)

    def test_validate(self):
        self.assertTrue(self.station.validate(None))


class TestStation1Kindergarten(unittest.TestCase):
    def setUp(self):
        self.station = KindergartenStation({"target_accuracy": 0.99, "batch_size": 2})
        
        # Mock storage to return dummy data for real training logic tests
        from unittest.mock import MagicMock
        self.station.storage = MagicMock()
        
        dummy_traj = {
            "steps": [
                {
                    "observation": {
                        "agent_position": [0.0, 0.0],
                        "goal_position": [1.0, 1.0],
                        "agent_angle": 0.0
                    },
                    "action": {"motor": [0.5, 0.1]}
                } for _ in range(10)
            ]
        }
        self.station.storage.load_trajectory.return_value = [dummy_traj]

    def test_lifecycle(self):
        self.assertTrue(self.station.warmup())
        success, model = self.station.train()
        self.assertTrue(success)
        self.assertTrue(self.station.validate(model))


class TestStation2Gym(unittest.TestCase):
    def setUp(self):
        # Disable JIT for tests involving Mocks
        import jax
        jax.config.update("jax_disable_jit", True)
        
        self.station = GymStation({"target_success_rate": 0.95})
        
        # Patch dependencies for DAggerTrainer
        from unittest.mock import MagicMock, patch
        
        # We need to patch where DAggerTrainer imports them, OR patch globally
        # Since I can't patch inside the function easily without context manager in test method
        # I will do it in test_dagger_trainer
        pass

    def test_dagger_trainer(self):
        trainer = DAggerTrainer({"max_rounds": 1, "episodes_per_round": 1, "gradient_steps": 1})
        
        # Mock dependencies
        with unittest.mock.patch('factory.stations.s2_gym.EntropyGymWrapper') as MockEnv:
            # Mock Env instance
            mock_env = MockEnv.return_value
            # reset returns (state, obs). Obs shape [NumAgents, ObsDim]
            mock_env.reset.return_value = (None, [[0.1, 0.1]*32]*10) # 10 agents
            # step returns (state, obs, reward, done, info)
            mock_env.step.return_value = (None, [[0.1, 0.1]*32]*10, [0]*10, [True]*10, {})
            
            with unittest.mock.patch('factory.stations.s2_gym.KindergartenStation') as MockS1:
               # Mock S1 helper
               mock_s1 = MockS1.return_value
               mock_s1._process_data.return_value = ([], []) # Empty init data
               
               # Create Dummy Model (TrainState)
               # TrainState needs apply_fn and params
               model = unittest.mock.MagicMock()
               
               # Mock apply_fn to return correct shapes
               def mock_apply(params, obs):
                   # obs shape (N, D) or list
                   import numpy as np
                   # Check if list or array
                   if isinstance(obs, list):
                       N = len(obs)
                   else:
                       N = obs.shape[0]
                   return np.zeros((N, 2)), None, None
               
               model.apply_fn.side_effect = mock_apply
               model.params = {}
               
               result = trainer.train(model, None, [])
               self.assertIsNotNone(result)

    def test_lifecycle(self):
        # Mock DAggerTrainer.train to avoid complex logic
        self.station.dagger.train = unittest.mock.MagicMock(return_value="TrainedModel")
        
        self.assertTrue(self.station.warmup())
        success, model = self.station.train()
        self.assertTrue(success)
        self.assertEqual(model, "TrainedModel")


class TestStation3LanguageSchool(unittest.TestCase):
    def setUp(self):
        import jax
        jax.config.update("jax_disable_jit", True)
        # We instantiate station inside test to ensure mocks apply if needed
        # or we patch attribute.
        pass

    def test_lifecycle(self):
        # Mock load_trajectory
        with unittest.mock.patch('factory.stations.s3_language_school.DemoStorage') as MockStorage:
            ms = MockStorage.return_value
            # S3 expects trajectories to process for consistency
            dummy_traj = {
                "steps": [
                    {
                        "observation": {
                            "agent_position": [0.0, 0.0],
                            "goal_position": [1.0, 1.0],
                            "agent_angle": 0.0
                        }
                    }for _ in range(5)
                ]
            }
            ms.load_trajectory.return_value = [dummy_traj]
            
            # Instantiate here so it uses the MockStorage class
            station = LanguageSchoolStation({"target_consistency": 0.90})
            
            # S3 warmsup
            self.assertTrue(station.warmup(model="S2Matches"))
            
            # Train needs S2 model (TrainState). Use REAL model to avoid JAX/Mock issues.
            from entropy.training.network import ActorCritic
            from flax.training.train_state import TrainState
            import optax
            import jax
            import jax.numpy as jnp
            
            key = jax.random.PRNGKey(0)
            # action_dim should cover motor(2) + comm(N)
            # S3 assumes slicing [:2] and [2:]. Let's use 4 dims.
            net = ActorCritic(action_dim=4) 
            obs_dummy = jnp.zeros((1, 4)) # S3 processing produces 4-dim obs
            params = net.init(key, obs_dummy)
            
            real_state = TrainState.create(
                apply_fn=net.apply,
                params=params,
                tx=optax.adam(1e-3)
            )
            
            # Run train
            success, new_model = station.train(model=real_state)
            self.assertTrue(success)
            self.assertTrue(station.validate(new_model))


class TestStation4TeamBuilding(unittest.TestCase):
    def setUp(self):
        import jax
        jax.config.update("jax_disable_jit", True)
        self.station = TeamBuildingStation({"reward_threshold": 100.0, "mappo": {"actor_updates_per_step": 1}})

    def test_mappo_update(self):
        mappo = OptimizedMAPPO({})
        actor_loss, critic_loss = mappo.update(None)
        self.assertIsInstance(actor_loss, float)
        self.assertIsInstance(critic_loss, float)

    def test_lifecycle(self):
        # Mock wrapper
        with unittest.mock.patch('factory.stations.s4_team_building.EntropyGymWrapper') as MockWrapper:
             mw = MockWrapper.return_value
             import numpy as np
             # reset -> state, obs [5, 64]
             mw.reset.return_value = (None, np.zeros((5, 64)))
             # step -> state, obs, reward[5], done[5], info
             mw.step.return_value = (None, np.zeros((5, 64)), np.ones(5), np.ones(5, dtype=bool), {})
             
             self.assertTrue(self.station.warmup())
             success, model = self.station.train()
             self.assertTrue(success)


class TestStation5WarRoom(unittest.TestCase):
    def setUp(self):
        import jax
        jax.config.update("jax_disable_jit", True)
        self.station = WarRoomStation({"survival_threshold": 0.95})

    def test_lifecycle(self):
        # Mock wrapper
        with unittest.mock.patch('factory.stations.s5_war_room.EntropyGymWrapper') as MockWrapper:
             mw = MockWrapper.return_value
             import numpy as np
             mw.reset.return_value = (None, np.zeros((5, 64)))
             mw.step.return_value = (None, np.zeros((5, 64)), np.ones(5), np.ones(5, dtype=bool), {})
             
             self.assertTrue(self.station.warmup())
             # S5 train expects model optionally
             success, model = self.station.train()
             self.assertTrue(success)
        self.assertTrue(self.station.validate(model))


class TestStation55DomainRandomization(unittest.TestCase):
    def setUp(self):
        self.station = DomainRandomizationStation({"max_generalization_gap": 0.10})

    def test_randomizer(self):
        from factory.stations.s55_domain_randomization import DomainRandomizationConfig
        config = DomainRandomizationConfig()
        randomizer = DomainRandomizer(config)
        # Should not raise
        randomizer.randomize_physics(None, None)
        obs = randomizer.randomize_sensors({}, None)
        self.assertEqual(obs, {})

    def test_lifecycle(self):
        self.assertTrue(self.station.warmup())
        success, model = self.station.train()
        self.assertTrue(success)


class TestAdaptiveSkepticism(unittest.TestCase):
    def test_increase_on_success(self):
        dial = AdaptiveSkepticismDial({
            "start_rate": 0.0, "max_rate": 0.20,
            "success_threshold": 0.95, "collision_threshold": 0.02
        })
        new_rate = dial.update({"success_rate": 0.98, "collision_rate": 0.01})
        self.assertGreater(new_rate, 0.0)

    def test_decrease_on_crisis(self):
        dial = AdaptiveSkepticismDial({
            "start_rate": 0.10, "min_rate": 0.0,
            "crisis_threshold": 0.70
        })
        new_rate = dial.update({"success_rate": 0.50, "collision_rate": 0.5})
        self.assertLess(new_rate, 0.10)


class TestTrainingUtils(unittest.TestCase):
    def test_prioritized_sampler(self):
        sampler = PrioritizedDemoSampler(alpha=0.6)
        demos = [{"obs": i} for i in range(100)]
        batch = sampler.sample(demos, None, batch_size=10)
        self.assertEqual(len(batch), 10)

    def test_coverage_bc_loss(self):
        loss = coverage_aware_bc_loss(None, None)
        self.assertIsInstance(loss, float)


class TestFactoryManager(unittest.TestCase):
    def test_manager_loads_all_stations(self):
        manager = FactoryManager({"stations": {}})
        self.assertEqual(len(manager.stations), 7)
        
    def test_station_names(self):
        manager = FactoryManager({"stations": {}})
        names = [s.name for s in manager.stations]
        self.assertIn("OracleStation", names)
        self.assertIn("KindergartenStation", names)
        self.assertIn("DomainRandomizationStation", names)


if __name__ == '__main__':
    unittest.main()
