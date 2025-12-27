import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../'))

import logging
import sys
import os

# Ensure we can import from root
sys.path.append(os.getcwd())

import jax
# Disable JIT for debugging ease and speed on small check
jax.config.update("jax_disable_jit", True)

from factory.manager import FactoryManager

def run_integration_test():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("IntegrationTest")
    
    logger.info("🚀 Starting Factory Integration Test...")
    
    shared_storage = {"path": "data/integration_test_demos/"}
    config = {
        "auto_retry": 1,
        "stations": {
            "oracle": {"storage": shared_storage, "generation": {"num_episodes": 10}}, # Small gen for speed
            "kindergarten": {"storage": shared_storage, "max_epochs": 1, "batch_size": 32},
            "gym": {"storage": shared_storage, "dagger": {"max_rounds": 1, "episodes_per_round": 2}},
            "language_school": {"storage": shared_storage, "max_epochs": 1},
            "team_building": {"storage": shared_storage, "max_epochs": 1},
            "war_room": {"storage": shared_storage, "survival_threshold": 0.0},
            "domain_randomization": {"storage": shared_storage}
        }
    }
    
    manager = FactoryManager(config)
    
    # We need an initial model?
    # S1 creates a model if None. Or Oracle provides it.
    # FactoryManager pipeline passes None initially.
    # OracleStation (S0) returns demos, but maybe not a model.
    # S1 (Kindergarten) warmup checks for demos.
    # We might need to ensure S0 loads demos or we provide a dummy model.
    # Let's see what happens.
    
    success = manager.run_pipeline()
    
    if success:
        logger.info("✅ Integration Test Passed!")
        sys.exit(0)
    else:
        logger.error("❌ Integration Test Failed!")
        sys.exit(1)

if __name__ == "__main__":
    run_integration_test()
