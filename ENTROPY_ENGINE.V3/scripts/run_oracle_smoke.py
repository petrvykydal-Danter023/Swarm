import sys
import os
import logging

# Add current dir to path to find factory package
sys.path.append(os.getcwd())

from factory.stations.s0_oracle import OracleStation
from factory.manager import FactoryManager
from omegaconf import OmegaConf

def main():
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("OracleSmokeTest")
    
    logger.info("💨 Running Oracle Factory Smoke Test...")
    
    # Load config (Mocking hydra loading)
    base_config = OmegaConf.load("configs/factory/oracle.yaml")
    
    # Override for test speed
    base_config.oracle_factory.generation.num_episodes = 5
    
    # Initialize station
    station = OracleStation(base_config.oracle_factory)
    
    # Run Lifecycle
    logger.info("1. Warmup")
    if not station.warmup():
        logger.error("Warmup failed")
        sys.exit(1)
        
    logger.info("2. Training (Generation)")
    success, result_path = station.train()
    
    if success:
        logger.info(f"✅ Success! Data generated at {result_path}")
    else:
        logger.error("❌ Generation failed")
        sys.exit(1)
        
    logger.info("3. Validation")
    if station.validate(None):
        logger.info("✅ Validation Passed")
    else:
        logger.error("❌ Validation Failed")
        sys.exit(1)

if __name__ == "__main__":
    main()
