"""
Full pipeline smoke test - runs all stations end-to-end.
"""
import sys
import os
import logging

sys.path.append(os.getcwd())

from factory.manager import FactoryManager
from omegaconf import OmegaConf

def main():
    logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
    logger = logging.getLogger("FullPipelineTest")
    
    logger.info("=" * 60)
    logger.info("🏭 SWARM AI FACTORY - FULL PIPELINE SMOKE TEST")
    logger.info("=" * 60)
    
    # Load config
    config = OmegaConf.create({
        "auto_retry": 2,
        "stations": {
            "oracle": {"generation": {"num_episodes": 500}}, # Increased for demo
            "kindergarten": {"target_accuracy": 0.99},
            "gym": {"target_success_rate": 0.95},
            "language_school": {"target_consistency": 0.90},
            "team_building": {"reward_threshold": 100.0},
            "war_room": {"survival_threshold": 0.95},
            "domain_randomization": {"max_generalization_gap": 0.10},
        }
    })
    
    # Initialize factory
    factory = FactoryManager(config)
    
    # Initialize Dashboard
    from entropy.dashboard.reporter import DashboardReporter
    dashboard = DashboardReporter()
    dashboard.log("Pipeline started via run_full_pipeline.py")
    
    # Run pipeline
    logger.info("\n" + "=" * 60)
    logger.info("Starting full pipeline run...")
    logger.info("=" * 60 + "\n")
    
    success = factory.run_pipeline()
    
    if success:
        logger.info("\n" + "=" * 60)
        logger.info("✅ FULL PIPELINE PASSED - Master Model Certified!")
        logger.info("=" * 60)
        print("\nDashboard is still running at http://localhost:8080")
        try:
            input("Press Enter to exit and stop the dashboard...")
        except EOFError:
            pass
        return 0
    else:
        logger.error("\n" + "=" * 60)
        logger.error("❌ PIPELINE FAILED")
        logger.error("=" * 60)
        return 1

if __name__ == "__main__":
    sys.exit(main())
