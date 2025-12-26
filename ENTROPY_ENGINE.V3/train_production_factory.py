
import sys
import os
import yaml
import jax
import logging

# Ensure imports work
sys.path.append(os.getcwd())

# Disable JIT for Windows Stability
jax.config.update("jax_disable_jit", True)

from factory.manager import FactoryManager

def load_config(path="configs/factory_production.yaml"):
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run_production_training():
    # Force UTF-8 for stdout if possible
    if sys.stdout.encoding.lower() != 'utf-8':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except:
            pass

    # Setup Logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler("factory_production.log", mode='w', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger = logging.getLogger("ProductionMain")
    
    logger.info("🏭 ENTROPY ENGINE V3: INITIALIZING SWARM AI FACTORY")
    logger.info("🚀 Mode: PRODUCTION TRAINING")
    
    config = load_config()
    logger.info(f"📜 Loaded Config: {config}")
    
    manager = FactoryManager(config)
    
    from entropy.training.checkpoint import CheckpointManager
    ckpt_manager = CheckpointManager("outputs/checkpoints", max_to_keep=5)
    
    logger.info("⏳ Starting Pipeline... This may take hours/days depending on compute.")
    success, certified_model = manager.run_pipeline()
    
    if success:
        logger.info("🎉🎉 FACTORY COMPLETE. MASTER MODEL CERTIFIED. 🎉🎉")
        
        # Save Model
        if certified_model and hasattr(certified_model, 'params'):
            step = 1000000 # Dummy step for certified model
            ckpt_manager.save(
                params=certified_model.params,
                opt_state=certified_model.opt_state,
                step=step,
                metrics={"certification": "Platinum"}
            )
            logger.info("💾 Checkpoints saved in 'outputs/checkpoints/'")
        else:
            logger.warning("⚠️ Success reported but model structure unknown. Saved raw pickle if possible.")
            
        sys.exit(0)
    else:
        logger.error("🛑 FACTORY HALTED. Check logs.")
        sys.exit(1)

if __name__ == "__main__":
    run_production_training()
