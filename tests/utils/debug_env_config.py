
import json
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from topdown_engine import TopDownSwarmEnv

def check_env_config():
    config_path = "examples/safety_training.json"
    if not os.path.exists(config_path):
        print(f"ERROR: {config_path} not found.")
        return

    with open(config_path, "r") as f:
        config = json.load(f)
        
    print(f"Loaded config: {json.dumps(config, indent=2)}")
    
    print("\n--- Initializing Environment ---")
    env = TopDownSwarmEnv(config)
    
    print(f"Env Num Agents: {env.num_agents}")
    print(f"Env Obstacles Count: {len(env.obstacles)}")
    print(f"Env Payloads Count: {len(env.payloads)}")
    print(f"Env Width: {env.width}")
    
    expected_agents = config["env_params"]["num_agents"]
    # expected_obstacles = len(config["env_params"]["special_objects"]) - 1 # Goal is not obstacle type in internal list? No, type is stored.
    # Actually build_world filters?
    # No, _build_world iterates special_objects.
    # If type == payload -> payload. Else -> obstacle.
    # Goal is type="goal", so it goes to self.obstacles.
    expected_obstacles = len(config["env_params"]["special_objects"])
    
    if env.num_agents != expected_agents:
        print(f"❌ FAIL: Expected {expected_agents} agents, got {env.num_agents}")
    else:
        print(f"✅ Agents Count OK")
        
    if len(env.obstacles) != expected_obstacles:
        print(f"❌ FAIL: Expected {expected_obstacles} obstacles (incl goals), got {len(env.obstacles)}")
    else:
        print(f"✅ Obstacles Count OK")

if __name__ == "__main__":
    check_env_config()
