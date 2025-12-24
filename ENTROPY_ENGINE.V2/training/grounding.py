import numpy as np
from core.communication import Vocab
from core.entities import Agent, Goal

# ============================================================================
# GROUNDING DEFINITION (The "Truth")
# ============================================================================

def get_expected_message(agent: Agent, goals_map: dict, nearby_agents: list) -> int:
    """
    Determines what the agent SHOULD say based on its current state (Ground Truth).
    Returns a Vocab ID or Vocab.SILENCE if no specific message is enforced.
    """
    
    # 1. PRIORITY: Goal Reached / Nearby
    # If very close to own goal -> FOUND_TARGET (7)
    
    min_dist = float('inf')
    closest_goal = None
    
    for goal in goals_map.values():
        dist = np.linalg.norm(agent.position - goal.body.position)
        if dist < min_dist:
            min_dist = dist
            closest_goal = goal
            
    if min_dist < 60.0: # Close enough to "see" or "touch"
        return Vocab.FOUND_TARGET
        
    # 2. PRIORITY: Danger (Obstacles/Walls)
    # Future work
    
    # 3. PRIORITY: Help (Stuck?)
    # Future work
    
    # 4. DEFAULT: Silence
    return Vocab.SILENCE
