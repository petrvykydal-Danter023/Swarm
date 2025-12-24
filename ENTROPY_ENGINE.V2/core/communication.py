from dataclasses import dataclass
import numpy as np
from enum import IntEnum

# ============================================================================
# VOCABULARY DEFINITION
# ============================================================================

class Vocab(IntEnum):
    # META
    SILENCE = 0
    ACK = 1
    NACK = 2
    
    # MOVEMENT & POSITION
    HERE = 3
    GOING_TO = 4
    BLOCKED = 5
    STOP = 6
    
    # OBJECTS & GOALS
    FOUND_TARGET = 7
    LOST_TARGET = 8
    CARRYING = 9
    DROPPED = 10
    
    # COOPERATION
    NEED_HELP = 11
    COMING = 12
    FOLLOW_ME = 13
    SYNC = 14
    
    # ROLES
    CLAIM_LEADER = 15
    CLAIM_SCOUT = 16
    CLAIM_DEFENDER = 17
    CLAIM_CARRIER = 18
    ROLE_ACCEPTED = 19
    
    # TACTICS
    ATTACK = 20
    DEFEND = 21
    SURROUND = 22
    RETREAT = 23
    
    # URGENCY
    ALERT_LOW = 24
    ALERT_MED = 25
    ALERT_HIGH = 26
    ALERT_CRIT = 27
    
    # RESERVED
    RESERVED_28 = 28
    RESERVED_29 = 29
    RESERVED_30 = 30
    RESERVED_31 = 31

VOCAB_SIZE = len(Vocab)

# ============================================================================
# MESSAGE STRUCTURE
# ============================================================================

@dataclass
class StructuredMessage:
    """
    Represents a single message in the system.
    Used for both Token Channel (High-Level) and Broadcast Channel (Low-Level).
    """
    sender_id: int
    msg_type: int          # 0-31 (Vocab)
    target_agent: int      # -1 (All) or specific Agent ID
    role_claim: int        # 0-7 (Role ID if applicable)
    payload: np.ndarray    # [4] Continuous data (x, y, force, urgency)

    @staticmethod
    def empty(sender_id=-1, payload_dim=4):
        return StructuredMessage(
            sender_id=sender_id,
            msg_type=Vocab.SILENCE,
            target_agent=-1,
            role_claim=0,
            payload=np.zeros(payload_dim, dtype=np.float32)
        )

    def to_array(self):
        """
        Serialize to flat array for observation:
        [sender_id, msg_type, target_agent, role_claim, payload...]
        """
        # Note: sender_id might be relative in observation, but absolute here
        return np.concatenate([
            [self.sender_id, self.msg_type, self.target_agent, self.role_claim],
            self.payload
        ], dtype=np.float32)

class CommunicationState:
    """
    Holds the global communication state for the environment step.
    """
    def __init__(self, num_agents, payload_dim=4):
        self.num_agents = num_agents
        self.payload_dim = payload_dim
        
        # Channel A: Token (One active speaker per step)
        self.token_holder = -1
        self.token_message = StructuredMessage.empty()
        
        # Channel B: Broadcast (All agents transmit)
        self.broadcast_messages = [StructuredMessage.empty(i) for i in range(num_agents)]
        
        # Priority Tracking (for Token arbitration)
        self.agent_priorities = np.zeros(num_agents, dtype=np.float32)
        
    def reset(self):
        self.token_holder = -1
        self.token_message = StructuredMessage.empty()
        self.broadcast_messages = [StructuredMessage.empty(i) for i in range(self.num_agents)]
        self.agent_priorities.fill(0.0)
