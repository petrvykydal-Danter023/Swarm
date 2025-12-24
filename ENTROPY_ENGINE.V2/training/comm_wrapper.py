import gymnasium
import numpy as np
from pettingzoo.utils.wrappers import BaseWrapper
from core.communication import Vocab, StructuredMessage
from training.grounding import get_expected_message

class CommunicationRewardWrapper(BaseWrapper):
    """
    Wrapper for EntropyEnv that adds:
    1. Communication Bandwidth Regularization (Penalty for excessive noise)
    2. Logging of Vocabulary Usage to 'infos'
    """
    def __init__(self, env, silence_penalty=0.0):
        super().__init__(env)
        self.silence_penalty = silence_penalty
        self.vocab_counts = np.zeros(32, dtype=np.int32)
        
        # Communication Credit Memory
        # Stores (timestep, speaker_id, message_type)
        self.last_token_speaker = None
        self.last_token_time = 0
        self.timestep = 0
        
    def reset(self, seed=None, options=None):
        self.timestep = 0
        self.vocab_counts.fill(0)
        self.last_token_speaker = None
        self.last_token_time = 0
        return self.env.reset(seed=seed, options=options)
        
    def step(self, actions):
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self.timestep += 1
        
        # Access the inner env's communication state
        base_env = self.env.unwrapped
        if hasattr(base_env, 'comm_state'):
            comm_state = base_env.comm_state
            
            # --- LOGGING & STATE TRACKING ---
            
            # Token Channel
            if comm_state.token_holder != -1:
                t_type = comm_state.token_message.msg_type
                self.vocab_counts[t_type] += 1
                
                # Update Memory for Credit Assignment
                self.last_token_speaker = comm_state.token_holder
                self.last_token_time = self.timestep
                
            # Broadcast Channel
            for msg in comm_state.broadcast_messages:
                b_type = msg.msg_type
                if b_type != Vocab.SILENCE:
                    self.vocab_counts[b_type] += 1
            
            # Info Logging
            first_agent = self.env.possible_agents[0]
            if first_agent in infos:
                infos[first_agent]["vocab_counts"] = self.vocab_counts.copy()
            
            # --- REWARD SHAPING ---
            
            # 1. Bandwidth Regularization / Silence Incentive
            # Penalty for speaking if it's just noise? 
            # Or small bonus for SILENCE to encourage efficiency?
            # Let's apply a small penalty for non-SILENCE broadcast messages to discourage channel flooding.
            # Penalty: -0.01 per broadcast message
            
            for agent_id in self.env.agents:
                # Find this agent's broadcast message
                agent_idx = self.env.agents.index(agent_id)
                msg = comm_state.broadcast_messages[agent_idx]
                
                if msg.msg_type != Vocab.SILENCE:
                    # Apply penalty
                    rewards[agent_id] -= 0.01
                    
            # 2. Communication Credit (Assist Bonus)
            # If ANY agent gets a positive reward (e.g. goal reached), 
            # and a Token message was sent recently (e.g. within 10 steps max),
            # Give bonus to the speaker provided they aren't the one who reached the goal (or maybe even if they are?)
            # Let's say: If Goal Reached (Reward > 5.0) and Speaker existed recently.
            
            recent_threshold = 15
            if self.last_token_speaker is not None and (self.timestep - self.last_token_time) < recent_threshold:
                # Did anyone succeed?
                for agent_id, r in rewards.items():
                    if r > 5.0: # Threshold for Goal Reached (+10.0)
                        # Award the Speaker!
                        speaker_id = self.env.agents[self.last_token_speaker] # Map index to ID string if needed? 
                        # Wait, last_token_speaker is an INT index. 
                        # We need to find the Agent ID string for that index.
                        # self.env.agents is a list of strings.
                        if self.last_token_speaker < len(self.env.agents):
                            speaker_str_id = self.env.agents[self.last_token_speaker]
                            
                            # Bonus!
                            # Don't give bonus if speaker is the one who reached it? 
                            # Actually, self-talk is fine in Phase 1 (learning grounding).
                            # But specifically "Assist" implies helping others.
                            # For simplicity Phase 1: Just reward valid communication that led to success.
                            
                            rewards[speaker_str_id] += 2.0
                            if speaker_str_id in infos:
                                infos[speaker_str_id]["comm_assist_bonus"] = 2.0
                            
                            # Only award once per step to avoid stacking if multiple agents succeed same frame (unlikely)
                            break
            
            # 3. Honesty Enforcement (Reward Hacking Safeguard) & Grounding
            for agent_id in self.env.agents:
                agent_idx = self.env.agents.index(agent_id)
                msg_type = comm_state.broadcast_messages[agent_idx].msg_type
                
                agent = base_env.agents_map[agent_id]
                
                # --- A) Grounding Bonus (The "Carrot") ---
                # Check what agent *should* have said
                expected_msg = get_expected_message(agent, base_env.goals_map, [])
                
                if expected_msg != Vocab.SILENCE:
                     if msg_type == expected_msg:
                         # Bingo!
                         rewards[agent_id] += 5.0
                         if agent_id in infos:
                             infos[agent_id]["grounding_bonus"] = 5.0
                
                # --- B) Honesty Penalty (The "Stick") ---
                # Vocab.FOUND_TARGET is 7
                if msg_type == 7: 
                    # Use existing check...
                    if agent_id in base_env.goals_map: # Safer check
                         goal = base_env.goals_map[agent_id]
                         dist = agent.body.position.get_distance(goal.body.position)
                         if dist > 60: # Threshold matching grounding.py (60)
                             rewards[agent_id] -= 0.5 
                             if agent_id in infos:
                                 infos[agent_id]["liar_penalty"] = -0.5

        return obs, rewards, terminations, truncations, infos
