
import os
import torch
import torch.nn as nn
import numpy as np
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.vec_env import DummyVecEnv
import gymnasium
from gymnasium import spaces

# Define Root Path
V2_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -------------------------------------------------------------------
# 1. Define Dummy Environments representing V2 and V3 spaces
# -------------------------------------------------------------------

class DummyEnvV2(gymnasium.Env):
    def __init__(self):
        # V2: 36 Obs, 2 Action
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    def reset(self, **kwargs): return np.zeros(36, dtype=np.float32), {}
    def step(self, action): return np.zeros(36, dtype=np.float32), 0.0, False, False, {}

class DummyEnvV3(gymnasium.Env):
    def __init__(self):
        # V3: 38 Obs, 4 Action
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(38,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)
    def reset(self, **kwargs): return np.zeros(38, dtype=np.float32), {}
    def step(self, action): return np.zeros(38, dtype=np.float32), 0.0, False, False, {}

# -------------------------------------------------------------------
# 2. Main Surgery Logic
# -------------------------------------------------------------------

def perform_surgery():
    print("🏥 Starting Brain Surgery (V2 -> V3)...")
    
    # Paths
    v2_model_path = os.path.join(V2_ROOT, "models", "entropy_v2_1.zip")
    v3_model_path = os.path.join(V2_ROOT, "models", "entropy_v3_0.zip")
    
    if not os.path.exists(v2_model_path):
        print(f"❌ Error: Source model not found at {v2_model_path}")
        return

    # Load V2 Model
    print("Loading V2 Model (Donor)...")
    env_v2 = DummyVecEnv([lambda: DummyEnvV2()])
    model_v2 = RecurrentPPO.load(v2_model_path, env=env_v2, custom_objects={"lr_schedule": 0.0003, "clip_range": 0.2})
    state_dict_v2 = model_v2.policy.state_dict()
    
    # Init V3 Model (Recipient)
    print("Initializing V3 Model (Recipient)...")
    env_v3 = DummyVecEnv([lambda: DummyEnvV3()])
    model_v3 = RecurrentPPO("MlpLstmPolicy", env_v3, verbose=1)
    state_dict_v3 = model_v3.policy.state_dict()
    
    # Transplant Weights
    print("Transplanting Organs...")
    
    for key, v3_param in state_dict_v3.items():
        if key in state_dict_v2:
            v2_param = state_dict_v2[key]
            
            # Case 1: Identical shapes (Hidden layers)
            if v2_param.shape == v3_param.shape:
                state_dict_v3[key] = v2_param.clone()
                # print(f"  ✅ Copied exact: {key}")
                
            # Case 2: Input Layer Mismatch (36 -> 38)
            # Typically features_extractor.0.weight (2D tensor)
            elif len(v2_param.shape) >= 2 and v2_param.shape[1] == 36 and v3_param.shape[1] == 38:
                print(f"  💉 Stitching Input Layer: {key} {v2_param.shape} -> {v3_param.shape}")
                # Copy old weights to first 36 columns
                # Typically weight is (out_features, in_features)
                v3_param.data[:, :36] = v2_param.data.clone()
                # Zero init new inputs (Radio RX)
                v3_param.data[:, 36:] = 0.0
                state_dict_v3[key] = v3_param
                
            # Case 3: Output Layer Mismatch (2 -> 4)
            # Action net (policy) weight: (out_features, in_features)
            elif len(v2_param.shape) >= 2 and v2_param.shape[0] == 2 and v3_param.shape[0] == 4:
                print(f"  💉 Stitching Output Layer (Policy): {key} {v2_param.shape} -> {v3_param.shape}")
                # Copy old weights to first 2 rows
                v3_param.data[:2, :] = v2_param.data.clone()
                # Zero init new outputs (Radio TX) to ensure zero logic initially
                v3_param.data[2:, :] = 0.0
                state_dict_v3[key] = v3_param
                
            # Case 4: Output Bias Mismatch (1D tensor)
            elif len(v2_param.shape) == 1 and v2_param.shape[0] == 2 and v3_param.shape[0] == 4:
                print(f"  💉 Stitching Output Bias: {key}")
                v3_param.data[:2] = v2_param.data.clone()
                v3_param.data[2:] = 0.0
                state_dict_v3[key] = v3_param
                
            # Case 5: Value Net (Critic) input mismatch?
            # Critic takes observations too?
            # In SB3, shared feature extractor means critic head is just linear from latent.
            # Latent size doesn't change usually.
            # But if separate extractors...
            
            else:
                 print(f"  ⚠️ Shape mismatch unhandled: {key} {v2_param.shape} vs {v3_param.shape}")
        else:
            print(f"  ⚠️ Key missing in donor: {key}")

    # Load modified state dict
    model_v3.policy.load_state_dict(state_dict_v3)
    
    # Save
    print(f"Saving V3 Model to {v3_model_path}...")
    model_v3.save(v3_model_path)
    print("✅ Surgery Successful! The patient is ready for V3.")

if __name__ == "__main__":
    perform_surgery()
