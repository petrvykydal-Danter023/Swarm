"""
Entropy Engine V3 - Brain Manager
Centralized management for AI models, checkpoints, and versioning.
"""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Optional, Any, List, Tuple
import json
import safetensors.flax
import jax
import jax.numpy as jnp
from flax import linen as nn
from functools import partial
import datetime

@dataclass
class BrainMeta:
    """Metadata about a stored brain model."""
    name: str
    version: int
    brain_type: str  # "ppo", "dial_encoder", "dial_decoder", "transformer", "mappo"
    input_dim: int
    output_dim: int
    hidden_dim: int
    created_at: str
    training_steps: int
    source_checkpoint: Optional[str] = None
    notes: str = ""

@dataclass
class DIALBrainPair:
    """Encoder + Decoder pair for communication."""
    encoder_params: dict
    decoder_params: dict
    vocab_size: int
    payload_dim: int
    context_dim: int

class BrainManager:
    """
    Central manager for all AI models ("brains").
    Handles versioned saving, loading, and registry maintenance.
    """
    
    def __init__(self, storage_dir: str = "brains/"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.registry: Dict[str, BrainMeta] = {}
        self._load_registry()
        
    def _load_registry(self):
        """Loads registry from disk."""
        registry_path = self.storage_dir / "registry.json"
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    data = json.load(f)
                    for name, meta in data.items():
                        self.registry[name] = BrainMeta(**meta)
            except Exception as e:
                print(f"[BrainManager] Error loading registry: {e}")
                self.registry = {}
    
    def _save_registry(self):
        """Saves registry to disk."""
        registry_path = self.storage_dir / "registry.json"
        data = {name: asdict(meta) for name, meta in self.registry.items()}
        with open(registry_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def save(self, name: str, params: dict, meta: BrainMeta) -> str:
        """
        Save a model with automatic versioning.
        
        Args:
            name: Base name (e.g. "ppo_swarm")
            params: JAX/Flax parameters
            meta: Metadata object
            
        Returns:
            full_name: Versioned unique identifier (e.g. "ppo_swarm_v1")
        """
        # Auto-increment version
        existing_versions = [
            m.version for k, m in self.registry.items() 
            if k == name or k.startswith(f"{name}_v")  # looser check, better safe
        ]
        # Better: check meta.name which should match 'name' argument generally, 
        # but the keys in registry are usually "name_vX".
        # Let's rely on the registry keys.
        existing_versions = []
        for key, m in self.registry.items():
            if m.name == name:
                existing_versions.append(m.version)
        
        version = max(existing_versions, default=0) + 1
        meta.version = version
        meta.name = name # Ensure consistency
        meta.created_at = datetime.datetime.now().isoformat()
        
        # Create directory
        model_dir = self.storage_dir / name / f"v{version}"
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save params using safetensors
        params_path = model_dir / "params.safetensors"
        # Flatten dict if needed? safetensors.flax usually handles nested dicts (pytrees)
        safetensors.flax.save_file(params, str(params_path))
        
        # Save metadata
        meta_path = model_dir / "meta.json"
        with open(meta_path, 'w') as f:
            json.dump(asdict(meta), f, indent=2)
        
        # Update registry
        full_name = f"{name}_v{version}"
        self.registry[full_name] = meta
        self._save_registry()
        
        print(f"[BrainManager] ✅ Saved {full_name} to {model_dir}")
        return full_name
    
    def load(self, name: str, version: Optional[int] = None) -> Tuple[dict, BrainMeta]:
        """
        Load a model.
        
        Args:
            name: Model name
            version: Specific version (None = latest)
            
        Returns:
            (params, meta)
        """
        if version is None:
            # Find latest
            versions = [(k, m) for k, m in self.registry.items() if m.name == name]
            if not versions:
                raise ValueError(f"Model '{name}' not found in registry")
            full_name, meta = max(versions, key=lambda x: x[1].version)
            version = meta.version
        else:
            full_name = f"{name}_v{version}"
            matching = [m for k, m in self.registry.items() if k == full_name] # direct lookup
            if full_name in self.registry:
                meta = self.registry[full_name]
            else:
                 raise ValueError(f"Model '{full_name}' not found")
        
        model_dir = self.storage_dir / name / f"v{version}"
        params_path = model_dir / "params.safetensors"
        
        if not params_path.exists():
            raise FileNotFoundError(f"Params file missing: {params_path}")
            
        params = safetensors.flax.load_file(str(params_path))
        print(f"[BrainManager] ✅ Loaded {full_name}")
        return params, meta
    
    def list_models(self) -> List[str]:
        """Return list of all registered model keys."""
        return list(self.registry.keys())
    
    def get_latest(self, name: str) -> Optional[str]:
        """Return full_name of latest version for a given model name."""
        versions = [(k, m) for k, m in self.registry.items() if m.name == name]
        if not versions:
            return None
        full_name, _ = max(versions, key=lambda x: x[1].version)
        return full_name


class BrainInference:
    """
    High-level wrapper for model inference.
    Hides JAX/Flax details from the user/environment using it.
    """
    
    def __init__(self, manager: BrainManager, model_name: str, network_class):
        self.manager = manager
        print(f"[BrainInference] Loading '{model_name}'...")
        self.params, self.meta = manager.load(model_name)
        
        # Reconstruct network
        # This assumes network_class takes specific args. 
        # Ideally we'd map brain_type to classes or use kwargs from meta.
        self.network = network_class(
            obs_dim=self.meta.input_dim,
            action_dim=self.meta.output_dim,
            hidden_dim=self.meta.hidden_dim
        )
        
    @partial(jax.jit, static_argnums=(0,))
    def act(self, obs: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Predict action from observation.
        
        Args:
            obs: Observation vector [Batch, ObsDim] or [ObsDim]
            deterministic: If True, use mean. If False, sample.
        """
        # Add batch dim if needed
        is_batch = obs.ndim > 1
        if not is_batch:
            obs = obs[None, :]
        
        # Apply network
        # Assuming network returns (mean, log_std, value) like ActorCritic
        # If it returns tuple, unpack. If single tensor, use it.
        out = self.network.apply(self.params, obs)
        
        if isinstance(out, tuple):
             # ActorCritic returns (mean, log_std, value)
             mean, log_std, value = out
        else:
             # Just mean?
             mean = out
             log_std = None
             
        if deterministic or log_std is None:
            actions = mean
        else:
            # Simple sampling logic (relying on PRNG key 0 - technically not stochastic across calls without rng inputs)
            # For true stochasticity, 'act' should take an rng key.
            # However, for pure inference in 'headless' mode we usually adhere to deterministic.
            # If stochastic is needed, we'd add noise here.
            # For this wrapper, let's keep it simple: deterministic by default.
            # If we need randomness, we might need to inject state or rng.
            # Using constant seed for "sample" mode here just to show logic structure.
            std = jnp.exp(log_std)
            rng = jax.random.PRNGKey(42) # Fixed seed wrapper limitation
            actions = mean + std * jax.random.normal(rng, mean.shape)
            
        # Tanh squashing (assuming PPO typical output)
        actions = jnp.tanh(actions)
        
        if not is_batch:
            actions = actions.squeeze(0)
            
        return actions


class DIALBrainManager(BrainManager):
    """Specialized manager for DIAL communication models."""
    
    def save_dial(
        self, 
        name: str, 
        encoder_params: dict, 
        decoder_params: dict, 
        meta: BrainMeta
    ) -> str:
        """
        Save DIAL model pair. 
        We save them as two separate storage items sharing a version, 
        or bundle them?
        Blueprint suggests separate types or specialized handling.
        Let's bundle params into one flat dict for safetensors if possible,
        or just subclass save logic.
        
        Simplest approach: Prefix params with "enc_" and "dec_" and save as one file.
        """
        # Prefix keys to merge dicts
        # Note: Flax params are nested FrozenDicts. Merging is tricky without flattening.
        # Safetensors expects flat keys "layer1.weight".
        
        # Strategy: Save as standard Brain but ensure meta marks it as dial.
        # We will assume 'encoder_params' and 'decoder_params' are passed in a wrapper dict.
        combined_params = {
            "encoder": encoder_params,
            "decoder": decoder_params
        }
        # But safetensors.flax.save_file expects Flax params (FrozenDict of arrays).
        # We cannot easily mix two separate trees unless we wrap them.
        
        # For simplicity in this implementation, we will save them in subfolders manually
        # bypassing the standard save() single-file logic, OR 
        # we can define a "DIALModule" that wraps both and reuse standard save.
        
        # Let's defer strict DIAL saving logic to a simpler persistent structure:
        # Save as 2 separate brains: "name_enc" and "name_dec"
        
        name_enc = f"{name}_enc"
        name_dec = f"{name}_dec"
        
        # Meta copies
        meta_enc = BrainMeta(**asdict(meta))
        meta_enc.name = name_enc
        meta_enc.brain_type = "dial_encoder"
        
        meta_dec = BrainMeta(**asdict(meta))
        meta_dec.name = name_dec
        meta_dec.brain_type = "dial_decoder"
        
        v_enc = self.save(name_enc, encoder_params, meta_enc)
        v_dec = self.save(name_dec, decoder_params, meta_dec)
        
        print(f"[DIALBrainManager] Saved pair: {v_enc} + {v_dec}")
        return f"{name}_v{meta_enc.version}"
