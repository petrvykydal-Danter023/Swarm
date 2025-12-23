# 🌀 ENTROPY ENGINE V2

> **High-performance 2D Reinforcement Learning framework for training swarm behaviors on consumer hardware.**

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![Stable-Baselines3](https://img.shields.io/badge/SB3-RecurrentPPO-green.svg)](https://sb3-contrib.readthedocs.io/)
[![WandB](https://img.shields.io/badge/Logging-WandB-orange.svg)](https://wandb.ai/)

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        ENTROPY ENGINE V2                        │
├─────────────┬─────────────┬─────────────────┬───────────────────┤
│   core/     │    env/     │    training/    │     shared/       │
│  ─────────  │  ─────────  │  ─────────────  │  ───────────────  │
│  • Physics  │  • PettingZoo│  • RecurrentPPO │  • Pygame Render  │
│  • Entities │  • Rewards   │  • Multicore    │  • Rich Logger    │
│  (Pymunk)   │  • Sensors   │  • Callbacks    │  • WandB          │
└─────────────┴─────────────┴─────────────────┴───────────────────┘
```

| Module | Description |
|--------|-------------|
| **`core/`** | Physics engine (Pymunk) + Entity classes (Agent, Wall, Goal) |
| **`env/`** | PettingZoo-compliant ParallelEnv with Lidar sensors and rewards |
| **`training/`** | SB3 training scripts, multicore wrapper, callbacks |
| **`shared/`** | Pygame renderer, Rich console logger, WandB integration |

---

## 🤖 The Agent

### Observations (36D Vector)
| Component | Dimensions | Description |
|-----------|------------|-------------|
| **Lidar** | 32 | Distance to obstacles in 360° (normalized) |
| **Velocity** | 2 | Current speed (x, y) |
| **Goal Vector** | 2 | Relative direction to target (ego-centric) |

### Actions (2D Continuous)
| Output | Range | Description |
|--------|-------|-------------|
| **Left Motor** | [-1, 1] | Power to left wheel |
| **Right Motor** | [-1, 1] | Power to right wheel |

### Brain (LSTM)
- **Architecture:** MlpLstmPolicy (256 hidden units)
- **Memory:** Agents remember past states for temporal reasoning

---

## 🚀 Training Pipeline

### Parallel Training (Multicore)
```
┌────────────────────────────────────────────────────────┐
│                    MAIN PROCESS                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │           RecurrentPPO (GPU)                    │   │
│  │           Shared Neural Network                 │   │
│  └─────────────────────────────────────────────────┘   │
│                         │                              │
│    ┌────────────────────┼────────────────────┐         │
│    ▼         ▼          ▼          ▼         ▼         │
│ ┌─────┐  ┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐        │
│ │Env 1│  │Env 2│   │Env 3│   │ ... │   │Env 8│        │
│ │10 ag│  │10 ag│   │10 ag│   │     │   │10 ag│        │
│ └─────┘  └─────┘   └─────┘   └─────┘   └─────┘        │
│   CPU      CPU       CPU       CPU       CPU           │
└────────────────────────────────────────────────────────┘
         = 80 agents training in parallel
```

### CTDE Paradigm
> **Centralized Training, Decentralized Execution**

| Phase | Behavior |
|-------|----------|
| **Training** | All 80 agents share ONE neural network (parameter sharing) |
| **Inference** | Each agent runs independently with local observations only |

---

## 📊 Training Metrics Reference

| Metric | Description | Good Values |
|--------|-------------|-------------|
| `fps` | Environment steps per second | Higher = faster training |
| `loss` | Total loss (policy + value + entropy) | Should decrease |
| `value_loss` | Critic prediction error | Should decrease |
| `explained_variance` | How well Critic understands the environment | 0→1 (higher = better) |
| `entropy_loss` | Exploration encouragement | Gradually decreases |
| `approx_kl` | Policy change magnitude | < 0.02 (PPO constraint) |
| `clip_fraction` | Updates clipped by PPO | < 0.2 |
| `std` | Action randomness | Decreases as agent becomes confident |

---

## ✅ Numba Optimalizace
*Výsledky benchmarku (100 agentů, Zrychlení: 10.3x!):*
- Bez Numby: +- 47 FPS
- S Numbou: +- 483 FPS

---

## 🛠️ Getting Started

### 1. Install Dependencies
```bash
pip install -r ENTROPY_ENGINE.V2/requirements.txt
```

### 2. (Optional) Setup WandB
```bash
wandb login
```

### 3. Run Training
```bash
# Single-core (10 agents)
python ENTROPY_ENGINE.V2/training/train_lstm.py

# Multi-core (80 agents, 8 processes)
python ENTROPY_ENGINE.V2/training/train_multicore.py
```

### 4. Monitor Progress
- **Console:** Rich progress bar + live FPS
- **WandB:** [wandb.ai/petr-vykydal/entropy-engine-v2](https://wandb.ai/petr-vykydal/entropy-engine-v2)
- **Local:** GIFs saved to `videos/`

---

## 🔄 Auto-Resume & Checkpointing

> **Crash recovery pro dlouhé tréninky na consumer hardware.**

### Jak to funguje
```
┌─────────────────────────────────────────────┐
│           START TRÉNINKU                     │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
        ┌─────────────────────┐
        │ Existuje checkpoint? │
        └─────────┬───────────┘
                  │
        ┌─────────┴─────────┐
        │                   │
        ▼                   ▼
   ✅ ANO               ❌ NE
        │                   │
        ▼                   ▼
  Načti checkpoint    Načti base model
  (crash recovery)   (prev_model)
        │                   │
        └─────────┬─────────┘
                  │
                  ▼
          Pokračuj v tréninku
                  │
                  ▼
    ┌─────────────────────────┐
    │  Trénink dokončen       │
    │  - Ulož finální model   │
    │  - Smaž checkpoint      │
    └─────────────────────────┘
```

### Klíčové body
| Vlastnost | Hodnota |
|-----------|---------|
| **Checkpoint interval** | Každých 50k kroků |
| **Lokace** | `checkpoints/checkpoint.zip` |
| **Přepisování** | Vždy se přepíše starý na nový |
| **Po dokončení** | Checkpoint se automaticky smaže |

### Obnovení po pádu
Stačí spustit trénink znovu:
```bash
python training/train_multicore.py
# 🔄 Resuming from checkpoint at 150000 steps...
```

## 📂 Directory Structure

```
ENTROPY_ENGINE.V2/
├── core/
│   ├── entities.py      # Agent, Wall, Goal classes
│   ├── physics.py       # Pymunk world wrapper
│   └── world.py         # PhysicsWorld manager
├── env/
│   └── entropy_env.py   # PettingZoo ParallelEnv
├── training/
│   ├── train_lstm.py    # Single-core training
│   ├── train_multicore.py # 8-process parallel training
│   ├── multicore_wrapper.py # AsyncVectorizedEntropyEnv
│   ├── custom_wrapper.py # PettingZoo → VecEnv adapter
│   └── callbacks.py     # GIF recording, Rich logging
├── shared/
│   ├── rendering.py     # Pygame renderer
│   └── logger.py        # Rich console logger
├── models/              # Saved .zip model checkpoints
├── checkpoints/         # Rolling checkpoint for crash recovery
├── videos/              # Generated GIFs (start/end/comparison)
├── runs/                # TensorBoard logs
├── wandb/               # WandB run metadata
└── requirements.txt
```

---

## 🔧 Configuration

Key hyperparameters in `train_multicore.py`:

```python
N_ENVS = 8              # Parallel processes
AGENTS_PER_ENV = 10     # Agents per world
total_timesteps = 1_000_000
learning_rate = 3e-4
n_steps = 512
batch_size = 4096
lstm_hidden_size = 256
```

---

## 📈 Performance

| Metric | Value |
|--------|-------|
| **Training Speed** | ~600-1600 FPS |
| **Parallel Agents** | 80 |
| **GPU Utilization** | ~25% (bottlenecked by CPU physics) |
| **Time to 1M steps** | ~25-30 minutes |

---

## 🚧 Roadmap

- [ ] Numba-accelerated Lidar raycasting
- [ ] Shared memory IPC (replace Pipe)
- [ ] Inter-agent communication channels
- [ ] JAX/Brax GPU physics migration
- [ ] Curriculum learning stages

---

<p align="center">
  <b>Built with 🧠 by the Entropy Team</b>
</p>