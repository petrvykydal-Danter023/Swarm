# 🌀 ENTROPY_ENGINE.V2

**Entropy Engine V2** is a high-performance, 2D Reinforcement Learning (RL) framework designed for training complex swarm behaviors on consumer hardware. It combines accurate physics simulation with efficient parallel training pipelines.

---

## 🏗 Architecture

The engine is built on four core pillars:

1.  **Core (`core/`)**:
    *   **Physics**: Powered by `Pymunk` (Chipmunk2D). Provides fast, stable rigid-body simulation with collision types, friction, and elasticity.
    *   **Entities**: Base classes for Agents, Walls, and Goals.
2.  **Environment (`env/`)**:
    *   **API**: `PettingZoo` compliant ParallelEnv (multi-agent standard).
    *   **Logic**: Handles resetting, stepping the physics world, sensor updates, and reward calculation.
3.  **Training (`training/`)**:
    *   **Library**: `Stable-Baselines3` (SB3).
    *   **Algorithm**: `RecurrentPPO` (PPO + LSTM). Agents have memory and "consciousness" of past states.
    *   **Vectorization**: Custom `PettingZooToVecEnv` wrapper enabling **Parameter Sharing** (one brain controls all 10+ agents).
4.  **Shared (`shared/`)**:
    *   **Visualization**: Custom `Pygame` renderer for video generation.
    *   **Logging**: `Rich` console output and `WandB` cloud integration.

---

## 🤖 The Agent

The agents are autonomous entities designed for swarm intelligence tasks.

### 🧠 Inputs (Observations)
Each agent perceives the world through a 36-dimensional vector:
*   **Lidar (32 rays):** Distance to obstacles (Walls, other Agents) in a 360° circle.
*   **Velocity (2 values):** Current linear velocity (x, y).
*   **Goal Vector (2 values):** Relative vector pointing to their assigned target.

### 🦾 Outputs (Actions)
*   **Differential Drive (2 values):** Continuous control signals [-1, 1] for Left and Right motor power.

### 🧠 Brain (LSTM)
*   Agents use a **Long Short-Term Memory (LSTM)** network.
*   This allows them to remember temporal patterns (e.g., "I was turning left a second ago", "The goal was behind that wall").

---

## 🌍 Simulation Environment

*   **World:** Continuous 2D space (800x600).
*   **Physics Step:** 60 Hz (consistent simulation stability).
*   **Goal:** Agents must navigate to their dynamically spawned green targets while avoiding collisions with walls (static) and each other (dynamic).
*   **Rewards:**
    *   `+` Moving closer to goal.
    *   `+10` Reaching goal (respawns immediately).
    *   `-` Time penalty (encourages speed).

---

## 🚀 Getting Started

### 1. Prerequisites
```bash
pip install -r ENTROPY_ENGINE.V2/requirements.txt
```

### 2. Login to Weights & Biases (Optional but Recommended)
For cloud logging and video usage:
```bash
wandb login
```

### 3. Run Training
Train a swarm of 10 agents with LSTM memory:
```bash
python ENTROPY_ENGINE.V2/training/train_lstm.py
```

### 4. Monitor
*   **Console:** Live metrics (FPS, Reward) via `Rich`.
*   **WandB:** Graphs and "Before vs After" GIFs at [wandb.ai](https://wandb.ai).
*   **Local:** Videos saved in `./videos/`.

---

## 📂 Directory Structure

```
ENTROPY_ENGINE.V2/
├── core/           # Physics & Entities (Agent, Wall)
├── env/            # PettingZoo Environment Logic
├── training/       # SB3 Scripts & Callbacks
├── shared/         # Rendering & Logging Utils
└── requirements.txt
```
