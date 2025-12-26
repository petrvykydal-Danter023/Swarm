
"""
Entropy Engine V3 - MASTER EXPERIMENT TEMPLATE (10/10 Setup)
============================================================
Tento soubor slouží jako kompletní šablona pro nastavení experimentu.
Všechny parametry jsou zde na jednom místě s vysvětlujícími komentáři.

JAK POUŽÍT:
1. Zkopírujte tento soubor (např. 'my_experiment_01.py').
2. Upravte parametry v sekci CONFIGURATION.
3. Spusťte: `python my_experiment_01.py`.
"""
import sys
import os
sys.path.insert(0, os.getcwd())

from entropy.config import (
    ExperimentConfig, 
    SimConfig, 
    AgentConfig, 
    RewardConfig, 
    PPOConfig, 
    HogConfig, 
    RenderConfig
)
from train_master import run_experiment

# =========================================================================
# 🛠️ CONFIGURATION (NASTAVENÍ)
# =========================================================================

experiment = ExperimentConfig(
    name = "universal_test_run",  # Jméno experimentu (pro logy/files)
    total_epochs = 10000,         # Délka tréninku
    
    # 📥 RESUME (POKRAČOVÁNÍ)
    # Cesta k .pkl souboru. Pokud None, jede od nuly.
    # Např.: "outputs/universal_test_checkpoints/best.pkl"
    load_checkpoint = None,       
    
    # 🌍 SIMULACE A PROSTŘEDÍ
    sim = SimConfig(
        num_envs = 64,            # 1 = Debug (pomalé učení), 64+ = Massive (stabilní, rychlé)
        max_steps = 200,          # Délka epizody (kroky)
        arena_width = 800.0,
        arena_height = 600.0
    ),
    
    # 🤖 AGENTI
    agent = AgentConfig(
        num_agents = 20,          # Agenti v jedné aréně
        lidar_rays = 32,
        
        # --- KOMUNIKACE ---
        use_communication = False, # ✅ Zapnout/Vypnout "řeč"
        vocab_size = 4,            # Kolik slov umí (pokud zapnuto)
        context_dim = 64           # Paměť na zprávy
    ),
    
    # 🎯 ODMĚNY (Co je dobré a co špatné?)
    reward = RewardConfig(
        w_dist = 1.0,      # Motivace jít k cíli (Lineární)
        w_reach = 10.0,    # Bonus za dosažení cíle (Skoková)
        w_energy = -0.01,  # Penalizace za plýtvání palivem
        
        # --- MÓD CÍLE ---
        shared_goal = False # ✅ False = Každý má jiný cíl (Těžké)
                            # ✅ True = Všichni mají jeden cíl (Lehké/Flocking)
    ),
    
    # 🧠 TRÉNINK (PPO HYPERPARAMETRY)
    ppo = PPOConfig(
        lr_actor = 3e-4,   # Rychlost učení pohybu
        lr_critic = 1e-3,  # Rychlost učení hodnocení
        actor_updates = 4, # Kolikrát přežvýkat data
        critic_updates = 1
    ),
    
    # 👻 HAND OF GOD (EXPERTNÍ ASISTENCE)
    hog = HogConfig(
        enabled = True,        # ✅ Zapnout "pomocná kolečka"?
        start_weight = 1.0,    # 100% pomoc na začátku
        end_weight = 0.0,      # 0% pomoc na konci
        decay_epochs = 2000,   # Jak rychle pomoc zmizí (Curriculum)
        
        # --- ADAPTIVNÍ MÓD ---
        adaptive = False,      # ✅ True = "Chytrý" ústup (jen když to agentovi jde)
        target_reward = -0.1   # Cílová odměna, při které snižujeme pomoc
    ),
    
    # 🎥 VIZUALIZACE (VIDEO)
    render = RenderConfig(
        enabled = True,            # Generovat GIFy?
        render_every = 1000,       # Jak často (každých X epoch)
        output_dir = "outputs/universal_test"
    )
)

# =========================================================================
# 🚀 SPUŠTĚNÍ
# =========================================================================

if __name__ == "__main__":
    # Spustí univerzální tréninkovou smyčku s tímto nastavením
    run_experiment(experiment)
