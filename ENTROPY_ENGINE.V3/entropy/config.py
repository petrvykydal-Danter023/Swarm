
"""
Entropy Engine V3 - Universal Configuration System
Defines the hierarchy of configuration objects for the entire engine.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class SimConfig:
    """
    Nastavení fyzikálního světa a simulace.
    """
    # Rozměry arény (jednotky)
    arena_width: float = 800.0  
    arena_height: float = 600.0
    
    # Délka jedné epizody (počet kroků)
    # 200 kroků @ 60 FPS = 3.3 sekundy reálného času simulace (ne tréninku)
    max_steps: int = 200  
    
    # Počet paralelních prostředí pro trénink (Batch Size)
    # 1 = Single Debug (train_fast_swarm)
    # 64+ = Massive Parallel (train_massive_swarm) - Doporučeno pro stabilitu
    num_envs: int = 64  
    
    # Fyzikální parametry (tření, kolize...) lze přidat zde
    dt: float = 0.1

@dataclass
class CommConfig:
    """
    Pokročilá komunikace (Coordinated Unit).
    """
    mode: str = "spatial"  # "spatial" nebo "broadcast" (legacy)
    msg_dim: int = 16      # Velikost zprávy (vektor)
    
    # --- Spatial & Attention ---
    max_neighbors: int = 5 # Top-K sousedů pro Attention/Inbox
    
    # --- Dual-Channel Attention ---
    dual_attention: bool = True     # Use Local/Global split
    local_radius: float = 300.0     # Radius for "Local" (Tactical) messages
    local_heads: int = 2
    global_heads: int = 2
    
    # --- Event-Triggered Communication ---
    surprise_gating: bool = True    # Use Surprise to open Gate
    surprise_threshold: float = 0.1 # Min prediction error to allow speaking
    info_gain_reward: float = 0.1   # Reward for useful messages
    
    # --- Virtual Pheromones ---
    pheromones_enabled: bool = True # Enable Stigmergy
    pheromone_radius: float = 50.0  # Detection/Influence range
    pheromone_ttl: int = 100        # Time-to-live in steps
    max_pheromones: int = 100       # Buffer size (max active markers)
    pheromone_dim: int = 8          # Dimension of pheromone message
    
    # --- Dynamic Hierarchy ---
    hierarchy_enabled: bool = True     # Use Squads & Leaders
    squad_size: int = 5                # Target agents per squad
    leader_election_mode: str = "proximity" # "proximity", "random"
    leader_broadcast_only: bool = True # Restrict broadcast to leaders
    
    # --- Gating & Penalty ---
    gating_threshold: float = 0.5  # Sigmoid > 0.5 => Speak
    spam_penalty: float = -0.01    # Adaptivní penalizace za mluvení
    comm_warmup_epochs: int = 1000 # Epochy zdarma (bez penalizace)

@dataclass
class AgentConfig:
    """
    Nastavení agentů a jejich schopností.
    """
    # Počet agentů v jedné aréně
    num_agents: int = 20
    
    # --- Senzory ---
    lidar_rays: int = 32
    lidar_range: float = 200.0
    
    # --- Komunikace ---
    # Pokud use_communication=True, použije se CommConfig níže
    use_communication: bool = False
    vocab_size: int = 4 # Legacy (pro zpětnou kompatibilitu, pokud mode!=spatial)
    context_dim: int = 64 # Legacy
    
    # Nová konfigurace pro Spatial comms
    comm: CommConfig = field(default_factory=CommConfig)

@dataclass
class RewardConfig:
    """
    Nastavení Váh Odměn (Reward Shaping).
    Určuje, co je pro agenty "dobré" a "špatné".
    """
    # 1. Vzdálenost k cíli (Dense Reward)
    # Motivuje k pohybu směrem k cíli.
    # Negativní hodnota = penalizace za vzdálenost (chce být blízko = 0)
    w_dist: float = 1.0  
    
    # 2. Dosažení cíle (Sparse Reward)
    # Velký bonus za dotknutí se cíle.
    w_reach: float = 10.0
    
    # 3. Penalizace za energii (Motor usage)
    # Motivuje k efektivnímu pohybu (neplýtvat palivem).
    # Záporná hodnota.
    w_energy: float = -0.01 
    
    # 4. Living Penalty (Time pressure)
    # Malá penalizace za každý krok, kdy agent NENÍ v cíli.
    # Zabraňuje strategii "stůj a šetři energii".
    w_living_penalty: float = -0.001
    
    # 5. Sdílení cíle (Shared Goal)
    # True = Všichni agenti mají jeden společný cíl (Flocking).
    # False = Každý má svůj unikátní cíl (Routing/Traffic).
    shared_goal: bool = False

@dataclass
class PPOConfig:
    """
    Hyperparametry pro MAPPO (Multi-Agent PPO).
    """
    # Learning Rate pro Actora (Pohyb/Mluvení)
    lr_actor: float = 3e-4  
    
    # Learning Rate pro Critica (Odhad hodnoty)
    # Obvykle vyšší než actor, aby se rychleji stabilizoval.
    lr_critic: float = 1e-3 
    
    # Počet aktualizací sítě na jeden batch dat
    actor_updates: int = 4
    critic_updates: int = 1
    
    # Clip Range pro PPO (jak moc se může změnit strategie v jednom kroku)
    clip_eps: float = 0.2
    
    # Gamma (Discount Factor) - jak moc záleží na budoucnosti
    gamma: float = 0.99

@dataclass
class HogConfig:
    """
    Hand of God (HOG) - Expertní Asistence
    """
    enabled: bool = True
    start_weight: float = 1.0 
    end_weight: float = 0.0
    decay_epochs: int = 5000 
    
    # Adaptivní mód
    # Pokud True, decay_epochs se ignoruje a pomoc klesá, 
    # jen když agent dosáhne target_reward.
    adaptive: bool = False
    target_reward: float = -0.1 # Nula je perfektní (být na cíli)

@dataclass
class RenderConfig:
    """
    Nastavení vizualizace a ukládání.
    """
    enabled: bool = True
    render_every: int = 1000
    fps: int = 20
    output_dir: str = "outputs/universal_experiment"

@dataclass(unsafe_hash=True)
class IntentConfig:
    """
    Konfigurace Intent-Based Actions (Phase 2).
    """
    enabled: bool = False             # Pokud False, používá se Direct Action (Motor L/R)
    
    # PID parametry pro převod Target -> Motor
    pid_pos_kp: float = 2.0
    pid_pos_kd: float = 0.5
    pid_rot_kp: float = 5.0
    pid_rot_kd: float = 0.5
    
    # Limity
    max_linear_accel: float = 5.0
    max_angular_accel: float = 10.0

@dataclass(unsafe_hash=True)
class SafetyConfig:
    """
    Konfigurace Safety Layer (Reflexy).
    """
    enabled: bool = True
    
    # === Collision Avoidance ===
    safety_radius: float = 30.0        # Start slowing down at this distance
    min_distance: float = 10.0         # Hard stop distance
    collision_check_radius: float = 60.0 # Only check agents within this radius (scaling)
    
    # === Repulsion (Liquid Swarm) ===
    enable_repulsion: bool = True
    repulsion_radius: float = 25.0     # Start repelling at this distance
    repulsion_force: float = 0.5       # Strength of push
    
    # === Speed Limits ===
    max_speed: float = 10.0            # Absolute max velocity
    emergency_brake_dist: float = 5.0  # Hard brake distance
    
    # === Communication Limits ===
    msg_rate_limit: int = 5            # Max messages per N steps
    msg_rate_window: int = 10          # Window size in steps
    
    # === Energy Management ===
    energy_enabled: bool = False  # Toggle
    low_battery_threshold: float = 0.2     # 20% - reduce speed
    critical_battery_threshold: float = 0.05  # 5% - force return
    low_battery_speed_mult: float = 0.5
    
    # === Watchdog (Anti-Stalemate) ===
    watchdog_enabled: bool = True  # Toggle
    stalemate_window: int = 100        # Check every N steps
    stalemate_min_distance: float = 5.0  # Must move at least this far
    stalemate_random_duration: int = 20  # Random walk duration
    stalemate_random_speed: float = 0.5
    
    # === Geo-Fence ===
    geofence_enabled: bool = True  # Toggle
    geofence_push_distance: float = 30.0
    geofence_push_force: float = 1.0
    
    # === Override ===
    allow_ai_override: bool = True     # Can AI disable reflexes?
    
    # === Metrics ===
    log_metrics: bool = True
    log_interval: int = 100   # Log every N steps

@dataclass
class ExperimentConfig:
    """
    MASTER CONFIG - Kořenový objekt pro celý experiment.
    """
    name: str = "default_experiment"
    total_epochs: int = 50000 
    
    # Resume / Transfer Learning
    # Cesta k .pkl souboru s checkpointem. Pokud None, jede od nuly.
    load_checkpoint: Optional[str] = None
    
    sim: SimConfig = field(default_factory=SimConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    hog: HogConfig = field(default_factory=HogConfig)
    render: RenderConfig = field(default_factory=RenderConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    intent: IntentConfig = field(default_factory=IntentConfig)


# =============================================================================
# 📦 ENTROPY ENGINE V3 - INVENTORY & CAPABILITIES
# =============================================================================
# Zde je seznam všeho, co je simulovatelné a podporované v aktuální verzi Enginu.
#
# 1. OBJEKTY A ENTITY
#    - [x] Agent (Circle): Má pozici, rotaci, rychlost, barvu (dle týmu/stavu).
#    - [x] Cíl (Target): Bod v prostoru (kruh), statický nebo dynamický (pohyblivý).
#    - [x] Překážky (Obstacles):
#          - [x] Hranice arény (Walls): Pevné zdi, odráží agenty.
#          - [ ] Vnitřní objekty (Boxy/Kruhy): Zatím statické v kódu, lze přidat do configu.
#    - [ ] Zóny (Zones): Oblasti s jiným třením nebo speciálním efektem (Damage/Heal).
#
# 2. SENZORY (VSTUPY)
#    - [x] LIDAR: Paprskový senzor detekující vzdálenost k překážkám/agentům.
#    - [x] Relativní Pozice Cíle (GPS): Vektor k cíli [dx, dy].
#    - [x] Rychlost (Velocity): Vlastní vektor pohybu [vx, vy].
#    - [x] Inbox (Spatial Comms): Příjem zpráv od Top-K sousedů + Metadata (Angle, Dist).
#    - [ ] Vizální Vstup (Pixel-based): Renderovaný pohled (příliš pomalé pro miliony kroků, nepoužíváme).
#
# 3. AKCE (VÝSTUPY)
#    - [x] Pohyb (Continuous): Tank-drive [Levý_Motor, Pravý_Motor] nebo [Speed, Rotate].
#    - [x] Komunikace (Complex): 
#          - Broadcast: Všesměrové vysílání.
#          - Spatial: Cílené vysílání na souřadnice (Angle/Dist).
#          - Gating: Možnost mlčet (šetří penalizaci).
#    - [ ] Manipulace: Chytání objektů (Gripper) - plánováno pro V4.
#
# 4. FYZIKA
#    - [x] Kinematika: Newtonovský pohyb, setrvačnost.
#    - [x] Kolize: Pružné srážky (Agent-Agent, Agent-Zeď).
#    - [x] Tření (Friction): Lineární zpomalování.
#    - [x] Energie: Spotřeba paliva dle výkonu motorů.
#
# 5. ML & TRÉNINK
#    - [x] algoritmus: MAPPO (Multi-Agent PPO) s CTDE architekturou.
#    - [x] Paměť: GRU (Recurrent Actor) pro udržení kontextu (Telepatie).
#    - [x] Hand of God: Expertní navigace (vektorová pole) pro guiding.
#    - [x] Curriculum: Postupné ztěžování (HOG decay, Spam penalty ramp-up).
#    - [x] Massive Parallelism: JAX VMAP (64+ vesmírů naráz).
#    - [x] Checkpointing: Ukládání/Načítání stavu sítě a optimizéru.
#
# 6. VIZUALIZACE (RENDERER)
#    - [x] Headless: Běží na serveru bez monitoru.
#    - [x] Elementy: Agenti (šipky směru), Cíle (tečky), Lidar (paprsky), Historie (stopy).
#    - [x] Výstup: GIF animace, MP4 video (přes imageio).
#    - [ ] Real-time GUI: Okno s ovládáním myší (nepodporováno v massive módu).
#
# 7. TYPY ÚLOH (SCÉNÁŘE)
#    - [x] Navigace (Routing): Každý agent má svůj cíl.
#    - [x] Shlukování (Flocking): Všichni mají jeden cíl.
#    - [ ] Pronásledování (Tag): Tým A honí Tým B.
#    - [ ] Fotbal/Tlačení: Manipulace s pasivním objektem.
#
# 8. SAFETY LAYER (Hybrid Architecture)
#    - [x] Collision Reflex: Automatické zpomalení u překážek (squad-aware).
#    - [x] Agent Repulsion: Odpuzování agentů mimo vlastní squad (liquid swarm).
#    - [x] Geo-Fence: Virtuální síla od hranic arény.
#    - [x] Token Bucket Comm Limiter: Anti-spam s rozloženým refillem.
#    - [x] Watchdog (Anti-Stalemate): Detekce zacyklení + náhodný útěk.
#    - [x] Safety Metrics: Telemetrie intervencí (speed_reductions, hard_stops).
#    - [ ] AI Override: Možnost AI přepsat safety (action space nepodporuje).
#    - [ ] Energy Governor: Zpomalení při nízké energii (logika neaktivní).
#
# 9. INTENT SYSTEM (High-Level Control)
#    - [x] Velocity Mode: Vstup [v, omega] → výstup [Motor_L, Motor_R].
#    - [x] Target Mode: Vstup [rel_x, rel_y] → PID kontroler → motory.
#    - [x] IntentConfig: PID parametry (pid_pos_kp, pid_rot_kp, atd.).
#    - [ ] Follow Intent: Sledování jiného agenta (nutný extra kód).
#    - [ ] Formation Intent: Pozice ve formaci (nutná squad logika).
# =============================================================================


# =============================================================================
# 📊 FINDINGS (Benchmark Results)
# =============================================================================
# Key insights from experiments comparing control architectures:
#
# BENCHMARK: Heuristic "Go To Goal" policy, 200 steps, 20 agents
# 
# | Mode                        | Reward   | Goals Reached | FPS  |
# |-----------------------------|----------|---------------|------|
# | Direct (Unsafe)             | -73      | 0             | 124  |
# | Direct + Safety             | -73      | 0             | 30   |
# | Hybrid (Intent + Safety)    | +1252    | 2541          | 28   |
#
# CONCLUSION:
# - Hybrid architecture is the CLEAR WINNER (2541 goals vs 0).
# - Direct control cannot properly translate direction vectors to 
#   differential drive (Motor L/R) without the Intent Translator's PID.
# - Safety Layer is working correctly (26 interventions in Hybrid mode).
# - FPS drop (~4x) is acceptable trade-off for massive goal achievement.
#
# RECOMMENDATION: Always use Hybrid mode (intent.enabled=True, safety.enabled=True)
# for any real training scenario.
# =============================================================================
