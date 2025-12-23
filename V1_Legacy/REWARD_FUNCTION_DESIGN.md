# Sim2Real Reward Function Design

Pro úspěšný přenos nátrenovaného chování do reality (a 3D světa) nesmí být Reward Function pouze o "splnění úkolu". Musí aktivně tvarovat chování tak, aby bylo proveditelné na fyzickém hardwaru.

## 1. Safety & Hardware Protection (Ochrana HW)

Roboti nejsou nezničitelní. V simulaci mohou do sebe narážet v plné rychlosti, v realitě to znamená zničené motory a plasty.

### ❌ Current State
```python
reward = -distance_to_goal
# Agent se snaží dostat k cíli za každou cenu, i když to znamená náraz v plné rychlosti.
```

### ✅ Proposed Upgrade: `collision_penalty` & `velocity_cap`
```python
# 1. High Velocity Collision Penalty
# Pokud dojde ke kolizi a rychlost byla vysoká -> velký trest.
min_safe_dist = agent.radius + 2.0
nearest_dist = 999
for other in env_state['agents']:
    if other['id'] != agent['id']:
        d = dist(agent, other)
        nearest_dist = min(nearest_dist, d)

if nearest_dist < min_safe_dist:
    # Penalizovat rychlost v blízkosti překážek
    speed = math.sqrt(agent['vx']**2 + agent['vy']**2)
    if speed > 0.5: # 50% speed limit near obstacles
        reward -= speed * 2.0 
```

## 2. Energy Efficiency (Spotřeba Energie)

Reální roboti mají omezenou baterii. Agenti se musí naučit "šetřit", ne jen "sprintovat".

### ❌ Current State
```python
# Žádná penalizace za pohyb. Agent kmitá sem a tam.
```

### ✅ Proposed Upgrade: `energy_cost` & `idle_reward`
```python
# 1. Action Magnitude Penalty (Jemnější pohyby)
# reward -= (abs(ax) + abs(ay)) * 0.01

# 2. Battery Awareness
# Pokud málo baterie -> Větší motivace nic nedělat (šetřit).
if agent['energy'] < 0.2:
    if abs(agent['vx']) < 0.01:
        reward += 0.1 # Odměna za odpočinek při vybití
```

## 3. Smoothness Control (Ochrana převodovek/Motorů)

Prudké změny směru (Jerk) ničí převodovky a způsobují prokluz kol.

### ❌ Current State
```python
# Agent může měnit směr okamžitě (pokud to fyzika dovolí).
```

### ✅ Proposed Upgrade: `action_smoothing`
```python
# Vyžaduje historii akcí (kterou máme v Motor Lag bufferu!)
# reward -= abs(current_action - last_action) * 0.5
# Nutí agenta měnit akce plynule.
```

## 4. Communication Efficiency (Bandwidth)

Vysílání zpráv stojí energii a zahlcuje síť.

### ❌ Current State
```python
# Agent může "křičet" (signal=1.0) neustále bez trestu (kromě malé energy cost v enginu).
```

### ✅ Proposed Upgrade: `silence_reward`
```python
# Odměna za mlčení, pokud zpráva není nutná.
if abs(agent['comm_signal']) < 0.1:
    reward += 0.05
# Tím se naučí komunikovat jen když je to důležité.
```

## 5. Sparse vs Dense Rewards (Tréninková strategie)

*   **Dense (Husté)**: `reward = -distance`. Navádí agenta krok po kroku. Rychlé učení, ale náchylné na lokální minima (zasekne se za zdí).
*   **Sparse (Řídké)**: `reward = 100 if reached_goal else 0`. Těžké na naučení (agent bloudí), ale robustnější strategie.

### 💡 Hybridní přístup (Curriculum Learning)
1.  Začít s **Dense** odměnami (aby pochopil, co má dělat).
2.  Postupně ($alpha \to 0$) přejít na **Sparse** (aby našel nejlepší cestu, ne jen následoval gradient vzdálenosti).

## Příklad komplexní "Sim2Real" funkce (Python Code String)

```python
reward = 0.0

# --- 1. Objective (The Goal) ---
goal = env_state['goals'][0]
dist = math.sqrt((agent['x']-goal['x'])**2 + (agent['y']-goal['y'])**2)
reward += -dist / 100.0 # Dense guidance
if dist < 5.0: reward += 50.0 # Success bonus

# --- 2. Safety (Collision Avoidance) ---
# Detect near obstacles using radar or positions
min_dist = 999
for o in env_state.get('obstacles', []): # Assuming obstacle list available in env_state or radar
    d = math.sqrt((agent['x']-o['x'])**2 + (agent['y']-o['y'])**2) - o['radius'] - agent['radius']
    min_dist = min(min_dist, d)

if min_dist < 2.0:
    reward -= 1.0 # Proximity warning
    if min_dist <= 0:
        reward -= 10.0 # Collision!

# --- 3. Efficiency (Energy & Smoothness) ---
speed = math.sqrt(agent['vx']**2 + agent['vy']**2)
accel = math.sqrt(agent.get('ax', 0)**2 + agent.get('ay', 0)**2) # Need to pass last action to env_state

reward -= speed * 0.01 # Cost of transport
reward -= accel * 0.05 # Cost of acceleration (Jerk/Motor load)

# --- 4. Comms ---
if abs(agent.get('comm', 0)) > 0.1:
    reward -= 0.05 # Talking costs bandwidth

return float(reward)
```
