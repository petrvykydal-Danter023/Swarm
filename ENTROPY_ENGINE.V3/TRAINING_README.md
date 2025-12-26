# 🧠 Entropy Engine V3 - Training Guide

Tento dokument popisuje tréninkové skripty pro Swarm Intelligence, specificky nový **Massive HOG (Hand of God)** režim.

## 🚀 Rychlý Start

Pro spuštění nejvýkonnějšího tréninku (Massive Parallel + Expert Guidance):

```bash
python train_massive_hog.py
```

Tento skript automaticky:
1.  Nastartuje **64 paralelních prostředí** (1280 agentů).
2.  Zkompiluje JAX computational graph (XLA).
3.  Začne trénink s **Hand of God** asistencí (100% -> 0%).
4.  Ukládá videa (GIF) do `outputs/massive_hog/`.

---

## 🛠️ Konfigurace (`TrainingConfig`)

Skript `train_massive_hog.py` používá dataclass `TrainingConfig` pro nastavení všech hyperparametrů.

| Parametr | Default | Popis |
| :--- | :--- | :--- |
| **Environment** | | |
| `NUM_ENVS` | `64` | Počet paralelních simulací. Vyšší číslo = stabilnější gradient. |
| `NUM_AGENTS` | `20` | Počet agentů v jedné aréně. Celkem agentů = Envs * Agents. |
| `MAX_STEPS` | `200` | Délka jedné epizody (kroků). |
| **Training** | | |
| `TOTAL_EPOCHS` | `50000` | Celkový počet epoch. |
| `LR_ACTOR` | `3e-4` | Learning Rate pro Actora (pohyb). |
| `LR_CRITIC` | `1e-3` | Learning Rate pro Critica (hodnocení stavu). |
| **Hand of God** | | |
| `HOG_START` | `1.0` | Počáteční síla asistence (1.0 = 100% expert). |
| `HOG_END` | `0.0` | Konečná síla asistence (0.0 = čistá AI). |
| `HOG_DECAY_EPOCHS` | `5000` | Počet epoch, během kterých asistence klesne na nulu. |
| **Rendering** | | |
| `RENDER` | `True` | Zapnout/Vypnout generování GIFů. |
| `RENDER_EVERY` | `1000` | Jak často (v epochách) generovat validační video. |

---

## 🔧 Jak vytvořit vlastní experiment?

Místo editace hlavního souboru můžete vytvořit vlastní spouštěcí skript importováním `run_training` a `TrainingConfig`.

**Příklad: `my_experiment.py`**

```python
from train_massive_hog import run_training, TrainingConfig

# 1. Definice vlastní konfigurace
my_config = TrainingConfig(
    NUM_ENVS=16,            # Méně prostředí pro debugging
    NUM_AGENTS=10, 
    TOTAL_EPOCHS=500,       # Krátký run
    HOG_DECAY_EPOCHS=100,   # Rychlejší ústup experta
    OUTPUT_DIR="outputs/my_debug_run",
    RENDER_EVERY=50         # Častější videa
)

# 2. Spuštění
if __name__ == "__main__":
    run_training(my_config)
```

---

## 🧠 Koncepty

### Hand of God (HOG) 👻
Trénink začíná s "pomocnými kolečky".
*   **Start**: Agentův pohyb je mixem jeho sítě a "Expertního Vektoru" (který zná cestu k cíli).
*   **Průběh**: Poměr experta lineárně klesá. Síť se učí predikovat to, co by udělal expert (Imitation Learning via PPO Rewards).
*   **Konec**: Expert zmizí a agent se pohybuje zcela samostatně.

### Massive Parallelism (JAX VMAP) 🌍
Místo jedné simulace běží 64 simulací naráz na jedné grafické kartě (nebo CPU via AVX).
*   **Výhoda**: Obrovské množství dat (Experience Replay) za zlomek času.
*   **Důsledek**: Extrémně stabilní učení, protože `mean_reward` je průměrován přes 1280 agentů, nikoliv jen 20.
