# 🔥 Brutální Vylepšení pro Entropy Engine V4

Seřazeno od nejvyššího impaktu.

---

## 1. Emergent Communication Protocol 🧠📡

**Co to je:** Nechat AI *naučit se jazyk* místo předem definovaných zpráv. Agenti si sami vytvoří protokol pro sdílení informací.

**Jak:**
- `msg_dim=8` → AI generuje embedding místo `[dist, angle, ...]`.
- **Listener Network**: Dekóduje zprávy od sousedů.
- **Reward Shaping**: Bonus za správnou predikci pozic sousedů (učí komunikovat).

**Proč je to brutální:** Emergentní jazyk = agenti spolupracují i na úkolech, které jsme nečekali.

---

## 2. Async Hierarchical Control (Macro-Actions) 🏛️

**Co to je:** Leader dává rozkazy na *T* kroků dopředu. Follower je plní autonomně.

**Jak:**
- Leader má `action_space = [Macro_Intent, Duration]`.
- Follower má `action_space = [Intent]` a drží se posledního rozkazu.
- Leader rozhoduje jen každých `T` kroků → 10x rychlejší inference pro leadery.

**Proč je to brutální:** Škáluje na stovky agentů bez explozivních nákladů na inference.

---

## 3. Full JAX JIT Wrapper ⚡ (TOP PRIORITA)

**Co to je:** Celý `env_wrapper.step()` jako jedna `jax.jit` funkce. Nula Pythonu.

**Jak:**
- Přepsat všechny loops jako `jax.lax.fori_loop`.
- Komunikaci, pheromony, lidary – všechno jako pure JAX ops.
- `jax.checkpoint` pro memory-efficient backprop.

**Proč je to brutální:** Z ~1500 FPS na potenciálně **50 000+ FPS** (GPU backend).

> [!IMPORTANT]
> Toto je nejvyšší priorita. 10-50x speedup umožní škálovat všechno ostatní.

---

## 4. Curriculum Learning Factory 🏭

**Co to je:** Automatický "Škola" systém. Agenti začínají na lehkých úkolech, postupně se difficulty zvyšuje.

**Jak:**
- `CurriculumManager` sleduje `success_rate` a `avg_reward`.
- Pokud `success_rate > 0.8` → unlock dalšího levelu (více agentů, menší arena, více překážek).
- Self-play: Nejlepší agenti z minulých epoch jako "soupeři".

**Proč je to brutální:** Model nikdy nepřeskakuje těžké úkoly, ale vždy se učí na hraně svých schopností.

---

## 5. World Model + Imagination Rollouts 🔮

**Co to je:** AI si "představuje" budoucnost bez simulace. Trénuje na svých vlastních snech.

**Jak:**
- `WorldModelPredictor` (základ už existuje v `mappo.py`).
- Rollout: `obs_t, action → pred_obs_{t+1}` opakovaně.
- AI plánuje 5-10 kroků dopředu v latentním prostoru.
- **Dreamer-style** policy update z imagined trajektorií.

**Proč je to brutální:** Dramaticky snižuje sample complexity. Agent se učí i když "nesimuluje".

---

## Shrnutí Priorit

| # | Název | Effort | Impact | Doporučení |
|---|-------|--------|--------|------------|
| 3 | Full JAX JIT | 🔴 High | ⭐⭐⭐⭐⭐ | **DO FIRST** |
| 1 | Emergent Comms | 🟡 Medium | ⭐⭐⭐⭐ | Po JAX JIT |
| 2 | Macro-Actions | 🟡 Medium | ⭐⭐⭐⭐ | Po JAX JIT |
| 4 | Curriculum | 🟢 Low | ⭐⭐⭐ | Kdykoliv |
| 5 | World Model | 🔴 High | ⭐⭐⭐⭐ | Long-term |
