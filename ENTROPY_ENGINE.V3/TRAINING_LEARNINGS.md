# 🧠 Entropy Engine V3 - Training Tips & Learnings

Tento dokument shrnuje klíčové poznatky získané během vývoje a optimalizace tréninkového procesu Swarm Intelligence.

## 1. Massive Parallelism (Velké množství prostředí) 🌍
*   **Co to je**: Spuštění desítek až stovek simulací (Envs) najednou pomocí `jax.vmap`.
*   **Proč to funguje**: PPO (Proximal Policy Optimization) je "on-policy" algoritmus, který je velmi citlivý na "šum" v datech.
*   **Efekt**:
    *   Když běží **1 prostředí** (20 agentů): Gradient je nestabilní, agenti se mohou "zacyklit" ve špatné strategii.
    *   Když běží **64 prostředí** (1280 agentů): Průměrný gradient je extrémně přesný. Síť se učí robustní chování, protože vidí "všechny možné situace" v každém kroku.
*   **Tip**: Vždy se snažte maximalizovat počet prostředí, co  vám paměť GPU/CPU dovolí. Více agentů = stabilnější a rychlejší konvergence (na počet iterací).

## 2. Hand of God (HOG) - Curriculum Learning 👻
*   **Problém**: Na začátku je síť náhodně inicializovaná. Agenti se motají v kruhu a trvá dlouho, než náhodou narazí na cíl a dostanou odměnu.
*   **Řešení**: Vnutit jim experimentálně "správný směr" (Expert Vector) na začátku tréninku.
*   **Implementace**: Lineární decay (100% pomoc -> 0% pomoc).
*   **Výsledek**: Agenti okamžitě "ochutnají" odměnu. Critic se rychle naučí, že "být u cíle je dobré". Actor se pak snaží tento stav zreprodukovat, i když pomoc slábne.
*   **Tip**: Pokud se agenti neučí, zkuste jim prvních 5-10% tréninku "vodit ruku".

## 3. Shared vs. Unique Goals 🎯
*   **Unique Goals (Standard)**: Každý agent má svůj vlastní cíl.
    *   *Výhoda*: Agenti jsou samostatní a robustní.
    *   *Nevýhoda*: Obtížné učení, agenti se navzájem pletou ("křižovatka").
*   **Shared Goal (Zjednodušení)**: Všichni mají jeden společný cíl.
    *   *Výhoda*: Úloha se mění na "shlukování" (Flocking). Snadnější učení, méně kolizí, možnost kopírovat souseda.
    *   *Nevýhoda*: Riziko "stádního efektu" (agent bez sousedů je ztracen).
*   **Tip**: Pro rychlý debug navigace použijte Shared Goal. Pro finální "inteligentní" roj použijte Unique Goals nebo Curriculum (nejdřív Shared, pak Unique).

## 4. CTDE Architektura (Cooperation) 🤝
*   **Centralized Training (Critic)**: Kritik vidí **celý stav světa** (všechny pozice). Díky tomu ví, zda je situace dobrá pro tým jako celek.
*   **Decentralized Execution (Actor)**: Agent (voják) vidí jen **lokální okolí** (Lidar). Musí se rozhodovat sám.
*   **Proč to funguje**: Během tréninku "Bůh" (Critic) radí vojákovi (Actor), co by měl udělat, aby pomohl týmu, i když voják nevidí celý obraz. Po tréninku už voják jedná sám, ale má v sobě "intuici" vštípenou kritikem.

## 5. Rychlost je všechno (JAX Scan + JIT) ⚡
*   **Python Loop**: Pomalý (cca 60 FPS). Nutnost komunikace CPU <-> GPU v každém kroku.
*   **JAX Scan**: Celá epizoda (200 kroků) se zkompiluje do jedné operace na GPU (XLA). Žádný Python overhead.
*   **Výsledek**: Zrychlení 10x až 100x (1000+ FPS). Umožňuje trénovat miliony kroků za minuty.

## 6. Komunikace 🗣️
*   **Cena**: Přidání komunikačních kanálů (vocab) zvětšuje výstupní prostor sítě. Trénink je pomalejší a náročnější na stabilitu.
*   **Tip**: Nejdřív naučte agenty chodit (pohyb only). Až to umí perfektně, přidejte "řeč". Učit se chodit a mluvit naráz je pro RL velmi těžké.
