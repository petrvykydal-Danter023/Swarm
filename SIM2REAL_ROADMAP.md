# Sim2Real & 3D Readiness Roadmap

Tento dokument shrnuje klíčové kroky, jak přiblížit `TopDownSwarmEnv` realitě a připravit agenty na nasazení do fyzických robotů (3D svět).

## 1. Pokročilá Fyzika a Hardware (The "Body" Gap)
Simulátor je aktuálně "příliš dokonalý". Roboti v realitě jsou nepřesní.

*   **Motor Lag & Latency ⚠️**:
    *   *Realita*: Když pošlete příkaz `speed=1.0`, robotovi trvá 100-200ms, než se rozjede.
    *   *Implementace*: Přidat buffer historie akcí. Agentova akce v čase `t` se projeví až v `t+k`.
*   **Prokluz kol (Slippage)**:
    *   *Realita*: Kola na prachu prokluzují. Odometrie (to, kde si robot myslí, že je) se rozchází s realitou.
    *   *Implementace*: Přidat náhodný šum do pohybu (`actual_move = command * random(0.8, 1.1)`) a oddělit "Real Pose" (pro fyziku) a "Odom Pose" (pro senzor).
*   **Nelineární vybíjení baterie**:
    *   *Realita*: Baterie nedrží napětí lineárně. Ke konci padá rychle.
    *   *Implementace*: Použít vybíjecí křivku (např. LiPo curve) pro výpočet `energy`.

## 2. Senzorika a Vnímání (The "Eye" Gap)
Aktuální "Radar" vidí skrz zdi a rozeznává objekty dokonale.

*   **Occlusion (Zákryt)**:
    *   *Realita*: LiDAR nevidí za překážku.
    *   *Implementace*: Raycasting. Paprsek se zastaví o první překážku. (Současný `_compute_radar` vidí vše v dosahu).
*   **Senzorová nejistota (Measurement Uncertainty)**:
    *   *Realita*: Senzory šumí různě podle vzdálenosti (čím dál, tím hůř).
    *   *Implementace*: Šum úměrný vzdálenosti (`noise = dist * 0.05`).
*   **Falešná pozitiva**:
    *   *Realita*: LiDAR občas "vidí" ducha (odraz od skla, prach).
    *   *Implementace*: Občas vložit do pozorování náhodný "objekt", který tam není.

## 3. Komunikace (The "Voice" Gap)
Už máme Packet Loss a Range, ale chybí:

*   **Bandwidth Limit (Šířka pásma)**:
    *   *Realita*: Nemůžete posílat floaty 60x za vteřinu od 100 agentů. Síť se zahltí.
    *   *Implementace*: Omezit počet zpráv za vteřinu (např. "Budget" na vysílání).
*   **Latency (Zpoždění)**:
    *   *Realita*: Zpráva letí vzduchem a zpracovává se (ping 20-500ms).
    *   *Implementace*: Zprávy doručovat až v příštím kroku (`mailbox` system).

## 4. Příprava na 3D World (Future Proofing)
Ačkoliv je engine 2D, můžeme trénovat myšlení pro 3D.

*   **Z-Axis Abstraction**:
    *   Definovat "výšku" objektů. (Může agent přejet payload? Ne. Může podjet most? Ano).
*   **Kamera / Vision Sensor**:
    *   Místo 8 paprsků radaru generovat 1D "Depth Map" (řádek pixelů). To je bližší vstupu z kamery/LiDARu, který se používá ve 3D.
*   **ROS 2 Bridge**:
    *   Vytvořit wrapper, který publikuje `cmd_vel` (Twist) a odebírá `Scan` (LaserScan).
    *   Tím pádem stejný mozek (RL model) může řídit simulovaného agenta i fyzického robota (např. TurtleBot).

## Doporučený postup (Next Steps)
1.  **Raycasting Radar**: Přepsat radar na paprskový, aby neviděl skrz zdi. (Kritické pro navigaci).
2.  **Motor Lag**: Přidat zpoždění reakcí. (Kritické pro control theory).
3.  **ROS Bridge**: Až bude model chytrý, napojit ho na ROS 2.
