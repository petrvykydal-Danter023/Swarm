# Emergentní Komunikace (Emergent Communication)

V `TopDownSwarmEnv` není komunikace předem definovaná (žádné "zprávy" jako text nebo kódy). Místo toho se agenti učí komunikovat od nuly formou **emergentního chování**.

## Jak to funguje?

### 1. Kanál (Raw Channel)
*   **Vysílání**: Každý agent má k dispozici **jeden spojitý kanál** (číslo `float`).
*   **Hodnota**: V každém kroku může agent nastavit tento signál na hodnotu v rozsahu `[-1.0, 1.0]`.
*   **Význam**: Engine tomuto číslu nedává **žádný význam**. Není to příkaz ani identifikace. Je to jen "tón", který agent vydává.

### 2. Učení (RL & Emergence)
Význam signálů vzniká evolučně během tréninku (Reinforcement Learning):
*   Agenti se učí korelace mezi vysílaným signálem a následnou odměnou.
*   **Příklad vzniku**:
    *   Agent A najde jídlo a náhodně vyšle signál `1.0`.
    *   Agent B (který je poblíž) "uslyší" `1.0`, přijde k A a oba dostanou odměnu za spolupráci.
    *   Postupně se tento vzor posílí a signál `1.0` začne pro swarm znamenat "Našel jsem jídlo/Pojďte sem".

### 3. Kontext a Prostor
*   Komunikace nepřenáší polohu ("Jsem na souřadnicích X,Y").
*   Příjemce informaci o poloze získává ze svých **senzorů** (`neighbor_vectors`).
*   **Vjem příjemce**: *"Vidím souseda vlevo (znám jeho polohu) a slyším od něj signál `0.8` (naučený význam: 'Pomoc')."*

## Technické Parametry

*   **Comm Range**: Signál má omezený dosah (nastavitelné v configu). Agenti mimo dosah slyší `0.0`.
*   **Packet Loss**: Simulace rušení. S určitou pravděpodobností signál nedorazí (přijat jako `0.0`).
*   **Cost**: Vysílání stojí energii (`Energy Cost = |signal| * 0.001 * dt`). To motivuje agenty, aby "nekřičeli" zbytečně, pokud to nepřináší užitek.
