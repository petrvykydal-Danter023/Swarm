# 🗣️ Jak spolu mluví naši agenti? (Vysvětlení pro lidi)

Tento dokument vysvětluje, jak funguje komunikace v Entropy Engine V2. Místo složité matematiky si představme skupinu lidí plnících úkol v hlučné místnosti.

---

## 📻 Dva Kanály: Jak se nepřekřičet

Agenti mají k dispozici dva způsoby, jak předat informaci. Představte si to jako **Vysílačku (Token)** a **Hlasité mluvení (Broadcast)**.

### 1. 📢 Kanál A: Vysílačka (Token Channel)
*   **Princip:** K dispozici je pouze **jedna** globální vysílačka.
*   **Kdo mluví:** V každém okamžiku může do vysílačky mluvit **jen jeden** agent.
*   **Jak se rozhodne:** Agenti se "hlásí" (generují číslo priority 0-1). Kdo se hlásí nejvíc urgentně, dostane vysílačku.
*   **Co slyší ostatní:** Všichni slyší zprávu z vysílačky, ať jsou kdekoli.
*   **Použití:** Pro důležité povely ("Jděte všichni na sever!", "Našel jsem cíl!", "Jsem Lídr!").

> **Pojistka:** Aby si jeden "ukecaný" agent nenechal vysílačku pro sebe, systém ho po použití na chvíli "umlčí" (sníží mu prioritu), aby se dostalo na ostatní.

### 2. 💬 Kanál B: Místní šum (Broadcast Channel)
*   **Princip:** Každý agent si mumlá pro sebe nebo křičí do svého okolí.
*   **Kdo mluví:** Všichni najednou.
*   **Co slyší ostatní:** Slyšíte jen ty, co jsou blízko vás (nebo v naší zjednodušené verzi slyšíte "šum" od všech, ale víte kdo co říká).
*   **Použití:** Pro sdílení stavu ("Jsem tady", "Mám objekt", "Jdu doleva"). Není to rozkaz, ale informace.

---

## 📖 Slovník: Co vlastně říkají?

Agenti nemluví česky ani anglicky. Mají předdefinovaný slovník **32 symbolů** (slov). Každé slovo má číslo (0-31), ale my jsme jim dali význam.

Příklady slovíček:
*   🟢 **Pohyb:** `GOING_TO` (Jdu tam), `STOP` (Stůj), `BLOCKED` (Jsem zaseklý)
*   🎯 **Cíle:** `FOUND_TARGET` (Našel jsem to!), `CARRYING` (Nesu to)
*   🤝 **Spolupráce:** `NEED_HELP` (Pomoc!), `FOLLOW_ME` (Za mnou)
*   👑 **Role:** `CLAIM_LEADER` (Já jsem šéf), `CLAIM_SCOUT` (Já budu průzkumník)
*   ⚔️ **Taktika:** `ATTACK` (Útok), `RETREAT` (Ústup)

Kromě slova pošlou i **Data (Payload)**: To jsou 4 čísla, která upřesňují zprávu.
*   *Příklad:* Slovo `GOING_TO` + Data `[0.5, 0.8, 0, 0]` znamená "Jdu na souřadnice X=0.5, Y=0.8".

---

## 🎭 Příklad ze života agentů

Představ si situaci: **Swarm má najít a přinést vlajku.**

1.  **Začátek epizody:** Všichni mlčí. Nikdo neví kde je vlajka.
2.  **Agent 3** najde vlajku v rohu místnosti.
    *   **Mozek:** "Heuréka! Musím to říct všem!" -> Zvedne Prioritu na 100%.
    *   **Systém:** Přidělí vysílačku Agentovi 3.
    *   **Agent 3 (Vysílačka):** 📢 `FOUND_TARGET` + `[pozice vlajky]`
3.  **Ostatní agenti:**
    *   Slyší z vysílačky: "Někdo (Agent 3) našel cíl na pozici X,Y!"
    *   Změní své chování: Přestanou bloudit a otočí se směrem k Agentovi 3.
4.  **Cesta zpět:**
    *   Agent 3 vezme vlajku.
    *   Agent 2 (který číhá u základny) si vezme vysílačku: 📢 `FOLLOW_ME` (Následujte mě k základně).
    *   Agenti utvoří formaci kolem Agenta 3 a chrání ho cestou zpět.

---

## 🧠 Jak se to učí? (Curriculum)

Agenti na začátku netuší, co `FOUND_TARGET` znamená. Je to pro ně jen náhodný šum "Slovo 7".

1.  **Pokus/Omyl:** Agent zkusí náhodně zařvat "Slovo 7", když stojí u cíle.
2.  **Odměna:** Dostane bod (reward), protože ostatní se k němu náhodou přiblížili a úkol splnili rychleji.
3.  **Spojení:** Agentův mozek si spojí: *"Když vidím cíl a řeknu 'Slovo 7', dostanu cukřík."*
4.  **Entropie:** Postupně se přestanou chovat náhodně a začnou 'Slovo 7' používat cíleně jen u cíle. Tím vzniká jazyk.

---

### Shrnutí pro tebe
Když se díváš na vizualizaci (ta barevná kolečka):
*   Pokud vidíš **Velkou bublinu** nebo čáru od jednoho agenta k ostatním -> To je **Vysílačka (Token)**. Ten agent právě velí.
*   Pokud vidíš malé blikání kolem všech -> To je **Broadcast**, sdílí si polohu.

::_______________________________________
::---------------------------------------

## 🏆 Cukr a Bič: Jak funguje odměňování? (Reward System)

Agenti se "učí" podle systému odměn (Rewards) a trestů (Penalites). Zde je tvůj "výchovný systém":

### 1. 🍭 Hlavní Cíl (The Big Prize)
*   **Dojdi k cíli:** Agent dostane obrovskou odměnu **+10 bodů**, když se dotkne svého cíle.
*   **Dostaň se blíž:** Každý krok dostává malinkou nápovědu (odměnu) podle toho, jestli se k cíli blíží nebo vzdaluje (tzv. *Shaping*).

### 2. 🤫 Ticho léčí (Bandwidth Penalty)
*   Agenti by nejraději "řvali" do vysílačky pořád, protože je to stojí 0 energie. To by zahltilo kanál.
*   **Pravidlo:** Pokud mluvíš do Broadcastu zbytečně (neříkáš `SILENCE`), stojí tě to **-0.01 bodu**.
*   **Výsledek:** Agenti mluví jen tehdy, když mají co říct.

### 3. 🤥 Detektor lži (Honesty Enforcement)
*   Agenti by mohli "hacknout" systém a klamat ostatní, aby si vylepšili skóre.
*   **Pravidlo:** Pokud agent zahlásí `FOUND_TARGET` ("Našel jsem to!"), ale ve skutečnosti je od cíle daleko (>50 metrů), dostane okamžitou facku **-0.5 bodu**.
*   **Výsledek:** Agenti nelžou o kritických věcech.

### 4. ⚖️ Fér Play (Token Fairness)
*   Pokud jeden agent drží "Vysílačku" (Token) moc dlouho, systém mu uměle sníží prioritu.
*   To není "trest" v bodech, ale "pravidlo hry" v prostředí. Zaručuje, že se ke slovu dostanou i ti tišší v koutě.

### 5. 🤝 Asistence (Communication Credit)
*   Co když jeden agent poradí druhému, ale sám cíl nenajde?
*   **Pravidlo:** Pokud někdo promluví do "Vysílačky" (Token) a **kdokoliv** z týmu do 15 kroků najde cíl, ten, co mluvil, dostane bonus **+2.0 body**.
*   **Výsledek:** Vyplatí se radit ostatním, i když z toho nemám přímý zisk hned. Vzniká altruismus.
