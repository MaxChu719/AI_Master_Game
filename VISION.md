# AI Master — Game Vision

> A pygame-ce idle/wave-defense game where you build, train, and evolve AI minions using real reinforcement learning algorithms.

---

## Core Concept

**AI Master** is a hybrid idle/wave-defense game in which the player acts as a god-like AI architect. The player never fights directly — instead they design neural networks, choose learning algorithms, purchase minion types, and watch their creations evolve in real time during battle.

### Core Pillars

| Pillar | Description |
|---|---|
| **Real Learning** | Minions genuinely improve via Rainbow DQN. Loss, buffer fill, and training steps are live. |
| **Deep Customization** | The "Brain Workshop" lets players compose neural architectures from unlockable building blocks. |
| **Idle Depth** | Battles can run while the player is away. Training continues, coins accumulate, and evolution happens in the background. |
| **Spectacle** | Every minion type, ability, and upgrade has a distinct visual signature. The battlefield is alive with particles, shaders, and procedural animations. |
| **Physics Feel** | Entities physically separate when overlapping, and attacks send targets flying with directional knockback — the battlefield has weight and impact. |

---

## Enhancements & Stretch Ideas

### Minion Mutation System
After a minion survives N waves, it can undergo a **Mutation Event** — a random or player-directed alteration to its architecture (e.g., a new skip connection is added, a layer doubles in width). This gives every minion a unique genealogy and creates attachment. Show a "genetic tree" of each minion's evolution.

### Enemy AI That Adapts Too
Enemies are also driven by simple policies that counter-evolve against the minion population. This creates an **arms race** dynamic. Players must constantly update their minions' architectures or get out-evolved. This keeps late-game from becoming trivial.

### Research Tree (Meta-Progression)
A separate meta-currency called **Insight** (earned by observing training milestones, e.g., "loss dropped 50%") unlocks permanent research nodes: new learning algorithms, new layer types, new minion classes, new activation functions. This separates short-term coin spending from long-term progression.

### Brain Sharing & Export
Players can export a minion's brain configuration and weights as a `.brain` file and share it. Others can import it as a starting point. This creates a community of "AI designs," mimicking real ML model sharing (like HuggingFace).

### Live Training Visualization
Render a small, live **loss/reward curve** as part of each minion's status card. Show the network graph (nodes and edges) in the Brain Workshop using a force-directed layout. This makes the invisible (learning) visible and exciting.

### Synergy Bonuses
Minion squads get **Synergy Bonuses** when composed thoughtfully — e.g., a Healer + 2 Fighters using the same learning algorithm get a shared experience replay buffer, making all three train faster. This rewards team-building strategy.

### Boss Encounters with Explainability Challenges
Bosses have a "shield" that can only be broken if your minions can demonstrate a correct policy decision in a challenge scenario (an in-game "interpretability test"). This turns RL concepts into puzzle mechanics.

### Prestige / New Game+
At the end of the final wave tier, the player can **Prestige** — losing coins but keeping all unlocked architecture components, and gaining a special prestige-only modifier (e.g., minions can now use ensemble policies — voting between multiple brains).

### Environmental Hazards That Are Observations
Add battlefield conditions (fog of war, lava zones, moving platforms) that directly enter the minion's observation vector. This forces the player to ensure their network can handle variable inputs and rewards the use of advanced architectures over basic ones.

### Minion Mood & Motivation System
Give each minion a simulated "morale" stat that degrades if it keeps losing. Low morale reduces exploration, making the minion more conservative. Players must spend coins on "motivation boosts" or rest cycles to recover morale — a deliberate idle mechanic.

---

## Minion Roster

Each minion type has a distinct role, personality, and visual style:

| Type | Role | Visual Style |
|---|---|---|
| **Fighter** | DPS melee | Blue rect, labeled "F", glowing armored knight; DQN with 16-action directional attack (8 move + 8 attack) |
| **Archer** | DPS ranged | Green rect, labeled "A", bow flash; DQN with 24-action directional shoot (8 move + 16 attack at 22.5° precision); velocity-aware observation for trajectory prediction |
| **Wizard** | DPS ranged/AoE | Floating robe, spell orbs |
| **Healer** | Support | Soft-light aura, healing rings |
| **Tank** | Frontline | Heavy, rune-etched armor |
| **Scout** | Utility/debuff | Sleek cloaked silhouette |
| **Summoner** | Minion of minion | Arcane circle, floating runes |

---

## Enemy Factions

Enemies are organized into **Factions**, each with distinct behaviors:

| Faction | Playstyle | Special Trait |
|---|---|---|
| **Swarm** | Massive numbers, low HP | Overwhelm by count |
| **Spider** | Medium groups, ranged, evasive | Shoots freezing web; keeps distance; freezes minions briefly; present every wave at ~25% of swarm count |
| **Berserkers** | High damage, low defense | Enrage below 30% HP |
| **Shieldbearers** | Frontline blockers | Reflect damage, must be flanked |
| **Sorcerers** | Ranged elemental attacks | Interrupt channels, apply debuffs |
| **Evolvers** | Counter-learning agents | Shift tactics based on minion policies |
| **Bosses** | Animated, multi-phase, fireball-launching, swarm-spawning | Every 5 waves; phase 2 at 50% HP |

---

## The Brain System

The Brain is the core gameplay artifact — a composed neural network that acts as a minion's decision-making policy.

Players build brains from unlockable **modules** (building blocks), **activation functions**, **normalization layers**, and **combiners** (how modules connect). Brain capacity can be upgraded with coins.

The player chooses a **learning algorithm** for each minion. Algorithms are unlocked via the Research Tree and each has distinct tradeoffs for different playstyles and minion roles.

### V1 Brain Architecture — CNN + Rainbow DQN

Observations are **84×84×4 stacked grayscale images** (crop of the arena centered on each minion, arena-height × arena-height, zero-padded, resized to 84×84). Each minion gets its own crop so the network learns from a self-centered perspective.

The V1 brain uses a **MobileNetV4-inspired CNN** (4 depthwise-separable conv layers) followed by the same **Noisy Dueling Distributional** (C51) head used in prior versions.

```
Input [B, 4, 84, 84]   (4 stacked grayscale frames, arena-crop centered on minion)
  → Conv(4→32, k=3, stride=2) → ReLU                [B, 32, 42, 42]
  → DSConv(32→64,   stride=2)                        [B, 64, 21, 21]
  → DSConv(64→128,  stride=2)                        [B,128, 10, 10]
  → DSConv(128→256, stride=2)                        [B,256,  5,  5]
  → GlobalAvgPool                                    [B, 256]
  → Value stream:     NoisyLinear(256→64) → ReLU → NoisyLinear(64→51 atoms)
  → Advantage stream: NoisyLinear(256→64) → ReLU → NoisyLinear(64→A*51)
  → Q = V + (A − mean(A))   [B, A, 51 atoms]
  → log_softmax → log-prob distribution
```

CNN layers (channels, kernel size) are configurable in `config.json → cnn`.
Number of stacked frames and crop size are configurable in `config.json → image_obs`.

The preset heuristic policy (used during warmup) still operates on the internal 41-dim vector observation so it can compute exact distances and aim lead without visual ambiguity.

**Rainbow components active in V1:**
| Component | Status |
|---|---|
| Double DQN | ✅ Policy net selects actions; target net evaluates (Memory Replay) |
| Expected SARSA | ✅ Softmax-policy expected value target (in-game online training) |
| Dueling Networks | ✅ Value + Advantage head |
| Prioritised Experience Replay (PER) | ✅ SumTree with IS weights |
| N-step returns | ✅ N=3 |
| Distributional (C51) | ✅ 51 atoms, v_min=-30, v_max=80 |
| Noisy Networks | ✅ Factorized noise replaces ε-greedy |
| Soft Target Update | ✅ Polyak averaging every step (τ=0.005) |

**Two training modes run concurrently:**
- **In-game (Expected SARSA)**: online, runs each frame in a background thread; uses the replay buffer with a softmax-policy expected-value target for stable online updates.
- **Memory Replay (Rainbow DQN)**: off-policy batch training from the ResearchLab; uses full Double DQN target (standard Rainbow).

Exploration uses a **role-specific preset heuristic policy** during the warmup phase (buffer filling), then relies entirely on NoisyNet exploration afterwards.

Before each battle, a **Training Setup menu** lets the player choose training hyperparameters and toggle **Preset+Train mode** — minions always act via the preset heuristic policy, but interactions are still collected into the replay buffer and the DQN trains in the background. This allows the player to compare heuristic action quality while simultaneously building up a trained network for later use.

---

## Wave Structure

100 waves, grouped into tiers of 5. **Every 5th wave is a Boss Wave**.

```
Waves  1–5:   Swarm tutorial → Wave 5 Boss
Waves  6–10:  Swarm escalation → Wave 10 Boss
Waves 11–50:  Increasing swarms, scaling boss difficulty
Waves 51–100: Late-game, large swarms, powerful bosses
```

Non-boss wave enemy count: min(60, 22 + (wave_idx − 10) × 2) for waves 11+.
Boss waves: 8 swarms + 1 Boss (HP = 800 + wave_idx × 80).

**Boss behaviour:**
- Slow movement toward nearest minion.
- Fires volleys of 3–5 fireballs every 4 seconds (fan spread, more in phase 2).
- Spawns 4–6 swarms near itself every 6 seconds.
- Phase 2 triggers at 50% HP: faster rotation, counter-rotating ring, extra orbs, extra fireballs.
- Death: multi-ring shockwave animation, 500 coin reward.

---

## AI Master & Spells

The player accumulates **MP** (mana points) that regenerate over time. Four spells are available:

| Spell | Cost | Effect | Targeting |
|---|---|---|---|
| **Healing** | 30 MP | Instant AOE heal to all minions within radius | Click to place; radius shown on cursor |
| **Fireball** | 50 MP | Meteor descends after 1s; AOE damage to enemies within explosion radius | Click to place; flight shown; radius shown |
| **Summon Fighter** | 50 MP | Instantly deploys a new Fighter minion (only during an active wave; cap enforced) | Click icon — instant |
| **Summon Archer** | 40 MP | Instantly deploys a new Archer minion (only during an active wave; cap enforced) | Click icon — instant |

Spell animations:
- **Healing**: soft expanding green circle, rising sparkles, cross shimmer.
- **Fireball**: growing meteor descends with fire trail → multi-ring explosion + debris particles.

All spell parameters (damage, radius, cooldown, MP cost) are upgradeable in the **AI Master** menu.  
Summon MP costs are configurable in `config.json → spells.summon_fighter / summon_archer`.

---

## Deployment & In-Wave Spawning

The player can deploy multiple minions of each type before waves start. All minions of the same type **share the same brain** (Rainbow DQN agent), so their transitions all feed into one replay buffer and one network.

Default: 1 Fighter + 1 Archer (fixed starting deployment, unaffected by upgrades).  
The **Deploy Limit** upgrade raises a single **global cap** on total minions (Fighters + Archers combined). Level 0 → cap 2; fully upgraded (level 5) → cap 20. Players can allocate those slots freely across both types. The cap governs both the starting deployment and in-wave spawning combined.

### Dynamic In-Wave Spawning

During active training waves, the AI Master can spend **MP** to summon additional minions via the **Summon Fighter** and **Summon Archer** spell icons. Newly spawned minions join immediately and contribute their experience to the same shared replay buffer as existing minions of their type.

- Summon cost is configurable per minion type (default: 50 MP for a Fighter, 40 MP for an Archer)
- A global cap prevents unbounded scaling (configurable; e.g., fully upgraded = 20 total across both types)
- Summon icons appear in the bottom-centre spell panel; greyed out when global cap reached, wave not active, or insufficient MP; badge shows `F/A type count + total/cap`
- Dead minions from prior waves count against the cap; summoning is the only way to replace losses mid-run

---

## Memory Replay Training

Between battles, the player can spend coins to run **off-policy training** on the accumulated replay buffer:
- Specify number of iterations (10–5 000).
- Cost = iterations × 0.01 coins (10 coins per 1 000 iterations).
- Training runs in a background thread; a **live progress bar** is shown while training, and the result (avg loss + avg reward per sampled batch) appears when done.
- After training completes, **only model checkpoints are saved** (fast); the replay buffer is not re-serialized here.

### Session-Based Replay Buffer Storage

The replay buffer is stored as a **folder of per-session snapshot files**, one per gaming session (run from new-game/load to game-over):

```
data/saves/
├── {name}.json
├── {name}_fighter.pt           ← model checkpoint
├── {name}_archer.pt
├── {name}_fighter_buffer/      ← session snapshot folder
│   ├── session_0001.pt         ← oldest retained session
│   ├── session_0002.pt
│   └── session_0003.pt         ← most recent session
└── {name}_archer_buffer/
    └── ...
```

**Saving (on game-over only):** At session end, extract the top `max_session_transitions` (default 10,000, configurable) transitions by PER priority from the current in-memory buffer and write a new `session_{N:04d}.pt` file. This preserves the most informative experience (highest TD error) rather than a random cross-section. If the folder exceeds `max_buffer_sessions` (default 10, configurable), the oldest file is deleted. No buffer save occurs mid-session (e.g. after each wave) — only the model checkpoint is saved then.

**Loading (on game load):** All session files are read in ascending order (oldest first) and re-inserted into the in-memory buffer. Loading oldest-first ensures the most recent sessions survive naturally when the ring buffer overflows.

**Priority handling:** Loaded transitions are re-inserted at maximum priority. Prior priorities are stale (network weights have changed since), so uniform max-priority is correct; priorities are re-ranked naturally during training.

**Session index tracking:** The save JSON stores monotonically incrementing counters (`fighter_session_idx`, `archer_session_idx`) so filenames are unique and never reused.

### Memory Storage Frequency

Since the game runs at 60 FPS, storing a transition every single frame would fill the buffer with near-identical, highly correlated observations. Instead, transitions are stored every **N frames** (default 10, configurable). This gives 6 transitions/second per minion — dense enough for effective learning, but with far less temporal correlation. All minions of the same type write into the same shared buffer, so the effective collection rate scales linearly with the number of active minions of that type.

---

## Economy & Progression

| Currency | Source | Spent On |
|---|---|---|
| **Coins** (⚙) | Kill enemies (+10), complete waves (+50), kill boss (+500) | Minions, brain upgrades, memory replay training |
| **Insight** (💡) | Training milestones, boss kills | Research tree nodes (future) |
| **Shard** (💎) | Prestige reward | Prestige-exclusive upgrades (future) |

---

## Research Lab (Upgrade Menu)

Two tabs:

| Tab | Upgrades |
|---|---|
| **AI Minions** | Fighter: HP, Attack, Move Speed, Atk Speed, Stamina (5 levels each) · Archer: HP, Attack, Move Speed, Stamina (5 levels each) |
| **AI Master** | Max MP, MP Regen, Heal Amount, Heal Radius, Heal Cooldown, Fireball DMG, Fireball Radius, Fireball CD, Deploy Limit (global cap: 2→5→8→12→16→20), Memory Replay Training UI |

---

## Visual Style

- All sprites are **pixel art** at 32×32 or 64×64
- Each entity has idle, walk, attack, and death animations
- Minions have a **glow layer** that intensifies when their loss is decreasing (visual learning feedback)
- Boss has animated orbital rings, breathing glow, and phase-based color shifts
- Spell effects: particle systems + shockwave rings

Particle effects mark key events: melee hits, spell casts, minion deaths, training milestones, coin drops, wave starts, and boss phase transitions.

---

## Audio Design

Dynamic music responds to battle intensity using a **layered stem** approach:

- **Stem 1 (always on)**: Ambient/pad — dark synth drone
- **Stem 2 (wave active)**: Rhythmic percussion fades in when enemies spawn
- **Stem 3 (high intensity)**: Lead melody fades in when HP is low or enemy count is high
- **Stem 4 (boss)**: Unique boss track, crossfaded on boss spawn

Sound effects are distinct and meaningful for every key game event: attacks, spells, coins, training milestones, brain upgrades, enemy deaths, boss roars, and wave completions. Entity sounds are panned left/right based on position on screen.

---

## UI & HUD

The **Battle HUD** gives the player a live view:
- Top-left: "Wave X/100" (red "★ BOSS WAVE ★" on boss waves)
- Top-center: control panel strip — Pause, Reset Brain, Speed, Menu
- Top-right: Player name + Coins
- Bottom-left: DQN Training panel — Fighter Brain (mode, loss, avg reward EMA, steps, buffer %, LR, PER β), Archer Brain (same), Speed; shows "Saving..." badge while async checkpoint save is in progress
- Bottom-center: Two-row spell panel — Row 1: MP bar; Row 2: Healing, Fireball, Summon Fighter, Summon Archer icons (with cooldown overlays and count badges)
- Bottom-right: All minion HP/SP bars stacked; enemy count + boss HP if active
- Center overlays: INTERMISSION countdown, GAME OVER, VICTORY

---

## Scenes

| Scene | Description |
|---|---|
| `Main Menu` | Title screen — Start New Game (name input + save), Load Saved Game, Quit |
| `Loading` | Animated loading screen shown while new-game/load-save initialises agents in a background thread, preventing UI freeze |
| `Research Lab` | Meta-upgrade screen — 3 tabs: Fighter, Archer, AI Master |
| `Battle` | Main gameplay — wave combat + live Rainbow DQN training; returns to Research Lab on end |
| `Brain Workshop` | Compose, preview, and purchase neural architectures (future) |
| `Shop` | Buy minion types, consumables, synergy items (future) |
| `Research` | Spend Insight on the meta-progression research tree (future) |
