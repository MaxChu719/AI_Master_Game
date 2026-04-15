# AI Master — Version 1 Implementation Reference

> Current working implementation: a wave-defense game where AI minions trained by full Rainbow DQN fight 100 waves of enemies, with boss encounters every 5 waves, player spells, and a deep upgrade system.

---

## Environment Setup

This project is developed on **Windows**. Use **Anaconda Prompt** (or a PowerShell/CMD session with conda initialized) to run the project.

Activate the project conda environment:

```cmd
conda activate playground
```

Then run from the `ai_master\` directory:

```cmd
python main.py
```

If you are using VS Code, open an integrated terminal with the `playground` environment selected, or run:

```cmd
conda run -n playground python main.py
```

---

## License

This project is released under the **Polyform Noncommercial License 1.0.0** (see `LICENSE`).

- **Non-commercial use** (personal, educational, contributions, research) is freely permitted.
- **Commercial use** (streaming for profit, selling, monetizing) is reserved exclusively for the copyright holder.
- Contributors may submit patches/PRs; their contributions remain under the same license.

---

## Table of Contents

1. [Implementation Guide for AI Sessions](#1-implementation-guide-for-ai-sessions)
2. [Architecture Overview](#2-architecture-overview)
3. [Game Loop & Scene System](#3-game-loop--scene-system)
4. [Core Entities](#4-core-entities)
5. [Physics System](#5-physics-system)
6. [AI Minion Brain System](#6-ai-minion-brain-system)
7. [Reinforcement Learning Integration](#7-reinforcement-learning-integration)
8. [Wave & Enemy System](#8-wave--enemy-system)
9. [Economy](#9-economy)
10. [UI & HUD](#10-ui--hud)
11. [V1 Scope & File Structure](#11-v1-scope--file-structure)
12. [Technical Dependencies](#12-technical-dependencies)
13. [Last Implementation Notes](#13-last-implementation-notes)

---

## 1. Implementation Guide for AI Sessions

> **READ THIS FIRST.** This section tells you (the AI assistant) how to approach implementing this project across multiple sessions.

### How Each Session Should Work

1. **Read VISION.md** to understand the full vision.
2. **Read CURRENT.md** to understand the current implementation details.
3. **Follow the existing code style.** - Match naming conventions, import patterns, and file organization from what's already there.
4. **Test that the game runs** after your changes. Run `python main.py` from the `ai_master/` directory and confirm no crashes. If you can't run it, at a minimum, verify there are no import errors.
5. **Update the Last Implementation Notes** - Override the section with your implementation summary. Don't keep appending new changes to the Last Implementation Notes.
6. **Make everything customizable in the config.json** - Whenever possible, please make all your changes customizable in the config.json.
7. **Make good physics, sound effects, and animations** - This will make the idle-like gameplay more engaging.

### Rules & Constraints

- **Use pygame-ce** (not plain pygame). Import as `import pygame` but install as `pygame-ce`.
- All files that use `X | Y` union type hints or `list[T]` generic hints at runtime must include `from __future__ import annotations` at the top (before the docstring if there is one). This ensures Python 3.9 compatibility.

### Cross-Step Dependencies

| Step | Depends On | What It Provides |
|---|---|---|
| 1 — Engine Foundation | Nothing | `main.py`, `GameManager`, `BaseScene`, `MainMenuScene` |
| 2 — Entities & Arena | Step 1 | `Minion`, `Enemy` classes, renders in `BattleScene` |
| 3 — Combat & Movement | Steps 1–2 | `MovementSystem`, `CombatSystem`, wired into battle loop |
| 4 — Wave System | Steps 1–3 | `WaveSystem`, coin tracking, game over / restart flow |
| 5 — DQN Brain | Steps 1–2 (needs entity classes) | `BrainNetwork`, `MinionEnv`, `DQNAgent` |
| 6 — Training Integration | Steps 1–5 (all prior) | `TrainingSystem`, minion acts on learned policy |
| 7 — HUD | Steps 1–6 | `HUD` overlay with stats |
| 8 — Playtest & Tune | Steps 1–7 | Balance tweaks, "R" to reset brain |
| 9 — Sound, Visuals, Stamina, Archer | Steps 1–8 | Archer minion, stamina, SFX, projectiles |

### How to Handle Problems

- **If existing code from a prior step has a bug that blocks you:** Fix it minimally. Note the fix in your Implementation Notes.
- **If the plan says to do X but it doesn't make sense given the existing code:** Ask for clarification.
- **If you're unsure about a design decision:** Ask for clarification.

### Implementation Notes Format

After completing a step, add notes like this under Last Implementation Notes:

```
> **Task summary (completed):**
> - Key change 1
> - Key change 2
> - Deviated from plan: ...
> - Known issue: ...
```

---

## 2. Architecture Overview

```
ai_master/
├── main.py                  # Entry point
├── engine/                  # Core pygame-ce engine wrappers
├── scenes/                  # Scene state machine
├── entities/                # Game objects (master, minions, enemies, boss, spells)
├── ai/                      # RL algorithms, neural networks, brain system
├── systems/                 # ECS-like systems (combat, movement, wave)
├── ui/                      # HUD, menus
├── audio/                   # Sound manager
└── data/                    # Configs, save files
```

The engine uses a **Scene Stack** managed by a central `GameManager`. Systems are updated each frame in fixed priority order. Entities are stored using plain Python dicts + lists.

---

## 3. Game Loop & Scene System

### Main Loop (pygame-ce)

```python
# main.py skeleton
import pygame
from engine.game_manager import GameManager

def main():
    pygame.init()
    screen = pygame.display.set_mode((1280, 720), pygame.SCALED | pygame.RESIZABLE)
    pygame.display.set_caption("AI Master")
    clock = pygame.time.Clock()
    manager = GameManager(screen)
    manager.push_scene("main_menu")

    while manager.running:
        dt = clock.tick(60) / 1000.0  # delta time in seconds
        for event in pygame.event.get():
            manager.handle_event(event)
        manager.update(dt)
        manager.draw(screen)
        pygame.display.flip()

    pygame.quit()
```

### Scenes

| Scene | Description |
|---|---|
| `MainMenuScene` | Start New Game (name input, override confirm) / Load Saved Game / Quit |
| `LoadingScene` | Animated loading screen (spinning arc + dots) shown while `new_game()` or `load_save()` runs in a background thread; transitions to ResearchLab when done |
| `ResearchLabScene` | 2-tab upgrade screen: AI Minions (Fighter + Archer columns side-by-side), AI Master; Memory Replay training UI in AI Master tab; "Start Battle" now pushes TrainingSetupScene |
| `TrainingSetupScene` | Pre-battle config: Mode (DQN Training / Preset+Train), LR, Warmup Preset Ratio, Min Buffer, Target Update Freq, Batch Size — applies to both agents before launching battle |
| `BattleScene` | Core battle scene — wave combat + live Rainbow DQN training; returns to Research Lab on end |

### Fixed Update vs. Render

- **Physics/AI tick**: each frame with delta-time
- **RL training step**: runs asynchronously in a background `ThreadPoolExecutor` thread
- **Memory Replay training**: dedicated daemon thread, launched from ResearchLab
- **Render**: 60 Hz

---

## 4. Core Entities

### Fighter Minion (`entities/minion.py`)

- Blue colored rectangle, size 24, labeled "F"
- HP: 120, speed: 200 px/s
- Attack: 15 damage, 50px range (+enemy body radius for hit detection), 0.5s cooldown
- Stamina: 100, regen 20/s, cost 20 per swing (5 swings before depleted)
- Stamina bar drawn above HP bar (yellow → orange as depleted)
- Directional sword arc: 70° cone at attack range; attack fires in the DQN-chosen direction
- Driven by Rainbow DQN policy (16 actions: 8 move directions + 8 directional attacks)
- Attack only triggers when DQN selects an attack action (no auto-attack)
- Multiple fighters share one `DQNAgent` in `game_manager.fighter_agent`
- **Dead state**: renders as a tombstone (rounded gray pillar with horizontal groove) with "F" label preserved in muted tone

### Archer Minion (`entities/archer.py`)

- Green colored rectangle, labeled "A"
- HP: 70, driven by Rainbow DQN
- Stamina: 80, regen 15/s, cost 25/shot — stamina bar drawn above HP bar
- Fires arrows every 1.3s within 270px range (gated by stamina)
- Driven by Rainbow DQN policy (16 actions: 8 move directions + 8 directional attack/shoot angles)
- Separate `DQNAgent` in `game_manager.archer_agent`; shares brain across all deployed archers
- **Dead state**: renders as a green-tinted tombstone with "A" label preserved in muted tone
- HP and stamina persist between waves (no revival — dead minion stays dead)

### Spider Enemy (`entities/spider.py`)

- Dark purple animated body with 8 jointed legs (drawn via line segments with wobble animation)
- HP: 50, speed: 100 px/s, size: 20; `enemy_type = 1`
- Keeps a **preferred distance** (~200 px) from the nearest minion — flees if too close (<120 px), strafes if at preferred range, approaches if too far
- Shoots a **SpiderWeb** projectile every 3 s at the nearest minion within 220 px
- Web hit: 10 damage + **freezes** the target for 1.5 s (frozen minion cannot move; blue web overlay shown)
- Web shoot flash animation (brief radial beam from body)
- Present on **every wave** — count = `max(1, round(swarm_count × spider_swarm_ratio))`, default ratio 0.25 (configurable via `config.json → wave.spider_swarm_ratio`)
- Movement handled by `MovementSystem._move_spider()`; melee skipped in `CombatSystem`
- Web shooting + freeze applied in `BattleScene._update_active()`

### SpiderWeb Projectile (`entities/spider_web.py`)

- Travels at 200 px/s, max lifetime 2 s
- Drawn as a gray-white blob with 4 crossed web-strand lines
- On hit: `target.frozen_timer = freeze_duration` (1.5 s); deals 10 damage
- Managed in `BattleScene.spider_webs` list; cleared on wave advance

### Swarm Enemy (`entities/enemy.py`)

- Red colored rectangle, size 18
- HP: 30, speed scales per wave: `base_speed + wave_index * speed_per_wave` px/s
- Chases nearest alive minion (from combined fighters+archers list), attacks on contact
- Attack: 8 damage, 30px range, 1.0s cooldown
- Displays yellow ring burst (0.15s) when attacking

### Boss (`entities/boss.py`)

- Spawned on every 5th wave (wave numbers 5, 10, 15, …)
- HP: `800 + wave_index × 80`; slow movement toward nearest alive minion
- **Phase 1**: rotating ring of orbiting orbs, glowing eyes, pulsing body
- **Phase 2** (at 50% HP): faster rotation, counter-rotating outer ring, extra orbs, more fireballs
- **Fireball volleys**: 3–5 projectiles fanned every 4s; each `BossFireball` has a particle trail + pulsing glow; on impact creates a `BossExplosion`
- **Swarm spawning**: requests 4–6 swarms near its position every 6s via `events["swarms_to_spawn"]`
- **Death animation**: multi-ring expanding shockwave over 1.8s; `boss_dead` event triggers 500-coin award
- Boss is stored on `wave_system.boss` (separate from the swarm `enemies` list)
- `boss.update(dt, targets)` returns an events dict: `{"swarms_to_spawn": N, "new_explosions": [...], "boss_dead": bool}`

### Projectile (Arrow) (`entities/projectile.py`)

- Moves at 420 px/s, 2s lifetime
- Drawn as brown shaft + arrowhead dot
- Checks `boss` for hit detection in addition to swarm enemies

### Fire Mage Minion (`entities/fire_mage.py`)

- Orange floating robe, glowing eyes, orbiting ember spark; size 20
- HP: 50, speed: 65 px/s; `minion_type = "fire_mage"`
- **Heuristic AI** (not DQN): `update_velocity(enemies, boss)` approaches/retreats/strafes relative to nearest enemy; preferred range 210 px, min safe dist 80 px
- `try_shoot(enemies, boss)` fires a `FireMageFireball` when cooldown ready (2.5s) and a target is in range (260 px); returns the projectile to be added to the scene list
- Fireball explodes on hit: `MageExplosion` deals AoE damage (75 px radius, 22 base) + applies **Burn** status (8 DPS for 3s) to all enemies in blast
- `knockback_vel`, `frozen_timer`, `stamina`/`max_stamina` attributes present for system compatibility
- **Dead state**: tombstone with orange-tinted "FM" label

### Ice Mage Minion (`entities/ice_mage.py`)

- Blue floating robe, counter-rotating diamond crystal; size 20
- HP: 50, speed: 65 px/s; `minion_type = "ice_mage"`
- **Heuristic AI**: same approach/retreat/strafe logic as Fire Mage; preferred range 200 px, min safe dist 80 px
- `try_shoot(enemies, boss)` fires an `IceMageIceball` when cooldown ready (3.0s) and a target is in range (240 px)
- Iceball hit: 15 direct damage + applies **Freeze** status (`frozen_timer = 2.0s`) — frozen enemies cannot move for the duration
- **Dead state**: tombstone with blue-tinted "IM" label

### Mage Projectiles (`entities/mage_projectile.py`)

**FireMageFireball**
- Moves at 310 px/s; particle trail rendered each frame
- `update(dt, enemies, boss)` detects hits (distance ≤ combined radii); returns a `MageExplosion` on contact; marks `is_alive = False`

**IceMageIceball**
- Moves at 290 px/s; cyan particle trail
- On hit: deals 15 damage + sets `target.frozen_timer = max(current, freeze_duration)`; marks `is_alive = False`

**MageExplosion**
- `apply(enemies, boss)`: AoE damage to all enemies within radius; sets `enemy.burn_timer = burn_duration` and `enemy.burn_dps = burn_dps` on each hit enemy
- Animation: 2 expanding rings + orange spark particles over 0.85s

**Status effect draw helpers** (used by Slime and Creeper):
- `draw_freeze_overlay(surface, pos, size, alpha)` — blue crystalline overlay
- `draw_burn_overlay(surface, pos, size, timer, max_timer)` — flickering orange/red overlay; intensity scales with remaining burn time

### Slime Enemy (`entities/slime.py`)

- Rounded blob with bobbing squash-and-stretch animation; `enemy_type = 2`
- **3 generations**: Large (gen 0: 80 HP, size 28, speed 55), Medium (gen 1: 40 HP, size 18, speed 75), Small (gen 2: 20 HP, size 12, speed 100)
- On death: gen 0 → spawns 2× gen 1; gen 1 → spawns 2× gen 2; gen 2 → dies outright
- Split tracked with `_split_done` flag (prevents double-split if death detected twice in same frame)
- Has `frozen_timer`, `burn_timer`, `burn_dps` — renders freeze/burn overlays when active
- Melee attack: range 28 px, damage scales with generation, cooldown 1.2s; melee enabled in CombatSystem (`enemy_type not in (0, 2)` gate changed to allow Slime melee)
- Spawns from wave 3 onward; 1 Large per 8 swarms; boss waves skip Slimes

### Creeper Enemy (`entities/creeper.py`)

- Pixel-art face (square eyes, horizontal mouth bar) with fuse-flash animation; `enemy_type = 4`
- HP: 60, speed: 85 px/s, size: 22
- **Explosion trigger**: `tick(dt, all_minions)` sets `should_explode = True` when:
  - Any alive minion is within `trigger_range` (35 px), OR
  - `hp < self._prev_hp` (took any damage this frame)
- Fuse flash rate: `math.sin(flash_counter * fuse_flash_rate * π)` — face flashes faster as HP drops
- **`CreeperExplosion`** (also in this file): `apply(minions)` deals 55 AoE damage to all alive minions within 100 px; animation: green rings + debris particles over 0.9s
- `BattleScene` detects `creeper.should_explode`, calls `CreeperExplosion(creeper.pos)`, plays SFX, marks creeper dead
- Does not melee (blocked by `enemy_type not in (0, 2)` check in CombatSystem; Creeper is type 4)
- Spawns from wave 5 onward; 1 per 10 swarms; boss waves skip Creepers

### Spell Effects (`entities/spell_effect.py`)

**HealingEffect**
- `apply(targets)` heals all alive minions within `radius` by `heal_amount` (instant)
- Animation: expanding green circle + rising sparkles + cross shimmer over 1.1s

**FireballPending**
- Placed at cursor; shows descending meteor + target reticle for `flight_time` (default 1.0s)
- `update(dt)` returns `True` when ready to detonate
- `detonate(damage, radius)` → returns a `FireballLanding`

**FireballLanding**
- `apply(targets)` deals damage to all alive entities (minions and enemies) within `explosion_radius`
- Animation: inner flash + expanding rings + debris particles over 0.8s

**SummonPortal**
- Created when a summon icon is clicked (replaces instant spawn); color-coded by role: blue (fighter), green (archer), orange-red (fire_mage), cyan (ice_mage)
- Animation: expanding ring, 8 orbiting rune dots, entry/exit flash over `duration` (default 1.2s, configurable in `config.json → summon_portal.duration`)
- `done` flag set `True` when animation completes; `BattleScene` detects this and calls `_complete_spawn(role, pos)` to create the actual entity
- Role and position stored so the right minion type spawns at the portal center

### Arena

- 1280×720, 10px margin on all sides (playable: 10,10 → 1270,710)
- All entity positions clamped to arena bounds

---

## 5. Physics System

### Entity Separation

All entities (minions and enemies) push each other apart when their rectangles overlap. This is handled in `MovementSystem` as a post-movement pass each frame.

**Algorithm:**
- Compare every pair of entities within the same group (minion–minion, enemy–enemy) and across groups (minion–enemy).
- If the distance between centers is less than `separation_radius` (sum of their radii + a small buffer), compute a repulsion vector from the deeper entity to the shallower one.
- Apply a separation impulse scaled by overlap depth: `impulse = separation_force × (separation_radius − dist) / separation_radius`.
- Split the impulse symmetrically between the two entities (each gets half), then clamp positions to arena bounds.

Config keys (`config.json → physics`):
```json
{
  "separation_radius_buffer": 4,
  "separation_force": 300.0
}
```

### Knockback on Hit

When a melee or ranged attack lands, the target receives a knockback impulse directed away from the attacker.

- **Melee (Fighter sword arc):** knockback applied at the moment of hit; direction = `normalize(target.pos − attacker.pos)`.
- **Ranged (Arrow, SpiderWeb, BossFireball):** knockback applied at projectile impact; direction = projectile's normalized velocity vector.
- Knockback decays over `knockback_duration` seconds using exponential falloff: `knockback_vel *= knockback_decay^dt`.
- Frozen minions still receive knockback position displacement (the freeze only disables voluntary movement).
- Knockback velocity is added on top of the entity's normal movement each frame and clamped to `max_knockback_speed`.

Config keys (`config.json → physics`):
```json
{
  "knockback_force": 280.0,
  "knockback_duration": 0.25,
  "knockback_decay": 0.05,
  "max_knockback_speed": 600.0
}
```

Both systems are implemented inside `MovementSystem` and enabled by default. They apply to all entity types (fighters, archers, swarms, spiders, and the boss receives a scaled-down knockback).

---

## 6. AI Minion Brain System

### Brain Architecture — CNN + Rainbow DQN (V1)

**MobileNetV4-inspired CNN** followed by a **Noisy Dueling Distributional (C51)** head.

Input is **4 stacked 84×84 grayscale frames** (image crop of arena height × arena height, centered on the minion, zero-padded at edges, then bilinear-resized to 84×84).

```
Input [B, 4, 84, 84]
  → Conv(4→32, k=3, stride=2) + ReLU                [B, 32, 42, 42]
  → DepthwiseSeparableConv(32→64,   stride=2)        [B, 64, 21, 21]
  → DepthwiseSeparableConv(64→128,  stride=2)        [B,128, 10, 10]
  → DepthwiseSeparableConv(128→256, stride=2)        [B,256,  5,  5]
  → GlobalAvgPool                                    [B, 256]
  → Value stream:     NoisyLinear(256→64) → ReLU → NoisyLinear(64→51)   → [B, 1, 51]
  → Advantage stream: NoisyLinear(256→64) → ReLU → NoisyLinear(64→A×51) → [B, A, 51]
  → Q = V + (A − mean(A))   [B, A, 51 atoms]
  → log_softmax(dim=-1)     → log-probability distribution
```

`DepthwiseSeparableConv`: depthwise conv (groups=C) → pointwise conv 1×1 → ReLU (no BatchNorm to avoid train/eval inconsistency issues with small batches).

CNN layers and kernel size are configurable in `config.json → cnn.channels / cnn.kernel_size`.
Number of stacked frames and final size configurable in `config.json → image_obs`.

`NoisyLinear` uses **factorized Gaussian noise**: `weight_mu/sigma`, `bias_mu/sigma` params; `weight_eps/bias_eps` buffers resampled every `reset_noise()` call. The noise function is `sign(x) * sqrt(|x|)`.

The legacy **Transformer encoder + BrainNetwork** (vector obs) is retained in `brain.py` for reference and can be re-enabled via `obs_type="vector"` in `DQNAgent`.

Default CNN hyperparameters:

| Param | Value | Config key |
|---|---|---|
| `channels` | [32, 64, 128, 256] | `cnn.channels` |
| `kernel_size` | 3 | `cnn.kernel_size` |
| `frame_size` | 84 | `image_obs.frame_size` |
| `n_frames` | 4 | `image_obs.n_frames` |
| `n_atoms` | 51 | `rainbow.n_atoms` |
| `v_min` | -30 | `rainbow.v_min` |
| `v_max` | 80 | `rainbow.v_max` |
| `noisy_sigma` | 0.5 | `rainbow.noisy_sigma` |

### Observation Space — Image (4 × 84 × 84)

The primary observation for the DQN is a stack of 4 grayscale 84×84 frames:

```
Shape: (4, 84, 84)  float32, values in [0, 1]

Each frame is a top-down arena-crop centered on the observing minion:
  1. Render full arena (ARENA_HEIGHT × ARENA_WIDTH) as a float32 grayscale array.
     Gray values per entity type:
       Fighter / Archer (ally)  0.40
       Self (the observing minion)  0.60  (overwritten in crop)
       Swarm enemy              0.80
       Spider enemy             0.87
       Boss                     0.95
       Arrow projectile         0.55
       Background               0.00
  2. Crop (ARENA_HEIGHT × ARENA_HEIGHT) centered on minion; zero-pad where outside bounds.
  3. Bilinear-resize to (84 × 84).
  4. Stack the 4 most-recent frames — frame 0 is oldest, frame 3 is newest.
```

Frame-stacking gives the network motion information (enemy velocity, approach direction).
The frame buffer is initialised with zero frames at battle start and fills over the first 4 ticks.

### Observation Space — Vector (43-dim, for preset heuristic only)

The preset policy (used during warmup) still runs on the 43-dim vector observation
(`MinionEnv.get_vector_observation()`):

```
[0]     self_hp_norm
[1]     self_x_norm              (pos relative to arena left / ARENA_WIDTH)
[2]     self_y_norm              (pos relative to arena top  / ARENA_HEIGHT)
[3]     self_stamina_norm
[4]     self_attack_damage_norm
[5]     self_attack_speed_norm
[6–10]  ally hp, x, y, stamina, is_alive
[11+i*6 .. 11+i*6+5]  i-th nearest enemy token (i = 0..4):
  type_norm, dx_norm, dy_norm, hp_norm, vx_norm, vy_norm
[41]    ally_last_action_norm  (ally's last action index / (ally_n_actions − 1); 0 if absent)
[42]    ally_is_attacking      (1.0 if ally's last_action ≥ 8, i.e. an attack action)
```

Each minion stores `last_action: int` (set to 0 at spawn); updated every step in `BattleScene`
immediately after `select_action()` is called. Ally action space: Archer has 24 actions,
Fighter has 16 — the correct divisor is chosen per env role.

### Action Space

**Fighter — 16 actions (unchanged)**

| Index | Type   | Direction |
|---|---|---|
| 0–7  | Move   | 8 directions at 45° intervals |
| 8–15 | Attack | 8 directions at 45° intervals |

**Archer — 24 actions (extended for precision aiming)**

| Index | Type   | Direction |
|---|---|---|
| 0–7   | Move   | 8 directions at 45° intervals |
| 8–23  | Attack | 16 directions at 22.5° intervals |

Archer attack direction granularity doubled (45° → 22.5°) so the agent can precisely lead fast-moving targets when velocity-lead-correcting its aim.
The aim-snap helper in `BattleScene._archer_aim_snap()` still corrects the chosen direction angle to the nearest in-range enemy.

---

## 7. Reinforcement Learning Integration

### Training Methods

Two distinct training methods run concurrently, both using the same PER replay buffer:

| Training Path | Method | Trigger | Key Difference |
|---|---|---|---|
| **In-game** | Rainbow DQN (default) or Expected SARSA | Every `train_interval` frames (default 10, background thread) | Configurable via `config.json → training.ingame_mode`; throttled for smooth gameplay |
| **Memory Replay** | Rainbow DQN (Double DQN) | ResearchLab UI | Target = max_a Q(s') via policy+target net (greedy) |

**Expected SARSA (in-game):**
```
π(a | s') = softmax(Q_target(s') / temperature)        [configurable: expected_sarsa.softmax_temperature]
Z_expected(s') = Σ_a π(a|s') · Z_target(s', a)        [mixture distribution over atoms]
target_dist    = C51-project(r + γⁿ · Z_expected(s'))
loss           = cross-entropy(target_dist, log Z_policy(s, a)) · IS_weight
```

**Modern Rainbow DQN (Memory Replay):**
```
next_act    = argmax_a E[Z_policy(s', a)]             [Double DQN action selection]
target_dist = C51-project(r + γⁿ · Z_target(s', next_act))
loss        = cross-entropy(target_dist, log Z_policy(s, a)) · IS_weight
```

### Active Rainbow Components

| Component | Status |
|---|---|
| Double DQN | ✅ Policy net selects actions; target net evaluates (Memory Replay) |
| Expected SARSA | ✅ Softmax expected-value target (in-game training) |
| Dueling Networks | ✅ Value + Advantage head |
| Prioritized Experience Replay (PER) | ✅ SumTree with IS weights |
| N-step returns | ✅ N=3 |
| Distributional (C51) | ✅ 51 atoms, v_min=-30, v_max=80 |
| Noisy Networks | ✅ Factorized noise replaces ε-greedy |

### DQN Agent (`ai/dqn.py`)

```python
class SumTree:
    # Binary sum tree for O(log n) PER sampling/update
    def add(self, priority, data): ...
    def update(self, idx, priority): ...
    def get(self, s) -> (tree_idx, priority, data): ...
    @property
    def total(self) -> float: ...

class DQNAgent:
    # role: "fighter" or "archer"
    # Replay buffer: SumTree PER, configurable size (default 10 000)
    # N-step buffer: deque of length N=3; flushed to PER on done or overflow
    # No epsilon — exploration via NoisyNet
    # Target net updated every step via Polyak averaging (soft_update_tau, default 0.005)

    def select_action(self, obs) -> int:
        # During warmup (buffer < warmup_preset_ratio * buffer_size):
        #   use role-specific preset heuristic
        # After warmup: greedy argmax over expected Q = sum(logprob.exp() * support)

    def store_transition(self, obs, action, reward, next_obs, done):
        # Accumulates into _n_buf; pushes full N-step transitions to PER
        # On done: flushes remaining transitions with shorter horizons

    def train_step(self) -> dict:
        # PER sample → Double DQN target via _project_distribution (C51 Bellman)
        # Cross-entropy loss weighted by IS weights
        # Priority update via TD error
        # Soft (Polyak) target update every step: θ_target ← τ·θ + (1-τ)·θ_target
        # Returns {"loss": float}

    def save_checkpoint(self, path): ...   # model weights + optimizer only (no buffer)
    def save_buffer_session(self, folder: str, session_idx: int): ...  # top-N by priority → folder/session_{idx:04d}.pt
    def load_checkpoint(self, path): ...   # try/except for architecture mismatch; backward-compat loads embedded buffer from old saves
    def load_buffer_sessions(self, folder: str): ...  # loads all session_*.pt files, inserts at max priority
```

### C51 Bellman Projection

```python
def _project_distribution(self, next_dist, rewards, dones):
    Tz = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * gamma_n * support
    Tz = Tz.clamp(v_min, v_max)
    b  = (Tz - v_min) / delta_z
    lo, hi = b.floor().long(), b.ceil().long()
    target.scatter_add(...lo..., next_dist * (hi.float() - b))
    target.scatter_add(...hi..., next_dist * (b - lo.float()))
    return target
```

### Agent Ownership

Both `DQNAgent` instances live on `GameManager`:
- `game_manager.fighter_agent` — shared by all deployed fighters
- `game_manager.archer_agent`  — shared by all deployed archers

`GameManager.init_agents()` creates agents and loads checkpoints + session buffer snapshots if a save exists. This is called by both `new_game()` and `load_save()`.

### Preset Policy (Warmup Exploration Fallback)

During warmup, agents use role-specific heuristic policies instead of pure noise:

**Fighter preset:** flee if enemies within min safe dist (blended with wall repulsion to escape corners) → attack nearest if in ideal range and stamina OK → back off if in range but stamina low → approach nearest (blended with wall push). Falls back to center-seeking movement when no enemies are visible.

**Archer preset (priority order):**
1. Panic-flee when any enemy is within min safe distance (blended with wall repulsion).
2. **Shoot** with velocity-lead correction if nearest enemy is in range and stamina > 25% — this fires even when mild repulsion is active, fixing the previous bug where archers never shot enemies within 150 px.
3. Approach if out of shoot range.
4. Back off to preferred range if too close but no stamina.

Wall repulsion helper `_wall_repulsion(play_x, play_y)` returns a push vector whenever the entity is within `wall_safe_dist` (80 px) of any arena edge; this vector is blended into flee/approach directions to prevent minions from getting stuck in corners.

### Reward Function (role-based, fully configurable via `config.json → rewards`)

All reward magnitudes are read from `config.json → rewards` at runtime — edit to retune without code changes.

**Fighter — individual component:**

| Event | Default Reward | Config key |
|---|---|---|
| Melee damage dealt | +1 × damage | `rewards.fighter_damage_scale` |
| Damage taken | −1 × damage | `rewards.fighter_damage_taken_scale` |
| Melee kill | +5 | `rewards.fighter.melee_kill_bonus` |
| Swing misses (0 enemies hit) | −2 | `rewards.fighter_miss_penalty` |
| Within `close_range` (100 px) of nearest enemy | +0.02/frame | `rewards.fighter.close_bonus` |
| Beyond `far_range` (200 px) from nearest enemy | −0.01/frame | `rewards.fighter.far_penalty` |
| Death | −10 | `rewards.death_penalty` |
| Wave cleared | 0 | `rewards.wave_cleared_bonus` |

**Archer — individual component:**

| Event | Default Reward | Config key |
|---|---|---|
| Arrow damage dealt | +1 × damage | `rewards.archer_damage_scale` |
| Damage taken | −1 × damage | `rewards.archer_damage_taken_scale` |
| Enemy killed by arrow | +5 | `rewards.archer.ranged_kill_bonus` |
| Shot with no enemy in cone | 0 (disabled) | `rewards.archer_miss_penalty` |
| Arrow expires without hitting | −2 | `rewards.archer_arrow_expired_penalty` |
| Within `ideal_range_min`–`ideal_range_max` (180–270 px) | +0.02/frame | `rewards.archer.ideal_range_bonus` |
| Beyond `too_far_range` (320 px) or within `too_close_range` (100 px) | −0.01/frame | `rewards.archer.bad_range_penalty` |
| Death | −10 | `rewards.death_penalty` |
| Wave cleared | 0 | `rewards.wave_cleared_bonus` |

**Team component** (added to both roles, scaled by `rewards.team_reward_weight` = 0.4; set to 0 for purely individual behaviour):

| Event | Fighter bonus | Archer bonus | Config key |
|---|---|---|---|
| Ally ranged (archer) kill | +`fighter.ranged_kill_bonus` (2.0) per kill | — | `rewards.fighter.ranged_kill_bonus` |
| Ally melee (fighter) kill | — | +`archer.melee_kill_bonus` (2.0) per kill | `rewards.archer.melee_kill_bonus` |
| Ally dies this step | −`team_ally_death_penalty` (5.0) | −`team_ally_death_penalty` (5.0) | `rewards.team_ally_death_penalty` |
| Per alive ally HP norm | +`team_ally_hp_bonus_scale` (0.002) × Σ(hp/max_hp) | same | `rewards.team_ally_hp_bonus_scale` |

**Spread / anti-stacking penalty** (both roles):

| Condition | Penalty | Config keys |
|---|---|---|
| Any ally within `ideal_spread_min` (100 px) | −`spread_penalty_scale` × overlap_fraction per ally | `rewards.ideal_spread_min`, `rewards.spread_penalty_scale` |

**Design notes:**
- `wave_cleared_bonus` is 0 because the AI Master can summon multiple minions during a wave; a per-wave reward would be diluted and misleading when minion counts vary.
- `archer_miss_penalty` (no enemy in cone) is 0 because archers legitimately lead shots at predicted positions; penalising shots into empty space discourages correct predictive aiming.
- `archer_arrow_expired_penalty` fires when a `Projectile` expires (`lifetime` elapsed) without setting `hit_enemy=True`. This penalises genuinely wasted shots while leaving predictive shots unpunished.
- Kill bonus differentiation: fighter gains more from its own melee kills (5.0) than from team ranged kills (2.0 × weight), and vice versa for archer. This shapes role specialisation without requiring a separate learning objective.
- Spread penalty fires per ally within threshold; scales linearly with overlap. Deters all minions stacking on one target where AoE (Creeper, Boss fireball) would wipe them simultaneously.

### Memory Replay Training

From the ResearchLab AI Master tab, the player can run off-policy training on the accumulated buffer:
- Player specifies iterations (10–5000, adjustable in ±50 steps)
- Cost = `max(1, round(iterations × 0.1))` coins — 10 coins per 1 000 iterations (configurable via `config.json → memory_replay.cost_per_iteration`)
- Launches a daemon thread that calls `agent.train_step()` repeatedly
- A **live progress bar** (0–100%) is shown while training runs
- After training, **only model checkpoints are saved** (fast); replay buffer is not re-serialized here
- When save completes, result is shown: `F  loss:X.XXXX  avg rew:X.XXX` / `A  loss:X.XXXX  avg rew:X.XXX`

### Session-Based Replay Buffer Storage

The replay buffer is persisted as a **folder of per-session snapshot files** — one file written at game-over per role:

```
data/saves/
├── {name}.json                      ← includes fighter_session_idx, archer_session_idx
├── {name}_fighter.pt                ← model checkpoint
├── {name}_archer.pt
├── {name}_fighter_buffer/           ← session snapshot folder
│   ├── session_0001.pt              ← oldest retained session
│   ├── session_0002.pt
│   └── session_0003.pt              ← most recent session (just written)
└── {name}_archer_buffer/
    └── ...
```

**Saving (game-over only):**
- `DQNAgent.save_buffer_session(folder, session_idx)` — extracts the top `max_session_transitions` (default 10,000) transitions by PER priority from the in-memory buffer, serializes as `session_{idx:04d}.pt`
- If the folder already contains `max_buffer_sessions` (default 10) files, the oldest is deleted before writing the new one
- `session_idx` is read from the save JSON (`fighter_session_idx` / `archer_session_idx`) and incremented after each write
- No buffer save occurs after individual waves or after Memory Replay — only model checkpoints are saved then

**Loading (game load):**
- `DQNAgent.load_buffer_sessions(folder)` — scans the folder for all `session_*.pt` files, sorts ascending, inserts each into the in-memory PER buffer in order (oldest first)
- Inserted with **max priority** (stale priorities discarded; priorities re-ranked during training)
- When total loaded transitions exceed the in-memory buffer capacity, the ring buffer naturally evicts the oldest, preserving the most recent sessions

**Config keys (`config.json → memory_buffer`):**
```json
{
  "max_session_transitions": 10000,
  "max_buffer_sessions": 10
}
```

**DQNAgent interface additions:**
```python
def save_buffer_session(self, folder: str, session_idx: int): ...
    # Extracts top min(max_session_transitions, buffer_size) transitions by PER priority,
    # writes to folder/session_{idx:04d}.pt, deletes oldest if folder > max_buffer_sessions

def load_buffer_sessions(self, folder: str): ...
    # Reads all session_*.pt files in sorted order, inserts into PER at max priority
```

`GameManager.buffer_folder(name, role)` returns the folder path. `init_agents()` calls `load_buffer_sessions` after `load_checkpoint`.

### Memory Storage Frame Skip

Transitions are **not** stored every frame. Each `MinionEnv` maintains a `frame_counter` that increments each game frame; a transition is pushed to the replay buffer only when `frame_counter % memory_store_interval == 0`.

- Default `memory_store_interval`: **10** frames (= 6 transitions/second/minion at 60 FPS)
- Configurable in `config.json → training.memory_store_interval`
- Reduces temporal correlation between stored transitions significantly
- All minions of the same type still write into the **same shared** `DQNAgent` buffer, so with N active minions the effective collection rate is `N × 6` transitions/second
- The `frame_counter` resets to 0 when a minion dies or a new wave starts

### Training Threads (Live Battle)

Two independent `TrainingSystem` instances run in parallel background threads — one for the fighter brain, one for the archer brain. `select_action` runs on the main thread; `train_step` runs on the background thread.

### MinionEnv (`ai/minion_env.py`)

```python
class MinionEnv:
    def __init__(self, minion, enemies, ally=None, role="fighter"): ...
    def get_observation(self) -> np.ndarray: ...  # 26-dim
    def get_reward(self, combat_events: dict) -> float: ...
    def is_done(self) -> bool: ...
    # frame_counter: int — increments each frame; transition stored when counter % memory_store_interval == 0
```

`MinionEnv` holds references to the live `enemies` list and `ally` entity. On wave reset, update `minion_env.minion`, `minion_env.enemies`, `minion_env.ally`.

---

## 8. Wave & Enemy System

### Wave System States

`INTERMISSION → SPAWNING → ACTIVE → (INTERMISSION or VICTORY)`, `GAME_OVER` only when **all** deployed fighters and archers are dead.

### Wave Count Formula

- **Waves 1–10**: counts from `config.json → wave.first_ten_counts` (e.g., `[3,5,7,9,12,14,16,18,19,20]`)
- **Waves 11–100**: `min(60, 22 + (wave_index - 10) * 2)` swarm enemies
- **Boss waves** (wave_index+1 divisible by 5): `boss_wave_swarm_count` swarms + 1 Boss (no Slimes or Creepers)
- **Spiders**: every wave, `max(1, round(swarm_count × 0.25))` spiders
- **Slimes**: from wave 3, `max(1, swarm_count // 8)` Large Slimes per wave (gen 0)
- **Creepers**: from wave 5, `max(1, swarm_count // 10)` Creepers per wave
- Spawn delay: 0.3s between enemies. Intermission: 3s with live countdown.

### Boss Wave Behavior

- Boss is stored on `wave_system.boss`, not in `enemies` list
- Wave clear requires: all swarms dead AND boss dead (no death animation running)
- `wave_system.spawn_swarms_near_boss(enemies, count)` called by BattleScene when boss requests swarms
- Boss coin reward: 500 coins, awarded on boss death transition

### Wave Clear Logic

```python
wave_cleared = (swarms_alive == 0 and boss_done and
                (len(enemies) > 0 or self.boss is not None))
```

`boss_done` is `True` if `self.boss is None` or `(not boss.is_alive and not boss._dying)`.

### Minion Persistence

Minion HP and alive status **carry over** between waves — dead minions stay dead, HP is not restored on wave advance. Game-over triggers when all deployed minions are dead.

### Dynamic In-Wave Spawning

During an active wave (state `ACTIVE`), the player can spend **AI Master MP** to summon additional minions via the **Summon Fighter** and **Summon Archer** spell icons in the bottom-centre HUD panel:

- **Summon Fighter** icon: costs `config.json → spells.summon_fighter.mp_cost` MP (default 50)
- **Summon Archer** icon: costs `config.json → spells.summon_archer.mp_cost` MP (default 40)
- Total minions across **both types combined** is capped by a single global cap from `config.json → spawning.deployment_caps_global` (indexed by Deploy Limit upgrade level; default at level 0 = 2, fully upgraded at level 5 = 20)
- Players may allocate the global pool freely — e.g., 20 Fighters and 0 Archers, or 10 of each, as long as total ≤ global cap
- Newly spawned minions are placed at a random safe position near the arena edge
- They immediately join the active lists and start feeding transitions into the **same shared `DQNAgent`** as existing minions of their type
- Summon icons are shown in the bottom-centre spell panel alongside Heal and Fireball; greyed out when global cap is reached, wave not active, or insufficient MP. Badge shows `F:{n}  total/cap` or `A:{n}  total/cap` so the shared pool is always visible.
- In-wave spawned minions count toward game-over (all minions dead = game over) and wave-clear tracking
- `BattleScene._try_spawn_minion(role)` handles MP deduction, position selection, entity creation, `MinionEnv` creation, and list registration

Config keys (`config.json → spawning`):
```json
{
  "deployment_caps_global": [2, 5, 8, 12, 16, 20]
}
```

Config keys (`config.json → spells`):
```json
{
  "summon_fighter":    { "mp_cost": 50 },
  "summon_archer":     { "mp_cost": 40 },
  "summon_fire_mage":  { "mp_cost": 60 },
  "summon_ice_mage":   { "mp_cost": 55 }
}
```

---

## 9. Economy

| Currency | Source | Spent On |
|---|---|---|
| **Coins** (⚙) | +10/kill, +50/wave, +500/boss kill | Upgrades in Research Lab, Memory Replay Training |
| **MP** | Regenerates over time (upgradeable) | Casting spells (Heal, Fireball, Summon Fighter, Summon Archer) |

Research Lab upgrade cost: 50/100/150/200/250 coins per level (max 5 levels per stat).

AI Master upgrades have their own cost schedule defined in `_AI_MASTER_ROWS` in `research_lab.py`.

---

## 10. UI & HUD

### HUD Layout

```
┌──────────────────────────────────────────────────────────────┐
│ ★ BOSS WAVE ★ X/100        [P] Pause  [R] Brain  [+/-] Nx   │
│                              [ESC] Menu        Name | Coins   │
│                                                               │
│  [Battle arena]                                               │
│                                                               │
│ Rainbow Training                                              │
│   Fighter Brain:              ████████ MP XX/XX ████████     │
│     Loss: X.XXX  Steps: XXXXX  [Heal][Fire][SumF][SumA]      │
│   Archer Brain:                               Fighters stacked│
│     Loss: X.XXX  Steps: XXXXX               Archers stacked  │
│   Speed: Nx                                 Boss HP (if boss) │
│                                              Enemies: X       │
└──────────────────────────────────────────────────────────────┘
```

### HUD Details

- **Top-left**: "Wave X/100" white; boss waves show "★ BOSS WAVE X/100 ★" in red
- **Top-center**: control strip — `[P] Pause`, `[R] Reset Brain`, `[+/-] Speed: Nx`, `[ESC] Menu`
- **Top-right**: Player name + "⚙ XXXX"
- **Bottom-left**: "Fighter/Archer Brain" panel — mode (DQN / Warmup / Preset+Train / Preset+Warmup), loss, **avg reward EMA**, steps, buffer fill (count/max + %), LR, PER β, Speed; "Saving..." line appended while an async checkpoint save is in progress after a wave ends
- **Bottom-center**: Two-row spell panel — **Row 1**: MP bar (fills left to right, shows `MP XX/XX`); **Row 2**: six spell icons `[Heal] [Fireball] [Summon F] [Summon A] [Summon FM] [Summon IM]` — each shows MP cost at top, name/count-badge at bottom, cooldown overlay when on cooldown; summon icons show combined badge `total/global_cap`; icons greyed when unavailable (global cap reached, wave inactive, or insufficient MP)
- **Bottom-right**: all fighters stacked (HP + stamina bars), all archers stacked, enemy count, boss HP bar if boss alive
- **Spell hint**: instruction text drawn in center when a placement spell (Heal/Fireball) is selected
- **Center overlays**: INTERMISSION countdown, GAME OVER (red-tinted), VICTORY ("All 100 waves cleared!", gold-tinted)

### Spell System

`hit_test_spell_panel(pos)` on HUD returns `"healing"`, `"fireball"`, `"summon_fighter"`, `"summon_archer"`, `"summon_fire_mage"`, `"summon_ice_mage"`, or `None`.

In BattleScene:
- Click any `summon_*` icon → calls `_try_spawn_minion(role)`: deducts MP, chooses spawn position, creates a `SummonPortal`, plays `summon_portal` SFX; actual entity created via `_complete_spawn(role, pos)` when portal animation finishes
- Click `healing` / `fireball` icon → `_activate_spell(name)` → toggles `self.spell_mode`; checks MP and cooldown
- Click arena with spell active → `_cast_spell(pos)`:
  - **Healing**: instantiates `HealingEffect`, applies heal, starts cooldown
  - **Fireball**: instantiates `FireballPending`, stored in `spell_effects`; detonation detected in update loop
- Right-click cancels placement spell mode
- MP regenerates at `_mp_regen` per second (capped at `max_mp`)

### MP System

- `self.mp`, `self.max_mp`, `self._mp_regen` computed from `ai_master` upgrade levels
- MP bar at bottom-center row 1, fills left to right
- Heal costs 30 MP, Fireball costs 50 MP, Summon Fighter 50 MP, Summon Archer 40 MP, Summon Fire Mage 60 MP, Summon Ice Mage 55 MP — all configurable in `config.json → spells`

### Controls

| Key / Click | Action |
|---|---|
| P / click `[P]` | Pause/unpause |
| R / click `[R]` | Reset brain (weights + buffer), shows "BRAIN RESET" for 2s |
| +/- / click `[+/-]` | Speed multiplier 1×–5× |
| ESC / click `[ESC]` | Return to main menu |
| Click Heal icon | Enter healing placement mode |
| Click Fireball icon | Enter fireball placement mode |
| Click Summon F icon | Begin portal animation → spawn Fighter when portal completes (costs MP, active wave only) |
| Click Summon A icon | Begin portal animation → spawn Archer when portal completes (costs MP, active wave only) |
| Click Summon FM icon | Begin portal animation → spawn Fire Mage when portal completes (costs MP, active wave only) |
| Click Summon IM icon | Begin portal animation → spawn Ice Mage when portal completes (costs MP, active wave only) |
| Right-click (during spell mode) | Cancel placement spell |
| Click arena (spell mode active) | Cast placement spell at location |

---

## 11. V1 Scope & File Structure

### V1 File Structure

```
ai_master/
├── main.py                  # Entry point + game loop
├── requirements.txt
├── config.json              # Central parameter file — edit to tune game + AI
├── config.py                # Loads config.json → CFG dict (imported project-wide)
├── VISION.md
├── CURRENT.md
├── engine/
│   ├── game_manager.py      # Scene stack, global state, save/load, DQNAgent ownership
│   └── scene.py             # BaseScene (abc.ABC + abstractmethod)
├── scenes/
│   ├── main_menu.py         # Start New Game / Load Saved Game / name input / override confirm
│   ├── loading.py           # Animated loading screen; runs new_game/load_save in background thread
│   ├── research_lab.py      # 3-tab upgrade screen: Fighter / Archer / AI Master + Memory Replay
│   ├── training_setup.py    # Pre-battle training config: mode, LR, warmup ratio, buffer, batch size
│   └── battle.py            # Core battle scene (multi-minion, spells, boss, Rainbow DQN)
├── entities/
│   ├── minion.py            # Fighter minion (stamina + sword arc + "F" label; frozen_timer)
│   ├── archer.py            # Archer minion (ranged, stamina, "A" label; frozen_timer)
│   ├── fire_mage.py         # Fire Mage minion (heuristic AI; fireball AoE + Burn status)
│   ├── ice_mage.py          # Ice Mage minion (heuristic AI; iceball + Freeze status)
│   ├── mage_projectile.py   # FireMageFireball, IceMageIceball, MageExplosion; freeze/burn overlay helpers
│   ├── enemy.py             # Swarm enemy (attack flash; enemy_type=0)
│   ├── spider.py            # Spider enemy (ranged, web shot, animated legs; enemy_type=1)
│   ├── spider_web.py        # SpiderWeb projectile (slow, applies freeze on hit)
│   ├── slime.py             # Slime enemy (3-generation split on death; enemy_type=2)
│   ├── creeper.py           # Creeper enemy + CreeperExplosion (proximity/damage explosion; enemy_type=4)
│   ├── boss.py              # Boss entity + BossFireball + BossExplosion
│   ├── spell_effect.py      # HealingEffect, FireballPending, FireballLanding, SummonPortal
│   └── projectile.py        # Arrow projectile
├── ai/
│   ├── brain.py             # BrainNetwork: Transformer encoder + NoisyLinear Dueling C51 head
│   ├── minion_env.py        # MinionEnv: 26-dim observation + role-based reward
│   └── dqn.py               # DQNAgent: full Rainbow (PER SumTree, N-step, C51, NoisyNet, Double+Dueling)
├── systems/
│   ├── combat_system.py     # Cone attack, stamina, projectiles, archer, boss damage
│   ├── movement_system.py   # Entity movement + boundaries (accepts lists of minions + boss)
│   ├── wave_system.py       # 100 waves, boss every 5th, spawn formulas
│   └── training_system.py   # Background RL training (ThreadPoolExecutor)
├── audio/
│   ├── __init__.py
│   └── sfx_manager.py       # Procedural SFX (numpy + sndarray)
├── ui/
│   └── hud.py               # Full HUD: wave banner, MP bar, spell icons, boss HP, all minions
└── data/
    └── saves/               # JSON save files ({name}.json) + brain checkpoints ({name}_{role}.pt)
                             # + per-role buffer folders ({name}_{role}_buffer/) containing session_000N.pt snapshots
```

### Systems Signatures (Updated)

Both `MovementSystem` and `CombatSystem` now accept **lists** of minions:

```python
# movement_system.py
def update(dt, fighters: list, archers: list, enemies: list, arena_bounds, boss=None)
# Also runs _apply_separation(all_entities) and _apply_knockback_decay(all_entities, dt)

# combat_system.py
def update(dt, fighters, enemies, projectiles, archers, fighter_attack_dirs, boss=None)
# Returns events dict including boss_damage_dealt, boss_killed
# Knockback impulse applied to hit target inside _apply_melee_hit() and _apply_projectile_hit()
```

### Deployment & In-Wave Spawning System

- `game_manager.save["ai_master"]["deployment"]` — Deploy Limit upgrade level (0–5)
- `_compute_deployment(am)` in `battle.py` → always `(1, 1)` — starting deployment is always 1 Fighter + 1 Archer
- `_compute_spawn_cap(am)` in `battle.py` → single `int` global cap indexed from `config.json → spawning.deployment_caps_global`; stored as `scene.spawn_cap_total`
- Deploy Limit upgrade **increases the global total cap**, not a per-type cap
- Global cap schedule (level → total): 0→2, 1→5, 2→8, 3→12, 4→16, 5→20
- All fighters share `game_manager.fighter_agent`; all archers share `game_manager.archer_agent`
- `MinionEnv.ally` = first alive minion of opposite type
- `BattleScene._try_spawn_minion(role)` — checks `len(fighters) + len(archers) + len(fire_mages) + len(ice_mages) < self.spawn_cap_total`; deducts MP, creates `SummonPortal`; `_complete_spawn(role, pos)` called when portal `done` flag is set
- HUD reads `scene.spawn_cap_total` (falls back to last entry of `spawning.deployment_caps_global` from config)

---

## 12. Technical Dependencies

```
# requirements.txt
pygame-ce>=2.4.0          # Game engine
torch>=2.2.0              # Neural networks + RL (Transformer requires torch>=2.0)
numpy>=1.26.0             # Array operations
```

> **Note:** `from __future__ import annotations` is required at the top of any file using `X | Y` or `list[T]` type hints at runtime (wave_system.py, battle.py, hud.py). This ensures Python 3.9 compatibility.

---

## 13. Last Implementation Notes

> **Version 1 RL reward & observation update (completed):**
>
> **1 — Team Reward Component**
> - `MinionEnv.get_reward()` now includes a team component scaled by `rewards.team_reward_weight` (default 0.4; set 0 to reproduce old individual-only behaviour)
> - Team component per env: ally-kill bonus (role-differentiated), ally-death penalty, per-frame HP-norm sum bonus
> - Fighter benefits from archer kills at `rewards.fighter.ranged_kill_bonus`; archer benefits from fighter kills at `rewards.archer.melee_kill_bonus`
> - `BattleScene._update_active()` injects `ally_ranged_kills_this_step`, `ally_melee_kills_this_step`, `ally_deaths_this_step` into the events dict (snapshot of alive fighters+archers taken at step start; deaths counted after all combat resolves)
> - `MinionEnv` receives `fighters_ref` and `archers_ref` constructor args (live list references from `BattleScene`) for HP-bonus and spread-penalty calculations
>
> **2 — Positional Pressure Rewards**
> - Fighter: +`rewards.fighter.close_bonus` per frame when within `close_range` (100 px) of nearest enemy; −`rewards.fighter.far_penalty` when beyond `far_range` (200 px)
> - Archer: +`rewards.archer.ideal_range_bonus` per frame when 180–270 px from nearest enemy; −`rewards.archer.bad_range_penalty` when < 100 px or > 320 px
> - Kill bonuses differentiated by method: old `fighter_kill_bonus` / `archer_kill_bonus` flat keys removed; replaced by `rewards.fighter.melee_kill_bonus` (5.0) and `rewards.archer.ranged_kill_bonus` (5.0) for own kills, with smaller cross-role bonuses in the team component
> - All thresholds and magnitudes live in `config.json → rewards.fighter` and `rewards.archer`
>
> **3 — Ally Last Action in Observation**
> - Vector obs expanded from **41-dim → 43-dim** (OBS_DIM constant updated)
> - New dims [41] `ally_last_action_norm` and [42] `ally_is_attacking` appended after the 5 enemy tokens
> - Fighter's ally (Archer) has 24 actions → normalise by 23; Archer's ally (Fighter) has 16 → normalise by 15
> - `Minion` and `Archer` entities gain `last_action: int = 0`; `BattleScene._update_active()` writes it immediately after `select_action()` returns, before movement/combat
>
> **4 — Spread / Anti-Stacking Penalty**
> - Per-frame penalty in `get_reward()` for each ally within `rewards.ideal_spread_min` (100 px); magnitude = `(threshold − dist) / threshold × spread_penalty_scale` (0.05)
> - Uses `fighters_ref` + `archers_ref` live list references; fires for both fighter and archer envs
> - Config keys: `rewards.ideal_spread_min`, `rewards.spread_penalty_scale`
>
> **Modified files**: `ai/minion_env.py`, `entities/minion.py`, `entities/archer.py`, `scenes/battle.py`, `config.json`
> **Known issues / design notes**:
> - Mage enemies are not DQN-trained; they do not contribute to fighter/archer replay buffers
> - `MinionEnv._N_ENEMY_TYPES = 4` left unchanged; Creeper (type 4) normalizes to 4/4 = 1.0, slightly above the old max — does not affect network architecture
> - Slime split is handled in `BattleScene._update_active()` by checking `_split_done` flag; splitting injects new Slime objects directly into `self.enemies`
