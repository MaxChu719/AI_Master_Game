from __future__ import annotations
"""
BattleScene — core game loop.

Supports:
  - Multi-minion deployment (controlled by ai_master.deployment level in save data).
  - Shared DQN agents (from GameManager) — all fighters share fighter_agent brain.
  - Boss waves every 5 waves (wave system handles spawning; BattleScene updates boss).
  - MP system: regenerates over time; used to cast Healing and Fireball spells.
  - Spell selection: click spell icon → cursor shows radius → click arena to cast.
  - Memory replay training: triggered from ResearchLab; handled via GameManager agents.
  - Full Rainbow DQN (agents on GameManager).
  - Dynamic in-wave minion spawning: player spends coins during active wave to call
    reinforcements; new minions share the existing DQNAgent and replay buffer.
  - Memory store interval: transitions stored every N frames (default 10) to reduce
    temporal correlation in the replay buffer.
"""
import math
import random
import pygame
from engine.scene import BaseScene
from entities.minion        import Minion
from entities.archer        import Archer, ARCHER_MISS_ANGLE
from entities.fire_mage     import FireMage
from entities.ice_mage      import IceMage
from entities.projectile    import Projectile
from entities.spider_web    import SpiderWeb
from entities.mage_projectile import FireMageFireball, IceMageIceball, MageExplosion
from entities.slime         import Slime
from entities.creeper       import Creeper, CreeperExplosion
from entities.spell_effect  import HealingEffect, FireballPending, FireballLanding, SummonPortal
from systems.movement_system import MovementSystem
from systems.combat_system   import CombatSystem
from systems.wave_system      import WaveSystem, WaveState
from systems.training_system  import TrainingSystem
from ai.dqn import (
    FIGHTER_ACTION_TO_DIRECTION, FIGHTER_ACTION_IS_ATTACK,
    ARCHER_ACTION_TO_DIRECTION,  ARCHER_ACTION_IS_ATTACK,
    # backward-compat names still used internally for fighter
    ACTION_TO_DIRECTION, ACTION_IS_ATTACK,
)
from ai.minion_env import MinionEnv
from audio.sfx_manager import SFXManager
from ui.hud import HUD
from config import CFG

import numpy as np

_M = CFG["arena"]["margin"]
ARENA_LEFT   = _M
ARENA_TOP    = _M
ARENA_RIGHT  = CFG["arena"]["width"]  - _M
ARENA_BOTTOM = CFG["arena"]["height"] - _M
ARENA_BOUNDS = (ARENA_LEFT, ARENA_TOP, ARENA_RIGHT, ARENA_BOTTOM)
_ARENA_H     = ARENA_BOTTOM - ARENA_TOP
_ARENA_W     = ARENA_RIGHT  - ARENA_LEFT

# Gray intensities used when rendering the observation frame [0, 1]
_OBS_GRAY_FIGHTER = 0.40
_OBS_GRAY_ARCHER  = 0.40
_OBS_GRAY_SWARM   = 0.80
_OBS_GRAY_SPIDER  = 0.87
_OBS_GRAY_BOSS    = 0.95
_OBS_GRAY_ARROW   = 0.55

_ARCHER_SHOOT_RANGE = CFG["archer"]["attack_range"]
_ARROW_SPEED        = float(CFG["projectile"]["speed"])
_ARROW_LIFETIME     = float(CFG["projectile"]["lifetime"])
_MP_CFG    = CFG["mp"]
_SP_CFG    = CFG["spells"]
_AIM_CFG   = CFG["ai_master_upgrades"]
_SPAWN_CFG = CFG.get("spawning", {})
_STORE_INTERVAL = int(CFG.get("training", {}).get("memory_store_interval", 10))
_PORTAL_DURATION = float(CFG.get("summon_portal", {}).get("duration", 1.2))

# Knockback force for boss fireball explosions (from physics config)
_PHYS = CFG.get("physics", {})
_FB_KNOCKBACK_FORCE = float(_PHYS.get("knockback_force", 260.0)) * 1.3
_MAX_KB_SPEED = float(_PHYS.get("max_knockback_speed", 600.0))

# Enemy grave duration: how long a dead enemy's tombstone is shown (seconds)
_GRAVE_DURATION = float(CFG.get("enemy", {}).get("grave_duration", 3.0))
_GRAVE_FADE     = 0.5   # seconds before grave expires during which it fades out


def _archer_aim_snap(archer, base_angle: float, enemies: list, boss=None) -> float:
    """
    Improve DQN arrow accuracy by snapping the fire angle to a velocity-lead
    corrected angle when an enemy falls within the DQN's chosen direction cone.

    The DQN selects one of 8 directions (45° apart).  This means even a
    perfectly chosen direction can be up to 22.5° off.  This function:
      1. Finds the nearest alive enemy within ARCHER_MISS_ANGLE of base_angle.
      2. Computes a velocity-lead-corrected intercept angle toward that target.
      3. Returns the corrected angle (or base_angle if no target in cone).
    """
    best_dist   = float('inf')
    best_target = None

    candidates = [e for e in enemies if getattr(e, 'is_alive', False)]
    if boss is not None and getattr(boss, 'is_alive', False):
        candidates.append(boss)

    for t in candidates:
        dist = archer.pos.distance_to(t.pos)
        if dist > _ARCHER_SHOOT_RANGE:
            continue
        e_angle = math.atan2(t.pos.y - archer.pos.y, t.pos.x - archer.pos.x)
        diff = abs(math.atan2(math.sin(e_angle - base_angle),
                              math.cos(e_angle - base_angle)))
        if diff <= ARCHER_MISS_ANGLE and dist < best_dist:
            best_dist   = dist
            best_target = t

    if best_target is None:
        return base_angle

    # Velocity-lead intercept: solve for travel time t such that
    # |target_pos + vel*t - archer_pos| == arrow_speed * t
    dx = best_target.pos.x - archer.pos.x
    dy = best_target.pos.y - archer.pos.y
    vel = getattr(best_target, 'velocity', None)
    vx  = vel.x if vel is not None else 0.0
    vy  = vel.y if vel is not None else 0.0

    a_coef = _ARROW_SPEED * _ARROW_SPEED - (vx * vx + vy * vy)
    b_coef = -2.0 * (dx * vx + dy * vy)
    c_coef = -(dx * dx + dy * dy)
    disc   = b_coef * b_coef - 4.0 * a_coef * c_coef

    if a_coef > 1.0 and disc >= 0.0:
        travel_time = (-b_coef + math.sqrt(disc)) / (2.0 * a_coef)
    else:
        # Fallback: direct-aim time
        travel_time = best_dist / _ARROW_SPEED

    travel_time = max(0.0, min(travel_time, _ARROW_LIFETIME))

    lead_x = dx + vx * travel_time
    lead_y = dy + vy * travel_time
    if lead_x == 0.0 and lead_y == 0.0:
        return base_angle
    return math.atan2(lead_y, lead_x)


def _compute_deployment(am_upgrades: dict) -> tuple[int, int]:
    """Return starting (n_fighters, n_archers) — always 1 of each."""
    return 1, 1


def _compute_spawn_cap(am_upgrades: dict) -> int:
    """Return the global total minion cap based on Deploy Limit upgrade level.

    Level 0 → global cap 2   (the two starting minions; no extra spawning)
    Level 1 → global cap 5
    Level 2 → global cap 8
    Level 3 → global cap 12
    Level 4 → global cap 16
    Level 5 → global cap 20
    Cap values are stored in config.json → spawning.deployment_caps_global.
    """
    level = am_upgrades.get("deployment", 0)
    caps  = _SPAWN_CFG.get("deployment_caps_global", [2, 5, 8, 12, 16, 20])
    return int(caps[min(level, len(caps) - 1)])


def _compute_mp_stats(am: dict) -> tuple[float, float]:
    """(max_mp, regen_per_second)"""
    max_mp = _MP_CFG["base_max"]  + am.get("max_mp", 0) * _MP_CFG["max_per_upgrade_level"]
    regen  = _MP_CFG["base_regen"] + am.get("mp_regen", 0) * _MP_CFG["regen_per_upgrade_level"]
    return float(max_mp), float(regen)


def _compute_heal_stats(am: dict) -> tuple[int, int, float]:
    """(heal_amount, radius, cooldown)"""
    sc   = _SP_CFG["healing"]
    amt  = sc["base_heal_amount"]  + am.get("heal_amount",   0) * sc["heal_per_level"]
    rad  = sc["base_radius"]       + am.get("heal_radius",   0) * sc["radius_per_level"]
    cd   = sc["base_cooldown"]     - am.get("heal_cooldown", 0) * sc["cooldown_reduction_per_level"]
    return int(amt), int(rad), max(1.0, float(cd))


def _compute_fireball_stats(am: dict) -> tuple[int, int, float]:
    """(damage, explosion_radius, cooldown)"""
    sc   = _SP_CFG["fireball"]
    dmg  = sc["base_damage"]   + am.get("fb_damage",   0) * sc["damage_per_level"]
    rad  = sc["base_radius"]   + am.get("fb_radius",   0) * sc["radius_per_level"]
    cd   = sc["base_cooldown"] - am.get("fb_cooldown", 0) * sc["cooldown_reduction_per_level"]
    return int(dmg), int(rad), max(2.0, float(cd))


class BattleScene(BaseScene):
    def __init__(self, game_manager):
        super().__init__(game_manager)
        self.movement_system = MovementSystem()
        self.combat_system   = CombatSystem()
        # Use shared agents from GameManager
        self.fighter_agent = game_manager.fighter_agent
        self.archer_agent  = game_manager.archer_agent
        # Per-agent training threads
        self.fighter_training = TrainingSystem()
        self.archer_training  = TrainingSystem()
        self.sfx = SFXManager()

        # DQN telemetry
        self.latest_loss           = 0.0
        self.latest_steps          = 0
        self.latest_buffer_size    = 0
        self.latest_avg_reward     = 0.0   # EMA of stored transition rewards (fighter)
        self.latest_archer_loss    = 0.0
        self.latest_archer_steps   = 0
        self.latest_archer_buffer_size  = 0
        self.latest_archer_avg_reward   = 0.0  # EMA of stored transition rewards (archer)

        # Async checkpoint saving state
        self._saving         = False   # wave-end checkpoint save in progress
        self._session_saving = False   # session-end buffer snapshot save in progress

        # Minion envs (list per agent role)
        self.fighter_envs: list[MinionEnv] = []
        self.archer_envs:  list[MinionEnv] = []

        self.paused           = False
        self.speed_multiplier = 1
        self.brain_reset_timer = 0.0
        self.damage_numbers   = []   # [x, y, text, color, timer]
        self._dmg_font        = pygame.font.SysFont("arial", 16, bold=True)
        self._session_ended   = False

        # Spell effects
        self.spell_effects: list = []   # HealingEffect | FireballPending | FireballLanding
        self._pending_fireball: FireballPending | None = None

        # Spell selection mode: None | "healing" | "fireball"
        self.spell_mode: str | None = None

        # Spell cooldowns
        self._heal_cd   = 0.0
        self._fb_cd     = 0.0

        # MP
        self.mp     = 0.0
        self.max_mp = 100.0
        self._mp_regen = 5.0

        self._reset()
        self.hud = HUD(self)

    # ── Properties for HUD backward compat ───────────────────────────────

    @property
    def minion(self):
        return self.fighters[0] if self.fighters else None

    @property
    def archer(self):
        return self.archers[0] if self.archers else None

    # ── Reset ─────────────────────────────────────────────────────────────

    def _reset(self):
        am = self.game_manager.save_data.get("ai_master", {}) \
             if self.game_manager.save_data else {}

        n_f, n_a = _compute_deployment(am)
        self.spawn_cap_total = _compute_spawn_cap(am)

        # Lay out minions in a loose cluster near the centre
        cx, cy = 640, 360
        spacing = 50
        self.fighters: list[Minion] = []
        self.archers:  list[Archer] = []
        for i in range(n_f):
            x = cx - (n_f - 1) * spacing // 2 + i * spacing
            self.fighters.append(Minion((x, cy - 50)))
        for i in range(n_a):
            x = cx - (n_a - 1) * spacing // 2 + i * spacing
            self.archers.append(Archer((x, cy + 50)))

        self.enemies        : list = []
        self.projectiles    : list = []
        self.spider_webs    : list = []
        self.fire_mages     : list[FireMage]  = []
        self.ice_mages      : list[IceMage]   = []
        self.mage_projectiles: list = []   # FireMageFireball | IceMageIceball
        self.mage_explosions : list[MageExplosion] = []
        self.creeper_explosions: list[CreeperExplosion] = []
        self.summon_portals  : list[SummonPortal] = []
        self.spell_effects      = []
        self._pending_fireball  = None
        self.spell_mode         = None
        self.wave_system        = WaveSystem(self.game_manager)
        self._session_ended     = False

        # MP stats from upgrades
        self.max_mp, self._mp_regen = _compute_mp_stats(am)
        self.mp = 0.0

        # Spell cooldowns reset on new run
        self._heal_cd = 0.0
        self._fb_cd   = 0.0

        # Build per-minion envs
        self._rebuild_envs()

        # Stats
        self.total_kills        = 0
        self.total_damage_dealt = 0.0
        self.waves_survived     = 0
        self.fighter_kills              = 0
        self.fighter_damage             = 0.0
        self.fighter_attacks_attempted  = 0
        self.fighter_attacks_hit        = 0
        self.archer_kills               = 0
        self.archer_damage              = 0.0
        self.archer_arrow_hits          = 0

        self._apply_research()

    def _rebuild_envs(self):
        """Build MinionEnv for every fighter and archer."""
        self.fighter_envs = []
        self.archer_envs  = []
        # Primary ally for each fighter = first alive archer (env holds live ref via list)
        for f in self.fighters:
            ally = self.archers[0] if self.archers else None
            self.fighter_envs.append(MinionEnv(
                f, self.enemies, ally=ally, role="fighter",
                fighters_ref=self.fighters, archers_ref=self.archers))
        for a in self.archers:
            ally = self.fighters[0] if self.fighters else None
            self.archer_envs.append(MinionEnv(
                a, self.enemies, ally=ally, role="archer",
                fighters_ref=self.fighters, archers_ref=self.archers))

    def _all_minions(self) -> list:
        """Return all fighter + archer + mage entities (alive or dead)."""
        return self.fighters + self.archers + self.fire_mages + self.ice_mages

    def _apply_research(self):
        if not self.game_manager.save_data:
            return
        r  = self.game_manager.save_data.get("research", {})
        rc = CFG["economy"]

        fr = r.get("fighter", {})
        for f in self.fighters:
            f.max_hp       += fr.get("hp",           0) * rc["research_hp_per_level"]
            f.hp            = f.max_hp
            f.attack_damage += fr.get("attack",      0) * rc["research_attack_per_level"]
            f.speed         += fr.get("move_speed",  0) * rc["research_speed_per_level"]
            f.attack_cooldown = max(
                rc["research_min_attack_cooldown"],
                f.attack_cooldown - fr.get("attack_speed", 0) * rc["research_atk_speed_per_level"])
            f.max_stamina  += fr.get("stamina",      0) * rc["research_stamina_per_level"]
            f.stamina       = f.max_stamina

        ar = r.get("archer", {})
        for a in self.archers:
            a.max_hp       += ar.get("hp",           0) * rc["research_hp_per_level"]
            a.hp            = a.max_hp
            a.attack_damage += ar.get("attack",      0) * rc["research_attack_per_level"]
            a.speed         += ar.get("move_speed",  0) * rc["research_speed_per_level"]
            a.max_stamina  += ar.get("stamina",      0) * rc["research_stamina_per_level"]
            a.stamina       = a.max_stamina

    def _apply_research_single(self, entity, role: str):
        """Apply research upgrades to a single newly-spawned minion."""
        if not self.game_manager.save_data:
            return
        r  = self.game_manager.save_data.get("research", {})
        rc = CFG["economy"]
        if role == "fighter":
            fr = r.get("fighter", {})
            entity.max_hp       += fr.get("hp",          0) * rc["research_hp_per_level"]
            entity.hp            = entity.max_hp
            entity.attack_damage += fr.get("attack",     0) * rc["research_attack_per_level"]
            entity.speed         += fr.get("move_speed", 0) * rc["research_speed_per_level"]
            entity.attack_cooldown = max(
                rc["research_min_attack_cooldown"],
                entity.attack_cooldown - fr.get("attack_speed", 0) * rc["research_atk_speed_per_level"])
            entity.max_stamina  += fr.get("stamina",     0) * rc["research_stamina_per_level"]
            entity.stamina       = entity.max_stamina
        else:
            ar = r.get("archer", {})
            entity.max_hp       += ar.get("hp",          0) * rc["research_hp_per_level"]
            entity.hp            = entity.max_hp
            entity.attack_damage += ar.get("attack",     0) * rc["research_attack_per_level"]
            entity.speed         += ar.get("move_speed", 0) * rc["research_speed_per_level"]
            entity.max_stamina  += ar.get("stamina",     0) * rc["research_stamina_per_level"]
            entity.stamina       = entity.max_stamina

    def _try_spawn_minion(self, role: str) -> bool:
        """
        Attempt to initiate a minion summon of `role`.
        Deducts MP and creates a SummonPortal animation; the actual minion
        spawns when the portal animation completes.
        Only valid during an ACTIVE wave.
        Returns True if the summon was initiated.
        """
        if self.wave_system.state != WaveState.ACTIVE:
            return False

        spawn_key     = f"summon_{role}"
        mp_cost       = float(_SP_CFG.get(spawn_key, {}).get("mp_cost", 50))
        total_minions = (len(self.fighters) + len(self.archers) +
                         len(self.fire_mages) + len(self.ice_mages))

        if total_minions >= self.spawn_cap_total:
            return False
        if self.mp < mp_cost:
            return False

        self.mp -= mp_cost

        # Choose a random spawn position away from arena edges
        margin = 60
        x = random.randint(ARENA_LEFT + margin, ARENA_RIGHT - margin)
        y = random.randint(ARENA_TOP  + margin, ARENA_BOTTOM - margin)

        portal = SummonPortal((x, y), role=role, duration=_PORTAL_DURATION)
        self.summon_portals.append(portal)
        if self.sfx:
            self.sfx.play("summon_portal")
        return True

    def _complete_spawn(self, role: str, pos) -> None:
        """Actually spawn the minion after the portal animation completes."""
        x, y = int(pos.x), int(pos.y)
        if role == "fighter":
            m = Minion((x, y))
            self._apply_research_single(m, "fighter")
            self.fighters.append(m)
            ally = self.archers[0] if self.archers else None
            self.fighter_envs.append(MinionEnv(
                m, self.enemies, ally=ally, role="fighter",
                fighters_ref=self.fighters, archers_ref=self.archers))
        elif role == "archer":
            a = Archer((x, y))
            self._apply_research_single(a, "archer")
            self.archers.append(a)
            ally = self.fighters[0] if self.fighters else None
            self.archer_envs.append(MinionEnv(
                a, self.enemies, ally=ally, role="archer",
                fighters_ref=self.fighters, archers_ref=self.archers))
        elif role == "fire_mage":
            fm = FireMage((x, y))
            self.fire_mages.append(fm)
        elif role == "ice_mage":
            im = IceMage((x, y))
            self.ice_mages.append(im)

    # ── Events ───────────────────────────────────────────────────────────

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            self._process_key(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            # Check spell cancel (right-click not needed — left click anywhere handles it)
            pos = event.pos
            result = self.hud.hit_test_control_panel(pos)
            if result is not None:
                self._process_key(result)
                return
            # Check spell icon hits (includes summon spells)
            spell = self.hud.hit_test_spell_panel(pos)
            if spell in ("summon_fighter", "summon_archer",
                         "summon_fire_mage", "summon_ice_mage"):
                role_map = {
                    "summon_fighter":   "fighter",
                    "summon_archer":    "archer",
                    "summon_fire_mage": "fire_mage",
                    "summon_ice_mage":  "ice_mage",
                }
                self._try_spawn_minion(role_map[spell])
                return
            elif spell is not None:
                self._activate_spell(spell)
                return
            # Spell placement (if a spell is selected)
            if self.spell_mode and self._is_in_arena(pos):
                self._cast_spell(pos)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 3:
            # Right-click cancels spell selection
            self.spell_mode = None
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            self.spell_mode = None

    def _is_in_arena(self, pos) -> bool:
        x, y = pos
        return ARENA_LEFT <= x <= ARENA_RIGHT and ARENA_TOP <= y <= ARENA_BOTTOM

    def _activate_spell(self, spell_name: str):
        """Toggle or set spell selection mode."""
        if self.spell_mode == spell_name:
            self.spell_mode = None
            return
        # Check resources
        am  = self.game_manager.save_data.get("ai_master", {}) if self.game_manager.save_data else {}
        if spell_name == "healing":
            _, _, _ = _compute_heal_stats(am)
            cost = CFG["spells"]["healing"]["mp_cost"]
            if self.mp >= cost and self._heal_cd <= 0:
                self.spell_mode = "healing"
        elif spell_name == "fireball":
            cost = CFG["spells"]["fireball"]["mp_cost"]
            if self.mp >= cost and self._fb_cd <= 0:
                self.spell_mode = "fireball"

    def _cast_spell(self, pos):
        am  = self.game_manager.save_data.get("ai_master", {}) if self.game_manager.save_data else {}
        if self.spell_mode == "healing":
            heal_amt, heal_rad, heal_cd = _compute_heal_stats(am)
            cost = CFG["spells"]["healing"]["mp_cost"]
            if self.mp < cost or self._heal_cd > 0:
                self.spell_mode = None
                return
            self.mp -= cost
            self._heal_cd = heal_cd
            effect = HealingEffect(pos, heal_rad, heal_amt)
            healed = effect.apply(self.fighters + self.archers)
            self.spell_effects.append(effect)
            for target, amt in healed:
                self.damage_numbers.append(
                    [float(target.pos.x), float(target.pos.y) - 20,
                     f"+{amt}", (80, 255, 120), 1.2])
            if self.sfx:
                self.sfx.play("hit_impact")
            self.spell_mode = None

        elif self.spell_mode == "fireball":
            fb_dmg, fb_rad, fb_cd = _compute_fireball_stats(am)
            cost = CFG["spells"]["fireball"]["mp_cost"]
            if self.mp < cost or self._fb_cd > 0:
                self.spell_mode = None
                return
            self.mp  -= cost
            self._fb_cd = fb_cd
            flight_t = float(CFG["spells"]["fireball"]["flight_time"])
            pending  = FireballPending(pos, fb_rad, flight_t)
            self.spell_effects.append(pending)
            self._pending_fireball = (pending, fb_dmg, fb_rad)
            self.spell_mode = None

    def _process_key(self, key: int):
        if key == pygame.K_ESCAPE:
            if self.spell_mode:
                self.spell_mode = None
            else:
                self.game_manager.pop_scene()
        elif key == pygame.K_RETURN:
            if (self.wave_system.state in (WaveState.GAME_OVER, WaveState.VICTORY)
                    and not self._session_saving):
                self.game_manager.pop_scene()
        elif key == pygame.K_r:
            self.fighter_agent.reset_brain()
            self.archer_agent.reset_brain()
            self.latest_steps         = 0
            self.latest_loss          = 0.0
            self.latest_archer_steps  = 0
            self.latest_archer_loss   = 0.0
            self.brain_reset_timer    = 2.0
        elif key == pygame.K_p:
            if self.wave_system.state not in (WaveState.GAME_OVER, WaveState.VICTORY):
                self.paused = not self.paused
        elif key in (pygame.K_EQUALS, pygame.K_PLUS, pygame.K_KP_PLUS):
            self.speed_multiplier = min(5, self.speed_multiplier + 1)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            self.speed_multiplier = max(1, self.speed_multiplier - 1)

    # ── Update ───────────────────────────────────────────────────────────

    def update(self, dt: float):
        if self.brain_reset_timer > 0:
            self.brain_reset_timer -= dt

        # Tick damage numbers
        for dn in self.damage_numbers:
            dn[1] -= 40 * dt
            dn[4] -= dt
        self.damage_numbers = [dn for dn in self.damage_numbers if dn[4] > 0]

        if self.paused:
            return

        state = self.wave_system.state
        if state in (WaveState.GAME_OVER, WaveState.VICTORY):
            if not self._session_ended:
                self._session_ended = True
                self._save_run_result()
            return

        game_dt = dt * self.speed_multiplier

        # ── MP regen ────────────────────────────────────────────────────
        self.mp = min(self.max_mp, self.mp + self._mp_regen * game_dt)
        if self._heal_cd > 0: self._heal_cd = max(0.0, self._heal_cd - game_dt)
        if self._fb_cd  > 0: self._fb_cd  = max(0.0, self._fb_cd  - game_dt)

        # ── Spell effects ────────────────────────────────────────────────
        for eff in self.spell_effects:
            if isinstance(eff, FireballPending):
                detonated = eff.update(game_dt)
                if detonated and self._pending_fireball:
                    pending, fb_dmg, fb_rad = self._pending_fireball
                    if pending is eff:
                        landing = FireballLanding(eff.pos, fb_dmg, fb_rad)
                        hits = landing.apply(self.enemies + (
                            [self.wave_system.boss] if self.wave_system.boss and
                            self.wave_system.boss.is_alive else []))
                        for target, dmg in hits:
                            self.damage_numbers.append(
                                [float(target.pos.x), float(target.pos.y) - 20,
                                 str(dmg), (255, 150, 30), 1.2])
                        self.spell_effects.append(landing)
                        self._pending_fireball = None
                        if self.sfx:
                            self.sfx.play("enemy_death")
            else:
                eff.update(game_dt)
        self.spell_effects = [e for e in self.spell_effects if e.is_alive]

        if state in (WaveState.SPAWNING, WaveState.ACTIVE):
            self._update_active(game_dt)
        else:
            # INTERMISSION
            self.wave_system.update(game_dt, self.fighters + self.fire_mages + self.ice_mages,
                                    self.archers, self.enemies)

    def _render_obs_frame(self) -> np.ndarray:
        """Render a simplified overhead view as a float32 grayscale array (H×W)."""
        frame = np.zeros((_ARENA_H, _ARENA_W), dtype=np.float32)

        def _fill(pos, size: int, val: float):
            x = int(pos.x - ARENA_LEFT)
            y = int(pos.y - ARENA_TOP)
            r = max(1, size // 2)
            x0, x1 = max(0, x - r), min(_ARENA_W, x + r + 1)
            y0, y1 = max(0, y - r), min(_ARENA_H, y + r + 1)
            if x0 < x1 and y0 < y1:
                frame[y0:y1, x0:x1] = val

        for e in self.enemies:
            if e.is_alive:
                etype = getattr(e, "enemy_type", 0)
                val = _OBS_GRAY_SPIDER if etype == 1 else _OBS_GRAY_SWARM
                _fill(e.pos, e.size, val)

        boss = self.wave_system.boss
        if boss is not None and boss.is_alive:
            _fill(boss.pos, boss.size, _OBS_GRAY_BOSS)

        for p in self.projectiles:
            if p.is_alive:
                x = int(p.pos.x - ARENA_LEFT)
                y = int(p.pos.y - ARENA_TOP)
                if 0 <= x < _ARENA_W and 0 <= y < _ARENA_H:
                    frame[y, x] = _OBS_GRAY_ARROW

        for f in self.fighters:
            if f.is_alive:
                _fill(f.pos, f.size, _OBS_GRAY_FIGHTER)

        for a in self.archers:
            if a.is_alive:
                _fill(a.pos, a.size, _OBS_GRAY_ARCHER)

        # Mages drawn with same gray intensity as fighters (allies)
        for fm in self.fire_mages:
            if fm.is_alive:
                _fill(fm.pos, fm.size, _OBS_GRAY_FIGHTER)
        for im in self.ice_mages:
            if im.is_alive:
                _fill(im.pos, im.size, _OBS_GRAY_ARCHER)

        return frame

    def _update_active(self, game_dt: float):
        # ── Propagate boss reference to all envs so boss is visible ────
        current_boss = self.wave_system.boss
        for env in self.fighter_envs + self.archer_envs:
            env.boss = current_boss

        # ── Sample observations ─────────────────────────────────────────
        # Image obs (for DQN network) comes from the current frame buffer.
        # Vector obs (for preset heuristic) is computed from live game state.
        f_obs_list  = []
        f_vobs_list = []   # vector obs for preset policy
        f_act_list  = []
        a_obs_list  = []
        a_vobs_list = []
        a_act_list  = []

        for env in self.fighter_envs:
            if env.minion.is_alive:
                obs  = env.get_observation()
                vobs = env.get_vector_observation()
                act  = self.fighter_agent.select_action(obs, preset_obs=vobs)
            else:
                obs, vobs, act = None, None, None
            f_obs_list.append(obs)
            f_vobs_list.append(vobs)
            f_act_list.append(act)

        for env in self.archer_envs:
            if env.minion.is_alive:
                obs  = env.get_observation()
                vobs = env.get_vector_observation()
                act  = self.archer_agent.select_action(obs, preset_obs=vobs)
            else:
                obs, vobs, act = None, None, None
            a_obs_list.append(obs)
            a_vobs_list.append(vobs)
            a_act_list.append(act)

        # ── Update last_action on each minion (read by ally's vector obs) ──
        for env, act in zip(self.fighter_envs, f_act_list):
            if act is not None:
                env.minion.last_action = act
        for env, act in zip(self.archer_envs, a_act_list):
            if act is not None:
                env.minion.last_action = act

        # ── Snapshot alive fighters/archers for team death tracking ────────
        _allies_alive_before = {id(m) for m in self.fighters + self.archers if m.is_alive}

        # ── Set fighter velocities (16-action space: 0–7 move, 8–15 attack) ──
        fighter_attack_dirs = []
        for f, act in zip(self.fighters, f_act_list):
            if f.is_alive and act is not None:
                if FIGHTER_ACTION_IS_ATTACK[act]:
                    f.velocity.update(0, 0)
                    fighter_attack_dirs.append(FIGHTER_ACTION_TO_DIRECTION[act])
                else:
                    dx, dy = FIGHTER_ACTION_TO_DIRECTION[act]
                    f.velocity.update(dx * f.speed, dy * f.speed)
                    fighter_attack_dirs.append(None)
            else:
                fighter_attack_dirs.append(None)

        # ── Set archer velocities and update each archer ──────────────────
        # Archer 24-action space: 0–7 move, 8–23 attack (16 directions)
        for a, act in zip(self.archers, a_act_list):
            if a.is_alive and act is not None:
                if ARCHER_ACTION_IS_ATTACK[act]:
                    a.velocity.update(0, 0)
                else:
                    dx, dy = ARCHER_ACTION_TO_DIRECTION[act]
                    a.velocity.update(dx * a.speed, dy * a.speed)
                a.update(game_dt, ARENA_BOUNDS)

        # ── Tick + move Fire Mages & Ice Mages ───────────────────────────
        boss = self.wave_system.boss
        for fm in self.fire_mages:
            fm.tick(game_dt)
            fm.update_velocity(self.enemies, boss)
        for im in self.ice_mages:
            im.tick(game_dt)
            im.update_velocity(self.enemies, boss)

        # ── Move projectiles; detect timeout-expired arrows ──────────────
        for proj in self.projectiles:
            proj.update(game_dt)
        # Arrows that died from timeout (not from hitting an enemy) → penalty
        _arrow_expired = any(
            not p.is_alive and not p.hit_enemy
            for p in self.projectiles
        )
        self.projectiles = [p for p in self.projectiles if p.is_alive]

        # ── Update mage projectiles ───────────────────────────────────────
        new_mage_exps = []
        for proj in self.mage_projectiles:
            if isinstance(proj, FireMageFireball):
                result = proj.update(game_dt, self.enemies, boss)
                if result is not None:
                    hits = result.apply(self.enemies, boss)
                    for tgt, dmg in hits:
                        self.damage_numbers.append(
                            [float(tgt.pos.x), float(tgt.pos.y) - 18,
                             str(dmg), (255, 130, 30), 1.0])
                    new_mage_exps.append(result)
                    if self.sfx:
                        self.sfx.play("mage_explosion")
            else:  # IceMageIceball
                old_alive = [(e, e.is_alive, e.hp) for e in self.enemies]
                proj.update(game_dt, self.enemies, boss)
                # Detect freeze hits for SFX and damage numbers
                for e, was_alive, old_hp in old_alive:
                    if was_alive and e.hp < old_hp:
                        dmg = old_hp - e.hp
                        self.damage_numbers.append(
                            [float(e.pos.x), float(e.pos.y) - 18,
                             str(dmg), (120, 210, 255), 1.0])
                        if self.sfx:
                            self.sfx.play("freeze_hit")
        self.mage_projectiles = [p for p in self.mage_projectiles if p.is_alive]

        # Update existing mage explosions
        for exp in self.mage_explosions:
            exp.update(game_dt)
        self.mage_explosions = [e for e in self.mage_explosions if e.is_alive]
        self.mage_explosions.extend(new_mage_exps)

        # ── Spider web projectiles ────────────────────────────────────────
        for web in self.spider_webs:
            web.update(game_dt)
        self.spider_webs = [w for w in self.spider_webs if w.is_alive]

        # ── Slime tick (animation) ────────────────────────────────────────
        for e in self.enemies:
            if e.is_alive and isinstance(e, Slime):
                e.tick(game_dt)

        # ── Creeper tick (proximity + fuse animation) ─────────────────────
        all_live_minions = [m for m in self._all_minions() if m.is_alive]
        for e in self.enemies:
            if e.is_alive and isinstance(e, Creeper):
                e.tick(game_dt, all_live_minions)

        # ── Burn damage on enemies ────────────────────────────────────────
        for e in self.enemies:
            if not e.is_alive:
                continue
            b_timer = getattr(e, 'burn_timer', 0.0)
            if b_timer > 0:
                b_dps = getattr(e, 'burn_dps', 0.0)
                dmg = b_dps * game_dt
                e.hp = max(0, e.hp - dmg)
                e.burn_timer = max(0.0, b_timer - game_dt)
                if e.hp <= 0:
                    e.is_alive = False

        # ── Move fighters/archers/mages/enemies ──────────────────────────
        all_minions_list = self.fighters + self.archers + self.fire_mages + self.ice_mages
        self.movement_system.update(game_dt, all_minions_list, [],
                                    self.enemies, ARENA_BOUNDS, boss)

        # ── Summon portals — update and spawn minions when done ───────────
        for portal in self.summon_portals:
            portal.update(game_dt)
            if portal.done:
                self._complete_spawn(portal.role, portal.pos)
        self.summon_portals = [p for p in self.summon_portals if p.is_alive]

        # ── Creeper explosions ────────────────────────────────────────────
        exploding_creepers = [e for e in self.enemies
                              if isinstance(e, Creeper) and e.is_alive and e.should_explode]
        for creeper in exploding_creepers:
            creeper.is_alive = False
            exp = CreeperExplosion(creeper.pos, creeper.explosion_damage,
                                   creeper.explosion_radius)
            hits = exp.apply(all_live_minions)
            for tgt, dmg in hits:
                self.damage_numbers.append(
                    [float(tgt.pos.x), float(tgt.pos.y) - 20,
                     str(dmg), (100, 255, 60), 1.2])
            self.creeper_explosions.append(exp)
            if self.sfx:
                self.sfx.play("creeper_explosion")
        for exp in self.creeper_explosions:
            exp.update(game_dt)
        self.creeper_explosions = [e for e in self.creeper_explosions if e.is_alive]

        # ── Slime split on death ──────────────────────────────────────────
        new_slimes = []
        for e in self.enemies:
            if not e.is_alive and isinstance(e, Slime) and e.generation < 2:
                if not getattr(e, '_split_done', False):
                    e._split_done = True
                    # If this slime was the last alive enemy, skip the split
                    # so the wave ends cleanly instead of spawning new children.
                    alive_others = sum(1 for x in self.enemies if x.is_alive)
                    if alive_others > 0:
                        e._grave_skip = True   # split happened — no grave
                        for _ in range(2):
                            offset = pygame.Vector2(
                                random.uniform(-15, 15), random.uniform(-15, 15))
                            s = Slime(e.pos + offset, generation=e.generation + 1)
                            new_slimes.append(s)
                        if self.sfx:
                            self.sfx.play("slime_split")
                    # else: last enemy — skip split; grave will appear instead
        self.enemies.extend(new_slimes)

        # ── Enemy grave timers ────────────────────────────────────────────
        # Initialize grave for newly dead enemies and tick existing ones down.
        for e in self.enemies:
            if not e.is_alive:
                if e.grave_timer < 0.0 and not getattr(e, '_grave_skip', False):
                    # First frame this enemy is dead → start the grave timer
                    e.grave_timer = _GRAVE_DURATION
                elif e.grave_timer > 0.0:
                    e.grave_timer = max(0.0, e.grave_timer - game_dt)

        # ── Update boss ───────────────────────────────────────────────────
        if boss is not None:
            all_targets = [m for m in self._all_minions() if m.is_alive]
            boss_events = boss.update(game_dt, all_targets)

            # Spawn swarms boss requests
            if boss_events["swarms_to_spawn"] > 0:
                self.wave_system.spawn_swarms_near_boss(
                    self.enemies, boss_events["swarms_to_spawn"])

            # Apply boss explosion damage + knockback to all minions (incl. mages)
            for exp in boss_events["new_explosions"]:
                for target in all_targets:
                    if not target.is_alive:
                        continue
                    if exp.pos.distance_to(target.pos) <= exp.radius:
                        dmg = boss._fb_damage
                        target.hp = max(0, target.hp - dmg)
                        self.damage_numbers.append(
                            [float(target.pos.x), float(target.pos.y) - 18,
                             str(dmg), (255, 80, 200), 1.0])
                        # Knockback away from explosion centre
                        kb_dir = target.pos - exp.pos
                        kv = getattr(target, 'knockback_vel', None)
                        if kv is not None and kb_dir.length_squared() > 0:
                            imp = kb_dir.normalize() * _FB_KNOCKBACK_FORCE
                            kv += imp
                            if kv.length() > _MAX_KB_SPEED:
                                kv.scale_to_length(_MAX_KB_SPEED)
                        if target.hp <= 0:
                            target.is_alive = False

        # ── Combat ────────────────────────────────────────────────────────
        events = self.combat_system.update(
            game_dt,
            fighters=self.fighters,
            enemies=self.enemies,
            projectiles=self.projectiles,
            archers=self.archers,
            fighter_attack_dirs=fighter_attack_dirs,
            boss=boss,
            mages=self.fire_mages + self.ice_mages,
        )

        # ── Fire Mage shooting ────────────────────────────────────────────
        for fm in self.fire_mages:
            if fm.is_alive:
                fb = fm.try_shoot(self.enemies, boss)
                if fb is not None:
                    self.mage_projectiles.append(fb)
                    if self.sfx:
                        self.sfx.play("fireball_shoot")

        # ── Ice Mage shooting ─────────────────────────────────────────────
        for im in self.ice_mages:
            if im.is_alive:
                ib = im.try_shoot(self.enemies, boss)
                if ib is not None:
                    self.mage_projectiles.append(ib)
                    if self.sfx:
                        self.sfx.play("iceball_shoot")

        # ── Spider tick + web shooting ────────────────────────────────────
        alive_minions = [m for m in self._all_minions() if m.is_alive]
        for enemy in self.enemies:
            if not enemy.is_alive:
                continue
            if getattr(enemy, 'enemy_type', 0) == 1:   # Spider
                enemy.tick(game_dt)
                web = enemy.try_shoot_web(alive_minions)
                if web is not None:
                    self.spider_webs.append(web)

        # ── Spider web hit detection ──────────────────────────────────────
        for web in self.spider_webs:
            if not web.is_alive:
                continue
            for target in self._all_minions():
                if not target.is_alive:
                    continue
                hit_dist = (web.size + target.size) // 2 + 2
                if web.pos.distance_to(target.pos) <= hit_dist:
                    target.hp = max(0, target.hp - web.damage)
                    target.frozen_timer = max(
                        getattr(target, 'frozen_timer', 0.0),
                        web.freeze_duration)
                    web.is_alive = False
                    if target in self.fighters:
                        key = "damage_taken"
                    elif target in self.archers:
                        key = "archer_damage_taken"
                    else:
                        key = "mage_damage_taken"
                    events[key] = events.get(key, 0.0) + web.damage
                    self.damage_numbers.append(
                        [float(target.pos.x), float(target.pos.y) - 18,
                         str(web.damage), (100, 160, 255), 1.0])
                    if target.hp <= 0:
                        target.is_alive = False
                    break

        # Propagate expired-arrow flag into events (read by archer reward)
        if _arrow_expired:
            events["archer_arrow_expired"] = True

        # ── Archer shooting ───────────────────────────────────────────────
        # Archer 24-action space: attack actions are indices 8–23 (16 dirs)
        for a, act in zip(self.archers, a_act_list):
            if a.is_alive and act is not None and ARCHER_ACTION_IS_ATTACK[act]:
                dx, dy     = ARCHER_ACTION_TO_DIRECTION[act]
                base_angle = math.atan2(dy, dx)
                # Snap to nearest target in cone + velocity-lead correction
                angle = _archer_aim_snap(a, base_angle, self.enemies,
                                         boss=self.wave_system.boss)
                fired = a.try_shoot(angle, self.projectiles, self.sfx)
                if fired:
                    events["archer_miss"] = self._check_archer_miss(a, angle)

        # ── Track stats ───────────────────────────────────────────────────
        self.total_kills        += events.get("enemies_killed",        0)
        self.total_damage_dealt += events.get("damage_dealt",          0.0)
        self.fighter_kills             += events.get("sword_kills",           0)
        self.fighter_damage            += events.get("sword_damage",          0.0)
        self.fighter_attacks_attempted += events.get("sword_attacks",         0)
        self.fighter_attacks_hit       += events.get("sword_hit_swings",      0)
        self.archer_kills              += events.get("archer_kills",          0)
        self.archer_damage             += events.get("archer_damage_dealt",   0.0)
        self.archer_arrow_hits         += events.get("arrow_hits",            0)

        # ── Team reward events (injected into events dict for get_reward) ──
        # ally_ranged_kills = archer kills (used by fighter's team component)
        # ally_melee_kills  = fighter kills (used by archer's team component)
        events["ally_ranged_kills_this_step"] = events.get("archer_kills", 0)
        events["ally_melee_kills_this_step"]  = events.get("sword_kills",  0)
        events["ally_deaths_this_step"] = sum(
            1 for m in self.fighters + self.archers
            if id(m) in _allies_alive_before and not m.is_alive
        )

        # ── SFX ───────────────────────────────────────────────────────────
        if events.get("sword_attacks", 0) > 0:
            self.sfx.play("sword_swing")
        if events.get("arrow_hits",   0) > 0:
            self.sfx.play("hit_impact")
        if events.get("enemies_killed", 0) > 0:
            self.sfx.play("enemy_death")

        # ── Wave state ────────────────────────────────────────────────────
        prev_state = self.wave_system.state
        self.wave_system.update(game_dt, self.fighters + self.fire_mages + self.ice_mages,
                                    self.archers, self.enemies)
        new_state  = self.wave_system.state

        if prev_state == WaveState.ACTIVE and new_state in (WaveState.INTERMISSION,
                                                             WaveState.VICTORY):
            events["wave_cleared"] = True
            self.waves_survived   += 1
            self.projectiles.clear()
            self.spider_webs.clear()
            self.mage_projectiles.clear()
            self.mage_explosions.clear()
            self.creeper_explosions.clear()
            self.summon_portals.clear()
            self._save_checkpoints_async()

        # ── Damage numbers ────────────────────────────────────────────────
        for hx, hy, val, color in events.get("hits", []):
            self.damage_numbers.append([float(hx), float(hy), str(val), color, 1.0])

        # ── Capture post-combat obs frame for next_obs ────────────────────
        # Rendering happens AFTER combat so next_obs reflects the updated state.
        post_frame = self._render_obs_frame()

        # ── RL training — fighter ─────────────────────────────────────────
        for obs, act, env in zip(f_obs_list, f_act_list, self.fighter_envs):
            if obs is None:
                continue
            env.frame_counter += 1
            reward = env.get_reward(events)
            # Accumulate reward across the store interval so rewards are not lost
            # between stored transitions.
            env._accumulated_reward += reward
            # Append post-combat frame to buffer; get_observation() returns new stack
            env.capture_frame(post_frame)
            next_obs = env.get_observation()
            done     = env.is_done()
            if env.frame_counter % _STORE_INTERVAL == 0 or done:
                acc = env._accumulated_reward
                self.fighter_agent.store_transition(obs, act, acc, next_obs, done)
                self.latest_avg_reward = (
                    0.95 * self.latest_avg_reward + 0.05 * acc)
                env._accumulated_reward = 0.0
            if done:
                env.frame_counter = 0
                env._accumulated_reward = 0.0
        self.latest_buffer_size = self.fighter_agent.tree.size
        result = self.fighter_training.collect_result()
        if result and result.get("steps", 0) > 0:
            self.latest_loss  = result["loss"]
            self.latest_steps = result["steps"]
        self.fighter_training.schedule_training(self.fighter_agent)

        # ── RL training — archer ──────────────────────────────────────────
        for obs, act, env in zip(a_obs_list, a_act_list, self.archer_envs):
            if obs is None:
                continue
            env.frame_counter += 1
            reward = env.get_reward(events)
            env._accumulated_reward += reward
            env.capture_frame(post_frame)
            next_obs = env.get_observation()
            done     = env.is_done()
            if env.frame_counter % _STORE_INTERVAL == 0 or done:
                acc = env._accumulated_reward
                self.archer_agent.store_transition(obs, act, acc, next_obs, done)
                self.latest_archer_avg_reward = (
                    0.95 * self.latest_archer_avg_reward + 0.05 * acc)
                env._accumulated_reward = 0.0
            if done:
                env.frame_counter = 0
                env._accumulated_reward = 0.0
        self.latest_archer_buffer_size = self.archer_agent.tree.size
        result = self.archer_training.collect_result()
        if result and result.get("steps", 0) > 0:
            self.latest_archer_loss  = result["loss"]
            self.latest_archer_steps = result["steps"]
        self.archer_training.schedule_training(self.archer_agent)

    # ── Async checkpoint save ─────────────────────────────────────────────

    def _save_checkpoints_async(self):
        """Save model checkpoints (only) in a background thread after each wave.
        Sets self._saving = True while running; skipped if a save is already in progress."""
        import threading
        if self._saving:
            return
        name = self.game_manager.player_name
        if not name:
            return
        self._saving = True
        fa  = self.fighter_agent
        aa  = self.archer_agent
        gm  = self.game_manager

        def _do_save():
            try:
                fa.save_checkpoint(gm.brain_path(name, "fighter"))
                aa.save_checkpoint(gm.brain_path(name, "archer"))
            finally:
                self._saving = False

        threading.Thread(target=_do_save, daemon=True).start()

    # ── Miss detection ────────────────────────────────────────────────────

    def _check_archer_miss(self, archer: Archer, shot_angle: float) -> bool:
        for e in self.enemies:
            if not e.is_alive:
                continue
            if archer.pos.distance_to(e.pos) > _ARCHER_SHOOT_RANGE:
                continue
            e_angle = math.atan2(e.pos.y - archer.pos.y, e.pos.x - archer.pos.x)
            diff = abs(math.atan2(math.sin(e_angle - shot_angle),
                                  math.cos(e_angle - shot_angle)))
            if diff <= ARCHER_MISS_ANGLE:
                return False
        return True

    # ── Save ─────────────────────────────────────────────────────────────

    def _save_run_result(self):
        if not self.game_manager.save_data:
            return
        sd = self.game_manager.save_data
        sd["waves_completed"] = max(sd.get("waves_completed", 0), self.waves_survived)

        stats = sd.setdefault("stats", {})
        fs = stats.setdefault("fighter", {})
        fs["total_kills"]       = fs.get("total_kills",       0) + self.fighter_kills
        fs["total_damage"]      = fs.get("total_damage",      0) + int(self.fighter_damage)
        fs["training_waves"]    = fs.get("training_waves",    0) + self.waves_survived
        fs["max_waves_survived"] = max(fs.get("max_waves_survived", 0), self.waves_survived)
        fs["attacks_attempted"] = fs.get("attacks_attempted", 0) + self.fighter_attacks_attempted
        fs["attacks_hit"]       = fs.get("attacks_hit",       0) + self.fighter_attacks_hit

        ar = stats.setdefault("archer", {})
        ar["total_kills"]       = ar.get("total_kills",       0) + self.archer_kills
        ar["total_damage"]      = ar.get("total_damage",      0) + int(self.archer_damage)
        ar["shots_fired"]       = ar.get("shots_fired",       0) + sum(
            getattr(a, "shots_fired", 0) for a in self.archers)
        ar["shots_hit"]         = ar.get("shots_hit",         0) + self.archer_arrow_hits
        ar["max_waves_survived"] = max(ar.get("max_waves_survived", 0), self.waves_survived)
        # Count how many mages were ever deployed this run
        stats["fire_mages_deployed"] = stats.get("fire_mages_deployed", 0) + len(self.fire_mages)
        stats["ice_mages_deployed"]  = stats.get("ice_mages_deployed",  0) + len(self.ice_mages)

        # Increment session indices before spawning save thread, then write JSON
        name = self.game_manager.player_name
        sd   = self.game_manager.save_data
        f_idx = sd.get("fighter_session_idx", 0)
        a_idx = sd.get("archer_session_idx",  0)
        sd["fighter_session_idx"] = f_idx + 1
        sd["archer_session_idx"]  = a_idx + 1
        self.game_manager.save_game()

        # Save brain checkpoints + session buffer snapshots asynchronously.
        # Priority extraction over the full buffer can be slow, so we do it in
        # a background thread and show a "Saving session..." indicator until done.
        if name:
            import threading
            self._session_saving = True
            fa = self.fighter_agent
            aa = self.archer_agent
            gm = self.game_manager

            def _do_session_save():
                try:
                    fa.save_checkpoint(gm.brain_path(name, "fighter"))
                    aa.save_checkpoint(gm.brain_path(name, "archer"))
                    fa.save_buffer_session(gm.buffer_folder(name, "fighter"), f_idx)
                    aa.save_buffer_session(gm.buffer_folder(name, "archer"),  a_idx)
                finally:
                    self._session_saving = False

            threading.Thread(target=_do_session_save, daemon=True).start()

    # ── Draw ─────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface):
        surface.fill((30, 30, 40))

        arena_rect = pygame.Rect(ARENA_LEFT, ARENA_TOP,
                                  ARENA_RIGHT - ARENA_LEFT,
                                  ARENA_BOTTOM - ARENA_TOP)
        pygame.draw.rect(surface, (255, 255, 255), arena_rect, 1)

        # Enemies (draw dead first so alive ones render on top)
        for enemy in self.enemies:
            enemy.draw(surface)

        # Creeper explosions (behind living enemies)
        for exp in self.creeper_explosions:
            exp.draw(surface)

        # Mage explosions
        for exp in self.mage_explosions:
            exp.draw(surface)

        # Boss
        boss = self.wave_system.boss
        if boss is not None:
            boss.draw(surface)

        # Projectiles (arrows)
        for proj in self.projectiles:
            proj.draw(surface)

        # Mage projectiles
        for proj in self.mage_projectiles:
            proj.draw(surface)

        # Spider webs
        for web in self.spider_webs:
            web.draw(surface)

        # Summon portals (drawn behind minions)
        for portal in self.summon_portals:
            portal.draw(surface)

        # Spell effects
        for eff in self.spell_effects:
            eff.draw(surface)

        # Minions
        for a in self.archers:
            a.draw(surface)
        for f in self.fighters:
            f.draw(surface)
        for fm in self.fire_mages:
            fm.draw(surface)
        for im in self.ice_mages:
            im.draw(surface)

        # Damage numbers
        for dn in self.damage_numbers:
            x, y, text, color, timer = dn
            alpha = min(255, int(255 * timer))
            surf  = self._dmg_font.render(text, True, color)
            surf.set_alpha(alpha)
            surface.blit(surf, (int(x) - surf.get_width() // 2, int(y)))

        # Spell radius preview on cursor
        if self.spell_mode:
            mx, my = pygame.mouse.get_pos()
            am = self.game_manager.save_data.get("ai_master", {}) \
                 if self.game_manager.save_data else {}
            if self.spell_mode == "healing":
                _, rad, _ = _compute_heal_stats(am)
                col = (60, 220, 80)
            else:
                _, rad, _ = _compute_fireball_stats(am)
                col = (220, 80, 20)
            pygame.draw.circle(surface, col, (mx, my), rad, 2)
            # Dashed inner circle
            for k in range(12):
                a = k * (math.tau / 12)
                px = mx + int(math.cos(a) * rad * 0.6)
                py = my + int(math.sin(a) * rad * 0.6)
                pygame.draw.circle(surface, col, (px, py), 2)

        self.hud.draw(surface)
