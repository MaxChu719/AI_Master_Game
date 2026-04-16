from __future__ import annotations
"""
BattleSimulationScene — Research Lab "Battle Simulation" mode.

Key differences from BattleScene:
  - No wave system.  Monsters spawn continuously at a configurable rate.
  - Boss spawns every `boss_every` seconds (if no boss is currently alive).
  - Dead AI minions auto-revive after a 3-second delay at a random arena position.
  - DQN training runs synchronously every frame for `train_steps` steps per agent,
    maximising buffer accumulation speed.
  - A foldable, scrollable control panel (right side of screen) lets the player
    adjust DQN hyper-parameters and monster spawn rates on the fly.
"""
import math
import random
import threading
import pygame

from engine.scene import BaseScene
from entities.minion        import Minion
from entities.archer        import Archer, ARCHER_MISS_ANGLE
from entities.fire_mage     import FireMage
from entities.ice_mage      import IceMage
from entities.projectile    import Projectile
from entities.spider_web    import SpiderWeb
from entities.enemy         import Enemy
from entities.spider        import Spider
from entities.slime         import Slime
from entities.creeper       import Creeper, CreeperExplosion
from entities.boss          import Boss
from entities.mage_projectile import FireMageFireball, IceMageIceball, MageExplosion
from systems.movement_system import MovementSystem
from systems.combat_system   import CombatSystem
from ai.dqn import (
    FIGHTER_ACTION_TO_DIRECTION, FIGHTER_ACTION_IS_ATTACK,
    ARCHER_ACTION_TO_DIRECTION,  ARCHER_ACTION_IS_ATTACK,
    MAGE_ACTION_TO_DIRECTION,    MAGE_ACTION_IS_ATTACK,
    ACTION_TO_DIRECTION, ACTION_IS_ATTACK,
)
from ai.minion_env import MinionEnv
from audio.sfx_manager import SFXManager
from config import CFG

import numpy as np

# ── Arena constants ───────────────────────────────────────────────────────────
_M           = CFG["arena"]["margin"]
ARENA_LEFT   = _M
ARENA_TOP    = _M
ARENA_RIGHT  = CFG["arena"]["width"]  - _M
ARENA_BOTTOM = CFG["arena"]["height"] - _M
ARENA_BOUNDS = (ARENA_LEFT, ARENA_TOP, ARENA_RIGHT, ARENA_BOTTOM)
_ARENA_W     = ARENA_RIGHT  - ARENA_LEFT
_ARENA_H     = ARENA_BOTTOM - ARENA_TOP

_STORE_INTERVAL = int(CFG.get("training", {}).get("memory_store_interval", 10))
_ARCHER_SHOOT_RANGE = CFG["archer"]["attack_range"]
_ARROW_SPEED        = float(CFG["projectile"]["speed"])
_ARROW_LIFETIME     = float(CFG["projectile"]["lifetime"])

# Revive delay for dead minions in simulation mode
_REVIVE_DELAY = 3.0

# Control panel layout
_PANEL_W = 260   # width of the right-side control panel when open
_PANEL_ROW_H = 34


def _random_arena_pos(margin: int = 60) -> tuple[int, int]:
    x = random.randint(ARENA_LEFT  + margin, ARENA_RIGHT  - margin)
    y = random.randint(ARENA_TOP   + margin, ARENA_BOTTOM - margin)
    return x, y


def _archer_aim_snap(archer, base_angle: float, enemies: list, boss=None) -> float:
    """Mirror of the same helper in battle.py."""
    best_dist   = float("inf")
    best_target = None
    candidates  = [e for e in enemies if getattr(e, "is_alive", False)]
    if boss is not None and getattr(boss, "is_alive", False):
        candidates.append(boss)
    for t in candidates:
        dist = archer.pos.distance_to(t.pos)
        if dist > _ARCHER_SHOOT_RANGE:
            continue
        e_ang = math.atan2(t.pos.y - archer.pos.y, t.pos.x - archer.pos.x)
        diff  = abs(math.atan2(math.sin(e_ang - base_angle),
                               math.cos(e_ang - base_angle)))
        if diff <= ARCHER_MISS_ANGLE and dist < best_dist:
            best_dist   = dist
            best_target = t
    if best_target is None:
        return base_angle
    dx = best_target.pos.x - archer.pos.x
    dy = best_target.pos.y - archer.pos.y
    vel = getattr(best_target, "velocity", None)
    vx  = vel.x if vel is not None else 0.0
    vy  = vel.y if vel is not None else 0.0
    a_c = _ARROW_SPEED ** 2 - (vx ** 2 + vy ** 2)
    b_c = -2.0 * (dx * vx + dy * vy)
    c_c = -(dx ** 2 + dy ** 2)
    if a_c > 0:
        disc = b_c ** 2 - 4 * a_c * c_c
        t_fly = (-b_c + math.sqrt(max(0.0, disc))) / (2 * a_c)
    else:
        t_fly = math.sqrt(dx ** 2 + dy ** 2) / _ARROW_SPEED
    t_fly = min(t_fly, 2.0)
    return math.atan2(dy + vy * t_fly, dx + vx * t_fly)


# ── BattleSimulationScene ─────────────────────────────────────────────────────

class BattleSimulationScene(BaseScene):
    """Infinite combat simulation for rapid DQN training."""

    def __init__(self, game_manager, sim_cfg: dict):
        super().__init__(game_manager)

        # ── Sim config ────────────────────────────────────────────────────
        counts              = sim_cfg.get("counts", {})
        self._train_steps   = max(1,   int(sim_cfg.get("train_steps",  1)))
        self._buffer_rate   = max(1,   int(sim_cfg.get("buffer_rate",  10)))
        self._swarm_rate    = max(1.0, float(sim_cfg.get("swarm_rate", 60)))
        self._boss_every    = max(30.0, float(sim_cfg.get("boss_every", 120)))

        # Apply DQN hyper-parameters to agents
        fa  = game_manager.fighter_agent
        aa  = game_manager.archer_agent
        fma = game_manager.fire_mage_agent
        ima = game_manager.ice_mage_agent
        for agent in (fa, aa, fma, ima):
            if agent is None:
                continue
            agent.apply_training_settings({
                "lr":         float(sim_cfg.get("lr",    1e-4)),
                "batch_size": int(sim_cfg.get("batch",  32)),
            })
            agent.gamma   = float(sim_cfg.get("gamma",       0.99))
            agent.gamma_n = agent.gamma ** agent.n_step

        self.fighter_agent   = fa
        self.archer_agent    = aa
        self.fire_mage_agent = fma
        self.ice_mage_agent  = ima

        self.movement_system = MovementSystem()
        self.combat_system   = CombatSystem()
        self.sfx             = SFXManager()

        # ── Entity lists ──────────────────────────────────────────────────
        self.fighters  : list[Minion]  = []
        self.archers   : list[Archer]  = []
        self.fire_mages: list[FireMage] = []
        self.ice_mages : list[IceMage]  = []
        self.enemies   : list         = []
        self.projectiles: list        = []
        self.spider_webs: list        = []
        self.mage_projectiles: list   = []
        self.mage_explosions : list   = []
        self.creeper_explosions: list = []
        self.boss: Boss | None        = None
        self.boss_explosions: list    = []

        # ── MinionEnvs ────────────────────────────────────────────────────
        self.fighter_envs  : list[MinionEnv] = []
        self.archer_envs   : list[MinionEnv] = []
        self.fire_mage_envs: list[MinionEnv] = []
        self.ice_mage_envs : list[MinionEnv] = []

        # ── Revive queue: list of {"minion": m, "role": str, "timer": float} ──
        self._pending_revives: list[dict] = []

        # ── Spawn timing ──────────────────────────────────────────────────
        self._spawn_timer = 60.0 / self._swarm_rate   # seconds until next enemy
        self._boss_timer  = self._boss_every
        self._enemy_wave_idx = 0   # used to scale enemy HP/speed like wave_index

        # ── Stats ─────────────────────────────────────────────────────────
        self.total_kills        = 0
        self.total_damage_dealt = 0.0
        self.sim_time           = 0.0   # total elapsed simulation seconds
        self.training_steps     = {"fighter": 0, "archer": 0,
                                   "fire_mage": 0, "ice_mage": 0}
        self.latest_loss        = {"fighter": 0.0, "archer": 0.0,
                                   "fire_mage": 0.0, "ice_mage": 0.0}

        # ── DQN telemetry (mirrored for potential HUD reuse) ──────────────
        self.latest_steps       = 0
        self.latest_archer_steps = 0
        self.latest_fm_steps    = 0
        self.latest_im_steps    = 0

        # ── Control panel state ───────────────────────────────────────────
        self._panel_open   = True
        self._panel_scroll = 0
        self._panel_rects  = []  # [(rect, kind, ...)]

        # ── UI state ──────────────────────────────────────────────────────
        self.paused           = False
        self.speed_multiplier = 1
        self.damage_numbers   = []   # [x, y, text, color, timer]
        self._dmg_font        = pygame.font.SysFont("arial", 16, bold=True)
        self._font_hdr  = pygame.font.SysFont("arial", 20, bold=True)
        self._font_stat = pygame.font.SysFont("arial", 18)
        self._font_sm   = pygame.font.SysFont("arial", 14)
        self._font_btn  = pygame.font.SysFont("arial", 22, bold=True)

        # ── Checkpoint save state ─────────────────────────────────────────
        self._saving = False

        # ── Spawn initial minions ─────────────────────────────────────────
        cx, cy = 640, 360
        for i in range(counts.get("fighter", 0)):
            x, y = _random_arena_pos()
            m    = Minion((x, y))
            self._apply_research_single(m, "fighter")
            self.fighters.append(m)
        for i in range(counts.get("archer", 0)):
            x, y = _random_arena_pos()
            a    = Archer((x, y))
            self._apply_research_single(a, "archer")
            self.archers.append(a)
        for i in range(counts.get("fire_mage", 0)):
            x, y = _random_arena_pos()
            fm   = FireMage((x, y))
            self._apply_research_single(fm, "fire_mage")
            self.fire_mages.append(fm)
        for i in range(counts.get("ice_mage", 0)):
            x, y = _random_arena_pos()
            im   = IceMage((x, y))
            self._apply_research_single(im, "ice_mage")
            self.ice_mages.append(im)

        self._rebuild_envs()

    # ── Research helpers ──────────────────────────────────────────────────────

    def _apply_research_single(self, entity, role: str):
        """Apply Research Lab upgrades to a freshly created minion."""
        if not self.game_manager.save_data:
            return
        r  = self.game_manager.save_data.get("research", {})
        rc = CFG["economy"]
        rr = r.get(role, {})
        entity.max_hp       += rr.get("hp",          0) * rc["research_hp_per_level"]
        entity.hp            = entity.max_hp
        entity.attack_damage += rr.get("attack",     0) * rc["research_attack_per_level"]
        entity.speed         += rr.get("move_speed", 0) * rc["research_speed_per_level"]
        entity.max_stamina   += rr.get("stamina",    0) * rc["research_stamina_per_level"]
        entity.stamina        = entity.max_stamina
        if hasattr(entity, "_shoot_cd"):
            cd_red = rr.get("attack_speed", 0) * rc["research_atk_speed_per_level"]
            entity._shoot_cd   = max(0.5, entity._shoot_cd - cd_red)
            entity.attack_cooldown = entity._shoot_cd
        elif hasattr(entity, "attack_cooldown"):
            cd_red = rr.get("attack_speed", 0) * rc["research_atk_speed_per_level"]
            entity.attack_cooldown = max(
                rc["research_min_attack_cooldown"],
                entity.attack_cooldown - cd_red)

    def _rebuild_envs(self):
        self.fighter_envs   = []
        self.archer_envs    = []
        self.fire_mage_envs = []
        self.ice_mage_envs  = []
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
        for fm in self.fire_mages:
            self.fire_mage_envs.append(MinionEnv(
                fm, self.enemies, ally=None, role="fire_mage",
                fighters_ref=self.fighters, archers_ref=self.archers))
        for im in self.ice_mages:
            self.ice_mage_envs.append(MinionEnv(
                im, self.enemies, ally=None, role="ice_mage",
                fighters_ref=self.fighters, archers_ref=self.archers))

    # ── Events ────────────────────────────────────────────────────────────────

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                self.game_manager.pop_scene()
            elif event.key == pygame.K_p:
                self.paused = not self.paused
            elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                self.speed_multiplier = min(5, self.speed_multiplier + 1)
            elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                self.speed_multiplier = max(1, self.speed_multiplier - 1)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._process_panel_click(event.pos)
        elif event.type == pygame.MOUSEWHEEL and self._panel_open:
            self._panel_scroll = max(0, self._panel_scroll - event.y * 30)

    def _process_panel_click(self, pos):
        for info in self._panel_rects:
            rect = info[0]
            if not rect.collidepoint(pos):
                continue
            kind = info[1]
            if kind == "panel_toggle":
                self._panel_open   = not self._panel_open
                self._panel_scroll = 0
            elif kind == "param":
                _, _, param_key, delta = info
                self._adjust_param(param_key, delta)
            elif kind == "scroll_up":
                self._panel_scroll = max(0, self._panel_scroll - 40)
            elif kind == "scroll_down":
                self._panel_scroll += 40
            elif kind == "exit":
                self.game_manager.pop_scene()
            break

    # Param step table — (min, step, max)
    _PARAM_STEPS = {
        "train_steps":  (1,    1,    50),
        "buffer_rate":  (1,    1,    60),
        "swarm_rate":   (10,   10,   300),
        "boss_every":   (30,   30,   600),
        "lr":           (1e-5, 1e-5, 1e-3),
        "batch":        (8,    8,    256),
        "gamma":        (0.90, 0.01, 0.9999),
    }

    def _adjust_param(self, key: str, delta: int):
        if key not in self._PARAM_STEPS:
            return
        lo, step, hi = self._PARAM_STEPS[key]
        if key == "train_steps":
            self._train_steps = max(int(lo), min(int(hi),
                                    self._train_steps + delta * int(step)))
        elif key == "buffer_rate":
            self._buffer_rate = max(int(lo), min(int(hi),
                                    self._buffer_rate + delta * int(step)))
        elif key == "swarm_rate":
            self._swarm_rate = max(lo, min(hi, self._swarm_rate + delta * step))
        elif key == "boss_every":
            self._boss_every = max(lo, min(hi, self._boss_every + delta * step))
        elif key == "lr":
            new_lr = round(max(lo, min(hi,
                     (self.fighter_agent.lr if self.fighter_agent else 1e-4)
                     + delta * step)), 6)
            for ag in (self.fighter_agent, self.archer_agent,
                       self.fire_mage_agent, self.ice_mage_agent):
                if ag is not None:
                    ag.apply_training_settings({"lr": new_lr})
        elif key == "batch":
            new_b = max(int(lo), min(int(hi),
                        (self.fighter_agent.batch_size if self.fighter_agent else 32)
                        + delta * int(step)))
            for ag in (self.fighter_agent, self.archer_agent,
                       self.fire_mage_agent, self.ice_mage_agent):
                if ag is not None:
                    ag.batch_size = new_b
        elif key == "gamma":
            new_g = round(max(lo, min(hi,
                        (self.fighter_agent.gamma if self.fighter_agent else 0.99)
                        + delta * step)), 4)
            for ag in (self.fighter_agent, self.archer_agent,
                       self.fire_mage_agent, self.ice_mage_agent):
                if ag is not None:
                    ag.gamma   = new_g
                    ag.gamma_n = new_g ** ag.n_step

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, dt: float):
        if self.paused:
            return
        for _ in range(self.speed_multiplier):
            self._tick(dt)

    def _tick(self, dt: float):
        self.sim_time += dt

        # Decay damage numbers
        self.damage_numbers = [
            [x, y - 30 * dt, t, c, life - dt]
            for x, y, t, c, life in self.damage_numbers
            if life - dt > 0
        ]

        # ── DQN: select actions for all minions ───────────────────────────
        all_minions = self.fighters + self.archers + self.fire_mages + self.ice_mages
        all_live    = [m for m in all_minions if m.is_alive]
        boss        = self.boss

        pre_frame = self._capture_arena_frame()

        f_obs_list  = []; f_act_list  = []
        a_obs_list  = []; a_act_list  = []
        fm_obs_list = []; fm_act_list = []
        im_obs_list = []; im_act_list = []

        fighter_attack_dirs: list = []
        for env, fighter in zip(self.fighter_envs, self.fighters):
            if not fighter.is_alive:
                f_obs_list.append(None); f_act_list.append(None)
                fighter_attack_dirs.append(None); continue
            env.capture_frame(pre_frame)
            obs = env.get_observation()
            act = self.fighter_agent.select_action(obs) if self.fighter_agent else 0
            fighter.last_action = act
            is_attack = FIGHTER_ACTION_IS_ATTACK[act]
            d = FIGHTER_ACTION_TO_DIRECTION[act]
            if is_attack:
                fighter.velocity.update(0, 0)
                fighter_attack_dirs.append(d)
            else:
                fighter.velocity.update(d[0] * fighter.speed, d[1] * fighter.speed)
                fighter_attack_dirs.append(None)
            f_obs_list.append(obs); f_act_list.append(act)

        for env, archer in zip(self.archer_envs, self.archers):
            if not archer.is_alive:
                a_obs_list.append(None); a_act_list.append(None); continue
            env.capture_frame(pre_frame)
            obs = env.get_observation()
            act = self.archer_agent.select_action(obs) if self.archer_agent else 0
            archer.last_action = act
            is_attack = ARCHER_ACTION_IS_ATTACK[act]
            d = ARCHER_ACTION_TO_DIRECTION[act]
            if is_attack:
                archer.velocity.update(0, 0)
            else:
                archer.velocity.update(d[0] * archer.speed, d[1] * archer.speed)
            archer.update(dt, ARENA_BOUNDS)
            a_obs_list.append(obs); a_act_list.append(act)

        for env, fm in zip(self.fire_mage_envs, self.fire_mages):
            fm.tick(dt)
            if not fm.is_alive:
                fm_obs_list.append(None); fm_act_list.append(None); continue
            env.capture_frame(pre_frame)
            obs = env.get_observation()
            act = self.fire_mage_agent.select_action(obs) if self.fire_mage_agent else 0
            fm.last_action = act
            is_attack = MAGE_ACTION_IS_ATTACK[act]
            d = MAGE_ACTION_TO_DIRECTION[act]
            if is_attack:
                fm.velocity.update(0, 0)
            else:
                fm.velocity.update(d[0] * fm.speed, d[1] * fm.speed)
            fm_obs_list.append(obs); fm_act_list.append(act)

        for env, im in zip(self.ice_mage_envs, self.ice_mages):
            im.tick(dt)
            if not im.is_alive:
                im_obs_list.append(None); im_act_list.append(None); continue
            env.capture_frame(pre_frame)
            obs = env.get_observation()
            act = self.ice_mage_agent.select_action(obs) if self.ice_mage_agent else 0
            im.last_action = act
            is_attack = MAGE_ACTION_IS_ATTACK[act]
            d = MAGE_ACTION_TO_DIRECTION[act]
            if is_attack:
                im.velocity.update(0, 0)
            else:
                im.velocity.update(d[0] * im.speed, d[1] * im.speed)
            im_obs_list.append(obs); im_act_list.append(act)

        # ── Burn damage on enemies ────────────────────────────────────────
        for e in self.enemies:
            if not e.is_alive:
                continue
            b_t = getattr(e, "burn_timer", 0.0)
            if b_t > 0:
                b_dps = getattr(e, "burn_dps", 0.0)
                e.hp = max(0, e.hp - b_dps * dt)
                e.burn_timer = max(0.0, b_t - dt)
                if e.hp <= 0:
                    e.is_alive = False

        # ── Movement / combat ─────────────────────────────────────────────
        self.movement_system.update(dt, all_minions, [], self.enemies,
                                    ARENA_BOUNDS, boss)
        events = self.combat_system.update(
            dt, self.fighters, self.enemies, self.projectiles,
            self.archers, fighter_attack_dirs, boss)

        self.total_kills        += events.get("kills", 0)
        self.total_damage_dealt += events.get("damage_dealt", 0.0)

        # ── Archer shooting (deferred to after combat system) ─────────────
        for a, act in zip(self.archers, a_act_list):
            if a.is_alive and act is not None and ARCHER_ACTION_IS_ATTACK[act]:
                dx, dy     = ARCHER_ACTION_TO_DIRECTION[act]
                base_angle = math.atan2(dy, dx)
                angle      = _archer_aim_snap(a, base_angle, self.enemies, boss)
                a.try_shoot(angle, self.projectiles, self.sfx)

        # ── Mage shooting (deferred to after combat system) ───────────────
        for fm, act in zip(self.fire_mages, fm_act_list):
            if fm.is_alive and act is not None and MAGE_ACTION_IS_ATTACK[act]:
                dx, dy    = MAGE_ACTION_TO_DIRECTION[act]
                proj = fm.try_shoot_aimed(math.atan2(dy, dx), self.enemies, boss)
                if proj:
                    self.mage_projectiles.append(proj)

        for im, act in zip(self.ice_mages, im_act_list):
            if im.is_alive and act is not None and MAGE_ACTION_IS_ATTACK[act]:
                dx, dy    = MAGE_ACTION_TO_DIRECTION[act]
                proj = im.try_shoot_aimed(math.atan2(dy, dx), self.enemies, boss)
                if proj:
                    self.mage_projectiles.append(proj)

        # ── Boss update ───────────────────────────────────────────────────
        if boss is not None and boss.is_alive:
            boss_events = boss.update(dt, all_live)
            n_swarms    = boss_events.get("swarms_to_spawn", 0)
            for _ in range(n_swarms):
                px = int(boss.pos.x + random.randint(-80, 80))
                py = int(boss.pos.y + random.randint(-80, 80))
                px = max(ARENA_LEFT + 20, min(ARENA_RIGHT  - 20, px))
                py = max(ARENA_TOP  + 20, min(ARENA_BOTTOM - 20, py))
                _e = Enemy((px, py))
                _ecc_b = CFG["enemy"]
                _e.speed = _ecc_b["base_speed"] + self._enemy_wave_idx * _ecc_b["speed_per_wave"]
                self.enemies.append(_e)
            for exp in boss_events.get("new_explosions", []):
                hits = exp.apply(all_live)
                for tgt, dmg in hits:
                    self.damage_numbers.append(
                        [float(tgt.pos.x), float(tgt.pos.y) - 20,
                         str(dmg), (255, 80, 0), 1.2])
                self.boss_explosions.append(exp)
            if boss_events.get("boss_dead", False):
                self.total_kills += 1
        if boss is not None and not boss.is_alive and not getattr(boss, "_dying", False):
            self.boss = None

        # ── Mage projectiles ──────────────────────────────────────────────
        for proj in self.mage_projectiles:
            result = proj.update(dt, self.enemies, boss)
            if result is not None:  # MageExplosion returned on hit
                self.mage_explosions.append(result)
        new_exps = [p for p in self.mage_projectiles if isinstance(p, FireMageFireball)]
        for exp in self.mage_explosions:
            hits = exp.apply(self.enemies, boss)
            for e, dmg in hits:
                self.total_damage_dealt += dmg
                self.damage_numbers.append(
                    [float(e.pos.x), float(e.pos.y) - 20,
                     str(dmg), (255, 140, 20), 1.2])
        self.mage_projectiles = [p for p in self.mage_projectiles if p.is_alive]
        self.mage_explosions  = [e for e in self.mage_explosions  if e.is_alive]

        # ── Spider webs ───────────────────────────────────────────────────
        for spider in [e for e in self.enemies if isinstance(e, Spider) and e.is_alive]:
            web = spider.try_shoot_web(all_live)
            if web:
                self.spider_webs.append(web)
        for web in self.spider_webs:
            web.update(dt)
            if not web.is_alive:
                continue
            for m in all_live:
                hit_dist = (web.size + m.size) // 2 + 2
                if web.pos.distance_to(m.pos) <= hit_dist:
                    m.hp = max(0, m.hp - web.damage)
                    m.frozen_timer = max(getattr(m, 'frozen_timer', 0.0),
                                        web.freeze_duration)
                    web.is_alive = False
                    break
        self.spider_webs = [w for w in self.spider_webs if w.is_alive]

        # ── Creeper explosions ────────────────────────────────────────────
        for creeper in [e for e in self.enemies
                        if isinstance(e, Creeper) and e.is_alive and e.should_explode]:
            creeper.is_alive = False
            exp = CreeperExplosion(creeper.pos, creeper.explosion_damage,
                                   creeper.explosion_radius)
            hits = exp.apply(all_live)
            for tgt, dmg in hits:
                self.damage_numbers.append(
                    [float(tgt.pos.x), float(tgt.pos.y) - 20,
                     str(dmg), (100, 255, 60), 1.2])
            self.creeper_explosions.append(exp)

        # ── Slime on-death splits ─────────────────────────────────────────
        new_slimes = []
        for sl in self.enemies:
            if not isinstance(sl, Slime) or sl.is_alive:
                continue
            if not getattr(sl, "_split_done", False) and sl.generation < 2:
                sl._split_done = True
                alive_others   = sum(1 for x in self.enemies if x.is_alive)
                if alive_others > 0:
                    sl._grave_skip = True
                    for _ in range(2):
                        offset = pygame.Vector2(
                            random.uniform(-15, 15), random.uniform(-15, 15))
                        new_slimes.append(Slime(sl.pos + offset,
                                                generation=sl.generation + 1))
        self.enemies.extend(new_slimes)

        # ── Clean dead enemies (after grace period for graves) ────────────
        _grave_dur = float(CFG.get("enemy", {}).get("grave_duration", 3.0))
        for e in self.enemies:
            if not e.is_alive:
                if e.grave_timer < 0.0 and not getattr(e, "_grave_skip", False):
                    e.grave_timer = _grave_dur
                elif e.grave_timer > 0.0:
                    e.grave_timer = max(0.0, e.grave_timer - dt)
        self.enemies = [e for e in self.enemies
                        if e.is_alive or e.grave_timer > 0]

        for exp in self.creeper_explosions:
            exp.update(dt)
        for exp in self.boss_explosions:
            exp.update(dt)
        self.creeper_explosions = [e for e in self.creeper_explosions if e.is_alive]
        self.boss_explosions    = [e for e in self.boss_explosions    if e.is_alive]

        # ── Auto-revive queue ─────────────────────────────────────────────
        self._process_revives(dt, all_minions)

        # Queue newly dead minions for revive
        for m in all_minions:
            if not m.is_alive and not any(r["minion"] is m for r in self._pending_revives):
                role = self._minion_role(m)
                self._pending_revives.append({"minion": m, "role": role, "timer": _REVIVE_DELAY})

        # ── Enemy spawning ────────────────────────────────────────────────
        self._spawn_timer -= dt
        if self._spawn_timer <= 0:
            self._spawn_enemies()
            self._spawn_timer = 60.0 / self._swarm_rate

        # ── Periodic boss spawn ───────────────────────────────────────────
        self._boss_timer -= dt
        if self._boss_timer <= 0 and self.boss is None:
            self._spawn_boss()
            self._boss_timer = self._boss_every

        # ── DQN training (synchronous, N steps per frame) ─────────────────
        post_frame = self._capture_arena_frame()
        self._run_training(f_obs_list, f_act_list,
                           a_obs_list, a_act_list,
                           fm_obs_list, fm_act_list,
                           im_obs_list, im_act_list,
                           post_frame, events)

    def _minion_role(self, m) -> str:
        if m in self.fighters:
            return "fighter"
        if m in self.archers:
            return "archer"
        if m in self.fire_mages:
            return "fire_mage"
        if m in self.ice_mages:
            return "ice_mage"
        return "fighter"

    def _process_revives(self, dt: float, all_minions: list):
        still_pending = []
        for entry in self._pending_revives:
            m     = entry["minion"]
            entry["timer"] -= dt
            if entry["timer"] <= 0:
                # Revive: reset HP/stamina, reposition
                m.hp        = m.max_hp
                m.stamina   = m.max_stamina
                m.is_alive  = True
                m.grave_timer = -1.0
                x, y = _random_arena_pos()
                m.pos = pygame.Vector2(x, y)
                # Reset corresponding MinionEnv
                role = entry["role"]
                if role == "fighter":
                    for env in self.fighter_envs:
                        if env.minion is m:
                            env.frame_counter = 0
                elif role == "archer":
                    for env in self.archer_envs:
                        if env.minion is m:
                            env.frame_counter = 0
                elif role == "fire_mage":
                    for env in self.fire_mage_envs:
                        if env.minion is m:
                            env.frame_counter = 0
                elif role == "ice_mage":
                    for env in self.ice_mage_envs:
                        if env.minion is m:
                            env.frame_counter = 0
            else:
                still_pending.append(entry)
        self._pending_revives = still_pending

    def _spawn_enemies(self):
        """Spawn one enemy (swarm or spider based on ratio)."""
        _ecc   = CFG["enemy"]
        ratio  = float(CFG.get("wave", {}).get("spider_swarm_ratio", 0.25))
        # Slowly scale difficulty: 1 wave-equivalent per minute of sim time
        self._enemy_wave_idx = min(100, int(self.sim_time / 60))

        if random.random() < ratio:
            x, y = _random_arena_pos(margin=40)
            self.enemies.append(Spider((x, y)))
        else:
            x, y = _random_arena_pos(margin=40)
            e = Enemy((x, y))
            e.speed = _ecc["base_speed"] + self._enemy_wave_idx * _ecc["speed_per_wave"]
            self.enemies.append(e)
        # Occasionally also spawn a slime or creeper (rate matches wave formula)
        # ~1 slime per 8 swarm spawns
        if random.random() < 1.0 / 8.0:
            x, y = _random_arena_pos(margin=40)
            self.enemies.append(Slime((x, y), generation=0))
        # ~1 creeper per 10 swarm spawns
        if random.random() < 1.0 / 10.0:
            x, y = _random_arena_pos(margin=40)
            self.enemies.append(Creeper((x, y)))

    def _spawn_boss(self):
        x, y = _random_arena_pos(margin=100)
        self.boss = Boss((x, y), wave_index=self._enemy_wave_idx)

    # ── Arena frame capture (image obs) ──────────────────────────────────────

    def _capture_arena_frame(self):
        """Return a float32 grayscale array (ARENA_H × ARENA_W) for image obs.

        Rendering is identical to BattleScene._render_obs_frame() so that
        observations are transferable between simulation and real waves.
        No HUD elements are painted — only arena entities.
        """
        arena_h = int(_ARENA_H)
        arena_w = int(_ARENA_W)
        frame   = np.zeros((arena_h, arena_w), dtype=np.float32)

        def _fill(pos, size: int, val: float):
            x = int(pos.x - ARENA_LEFT)
            y = int(pos.y - ARENA_TOP)
            r = max(1, size // 2)
            x0, x1 = max(0, x - r), min(arena_w, x + r + 1)
            y0, y1 = max(0, y - r), min(arena_h, y + r + 1)
            if x0 < x1 and y0 < y1:
                frame[y0:y1, x0:x1] = val

        for e in self.enemies:
            if e.is_alive:
                val = 0.87 if isinstance(e, Spider) else 0.80
                _fill(e.pos, e.size, val)
        if self.boss is not None and self.boss.is_alive:
            _fill(self.boss.pos, self.boss.size, 0.95)
        for p in self.projectiles:
            if p.is_alive:
                x = int(p.pos.x - ARENA_LEFT)
                y = int(p.pos.y - ARENA_TOP)
                if 0 <= x < arena_w and 0 <= y < arena_h:
                    frame[y, x] = 0.55
        for f in self.fighters:
            if f.is_alive:
                _fill(f.pos, f.size, 0.40)
        for a in self.archers:
            if a.is_alive:
                _fill(a.pos, a.size, 0.40)
        for fm in self.fire_mages:
            if fm.is_alive:
                _fill(fm.pos, fm.size, 0.40)
        for im in self.ice_mages:
            if im.is_alive:
                _fill(im.pos, im.size, 0.40)
        return frame

    # ── DQN training ─────────────────────────────────────────────────────────

    def _run_training(self, f_obs, f_act, a_obs, a_act,
                      fm_obs, fm_act, im_obs, im_act,
                      post_frame, events):
        """Store transitions and run N synchronous training steps per frame."""
        def _store_all(obs_list, act_list, envs, agent, avg_key):
            if agent is None:
                return
            for obs, act, env in zip(obs_list, act_list, envs):
                if obs is None:
                    continue
                env.frame_counter += 1
                reward  = env.get_reward(events)
                env._accumulated_reward += reward
                env.capture_frame(post_frame)
                next_obs = env.get_observation()
                done     = env.is_done()
                if env.frame_counter % self._buffer_rate == 0 or done:
                    acc = env._accumulated_reward
                    agent.store_transition(obs, act, acc, next_obs, done)
                    env._accumulated_reward = 0.0
                if done:
                    env.frame_counter = 0
                    env._accumulated_reward = 0.0

        _store_all(f_obs,  f_act,  self.fighter_envs,   self.fighter_agent,   "fighter")
        _store_all(a_obs,  a_act,  self.archer_envs,    self.archer_agent,    "archer")
        _store_all(fm_obs, fm_act, self.fire_mage_envs, self.fire_mage_agent, "fire_mage")
        _store_all(im_obs, im_act, self.ice_mage_envs,  self.ice_mage_agent,  "ice_mage")

        # Run N training steps per frame (synchronous — maximises throughput)
        for _ in range(self._train_steps):
            for role, agent in (
                ("fighter",   self.fighter_agent),
                ("archer",    self.archer_agent),
                ("fire_mage", self.fire_mage_agent),
                ("ice_mage",  self.ice_mage_agent),
            ):
                if agent is None:
                    continue
                result = agent.train_step()
                if result.get("steps", 0) > 0:
                    self.latest_loss[role]   = result["loss"]
                    self.training_steps[role] += 1
        self.latest_steps        = self.training_steps["fighter"]
        self.latest_archer_steps = self.training_steps["archer"]
        self.latest_fm_steps     = self.training_steps["fire_mage"]
        self.latest_im_steps     = self.training_steps["ice_mage"]

    # ── Async checkpoint save ────────────────────────────────────────────────

    def _save_checkpoints_async(self):
        if self._saving:
            return
        name = self.game_manager.player_name
        if not name:
            return
        self._saving = True
        fa, aa, fma, ima = (self.fighter_agent, self.archer_agent,
                            self.fire_mage_agent, self.ice_mage_agent)
        gm = self.game_manager

        def _do():
            try:
                if fa:  fa.save_checkpoint(gm.brain_path(name, "fighter"))
                if aa:  aa.save_checkpoint(gm.brain_path(name, "archer"))
                if fma: fma.save_checkpoint(gm.brain_path(name, "fire_mage"))
                if ima: ima.save_checkpoint(gm.brain_path(name, "ice_mage"))
            finally:
                self._saving = False

        threading.Thread(target=_do, daemon=True).start()

    # ── Draw ──────────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface):
        surface.fill((10, 10, 18))
        sw, sh = surface.get_size()
        self._panel_rects = []

        # Arena area — shrink if panel open
        arena_right = sw - (_PANEL_W if self._panel_open else 0)

        # Draw arena boundary
        pygame.draw.rect(surface, (30, 30, 50),
                         (ARENA_LEFT - 2, ARENA_TOP - 2,
                          ARENA_RIGHT - ARENA_LEFT + 4,
                          ARENA_BOTTOM - ARENA_TOP + 4), 1)

        # ── Entities ──────────────────────────────────────────────────────
        for e in self.enemies:
            e.draw(surface)
        if self.boss is not None:
            self.boss.draw(surface)
        for m in self.fighters + self.archers + self.fire_mages + self.ice_mages:
            m.draw(surface)
        for p in self.projectiles:
            p.draw(surface)
        for proj in self.mage_projectiles:
            proj.draw(surface)
        for exp in self.mage_explosions + self.creeper_explosions + self.boss_explosions:
            exp.draw(surface)
        for web in self.spider_webs:
            web.draw(surface)

        # ── Pending revive overlays ────────────────────────────────────────
        for entry in self._pending_revives:
            m   = entry["minion"]
            t   = entry["timer"]
            col = (120, 120, 200)
            s   = self._font_sm.render(f"Revive {t:.1f}s", True, col)
            surface.blit(s, s.get_rect(center=(int(m.pos.x), int(m.pos.y) - 24)))

        # ── Damage numbers ────────────────────────────────────────────────
        for x, y, text, col, _ in self.damage_numbers:
            ds = self._dmg_font.render(text, True, col)
            surface.blit(ds, (int(x), int(y)))

        # ── Simulation HUD (top-left) ─────────────────────────────────────
        hud_lines = [
            f"Sim Time: {self.sim_time:.0f}s  |  Speed: {self.speed_multiplier}x",
            f"Kills: {self.total_kills:,}  |  Enemies: {len([e for e in self.enemies if e.is_alive])}",
            f"F steps:{self.training_steps['fighter']:,}  A:{self.training_steps['archer']:,}"
            f"  FM:{self.training_steps['fire_mage']:,}  IM:{self.training_steps['ice_mage']:,}",
            f"Loss — F:{self.latest_loss['fighter']:.4f}"
            f"  A:{self.latest_loss['archer']:.4f}"
            f"  FM:{self.latest_loss['fire_mage']:.4f}"
            f"  IM:{self.latest_loss['ice_mage']:.4f}",
        ]
        for i, line in enumerate(hud_lines):
            s = self._font_sm.render(line, True, (180, 200, 180))
            surface.blit(s, (8, 8 + i * 18))

        # Minion status bar (bottom left)
        bar_y = sh - 80
        for ci, (role, lst) in enumerate([
                ("F", self.fighters), ("A", self.archers),
                ("FM", self.fire_mages), ("IM", self.ice_mages)]):
            for j, m in enumerate(lst):
                bx   = 8 + ci * 120 + j * 14
                col  = (80, 200, 80) if m.is_alive else (100, 60, 60)
                pygame.draw.rect(surface, col, (bx, bar_y, 12, 16), border_radius=3)
                ns = self._font_sm.render(role, True, (220, 220, 220))
                surface.blit(ns, ns.get_rect(center=(bx + 6, bar_y + 8)))

        # Key hints
        hints = self._font_sm.render(
            "[P] Pause    [+/-] Speed    [ESC] Exit Simulation", True, (60, 60, 80))
        surface.blit(hints, hints.get_rect(center=(sw // 2, sh - 12)))

        # ── Control panel ─────────────────────────────────────────────────
        self._draw_control_panel(surface, sw, sh)

    def _draw_control_panel(self, surface: pygame.Surface, sw: int, sh: int):
        """Draw the foldable, scrollable right-side control panel."""
        # Toggle button (always visible)
        toggle_w, toggle_h = 36, 36
        toggle_x = sw - toggle_w - 4
        toggle_y = ARENA_TOP
        toggle_r = pygame.Rect(toggle_x, toggle_y, toggle_w, toggle_h)
        pygame.draw.rect(surface, (30, 25, 50), toggle_r, border_radius=6)
        pygame.draw.rect(surface, (120, 80, 200), toggle_r, 2, border_radius=6)
        arrow = "◀" if self._panel_open else "▶"
        as_   = self._font_btn.render(arrow, True, (180, 150, 240))
        surface.blit(as_, as_.get_rect(center=toggle_r.center))
        self._panel_rects.append((toggle_r, "panel_toggle"))

        if not self._panel_open:
            return

        panel_x = sw - _PANEL_W
        panel_h = sh
        bg_surf = pygame.Surface((_PANEL_W, panel_h), pygame.SRCALPHA)
        bg_surf.fill((15, 12, 28, 220))
        surface.blit(bg_surf, (panel_x, 0))
        pygame.draw.line(surface, (60, 40, 90), (panel_x, 0), (panel_x, panel_h), 2)

        # Header
        hdr_s = self._font_hdr.render("Sim Controls", True, (200, 160, 255))
        surface.blit(hdr_s, hdr_s.get_rect(center=(panel_x + _PANEL_W // 2, 20)))

        # Exit button
        exit_r = pygame.Rect(panel_x + 8, 40, _PANEL_W - 16, 28)
        pygame.draw.rect(surface, (60, 20, 20), exit_r, border_radius=5)
        pygame.draw.rect(surface, (160, 60, 60), exit_r, 1, border_radius=5)
        ex_s   = self._font_sm.render("Exit Simulation (ESC)", True, (220, 120, 120))
        surface.blit(ex_s, ex_s.get_rect(center=exit_r.center))
        self._panel_rects.append((exit_r, "exit"))

        # Scrollable params
        params = [
            # (label, current_value_fn, fmt, key)
            ("Train Steps/Frame",  lambda: self._train_steps,     lambda v: str(int(v)),   "train_steps"),
            ("Buffer Rate (frames)", lambda: self._buffer_rate,   lambda v: str(int(v)),   "buffer_rate"),
            ("Swarm Rate/min",     lambda: self._swarm_rate,      lambda v: str(int(v)),   "swarm_rate"),
            ("Boss Interval (s)",  lambda: self._boss_every,      lambda v: str(int(v)),   "boss_every"),
            ("Learning Rate",      lambda: (self.fighter_agent.lr
                                            if self.fighter_agent else 1e-4),
                                            lambda v: f"{v:.1e}", "lr"),
            ("Batch Size",         lambda: (self.fighter_agent.batch_size
                                            if self.fighter_agent else 32),
                                            lambda v: str(int(v)), "batch"),
            ("Gamma",              lambda: (self.fighter_agent.gamma
                                            if self.fighter_agent else 0.99),
                                            lambda v: f"{v:.4f}", "gamma"),
        ]

        content_top = 76
        content_h   = sh - content_top - 10
        max_scroll  = max(0, len(params) * _PANEL_ROW_H - content_h + 8)
        self._panel_scroll = min(self._panel_scroll, max_scroll)

        clip = pygame.Rect(panel_x, content_top, _PANEL_W, content_h)
        old_clip = surface.get_clip()
        surface.set_clip(clip)

        draw_y = content_top - self._panel_scroll
        for param_label, val_fn, fmt, pkey in params:
            ry     = draw_y
            draw_y += _PANEL_ROW_H
            if ry + _PANEL_ROW_H < content_top or ry > content_top + content_h:
                continue

            pl_s = self._font_sm.render(param_label, True, (160, 150, 200))
            surface.blit(pl_s, pl_s.get_rect(midleft=(panel_x + 6, ry + 8)))

            val_str = fmt(val_fn())
            btn_w   = 26
            vw      = 70
            vx      = panel_x + _PANEL_W - btn_w - vw - btn_w - 8

            dec_r = pygame.Rect(vx,                ry + 4, btn_w, _PANEL_ROW_H - 8)
            val_r = pygame.Rect(vx + btn_w + 2,    ry + 4, vw,    _PANEL_ROW_H - 8)
            inc_r = pygame.Rect(vx + btn_w + vw + 4, ry + 4, btn_w, _PANEL_ROW_H - 8)

            for r, lbl in ((dec_r, "−"), (inc_r, "+")):
                pygame.draw.rect(surface, (40, 28, 60), r, border_radius=4)
                pygame.draw.rect(surface, (100, 70, 150), r, 1, border_radius=4)
                bs = self._font_btn.render(lbl, True, (200, 180, 255))
                surface.blit(bs, bs.get_rect(center=r.center))
            pygame.draw.rect(surface, (22, 16, 38), val_r, border_radius=3)
            pygame.draw.rect(surface, (70, 50, 110), val_r, 1, border_radius=3)
            vs = self._font_sm.render(val_str, True, (255, 240, 160))
            surface.blit(vs, vs.get_rect(center=val_r.center))

            self._panel_rects.append((dec_r, "param", pkey, -1))
            self._panel_rects.append((inc_r, "param", pkey,  1))

        # Scroll arrows
        if max_scroll > 0:
            up_r   = pygame.Rect(sw - 22, content_top,               20, 20)
            down_r = pygame.Rect(sw - 22, content_top + content_h - 20, 20, 20)
            surface.set_clip(old_clip)
            for r, lbl in ((up_r, "▲"), (down_r, "▼")):
                pygame.draw.rect(surface, (30, 20, 50), r, border_radius=3)
                pygame.draw.rect(surface, (90, 60, 140), r, 1, border_radius=3)
                as_ = self._font_sm.render(lbl, True, (160, 140, 200))
                surface.blit(as_, as_.get_rect(center=r.center))
            self._panel_rects.append((up_r,   "scroll_up"))
            self._panel_rects.append((down_r, "scroll_down"))
        else:
            surface.set_clip(old_clip)
