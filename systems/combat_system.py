"""
Combat system — multi-minion, boss support.

Handles:
  - Fighter melee (cone attack, stamina-gated, DQN-directed).
  - Archer stamina regen (update() is called per archer inside BattleScene).
  - Enemy melee on nearest alive minion.
  - Boss melee on nearest alive minion.
  - Arrow projectile collisions (with swarms and boss).
  - Knockback impulses applied to all hit targets.
"""
import math
import pygame
from config import CFG

_PHYS = CFG.get("physics", {})
_KNOCKBACK_FORCE     = float(_PHYS.get("knockback_force",     260.0))
_MAX_KNOCKBACK_SPEED = float(_PHYS.get("max_knockback_speed", 600.0))
_BOSS_KB_SCALE       = float(_PHYS.get("boss_knockback_scale", 0.15))


def _apply_knockback(target, direction: pygame.Vector2, force: float):
    """Add a knockback impulse to target in the given direction."""
    kv = getattr(target, 'knockback_vel', None)
    if kv is None or direction.length_squared() == 0:
        return
    impulse = direction.normalize() * force
    kv += impulse
    spd = kv.length()
    if spd > _MAX_KNOCKBACK_SPEED:
        kv.scale_to_length(_MAX_KNOCKBACK_SPEED)


class CombatSystem:

    def update(self, dt: float,
               fighters: list,
               enemies: list,
               projectiles: list = None,
               archers: list = None,
               fighter_attack_dirs: list = None,
               boss=None,
               mages: list = None):
        """
        fighters            : list of Fighter objects
        enemies             : list of swarm Enemy objects
        projectiles         : list of Projectile objects
        archers             : list of Archer objects
        fighter_attack_dirs : list matching fighters — (dx,dy) or None per fighter
        boss                : optional Boss (can receive damage from arrows/sword)
        mages               : optional list of FireMage/IceMage objects (for enemy targeting)

        Returns events dict.
        """
        archers             = archers             or []
        projectiles         = projectiles         or []
        fighter_attack_dirs = fighter_attack_dirs or [None] * len(fighters)
        mages               = mages               or []

        events = {
            "enemies_killed":      0,
            "damage_dealt":        0.0,
            "damage_taken":        0.0,
            "archer_damage_taken": 0.0,
            "sword_attacks":       0,
            "sword_damage":        0.0,
            "sword_kills":         0,
            "sword_hit_swings":    0,
            "sword_miss":          False,
            "archer_kills":        0,
            "archer_damage_dealt": 0.0,
            "archer_miss":         False,
            "arrow_hits":          0,
            "boss_damage_dealt":   0.0,
            "boss_killed":         False,
            "hits":                [],
        }

        all_alive_minions = [m for m in fighters + archers + mages if m.is_alive]

        # ── Regen stamina for all alive fighters ───────────────────────────
        for f in fighters:
            if f.is_alive:
                f.stamina = min(f.max_stamina, f.stamina + f.stamina_regen * dt)

        # ── Tick flash timers ──────────────────────────────────────────────
        for f in fighters:
            if f.attack_flash_timer > 0:
                f.attack_flash_timer = max(0.0, f.attack_flash_timer - dt)
        for e in enemies:
            if e.is_alive and e.attack_flash_timer > 0:
                e.attack_flash_timer = max(0.0, e.attack_flash_timer - dt)

        # ── Tick attack cooldowns ──────────────────────────────────────────
        for f in fighters:
            if f.attack_timer > 0:
                f.attack_timer = max(0.0, f.attack_timer - dt)
        for e in enemies:
            if e.is_alive and e.attack_timer > 0:
                e.attack_timer = max(0.0, e.attack_timer - dt)
        if boss and boss.is_alive:
            if boss.attack_timer > 0:
                boss.attack_timer = max(0.0, boss.attack_timer - dt)
            if boss.attack_flash_timer > 0:
                boss.attack_flash_timer = max(0.0, boss.attack_flash_timer - dt)

        # ── Fighter sword swings ───────────────────────────────────────────
        for fighter, atk_dir in zip(fighters, fighter_attack_dirs):
            if (not fighter.is_alive
                    or atk_dir is None
                    or fighter.attack_timer > 0
                    or fighter.stamina < fighter.stamina_cost):
                continue

            dx, dy = atk_dir
            swing_angle = math.atan2(dy, dx)
            fighter.attack_flash_angle = swing_angle
            fighter.attack_flash_timer = 0.2
            fighter.attack_timer       = fighter.attack_cooldown
            fighter.stamina           -= fighter.stamina_cost
            events["sword_attacks"]   += 1

            half_arc     = math.radians(35)
            swing_hits   = 0

            # Hit swarms
            # Use center-to-center distance minus enemy radius so that the edge
            # of the enemy body counts as in-range (matches the visual arc).
            for enemy in enemies:
                if not enemy.is_alive:
                    continue
                d = fighter.pos.distance_to(enemy.pos)
                if d > fighter.attack_range + enemy.size // 2:
                    continue
                e_angle = math.atan2(enemy.pos.y - fighter.pos.y,
                                     enemy.pos.x - fighter.pos.x)
                diff = abs(math.atan2(math.sin(e_angle - swing_angle),
                                      math.cos(e_angle - swing_angle)))
                if diff <= half_arc:
                    enemy.hp            -= fighter.attack_damage
                    events["damage_dealt"] += fighter.attack_damage
                    events["sword_damage"] += fighter.attack_damage
                    swing_hits             += 1
                    events["hits"].append((enemy.pos.x, enemy.pos.y - 14,
                                           fighter.attack_damage, (255, 230, 80)))
                    # Knockback: push enemy away from fighter
                    _apply_knockback(enemy, enemy.pos - fighter.pos, _KNOCKBACK_FORCE)
                    if enemy.hp <= 0:
                        enemy.is_alive          = False
                        events["enemies_killed"] += 1
                        events["sword_kills"]    += 1

            # Hit boss
            if boss and boss.is_alive:
                d = fighter.pos.distance_to(boss.pos)
                if d <= fighter.attack_range + boss.size // 2:
                    b_angle = math.atan2(boss.pos.y - fighter.pos.y,
                                         boss.pos.x - fighter.pos.x)
                    diff = abs(math.atan2(math.sin(b_angle - swing_angle),
                                          math.cos(b_angle - swing_angle)))
                    if diff <= half_arc:
                        boss.take_damage(fighter.attack_damage)
                        events["boss_damage_dealt"] += fighter.attack_damage
                        events["damage_dealt"]      += fighter.attack_damage
                        swing_hits                  += 1
                        events["hits"].append((boss.pos.x, boss.pos.y - 20,
                                               fighter.attack_damage, (255, 230, 80)))
                        # Boss receives a reduced knockback (it's very large)
                        _apply_knockback(boss, boss.pos - fighter.pos,
                                         _KNOCKBACK_FORCE * _BOSS_KB_SCALE)
                        if not boss.is_alive:
                            events["boss_killed"] = True

            if swing_hits > 0:
                events["sword_hit_swings"] += 1
            else:
                events["sword_miss"] = True

        # ── Enemy melee attacks on nearest alive minion ────────────────────
        for enemy in enemies:
            if not enemy.is_alive or enemy.attack_timer > 0:
                continue
            # Only melee-capable types: Swarm (0) and Slime (2)
            # Spider (1) uses ranged; Creeper (4) explodes instead
            if getattr(enemy, 'enemy_type', 0) not in (0, 2):
                continue
            if not all_alive_minions:
                continue

            nearest = min(all_alive_minions, key=lambda m: enemy.pos.distance_to(m.pos))
            d       = enemy.pos.distance_to(nearest.pos)
            if d <= enemy.attack_range:
                nearest.hp        -= enemy.attack_damage
                nearest.attack_timer = 0   # irrelevant but keep struct clean
                if nearest in fighters:
                    key = "damage_taken"
                elif nearest in archers:
                    key = "archer_damage_taken"
                else:
                    key = "mage_damage_taken"
                events[key] = events.get(key, 0.0) + enemy.attack_damage
                events["hits"].append((nearest.pos.x, nearest.pos.y - 14,
                                       enemy.attack_damage, (255, 80, 80)))
                # Knockback: push minion away from attacker
                _apply_knockback(nearest, nearest.pos - enemy.pos,
                                 _KNOCKBACK_FORCE * 0.5)
                enemy.attack_timer      = enemy.attack_cooldown
                enemy.attack_flash_timer = 0.15
                if nearest.hp <= 0:
                    nearest.is_alive = False
                    all_alive_minions = [m for m in all_alive_minions if m.is_alive]

        # ── Boss melee attack on nearest alive minion ──────────────────────
        if boss and boss.is_alive and boss.attack_timer <= 0:
            if all_alive_minions:
                nearest = min(all_alive_minions,
                              key=lambda m: boss.pos.distance_to(m.pos))
                d = boss.pos.distance_to(nearest.pos)
                if d <= boss.attack_range:
                    nearest.hp -= boss.attack_damage
                    if nearest in fighters:
                        key = "damage_taken"
                    elif nearest in archers:
                        key = "archer_damage_taken"
                    else:
                        key = "mage_damage_taken"
                    events[key] = events.get(key, 0.0) + boss.attack_damage
                    events["hits"].append((nearest.pos.x, nearest.pos.y - 18,
                                           boss.attack_damage, (255, 80, 200)))
                    # Boss melee hits hard — full knockback force
                    _apply_knockback(nearest, nearest.pos - boss.pos, _KNOCKBACK_FORCE)
                    boss.attack_timer      = boss.attack_cooldown
                    boss.attack_flash_timer = 0.15
                    if nearest.hp <= 0:
                        nearest.is_alive = False
                        all_alive_minions = [m for m in all_alive_minions if m.is_alive]

        # ── Projectile (arrow) collisions ──────────────────────────────────
        for proj in projectiles:
            if not proj.is_alive:
                continue

            # Check swarms
            for enemy in enemies:
                if not enemy.is_alive:
                    continue
                hit_dist = (enemy.size + proj.size) // 2 + 2
                if proj.pos.distance_to(enemy.pos) <= hit_dist:
                    enemy.hp                    -= proj.damage
                    proj.is_alive                = False
                    proj.hit_enemy               = True
                    events["damage_dealt"]       += proj.damage
                    events["archer_damage_dealt"] += proj.damage
                    events["arrow_hits"]         += 1
                    events["hits"].append((enemy.pos.x, enemy.pos.y - 14,
                                           proj.damage, (120, 200, 255)))
                    # Knockback in projectile travel direction
                    _apply_knockback(enemy, proj.vel, _KNOCKBACK_FORCE)
                    if enemy.hp <= 0:
                        enemy.is_alive           = False
                        events["enemies_killed"] += 1
                        events["archer_kills"]   += 1
                    break

            if not proj.is_alive:
                continue

            # Check boss
            if boss and boss.is_alive:
                hit_dist = (boss.size // 2 + proj.size) + 4
                if proj.pos.distance_to(boss.pos) <= hit_dist:
                    boss.take_damage(proj.damage)
                    proj.is_alive               = False
                    proj.hit_enemy              = True
                    events["boss_damage_dealt"] += proj.damage
                    events["damage_dealt"]      += proj.damage
                    events["arrow_hits"]        += 1
                    events["hits"].append((boss.pos.x, boss.pos.y - 20,
                                           proj.damage, (120, 200, 255)))
                    _apply_knockback(boss, proj.vel,
                                     _KNOCKBACK_FORCE * _BOSS_KB_SCALE)
                    if not boss.is_alive:
                        events["boss_killed"] = True

        return events
