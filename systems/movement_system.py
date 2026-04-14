"""
Movement system — handles multi-minion lists and an optional boss entity.

Swarm enemies always chase the nearest alive minion.
Spider enemies (enemy_type == 1) keep a preferred distance, flee when too
close, and strafe when at their preferred range.
Frozen minions (frozen_timer > 0) are held in place for the duration, but
still receive knockback displacement.

Physics additions:
  - Knockback: entities with knockback_vel have that velocity applied each
    frame then exponentially decayed.
  - Separation: after all movement, overlapping entities are pushed apart so
    they never visually stack on top of each other.
"""
import pygame
from config import CFG

_PHYS = CFG.get("physics", {})
_KNOCKBACK_DECAY     = float(_PHYS.get("knockback_decay",     12.0))
_MAX_KNOCKBACK_SPEED = float(_PHYS.get("max_knockback_speed", 600.0))
_SEP_BUFFER          = float(_PHYS.get("separation_radius_buffer", 4))
_SEP_FORCE           = float(_PHYS.get("separation_force", 1.0))


class MovementSystem:
    def update(self, dt: float,
               fighters: list,
               archers: list,
               enemies: list,
               arena_bounds: tuple,
               boss=None):
        """
        fighters      : list of Fighter objects
        archers       : list of Archer objects
        enemies       : list of Enemy / Spider objects (mixed)
        arena_bounds  : (left, top, right, bottom)
        boss          : optional Boss object (moves itself in its own update())
        """
        left, top, right, bottom = arena_bounds
        all_minions = [m for m in fighters + archers if m.is_alive]
        all_enemies = [e for e in enemies if e.is_alive]

        # ── Apply & decay knockback for all living entities ───────────────
        for entity in all_minions + all_enemies:
            kv = entity.knockback_vel
            if kv.x != 0 or kv.y != 0:
                # Clamp magnitude before applying
                spd = kv.length()
                if spd > _MAX_KNOCKBACK_SPEED:
                    kv.scale_to_length(_MAX_KNOCKBACK_SPEED)
                entity.pos += kv * dt
                # Clamp position
                half = entity.size // 2
                entity.pos.x = max(left + half, min(right - half, entity.pos.x))
                entity.pos.y = max(top  + half, min(bottom - half, entity.pos.y))
                # Exponential decay
                factor = max(0.0, 1.0 - _KNOCKBACK_DECAY * dt)
                kv.x *= factor
                kv.y *= factor
                if kv.length_squared() < 0.5:
                    kv.update(0, 0)

        # ── Move each alive Fighter/Archer ───────────────────────────────
        for minion in fighters + archers:
            if not minion.is_alive:
                continue

            # Tick freeze timer; block movement while frozen
            if minion.frozen_timer > 0:
                minion.frozen_timer = max(0.0, minion.frozen_timer - dt)
                half = minion.size // 2
                minion.pos.x = max(left + half, min(right - half, minion.pos.x))
                minion.pos.y = max(top  + half, min(bottom - half, minion.pos.y))
                continue

            minion.pos += minion.velocity * dt
            half = minion.size // 2
            minion.pos.x = max(left + half, min(right - half, minion.pos.x))
            minion.pos.y = max(top  + half, min(bottom - half, minion.pos.y))

        # ── Move each alive enemy ────────────────────────────────────────
        for enemy in enemies:
            if not enemy.is_alive:
                continue

            etype = getattr(enemy, 'enemy_type', 0)

            if etype == 1:
                # ── Spider: keep preferred distance, strafe, flee ────────
                self._move_spider(enemy, dt, all_minions, left, top, right, bottom)
            else:
                # ── Swarm: chase nearest alive minion ────────────────────
                target      = None
                target_dist = float("inf")
                for candidate in all_minions:
                    d = enemy.pos.distance_to(candidate.pos)
                    if d < target_dist:
                        target      = candidate
                        target_dist = d

                if target is None:
                    enemy.velocity = pygame.Vector2(0, 0)
                    continue

                diff = target.pos - enemy.pos
                if diff.length() > 0:
                    enemy.velocity = diff.normalize() * enemy.speed
                    enemy.pos     += enemy.velocity * dt
                else:
                    enemy.velocity = pygame.Vector2(0, 0)

                half_e = enemy.size // 2
                enemy.pos.x = max(left + half_e, min(right - half_e, enemy.pos.x))
                enemy.pos.y = max(top  + half_e, min(bottom - half_e, enemy.pos.y))

        # ── Separation pass — push overlapping entities apart ────────────
        self._apply_separation(all_minions, all_enemies, left, top, right, bottom)

    # ------------------------------------------------------------------
    # Spider movement helper
    # ------------------------------------------------------------------

    @staticmethod
    def _move_spider(spider, dt: float, all_minions: list,
                     left: float, top: float, right: float, bottom: float):
        """
        Spider maintains a preferred distance from the nearest alive minion:
          - Too close (< min_safe_dist): flee directly away
          - Too far  (> preferred_dist): approach
          - In range (between safe & preferred): strafe perpendicular to target
        Wall repulsion blended into all three modes.
        """
        half_e = spider.size // 2

        if not all_minions:
            spider.velocity = pygame.Vector2(0, 0)
            return

        target = min(all_minions, key=lambda m: spider.pos.distance_to(m.pos))
        diff   = target.pos - spider.pos
        dist   = diff.length()
        if dist == 0:
            spider.velocity = pygame.Vector2(0, 0)
            return

        norm_diff = diff.normalize()
        pref      = spider.preferred_dist
        safe      = spider.min_safe_dist

        # Wall repulsion
        wx, wy = 0.0, 0.0
        wall_d = 70.0
        gap_l  = spider.pos.x - left
        gap_r  = right - spider.pos.x
        gap_t  = spider.pos.y - top
        gap_b  = bottom - spider.pos.y
        if gap_l < wall_d: wx += (wall_d - gap_l)
        if gap_r < wall_d: wx -= (wall_d - gap_r)
        if gap_t < wall_d: wy += (wall_d - gap_t)
        if gap_b < wall_d: wy -= (wall_d - gap_b)
        wall_push = pygame.Vector2(wx, wy)

        if dist < safe:
            move_dir = -norm_diff
            if wall_push.length() > 0:
                move_dir += wall_push.normalize() * 0.4
        elif dist > pref:
            move_dir = norm_diff
            if wall_push.length() > 0:
                move_dir += wall_push.normalize() * 0.3
        else:
            perp = pygame.Vector2(-norm_diff.y, norm_diff.x)
            move_dir = perp
            if wall_push.length() > 0:
                move_dir += wall_push.normalize() * 0.5

        if move_dir.length() > 0:
            spider.velocity = move_dir.normalize() * spider.speed
        else:
            spider.velocity = pygame.Vector2(0, 0)

        spider.pos += spider.velocity * dt
        spider.pos.x = max(left + half_e, min(right - half_e, spider.pos.x))
        spider.pos.y = max(top  + half_e, min(bottom - half_e, spider.pos.y))

    # ------------------------------------------------------------------
    # Separation helper — prevents entities from visually stacking
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_separation(all_minions: list, all_enemies: list,
                          left: float, top: float, right: float, bottom: float):
        """
        Push overlapping entities apart.  Checks three groups of pairs:
          - minion vs minion
          - enemy  vs enemy
          - minion vs enemy
        Each entity in an overlapping pair is nudged half the overlap distance
        in opposite directions.
        """
        def _separate_pair(a, b):
            diff = a.pos - b.pos
            dist = diff.length()
            min_dist = (a.size + b.size) / 2.0 + _SEP_BUFFER
            if dist < min_dist:
                if dist < 0.01:
                    # Perfectly overlapping — nudge slightly off-axis
                    diff = pygame.Vector2(0.5, 0.5)
                    dist = diff.length()
                overlap = min_dist - dist
                push = diff.normalize() * (overlap * 0.5 * _SEP_FORCE)
                a.pos += push
                b.pos -= push
                # Clamp both to arena
                ha = a.size // 2
                a.pos.x = max(left + ha, min(right - ha, a.pos.x))
                a.pos.y = max(top  + ha, min(bottom - ha, a.pos.y))
                hb = b.size // 2
                b.pos.x = max(left + hb, min(right - hb, b.pos.x))
                b.pos.y = max(top  + hb, min(bottom - hb, b.pos.y))

        n_m = len(all_minions)
        n_e = len(all_enemies)

        # minion–minion
        for i in range(n_m):
            for j in range(i + 1, n_m):
                _separate_pair(all_minions[i], all_minions[j])

        # enemy–enemy
        for i in range(n_e):
            for j in range(i + 1, n_e):
                _separate_pair(all_enemies[i], all_enemies[j])

        # minion–enemy
        for m in all_minions:
            for e in all_enemies:
                _separate_pair(m, e)
