"""
Fire Mage — a heuristic-controlled AI minion.

Behaviour:
  - Floats slowly toward enemies, maintaining a preferred range.
  - Shoots explosive fireballs at the nearest enemy within attack range.
  - Fireball impact triggers an AoE explosion that applies a Burn status
    (continuous damage over time) to all hit enemies.

Visual: floating robed figure with glowing fire eyes and an orbiting spark.
enemy_type: N/A (it is a minion, not an enemy).
minion_type = "fire_mage"   (used by BattleScene for dispatch).
"""
from __future__ import annotations
import math
import pygame
from config import CFG

_FMC = CFG.get("fire_mage", {})

_ROBE_COL  = (160,  30,  10)
_HEAD_COL  = (200,  60,  20)
_EYE_COL   = (255, 160,  20)
_SPARK_COL = (255, 120,  30)


class FireMage:
    minion_type = "fire_mage"

    def __init__(self, pos):
        self.pos            = pygame.Vector2(pos)
        self.hp             = int(_FMC.get("hp", 50))
        self.max_hp         = self.hp
        self.speed          = float(_FMC.get("speed", 65.0))
        self.size           = int(_FMC.get("size", 20))
        self.is_alive       = True

        # Heuristic ranging
        self.attack_range   = float(_FMC.get("attack_range",   260.0))
        self.preferred_range = float(_FMC.get("preferred_range", 210.0))
        self.min_safe_dist  = float(_FMC.get("min_safe_dist",   80.0))

        # Shooting
        self._shoot_cd      = float(_FMC.get("fireball_cooldown", 2.5))
        self._shoot_timer   = 0.0
        self._cast_flash    = 0.0   # brief visual flash on cast (seconds)
        self._cast_angle    = 0.0   # direction of last cast (for draw)

        # Physics
        self.velocity       = pygame.Vector2(0, 0)
        self.knockback_vel  = pygame.Vector2(0, 0)
        self.frozen_timer   = 0.0

        # MP (magic points) — gates shooting; regenerates over time
        self.max_stamina    = float(_FMC.get("mp", 80.0))
        self.stamina        = self.max_stamina
        self.stamina_regen  = float(_FMC.get("mp_regen", 12.0))
        self.stamina_cost   = float(_FMC.get("mp_cost", 35.0))
        self.attack_damage  = int(_FMC.get("fireball_damage", 22))
        self.attack_cooldown = self._shoot_cd

        # DQN integration
        self.last_action    = 0   # most recent DQN action (read by ally obs)

        # Animation
        self._anim_timer    = 0.0

    # ------------------------------------------------------------------
    # Per-frame update (call from BattleScene before movement)
    # ------------------------------------------------------------------

    def tick(self, dt: float):
        if not self.is_alive:
            return
        self._anim_timer += dt
        if self._shoot_timer > 0:
            self._shoot_timer = max(0.0, self._shoot_timer - dt)
        if self._cast_flash > 0:
            self._cast_flash = max(0.0, self._cast_flash - dt)
        # Regen MP
        self.stamina = min(self.max_stamina, self.stamina + self.stamina_regen * dt)
        # Tick freeze
        if self.frozen_timer > 0:
            self.frozen_timer = max(0.0, self.frozen_timer - dt)

    # ------------------------------------------------------------------
    # Heuristic velocity update — called by BattleScene each frame
    # ------------------------------------------------------------------

    def update_velocity(self, enemies: list, boss=None):
        """Set self.velocity toward/away from nearest enemy to maintain preferred range."""
        if not self.is_alive or self.frozen_timer > 0:
            self.velocity.update(0, 0)
            return

        candidates = [e for e in enemies if e.is_alive]
        if boss is not None and getattr(boss, 'is_alive', False):
            candidates.append(boss)

        if not candidates:
            self.velocity.update(0, 0)
            return

        target = min(candidates, key=lambda e: self.pos.distance_to(e.pos))
        diff   = target.pos - self.pos
        dist   = diff.length()
        if dist < 0.1:
            self.velocity.update(0, 0)
            return

        norm = diff.normalize()
        if dist < self.min_safe_dist:
            self.velocity = -norm * self.speed
        elif dist > self.preferred_range:
            self.velocity = norm * self.speed
        else:
            # Strafe perpendicular
            perp = pygame.Vector2(-norm.y, norm.x)
            self.velocity = perp * self.speed * 0.6

    # ------------------------------------------------------------------
    # Shooting
    # ------------------------------------------------------------------

    def try_shoot_aimed(self, base_angle: float, enemies: list, boss=None) -> "FireMageFireball | None":
        """DQN-directed shoot: aim at nearest enemy within a 90° cone of base_angle.
        Falls back to nearest enemy overall if none are in the cone."""
        from entities.mage_projectile import FireMageFireball
        if self._shoot_timer > 0 or self.stamina < self.stamina_cost:
            return None

        candidates = [e for e in enemies if e.is_alive
                      and self.pos.distance_to(e.pos) <= self.attack_range]
        if boss is not None and getattr(boss, 'is_alive', False):
            if self.pos.distance_to(boss.pos) <= self.attack_range:
                candidates.append(boss)
        if not candidates:
            return None

        CONE_HALF = math.pi / 2   # 90° half-angle
        in_cone = [c for c in candidates
                   if abs(math.atan2(
                       math.sin(math.atan2(c.pos.y - self.pos.y,
                                           c.pos.x - self.pos.x) - base_angle),
                       math.cos(math.atan2(c.pos.y - self.pos.y,
                                           c.pos.x - self.pos.x) - base_angle)
                   )) <= CONE_HALF]
        target_pool = in_cone if in_cone else candidates
        target = min(target_pool, key=lambda e: self.pos.distance_to(e.pos))
        angle  = math.atan2(target.pos.y - self.pos.y, target.pos.x - self.pos.x)

        self._shoot_timer = self._shoot_cd
        self._cast_flash  = 0.35
        self._cast_angle  = angle
        self.stamina     -= self.stamina_cost

        return FireMageFireball(
            pos=(self.pos.x, self.pos.y),
            angle=angle,
            speed=float(_FMC.get("fireball_speed", 310.0)),
        )

    def try_shoot(self, enemies: list, boss=None) -> "FireMageFireball | None":
        from entities.mage_projectile import FireMageFireball
        if self._shoot_timer > 0 or self.stamina < self.stamina_cost:
            return None

        candidates = [e for e in enemies if e.is_alive
                      and self.pos.distance_to(e.pos) <= self.attack_range]
        if boss is not None and getattr(boss, 'is_alive', False):
            if self.pos.distance_to(boss.pos) <= self.attack_range:
                candidates.append(boss)

        if not candidates:
            return None

        target = min(candidates, key=lambda e: self.pos.distance_to(e.pos))
        angle  = math.atan2(target.pos.y - self.pos.y, target.pos.x - self.pos.x)

        self._shoot_timer = self._shoot_cd
        self._cast_flash  = 0.35
        self._cast_angle  = angle
        self.stamina     -= self.stamina_cost

        return FireMageFireball(
            pos=(self.pos.x, self.pos.y),
            angle=angle,
            speed=float(_FMC.get("fireball_speed", 310.0)),
        )

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface):
        cx, cy = int(self.pos.x), int(self.pos.y)
        r      = self.size // 2

        if not self.is_alive:
            pygame.draw.rect(surface, (80, 80, 80),
                             (cx - r + 2, cy - r, r * 2 - 4, r * 2), border_radius=5)
            pygame.draw.rect(surface, (60, 60, 60),
                             (cx - r + 2, cy - r, r * 2 - 4, r * 2), 2, border_radius=5)
            _lbl = pygame.font.SysFont("arial", 9, bold=True).render("FM", True, (160, 50, 20))
            surface.blit(_lbl, _lbl.get_rect(center=(cx, cy)))
            return

        # Floating bob
        bob   = math.sin(self._anim_timer * 2.5) * 2.0
        cy_f  = cy + int(bob)

        # Cast glow aura
        if self._cast_flash > 0:
            t    = self._cast_flash / 0.35
            gr   = r + int(10 * t)
            gsurf = pygame.Surface((gr * 2 + 4, gr * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(gsurf, (255, 100, 20, int(130 * t)), (gr + 2, gr + 2), gr)
            surface.blit(gsurf, (cx - gr - 2, cy_f - gr - 2))

        # Robe body
        pygame.draw.circle(surface, _ROBE_COL, (cx, cy_f + 3), r)
        # Head
        pygame.draw.circle(surface, _HEAD_COL, (cx, cy_f - r // 2), r // 2 + 1)
        # Glowing fire eyes
        pulse = 0.5 + 0.5 * math.sin(self._anim_timer * 5.0)
        eye_r = int(200 + 55 * pulse)
        eye_g = int(80 * pulse)
        pygame.draw.circle(surface, (eye_r, eye_g, 0), (cx - 3, cy_f - r // 2 - 1), 2)
        pygame.draw.circle(surface, (eye_r, eye_g, 0), (cx + 3, cy_f - r // 2 - 1), 2)
        # "FM" label on robe
        lbl = pygame.font.SysFont("arial", 8, bold=True).render("FM", True, (255, 200, 100))
        surface.blit(lbl, lbl.get_rect(center=(cx, cy_f + 3)))

        # Orbiting fire spark
        sa = self._anim_timer * 3.5
        sx = cx + int(math.cos(sa) * (r + 5))
        sy = cy_f + int(math.sin(sa) * (r + 5))
        pygame.draw.circle(surface, _SPARK_COL, (sx, sy), 3)
        pygame.draw.circle(surface, (255, 220, 80), (sx, sy), 1)

        # Cast beam
        if self._cast_flash > 0:
            t  = self._cast_flash / 0.35
            ex = cx + int(math.cos(self._cast_angle) * (r + 18))
            ey = cy_f + int(math.sin(self._cast_angle) * (r + 18))
            pygame.draw.line(surface, (255, int(180 * t), 0), (cx, cy_f), (ex, ey), 2)

        # Freeze overlay
        if self.frozen_timer > 0:
            ice = pygame.Surface((self.size + 8, self.size + 8), pygame.SRCALPHA)
            ice.fill((80, 160, 255, 100))
            surface.blit(ice, (cx - (self.size + 8) // 2, cy_f - (self.size + 8) // 2))

        # HP bar
        bw = self.size + 8
        bh = 4
        bx = cx - bw // 2
        by = cy_f - r - 14
        pygame.draw.rect(surface, (120, 30, 30), (bx, by, bw, bh))
        fw = int(bw * max(0, self.hp) / max(1, self.max_hp))
        if fw > 0:
            col = (50, 200, 80) if self.hp > self.max_hp * 0.4 else (220, 100, 30)
            pygame.draw.rect(surface, col, (bx, by, fw, bh))

        # MP bar (above HP bar — magenta/purple)
        mp_by = by - 6
        pygame.draw.rect(surface, (50, 10, 50), (bx, mp_by, bw, bh))
        mp_fill = int(bw * max(0, self.stamina) / max(1, self.max_stamina))
        if mp_fill > 0:
            ratio = self.stamina / self.max_stamina
            mp_col = (int(200 * ratio + 55), int(20 * ratio), int(220 * ratio))
            pygame.draw.rect(surface, mp_col, (bx, mp_by, mp_fill, bh))
