"""
Ice Mage — a heuristic-controlled AI minion.

Behaviour:
  - Floats slowly toward enemies, maintaining a preferred range.
  - Shoots iceballs at the nearest enemy within attack range.
  - Iceball impact deals low direct damage and applies a Freeze status
    (immobilises the target for a short duration).

Visual: floating robed figure with icy blue eyes and orbiting ice crystal.
minion_type = "ice_mage"   (used by BattleScene for dispatch).
"""
from __future__ import annotations
import math
import pygame
from config import CFG

_IMC = CFG.get("ice_mage", {})

_ROBE_COL    = ( 20,  60, 160)
_HEAD_COL    = ( 40,  90, 200)
_EYE_COL     = (160, 230, 255)
_CRYSTAL_COL = (180, 220, 255)


class IceMage:
    minion_type = "ice_mage"

    def __init__(self, pos):
        self.pos            = pygame.Vector2(pos)
        self.hp             = int(_IMC.get("hp", 50))
        self.max_hp         = self.hp
        self.speed          = float(_IMC.get("speed", 65.0))
        self.size           = int(_IMC.get("size", 20))
        self.is_alive       = True

        # Heuristic ranging
        self.attack_range   = float(_IMC.get("attack_range",    240.0))
        self.preferred_range = float(_IMC.get("preferred_range", 200.0))
        self.min_safe_dist  = float(_IMC.get("min_safe_dist",   80.0))

        # Shooting
        self._shoot_cd      = float(_IMC.get("iceball_cooldown", 3.0))
        self._shoot_timer   = 0.0
        self._cast_flash    = 0.0
        self._cast_angle    = 0.0

        # Physics
        self.velocity       = pygame.Vector2(0, 0)
        self.knockback_vel  = pygame.Vector2(0, 0)
        self.frozen_timer   = 0.0

        # Vector obs compat
        self.stamina        = 100.0
        self.max_stamina    = 100.0
        self.attack_damage  = int(_IMC.get("iceball_damage", 15))
        self.attack_cooldown = self._shoot_cd

        # DQN integration
        self.last_action    = 0   # most recent DQN action

        # Animation
        self._anim_timer    = 0.0

    # ------------------------------------------------------------------

    def tick(self, dt: float):
        if not self.is_alive:
            return
        self._anim_timer += dt
        if self._shoot_timer > 0:
            self._shoot_timer = max(0.0, self._shoot_timer - dt)
        if self._cast_flash > 0:
            self._cast_flash = max(0.0, self._cast_flash - dt)
        if self.frozen_timer > 0:
            self.frozen_timer = max(0.0, self.frozen_timer - dt)

    # ------------------------------------------------------------------

    def update_velocity(self, enemies: list, boss=None):
        """Set self.velocity toward/away from nearest enemy."""
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
            perp = pygame.Vector2(-norm.y, norm.x)
            self.velocity = perp * self.speed * 0.6

    # ------------------------------------------------------------------

    def try_shoot_aimed(self, base_angle: float, enemies: list, boss=None) -> "IceMageIceball | None":
        """DQN-directed shoot: aim at nearest enemy within a 90° cone of base_angle."""
        from entities.mage_projectile import IceMageIceball
        if self._shoot_timer > 0:
            return None

        candidates = [e for e in enemies if e.is_alive
                      and self.pos.distance_to(e.pos) <= self.attack_range]
        if boss is not None and getattr(boss, 'is_alive', False):
            if self.pos.distance_to(boss.pos) <= self.attack_range:
                candidates.append(boss)
        if not candidates:
            return None

        CONE_HALF = math.pi / 2
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

        return IceMageIceball(
            pos=(self.pos.x, self.pos.y),
            angle=angle,
            speed=float(_IMC.get("iceball_speed", 290.0)),
        )

    def try_shoot(self, enemies: list, boss=None) -> "IceMageIceball | None":
        from entities.mage_projectile import IceMageIceball
        if self._shoot_timer > 0:
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

        return IceMageIceball(
            pos=(self.pos.x, self.pos.y),
            angle=angle,
            speed=float(_IMC.get("iceball_speed", 290.0)),
        )

    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface):
        cx, cy = int(self.pos.x), int(self.pos.y)
        r      = self.size // 2

        if not self.is_alive:
            pygame.draw.rect(surface, (60, 60, 90),
                             (cx - r + 2, cy - r, r * 2 - 4, r * 2), border_radius=5)
            pygame.draw.rect(surface, (40, 40, 70),
                             (cx - r + 2, cy - r, r * 2 - 4, r * 2), 2, border_radius=5)
            _lbl = pygame.font.SysFont("arial", 9, bold=True).render("IM", True, (80, 140, 220))
            surface.blit(_lbl, _lbl.get_rect(center=(cx, cy)))
            return

        bob  = math.sin(self._anim_timer * 2.2) * 2.0
        cy_f = cy + int(bob)

        # Cast glow aura
        if self._cast_flash > 0:
            t    = self._cast_flash / 0.35
            gr   = r + int(10 * t)
            gsurf = pygame.Surface((gr * 2 + 4, gr * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(gsurf, (100, 200, 255, int(130 * t)), (gr + 2, gr + 2), gr)
            surface.blit(gsurf, (cx - gr - 2, cy_f - gr - 2))

        # Robe body
        pygame.draw.circle(surface, _ROBE_COL, (cx, cy_f + 3), r)
        # Head
        pygame.draw.circle(surface, _HEAD_COL, (cx, cy_f - r // 2), r // 2 + 1)
        # Glowing icy eyes
        pulse = 0.5 + 0.5 * math.sin(self._anim_timer * 4.5)
        eb = int(200 + 55 * pulse)
        pygame.draw.circle(surface, (80, 180, eb), (cx - 3, cy_f - r // 2 - 1), 2)
        pygame.draw.circle(surface, (80, 180, eb), (cx + 3, cy_f - r // 2 - 1), 2)
        # "IM" label
        lbl = pygame.font.SysFont("arial", 8, bold=True).render("IM", True, (160, 220, 255))
        surface.blit(lbl, lbl.get_rect(center=(cx, cy_f + 3)))

        # Orbiting ice crystal (diamond shape)
        ca = self._anim_timer * -2.8   # counter-rotate
        kx = cx + int(math.cos(ca) * (r + 5))
        ky = cy_f + int(math.sin(ca) * (r + 5))
        hs = 3
        pts = [(kx, ky - hs), (kx + hs, ky), (kx, ky + hs), (kx - hs, ky)]
        pygame.draw.polygon(surface, _CRYSTAL_COL, pts)
        pygame.draw.polygon(surface, (255, 255, 255), pts, 1)

        # Cast beam
        if self._cast_flash > 0:
            t  = self._cast_flash / 0.35
            ex = cx + int(math.cos(self._cast_angle) * (r + 18))
            ey = cy_f + int(math.sin(self._cast_angle) * (r + 18))
            pygame.draw.line(surface, (120, 200, int(255 * t)), (cx, cy_f), (ex, ey), 2)

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
        pygame.draw.rect(surface, (30, 40, 100), (bx, by, bw, bh))
        fw = int(bw * max(0, self.hp) / max(1, self.max_hp))
        if fw > 0:
            col = (60, 180, 255) if self.hp > self.max_hp * 0.4 else (180, 100, 220)
            pygame.draw.rect(surface, col, (bx, by, fw, bh))
