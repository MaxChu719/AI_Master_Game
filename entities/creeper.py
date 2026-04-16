"""
Creeper enemy — inspired by Minecraft.

Behaviour:
  - Walks toward the nearest alive AI minion.
  - Explodes when either:
      (a) it takes any damage from an attack, OR
      (b) it gets within trigger_range pixels of a minion.
  - Explosion deals AoE damage to all nearby minions.
  - Before exploding it flashes (alternates bright green / black) to warn
    the player.

enemy_type = 4  (for MinionEnv observation encoding)
"""
from __future__ import annotations
import math
import random
import pygame
from config import CFG

_CC = CFG.get("creeper", {})

_HP             = int(_CC.get("hp",               60))
_SPEED          = float(_CC.get("speed",          85.0))
_SIZE           = int(_CC.get("size",             22))
_EXP_RADIUS     = float(_CC.get("explosion_radius", 100.0))
_EXP_DAMAGE     = int(_CC.get("explosion_damage", 55))
_TRIGGER_RANGE  = float(_CC.get("trigger_range",  35.0))
_FUSE_FLASH_RATE = float(_CC.get("fuse_flash_rate", 8.0))

_BODY_COL  = (60, 160, 50)
_DARK_COL  = (30,  90, 20)
_FLASH_COL = (220, 255, 50)


class Creeper:
    enemy_type = 4

    def __init__(self, pos):
        self.pos             = pygame.Vector2(pos)
        self.hp              = _HP
        self.max_hp          = _HP
        self.speed           = _SPEED
        self.size            = _SIZE
        self.is_alive        = True
        self.attack_damage   = 0      # no normal melee — explosion only
        self.attack_range    = 0.0
        self.attack_cooldown = 999.0
        self.attack_timer    = 0.0
        self.attack_flash_timer = 0.0
        self.explosion_radius = _EXP_RADIUS
        self.explosion_damage = _EXP_DAMAGE
        self.trigger_range   = _TRIGGER_RANGE

        # Status effects
        self.frozen_timer    = 0.0
        self.burn_timer      = 0.0
        self.burn_dps        = 0.0

        # Physics
        self.velocity        = pygame.Vector2(0, 0)
        self.knockback_vel   = pygame.Vector2(0, 0)

        # Explosion state
        self.should_explode  = False    # set by BattleScene when triggered
        self._prev_hp        = _HP      # for damage-detection
        self._fuse_timer     = 0.0      # counts up to show fuse animation
        self._exploding      = False    # brief pre-explosion flash phase
        self._explode_flash  = 0.0     # timer for white flash just before boom

        # Animation
        self._walk_timer     = 0.0
        self._leg_phase      = 0.0
        # Grave timer: -1 = uninitialized; >0 = showing grave; 0 = expired
        self.grave_timer     = -1.0

    # ------------------------------------------------------------------

    def tick(self, dt: float, all_minions: list):
        """Update timers and check proximity trigger."""
        if not self.is_alive:
            return
        self._walk_timer += dt

        # Proximity check
        alive_minions = [m for m in all_minions if m.is_alive]
        if alive_minions:
            nearest_d = min(m.pos.distance_to(self.pos) for m in alive_minions)
            if nearest_d <= self.trigger_range:
                self.should_explode = True

        # Damage-received check (hp was lowered by combat system)
        if self.hp < self._prev_hp:
            self.should_explode = True
        self._prev_hp = self.hp

        # Fuse animation when triggered
        if self.should_explode:
            self._fuse_timer += dt
            self._explode_flash += dt

    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface):
        if not self.is_alive:
            if self.grave_timer > 0:
                from entities.mage_projectile import draw_enemy_grave
                draw_enemy_grave(surface, self.pos, self.size,
                                 stone=(62, 88, 55),
                                 alpha=min(1.0, self.grave_timer / 0.5))
            return

        cx, cy = int(self.pos.x), int(self.pos.y)
        r      = self.size // 2

        # Flash when about to explode
        if self.should_explode:
            flash_ratio = math.sin(self._explode_flash * _FUSE_FLASH_RATE * math.pi)
            flash_on = flash_ratio > 0
            body_col = _FLASH_COL if flash_on else (10, 10, 10)
            dark_col = _FLASH_COL if flash_on else _DARK_COL
        else:
            body_col = _BODY_COL
            dark_col = _DARK_COL

        # Walking sway
        sway = int(math.sin(self._walk_timer * 5.0) * 1.5)

        # Body (rectangle with rounded corners approximated by a circle stack)
        body_rect = pygame.Rect(cx - r + sway, cy - r, r * 2, r * 2)
        pygame.draw.rect(surface, body_col, body_rect, border_radius=4)
        pygame.draw.rect(surface, dark_col, body_rect, 2, border_radius=4)

        # Face: two square "pixel" eyes and mouth
        eye_size = max(2, r // 3)
        eye_y    = cy - r // 3
        pygame.draw.rect(surface, (0, 0, 0),
                         (cx - r // 2 - eye_size + sway, eye_y, eye_size * 2, eye_size * 2))
        pygame.draw.rect(surface, (0, 0, 0),
                         (cx + r // 2 - eye_size + sway, eye_y, eye_size * 2, eye_size * 2))

        # Mouth (pixelated horizontal bar)
        mouth_y = cy + r // 5
        pygame.draw.rect(surface, (0, 0, 0),
                         (cx - r // 3 + sway, mouth_y, (r // 3) * 2, eye_size))

        # Freeze overlay
        if self.frozen_timer > 0:
            from entities.mage_projectile import draw_freeze_overlay
            draw_freeze_overlay(surface, self.pos, self.size, min(1.0, self.frozen_timer))

        # Burn overlay
        if self.burn_timer > 0:
            from entities.mage_projectile import draw_burn_overlay
            draw_burn_overlay(surface, self.pos, self.size, self.burn_timer, 3.0)

        # HP bar
        bar_w = self.size
        bar_h = 3
        bar_x = cx - bar_w // 2
        bar_y = cy - r - 8
        pygame.draw.rect(surface, (140, 30, 30), (bar_x, bar_y, bar_w, bar_h))
        fw = int(bar_w * max(0, self.hp) / max(1, self.max_hp))
        if fw > 0:
            pygame.draw.rect(surface, (60, 200, 60), (bar_x, bar_y, fw, bar_h))


# ── CreeperExplosion — visual-only animation ──────────────────────────────────

class CreeperExplosion:
    """Visual AoE explosion triggered by a Creeper.
    Call apply(minions) once after creation to deal damage."""

    DURATION = 0.7

    def __init__(self, pos, damage: int, radius: float):
        self.pos      = pygame.Vector2(pos)
        self.damage   = damage
        self.radius   = radius
        self.timer    = 0.0
        self.is_alive = True

        self._particles: list[dict] = []
        for _ in range(30):
            a     = random.uniform(0, math.tau)
            speed = random.uniform(80, 260)
            life  = random.uniform(0.3, 0.65)
            self._particles.append({
                "x": float(pos.x), "y": float(pos.y),
                "vx": math.cos(a) * speed, "vy": math.sin(a) * speed,
                "life": life, "max_life": life,
                "size": random.randint(4, 12),
            })

    def apply(self, minions: list) -> list:
        """Deal damage to all alive minions within radius. Returns (minion, dmg) list."""
        hits = []
        for m in minions:
            if m.is_alive and self.pos.distance_to(m.pos) <= self.radius:
                m.hp = max(0, m.hp - self.damage)
                hits.append((m, self.damage))
                if m.hp <= 0:
                    m.is_alive = False
        return hits

    def update(self, dt: float):
        self.timer += dt
        if self.timer >= self.DURATION:
            self.is_alive = False
        for p in self._particles:
            p["x"]    += p["vx"] * dt
            p["y"]    += p["vy"] * dt
            p["life"] -= dt
        self._particles = [p for p in self._particles if p["life"] > 0]

    def draw(self, surface: pygame.Surface):
        t  = self.timer / self.DURATION
        cx = int(self.pos.x)
        cy = int(self.pos.y)

        # Inner flash
        if t < 0.22:
            flash_r = int(self.radius * 0.5 * (1 - t / 0.22))
            if flash_r > 0:
                pygame.draw.circle(surface, (255, 255, 255), (cx, cy), flash_r)

        # Green-tinted expanding rings
        ring_data = [
            (0.0,  0.80, (80, 220,  50), 5),
            (0.05, 0.95, (180, 255, 80), 3),
            (0.10, 1.10, (255, 255, 150), 2),
        ]
        for delay, scale, col, width in ring_data:
            rt = max(0.0, t - delay)
            r  = int(self.radius * scale * rt / max(0.01, 1.0 - delay))
            r  = min(r, int(self.radius * scale))
            if r > 0:
                w = max(1, int(width * (1 - rt)))
                pygame.draw.circle(surface, col, (cx, cy), r, w)

        # Particles
        for p in self._particles:
            ratio = p["life"] / p["max_life"]
            col   = (int(80 * ratio), int(200 * ratio), int(50 * ratio ** 2))
            sz    = max(1, int(p["size"] * ratio))
            pygame.draw.circle(surface, col, (int(p["x"]), int(p["y"])), sz)
