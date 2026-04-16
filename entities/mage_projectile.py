"""
Mage projectiles and their impact effects.

Classes:
  FireMageFireball  — travelling projectile from Fire Mage; on impact creates a
                      MageExplosion that deals AoE damage and applies Burn.
  IceMageIceball    — travelling projectile from Ice Mage; on impact deals
                      direct damage and applies Freeze to the target.
  MageExplosion     — Fire Mage AoE explosion (visual + damage data).
  BurnEffect        — Per-enemy burn state (applied via enemy attributes, not a
                      separate object, but helper drawn on each burning enemy).

Usage (from BattleScene):
  # Shooting
  fb = fire_mage.try_shoot(enemies)
  if fb:
      mage_projectiles.append(fb)

  # Update loop
  for proj in mage_projectiles:
      result = proj.update(dt, enemies, boss)
      if result is not None:
          # result is a MageExplosion (fire) or None (ice — side-effects applied)
          mage_explosions.append(result)
  mage_projectiles = [p for p in mage_projectiles if p.is_alive]

  # Draw
  for proj in mage_projectiles:
      proj.draw(surface)
  for exp in mage_explosions:
      exp.update(dt)
      exp.draw(surface)
  mage_explosions = [e for e in mage_explosions if e.is_alive]
"""
from __future__ import annotations
import math
import random
import pygame
from config import CFG

_FMC = CFG.get("fire_mage", {})
_IMC = CFG.get("ice_mage",  {})

_FIREBALL_DAMAGE  = int(_FMC.get("fireball_damage",         22))
_FIREBALL_EXP_RAD = int(_FMC.get("fireball_explosion_radius", 75))
_BURN_DPS         = float(_FMC.get("burn_dps",              8.0))
_BURN_DURATION    = float(_FMC.get("burn_duration",         3.0))

_ICEBALL_DAMAGE   = int(_IMC.get("iceball_damage",          15))
_FREEZE_DURATION  = float(_IMC.get("freeze_duration",       2.0))

_PROJECTILE_LIFETIME = 3.0   # seconds before auto-expire


# ── FireMageFireball ─────────────────────────────────────────────────────────

class FireMageFireball:
    """Slow fiery orb that explodes on impact."""

    def __init__(self, pos, angle: float, speed: float):
        self.pos      = pygame.Vector2(pos)
        self.vel      = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.is_alive = True
        self.size     = 7
        self._timer   = 0.0
        # Particle trail
        self._trail: list[dict] = []

    def update(self, dt: float, enemies: list, boss=None) -> "MageExplosion | None":
        """Move the fireball; return MageExplosion on hit, None otherwise."""
        if not self.is_alive:
            return None

        self._timer += dt
        if self._timer > _PROJECTILE_LIFETIME:
            self.is_alive = False
            return None

        self.pos += self.vel * dt

        # Trail particle
        self._trail.append({
            "x": self.pos.x, "y": self.pos.y,
            "life": 0.25, "max_life": 0.25,
            "r": random.randint(4, 9),
        })
        for p in self._trail:
            p["life"] -= dt
        self._trail = [p for p in self._trail if p["life"] > 0]

        # Hit detection
        candidates = [e for e in enemies if e.is_alive]
        if boss is not None and getattr(boss, 'is_alive', False):
            candidates.append(boss)

        for target in candidates:
            if self.pos.distance_to(target.pos) <= (self.size + target.size // 2) + 2:
                self.is_alive = False
                exp = MageExplosion(self.pos, _FIREBALL_DAMAGE, _FIREBALL_EXP_RAD,
                                    _BURN_DPS, _BURN_DURATION)
                return exp

        return None

    def draw(self, surface: pygame.Surface):
        if not self.is_alive:
            return

        # Trail
        for p in self._trail:
            ratio = p["life"] / p["max_life"]
            r     = max(1, int(p["r"] * ratio))
            col   = (int(255 * ratio), int(100 * ratio ** 2), 0)
            pygame.draw.circle(surface, col,
                               (int(p["x"]), int(p["y"])), r)

        # Core orb
        cx, cy = int(self.pos.x), int(self.pos.y)
        t = (pygame.time.get_ticks() % 400) / 400.0
        glow = self.size + int(3 * math.sin(t * math.tau))
        pygame.draw.circle(surface, (180, 40, 0), (cx, cy), glow)
        pygame.draw.circle(surface, (255, 140, 20), (cx, cy), self.size)
        pygame.draw.circle(surface, (255, 240, 140), (cx, cy), max(2, self.size - 3))


# ── IceMageIceball ───────────────────────────────────────────────────────────

class IceMageIceball:
    """Fast icy orb that freezes and damages a single target on impact."""

    def __init__(self, pos, angle: float, speed: float):
        self.pos      = pygame.Vector2(pos)
        self.vel      = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.is_alive = True
        self.size     = 6
        self._timer   = 0.0
        self._trail: list[dict] = []

    def update(self, dt: float, enemies: list, boss=None) -> None:
        """Move and detect hits; applies freeze + damage directly to target."""
        if not self.is_alive:
            return

        self._timer += dt
        if self._timer > _PROJECTILE_LIFETIME:
            self.is_alive = False
            return

        self.pos += self.vel * dt

        # Ice trail
        self._trail.append({
            "x": self.pos.x, "y": self.pos.y,
            "life": 0.18, "max_life": 0.18,
        })
        for p in self._trail:
            p["life"] -= dt
        self._trail = [p for p in self._trail if p["life"] > 0]

        # Hit detection — single target only
        candidates = [e for e in enemies if e.is_alive]
        if boss is not None and getattr(boss, 'is_alive', False):
            candidates.append(boss)

        for target in candidates:
            if self.pos.distance_to(target.pos) <= (self.size + target.size // 2) + 2:
                self.is_alive = False
                # Apply direct damage
                target.hp = max(0, target.hp - _ICEBALL_DAMAGE)
                if target.hp <= 0:
                    target.is_alive = False
                # Apply freeze status
                current_freeze = getattr(target, 'frozen_timer', 0.0)
                target.frozen_timer = max(current_freeze, _FREEZE_DURATION)
                break

    def draw(self, surface: pygame.Surface):
        if not self.is_alive:
            return

        # Trail
        for p in self._trail:
            ratio = p["life"] / p["max_life"]
            r     = max(1, int(5 * ratio))
            pygame.draw.circle(surface, (120, 200, int(200 + 55 * ratio)),
                               (int(p["x"]), int(p["y"])), r)

        cx, cy = int(self.pos.x), int(self.pos.y)
        t  = (pygame.time.get_ticks() % 600) / 600.0
        gw = self.size + int(3 * math.sin(t * math.tau))
        pygame.draw.circle(surface, (40, 100, 200), (cx, cy), gw)
        pygame.draw.circle(surface, (100, 190, 255), (cx, cy), self.size)
        pygame.draw.circle(surface, (210, 240, 255), (cx, cy), max(2, self.size - 3))
        # Ice glint
        pygame.draw.line(surface, (255, 255, 255),
                         (cx - 2, cy - 2), (cx + 2, cy + 2), 1)
        pygame.draw.line(surface, (255, 255, 255),
                         (cx + 2, cy - 2), (cx - 2, cy + 2), 1)


# ── MageExplosion ────────────────────────────────────────────────────────────

class MageExplosion:
    """
    Fire Mage AoE explosion.
    Call apply(enemies, boss) immediately after creation to deal damage and
    apply burn.  The object then persists purely for animation.
    """
    DURATION = 0.85

    def __init__(self, pos, damage: int, radius: int,
                 burn_dps: float, burn_duration: float):
        self.pos          = pygame.Vector2(pos)
        self.damage       = damage
        self.radius       = radius
        self.burn_dps     = burn_dps
        self.burn_duration = burn_duration
        self.timer        = 0.0
        self.is_alive     = True

        self._particles: list[dict] = []
        for _ in range(28):
            a     = random.uniform(0, math.tau)
            speed = random.uniform(70, 240)
            life  = random.uniform(0.35, 0.75)
            self._particles.append({
                "x": float(pos.x), "y": float(pos.y),
                "vx": math.cos(a) * speed, "vy": math.sin(a) * speed,
                "life": life, "max_life": life,
                "size": random.randint(3, 10),
            })

    def apply(self, enemies: list, boss=None) -> list:
        """
        Deal damage + apply Burn to all alive targets within radius.
        Returns list of (target, damage_dealt).
        """
        hits = []
        candidates = [e for e in enemies if e.is_alive]
        if boss is not None and getattr(boss, 'is_alive', False):
            candidates.append(boss)

        for t in candidates:
            if self.pos.distance_to(t.pos) <= self.radius:
                t.hp = max(0, t.hp - self.damage)
                hits.append((t, self.damage))
                # Apply burn (max with existing burn timer)
                existing_burn = getattr(t, 'burn_timer', 0.0)
                t.burn_timer    = max(existing_burn, self.burn_duration)
                t.burn_dps      = self.burn_dps
                if t.hp <= 0:
                    t.is_alive = False
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

        # Inner white flash
        if t < 0.2:
            flash_r = int(self.radius * 0.45 * (1 - t / 0.2))
            if flash_r > 0:
                pygame.draw.circle(surface, (255, 255, 200), (cx, cy), flash_r)

        # Expanding rings
        ring_data = [
            (0.0,  0.75, (255,  70,   0), 5),
            (0.06, 0.88, (255, 150,  30), 3),
            (0.12, 1.00, (255, 220, 100), 2),
        ]
        for delay, scale, col, width in ring_data:
            rt = max(0.0, t - delay)
            r  = int(self.radius * scale * rt / max(0.01, 1.0 - delay))
            r  = min(r, int(self.radius * scale))
            if r > 0:
                w = max(1, int(width * (1 - rt)))
                pygame.draw.circle(surface, col, (cx, cy), r, w)

        # Scorched ground
        scorch_r = int(self.radius * 0.5 * t)
        if scorch_r > 2:
            pygame.draw.circle(surface, (40, 20, 0), (cx, cy), scorch_r, 2)

        # Particles
        for p in self._particles:
            ratio = p["life"] / p["max_life"]
            col   = (int(255 * ratio), int(100 * ratio ** 2), 0)
            sz    = max(1, int(p["size"] * ratio))
            pygame.draw.circle(surface, col, (int(p["x"]), int(p["y"])), sz)


# ── FreezeEffect visual helper ────────────────────────────────────────────────

def draw_freeze_overlay(surface: pygame.Surface, pos, size: int, alpha: float = 1.0):
    """Draw an icy freeze overlay on a frozen entity."""
    cx, cy = int(pos.x), int(pos.y)
    r      = size // 2 + 4
    ice    = pygame.Surface((r * 2, r * 2), pygame.SRCALPHA)
    a      = int(110 * alpha)
    pygame.draw.circle(ice, (100, 190, 255, a), (r, r), r)
    # Ice crystal lines
    for i in range(4):
        angle = i * math.pi / 4
        ex    = r + int(math.cos(angle) * (r - 2))
        ey    = r + int(math.sin(angle) * (r - 2))
        pygame.draw.line(ice, (200, 230, 255, min(200, int(180 * alpha))),
                         (r, r), (ex, ey), 1)
    surface.blit(ice, (cx - r, cy - r))


def draw_burn_overlay(surface: pygame.Surface, pos, size: int, timer: float, max_timer: float):
    """Draw a fiery burn overlay on a burning entity (flickers)."""
    import pygame as pg
    cx, cy = int(pos.x), int(pos.y)
    r      = size // 2 + 2
    ratio  = timer / max(0.1, max_timer)
    t_ms   = pg.time.get_ticks()
    flick  = 0.5 + 0.5 * math.sin(t_ms / 120.0)
    a      = int(80 * ratio * flick)
    if a <= 0:
        return
    flame = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
    pygame.draw.circle(flame, (255, 100, 0, a), (r + 2, r + 2), r)
    pygame.draw.circle(flame, (255, 200, 0, a // 2), (r + 2, r + 2), r // 2)
    surface.blit(flame, (cx - r - 2, cy - r - 2))


def draw_enemy_grave(surface: pygame.Surface, pos, size: int,
                     stone=(84, 78, 90), alpha: float = 1.0):
    """Draw a compact tombstone for a dead enemy at pos.

    stone — base RGB stone color (varies by enemy type for a subtle tint).
    alpha — opacity 0.0 → 1.0; pass a value < 1.0 for the fade-out effect.
    """
    cx, cy = int(pos.x), int(pos.y)
    w      = max(8, size - 4)
    h      = max(10, int(size * 0.95))
    hw     = w // 2
    dark   = tuple(max(0, c - 30) for c in stone)
    a      = int(255 * max(0.0, min(1.0, alpha)))

    # Render onto an SRCALPHA surface so alpha compositing works correctly
    s    = pygame.Surface((w + 4, h + 6), pygame.SRCALPHA)
    # Tombstone body (pill-shaped rounded rect)
    body = pygame.Rect(2, 3, w, h)
    pygame.draw.rect(s, (*stone, a), body, border_radius=hw)
    pygame.draw.rect(s, (*dark,  a), body, 1, border_radius=hw)

    # Horizontal groove (lower third of stone)
    gy = 3 + h - h // 3
    if w > 6:
        pygame.draw.line(s, (*dark, a), (4, gy), (w - 2, gy), 1)

    # Cross (†) engraved near the top of the stone
    cw   = max(1, w // 5)
    ch   = max(3, h // 3 - 1)
    cx_s = w // 2 + 2        # x-center on the small surface
    ct   = 5                  # top of the vertical bar

    pygame.draw.rect(s, (*dark, a), (cx_s - cw // 2, ct,           cw,       ch))
    pygame.draw.rect(s, (*dark, a), (cx_s - cw - 1,  ct + ch // 3, cw*2 + 2, cw))

    surface.blit(s, (cx - hw - 2, cy - h // 2 - 3))
