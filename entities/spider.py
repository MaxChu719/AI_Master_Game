"""
Spider enemy — a ranged enemy that keeps its distance from minions and
shoots sticky web projectiles to freeze them temporarily.

Behaviour:
  - Maintains a preferred distance (~200 px) from the nearest alive minion,
    similar to the Archer preset heuristic.
  - Flees when a minion closes within min_safe_dist (~120 px).
  - Orbits (strafes) at preferred range when no threat is imminent.
  - Shoots a SpiderWeb projectile at the nearest in-range minion every ~3 s.
    Web hitting a minion deals small damage and freezes it for 1.5 s.

enemy_type = 1  (used by MinionEnv to encode type in the observation token)
"""
from __future__ import annotations
import math
import pygame
from config import CFG

_SC = CFG["spider"]

_BODY_COLOR   = (110, 20,  160)   # dark purple abdomen
_BODY_HI      = (155, 60,  210)   # highlight on abdomen
_LEG_COLOR    = (80,   0,  120)
_LEG_LOW      = (55,   0,   90)   # lower-segment leg (darker)
_EYE_COLOR    = (255, 230,  80)   # bright yellow eyes

# Each tuple: (upper_angle_deg, lower_delta_deg, upper_len, lower_len)
# upper_angle is from body-centre (+X = right); lower_delta bends the knee outward.
_LEG_DEFS = [
    (-150, -30, 9, 8),  # left rear
    (-120, -25, 9, 8),  # left mid-rear
    (-95,  -15, 9, 8),  # left mid-front
    (-70,  -10, 8, 7),  # left front
    ( 70,   10, 8, 7),  # right front
    ( 95,   15, 9, 8),  # right mid-front
    (120,   25, 9, 8),  # right mid-rear
    (150,   30, 9, 8),  # right rear
]


class Spider:
    """Ranged spider enemy that shoots freezing web projectiles."""

    enemy_type = 1   # class constant — used by MinionEnv observation encoding

    def __init__(self, pos):
        self.pos            = pygame.Vector2(pos)
        self.hp             = _SC["hp"]
        self.max_hp         = _SC["hp"]
        self.speed          = _SC["speed"]
        self.size           = _SC["size"]
        self.is_alive       = True
        self.attack_damage  = _SC["web_damage"]
        self.attack_range   = _SC["web_range"]
        self.attack_cooldown = _SC["web_cooldown"]
        self.attack_timer   = 0.0
        self.attack_flash_timer = 0.0   # shared name with Enemy for compat
        self.preferred_dist = _SC["preferred_dist"]
        self.min_safe_dist  = _SC["min_safe_dist"]
        self.freeze_duration = _SC["freeze_duration"]
        # Velocity is set by MovementSystem each frame
        self.velocity       = pygame.Vector2(0, 0)
        # Knockback velocity — applied and decayed by MovementSystem each frame
        self.knockback_vel  = pygame.Vector2(0, 0)
        # Animation
        self._anim_timer    = 0.0        # drives leg-wobble

    # ------------------------------------------------------------------
    # Per-frame timer tick (called from BattleScene, not MovementSystem)
    # ------------------------------------------------------------------

    def tick(self, dt: float):
        """Update animation timer.  attack_flash_timer is ticked by CombatSystem."""
        if self.is_alive:
            self._anim_timer += dt

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface):
        if not self.is_alive:
            return

        cx, cy = int(self.pos.x), int(self.pos.y)
        r      = self.size // 2          # abdomen radius
        wobble = math.sin(self._anim_timer * 7.0) * 1.8   # leg animation phase

        # ── Legs ─────────────────────────────────────────────────────────
        for (a1_deg, a_delta, len1, len2) in _LEG_DEFS:
            # Apply a small wobble to upper angle; halved wobble to lower
            a1 = math.radians(a1_deg + wobble)
            a2 = math.radians(a1_deg + a_delta + wobble * 0.4)
            # Upper segment: body centre → knee joint
            kx = cx + math.cos(a1) * (r + len1)
            ky = cy + math.sin(a1) * (r + len1) + 1
            # Lower segment: knee → foot tip
            fx = kx + math.cos(a2) * len2
            fy = ky + math.sin(a2) * len2 + 2   # slight gravity droop
            pygame.draw.line(surface, _LEG_COLOR, (cx, cy), (int(kx), int(ky)), 1)
            pygame.draw.line(surface, _LEG_LOW,   (int(kx), int(ky)), (int(fx), int(fy)), 1)

        # ── Abdomen (main body circle) ────────────────────────────────────
        pygame.draw.circle(surface, _BODY_COLOR, (cx, cy), r)
        # Subtle highlight in the upper-left quadrant
        pygame.draw.circle(surface, _BODY_HI, (cx - r // 3, cy - r // 3), max(2, r // 3))

        # ── Cephalothorax (head, smaller circle at front-top) ─────────────
        hx = cx
        hy = cy - r + 2
        pygame.draw.circle(surface, _BODY_COLOR, (hx, hy), max(3, r // 2))

        # ── Eyes (two bright dots) ────────────────────────────────────────
        ex = r // 3
        pygame.draw.circle(surface, _EYE_COLOR, (cx - ex, cy - r + 1), 2)
        pygame.draw.circle(surface, _EYE_COLOR, (cx + ex, cy - r + 1), 2)

        # ── Web-shoot flash (brief beam when firing) ──────────────────────
        if self.attack_flash_timer > 0:
            t   = self.attack_flash_timer / 0.3
            col = (min(255, int(200 * t)), min(255, int(200 * t)), 220)
            end_x = cx + int(math.cos(self._shoot_angle) * (r + 20))
            end_y = cy + int(math.sin(self._shoot_angle) * (r + 20))
            pygame.draw.line(surface, col, (cx, cy), (end_x, end_y), 2)

        # ── HP bar ────────────────────────────────────────────────────────
        bar_w = self.size
        bar_h = 4
        bar_x = cx - bar_w // 2
        bar_y = cy - r - 10
        pygame.draw.rect(surface, (180, 40, 40), (bar_x, bar_y, bar_w, bar_h))
        fill_w = int(bar_w * self.hp / self.max_hp)
        if fill_w > 0:
            pygame.draw.rect(surface, (50, 200, 80), (bar_x, bar_y, fill_w, bar_h))

    # ------------------------------------------------------------------
    # Shoot helper — called by BattleScene
    # ------------------------------------------------------------------

    def try_shoot_web(self, targets: list) -> "SpiderWeb | None":
        """
        Fire a SpiderWeb at the nearest target in range if the cooldown has
        expired.  Returns the SpiderWeb object on success, None otherwise.
        targets : list of alive minions
        """
        from entities.spider_web import SpiderWeb   # deferred to avoid circular import

        if self.attack_timer > 0 or not targets:
            return None

        nearest = min(targets, key=lambda m: self.pos.distance_to(m.pos))
        dist = self.pos.distance_to(nearest.pos)
        if dist > self.attack_range:
            return None

        dx = nearest.pos.x - self.pos.x
        dy = nearest.pos.y - self.pos.y
        angle = math.atan2(dy, dx)

        self.attack_timer = self.attack_cooldown
        self.attack_flash_timer = 0.3
        self._shoot_angle = angle   # used by draw()
        return SpiderWeb(
            pos=(self.pos.x, self.pos.y),
            angle=angle,
            speed=_SC["web_speed"],
            damage=self.attack_damage,
            freeze_duration=self.freeze_duration,
        )
