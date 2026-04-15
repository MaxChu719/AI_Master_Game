"""
Archer — a DQN-driven ranged minion.
Movement and shoot direction are selected by a DQN policy.
The battle scene sets self.velocity each frame, then calls update().
Shooting is triggered externally via try_shoot().
"""
import math
import pygame

from entities.projectile import Projectile
from config import CFG

_AC = CFG["archer"]
_PREFERRED_RANGE  = _AC["preferred_range"]
_SHOOT_RANGE      = _AC["attack_range"]
_LABEL_FONT       = None  # lazy-initialised once per process

ARCHER_MISS_ANGLE = math.radians(30)  # cone half-angle used for miss detection


def _get_font():
    global _LABEL_FONT
    if _LABEL_FONT is None:
        _LABEL_FONT = pygame.font.SysFont("arial", 13, bold=True)
    return _LABEL_FONT


class Archer:
    def __init__(self, pos):
        self.pos = pygame.Vector2(pos)
        self.hp = _AC["hp"]
        self.max_hp = _AC["hp"]
        self.speed = _AC["speed"]
        self.size = _AC["size"]
        self.color = (60, 200, 100)   # green
        self.is_alive = True
        self.attack_damage = _AC["attack_damage"]
        self.attack_cooldown = _AC["attack_cooldown"]
        self._shoot_timer = 0.0
        # Velocity — set externally by DQN each frame
        self.velocity = pygame.Vector2(0, 0)
        # Stamina — limits how rapidly the archer can fire
        self.stamina = _AC["stamina"]
        self.max_stamina = _AC["stamina"]
        self.stamina_regen = _AC["stamina_regen"]
        self.stamina_cost = _AC["stamina_cost"]
        # Visual flash when an arrow is released
        self.shoot_flash_timer = 0.0
        self.shoot_flash_angle = 0.0
        # Lifetime stat counters (accumulate across waves within a session)
        self.shots_fired = 0
        # Freeze state (applied by spider web hits)
        self.frozen_timer = 0.0         # counts down; >0 means movement is blocked
        # Knockback velocity — applied and decayed by MovementSystem each frame
        self.knockback_vel = pygame.Vector2(0, 0)
        # Last action index chosen by the DQN (or preset); used by ally observation
        self.last_action = 0

    # ------------------------------------------------------------------
    # Update — regen, move by velocity, clamp.  No autonomous AI here.
    # ------------------------------------------------------------------

    def update(self, dt: float, arena_bounds: tuple, sfx=None):
        if not self.is_alive:
            return

        # Regen stamina
        self.stamina = min(self.max_stamina, self.stamina + self.stamina_regen * dt)

        # Move by velocity (set externally by DQN)
        self.pos += self.velocity * dt

        # Clamp to arena
        left, top, right, bottom = arena_bounds
        half = self.size // 2
        self.pos.x = max(left + half, min(right - half, self.pos.x))
        self.pos.y = max(top + half, min(bottom - half, self.pos.y))

        # Tick timers
        if self._shoot_timer > 0:
            self._shoot_timer = max(0.0, self._shoot_timer - dt)
        if self.shoot_flash_timer > 0:
            self.shoot_flash_timer = max(0.0, self.shoot_flash_timer - dt)

    # ------------------------------------------------------------------
    # Shoot — called by BattleScene when DQN picks an attack action
    # ------------------------------------------------------------------

    def try_shoot(self, direction_angle: float, projectiles: list, sfx=None) -> bool:
        """Attempt to fire an arrow in direction_angle (radians).
        Returns True if a shot was fired, False if on cooldown or out of stamina."""
        if self._shoot_timer > 0 or self.stamina < self.stamina_cost:
            return False

        projectiles.append(Projectile(self.pos, direction_angle, damage=self.attack_damage))
        self._shoot_timer = self.attack_cooldown
        self.stamina -= self.stamina_cost
        self.shoot_flash_timer = 0.22
        self.shoot_flash_angle = direction_angle
        self.shots_fired += 1
        if sfx:
            sfx.play("arrow_shoot")
        return True

    # ------------------------------------------------------------------
    # Draw
    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface):
        if not self.is_alive:
            self._draw_grave(surface)
            return

        # Shoot flash: a fading beam in the direction of the shot
        if self.shoot_flash_timer > 0:
            self._draw_shoot_flash(surface)

        rect = pygame.Rect(
            int(self.pos.x - self.size // 2),
            int(self.pos.y - self.size // 2),
            self.size,
            self.size,
        )
        pygame.draw.rect(surface, self.color, rect)

        # HP bar
        bar_w = self.size
        bar_h = 4
        bar_x = rect.x
        bar_y = rect.y - 8
        pygame.draw.rect(surface, (180, 40, 40), (bar_x, bar_y, bar_w, bar_h))
        fill_w = int(bar_w * self.hp / self.max_hp)
        if fill_w > 0:
            pygame.draw.rect(surface, (50, 200, 80), (bar_x, bar_y, fill_w, bar_h))

        # Stamina bar (above HP bar, yellow → orange as depleted)
        stam_y = bar_y - 6
        pygame.draw.rect(surface, (60, 45, 10), (bar_x, stam_y, bar_w, bar_h))
        stam_fill = int(bar_w * self.stamina / self.max_stamina)
        if stam_fill > 0:
            ratio = self.stamina / self.max_stamina
            sc = (int(255 * (1.0 - ratio * 0.4)), int(200 * ratio), 0)
            pygame.draw.rect(surface, sc, (bar_x, stam_y, stam_fill, bar_h))

        # Freeze overlay (blue-tinted web effect when frozen by spider)
        if self.frozen_timer > 0:
            alpha = min(180, int(180 * self.frozen_timer / 1.5))
            ice = pygame.Surface((self.size, self.size), pygame.SRCALPHA)
            ice.fill((80, 160, 255, alpha))
            surface.blit(ice, (rect.x, rect.y))
            cx2, cy2 = int(self.pos.x), int(self.pos.y)
            hs = self.size // 2 - 2
            pygame.draw.line(surface, (180, 220, 255), (cx2 - hs, cy2 - hs), (cx2 + hs, cy2 + hs), 1)
            pygame.draw.line(surface, (180, 220, 255), (cx2 + hs, cy2 - hs), (cx2 - hs, cy2 + hs), 1)

        # "A" label
        label = _get_font().render("A", True, (255, 255, 255))
        surface.blit(label, label.get_rect(center=(int(self.pos.x), int(self.pos.y))))

    def _draw_grave(self, surface: pygame.Surface):
        """Draw a tombstone icon for dead archers, keeping the 'A' label."""
        cx, cy = int(self.pos.x), int(self.pos.y)
        w = self.size - 4
        half_w = w // 2
        stone     = (78, 95, 82)   # slightly green-tinted to hint at archer (green minion)
        stone_drk = (50, 65, 54)

        # Tombstone body: tall rounded rectangle
        body_rect = pygame.Rect(cx - half_w, cy - self.size // 2, w, self.size)
        pygame.draw.rect(surface, stone, body_rect, border_radius=half_w)
        pygame.draw.rect(surface, stone_drk, body_rect, 1, border_radius=half_w)

        # Horizontal groove across the stone
        groove_y = cy + 2
        pygame.draw.line(surface, stone_drk,
                         (cx - half_w + 3, groove_y), (cx + half_w - 3, groove_y), 1)

        # "A" label in muted tone
        label = _get_font().render("A", True, (110, 130, 115))
        surface.blit(label, label.get_rect(center=(cx, cy + 7)))

    def _draw_shoot_flash(self, surface: pygame.Surface):
        """Draw a fading beam line in the shot direction."""
        t = self.shoot_flash_timer / 0.22        # 1.0 → 0.0
        brightness = int(220 * t)
        color = (brightness // 3, brightness, brightness // 3)
        cx, cy = int(self.pos.x), int(self.pos.y)
        end_x = cx + int(math.cos(self.shoot_flash_angle) * _SHOOT_RANGE)
        end_y = cy + int(math.sin(self.shoot_flash_angle) * _SHOOT_RANGE)
        pygame.draw.line(surface, color, (cx, cy), (end_x, end_y), 2)
        # Small circle at muzzle
        dot_r = max(2, int(5 * t))
        pygame.draw.circle(surface, (min(255, brightness + 40), 255, min(255, brightness + 40)),
                            (cx, cy), dot_r, 1)
