import math
import pygame
from config import CFG

_FC = CFG["fighter"]
ATTACK_ARC = math.radians(_FC["attack_arc_deg"])

_LABEL_FONT = None


def _get_font():
    global _LABEL_FONT
    if _LABEL_FONT is None:
        _LABEL_FONT = pygame.font.SysFont("arial", 13, bold=True)
    return _LABEL_FONT


class Minion:
    def __init__(self, pos):
        self.pos = pygame.Vector2(pos)
        self.hp = _FC["hp"]
        self.max_hp = _FC["hp"]
        self.speed = _FC["speed"]
        self.size = _FC["size"]
        self.color = (60, 120, 220)
        self.is_alive = True
        self.attack_damage = _FC["attack_damage"]
        self.attack_range = _FC["attack_range"]
        self.attack_cooldown = _FC["attack_cooldown"]
        self.attack_timer = 0.0
        self.velocity = pygame.Vector2(0, 0)
        # Stamina — limits how many swings the fighter can chain
        self.stamina = _FC["stamina"]
        self.max_stamina = _FC["stamina"]
        self.stamina_regen = _FC["stamina_regen"]
        self.stamina_cost = _FC["stamina_cost"]
        # Attack visual state
        self.attack_flash_timer = 0.0   # counts down from 0.2 to 0
        self.attack_flash_angle = 0.0   # radians toward target
        # Freeze state (applied by spider web hits)
        self.frozen_timer = 0.0         # counts down; >0 means movement is blocked
        # Knockback velocity — applied and decayed by MovementSystem each frame
        self.knockback_vel = pygame.Vector2(0, 0)

    def draw(self, surface: pygame.Surface):
        if not self.is_alive:
            self._draw_grave(surface)
            return

        # Sword arc drawn behind the entity so it looks like a swing
        if self.attack_flash_timer > 0:
            self._draw_attack_arc(surface)

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

        # Stamina bar (above HP bar, yellow when full → orange when low)
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
            # Draw small X pattern to suggest web strands
            cx2, cy2 = int(self.pos.x), int(self.pos.y)
            hs = self.size // 2 - 2
            pygame.draw.line(surface, (180, 220, 255), (cx2 - hs, cy2 - hs), (cx2 + hs, cy2 + hs), 1)
            pygame.draw.line(surface, (180, 220, 255), (cx2 + hs, cy2 - hs), (cx2 - hs, cy2 + hs), 1)

        # "F" label
        label = _get_font().render("F", True, (255, 255, 255))
        surface.blit(label, label.get_rect(center=(int(self.pos.x), int(self.pos.y))))

    def _draw_grave(self, surface: pygame.Surface):
        """Draw a tombstone icon for dead fighters, keeping the 'F' label."""
        cx, cy = int(self.pos.x), int(self.pos.y)
        w = self.size - 4
        half_w = w // 2
        stone     = (88, 82, 98)
        stone_drk = (55, 50, 65)

        # Tombstone body: tall rounded rectangle
        body_rect = pygame.Rect(cx - half_w, cy - self.size // 2, w, self.size)
        pygame.draw.rect(surface, stone, body_rect, border_radius=half_w)
        pygame.draw.rect(surface, stone_drk, body_rect, 1, border_radius=half_w)

        # Horizontal groove across the stone
        groove_y = cy + 2
        pygame.draw.line(surface, stone_drk,
                         (cx - half_w + 3, groove_y), (cx + half_w - 3, groove_y), 1)

        # "F" label in muted tone
        label = _get_font().render("F", True, (120, 115, 135))
        surface.blit(label, label.get_rect(center=(cx, cy + 7)))

    def _draw_attack_arc(self, surface: pygame.Surface):
        r = int(self.attack_range)
        cx, cy = int(self.pos.x), int(self.pos.y)
        half_arc = ATTACK_ARC / 2

        # Build a wedge polygon (fan from center outward)
        segments = 8
        points = [(cx, cy)]
        for i in range(segments + 1):
            a = self.attack_flash_angle - half_arc + i * (ATTACK_ARC / segments)
            points.append((int(cx + r * math.cos(a)), int(cy + r * math.sin(a))))

        # Fade alpha from full (start) to 0 (end of 0.2s flash)
        t = self.attack_flash_timer / 0.2
        alpha = int(200 * t)

        # Draw on a small surface centered on the arc
        s_size = r * 2 + 4
        s = pygame.Surface((s_size, s_size), pygame.SRCALPHA)
        ox = cx - r - 2
        oy = cy - r - 2
        shifted = [(px - ox, py - oy) for px, py in points]
        pygame.draw.polygon(s, (255, 160, 0, alpha), shifted)
        # Bright arc edge (outer arc only, not the two radial edges)
        pygame.draw.lines(s, (255, 230, 60, min(255, alpha + 40)),
                          False, shifted[1:], 2)
        surface.blit(s, (ox, oy))
