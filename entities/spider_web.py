"""
SpiderWeb projectile — fired by Spider enemies.

Travels at 200 px/s, max lifetime 2 s.  On hitting a minion it deals a small
amount of damage and applies a freeze (frozen_timer) for freeze_duration seconds,
preventing the minion from moving.
"""
import math
import pygame


class SpiderWeb:
    size = 8   # collision radius (also used for draw)

    def __init__(self, pos, angle: float, speed: float,
                 damage: int, freeze_duration: float):
        self.pos            = pygame.Vector2(pos)
        self.velocity       = pygame.Vector2(math.cos(angle), math.sin(angle)) * speed
        self.damage         = damage
        self.freeze_duration = freeze_duration
        self.is_alive       = True
        self._timer         = 0.0
        self._max_life      = 2.0   # max flight time in seconds

    # ------------------------------------------------------------------

    def update(self, dt: float):
        self.pos    += self.velocity * dt
        self._timer += dt
        if self._timer >= self._max_life:
            self.is_alive = False

    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface):
        if not self.is_alive:
            return
        cx, cy = int(self.pos.x), int(self.pos.y)
        r = self.size // 2

        # Outer blob (slightly transparent-looking with two concentric circles)
        pygame.draw.circle(surface, (190, 185, 210), (cx, cy), r)
        pygame.draw.circle(surface, (230, 225, 255), (cx, cy), r, 1)

        # Four web-strand lines (compass + diagonal cross)
        for angle in (0, 45, 90, 135):
            rad = math.radians(angle)
            x1  = cx + int(math.cos(rad) * r)
            y1  = cy + int(math.sin(rad) * r)
            x2  = cx - int(math.cos(rad) * r)
            y2  = cy - int(math.sin(rad) * r)
            pygame.draw.line(surface, (215, 210, 240), (x1, y1), (x2, y2), 1)
