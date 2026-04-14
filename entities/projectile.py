import math
import pygame
from config import CFG

_PC = CFG["projectile"]


class Projectile:
    """Arrow shot by the Archer minion."""

    speed    = _PC["speed"]
    damage   = CFG["archer"]["attack_damage"]
    size     = _PC["size"]
    lifetime = _PC["lifetime"]

    def __init__(self, pos, angle: float, damage: int = None):
        self.pos = pygame.Vector2(pos)
        self.angle = angle          # radians
        self.vel = pygame.Vector2(math.cos(angle), math.sin(angle)) * self.speed
        self.is_alive = True
        self._elapsed = 0.0
        self.hit_enemy = False      # set True by CombatSystem on successful hit
        if damage is not None:
            self.damage = damage

    def update(self, dt: float):
        self.pos += self.vel * dt
        self._elapsed += dt
        if self._elapsed >= self.lifetime:
            self.is_alive = False

    def draw(self, surface: pygame.Surface):
        # Tail → tip line (brown shaft)
        tail = (self.pos.x - math.cos(self.angle) * 6,
                self.pos.y - math.sin(self.angle) * 6)
        tip  = (self.pos.x + math.cos(self.angle) * 8,
                self.pos.y + math.sin(self.angle) * 8)
        pygame.draw.line(surface, (139, 90, 43),
                         (int(tail[0]), int(tail[1])),
                         (int(tip[0]),  int(tip[1])), 2)
        # Arrowhead dot
        pygame.draw.circle(surface, (200, 140, 50),
                           (int(tip[0]), int(tip[1])), 2)
