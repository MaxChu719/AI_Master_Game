import pygame
from config import CFG

_EC = CFG["enemy"]


class Enemy:
    enemy_type = 0   # 0 = Swarm (used by MinionEnv for observation encoding)

    def __init__(self, pos):
        self.pos = pygame.Vector2(pos)
        self.hp = _EC["hp"]
        self.max_hp = _EC["hp"]
        self.speed = _EC["base_speed"]
        self.size = _EC["size"]
        self.color = (220, 50, 50)
        self.is_alive = True
        self.attack_damage = _EC["attack_damage"]
        self.attack_range = _EC["attack_range"]
        self.attack_cooldown = _EC["attack_cooldown"]
        self.attack_timer = 0.0
        # Attack flash: brief bright ring drawn when swarm bites
        self.attack_flash_timer = 0.0  # counts down from 0.15 to 0
        # Velocity vector — updated each frame by MovementSystem for DQN observation
        self.velocity = pygame.Vector2(0, 0)
        # Knockback velocity — applied and decayed by MovementSystem each frame
        self.knockback_vel = pygame.Vector2(0, 0)

    def draw(self, surface: pygame.Surface):
        # Attack flash ring (behind body so it looks like an aura burst)
        if self.attack_flash_timer > 0:
            t = self.attack_flash_timer / 0.15
            intensity = int(255 * t)
            ring_r = self.size + 5
            pygame.draw.circle(surface, (255, intensity, 0),
                                (int(self.pos.x), int(self.pos.y)), ring_r, 2)

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
