"""
Slime enemy — melee attacker that splits on death.

Three generations:
  Generation 0 (Large):  80 HP, size 28, splits into 2× Medium on death.
  Generation 1 (Medium): 40 HP, size 18, splits into 2× Small on death.
  Generation 2 (Small):  20 HP, size 12, dies without splitting.

Behaviour:
  - Chases the nearest alive minion (like a swarm enemy).
  - Bounces/wobbles visually; changes color slightly per generation.

enemy_type = 2  (used for MinionEnv observation encoding)
"""
from __future__ import annotations
import math
import random
import pygame
from config import CFG

_SC = CFG.get("slime", {})

_HP      = [_SC.get("large_hp",   80), _SC.get("medium_hp", 40), _SC.get("small_hp", 20)]
_SIZE    = [_SC.get("large_size", 28), _SC.get("medium_size", 18), _SC.get("small_size", 12)]
_SPEED   = [_SC.get("large_speed", 55.0), _SC.get("medium_speed", 75.0), _SC.get("small_speed", 100.0)]
_ATTACK  = [_SC.get("large_attack", 10),  _SC.get("medium_attack", 6),   _SC.get("small_attack", 3)]

# Green tones per generation
_BODY_COL  = [(60, 200,  40), (100, 220,  60), (150, 240, 100)]
_DARK_COL  = [(30, 150,  20), ( 60, 170,  40), (110, 200,  70)]
_EYE_COL   = [(255, 255, 200), (255, 255, 200), (255, 255, 200)]


class Slime:
    enemy_type = 2   # for MinionEnv observation encoding

    def __init__(self, pos, generation: int = 0):
        self.pos            = pygame.Vector2(pos)
        self.generation     = max(0, min(2, generation))
        self.hp             = _HP[self.generation]
        self.max_hp         = self.hp
        self.speed          = _SPEED[self.generation]
        self.size           = _SIZE[self.generation]
        self.is_alive       = True
        self.attack_damage  = _ATTACK[self.generation]
        self.attack_range   = float(_SC.get("attack_range",   28.0))
        self.attack_cooldown = float(_SC.get("attack_cooldown", 1.2))
        self.attack_timer   = 0.0
        self.attack_flash_timer = 0.0

        # Status effects
        self.frozen_timer   = 0.0
        self.burn_timer     = 0.0
        self.burn_dps       = 0.0

        # Physics
        self.velocity       = pygame.Vector2(0, 0)
        self.knockback_vel  = pygame.Vector2(0, 0)

        # Animation
        self._anim_timer    = random.uniform(0, math.tau)   # randomise phase
        self._split_flash   = 0.0   # brief flash when hit
        # Grave timer: -1 = uninitialized; >0 = showing grave; 0 = expired
        self.grave_timer    = -1.0

    # ------------------------------------------------------------------

    def tick(self, dt: float):
        if self.is_alive:
            self._anim_timer += dt
            if self._split_flash > 0:
                self._split_flash = max(0.0, self._split_flash - dt)

    # ------------------------------------------------------------------

    def draw(self, surface: pygame.Surface):
        if not self.is_alive:
            if self.grave_timer > 0:
                from entities.mage_projectile import draw_enemy_grave
                draw_enemy_grave(surface, self.pos, self.size,
                                 stone=(58, 92, 62),
                                 alpha=min(1.0, self.grave_timer / 0.5))
            return

        cx, cy = int(self.pos.x), int(self.pos.y)
        r      = self.size // 2

        # Squash-and-stretch bob (generation affects frequency slightly)
        freq  = 3.0 + self.generation * 0.5
        bob   = math.sin(self._anim_timer * freq)
        rx    = r + int(bob * r * 0.18)     # stretch horizontal
        ry    = r - int(bob * r * 0.18)     # squash vertical
        cy_f  = cy + int(abs(bob) * r * 0.08)   # ground contact shift

        body_col = _BODY_COL[self.generation]
        dark_col = _DARK_COL[self.generation]

        # Attack flash
        if self.attack_flash_timer > 0:
            t   = self.attack_flash_timer / 0.15
            bright = min(255, int(80 + 175 * t))
            body_col = (bright, min(255, body_col[1] + 30), bright // 3)

        # Split flash (white-out on taking damage)
        if self._split_flash > 0:
            t = self._split_flash / 0.15
            body_col = (min(255, body_col[0] + int(200 * t)),
                        min(255, body_col[1] + int(200 * t)),
                        min(255, body_col[2] + int(200 * t)))

        # Body (ellipse)
        pygame.draw.ellipse(surface, body_col,
                            (cx - rx, cy_f - ry, rx * 2, ry * 2))
        # Dark underside
        pygame.draw.ellipse(surface, dark_col,
                            (cx - rx, cy_f, rx * 2, ry), 0)
        # Outline
        pygame.draw.ellipse(surface, dark_col,
                            (cx - rx, cy_f - ry, rx * 2, ry * 2), 1)

        # Eyes (two dots near top)
        eo = max(2, r // 3)
        ey = cy_f - ry // 2
        pygame.draw.circle(surface, _EYE_COL[self.generation], (cx - eo, ey), max(1, r // 5))
        pygame.draw.circle(surface, _EYE_COL[self.generation], (cx + eo, ey), max(1, r // 5))

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
        bar_y = cy_f - ry - 8
        pygame.draw.rect(surface, (140, 30, 30), (bar_x, bar_y, bar_w, bar_h))
        fw = int(bar_w * max(0, self.hp) / max(1, self.max_hp))
        if fw > 0:
            pygame.draw.rect(surface, (60, 200, 60), (bar_x, bar_y, fw, bar_h))
