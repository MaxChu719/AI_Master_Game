"""
Spell effects for the AI Master.

Classes:
  HealingEffect   — instant AOE heal in a radius; plays a rising-sparkle animation.
  FireballPending — visual indicator while the meteor is in flight (countdown reticle).
  FireballLanding — explosion animation + AOE damage frame.

Usage (from BattleScene):
  # Healing
  effect = HealingEffect(pos, radius, heal_amount)
  effect.apply(alive_minions)       # heals all within radius immediately
  spell_effects.append(effect)      # kept for animation

  # Fireball
  pending = FireballPending(pos, radius, flight_time)
  pending = FireballPending(...)
  # After flight_time seconds, call pending.detonate() → FireballLanding
  landing = pending.detonate()
  landing.apply(alive_enemies + [boss] if boss else alive_enemies)
  spell_effects.append(landing)
"""
import math
import random
import pygame


# ── HealingEffect ─────────────────────────────────────────────────────────────

class HealingEffect:
    """
    Instant-heal AOE.  Animation: soft expanding green circle + rising sparkles.
    """
    DURATION = 1.1

    def __init__(self, pos, radius: int, heal_amount: int):
        self.pos         = pygame.Vector2(pos)
        self.radius      = radius
        self.heal_amount = heal_amount
        self.timer       = 0.0
        self.is_alive    = True

        # Pre-generate sparkle particles
        self._sparks = []
        for _ in range(30):
            a     = random.uniform(0, math.tau)
            r_off = random.uniform(0, radius * 0.9)
            self._sparks.append({
                "x":  float(pos[0]) + math.cos(a) * r_off,
                "y":  float(pos[1]) + math.sin(a) * r_off,
                "vy": -random.uniform(30, 90),   # float upward
                "life": random.uniform(0.5, 1.0),
                "max_life": 1.0,
                "size": random.randint(3, 7),
            })

    def apply(self, targets: list) -> list:
        """
        Heal every target whose position is within radius.
        Returns list of (target, amount_healed) tuples.
        """
        healed = []
        for t in targets:
            if not t.is_alive:
                continue
            if self.pos.distance_to(t.pos) <= self.radius:
                heal = min(t.max_hp - t.hp, self.heal_amount)
                t.hp += heal
                healed.append((t, heal))
        return healed

    def update(self, dt: float):
        self.timer += dt
        if self.timer >= self.DURATION:
            self.is_alive = False
        for s in self._sparks:
            s["y"]    += s["vy"] * dt
            s["life"] -= dt
        self._sparks = [s for s in self._sparks if s["life"] > 0]

    def draw(self, surface: pygame.Surface):
        t   = self.timer / self.DURATION
        cx  = int(self.pos.x)
        cy  = int(self.pos.y)

        # Expanding translucent circle
        r = int(self.radius * min(1.0, t * 2.5))
        if r > 0:
            alpha = int(100 * (1 - t))
            circ_surf = pygame.Surface((r * 2 + 4, r * 2 + 4), pygame.SRCALPHA)
            pygame.draw.circle(circ_surf, (60, 220, 80, alpha),
                               (r + 2, r + 2), r)
            surface.blit(circ_surf, (cx - r - 2, cy - r - 2))
        # Ring
        ring_r = int(self.radius * min(1.0, t * 1.8))
        if ring_r > 0:
            col_a = int(200 * (1 - t))
            pygame.draw.circle(surface, (80, 255, 100), (cx, cy), ring_r, 2)

        # Sparkles
        for s in self._sparks:
            ratio = s["life"] / s["max_life"]
            col   = (int(100 + 155 * ratio), 255, int(100 + 100 * ratio))
            sz    = max(1, int(s["size"] * ratio))
            pygame.draw.circle(surface, col, (int(s["x"]), int(s["y"])), sz)

        # Cross shimmer at centre
        shimmer = int(255 * (1 - t) * abs(math.sin(self.timer * 12)))
        if shimmer > 30:
            col = (shimmer, 255, shimmer)
            pygame.draw.line(surface, col, (cx - 12, cy), (cx + 12, cy), 2)
            pygame.draw.line(surface, col, (cx, cy - 12), (cx, cy + 12), 2)


# ── FireballPending ───────────────────────────────────────────────────────────

class FireballPending:
    """
    Visual placeholder while the fireball is in flight (flight_time seconds).
    Shows a target reticle + growing meteor icon descending from above.
    """

    def __init__(self, pos, radius: int, flight_time: float):
        self.pos         = pygame.Vector2(pos)
        self.radius      = radius
        self.flight_time = flight_time
        self.timer       = 0.0
        self.is_alive    = True   # becomes False when detonated

        # Meteor starts high up and descends
        self._meteor_start_y = pos[1] - 300
        self._meteor_end_y   = pos[1]

    @property
    def _progress(self) -> float:
        return min(1.0, self.timer / self.flight_time)

    def update(self, dt: float) -> bool:
        """Returns True when flight time elapsed (ready to detonate)."""
        self.timer += dt
        if self.timer >= self.flight_time:
            self.is_alive = False
            return True
        return False

    def detonate(self, damage: int, explosion_radius: int) -> "FireballLanding":
        """Create and return the landing explosion."""
        return FireballLanding(self.pos, damage, explosion_radius)

    def draw(self, surface: pygame.Surface):
        cx  = int(self.pos.x)
        cy  = int(self.pos.y)
        p   = self._progress

        # Danger reticle: pulsing outer ring
        pulse = 1.0 + 0.15 * math.sin(self.timer * 10)
        r     = int(self.radius * pulse)
        alpha_r = int(180 * (0.4 + 0.6 * p))
        pygame.draw.circle(surface, (220, 60, 0), (cx, cy), r, 2)

        # Inner dashed indicator (approximate with dots)
        n_dash = 16
        for k in range(n_dash):
            a  = k * (math.tau / n_dash)
            px = cx + int(math.cos(a) * self.radius * 0.6)
            py = cy + int(math.sin(a) * self.radius * 0.6)
            pygame.draw.circle(surface, (255, 120, 0), (px, py), 2)

        # Crosshair lines
        length = self.radius // 2
        pygame.draw.line(surface, (255, 60, 0), (cx - length, cy), (cx + length, cy), 1)
        pygame.draw.line(surface, (255, 60, 0), (cx, cy - length), (cx, cy + length), 1)

        # Countdown text (seconds remaining)
        remaining = max(0.0, self.flight_time - self.timer)

        # Descending meteor
        meteor_y = int(self._meteor_start_y + p * (self._meteor_end_y - self._meteor_start_y - 30))
        meteor_x = cx
        # Grow as it approaches
        meteor_r = max(4, int(6 + p * 14))
        glow_r   = meteor_r + int(4 * abs(math.sin(self.timer * 15)))
        pygame.draw.circle(surface, (180, 40, 0), (meteor_x, meteor_y), glow_r)
        pygame.draw.circle(surface, (255, 160, 20), (meteor_x, meteor_y), meteor_r)
        pygame.draw.circle(surface, (255, 255, 180), (meteor_x, meteor_y), max(2, meteor_r - 4))

        # Trail
        trail_len = int(40 * p)
        if trail_len > 0:
            for i in range(trail_len):
                ratio = (trail_len - i) / trail_len
                ty    = meteor_y - i * 3
                tr    = max(1, int(meteor_r * ratio * 0.7))
                tc    = (int(200 * ratio), int(80 * ratio), 0)
                pygame.draw.circle(surface, tc, (meteor_x, ty), tr)


# ── FireballLanding ───────────────────────────────────────────────────────────

class FireballLanding:
    """
    Fireball landing explosion. apply() deals damage; animation persists for DURATION.
    """
    DURATION = 0.8

    def __init__(self, pos, damage: int, explosion_radius: int):
        self.pos    = pygame.Vector2(pos)
        self.damage = damage
        self.radius = explosion_radius
        self.timer  = 0.0
        self.is_alive = True

        self._particles = []
        for _ in range(32):
            a     = random.uniform(0, math.tau)
            speed = random.uniform(80, 280)
            life  = random.uniform(0.4, 0.8)
            self._particles.append({
                "x": float(pos[0]), "y": float(pos[1]),
                "vx": math.cos(a) * speed, "vy": math.sin(a) * speed,
                "life": life, "max_life": life,
                "size": random.randint(3, 11),
            })

    def apply(self, targets: list) -> list:
        """
        Deal damage to every target within explosion radius.
        Returns list of (target, damage_dealt).
        """
        hits = []
        for t in targets:
            if not t.is_alive:
                continue
            if self.pos.distance_to(t.pos) <= self.radius:
                t.hp = max(0, t.hp - self.damage)
                hits.append((t, self.damage))
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
        t   = self.timer / self.DURATION
        cx  = int(self.pos.x)
        cy  = int(self.pos.y)

        # Inner blast flash
        if t < 0.25:
            flash_r = int(self.radius * 0.5 * (1 - t / 0.25))
            if flash_r > 0:
                pygame.draw.circle(surface, (255, 255, 230), (cx, cy), flash_r)

        # Expanding concentric rings
        ring_data = [
            (0.0,  0.80, (255,  60,   0), 5),
            (0.05, 0.90, (255, 140,  30), 3),
            (0.10, 1.05, (255, 220, 100), 2),
        ]
        for delay, scale, col, width in ring_data:
            rt = max(0.0, t - delay)
            r  = int(self.radius * scale * rt / (1.0 - delay + 0.01))
            r  = min(r, int(self.radius * scale))
            if r > 0:
                w = max(1, int(width * (1 - rt)))
                pygame.draw.circle(surface, col, (cx, cy), r, w)

        # Scorched ground (dark circle fading in)
        scorch_r = int(self.radius * 0.55 * t)
        if scorch_r > 2:
            pygame.draw.circle(surface, (40, 20, 0), (cx, cy), scorch_r, 3)

        # Particles
        for p in self._particles:
            ratio = p["life"] / p["max_life"]
            col   = (int(255 * ratio), int(100 * ratio ** 2), 0)
            sz    = max(1, int(p["size"] * ratio))
            pygame.draw.circle(surface, col, (int(p["x"]), int(p["y"])), sz)
