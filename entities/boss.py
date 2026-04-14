"""
Boss entity — slow, heavily animated, fires volleys of fireballs, spawns swarms.

Classes:
  BossFireball   — projectile that flies to a target position then explodes.
  BossExplosion  — particle + shockwave animation; deals AOE damage.
  Boss           — main boss entity with phase transitions and cool animations.
"""
import math
import random
import pygame
from config import CFG

_BC = CFG["boss"]


# ── BossFireball ──────────────────────────────────────────────────────────────

class BossFireball:
    """A fireball projectile fired by the boss toward a target position."""

    def __init__(self, pos, target_pos, damage: int, explosion_radius: int):
        self.pos              = pygame.Vector2(pos)
        self._target          = pygame.Vector2(target_pos)
        self.damage           = damage
        self.explosion_radius = explosion_radius
        self.size             = 10
        self.is_alive         = True

        diff = self._target - self.pos
        dist = diff.length()
        speed = float(_BC["fireball_speed"])
        self._travel_time = dist / speed if dist > 0 else 0.01
        self._elapsed     = 0.0

        if dist > 0:
            self.velocity = diff.normalize() * speed
        else:
            self.velocity = pygame.Vector2(speed, 0)

        # Particle trail
        self._trail: list = []   # [x, y, life, max_life, r]
        self._trail_acc   = 0.0

    def update(self, dt: float) -> bool:
        """Advance by dt. Returns True if impact occurred this frame."""
        self._elapsed += dt

        # Trail
        self._trail_acc += dt
        if self._trail_acc >= 0.04:
            self._trail_acc = 0.0
            self._trail.append([self.pos.x, self.pos.y, 0.35, 0.35,
                                 random.randint(4, 9)])
        for p in self._trail:
            p[2] -= dt
        self._trail = [p for p in self._trail if p[2] > 0]

        self.pos += self.velocity * dt

        if self._elapsed >= self._travel_time:
            self.is_alive = False
            return True
        return False

    def draw(self, surface: pygame.Surface):
        # Trail (older = dimmer, smaller)
        for px, py, life, max_life, pr in self._trail:
            ratio = life / max_life
            r     = max(1, int(pr * ratio))
            col   = (int(200 * ratio), int(70 * ratio), 0)
            pygame.draw.circle(surface, col, (int(px), int(py)), r)

        # Pulsing glow
        t        = self._elapsed / max(0.001, self._travel_time)
        pulse    = int(5 * abs(math.sin(self._elapsed * 18)))
        glow_r   = self.size + pulse + 3
        pygame.draw.circle(surface, (200, 60, 0),
                           (int(self.pos.x), int(self.pos.y)), glow_r)
        # Core
        pygame.draw.circle(surface, (255, 180, 30),
                           (int(self.pos.x), int(self.pos.y)), self.size - 1)
        # Bright centre
        pygame.draw.circle(surface, (255, 255, 200),
                           (int(self.pos.x), int(self.pos.y)), max(1, self.size - 5))


# ── BossExplosion ─────────────────────────────────────────────────────────────

class BossExplosion:
    """Visual-only explosion animation. Caller handles actual damage separately."""

    DURATION = 0.65

    def __init__(self, pos, radius: int):
        self.pos      = pygame.Vector2(pos)
        self.radius   = radius
        self.timer    = 0.0
        self.is_alive = True

        # Debris particles
        self._particles = []
        for _ in range(24):
            ang   = random.uniform(0, math.tau)
            speed = random.uniform(60, 220)
            life  = random.uniform(0.35, 0.65)
            self._particles.append({
                "x": float(pos[0]), "y": float(pos[1]),
                "vx": math.cos(ang) * speed, "vy": math.sin(ang) * speed,
                "life": life, "max_life": life,
                "size": random.randint(3, 9),
            })

    def update(self, dt: float):
        self.timer += dt
        if self.timer >= self.DURATION:
            self.is_alive = False
        for p in self._particles:
            p["x"] += p["vx"] * dt
            p["y"] += p["vy"] * dt
            p["life"] -= dt
        self._particles = [p for p in self._particles if p["life"] > 0]

    def draw(self, surface: pygame.Surface):
        t   = self.timer / self.DURATION          # 0 → 1
        cx  = int(self.pos.x)
        cy  = int(self.pos.y)

        # Inner flash (only early)
        if t < 0.3:
            inner_r = int(self.radius * 0.4 * (1 - t / 0.3))
            if inner_r > 0:
                pygame.draw.circle(surface, (255, 255, 210), (cx, cy), inner_r)

        # Expanding rings
        for offset, col in [
            (0,    (255,  90,   0)),
            (0.06, (255, 170,  40)),
            (0.12, (255, 220, 100)),
        ]:
            rt = max(0.0, t - offset)
            r  = int(self.radius * rt / (1.0 - offset + 0.01))
            r  = min(r, int(self.radius * 1.15))
            if r > 0:
                width = max(1, int(4 * (1 - rt)))
                pygame.draw.circle(surface, col, (cx, cy), r, width)

        # Debris particles
        for p in self._particles:
            ratio = p["life"] / p["max_life"]
            col   = (int(255 * ratio), int(100 * ratio * ratio), 0)
            sz    = max(1, int(p["size"] * ratio))
            pygame.draw.circle(surface, col, (int(p["x"]), int(p["y"])), sz)


# ── Boss ─────────────────────────────────────────────────────────────────────

class Boss:
    """
    Boss enemy.

    • Moves slowly toward the nearest alive target.
    • Every fireball_cooldown seconds fires fireball_count fireballs (more in phase 2).
    • Every swarm_spawn_cooldown seconds requests swarm spawns via events.
    • Phase 2 triggers at phase2_hp_fraction (more intense visuals, extra fireballs/swarms).
    • Death triggers a multi-ring shockwave animation.
    """

    def __init__(self, pos, wave_index: int):
        self.pos        = pygame.Vector2(pos)
        self.wave_index = wave_index

        # Stats (scale with wave)
        self.max_hp      = int(_BC["hp_base"]) + wave_index * int(_BC["hp_per_wave"])
        self.hp          = self.max_hp
        self.speed       = float(_BC["speed"])
        self.size        = int(_BC["size"])
        self.is_alive    = True

        self.attack_damage   = int(_BC["attack_damage"])
        self.attack_range    = float(_BC["attack_range"])
        self.attack_cooldown = 1.2
        self.attack_timer    = 0.0
        self.attack_flash_timer = 0.0

        self.velocity   = pygame.Vector2(0, 0)
        self.enemy_type = 3   # distinct type for observation encoding (boss)

        # ── Animation state ─────────────────────────────────────────────
        self._anim_t    = 0.0
        self._ring_ang  = 0.0
        self._orb_angs  = [i * math.tau / 3 for i in range(3)]   # 3 base orbs

        # ── Fireball attack ──────────────────────────────────────────────
        self._fb_timer     = 0.0
        self._fb_cooldown  = float(_BC["fireball_cooldown"])
        self._fb_damage    = int(_BC["fireball_damage"]) + wave_index * 3
        self._fb_radius    = int(_BC["fireball_explosion_radius"])
        self.fireballs: list[BossFireball]  = []
        self.explosions: list[BossExplosion] = []

        # ── Swarm spawn ──────────────────────────────────────────────────
        self._swarm_timer   = 0.0
        self._swarm_cd      = float(_BC["swarm_spawn_cooldown"])

        # ── Death ────────────────────────────────────────────────────────
        self._dying           = False
        self._death_t         = 0.0
        self._death_duration  = 1.8
        self._death_rings: list = []  # [current_r, max_r, elapsed]

        # Cached font for label
        self._font = pygame.font.SysFont("arial", 14, bold=True)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def is_phase2(self) -> bool:
        return self.hp / self.max_hp <= float(_BC.get("phase2_hp_fraction", 0.5))

    # ── Update ────────────────────────────────────────────────────────────────

    def update(self, dt: float, targets: list) -> dict:
        """
        Update the boss.

        targets: list of alive minion/archer objects.
        Returns event dict:
          swarms_to_spawn  (int)       — number of new swarms to spawn near the boss
          new_explosions   (list)      — BossExplosion objects this frame
          boss_dead        (bool)
        """
        events = {"swarms_to_spawn": 0, "new_explosions": [], "boss_dead": False}

        # Always update projectiles/explosions even when dying/dead
        for fb in self.fireballs:
            hit = fb.update(dt)
            if hit:
                exp = BossExplosion((fb.pos.x, fb.pos.y), self._fb_radius)
                self.explosions.append(exp)
                events["new_explosions"].append(exp)
        self.fireballs = [fb for fb in self.fireballs if fb.is_alive]

        for exp in self.explosions:
            exp.update(dt)
        self.explosions = [e for e in self.explosions if e.is_alive]

        # Death animation
        if self._dying:
            self._death_t += dt
            for ring in self._death_rings:
                ring[0] += 180 * dt
                ring[2] += dt
            if self._death_t >= self._death_duration:
                self._dying = False
            return events

        if not self.is_alive:
            return events

        # ── Animate ─────────────────────────────────────────────────────
        self._anim_t   += dt
        rot_speed       = 1.8 if self.is_phase2 else 0.9
        self._ring_ang += dt * rot_speed
        orb_speed       = 1.4 if self.is_phase2 else 0.7
        for i in range(len(self._orb_angs)):
            self._orb_angs[i] += dt * (orb_speed + i * 0.25)

        # ── Move toward nearest target ──────────────────────────────────
        alive_tgts = [t for t in targets if t.is_alive]
        if alive_tgts:
            nearest = min(alive_tgts, key=lambda t: self.pos.distance_to(t.pos))
            diff    = nearest.pos - self.pos
            if diff.length() > self.size * 0.6:
                self.velocity = diff.normalize() * self.speed
                self.pos     += self.velocity * dt
            else:
                self.velocity = pygame.Vector2(0, 0)

        # ── Attack cooldowns ────────────────────────────────────────────
        if self.attack_timer        > 0: self.attack_timer        -= dt
        if self.attack_flash_timer  > 0: self.attack_flash_timer  -= dt

        # ── Fireball volleys ────────────────────────────────────────────
        self._fb_timer += dt
        if self._fb_timer >= self._fb_cooldown and alive_tgts:
            self._fb_timer = 0.0
            n_fb = int(_BC["fireball_count"]) + (2 if self.is_phase2 else 0)
            target = min(alive_tgts, key=lambda t: self.pos.distance_to(t.pos))
            for i in range(n_fb):
                # Fan the fireballs around the target
                angle_offset = (i - (n_fb - 1) / 2.0) * 0.35
                offset_vec   = pygame.Vector2(
                    math.cos(angle_offset) * 25 * (i - n_fb // 2),
                    math.sin(angle_offset) * 25 * (i - n_fb // 2),
                )
                tgt_pos = target.pos + offset_vec
                fb = BossFireball(
                    (self.pos.x, self.pos.y),
                    (tgt_pos.x,  tgt_pos.y),
                    self._fb_damage,
                    self._fb_radius,
                )
                self.fireballs.append(fb)

        # ── Swarm spawning ───────────────────────────────────────────────
        self._swarm_timer += dt
        if self._swarm_timer >= self._swarm_cd:
            self._swarm_timer = 0.0
            n = int(_BC["swarm_count_per_spawn"]) + (2 if self.is_phase2 else 0)
            events["swarms_to_spawn"] = n

        # ── HP death check ───────────────────────────────────────────────
        if self.hp <= 0:
            self.is_alive = False
            self._dying   = True
            self._death_t = 0.0
            # Seed death rings
            for k in range(5):
                self._death_rings.append([0.0, self.size * (2.0 + k * 1.6), 0.0])
            events["boss_dead"] = True

        return events

    def take_damage(self, damage: int):
        if not self.is_alive:
            return
        self.hp = max(0, self.hp - damage)

    # ── Draw ─────────────────────────────────────────────────────────────────

    def draw(self, surface: pygame.Surface):
        # Always draw active fireballs/explosions
        for fb in self.fireballs:
            fb.draw(surface)
        for exp in self.explosions:
            exp.draw(surface)

        # Death rings
        if self._dying:
            cx, cy = int(self.pos.x), int(self.pos.y)
            for ring in self._death_rings:
                r     = int(ring[0])
                ratio = 1.0 - ring[2] / self._death_duration
                if r > 0:
                    col = (int(255 * ratio), int(200 * ratio), int(50 * ratio))
                    pygame.draw.circle(surface, col, (cx, cy), r, max(1, int(4 * ratio)))
            return

        if not self.is_alive:
            return

        cx, cy = int(self.pos.x), int(self.pos.y)
        t       = self._anim_t
        p2      = self.is_phase2

        # ── Background glow ──────────────────────────────────────────────
        glow_r = self.size + int(10 * math.sin(t * 2.5)) + (6 if p2 else 0)
        glow_c = (160, 10, 60) if p2 else (90, 10, 80)
        pygame.draw.circle(surface, glow_c, (cx, cy), glow_r)

        # ── Main body ────────────────────────────────────────────────────
        body_c = (230, 30, 50) if p2 else (150, 15, 110)
        half   = self.size // 2
        rect   = pygame.Rect(cx - half, cy - half, self.size, self.size)
        pygame.draw.rect(surface, body_c, rect, border_radius=10)
        # Inner highlight
        inner_c = (255, 80, 90) if p2 else (200, 50, 160)
        irect   = rect.inflate(-12, -12)
        pygame.draw.rect(surface, inner_c, irect, border_radius=6)

        # ── Rotating outer ring (octagonal) ─────────────────────────────
        ring_r  = self.size + 18
        n_pts   = 8
        ring_c  = (255, 110, 30) if p2 else (180, 60, 220)
        pts     = []
        for k in range(n_pts):
            a    = self._ring_ang + k * (math.tau / n_pts)
            pts.append((cx + math.cos(a) * ring_r, cy + math.sin(a) * ring_r))
        for k in range(n_pts):
            pygame.draw.line(surface, ring_c, pts[k], pts[(k + 1) % n_pts], 2)

        # Second counter-rotating ring (phase 2 only)
        if p2:
            ring2_r = self.size + 30
            ring2_c = (255, 200, 0)
            pts2    = []
            for k in range(n_pts):
                a    = -self._ring_ang * 1.3 + k * (math.tau / n_pts)
                pts2.append((cx + math.cos(a) * ring2_r, cy + math.sin(a) * ring2_r))
            for k in range(n_pts):
                pygame.draw.line(surface, ring2_c, pts2[k], pts2[(k + 1) % n_pts], 1)

        # ── Orbiting orbs ────────────────────────────────────────────────
        n_orbs  = 5 if p2 else 3
        orb_r   = self.size + 26
        orb_c   = (255, 150, 0) if p2 else (160, 80, 255)
        for i in range(n_orbs):
            ang = self._orb_angs[i % len(self._orb_angs)]
            ox  = cx + math.cos(ang) * orb_r
            oy  = cy + math.sin(ang) * orb_r
            pygame.draw.circle(surface, orb_c,        (int(ox), int(oy)), 7)
            pygame.draw.circle(surface, (255, 255, 200), (int(ox), int(oy)), 3)

        # ── Eyes ─────────────────────────────────────────────────────────
        eye_c    = (255, 255, 50) if p2 else (255, 160, 0)
        eye_off  = self.size // 4
        for ex, ey in [(cx - eye_off, cy - eye_off // 2),
                       (cx + eye_off, cy - eye_off // 2)]:
            pygame.draw.circle(surface, eye_c, (ex, ey), 6)
            pygame.draw.circle(surface, (255, 255, 255), (ex, ey), 2)

        # ── Rune markings on body ────────────────────────────────────────
        rune_c = (255, 200, 255) if p2 else (180, 100, 200)
        r_half = half - 6
        for k in range(4):
            a  = t * 0.5 + k * (math.tau / 4)
            rx = cx + int(math.cos(a) * r_half * 0.5)
            ry = cy + int(math.sin(a) * r_half * 0.5)
            pygame.draw.circle(surface, rune_c, (rx, ry), 3)

        # ── HP bar ───────────────────────────────────────────────────────
        bar_w = self.size * 2 + 14
        bar_h = 8
        bx    = cx - bar_w // 2
        by    = cy - half - 18
        pygame.draw.rect(surface, (60, 15, 15), (bx, by, bar_w, bar_h), border_radius=4)
        fill_w = max(0, int(bar_w * self.hp / self.max_hp))
        if fill_w > 0:
            fc = (255, 40, 40) if p2 else (200, 20, 140)
            pygame.draw.rect(surface, fc, (bx, by, fill_w, bar_h), border_radius=4)
        pygame.draw.rect(surface, (200, 80, 80), (bx, by, bar_w, bar_h), 1, border_radius=4)

        # "BOSS" label
        lbl   = self._font.render("BOSS", True, (255, 220, 255) if p2 else (220, 180, 255))
        lx    = cx - lbl.get_width() // 2
        ly    = by - lbl.get_height() - 2
        surface.blit(lbl, (lx, ly))
