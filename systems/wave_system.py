"""
Wave system — 100 waves, boss every 5th wave.

Wave count formula:
  Waves 1-10 : from config (first_ten_counts).
  Waves 11+  : min(60, 22 + (wave_index - 10) * 2).
  Boss waves (indices 4,9,14,...): fewer swarms (boss_wave_swarm_count) + one Boss.
"""
from __future__ import annotations
import random
import pygame
from entities.enemy   import Enemy
from entities.spider  import Spider
from entities.slime   import Slime
from entities.creeper import Creeper
from entities.boss    import Boss
from config import CFG

_WC  = CFG["wave"]
_EC  = CFG["economy"]
_BC  = CFG["boss"]
_M   = CFG["arena"]["margin"]
_SLC = CFG.get("slime",   {})
_CRC = CFG.get("creeper", {})

MAX_WAVES             = int(_WC["max_waves"])
FIRST_TEN             = list(_WC["first_ten_counts"])
BOSS_INTERVAL         = int(_WC["boss_interval"])
BOSS_WAVE_SWARMS      = int(_WC["boss_wave_swarm_count"])
SPAWN_DELAY           = float(_WC["spawn_delay"])
INTERMISSION_DURATION = float(_WC["intermission_duration"])
SPIDER_SWARM_RATIO    = float(_WC.get("spider_swarm_ratio", 0.25))
COIN_PER_KILL         = int(_EC["coins_per_kill"])
COIN_PER_WAVE         = int(_EC["coins_per_wave"])
BOSS_COIN_REWARD      = int(_BC["coin_reward"])

# New enemy wave start thresholds (1-based wave number)
SLIME_WAVE_START   = int(_SLC.get("wave_start",   3))
CREEPER_WAVE_START = int(_CRC.get("wave_start",   5))

ARENA_LEFT   = _M
ARENA_TOP    = _M
ARENA_RIGHT  = CFG["arena"]["width"]  - _M
ARENA_BOTTOM = CFG["arena"]["height"] - _M


def _wave_swarm_count(wave_index: int) -> int:
    """Number of swarm enemies to spawn (excludes boss and spiders)."""
    if wave_index < 10:
        return FIRST_TEN[wave_index]
    return min(60, 22 + (wave_index - 10) * 2)


def _wave_spider_count(wave_index: int) -> int:
    """Number of spiders to spawn this wave — ~25% of the swarm count, every wave."""
    return max(1, round(_wave_swarm_count(wave_index) * SPIDER_SWARM_RATIO))


def _wave_slime_count(wave_index: int) -> int:
    """Large Slimes to spawn — 1 per 8 swarms, starting at SLIME_WAVE_START."""
    if wave_index + 1 < SLIME_WAVE_START:
        return 0
    return max(1, _wave_swarm_count(wave_index) // 8)


def _wave_creeper_count(wave_index: int) -> int:
    """Creepers to spawn — 1 per 10 swarms, starting at CREEPER_WAVE_START."""
    if wave_index + 1 < CREEPER_WAVE_START:
        return 0
    return max(1, _wave_swarm_count(wave_index) // 10)


def _is_boss_wave(wave_index: int) -> bool:
    """Return True if this wave (0-based index) includes a boss."""
    return (wave_index + 1) % BOSS_INTERVAL == 0


def _random_edge_pos():
    edge = random.randint(0, 3)
    if edge == 0:
        return (random.uniform(ARENA_LEFT, ARENA_RIGHT), float(ARENA_TOP))
    elif edge == 1:
        return (random.uniform(ARENA_LEFT, ARENA_RIGHT), float(ARENA_BOTTOM))
    elif edge == 2:
        return (float(ARENA_LEFT), random.uniform(ARENA_TOP, ARENA_BOTTOM))
    else:
        return (float(ARENA_RIGHT), random.uniform(ARENA_TOP, ARENA_BOTTOM))


def _random_edge_near_boss(boss_pos, spread: int = 80):
    """Spawn position near a boss (for swarms it summons mid-wave)."""
    angle = random.uniform(0, 3.14159 * 2)
    dist  = random.uniform(spread * 0.5, spread)
    x     = boss_pos.x + dist * __import__("math").cos(angle)
    y     = boss_pos.y + dist * __import__("math").sin(angle)
    x     = max(ARENA_LEFT  + 10, min(ARENA_RIGHT  - 10, x))
    y     = max(ARENA_TOP   + 10, min(ARENA_BOTTOM - 10, y))
    return (x, y)


class WaveState:
    INTERMISSION = "INTERMISSION"
    SPAWNING     = "SPAWNING"
    ACTIVE       = "ACTIVE"
    VICTORY      = "VICTORY"
    GAME_OVER    = "GAME_OVER"


class WaveSystem:
    def __init__(self, game_manager):
        self.game_manager = game_manager
        self.boss: Boss | None = None   # current boss (None between boss waves)
        self.reset()

    def reset(self):
        self.wave_index         = 0
        self.state              = WaveState.INTERMISSION
        self.intermission_timer = INTERMISSION_DURATION
        self.spawn_timer        = 0.0
        self.enemies_to_spawn   = 0
        self.enemies_spawned    = 0
        self._prev_alive_count  = 0
        self._boss_was_alive    = False
        self.boss               = None
        self._spawn_queue: list = []   # ordered list of "swarm" / "spider" tokens
        self.game_manager.wave_number = 1

    @property
    def wave_number(self) -> int:
        return self.wave_index + 1

    @property
    def intermission_seconds_left(self) -> int:
        return max(1, int(self.intermission_timer) + 1)

    @property
    def is_boss_wave(self) -> bool:
        return _is_boss_wave(self.wave_index)

    # ------------------------------------------------------------------

    def update(self, dt: float, fighters: list, archers: list, enemies: list):
        """
        fighters / archers: lists of Fighter / Archer objects.
        enemies           : mutable list shared with BattleScene.
        Boss is on self.boss (separate from swarm enemies list).
        """
        if self.state in (WaveState.GAME_OVER, WaveState.VICTORY):
            return

        all_minions = fighters + archers

        def _all_dead():
            return (not any(f.is_alive for f in fighters) and
                    not any(a.is_alive for a in archers))

        if self.state == WaveState.INTERMISSION:
            if _all_dead():
                self.state = WaveState.GAME_OVER
                return
            self.intermission_timer -= dt
            if self.intermission_timer <= 0:
                self._start_spawning(enemies)

        elif self.state == WaveState.SPAWNING:
            if _all_dead():
                self.state = WaveState.GAME_OVER
                return
            self._award_kills(enemies)

            self.spawn_timer -= dt
            if self.spawn_timer <= 0 and self.enemies_spawned < self.enemies_to_spawn:
                etype = self._spawn_queue[self.enemies_spawned]
                if etype == "spider":
                    e = Spider(_random_edge_pos())
                elif etype == "slime":
                    e = Slime(_random_edge_pos(), generation=0)
                elif etype == "creeper":
                    e = Creeper(_random_edge_pos())
                else:
                    e = Enemy(_random_edge_pos())
                    _ecc = CFG["enemy"]
                    e.speed = _ecc["base_speed"] + self.wave_index * _ecc["speed_per_wave"]
                enemies.append(e)
                self.enemies_spawned += 1
                self.spawn_timer = SPAWN_DELAY

            self._prev_alive_count = sum(1 for e in enemies if e.is_alive)

            if self.enemies_spawned >= self.enemies_to_spawn:
                self.state = WaveState.ACTIVE

        elif self.state == WaveState.ACTIVE:
            self._award_kills(enemies)
            self._award_boss_kill()

            swarms_alive = sum(1 for e in enemies if e.is_alive)
            boss_alive   = (self.boss is not None and self.boss.is_alive)

            wave_cleared = (swarms_alive == 0 and not boss_alive and len(enemies) > 0) or \
                           (swarms_alive == 0 and not self.is_boss_wave and len(enemies) > 0) or \
                           (swarms_alive == 0 and self.boss is None and len(enemies) > 0)

            # Also check if boss was present and now dead, all swarms gone
            if self.boss is not None:
                boss_done = not self.boss.is_alive and not self.boss._dying
            else:
                boss_done = True

            wave_cleared = (swarms_alive == 0 and boss_done and
                            (len(enemies) > 0 or self.boss is not None))

            if wave_cleared:
                self.game_manager.coins += COIN_PER_WAVE
                self._advance_wave(fighters, archers, enemies)
                return

            if _all_dead():
                self.state = WaveState.GAME_OVER

    def _award_kills(self, enemies: list):
        alive = sum(1 for e in enemies if e.is_alive)
        killed = self._prev_alive_count - alive
        if killed > 0:
            self.game_manager.coins += killed * COIN_PER_KILL
        self._prev_alive_count = alive

    def _award_boss_kill(self):
        """Check if boss just died and award coins."""
        if self.boss is None:
            return
        boss_alive_now = self.boss.is_alive
        if self._boss_was_alive and not boss_alive_now:
            self.game_manager.coins += BOSS_COIN_REWARD
        self._boss_was_alive = boss_alive_now

    def _start_spawning(self, enemies: list):
        enemies.clear()
        self.boss = None

        is_bw      = _is_boss_wave(self.wave_index)
        n_swarms   = BOSS_WAVE_SWARMS if is_bw else _wave_swarm_count(self.wave_index)
        n_spiders  = _wave_spider_count(self.wave_index)
        n_slimes   = 0 if is_bw else _wave_slime_count(self.wave_index)
        n_creepers = 0 if is_bw else _wave_creeper_count(self.wave_index)

        # Build a shuffled spawn queue
        self._spawn_queue = (["swarm"]   * n_swarms  +
                             ["spider"]  * n_spiders +
                             ["slime"]   * n_slimes  +
                             ["creeper"] * n_creepers)
        random.shuffle(self._spawn_queue)
        self.enemies_to_spawn = len(self._spawn_queue)
        self.enemies_spawned  = 0
        self.spawn_timer      = 0.0
        self._prev_alive_count = 0

        if is_bw:
            self.boss = Boss(_random_edge_pos(), self.wave_index)
            self._boss_was_alive = True
        else:
            self._boss_was_alive = False

        self.state = WaveState.SPAWNING

    def _advance_wave(self, fighters: list, archers: list, enemies: list):
        """Advance to next wave, preserving minion HP/alive status."""
        enemies.clear()
        self.boss       = None
        self.wave_index += 1

        def _all_dead():
            return (not any(f.is_alive for f in fighters) and
                    not any(a.is_alive for a in archers))

        if _all_dead():
            self.state = WaveState.GAME_OVER
            return

        if self.wave_index >= MAX_WAVES:
            self.state = WaveState.VICTORY
            self.game_manager.wave_number = MAX_WAVES
        else:
            self.game_manager.wave_number  = self.wave_index + 1
            self.state                     = WaveState.INTERMISSION
            self.intermission_timer        = INTERMISSION_DURATION

    # ------------------------------------------------------------------
    # Boss-spawned swarms (called by BattleScene when boss requests them)

    def spawn_swarms_near_boss(self, enemies: list, count: int):
        """Spawn 'count' swarm enemies near the current boss position."""
        if self.boss is None:
            return
        for _ in range(count):
            pos = _random_edge_near_boss(self.boss.pos)
            e   = Enemy(pos)
            _ec = CFG["enemy"]
            e.speed = _ec["base_speed"] + self.wave_index * _ec["speed_per_wave"]
            enemies.append(e)
