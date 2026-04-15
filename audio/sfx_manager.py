"""
Procedurally generated sound effects using numpy + pygame.sndarray.
No audio asset files are required — all sounds are synthesized at startup.
"""
import numpy as np
import pygame

_RATE = 44100  # must match pygame.mixer.pre_init settings


def _to_sound(mono: np.ndarray) -> pygame.mixer.Sound:
    """Convert a mono float32 array (range -1..1) to a stereo int16 Sound."""
    pcm = np.clip(mono, -1.0, 1.0)
    pcm16 = (pcm * 16000).astype(np.int16)
    stereo = np.ascontiguousarray(np.column_stack([pcm16, pcm16]))
    return pygame.sndarray.make_sound(stereo)


def _sword_swing() -> pygame.mixer.Sound:
    """Metallic clang: noise burst + short resonant tone, fast decay."""
    dur = 0.12
    n = int(_RATE * dur)
    t = np.linspace(0, dur, n, dtype=np.float32)
    noise = np.random.default_rng(1).uniform(-1, 1, n).astype(np.float32)
    tone = (np.sin(2 * np.pi * 520 * t) * 0.5 + np.sin(2 * np.pi * 1040 * t) * 0.25)
    env = np.exp(-20 * t)
    return _to_sound((noise * 0.5 + tone) * env * 0.8)


def _arrow_shoot() -> pygame.mixer.Sound:
    """Quick whoosh: filtered noise with a fast drop in pitch."""
    dur = 0.14
    n = int(_RATE * dur)
    t = np.linspace(0, dur, n, dtype=np.float32)
    noise = np.random.default_rng(2).uniform(-1, 1, n).astype(np.float32)
    freq = np.linspace(900, 200, n)
    sweep = np.sin(2 * np.pi * np.cumsum(freq / _RATE)).astype(np.float32)
    env = np.exp(-10 * t) * np.clip(t * 80, 0, 1)
    return _to_sound((noise * 0.35 + sweep * 0.65) * env)


def _hit_impact() -> pygame.mixer.Sound:
    """Short thud for arrow or projectile hitting an enemy."""
    dur = 0.09
    n = int(_RATE * dur)
    t = np.linspace(0, dur, n, dtype=np.float32)
    noise = np.random.default_rng(3).uniform(-1, 1, n).astype(np.float32)
    tone = np.sin(2 * np.pi * 100 * t).astype(np.float32) * 0.7
    env = np.exp(-35 * t)
    return _to_sound((noise * 0.4 + tone) * env)


def _enemy_death() -> pygame.mixer.Sound:
    """Descending pitch burst when an enemy dies."""
    dur = 0.10
    n = int(_RATE * dur)
    t = np.linspace(0, dur, n, dtype=np.float32)
    freq = np.linspace(280, 80, n)
    sweep = np.sin(2 * np.pi * np.cumsum(freq / _RATE)).astype(np.float32) * 0.7
    noise = np.random.default_rng(4).uniform(-1, 1, n).astype(np.float32) * 0.2
    env = np.exp(-18 * t)
    return _to_sound((sweep + noise) * env)


def _fireball_shoot() -> pygame.mixer.Sound:
    """Deep whoosh with a rising crackle — fireball leaving the mage's hand."""
    dur = 0.20
    n   = int(_RATE * dur)
    t   = np.linspace(0, dur, n, dtype=np.float32)
    noise = np.random.default_rng(10).uniform(-1, 1, n).astype(np.float32)
    freq  = np.linspace(200, 600, n)
    sweep = np.sin(2 * np.pi * np.cumsum(freq / _RATE)).astype(np.float32)
    env   = np.clip(t * 40, 0, 1) * np.exp(-8 * t)
    return _to_sound((noise * 0.3 + sweep * 0.7) * env * 0.9)


def _mage_explosion() -> pygame.mixer.Sound:
    """Booming crunch for Fire Mage AoE explosion."""
    dur = 0.30
    n   = int(_RATE * dur)
    t   = np.linspace(0, dur, n, dtype=np.float32)
    noise = np.random.default_rng(11).uniform(-1, 1, n).astype(np.float32)
    sub   = np.sin(2 * np.pi * 55 * t).astype(np.float32) * 0.8
    crack = np.sin(2 * np.pi * 280 * t).astype(np.float32) * 0.35
    env   = np.exp(-7 * t) * np.clip(t * 60, 0, 1)
    return _to_sound((noise * 0.4 + sub + crack) * env)


def _iceball_shoot() -> pygame.mixer.Sound:
    """Crisp icy whoosh for iceball projectile."""
    dur = 0.16
    n   = int(_RATE * dur)
    t   = np.linspace(0, dur, n, dtype=np.float32)
    noise = np.random.default_rng(12).uniform(-1, 1, n).astype(np.float32)
    freq  = np.linspace(800, 1800, n)
    sweep = np.sin(2 * np.pi * np.cumsum(freq / _RATE)).astype(np.float32)
    env   = np.exp(-12 * t) * np.clip(t * 80, 0, 1)
    return _to_sound((noise * 0.2 + sweep * 0.8) * env * 0.7)


def _freeze_hit() -> pygame.mixer.Sound:
    """Glassy crack when Ice Mage freeze hits an enemy."""
    dur = 0.14
    n   = int(_RATE * dur)
    t   = np.linspace(0, dur, n, dtype=np.float32)
    noise = np.random.default_rng(13).uniform(-1, 1, n).astype(np.float32)
    tone  = (np.sin(2 * np.pi * 900 * t) * 0.5 +
             np.sin(2 * np.pi * 1400 * t) * 0.3).astype(np.float32)
    env   = np.exp(-22 * t)
    return _to_sound((noise * 0.3 + tone) * env * 0.8)


def _slime_split() -> pygame.mixer.Sound:
    """Wet bloop when a slime splits into smaller pieces."""
    dur = 0.18
    n   = int(_RATE * dur)
    t   = np.linspace(0, dur, n, dtype=np.float32)
    noise = np.random.default_rng(14).uniform(-1, 1, n).astype(np.float32)
    freq  = np.linspace(320, 80, n)
    sweep = np.sin(2 * np.pi * np.cumsum(freq / _RATE)).astype(np.float32) * 0.6
    env   = np.exp(-14 * t) * np.clip(t * 50, 0, 1)
    return _to_sound((noise * 0.35 + sweep) * env * 0.9)


def _creeper_fuse() -> pygame.mixer.Sound:
    """Sputtering hiss of a creeper's fuse activating."""
    dur = 0.22
    n   = int(_RATE * dur)
    t   = np.linspace(0, dur, n, dtype=np.float32)
    noise = np.random.default_rng(15).uniform(-1, 1, n).astype(np.float32)
    freq  = 1200 + 200 * np.sin(t * 40)
    sweep = np.sin(2 * np.pi * np.cumsum(freq / _RATE)).astype(np.float32) * 0.3
    env   = np.clip(t * 30, 0, 1) * (1 - t / dur * 0.3)
    return _to_sound((noise * 0.65 + sweep) * env * 0.6)


def _creeper_explosion() -> pygame.mixer.Sound:
    """Deep, punchy explosion for the Creeper boom."""
    dur = 0.35
    n   = int(_RATE * dur)
    t   = np.linspace(0, dur, n, dtype=np.float32)
    noise = np.random.default_rng(16).uniform(-1, 1, n).astype(np.float32)
    sub   = np.sin(2 * np.pi * 40 * t).astype(np.float32) * 0.9
    mid   = np.sin(2 * np.pi * 120 * t).astype(np.float32) * 0.4
    env   = np.exp(-5 * t) * np.clip(t * 80, 0, 1)
    return _to_sound((noise * 0.5 + sub + mid) * env)


def _summon_portal() -> pygame.mixer.Sound:
    """Rising mystical chime for the AI Master's summoning portal."""
    dur = 0.60
    n   = int(_RATE * dur)
    t   = np.linspace(0, dur, n, dtype=np.float32)
    freq  = np.linspace(300, 900, n)
    sweep = np.sin(2 * np.pi * np.cumsum(freq / _RATE)).astype(np.float32)
    harm2 = np.sin(2 * np.pi * np.cumsum(freq * 1.5 / _RATE)).astype(np.float32) * 0.35
    env   = np.clip(t * 8, 0, 1) * np.exp(-3 * t)
    return _to_sound((sweep + harm2) * env * 0.55)


class SFXManager:
    """Loads (synthesizes) all SFX once at init and exposes a play() method."""

    def __init__(self):
        self._enabled = False
        self._sounds: dict[str, pygame.mixer.Sound] = {}
        try:
            self._sounds = {
                "sword_swing":       _sword_swing(),
                "arrow_shoot":       _arrow_shoot(),
                "hit_impact":        _hit_impact(),
                "enemy_death":       _enemy_death(),
                "fireball_shoot":    _fireball_shoot(),
                "mage_explosion":    _mage_explosion(),
                "iceball_shoot":     _iceball_shoot(),
                "freeze_hit":        _freeze_hit(),
                "slime_split":       _slime_split(),
                "creeper_fuse":      _creeper_fuse(),
                "creeper_explosion": _creeper_explosion(),
                "summon_portal":     _summon_portal(),
            }
            for s in self._sounds.values():
                s.set_volume(0.28)
            self._enabled = True
        except Exception:
            # Audio unavailable on this platform — silently disable
            pass

    def play(self, name: str):
        if not self._enabled:
            return
        sound = self._sounds.get(name)
        if sound:
            sound.play()
