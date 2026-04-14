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


class SFXManager:
    """Loads (synthesizes) all SFX once at init and exposes a play() method."""

    def __init__(self):
        self._enabled = False
        self._sounds: dict[str, pygame.mixer.Sound] = {}
        try:
            self._sounds = {
                "sword_swing": _sword_swing(),
                "arrow_shoot": _arrow_shoot(),
                "hit_impact":  _hit_impact(),
                "enemy_death": _enemy_death(),
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
