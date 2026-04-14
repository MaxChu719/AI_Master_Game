from __future__ import annotations
import collections
import numpy as np
import torch
import torch.nn.functional as F
from config import CFG

_M = CFG["arena"]["margin"]
ARENA_LEFT   = _M
ARENA_TOP    = _M
ARENA_RIGHT  = CFG["arena"]["width"]  - _M
ARENA_BOTTOM = CFG["arena"]["height"] - _M
ARENA_WIDTH  = ARENA_RIGHT  - ARENA_LEFT
ARENA_HEIGHT = ARENA_BOTTOM - ARENA_TOP

# Normalisation constants (vector obs)
_OBS = CFG["observation"]
MAX_ATTACK_DAMAGE = float(_OBS["max_attack_damage"])
MAX_ATTACK_RATE   = float(_OBS["max_attack_rate"])
MAX_ENEMY_VEL     = float(_OBS["max_enemy_vel"])
_N_ENEMIES        = int(_OBS["max_enemies_tracked"])   # 5

# Enemy type IDs
ENEMY_TYPE_SWARM  = 0
ENEMY_TYPE_SPIDER = 1
ENEMY_TYPE_BOSS   = 3
_N_ENEMY_TYPES    = 4

# Vector observation layout
_ENEMY_TOKEN_DIM = 6
_SELF_ALLY_DIM   = 11
OBS_DIM          = _SELF_ALLY_DIM + _N_ENEMIES * _ENEMY_TOKEN_DIM   # 41

# Image observation config
_IOBS = CFG["image_obs"]
FRAME_SIZE = int(_IOBS["frame_size"])   # 84
N_FRAMES   = int(_IOBS["n_frames"])     # 4

# Reward config
_RCFG = CFG["rewards"]

# Gray values for each entity type in the rendered obs frame [0, 1]
_GRAY_FIGHTER  = 0.40   # ally fighter
_GRAY_ARCHER   = 0.40   # ally archer
_GRAY_SELF     = 0.60   # the minion this env belongs to (brighter than allies)
_GRAY_SWARM    = 0.80   # swarm enemy
_GRAY_SPIDER   = 0.87   # spider enemy
_GRAY_BOSS     = 0.95   # boss
_GRAY_ARROW    = 0.55   # arrow projectile


class MinionEnv:
    def __init__(self, minion, enemies, ally=None, role: str = "fighter"):
        """
        minion  : the entity this env observes and rewards (Fighter or Archer)
        enemies : shared live enemies list (Swarm + Spider mixed)
        ally    : the other AI minion (opposite type)
        role    : "fighter" or "archer"
        """
        self.minion  = minion
        self.enemies = enemies
        self.ally    = ally
        self.role    = role
        # Boss reference — updated by BattleScene each frame
        self.boss = None
        # Frame counter for memory store interval
        self.frame_counter = 0
        # Accumulated reward across the store interval
        self._accumulated_reward = 0.0

        # Image observation state
        # obs_frame: numpy array (ARENA_HEIGHT, ARENA_WIDTH) float32 [0,1]
        # Set externally by BattleScene each tick (after combat).
        self.obs_frame: np.ndarray | None = None
        # Circular frame buffer of 4 resized (FRAME_SIZE×FRAME_SIZE) frames.
        self._frame_buffer: collections.deque = collections.deque(
            [np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32)] * N_FRAMES,
            maxlen=N_FRAMES,
        )

    # ------------------------------------------------------------------
    # Image Observation
    # ------------------------------------------------------------------

    def capture_frame(self, obs_frame: np.ndarray | None = None):
        """Append the current obs frame (or self.obs_frame) to the frame buffer.

        Call once per game tick AFTER combat/physics so next_obs reflects
        the post-action state.  obs_frame is (ARENA_HEIGHT, ARENA_WIDTH) float32.
        """
        src = obs_frame if obs_frame is not None else self.obs_frame
        if src is None:
            self._frame_buffer.append(np.zeros((FRAME_SIZE, FRAME_SIZE), dtype=np.float32))
            return
        self._frame_buffer.append(self._crop_and_resize(src))

    def get_observation(self) -> np.ndarray:
        """Return the 4-stacked image observation (N_FRAMES, FRAME_SIZE, FRAME_SIZE)."""
        return np.stack(list(self._frame_buffer), axis=0)

    def _crop_and_resize(self, frame: np.ndarray) -> np.ndarray:
        """Crop HxH centered on this minion (zero-padded), resize to 84x84."""
        H_arena, W_arena = frame.shape
        H = H_arena  # crop side = arena height (square)
        half = H // 2

        # Minion centre in arena-local coordinates
        cx = int(self.minion.pos.x - ARENA_LEFT)
        cy = int(self.minion.pos.y - ARENA_TOP)

        x0, x1 = cx - half, cx + half
        y0, y1 = cy - half, cy + half

        # Zero-padded crop
        crop = np.zeros((H, H), dtype=np.float32)
        sx0, sx1 = max(0, x0), min(W_arena, x1)
        sy0, sy1 = max(0, y0), min(H_arena, y1)
        if sx0 < sx1 and sy0 < sy1:
            dx0 = sx0 - x0
            dx1 = dx0 + (sx1 - sx0)
            dy0 = sy0 - y0
            dy1 = dy0 + (sy1 - sy0)
            crop[dy0:dy1, dx0:dx1] = frame[sy0:sy1, sx0:sx1]

        # Overwrite self position with brighter value so agent can locate itself
        r = self.minion.size // 2
        mx0 = max(0, half - r)
        mx1 = min(H, half + r + 1)
        crop[mx0:mx1, mx0:mx1] = _GRAY_SELF

        # Resize H×H → FRAME_SIZE×FRAME_SIZE using PyTorch bilinear
        t = torch.tensor(crop, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        t84 = F.interpolate(t, size=(FRAME_SIZE, FRAME_SIZE),
                            mode="bilinear", align_corners=False)
        return t84.squeeze().numpy()

    # ------------------------------------------------------------------
    # Vector Observation (used by preset policy during warmup)
    # ------------------------------------------------------------------

    def get_vector_observation(self) -> np.ndarray:
        """Return the legacy 41-dim vector observation for the preset heuristic."""
        m   = self.minion
        obs = np.zeros(OBS_DIM, dtype=np.float32)

        obs[0] = m.hp / m.max_hp
        obs[1] = (m.pos.x - ARENA_LEFT) / ARENA_WIDTH
        obs[2] = (m.pos.y - ARENA_TOP)  / ARENA_HEIGHT
        obs[3] = m.stamina / m.max_stamina
        obs[4] = m.attack_damage / MAX_ATTACK_DAMAGE
        obs[5] = min(1.0, (1.0 / m.attack_cooldown) / MAX_ATTACK_RATE)

        a = self.ally
        if a is not None and a.is_alive:
            obs[6]  = a.hp / a.max_hp
            obs[7]  = (a.pos.x - ARENA_LEFT) / ARENA_WIDTH
            obs[8]  = (a.pos.y - ARENA_TOP)  / ARENA_HEIGHT
            obs[9]  = a.stamina / a.max_stamina
            obs[10] = 1.0

        alive = [e for e in self.enemies if e.is_alive]
        if self.boss is not None and self.boss.is_alive:
            alive.append(self.boss)
        alive.sort(key=lambda e: m.pos.distance_to(e.pos))
        for i, e in enumerate(alive[:_N_ENEMIES]):
            base = _SELF_ALLY_DIM + i * _ENEMY_TOKEN_DIM
            etype = float(getattr(e, "enemy_type", ENEMY_TYPE_SWARM))
            obs[base]     = etype / _N_ENEMY_TYPES
            obs[base + 1] = (e.pos.x - m.pos.x) / ARENA_WIDTH
            obs[base + 2] = (e.pos.y - m.pos.y) / ARENA_HEIGHT
            obs[base + 3] = e.hp / e.max_hp
            obs[base + 4] = e.velocity.x / MAX_ENEMY_VEL
            obs[base + 5] = e.velocity.y / MAX_ENEMY_VEL

        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def get_reward(self, combat_events: dict) -> float:
        reward = 0.0

        if self.role == "fighter":
            reward += combat_events.get("sword_damage",    0.0) * _RCFG["fighter_damage_scale"]
            reward -= combat_events.get("damage_taken",    0.0) * _RCFG["fighter_damage_taken_scale"]
            reward += combat_events.get("sword_kills",     0)   * _RCFG["fighter_kill_bonus"]
            if combat_events.get("sword_miss", False):
                reward -= _RCFG["fighter_miss_penalty"]
        else:  # archer
            reward += combat_events.get("archer_damage_dealt", 0.0) * _RCFG["archer_damage_scale"]
            reward -= combat_events.get("archer_damage_taken", 0.0) * _RCFG["archer_damage_taken_scale"]
            reward += combat_events.get("archer_kills",        0)   * _RCFG["archer_kill_bonus"]
            if combat_events.get("archer_miss", False):
                reward -= _RCFG["archer_miss_penalty"]
            # Penalty for arrows that expired without hitting anything
            if combat_events.get("archer_arrow_expired", False):
                reward -= _RCFG["archer_arrow_expired_penalty"]

        if not self.minion.is_alive:
            reward -= _RCFG["death_penalty"]
        if combat_events.get("wave_cleared", False):
            reward += _RCFG["wave_cleared_bonus"]

        return reward

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------

    def is_done(self) -> bool:
        if not self.minion.is_alive:
            return True
        return not any(e.is_alive for e in self.enemies)
