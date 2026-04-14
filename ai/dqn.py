"""
Full Rainbow DQN agent.

Components:
  1. Double DQN   — policy net selects action, target net evaluates value.
  2. Dueling      — handled in BrainNetwork / CNNBrainNetwork.
  3. PER          — Prioritised Experience Replay via SumTree.
  4. N-step       — N-step TD returns accumulated before inserting into PER.
  5. Distributional (C51) — returns are distributions; cross-entropy loss.
  6. Noisy Networks — handled in BrainNetwork; replaces ε-greedy post-warmup.

Two training methods:
  train_step()                — Modern Rainbow DQN (used for Memory Replay training).
  train_step_expected_sarsa() — Expected SARSA (used for in-game online training).
"""
from __future__ import annotations
import math
import collections
import random
import numpy as np
import torch
import torch.nn as nn
from ai.brain import BrainNetwork, CNNBrainNetwork
from config import CFG

# ── Direction helpers ─────────────────────────────────────────────────────────

def _clean(v):
    """Round float tuple to 8 decimal places to avoid fp noise."""
    return tuple(round(x, 8) for x in v)


# 8 move directions (N, NE, E, SE, S, SW, W, NW) — used by both roles
_MOVE_VECS_8 = [
    _clean((math.cos(i * math.pi / 4), math.sin(i * math.pi / 4)))
    for i in range(8)
]

# 8 attack directions for Fighter (same 45° spacing as move dirs)
_FIGHTER_ATTACK_DIRS = _MOVE_VECS_8

# 16 attack directions for Archer (22.5° spacing — higher precision for aiming)
_ARCHER_ATTACK_DIRS = [
    _clean((math.cos(i * math.pi / 8), math.sin(i * math.pi / 8)))
    for i in range(16)
]

# Fighter: 16 actions total  (0–7 move, 8–15 attack)
FIGHTER_ACTION_TO_DIRECTION = list(_MOVE_VECS_8) + list(_FIGHTER_ATTACK_DIRS)
FIGHTER_ACTION_IS_ATTACK    = [False] * 8 + [True] * 8

# Archer: 24 actions total   (0–7 move, 8–23 attack in 16 directions)
ARCHER_ACTION_TO_DIRECTION  = list(_MOVE_VECS_8) + list(_ARCHER_ATTACK_DIRS)
ARCHER_ACTION_IS_ATTACK     = [False] * 8 + [True] * 16

# Backward-compat aliases (fighter tables)
ACTION_TO_DIRECTION = FIGHTER_ACTION_TO_DIRECTION
ACTION_IS_ATTACK    = FIGHTER_ACTION_IS_ATTACK


def _vec_to_dir_index(dx: float, dy: float, dirs: list) -> int:
    """Return the index in `dirs` whose unit vector best matches (dx, dy)."""
    best_idx, best_dot = 0, -float("inf")
    for i, (vx, vy) in enumerate(dirs):
        dot = dx * vx + dy * vy
        if dot > best_dot:
            best_dot = dot
            best_idx = i
    return best_idx


# ── Observation constants (for preset policy) ─────────────────────────────────

_OBS_CFG   = CFG["observation"]
_ARENA_W   = float(CFG["arena"]["width"]  - 2 * CFG["arena"]["margin"])
_ARENA_H   = float(CFG["arena"]["height"] - 2 * CFG["arena"]["margin"])
_MAX_VEL   = float(_OBS_CFG["max_enemy_vel"])
_N_ENEMIES = int(_OBS_CFG["max_enemies_tracked"])

_PP = CFG["preset_policy"]
_FIGHTER_IDEAL  = float(_PP["fighter_ideal_dist"])
_FIGHTER_MIN_SF = float(_PP["fighter_min_safe_dist"])
_ARCHER_PREF    = float(_PP["archer_preferred_dist"])
_ARCHER_MIN_SF  = float(_PP["archer_min_safe_dist"])
_ARCHER_SHOOT_R = float(_PP["archer_shoot_range"])
_ARROW_SPEED    = float(_PP["arrow_speed"])
_WALL_SAFE_DIST = float(_PP.get("wall_safe_dist", 80.0))


def _wall_repulsion(play_x: float, play_y: float) -> tuple:
    """Return (rx, ry) that pushes entity away from arena edges."""
    rx, ry = 0.0, 0.0
    d = _WALL_SAFE_DIST
    gap_l = play_x
    gap_r = _ARENA_W - play_x
    if gap_l < d: rx += (d - gap_l)
    if gap_r < d: rx -= (d - gap_r)
    gap_t = play_y
    gap_b = _ARENA_H - play_y
    if gap_t < d: ry += (d - gap_t)
    if gap_b < d: ry -= (d - gap_b)
    return rx, ry


# ── Preset policies ───────────────────────────────────────────────────────────

_ENEMY_TOKEN_DIM = 6   # (type_norm, dx, dy, hp, vx, vy)
_VEC_OBS_DIM = 11 + _N_ENEMIES * _ENEMY_TOKEN_DIM   # 41


def _fighter_preset_action(obs: np.ndarray) -> int:
    enemies_rel = []
    for i in range(_N_ENEMIES):
        base = 11 + i * _ENEMY_TOKEN_DIM
        dx = obs[base + 1] * _ARENA_W
        dy = obs[base + 2] * _ARENA_H
        if dx != 0.0 or dy != 0.0:
            enemies_rel.append((dx, dy))

    play_x = obs[1] * _ARENA_W
    play_y = obs[2] * _ARENA_H
    wall_rx, wall_ry = _wall_repulsion(play_x, play_y)

    if not enemies_rel:
        cx_rel = _ARENA_W / 2.0 - play_x
        cy_rel = _ARENA_H / 2.0 - play_y
        move_dx = cx_rel + wall_rx
        move_dy = cy_rel + wall_ry
        return _vec_to_dir_index(move_dx, move_dy, _MOVE_VECS_8)

    dx_n, dy_n = enemies_rel[0]
    dist_n = math.sqrt(dx_n ** 2 + dy_n ** 2)
    stamina_norm = float(obs[3])

    too_close = [(dx, dy) for dx, dy in enemies_rel
                 if math.sqrt(dx ** 2 + dy ** 2) < _FIGHTER_MIN_SF]
    if too_close:
        avg_dx = sum(dx for dx, _ in too_close) / len(too_close)
        avg_dy = sum(dy for _, dy in too_close) / len(too_close)
        flee_dx = -avg_dx + wall_rx
        flee_dy = -avg_dy + wall_ry
        return _vec_to_dir_index(flee_dx, flee_dy, _MOVE_VECS_8)

    if dist_n <= _FIGHTER_IDEAL and stamina_norm > 0.15:
        # Attack action: base 8 + direction index (8 fighter attack dirs)
        return 8 + _vec_to_dir_index(dx_n, dy_n, _FIGHTER_ATTACK_DIRS)

    if dist_n <= _FIGHTER_IDEAL:
        back_dx = -dx_n + wall_rx
        back_dy = -dy_n + wall_ry
        return _vec_to_dir_index(back_dx, back_dy, _MOVE_VECS_8)

    approach_dx = dx_n + wall_rx
    approach_dy = dy_n + wall_ry
    return _vec_to_dir_index(approach_dx, approach_dy, _MOVE_VECS_8)


def _archer_preset_action(obs: np.ndarray) -> int:
    enemies_rel = []
    for i in range(_N_ENEMIES):
        base = 11 + i * _ENEMY_TOKEN_DIM
        dx = obs[base + 1] * _ARENA_W
        dy = obs[base + 2] * _ARENA_H
        vx = obs[base + 4] * _MAX_VEL
        vy = obs[base + 5] * _MAX_VEL
        if dx != 0.0 or dy != 0.0:
            enemies_rel.append((dx, dy, vx, vy))

    play_x = obs[1] * _ARENA_W
    play_y = obs[2] * _ARENA_H
    wall_rx, wall_ry = _wall_repulsion(play_x, play_y)

    if not enemies_rel:
        cx_rel = _ARENA_W / 2.0 - play_x
        cy_rel = _ARENA_H / 2.0 - play_y
        move_dx = cx_rel + wall_rx
        move_dy = cy_rel + wall_ry
        return _vec_to_dir_index(move_dx, move_dy, _MOVE_VECS_8)

    dx_n, dy_n, vx_n, vy_n = enemies_rel[0]
    dist_n = math.sqrt(dx_n ** 2 + dy_n ** 2)
    stamina_norm = float(obs[3])

    panic = [(dx, dy) for dx, dy, _, __ in enemies_rel
             if math.sqrt(dx ** 2 + dy ** 2) < _ARCHER_MIN_SF]
    if panic:
        avg_dx = sum(dx for dx, _ in panic) / len(panic)
        avg_dy = sum(dy for _, dy in panic) / len(panic)
        flee_dx = -avg_dx + wall_rx
        flee_dy = -avg_dy + wall_ry
        return _vec_to_dir_index(flee_dx, flee_dy, _MOVE_VECS_8)

    if dist_n <= _ARCHER_SHOOT_R and stamina_norm > 0.25:
        # Velocity-lead intercept
        v_sq = vx_n ** 2 + vy_n ** 2
        a_coef = _ARROW_SPEED ** 2 - v_sq
        b_coef = -2.0 * (dx_n * vx_n + dy_n * vy_n)
        c_coef = -(dist_n ** 2)
        if a_coef > 0:
            disc = b_coef ** 2 - 4 * a_coef * c_coef
            travel_time = (-b_coef + math.sqrt(max(0.0, disc))) / (2 * a_coef)
        else:
            travel_time = dist_n / _ARROW_SPEED
        travel_time = min(travel_time, 2.0)
        lead_dx = dx_n + vx_n * travel_time
        lead_dy = dy_n + vy_n * travel_time
        # Attack action: base 8 + direction index (16 archer attack dirs)
        return 8 + _vec_to_dir_index(lead_dx, lead_dy, _ARCHER_ATTACK_DIRS)

    if dist_n > _ARCHER_SHOOT_R:
        approach_dx = dx_n + wall_rx
        approach_dy = dy_n + wall_ry
        return _vec_to_dir_index(approach_dx, approach_dy, _MOVE_VECS_8)

    if dist_n < _ARCHER_PREF:
        back_dx = -dx_n + wall_rx
        back_dy = -dy_n + wall_ry
        return _vec_to_dir_index(back_dx, back_dy, _MOVE_VECS_8)

    return _vec_to_dir_index(dx_n + wall_rx, dy_n + wall_ry, _MOVE_VECS_8)


# ── SumTree for Prioritised Experience Replay ─────────────────────────────────

class SumTree:
    """Binary SumTree: O(log n) sampling, O(log n) update."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data     = [None] * capacity
        self.write    = 0
        self.size     = 0

    def _propagate(self, idx: int, delta: float):
        parent = (idx - 1) // 2
        self.tree[parent] += delta
        if parent != 0:
            self._propagate(parent, delta)

    def _retrieve(self, idx: int, s: float) -> int:
        left  = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        return self._retrieve(right, s - self.tree[left])

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def add(self, priority: float, data):
        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.size  = min(self.size + 1, self.capacity)

    def update(self, idx: int, priority: float):
        delta = priority - self.tree[idx]
        self.tree[idx] = priority
        self._propagate(idx, delta)

    def get(self, s: float):
        """Return (tree_idx, priority, data)."""
        idx      = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return idx, float(self.tree[idx]), self.data[data_idx]


# ── Rainbow DQN Agent ─────────────────────────────────────────────────────────

_DCFG = CFG["dqn"]
_RCFG = CFG["rainbow"]
_SARSA_TEMP = float(CFG.get("expected_sarsa", {}).get("softmax_temperature", 1.0))


class DQNAgent:
    """
    Shared Rainbow DQN agent supporting both vector and image observations.

    obs_type="image"  → uses CNNBrainNetwork, obs shape (N_FRAMES, H, W)
    obs_type="vector" → uses BrainNetwork,    obs shape (OBS_DIM,)

    Training methods:
      train_step()                → Modern Rainbow DQN (Memory Replay)
      train_step_expected_sarsa() → Expected SARSA    (in-game online training)
    """

    def __init__(
        self,
        obs_dim:     int   = _VEC_OBS_DIM,
        action_dim:  int   = 16,
        lr:          float = _DCFG["learning_rate"],
        role:        str   = "fighter",
        buffer_size: int   = _DCFG["replay_buffer_size"],
        obs_type:    str   = "image",   # "image" or "vector"
    ):
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.role       = role
        self.obs_type   = obs_type

        # Distributional support
        self.n_atoms = int(_RCFG["n_atoms"])
        self.v_min   = float(_RCFG["v_min"])
        self.v_max   = float(_RCFG["v_max"])
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        # N-step
        self.n_step      = int(_RCFG["n_step"])
        self.gamma       = float(_DCFG["gamma"])
        self.gamma_n     = self.gamma ** self.n_step
        self._n_buf      = collections.deque(maxlen=self.n_step)

        # PER params
        self.per_alpha        = float(_RCFG["per_alpha"])
        self.per_beta         = float(_RCFG["per_beta_start"])
        self.per_beta_end     = float(_RCFG["per_beta_end"])
        self.per_beta_steps   = int(_RCFG["per_beta_steps"])
        self.per_beta_inc     = (self.per_beta_end - self.per_beta) / self.per_beta_steps
        self.per_epsilon      = float(_RCFG["per_epsilon"])
        self.grad_clip        = float(_RCFG["grad_clip"])

        # Replay buffer
        self.buffer_size = buffer_size
        self.tree        = SumTree(buffer_size)

        # Training params
        self.batch_size          = int(_DCFG["batch_size"])
        self.min_buffer_size     = int(_DCFG["min_buffer_size"])
        self.soft_update_tau     = float(_DCFG.get("soft_update_tau", 0.005))
        self.warmup_preset_ratio = float(_DCFG.get("warmup_preset_ratio", 0.8))

        # Expected SARSA temperature (for softmax policy in target computation)
        self.sarsa_temperature   = _SARSA_TEMP

        # Build networks (CNN for image obs, Transformer for vector obs)
        if obs_type == "image":
            self.policy_net = CNNBrainNetwork(action_dim, self.n_atoms)
            self.target_net = CNNBrainNetwork(action_dim, self.n_atoms)
        else:
            self.policy_net = BrainNetwork(action_dim, self.n_atoms)
            self.target_net = BrainNetwork(action_dim, self.n_atoms)

        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer    = torch.optim.Adam(self.policy_net.parameters(), lr=lr)
        self.lr           = lr
        self.step_counter = 0

        # Preset-only mode: always use heuristic, still collect transitions
        self.preset_only = False

    # ── Buffer helpers ────────────────────────────────────────────────────────

    def _max_priority(self) -> float:
        if self.tree.size == 0:
            return 1.0
        return float(
            self.tree.tree[self.tree.capacity - 1:
                           self.tree.capacity - 1 + self.tree.size].max()
        ) or 1.0

    def _add_to_per(self, transition):
        p = self._max_priority() ** self.per_alpha
        self.tree.add(p, transition)

    def _build_n_step_transition(self):
        s0, a0, _, _, _ = self._n_buf[0]
        n_reward = 0.0
        n_done   = False
        last_ns  = None
        for i, (_, _, r, ns, d) in enumerate(self._n_buf):
            n_reward += (self.gamma ** i) * r
            last_ns   = ns
            if d:
                n_done = True
                break
        return (s0, a0, n_reward, last_ns, n_done)

    # ── Public API ────────────────────────────────────────────────────────────

    def preset_action(self, obs: np.ndarray) -> int:
        """Role-specific heuristic action from the 41-dim vector observation."""
        if self.role == "archer":
            return _archer_preset_action(obs)
        return _fighter_preset_action(obs)

    def apply_training_settings(self, settings: dict):
        """Apply a dict of training hyperparameters to this agent at runtime."""
        if "preset_only" in settings:
            self.preset_only = bool(settings["preset_only"])
        if "lr" in settings:
            self.lr = float(settings["lr"])
            for g in self.optimizer.param_groups:
                g["lr"] = self.lr
        if "warmup_preset_ratio" in settings:
            self.warmup_preset_ratio = float(settings["warmup_preset_ratio"])
        if "min_buffer_size" in settings:
            self.min_buffer_size = int(settings["min_buffer_size"])
        if "soft_update_tau" in settings:
            self.soft_update_tau = float(settings["soft_update_tau"])
        if "batch_size" in settings:
            self.batch_size = int(settings["batch_size"])

    def select_action(self, obs: np.ndarray, preset_obs: np.ndarray | None = None) -> int:
        """
        Select an action.

        obs        : network input (image: (N,H,W) or vector: (D,))
        preset_obs : 41-dim vector obs for the heuristic preset policy.
                     If None, falls back to obs (legacy vector-obs path).

        Warmup/preset-only modes use the preset heuristic (via preset_obs).
        After warmup: greedy argmax over expected Q from the noisy policy net.
        """
        _vec = preset_obs if preset_obs is not None else obs

        if self.preset_only:
            return self.preset_action(_vec)

        if self.tree.size < self.min_buffer_size:
            if random.random() < self.warmup_preset_ratio:
                return self.preset_action(_vec)

        self.policy_net.eval()
        with torch.no_grad():
            obs_t    = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            log_dist = self.policy_net(obs_t)                           # [1, A, N]
            dist     = log_dist.exp()
            q_vals   = (dist * self.support.to(obs_t.device)).sum(-1)  # [1, A]
            return int(q_vals.argmax(dim=1).item())

    def store_transition(self, obs, action, reward, next_obs, done):
        """Store a transition into the N-step buffer → PER."""
        self._n_buf.append((obs, action, reward, next_obs, done))
        if len(self._n_buf) == self.n_step:
            transition = self._build_n_step_transition()
            self._add_to_per(transition)
        if done:
            for k in range(1, len(self._n_buf)):
                sub = list(self._n_buf)[k:]
                s0, a0, _, _, _ = sub[0]
                nr, nd, lns = 0.0, False, None
                for i, (_, _, r, ns, d) in enumerate(sub):
                    nr  += (self.gamma ** i) * r
                    lns  = ns
                    if d:
                        nd = True
                        break
                self._add_to_per((s0, a0, nr, lns, nd))
            self._n_buf.clear()

    # ── Modern Rainbow DQN (Memory Replay training) ───────────────────────────

    def train_step(self) -> dict:
        """
        One step of full Rainbow DQN:
        Double DQN action selection + C51 Bellman projection + PER IS weights.
        Used for off-policy Memory Replay training from the ResearchLab.
        """
        if self.tree.size < self.min_buffer_size:
            return {"loss": 0.0, "steps": self.step_counter}

        batch, tree_idxs, is_weights = self._sample_per()
        if not batch or any(b is None for b in batch):
            return {"loss": 0.0, "steps": self.step_counter}

        obs_b, act_b, rew_b, next_b, done_b = zip(*batch)
        obs_t      = torch.tensor(np.array(obs_b),  dtype=torch.float32)
        act_t      = torch.tensor(act_b,            dtype=torch.long)
        rew_t      = torch.tensor(rew_b,            dtype=torch.float32)
        next_obs_t = torch.tensor(np.array(next_b), dtype=torch.float32)
        done_t     = torch.tensor(done_b,           dtype=torch.float32)

        with torch.no_grad():
            # Double DQN: policy net selects actions
            self.policy_net.eval()
            self.policy_net.reset_noise()
            next_log_p = self.policy_net(next_obs_t)               # [B, A, N]
            next_q     = (next_log_p.exp() * self.support).sum(-1) # [B, A]
            next_acts  = next_q.argmax(dim=1)                      # [B]

            # Target net evaluates
            self.target_net.reset_noise()
            next_dist_log = self.target_net(next_obs_t)            # [B, A, N]
            next_dist = next_dist_log.exp()[range(self.batch_size), next_acts]  # [B, N]

            target_dist = self._project_distribution(next_dist, rew_t, done_t)

        self.policy_net.train()
        self.policy_net.reset_noise()
        log_dist = self.policy_net(obs_t)                          # [B, A, N]
        log_p    = log_dist[range(self.batch_size), act_t]         # [B, N]

        element_loss  = -(target_dist * log_p).sum(dim=-1)         # [B]
        weighted_loss = (is_weights.to(element_loss.device) * element_loss).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        td_errors = element_loss.detach().abs().numpy() + self.per_epsilon
        for idx, pri in zip(tree_idxs, td_errors):
            self.tree.update(idx, float(pri ** self.per_alpha))

        self.per_beta = min(self.per_beta_end, self.per_beta + self.per_beta_inc)
        self.step_counter += 1
        tau = self.soft_update_tau
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(),
                              self.policy_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(pp.data, alpha=tau)

        return {
            "loss":       weighted_loss.item(),
            "avg_reward": float(rew_t.mean().item()),
            "epsilon":    0.0,
            "steps":      self.step_counter,
        }

    # ── Expected SARSA (in-game online training) ──────────────────────────────

    def train_step_expected_sarsa(self) -> dict:
        """
        One step of Expected SARSA with distributional RL.

        Instead of taking max_a Q(s', a) (Double DQN), computes the expected
        distribution under a softmax policy over the target-net Q values:

          π(a | s') = softmax(Q_target(s') / temperature)
          Z_expected(s') = Σ_a π(a | s') · Z_target(s', a)

        This mixture distribution is then projected via the standard C51
        Bellman operator and used as the training target.

        Used for online in-game training (TrainingSystem in battle.py).
        Memory Replay training continues to use train_step() (Rainbow DQN).
        """
        if self.tree.size < self.min_buffer_size:
            return {"loss": 0.0, "steps": self.step_counter}

        batch, tree_idxs, is_weights = self._sample_per()
        if not batch or any(b is None for b in batch):
            return {"loss": 0.0, "steps": self.step_counter}

        obs_b, act_b, rew_b, next_b, done_b = zip(*batch)
        obs_t      = torch.tensor(np.array(obs_b),  dtype=torch.float32)
        act_t      = torch.tensor(act_b,            dtype=torch.long)
        rew_t      = torch.tensor(rew_b,            dtype=torch.float32)
        next_obs_t = torch.tensor(np.array(next_b), dtype=torch.float32)
        done_t     = torch.tensor(done_b,           dtype=torch.float32)

        with torch.no_grad():
            # Expected SARSA: compute policy π from target net Q values
            self.target_net.reset_noise()
            next_dist_log = self.target_net(next_obs_t)             # [B, A, N]
            next_dist     = next_dist_log.exp()                     # [B, A, N]

            # Q values for softmax policy
            next_q = (next_dist * self.support).sum(-1)             # [B, A]
            pi     = torch.softmax(next_q / self.sarsa_temperature, dim=-1)  # [B, A]

            # Expected next-state distribution: weighted mixture over actions
            # next_dist_exp[b, n] = Σ_a  π[b, a] · next_dist[b, a, n]
            next_dist_exp = (pi.unsqueeze(-1) * next_dist).sum(1)  # [B, N]

            target_dist = self._project_distribution(next_dist_exp, rew_t, done_t)

        self.policy_net.train()
        self.policy_net.reset_noise()
        log_dist = self.policy_net(obs_t)                           # [B, A, N]
        log_p    = log_dist[range(self.batch_size), act_t]          # [B, N]

        element_loss  = -(target_dist * log_p).sum(dim=-1)          # [B]
        weighted_loss = (is_weights.to(element_loss.device) * element_loss).mean()

        self.optimizer.zero_grad()
        weighted_loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.grad_clip)
        self.optimizer.step()

        td_errors = element_loss.detach().abs().numpy() + self.per_epsilon
        for idx, pri in zip(tree_idxs, td_errors):
            self.tree.update(idx, float(pri ** self.per_alpha))

        self.per_beta = min(self.per_beta_end, self.per_beta + self.per_beta_inc)
        self.step_counter += 1
        tau = self.soft_update_tau
        with torch.no_grad():
            for tp, pp in zip(self.target_net.parameters(),
                              self.policy_net.parameters()):
                tp.data.mul_(1.0 - tau).add_(pp.data, alpha=tau)

        return {
            "loss":       weighted_loss.item(),
            "avg_reward": float(rew_t.mean().item()),
            "steps":      self.step_counter,
        }

    # ── PER sampling ─────────────────────────────────────────────────────────

    def _sample_per(self):
        batch, idxs, pris = [], [], []
        segment = self.tree.total / self.batch_size
        for i in range(self.batch_size):
            s = random.uniform(segment * i, segment * (i + 1))
            idx, pri, data = self.tree.get(s)
            idxs.append(idx)
            pris.append(max(pri, self.per_epsilon))
            batch.append(data)
        sampling_probs = np.array(pris) / self.tree.total
        is_weights = (self.tree.size * sampling_probs) ** (-self.per_beta)
        is_weights = torch.tensor(is_weights / is_weights.max(), dtype=torch.float32)
        return batch, idxs, is_weights

    # ── C51 projection ────────────────────────────────────────────────────────

    def _project_distribution(
        self,
        next_dist: torch.Tensor,
        rewards:   torch.Tensor,
        dones:     torch.Tensor,
    ) -> torch.Tensor:
        """Project the Bellman-updated distribution onto the support atoms."""
        B   = rewards.shape[0]
        sup = self.support

        Tz = rewards.unsqueeze(1) + (1.0 - dones.unsqueeze(1)) * self.gamma_n * sup.unsqueeze(0)
        Tz = Tz.clamp(self.v_min, self.v_max)

        b  = (Tz - self.v_min) / self.delta_z
        lo = b.floor().long().clamp(0, self.n_atoms - 1)
        hi = b.ceil().long().clamp(0, self.n_atoms - 1)

        lo_coef = hi.float() - b
        hi_coef = b - lo.float()
        eq_mask = (lo == hi).float()
        lo_coef = lo_coef + eq_mask

        target = torch.zeros(B, self.n_atoms)
        offset = torch.arange(B).unsqueeze(1) * self.n_atoms
        target.view(-1).index_add_(0, (lo + offset).view(-1),
                                   (next_dist * lo_coef).view(-1))
        target.view(-1).index_add_(0, (hi + offset).view(-1),
                                   (next_dist * hi_coef).view(-1))
        return target

    # ── Checkpoint ───────────────────────────────────────────────────────────

    def reset_brain(self):
        if self.obs_type == "image":
            self.policy_net = CNNBrainNetwork(self.action_dim, self.n_atoms)
            self.target_net = CNNBrainNetwork(self.action_dim, self.n_atoms)
        else:
            self.policy_net = BrainNetwork(self.action_dim, self.n_atoms)
            self.target_net = BrainNetwork(self.action_dim, self.n_atoms)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer    = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.tree         = SumTree(self.buffer_size)
        self._n_buf.clear()
        self.per_beta     = float(_RCFG["per_beta_start"])
        self.step_counter = 0

    def resize_buffer(self, new_size: int):
        """Resize the replay buffer, preserving existing transitions up to new_size."""
        if new_size == self.buffer_size:
            return
        old_tree = self.tree
        old_size = old_tree.size
        new_tree = SumTree(new_size)
        if old_size > 0:
            capacity = old_tree.capacity
            start = old_tree.write if old_size == capacity else 0
            count = min(old_size, new_size)
            for i in range(count):
                slot     = (start + (old_size - count) + i) % capacity
                priority = float(old_tree.tree[slot + capacity - 1])
                data     = old_tree.data[slot]
                if data is not None:
                    new_tree.add(priority if priority > 0 else self._max_priority(), data)
        self.buffer_size = new_size
        self.tree        = new_tree

    def save_checkpoint(self, path: str):
        """Save model weights, optimizer state, and training counters (no buffer)."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy_net":   self.policy_net.state_dict(),
            "target_net":   self.target_net.state_dict(),
            "optimizer":    self.optimizer.state_dict(),
            "step_counter": self.step_counter,
            "per_beta":     self.per_beta,
            "buffer_size":  self.buffer_size,
        }, path)

    def save_buffer(self, path: str):
        """Save the replay buffer (PER priorities + transitions) to a separate file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        capacity    = self.tree.capacity
        start       = self.tree.write if self.tree.size == capacity else 0
        buf_entries = []
        for i in range(self.tree.size):
            slot     = (start + i) % capacity
            priority = float(self.tree.tree[slot + capacity - 1])
            data     = self.tree.data[slot]
            if data is not None:
                buf_entries.append((priority, data))
        torch.save({"buffer": buf_entries}, path)

    def load_checkpoint(self, path: str):
        """Load model weights and training state. Buffer loaded separately via load_buffer."""
        import os
        if not os.path.isfile(path):
            return
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            self.policy_net.load_state_dict(ck["policy_net"])
            self.target_net.load_state_dict(ck["target_net"])
            self.optimizer.load_state_dict(ck["optimizer"])
            self.step_counter = int(ck.get("step_counter", 0))
            self.per_beta     = float(ck.get("per_beta", self.per_beta))
            self.target_net.eval()
            # Backward compat: load buffer if embedded in old checkpoint format
            buf_entries = ck.get("buffer", [])
            if buf_entries:
                for priority, data in buf_entries:
                    self.tree.add(priority if priority > 0 else 1.0, data)
        except Exception:
            pass  # Architecture changed; start fresh

    def load_buffer(self, path: str):
        """Load replay buffer from a separate buffer file."""
        import os
        if not os.path.isfile(path):
            return
        try:
            ck = torch.load(path, map_location="cpu", weights_only=False)
            for priority, data in ck.get("buffer", []):
                self.tree.add(priority if priority > 0 else 1.0, data)
        except Exception:
            pass
