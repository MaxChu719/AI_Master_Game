"""
Rainbow DQN Brains: two network architectures share the same Noisy Dueling C51 head.

BrainNetwork (vector input):
  Input [B, 41]   (11 self+ally dims + 5 enemy tokens × 6 dims each)
    → CLS token + 5 enemy tokens → TransformerEncoder(layers=4, heads=4) → CLS out [B, d_model]
    → Value stream:     NoisyLinear(d_model→64) → ReLU → NoisyLinear(64→n_atoms)
    → Advantage stream: NoisyLinear(d_model→64) → ReLU → NoisyLinear(64→A*n_atoms)
    → Q = V + (A - mean(A))  [B, A, n_atoms]  → log_softmax

CNNBrainNetwork (image input):
  Input [B, 4, 84, 84]  (4 stacked grayscale frames)
    → Conv(4→ch[0], k, stride=2) + 3 DepthwiseSeparableConv layers → GlobalAvgPool → [B, ch[-1]]
    → same Noisy Dueling C51 head as BrainNetwork

Both return LOG-probabilities: [B, action_dim, n_atoms].
Layers and channel counts for CNNBrainNetwork are configurable in config.json → cnn.

Components:
  - NoisyLinear: factorized Gaussian noise for exploration (Noisy Networks).
  - Dueling architecture: separate value and advantage streams.
  - Distributional: C51-style atom-based return distributions.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import CFG

_TC = CFG["transformer"]
_RC = CFG["rainbow"]
_CC = CFG["cnn"]
_IO = CFG["image_obs"]

D_MODEL   = int(_TC["d_model"])
NHEAD     = int(_TC["nhead"])
N_LAYERS  = int(_TC["num_layers"])
DIM_FF    = int(_TC["dim_feedforward"])
N_ATOMS   = int(_RC["n_atoms"])
NOISY_SIG = float(_RC["noisy_sigma"])

_SELF_ALLY_DIM = 11
_ENEMY_DIM     = 6   # (type_norm, dx, dy, hp, vx, vy)
_MAX_ENEMIES   = CFG["observation"]["max_enemies_tracked"]


# ── NoisyLinear (factorized noise) ─────────────────────────────────────────────

class NoisyLinear(nn.Module):
    """Linear layer with factorized Gaussian noise (Fortunato et al. 2017)."""

    def __init__(self, in_features: int, out_features: int, sigma: float = NOISY_SIG):
        super().__init__()
        self.in_features  = in_features
        self.out_features = out_features
        self.sigma        = sigma

        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_eps", torch.empty(out_features, in_features))
        self.register_buffer("bias_eps",   torch.empty(out_features))

        self._init_params()
        self.reset_noise()

    def _init_params(self):
        mu_range = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.sigma / math.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.sigma / math.sqrt(self.out_features))

    @staticmethod
    def _scaled_noise(size: int) -> torch.Tensor:
        x = torch.randn(size)
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        eps_p = self._scaled_noise(self.in_features)
        eps_q = self._scaled_noise(self.out_features)
        self.weight_eps.copy_(eps_q.outer(eps_p))
        self.bias_eps.copy_(eps_q)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            w = self.weight_mu + self.weight_sigma * self.weight_eps
            b = self.bias_mu   + self.bias_sigma   * self.bias_eps
        else:
            w = self.weight_mu
            b = self.bias_mu
        return F.linear(x, w, b)


# ── Rainbow Brain ──────────────────────────────────────────────────────────────

class BrainNetwork(nn.Module):
    """
    Rainbow DQN brain (Noisy + Dueling + Distributional).

    forward() returns LOG-probabilities: [B, action_dim, n_atoms].
    Call reset_noise() before each forward pass during training to re-sample noise.
    """

    def __init__(
        self,
        action_dim:      int = 16,
        n_atoms:         int = N_ATOMS,
        d_model:         int = D_MODEL,
        nhead:           int = NHEAD,
        num_layers:      int = N_LAYERS,
        dim_feedforward: int = DIM_FF,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms    = n_atoms

        # Token projections (same as original BrainNetwork)
        self.self_proj  = nn.Linear(_SELF_ALLY_DIM, d_model)
        self.enemy_proj = nn.Linear(_ENEMY_DIM,     d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=0.0,
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers, enable_nested_tensor=False)

        hidden = 64

        # Value stream  [B, d_model] → [B, n_atoms]
        self.val1 = NoisyLinear(d_model, hidden)
        self.val2 = NoisyLinear(hidden,  n_atoms)

        # Advantage stream  [B, d_model] → [B, action_dim * n_atoms]
        self.adv1 = NoisyLinear(d_model, hidden)
        self.adv2 = NoisyLinear(hidden,  action_dim * n_atoms)

    # ── Noise ─────────────────────────────────────────────────────────────────

    def reset_noise(self):
        """Re-sample factorized noise in all NoisyLinear layers."""
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs : [B, 11 + N*5]
        Returns: [B, action_dim, n_atoms]  log-probabilities
        """
        B = obs.shape[0]

        self_ally  = obs[:, :_SELF_ALLY_DIM]
        enemy_flat = obs[:, _SELF_ALLY_DIM:]
        n_enemies  = enemy_flat.shape[1] // _ENEMY_DIM

        cls_tok  = self.self_proj(self_ally).unsqueeze(1)               # [B,1,d]
        e_chunks = enemy_flat.view(B, n_enemies, _ENEMY_DIM)            # [B,N,5]
        e_tokens = self.enemy_proj(e_chunks)                            # [B,N,d]

        # Mask all-zero enemy slots
        padded = (e_chunks.abs().sum(-1) == 0)                         # [B, N]
        cls_mask = torch.zeros(B, 1, dtype=torch.bool, device=obs.device)
        key_mask = torch.cat([cls_mask, padded], dim=1)                # [B, 1+N]

        seq = torch.cat([cls_tok, e_tokens], dim=1)                    # [B,1+N,d]
        enc = self.transformer(seq, src_key_padding_mask=key_mask)
        feat = enc[:, 0]                                               # [B, d_model]

        # Value stream
        v = F.relu(self.val1(feat))
        v = self.val2(v).unsqueeze(1)                                  # [B, 1, n_atoms]

        # Advantage stream
        a = F.relu(self.adv1(feat))
        a = self.adv2(a).view(B, self.action_dim, self.n_atoms)        # [B, A, n_atoms]
        a = a - a.mean(dim=1, keepdim=True)                            # centred advantages

        q = v + a                                                      # [B, A, n_atoms]
        return F.log_softmax(q, dim=-1)                                # log-probabilities


# ── CNN Brain (image input) ────────────────────────────────────────────────────

class _DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable conv block (MobileNet-style): DW → PW → ReLU."""

    def __init__(self, in_c: int, out_c: int, kernel: int = 3, stride: int = 2):
        super().__init__()
        padding = kernel // 2
        self.dw = nn.Conv2d(in_c, in_c, kernel, stride=stride,
                            padding=padding, groups=in_c, bias=True)
        self.pw = nn.Conv2d(in_c, out_c, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.pw(self.dw(x)))


class CNNBrainNetwork(nn.Module):
    """
    CNN brain for image-based observations.

    Architecture (default 4 layers, configurable via config.json → cnn):
      Input [B, 4, 84, 84]  (4 stacked grayscale frames)
        → Conv(4 → ch[0], k, stride=2)                    → ReLU  [B, ch[0], 42, 42]
        → DepthwiseSeparableConv(ch[0] → ch[1], stride=2)         [B, ch[1], 21, 21]
        → DepthwiseSeparableConv(ch[1] → ch[2], stride=2)         [B, ch[2], 10, 10]
        → DepthwiseSeparableConv(ch[2] → ch[3], stride=2)         [B, ch[3],  5,  5]
        → GlobalAvgPool                                            [B, ch[3]]
        → Noisy Dueling C51 head (same as BrainNetwork)
          → [B, action_dim, n_atoms]  log-probabilities

    forward() returns LOG-probabilities: [B, action_dim, n_atoms].
    Call reset_noise() before each forward pass during training.
    """

    def __init__(
        self,
        action_dim: int = 16,
        n_atoms:    int = N_ATOMS,
        channels:   list | None = None,
        kernel_size: int | None = None,
        n_frames:   int | None = None,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.n_atoms    = n_atoms

        if channels is None:
            channels = list(_CC["channels"])
        if kernel_size is None:
            kernel_size = int(_CC["kernel_size"])
        if n_frames is None:
            n_frames = int(_IO["n_frames"])

        assert len(channels) >= 1, "cnn.channels must have at least 1 entry"

        # Build backbone: first layer is a standard conv, rest are DS-convs
        layers: list[nn.Module] = [
            nn.Conv2d(n_frames, channels[0], kernel_size=kernel_size,
                      stride=2, padding=kernel_size // 2, bias=True),
            nn.ReLU(),
        ]
        for i in range(len(channels) - 1):
            layers.append(
                _DepthwiseSeparableConv(channels[i], channels[i + 1],
                                        kernel=kernel_size, stride=2)
            )

        self.backbone = nn.Sequential(*layers)
        self.gap      = nn.AdaptiveAvgPool2d(1)          # global average pool

        feat_dim = channels[-1]

        # Noisy Dueling head (same pattern as BrainNetwork)
        self.val1 = NoisyLinear(feat_dim, 64)
        self.val2 = NoisyLinear(64, n_atoms)

        self.adv1 = NoisyLinear(feat_dim, 64)
        self.adv2 = NoisyLinear(64, action_dim * n_atoms)

    def reset_noise(self):
        for m in self.modules():
            if isinstance(m, NoisyLinear):
                m.reset_noise()

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs : [B, n_frames, FRAME_SIZE, FRAME_SIZE]
        Returns: [B, action_dim, n_atoms]  log-probabilities
        """
        B    = obs.shape[0]
        feat = self.gap(self.backbone(obs)).view(B, -1)    # [B, feat_dim]

        # Value stream
        v = F.relu(self.val1(feat))
        v = self.val2(v).unsqueeze(1)                      # [B, 1, n_atoms]

        # Advantage stream
        a = F.relu(self.adv1(feat))
        a = self.adv2(a).view(B, self.action_dim, self.n_atoms)  # [B, A, n_atoms]
        a = a - a.mean(dim=1, keepdim=True)

        q = v + a                                          # [B, A, n_atoms]
        return F.log_softmax(q, dim=-1)
