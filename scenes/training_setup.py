from __future__ import annotations
"""
TrainingSetupScene — shown between ResearchLab and BattleScene.

Lets the player choose:
  • Mode: DQN Training  vs  Preset+Train (preset actions, DQN trains in background)
  • Learning Rate
  • Warmup Preset Ratio  (fraction of buffer-fill phase using preset actions)
  • Min Buffer before Training  (transitions needed before gradient updates start)
  • Soft Update Tau (Polyak averaging rate for target network)
  • Batch Size

Changes apply immediately to both fighter_agent and archer_agent before battle starts.
"""
import pygame
from engine.scene import BaseScene
from config import CFG


# ── Parameter definitions ─────────────────────────────────────────────────────
# Each entry: (display_label, settings_key, options, default_index, format_fn)

def _fmt_mode(v):    return "Preset+Train" if v else "DQN Training"
def _fmt_lr(v):      return f"{v:.4f}"
def _fmt_pct(v):     return f"{int(v * 100)}%"
def _fmt_int(v):     return str(int(v))


_PARAMS = [
    # label,                  key,                    options,                              default_idx, fmt_fn
    ("Mode",                  "preset_only",          [False, True],                        0,           _fmt_mode),
    ("Learning Rate",         "lr",                   [0.0001, 0.0003, 0.0005, 0.001, 0.003], 2,        _fmt_lr),
    ("Warmup Preset Ratio",   "warmup_preset_ratio",  [0.0, 0.3, 0.5, 0.7, 0.8, 0.9, 1.0], 4,          _fmt_pct),
    ("Min Buffer (warmup)",   "min_buffer_size",      [50, 100, 200, 500, 1000],            2,           _fmt_int),
    ("Soft Update Tau",        "soft_update_tau",      [0.001, 0.003, 0.005, 0.01, 0.05],   2,           _fmt_lr),
    ("Batch Size",            "batch_size",           [32, 64, 128],                        1,           _fmt_int),
]

# Short description shown next to each param
_HINTS = [
    "Always act via preset heuristics, but DQN still collects experience and trains in background",
    "Adam optimiser learning rate for both agents",
    "How often to use preset heuristics while filling the replay buffer",
    "Replay buffer must hold at least this many transitions before training starts",
    "Polyak averaging rate for target network (θ_target ← τ·θ + (1-τ)·θ_target, per step)",
    "Transitions sampled per gradient update",
]


class TrainingSetupScene(BaseScene):
    def __init__(self, game_manager):
        super().__init__(game_manager)

        # Current option indices for each param
        self._indices = [p[3] for p in _PARAMS]
        self._selected = 0  # highlighted row

        # Fonts
        self._f_title  = pygame.font.SysFont("arial", 40, bold=True)
        self._f_sub    = pygame.font.SysFont("arial", 18)
        self._f_label  = pygame.font.SysFont("arial", 22, bold=True)
        self._f_val    = pygame.font.SysFont("arial", 22)
        self._f_hint   = pygame.font.SysFont("arial", 15)
        self._f_btn    = pygame.font.SysFont("arial", 26, bold=True)

        self._click_rects: list = []   # [(rect, action)]

    # ── Accessors ─────────────────────────────────────────────────────────────

    def _current_value(self, row: int):
        _, _, options, _, _ = _PARAMS[row]
        return options[self._indices[row]]

    def _step(self, row: int, delta: int):
        _, _, options, _, _ = _PARAMS[row]
        self._indices[row] = (self._indices[row] + delta) % len(options)

    # ── Events ────────────────────────────────────────────────────────────────

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            self._process_key(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._process_click(event.pos)

    def _process_key(self, key: int):
        if key == pygame.K_ESCAPE:
            self.game_manager.pop_scene()
        elif key == pygame.K_UP:
            self._selected = (self._selected - 1) % len(_PARAMS)
        elif key == pygame.K_DOWN:
            self._selected = (self._selected + 1) % len(_PARAMS)
        elif key in (pygame.K_LEFT, pygame.K_MINUS, pygame.K_KP_MINUS):
            self._step(self._selected, -1)
        elif key in (pygame.K_RIGHT, pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
            self._step(self._selected, +1)
        elif key in (pygame.K_RETURN, pygame.K_b):
            self._start_battle()

    def _process_click(self, pos):
        for rect, action in self._click_rects:
            if rect.collidepoint(pos):
                if action == "start":
                    self._start_battle()
                elif action == "back":
                    self.game_manager.pop_scene()
                elif isinstance(action, tuple):
                    row, delta = action
                    self._selected = row
                    self._step(row, delta)
                break

    def _start_battle(self):
        """Apply current settings to both agents, then push BattleScene."""
        fa = self.game_manager.fighter_agent
        aa = self.game_manager.archer_agent
        if fa is None or aa is None:
            return

        settings = {key: self._current_value(i)
                    for i, (_, key, _, _, _) in enumerate(_PARAMS)}
        fa.apply_training_settings(settings)
        aa.apply_training_settings(settings)

        from scenes.battle import BattleScene
        self.game_manager.push_scene(BattleScene(self.game_manager))

    # ── Update / Draw ─────────────────────────────────────────────────────────

    def update(self, dt: float):
        pass

    def draw(self, surface: pygame.Surface):
        surface.fill((22, 22, 32))
        sw, sh = surface.get_size()
        self._click_rects = []

        # ── Title ─────────────────────────────────────────────────────────────
        title = self._f_title.render("Training Setup", True, (180, 220, 255))
        surface.blit(title, title.get_rect(center=(sw // 2, 52)))

        sub = self._f_sub.render(
            "Configure AI training parameters before the battle starts   |   "
            "Arrow keys to navigate, Left/Right to change, Enter to start",
            True, (130, 130, 160))
        surface.blit(sub, sub.get_rect(center=(sw // 2, 92)))

        # ── Param rows ────────────────────────────────────────────────────────
        row_h   = 64
        start_y = 128
        lbl_x   = sw // 2 - 340
        val_x   = sw // 2 + 20
        btn_w   = 34
        btn_h   = 30

        for i, (label, key, options, _, fmt) in enumerate(_PARAMS):
            y = start_y + i * row_h
            selected = (i == self._selected)
            cur_val  = self._current_value(i)

            # Row highlight
            if selected:
                hl = pygame.Surface((sw - 80, row_h - 6), pygame.SRCALPHA)
                hl.fill((60, 80, 120, 90))
                surface.blit(hl, (40, y))

            # Parameter label
            lbl_col = (200, 230, 255) if selected else (160, 160, 190)
            lbl_s   = self._f_label.render(label, True, lbl_col)
            surface.blit(lbl_s, (lbl_x, y + 6))

            # Hint
            hint_s = self._f_hint.render(_HINTS[i], True, (100, 100, 130))
            surface.blit(hint_s, (lbl_x, y + 32))

            # "◄" button
            minus_r = pygame.Rect(val_x - btn_w - 8, y + (row_h - btn_h) // 2, btn_w, btn_h)
            pygame.draw.rect(surface, (50, 70, 110), minus_r, border_radius=5)
            pygame.draw.rect(surface, (90, 120, 180), minus_r, 1, border_radius=5)
            m_s = self._f_val.render("◄", True, (180, 210, 255))
            surface.blit(m_s, m_s.get_rect(center=minus_r.center))
            self._click_rects.append((minus_r, (i, -1)))

            # Value
            val_str = fmt(cur_val)
            val_col = (120, 255, 120) if key == "preset_only" and cur_val else \
                      (255, 200, 80)  if key == "preset_only" else \
                      (255, 220, 80)
            val_s   = self._f_val.render(val_str, True, val_col)
            val_r   = val_s.get_rect(midleft=(val_x + 6, y + row_h // 2))
            surface.blit(val_s, val_r)

            # "►" button  (placed after value)
            plus_x  = val_r.right + 14
            plus_r  = pygame.Rect(plus_x, y + (row_h - btn_h) // 2, btn_w, btn_h)
            pygame.draw.rect(surface, (50, 70, 110), plus_r, border_radius=5)
            pygame.draw.rect(surface, (90, 120, 180), plus_r, 1, border_radius=5)
            p_s = self._f_val.render("►", True, (180, 210, 255))
            surface.blit(p_s, p_s.get_rect(center=plus_r.center))
            self._click_rects.append((plus_r, (i, +1)))

            # Separator line
            if i < len(_PARAMS) - 1:
                pygame.draw.line(surface, (40, 45, 65),
                                 (40, y + row_h - 2), (sw - 40, y + row_h - 2))

        # ── Buttons ────────────────────────────────────────────────────────────
        btn_y    = start_y + len(_PARAMS) * row_h + 20
        btn_h2   = 46
        btn_gap  = 20

        back_w   = 160
        start_w  = 200
        total_bw = back_w + btn_gap + start_w
        bx       = sw // 2 - total_bw // 2

        # Back button
        back_r = pygame.Rect(bx, btn_y, back_w, btn_h2)
        pygame.draw.rect(surface, (50, 40, 55), back_r, border_radius=8)
        pygame.draw.rect(surface, (120, 90, 130), back_r, 2, border_radius=8)
        b_s = self._f_btn.render("← Back", True, (200, 180, 210))
        surface.blit(b_s, b_s.get_rect(center=back_r.center))
        self._click_rects.append((back_r, "back"))

        # Start Battle button
        start_r = pygame.Rect(bx + back_w + btn_gap, btn_y, start_w, btn_h2)
        pygame.draw.rect(surface, (30, 70, 40), start_r, border_radius=8)
        pygame.draw.rect(surface, (80, 200, 100), start_r, 2, border_radius=8)
        s_s = self._f_btn.render("Start Battle ▶", True, (120, 255, 140))
        surface.blit(s_s, s_s.get_rect(center=start_r.center))
        self._click_rects.append((start_r, "start"))

        # Keyboard hint
        kh = self._f_hint.render(
            "[ ↑↓ ] select row     [ ←/► ] change value     [ Enter ] start battle     [ ESC ] back",
            True, (80, 80, 110))
        surface.blit(kh, kh.get_rect(center=(sw // 2, btn_y + btn_h2 + 22)))
