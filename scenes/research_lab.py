"""
Research Lab — meta-upgrade screen between battles.

Tabs:
  AI Minions — spend Coins to raise HP, Attack, Speed, Atk Speed, Stamina/Shoot CD
               for all four minion types (Fighter, Archer, Fire Mage, Ice Mage).
  AI Master  — upgrade MP, spells, deployment limit, replay buffer;
               trigger memory replay training (all 4 agents).
"""
import threading
import pygame
from engine.scene import BaseScene
from config import CFG

UPGRADE_COSTS = [50, 100, 150, 200, 250]
MAX_LEVEL     = 5

# Fighter / Archer: 5 stats (includes Stamina)
STATS = [
    ("HP",          "hp"),
    ("Attack",      "attack"),
    ("Move Speed",  "move_speed"),
    ("Atk Speed",   "attack_speed"),
    ("Stamina",     "stamina"),
]

# Fire Mage / Ice Mage: 5 stats (includes MP; Atk Speed = Shoot CD)
MAGE_STATS = [
    ("HP",          "hp"),
    ("Attack",      "attack"),
    ("Move Speed",  "move_speed"),
    ("Shoot CD",    "attack_speed"),
    ("MP",          "stamina"),
]

_MINIONS       = ["fighter", "archer", "fire_mage", "ice_mage"]
_MINION_LABELS = ["Fighter", "Archer", "Fire Mage", "Ice Mage"]
_MINION_COLORS = [(100, 160, 255), (100, 255, 140), (255, 120, 40), (80, 180, 255)]

_STAT_EFFECTS = [
    "+20 max HP per level",
    "+5 attack damage per level",
    "+20 move speed per level",
    "-0.05 s attack cooldown per level",
    "+20 max stamina per level",
]
_MAGE_STAT_EFFECTS = [
    "+20 max HP per level",
    "+5 attack damage per level",
    "+20 move speed per level",
    "-0.05 s shoot cooldown per level",
    "+20 max MP per level",
]


def _col_stats(col: int):
    """Return the stat list for the given column index."""
    return STATS if col < 2 else MAGE_STATS


_AIM = CFG["ai_master_upgrades"]
_MR  = CFG["memory_replay"]

# AI Master upgrade rows: (display_name, save_key, costs_list, max_level, effect_desc)
_AI_MASTER_ROWS = [
    ("Max MP",          "max_mp",        _AIM["max_mp_costs"],        5, "+20 max MP"),
    ("MP Regen",        "mp_regen",      _AIM["mp_regen_costs"],      5, "+2 MP/s"),
    ("Heal Amount",     "heal_amount",   _AIM["heal_amount_costs"],   5, "+15 heal HP"),
    ("Heal Radius",     "heal_radius",   _AIM["heal_radius_costs"],   5, "+20 px"),
    ("Heal Cooldown",   "heal_cooldown", _AIM["heal_cooldown_costs"], 5, "-1.0 s cooldown"),
    ("Fireball DMG",    "fb_damage",     _AIM["fb_damage_costs"],     5, "+20 damage"),
    ("Fireball Radius", "fb_radius",     _AIM["fb_radius_costs"],     5, "+15 px radius"),
    ("Fireball CD",     "fb_cooldown",   _AIM["fb_cooldown_costs"],   5, "-1.5 s cooldown"),
    ("Deploy Limit",    "deployment",    _AIM["deployment_costs"],    5, "+global cap (2→5→8→12→16→20)"),
]

_ROW_H = 52


class ResearchLabScene(BaseScene):
    def __init__(self, game_manager):
        super().__init__(game_manager)
        self._font_title  = pygame.font.SysFont("arial", 44, bold=True)
        self._font_header = pygame.font.SysFont("arial", 24, bold=True)
        self._font_stat   = pygame.font.SysFont("arial", 20)
        self._font_small  = pygame.font.SysFont("arial", 15)
        self._font_stats  = pygame.font.SysFont("arial", 13)
        self._font_btn    = pygame.font.SysFont("arial", 26, bold=True)
        self._font_tab    = pygame.font.SysFont("arial", 20, bold=True)

        # Active tab: 0 = AI Minions, 1 = AI Master, 2 = Battle Simulation
        self._tab    = 0
        self._col    = 0   # 0=fighter,1=archer,2=fire_mage,3=ice_mage
        self._row    = 0   # selected row within current column
        self._aim_row = 0  # selected row in AI Master tab

        self._msg       = ""
        self._msg_timer = 0.0

        self._click_rects = []   # [(rect, info...)]

        # Memory replay training
        self._replay_iters        = 100
        self._replay_running      = False
        self._replay_result_lines = []
        self._replay_saving       = False
        self._replay_progress     = 0.0
        self._replay_thread : threading.Thread | None = None

        # Battle Simulation config
        self._sim_counts = {"fighter": 0, "archer": 0, "fire_mage": 0, "ice_mage": 0}
        self._sim_panel_open  = True    # DQN params panel expanded/collapsed
        self._sim_scroll      = 0       # scroll offset (pixels) inside expanded panel
        # DQN/training params (passed to BattleSimulationScene on launch)
        _bsim = CFG.get("battle_simulation", {})
        self._sim_train_steps  = int(_bsim.get("default_train_steps", 1))
        self._sim_buffer_rate  = int(_bsim.get("default_buffer_rate", 10))
        self._sim_lr           = float(_bsim.get("default_lr",          1e-4))
        self._sim_batch        = int(_bsim.get("default_batch",         32))
        self._sim_gamma        = float(_bsim.get("default_gamma",       0.99))
        self._sim_noise_sigma  = float(_bsim.get("default_noise_sigma", 0.5))
        # Monster spawn rates
        self._sim_swarm_rate  = int(_bsim.get("default_swarm_rate",   60))
        self._sim_boss_every  = int(_bsim.get("default_boss_every",   120))

    # ── Accessors ─────────────────────────────────────────────────────────

    def _research(self):
        return self.game_manager.save_data["research"]

    def _ai_master(self):
        return self.game_manager.save_data.setdefault("ai_master", {})

    def _stats(self):
        return self.game_manager.save_data.get("stats", {})

    def _level(self, col: int, row: int) -> int:
        stats = _col_stats(col)
        return self._research()[_MINIONS[col]].get(stats[row][1], 0)

    def _set_level(self, col: int, row: int, val: int):
        stats = _col_stats(col)
        self._research()[_MINIONS[col]][stats[row][1]] = val

    def _aim_level(self, row: int) -> int:
        _, key, _, _, _ = _AI_MASTER_ROWS[row]
        return self._ai_master().get(key, 0)

    def _clamp_row(self):
        """Clamp _row to valid range for the current column."""
        max_r = len(_col_stats(self._col)) - 1
        self._row = min(self._row, max_r)

    # ── Events ────────────────────────────────────────────────────────────

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            self._process_key(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._process_click(event.pos)
        elif event.type == pygame.MOUSEWHEEL and self._tab == 2 and self._sim_panel_open:
            self._sim_scroll = max(0, self._sim_scroll - event.y * 30)

    def _process_key(self, key: int):
        if key == pygame.K_ESCAPE:
            self.game_manager.pop_scene()
        elif key == pygame.K_TAB:
            self._tab = (self._tab + 1) % 3
        elif self._tab == 2:   # Battle Simulation tab keyboard nav
            if key == pygame.K_b:
                self._start_simulation()
        elif self._tab == 1:   # AI Master tab keyboard nav
            if key == pygame.K_UP:
                self._aim_row = (self._aim_row - 1) % len(_AI_MASTER_ROWS)
            elif key == pygame.K_DOWN:
                self._aim_row = (self._aim_row + 1) % len(_AI_MASTER_ROWS)
            elif key in (pygame.K_RETURN, pygame.K_SPACE):
                self._try_aim_upgrade(self._aim_row)
            elif key == pygame.K_b:
                self._start_battle()
        else:
            n_rows = len(_col_stats(self._col))
            if key == pygame.K_UP:
                self._row = (self._row - 1) % n_rows
            elif key == pygame.K_DOWN:
                self._row = (self._row + 1) % n_rows
            elif key == pygame.K_LEFT:
                self._col = (self._col - 1) % len(_MINIONS)
                self._clamp_row()
            elif key == pygame.K_RIGHT:
                self._col = (self._col + 1) % len(_MINIONS)
                self._clamp_row()
            elif key in (pygame.K_RETURN, pygame.K_SPACE):
                self._try_upgrade()
            elif key == pygame.K_b:
                self._start_battle()

    def _process_click(self, pos):
        for info in self._click_rects:
            rect = info[0]
            if not rect.collidepoint(pos):
                continue
            kind = info[1]
            if kind == "tab":
                self._tab = info[2]
            elif kind == "stat":
                _, _, col, row = info
                if col == self._col and row == self._row:
                    self._try_upgrade()
                else:
                    self._col, self._row = col, row
            elif kind == "header":
                self._col = info[2]
                self._clamp_row()
            elif kind == "aim_stat":
                row = info[2]
                if row == self._aim_row:
                    self._try_aim_upgrade(row)
                else:
                    self._aim_row = row
            elif kind == "battle":
                self._start_battle()
            elif kind == "replay_minus":
                self._replay_iters = max(10, self._replay_iters - 50)
            elif kind == "replay_plus":
                self._replay_iters = min(int(_MR["max_iterations_input"]),
                                         self._replay_iters + 50)
            elif kind == "replay_train":
                self._start_replay_training()
            elif kind == "sim_count":
                _, _, minion_key, delta = info
                self._sim_counts[minion_key] = max(0, self._sim_counts[minion_key] + delta)
            elif kind == "sim_panel_toggle":
                self._sim_panel_open = not self._sim_panel_open
                self._sim_scroll = 0
            elif kind == "sim_param":
                _, _, param_key, delta = info
                self._adjust_sim_param(param_key, delta)
            elif kind == "sim_scroll_up":
                self._sim_scroll = max(0, self._sim_scroll - 40)
            elif kind == "sim_scroll_down":
                self._sim_scroll += 40
            elif kind == "sim_start":
                self._start_simulation()
            break

    # ── Upgrades ──────────────────────────────────────────────────────────

    def _try_upgrade(self):
        level = self._level(self._col, self._row)
        if level >= MAX_LEVEL:
            self._flash("Already at max level!")
            return
        cost = UPGRADE_COSTS[level]
        if self.game_manager.coins < cost:
            self._flash(f"Need {cost} coins  (have {self.game_manager.coins})")
            return
        self.game_manager.coins -= cost
        self._set_level(self._col, self._row, level + 1)
        self.game_manager.save_game()
        stat_name = _col_stats(self._col)[self._row][0]
        self._flash(f"{_MINION_LABELS[self._col]} {stat_name} → Lv.{level + 1}!")

    def _try_aim_upgrade(self, row: int):
        label, key, costs, max_lvl, _ = _AI_MASTER_ROWS[row]
        level = self._aim_level(row)
        if level >= max_lvl:
            self._flash(f"{label} is at max level!")
            return
        cost = costs[level]
        if self.game_manager.coins < cost:
            self._flash(f"Need {cost} coins  (have {self.game_manager.coins})")
            return
        self.game_manager.coins -= cost
        self._ai_master()[key] = level + 1
        self.game_manager.save_game()
        self._flash(f"{label} → Lv.{level + 1}!")

    def _start_battle(self):
        self.game_manager.save_game()
        from scenes.training_setup import TrainingSetupScene
        self.game_manager.push_scene(TrainingSetupScene(self.game_manager))

    def _sim_cost(self) -> int:
        """Total coin cost for the configured minion composition."""
        dc = CFG.get("battle_simulation", {}).get("deploy_costs", {})
        cost = 0
        cost += self._sim_counts["fighter"]   * int(dc.get("fighter",   50))
        cost += self._sim_counts["archer"]    * int(dc.get("archer",    40))
        cost += self._sim_counts["fire_mage"] * int(dc.get("fire_mage", 60))
        cost += self._sim_counts["ice_mage"]  * int(dc.get("ice_mage",  55))
        return cost

    # Param step sizes for +/- buttons in the DQN panel
    _SIM_PARAM_STEPS = {
        "train_steps":  (1, 1, 50),    # (min, step, max)
        "buffer_rate":  (1, 1, 60),
        "lr":           (1e-5, 1e-5, 1e-3),
        "batch":        (8, 8, 256),
        "gamma":        (0.90, 0.01, 0.9999),
        "noise_sigma":  (0.1, 0.05, 1.0),
        "swarm_rate":   (10, 10, 300),
        "boss_every":   (30, 30, 600),
    }

    def _adjust_sim_param(self, key: str, delta: int):
        """Increment (+1) or decrement (-1) a simulation parameter by one step."""
        if key not in self._SIM_PARAM_STEPS:
            return
        lo, step, hi = self._SIM_PARAM_STEPS[key]
        attr = f"_sim_{key}"
        cur  = getattr(self, attr)
        if isinstance(lo, int):
            nv = max(lo, min(hi, cur + delta * step))
        else:
            nv = round(max(lo, min(hi, cur + delta * step)), 6)
        setattr(self, attr, nv)

    def _start_simulation(self):
        if self.game_manager.fighter_agent is None:
            self._flash("No agents — start a regular battle first.")
            return
        sim_cfg = {
            "counts":       dict(self._sim_counts),
            "train_steps":  self._sim_train_steps,
            "buffer_rate":  self._sim_buffer_rate,
            "lr":           self._sim_lr,
            "batch":        self._sim_batch,
            "gamma":        self._sim_gamma,
            "noise_sigma":  self._sim_noise_sigma,
            "swarm_rate":   self._sim_swarm_rate,
            "boss_every":   self._sim_boss_every,
        }
        self.game_manager.save_game()
        from scenes.battle_simulation import BattleSimulationScene
        self.game_manager.push_scene(BattleSimulationScene(self.game_manager, sim_cfg))

    def _flash(self, text: str):
        self._msg       = text
        self._msg_timer = 2.5

    # ── Memory replay training ────────────────────────────────────────────

    def _replay_cost(self) -> int:
        return max(1, round(self._replay_iters * float(_MR["cost_per_iteration"])))

    def _start_replay_training(self):
        if self._replay_running or self._replay_saving:
            return
        cost = self._replay_cost()
        fa   = self.game_manager.fighter_agent
        aa   = self.game_manager.archer_agent
        fma  = self.game_manager.fire_mage_agent
        ima  = self.game_manager.ice_mage_agent
        if fa is None or aa is None:
            self._flash("No agents loaded — start a game first.")
            return
        if self.game_manager.coins < cost:
            self._flash(f"Need {cost} coins  (have {self.game_manager.coins})")
            return
        # Allow training if at least one agent has enough buffer
        any_ready = (fa.tree.size >= fa.min_buffer_size or
                     aa.tree.size >= aa.min_buffer_size or
                     (fma is not None and fma.tree.size >= fma.min_buffer_size) or
                     (ima is not None and ima.tree.size >= ima.min_buffer_size))
        if not any_ready:
            self._flash("Buffer too small — play a battle first to collect data.")
            return
        self.game_manager.coins -= cost
        self.game_manager.save_game()
        self._replay_running      = True
        self._replay_result_lines = []
        self._replay_saving       = False
        self._replay_progress     = 0.0

        n  = self._replay_iters
        gm = self.game_manager

        def _train():
            total_f_loss  = total_a_loss  = 0.0
            total_fm_loss = total_im_loss = 0.0
            total_f_rew   = total_a_rew   = 0.0
            total_fm_rew  = total_im_rew  = 0.0
            steps = 0
            for i in range(n):
                rf  = fa.train_step()
                ra  = aa.train_step()
                rfm = fma.train_step() if fma else {}
                rim = ima.train_step() if ima else {}
                if rf.get("steps",  0) > 0:
                    total_f_loss  += rf["loss"];  total_f_rew  += rf.get("avg_reward",  0.0)
                if ra.get("steps",  0) > 0:
                    total_a_loss  += ra["loss"];  total_a_rew  += ra.get("avg_reward",  0.0)
                if rfm.get("steps", 0) > 0:
                    total_fm_loss += rfm["loss"]; total_fm_rew += rfm.get("avg_reward", 0.0)
                if rim.get("steps", 0) > 0:
                    total_im_loss += rim["loss"]; total_im_rew += rim.get("avg_reward", 0.0)
                steps += 1
                self._replay_progress = (i + 1) / n
            d = max(1, steps)
            self._replay_running = False

            # Save model checkpoints only (buffer saved at session end)
            name = gm.player_name
            if name:
                self._replay_saving = True
                try:
                    fa.save_checkpoint(gm.brain_path(name, "fighter"))
                    aa.save_checkpoint(gm.brain_path(name, "archer"))
                    if fma:
                        fma.save_checkpoint(gm.brain_path(name, "fire_mage"))
                    if ima:
                        ima.save_checkpoint(gm.brain_path(name, "ice_mage"))
                finally:
                    self._replay_saving = False

            self._replay_result_lines = [
                f"F  loss:{total_f_loss/d:.4f}  rew:{total_f_rew/d:.3f}",
                f"A  loss:{total_a_loss/d:.4f}  rew:{total_a_rew/d:.3f}",
                f"FM loss:{total_fm_loss/d:.4f}  rew:{total_fm_rew/d:.3f}",
                f"IM loss:{total_im_loss/d:.4f}  rew:{total_im_rew/d:.3f}",
            ]

        self._replay_thread = threading.Thread(target=_train, daemon=True)
        self._replay_thread.start()

    # ── Update / Draw ─────────────────────────────────────────────────────

    def update(self, dt: float):
        if self._msg_timer > 0:
            self._msg_timer -= dt

    def draw(self, surface: pygame.Surface):
        surface.fill((18, 18, 28))
        sw, sh = surface.get_size()
        cx = sw // 2
        self._click_rects = []

        # ── Title ──────────────────────────────────────────────────────────
        title = self._font_title.render("Research Lab", True, (180, 220, 255))
        surface.blit(title, title.get_rect(center=(cx, 36)))

        # ── Player info ────────────────────────────────────────────────────
        info = self._font_stat.render(
            f"Commander: {self.game_manager.player_name}          "
            f"Coins: {self.game_manager.coins}", True, (255, 215, 0))
        surface.blit(info, info.get_rect(center=(cx, 78)))

        pygame.draw.line(surface, (50, 50, 80), (40, 96), (sw - 40, 96), 1)

        # ── Tab strip ──────────────────────────────────────────────────────
        tab_labels = ["AI Minions", "AI Master", "Battle Simulation"]
        tab_colors = [(100, 220, 150), (200, 130, 255), (255, 180, 60)]
        tab_w = 200
        tab_h = 36
        tab_y = 102
        tab_start_x = cx - (len(tab_labels) * tab_w) // 2
        for ti, (tl, tc) in enumerate(zip(tab_labels, tab_colors)):
            tx    = tab_start_x + ti * tab_w
            trect = pygame.Rect(tx, tab_y, tab_w - 4, tab_h)
            bg    = (35, 35, 60) if ti == self._tab else (20, 20, 35)
            bord  = tc if ti == self._tab else (50, 50, 80)
            pygame.draw.rect(surface, bg, trect, border_radius=6)
            pygame.draw.rect(surface, bord, trect, 2, border_radius=6)
            ts = self._font_tab.render(tl, True, tc if ti == self._tab else (80, 80, 100))
            surface.blit(ts, ts.get_rect(center=trect.center))
            self._click_rects.append((trect, "tab", ti))

        content_top = tab_y + tab_h + 8

        if self._tab == 0:
            self._draw_minion_tab(surface, sw, sh, cx, content_top)
        elif self._tab == 1:
            self._draw_ai_master_tab(surface, sw, sh, cx, content_top)
        else:
            self._draw_battle_simulation_tab(surface, sw, sh, cx, content_top)

        # ── Flash message ──────────────────────────────────────────────────
        if self._msg and self._msg_timer > 0:
            alpha   = min(255, int(255 * min(1.0, self._msg_timer)))
            msg_s   = self._font_stat.render(self._msg, True, (255, 230, 100))
            tmp     = pygame.Surface(msg_s.get_size(), pygame.SRCALPHA)
            tmp.blit(msg_s, (0, 0))
            tmp.set_alpha(alpha)
            surface.blit(tmp, tmp.get_rect(center=(cx, sh - 56)))

        # ── Controls hint ──────────────────────────────────────────────────
        ctrl = self._font_stats.render(
            "Tab  Switch tab     Up/Down  Select row     Left/Right  Switch minion"
            "     Enter/Click  Upgrade     [B] Battle     ESC  Back",
            True, (60, 60, 90))
        surface.blit(ctrl, ctrl.get_rect(center=(cx, sh - 16)))

    # ── AI Minions tab ────────────────────────────────────────────────────

    def _draw_minion_tab(self, surface, sw, sh, cx, top_y):
        # 4 columns, evenly spaced across the screen
        n_cols = len(_MINIONS)
        col_xs = [sw * (i + 1) // (n_cols + 1) for i in range(n_cols)]

        max_rows = max(len(STATS), len(MAGE_STATS))
        rows_bottom = top_y + 32 + max_rows * _ROW_H

        for ci in range(n_cols):
            self._draw_minion_column(surface, ci, col_xs[ci], top_y)

        stats_top = rows_bottom + 10
        pygame.draw.line(surface, (40, 40, 68), (60, stats_top), (sw - 60, stats_top), 1)
        hdr = self._font_stats.render("— Battle Statistics —", True, (120, 130, 160))
        surface.blit(hdr, hdr.get_rect(center=(cx, stats_top + 10)))
        for ci in range(n_cols):
            self._draw_stats_column(surface, ci, col_xs[ci], stats_top + 22)

        # Effect hint for selected row
        cur_effects = _MAGE_STAT_EFFECTS if self._col >= 2 else _STAT_EFFECTS
        effect_y = stats_top + 22 + 5 * 15 + 8
        eff_surf = self._font_small.render(
            f"Effect: {cur_effects[self._row]}", True, (140, 160, 200))
        surface.blit(eff_surf, eff_surf.get_rect(center=(cx, effect_y)))

        btn_top  = effect_y + 22
        btn_rect = pygame.Rect(cx - 150, btn_top, 300, 44)
        pygame.draw.rect(surface, (25, 70, 30), btn_rect, border_radius=8)
        pygame.draw.rect(surface, (60, 180, 70), btn_rect, 2, border_radius=8)
        btn_t = self._font_btn.render("[B] Start Battle", True, (140, 255, 150))
        surface.blit(btn_t, btn_t.get_rect(center=btn_rect.center))
        self._click_rects.append((btn_rect, "battle"))

    def _draw_minion_column(self, surface, ci, col_x, top_y):
        """Draw one minion upgrade column (narrower for 4-column layout)."""
        lc      = _MINION_COLORS[ci]
        stats   = _col_stats(ci)
        half_w  = (surface.get_width() // (len(_MINIONS) + 1)) // 2 - 4

        hdr = self._font_header.render(_MINION_LABELS[ci], True, lc)
        hdr_r = hdr.get_rect(center=(col_x, top_y + 14))
        surface.blit(hdr, hdr_r)
        self._click_rects.append((hdr_r.inflate(20, 8), "header", ci))

        for ri, (sl, _) in enumerate(stats):
            level = self._level(ci, ri)
            sel   = (ci == self._col and ri == self._row and self._tab == 0)
            ry    = top_y + 32 + ri * _ROW_H

            row_rect = pygame.Rect(col_x - half_w, ry - 4, half_w * 2, _ROW_H - 4)
            self._click_rects.append((row_rect, "stat", ci, ri))

            if sel:
                pygame.draw.rect(surface, (35, 45, 70), row_rect, border_radius=6)
                pygame.draw.rect(surface, (70, 90, 170), row_rect, 2, border_radius=6)

            sc = (255, 255, 255) if sel else (170, 170, 190)
            ss = self._font_stat.render(sl, True, sc)
            surface.blit(ss, ss.get_rect(midleft=(col_x - half_w + 6, ry + _ROW_H // 2 - 8)))

            # Pip dots (below label)
            pip_y = ry + _ROW_H // 2 + 6
            for p in range(MAX_LEVEL):
                px = col_x - half_w + 6 + p * 20
                pc = (60, 200, 90) if p < level else (40, 42, 58)
                pygame.draw.rect(surface, pc, (px, pip_y, 17, 13), border_radius=3)
                if p < level:
                    pygame.draw.rect(surface, (90, 230, 110), (px, pip_y, 17, 13), 1, border_radius=3)

            if level < MAX_LEVEL:
                cost = UPGRADE_COSTS[level]
                ca   = self.game_manager.coins >= cost
                cc   = (255, 220, 60) if (sel and ca) else (180, 140, 40) if ca else (160, 80, 80)
                cs   = self._font_small.render(f"{cost}¢", True, cc)
            else:
                cs = self._font_small.render("MAX", True, (80, 200, 90))
            surface.blit(cs, cs.get_rect(midright=(col_x + half_w - 4, ry + _ROW_H // 2 - 8)))

    def _draw_stats_column(self, surface, ci, col_x, top_y):
        mk  = _MINIONS[ci]
        s   = self._stats().get(mk, {})
        col = (140, 150, 175)
        if mk == "fighter":
            rows = [
                f"Kills: {s.get('total_kills', 0):,}",
                f"Damage: {s.get('total_damage', 0):,}",
                f"Max Waves: {s.get('max_waves_survived', 0)}",
                f"Train Waves: {s.get('training_waves', 0)}",
                f"Swing Acc: {s.get('attacks_hit', 0) / max(1, s.get('attacks_attempted', 1)) * 100:.1f}%",
            ]
        elif mk == "archer":
            f = s.get("shots_fired", 0)
            rows = [
                f"Kills: {s.get('total_kills', 0):,}",
                f"Damage: {s.get('total_damage', 0):,}",
                f"Max Waves: {s.get('max_waves_survived', 0)}",
                f"Shots Fired: {f:,}",
                f"Shot Acc: {s.get('shots_hit', 0) / max(1, f) * 100:.1f}%",
            ]
        elif mk == "fire_mage":
            rows = [
                f"Kills: {s.get('total_kills', 0):,}",
                f"Damage: {s.get('total_damage', 0):,}",
                f"DQN Role: FM",
            ]
        else:  # ice_mage
            rows = [
                f"Kills: {s.get('total_kills', 0):,}",
                f"Damage: {s.get('total_damage', 0):,}",
                f"DQN Role: IM",
            ]
        for i, text in enumerate(rows):
            surf = self._font_stats.render(text, True, col)
            surface.blit(surf, surf.get_rect(midleft=(col_x - 80, top_y + i * 15 + 6)))

    # ── AI Master tab ─────────────────────────────────────────────────────

    def _draw_ai_master_tab(self, surface, sw, sh, cx, top_y):
        fa  = self.game_manager.fighter_agent
        aa  = self.game_manager.archer_agent
        fma = self.game_manager.fire_mage_agent
        ima = self.game_manager.ice_mage_agent

        f_buf  = fa.tree.size  if fa  else 0
        a_buf  = aa.tree.size  if aa  else 0
        fm_buf = fma.tree.size if fma else 0
        im_buf = ima.tree.size if ima else 0

        row_h  = 25
        col_w  = sw // 2 - 30
        n_rows = len(_AI_MASTER_ROWS)

        hdr = self._font_header.render("AI Master Upgrades", True, (200, 130, 255))
        surface.blit(hdr, hdr.get_rect(center=(cx, top_y + 16)))

        row_start = top_y + 46

        for ri, (label, key, costs, max_lvl, effect) in enumerate(_AI_MASTER_ROWS):
            level = self._aim_level(ri)
            sel   = (ri == self._aim_row)
            ry    = row_start + ri * row_h

            row_rect = pygame.Rect(cx - col_w // 2 - 60, ry, col_w + 120, row_h - 3)
            self._click_rects.append((row_rect, "aim_stat", ri))

            if sel:
                pygame.draw.rect(surface, (35, 25, 55), row_rect, border_radius=6)
                pygame.draw.rect(surface, (130, 70, 200), row_rect, 2, border_radius=6)

            tc = (255, 255, 255) if sel else (180, 160, 220)
            ts = self._font_stat.render(label, True, tc)
            surface.blit(ts, ts.get_rect(midleft=(row_rect.x + 10, ry + row_h // 2 - 2)))

            for p in range(max_lvl):
                px = cx + 10 + p * 22
                py = ry + row_h // 2 - 8
                pc = (160, 80, 255) if p < level else (40, 35, 58)
                pygame.draw.rect(surface, pc, (px, py, 18, 18), border_radius=4)
                if p < level:
                    pygame.draw.rect(surface, (200, 120, 255), (px, py, 18, 18), 1, border_radius=4)

            if level < max_lvl:
                cost = costs[level]
                ca   = self.game_manager.coins >= cost
                cc   = (255, 220, 60) if (sel and ca) else (180, 140, 40) if ca else (160, 80, 80)
                cs   = self._font_small.render(f"{cost} coins", True, cc)
            else:
                cs = self._font_small.render("MAX", True, (80, 200, 90))
            surface.blit(cs, cs.get_rect(midright=(row_rect.right - 10, ry + row_h // 2 - 2)))

            if sel:
                es = self._font_stats.render(f"  {effect}", True, (160, 140, 200))
                surface.blit(es, (row_rect.x + ts.get_width() + 16, ry + row_h // 2 + 6))

        # ── Memory Replay Training ─────────────────────────────────────────
        mr_top = row_start + n_rows * row_h + 18
        pygame.draw.line(surface, (60, 40, 80), (cx - 300, mr_top), (cx + 300, mr_top), 1)

        mr_title = self._font_header.render("Memory Replay Training", True, (200, 160, 255))
        surface.blit(mr_title, mr_title.get_rect(center=(cx, mr_top + 20)))

        # Buffer status — all 4 agents
        bs_text = self._font_small.render(
            f"F:{f_buf:,}  A:{a_buf:,}  FM:{fm_buf:,}  IM:{im_buf:,}",
            True, (150, 140, 170))
        surface.blit(bs_text, bs_text.get_rect(center=(cx, mr_top + 44)))

        cost_total = self._replay_cost()
        iter_text  = self._font_stat.render(
            f"Iterations: {self._replay_iters:,}   Cost: {cost_total:,} coins",
            True, (220, 200, 255))
        surface.blit(iter_text, iter_text.get_rect(center=(cx, mr_top + 70)))

        btn_y = mr_top + 98
        # [−] button
        minus_r = pygame.Rect(cx - 220, btn_y, 60, 34)
        pygame.draw.rect(surface, (50, 35, 70), minus_r, border_radius=6)
        pygame.draw.rect(surface, (120, 80, 180), minus_r, 2, border_radius=6)
        ms = self._font_btn.render("−", True, (200, 180, 255))
        surface.blit(ms, ms.get_rect(center=minus_r.center))
        self._click_rects.append((minus_r, "replay_minus"))

        # [+] button
        plus_r = pygame.Rect(cx + 160, btn_y, 60, 34)
        pygame.draw.rect(surface, (50, 35, 70), plus_r, border_radius=6)
        pygame.draw.rect(surface, (120, 80, 180), plus_r, 2, border_radius=6)
        ps = self._font_btn.render("+", True, (200, 180, 255))
        surface.blit(ps, ps.get_rect(center=plus_r.center))
        self._click_rects.append((plus_r, "replay_plus"))

        # [Train Now] button
        can_afford  = self.game_manager.coins >= cost_total
        _busy       = self._replay_running or self._replay_saving
        train_col   = (60, 30, 100) if not _busy else (30, 40, 70)
        train_bord  = (160, 80, 255) if (can_afford and not _busy) else (70, 50, 110)
        train_r     = pygame.Rect(cx - 110, btn_y, 220, 34)
        pygame.draw.rect(surface, train_col, train_r, border_radius=6)
        pygame.draw.rect(surface, train_bord, train_r, 2, border_radius=6)
        if self._replay_running:
            ts_text = "Training…"
            tc      = (180, 160, 220)
        else:
            ts_text = "Train Now"
            tc      = (200, 160, 255) if (can_afford and not _busy) else (100, 80, 120)
        ts = self._font_btn.render(ts_text, True, tc)
        surface.blit(ts, ts.get_rect(center=train_r.center))
        if not _busy:
            self._click_rects.append((train_r, "replay_train"))

        # Progress bar / saving indicator / result text
        if self._replay_running:
            bar_w = 280
            bar_h = 10
            bar_x = cx - bar_w // 2
            bar_y = btn_y + 44
            pygame.draw.rect(surface, (35, 22, 55), (bar_x, bar_y, bar_w, bar_h), border_radius=4)
            fill_w = int(bar_w * self._replay_progress)
            if fill_w > 0:
                pygame.draw.rect(surface, (140, 80, 220), (bar_x, bar_y, fill_w, bar_h), border_radius=4)
            pygame.draw.rect(surface, (100, 60, 170), (bar_x, bar_y, bar_w, bar_h), 1, border_radius=4)
            pct_s = self._font_small.render(
                f"Training… {int(self._replay_progress * 100)}%", True, (190, 165, 230))
            surface.blit(pct_s, pct_s.get_rect(center=(cx, bar_y + bar_h + 14)))
        elif self._replay_saving:
            sv = self._font_small.render("Saving checkpoints...", True, (255, 200, 60))
            surface.blit(sv, sv.get_rect(center=(cx, btn_y + 46)))
        elif self._replay_result_lines:
            for j, line in enumerate(self._replay_result_lines):
                rs = self._font_small.render(line, True, (180, 220, 180))
                surface.blit(rs, rs.get_rect(center=(cx, btn_y + 44 + j * 16)))

        # Start battle button
        bb_y     = btn_y + 110
        btn_rect = pygame.Rect(cx - 150, bb_y, 300, 44)
        pygame.draw.rect(surface, (25, 70, 30), btn_rect, border_radius=8)
        pygame.draw.rect(surface, (60, 180, 70), btn_rect, 2, border_radius=8)
        btn_t = self._font_btn.render("[B] Start Battle", True, (140, 255, 150))
        surface.blit(btn_t, btn_t.get_rect(center=btn_rect.center))
        self._click_rects.append((btn_rect, "battle"))

    # ── Battle Simulation tab ─────────────────────────────────────────────

    def _draw_battle_simulation_tab(self, surface, sw, sh, cx, top_y):
        """Render the Battle Simulation configuration tab."""
        font_hdr   = self._font_header
        font_stat  = self._font_stat
        font_small = self._font_small
        font_btn   = self._font_btn

        # ── Section: Minion Composition ───────────────────────────────────
        hdr = font_hdr.render("Minion Composition", True, (255, 200, 80))
        surface.blit(hdr, hdr.get_rect(center=(cx, top_y + 16)))

        minion_defs = [
            ("fighter",   "Fighter",   (100, 160, 255)),
            ("archer",    "Archer",    (100, 255, 140)),
            ("fire_mage", "Fire Mage", (255, 120, 40)),
            ("ice_mage",  "Ice Mage",  (80, 200, 255)),
        ]
        _dc = CFG.get("battle_simulation", {}).get("deploy_costs", {})
        deploy_costs = {
            "fighter":   int(_dc.get("fighter",   50)),
            "archer":    int(_dc.get("archer",    40)),
            "fire_mage": int(_dc.get("fire_mage", 60)),
            "ice_mage":  int(_dc.get("ice_mage",  55)),
        }

        comp_y = top_y + 48
        for ci, (key, label, col) in enumerate(minion_defs):
            col_cx = sw * (ci + 1) // (len(minion_defs) + 1)
            dot_r  = 7
            pygame.draw.circle(surface, col, (col_cx - 50, comp_y + 10), dot_r)
            ls = font_stat.render(label, True, col)
            surface.blit(ls, ls.get_rect(midleft=(col_cx - 40, comp_y + 10)))

            btn_h  = 30
            btn_w  = 30
            minus_r = pygame.Rect(col_cx - 55, comp_y + 34, btn_w, btn_h)
            plus_r  = pygame.Rect(col_cx + 25, comp_y + 34, btn_w, btn_h)
            for r, lbl in ((minus_r, "−"), (plus_r, "+")):
                pygame.draw.rect(surface, (50, 35, 70), r, border_radius=5)
                pygame.draw.rect(surface, (100, 70, 140), r, 2, border_radius=5)
                bs = font_btn.render(lbl, True, (200, 180, 255))
                surface.blit(bs, bs.get_rect(center=r.center))
            self._click_rects.append((minus_r, "sim_count", key, -1))
            self._click_rects.append((plus_r,  "sim_count", key,  1))

            count = self._sim_counts[key]
            cs    = font_stat.render(str(count), True, (255, 255, 255))
            surface.blit(cs, cs.get_rect(center=(col_cx, comp_y + 49)))

            unit_cost      = deploy_costs[key]
            total_for_role = count * unit_cost
            cost_s = font_small.render(
                f"{unit_cost}¢ each  ×{count} = {total_for_role}¢",
                True, (180, 160, 220))
            surface.blit(cost_s, cost_s.get_rect(center=(col_cx, comp_y + 74)))

        total_cost = self._sim_cost()
        tc_s = font_stat.render(f"Total Cost: {total_cost} Coins", True, (255, 215, 0))
        surface.blit(tc_s, tc_s.get_rect(center=(cx, comp_y + 100)))

        # ── Section: Advanced DQN Parameters (collapsible) ────────────────
        panel_top = comp_y + 132
        pygame.draw.line(surface, (60, 50, 80), (cx - 320, panel_top), (cx + 320, panel_top), 1)

        arrow = "▼" if self._sim_panel_open else "▶"
        ph    = font_hdr.render(f"{arrow}  Advanced Parameters", True, (200, 160, 255))
        toggle_rect = pygame.Rect(cx - ph.get_width() // 2 - 6, panel_top + 6,
                                   ph.get_width() + 12, 28)
        pygame.draw.rect(surface, (30, 22, 50), toggle_rect, border_radius=5)
        surface.blit(ph, ph.get_rect(center=toggle_rect.center))
        self._click_rects.append((toggle_rect, "sim_panel_toggle"))

        if self._sim_panel_open:
            panel_inner_top = panel_top + 40
            panel_h         = sh - panel_inner_top - 80
            clip_rect       = pygame.Rect(0, panel_inner_top, sw, panel_h)
            old_clip        = surface.get_clip()
            surface.set_clip(clip_rect)

            params = [
                ("DQN Training Steps / Frame", "_sim_train_steps",  lambda v: str(int(v)),   "train_steps"),
                ("Memory Buffer Rate (frames)", "_sim_buffer_rate", lambda v: str(int(v)),   "buffer_rate"),
                ("Learning Rate",              "_sim_lr",           lambda v: f"{v:.1e}",    "lr"),
                ("Batch Size",                 "_sim_batch",        lambda v: str(int(v)),   "batch"),
                ("Discount Factor (γ)",        "_sim_gamma",        lambda v: f"{v:.4f}",    "gamma"),
                ("NoisyNet Sigma (exploration)", "_sim_noise_sigma", lambda v: f"{v:.2f}",  "noise_sigma"),
                ("Swarm Spawn Rate (/ min)",   "_sim_swarm_rate",   lambda v: str(int(v)),   "swarm_rate"),
                ("Boss Spawn Interval (s)",    "_sim_boss_every",   lambda v: str(int(v)),   "boss_every"),
            ]

            row_h_p    = 34
            max_scroll = max(0, len(params) * row_h_p - panel_h + 8)
            self._sim_scroll = min(self._sim_scroll, max_scroll)
            draw_y = panel_inner_top - self._sim_scroll

            for param_label, attr, fmt, pkey in params:
                ry     = draw_y
                draw_y += row_h_p
                if ry + row_h_p < panel_inner_top or ry > panel_inner_top + panel_h:
                    continue

                pl_s = font_stat.render(param_label, True, (180, 170, 220))
                surface.blit(pl_s, pl_s.get_rect(midleft=(cx - 300, ry + row_h_p // 2)))

                val     = getattr(self, attr)
                val_str = fmt(val)
                vw      = 80
                vx      = cx + 80

                dec_r = pygame.Rect(vx - 48, ry + 3, 40, row_h_p - 6)
                inc_r = pygame.Rect(vx + vw + 8, ry + 3, 40, row_h_p - 6)
                val_r = pygame.Rect(vx,       ry + 3, vw, row_h_p - 6)

                for r, lbl in ((dec_r, "−"), (inc_r, "+")):
                    pygame.draw.rect(surface, (45, 32, 65), r, border_radius=5)
                    pygame.draw.rect(surface, (100, 70, 150), r, 1, border_radius=5)
                    bs = font_btn.render(lbl, True, (200, 180, 255))
                    surface.blit(bs, bs.get_rect(center=r.center))

                pygame.draw.rect(surface, (28, 20, 45), val_r, border_radius=4)
                pygame.draw.rect(surface, (80, 60, 120), val_r, 1, border_radius=4)
                vs = font_stat.render(val_str, True, (255, 240, 160))
                surface.blit(vs, vs.get_rect(center=val_r.center))

                self._click_rects.append((dec_r, "sim_param", pkey, -1))
                self._click_rects.append((inc_r, "sim_param", pkey,  1))

            if max_scroll > 0:
                scroll_x = cx + 340
                up_r   = pygame.Rect(scroll_x, panel_inner_top,               24, 24)
                down_r = pygame.Rect(scroll_x, panel_inner_top + panel_h - 24, 24, 24)
                for r, lbl in ((up_r, "▲"), (down_r, "▼")):
                    pygame.draw.rect(surface, (40, 30, 60), r, border_radius=4)
                    pygame.draw.rect(surface, (100, 70, 150), r, 1, border_radius=4)
                    as_ = font_small.render(lbl, True, (180, 160, 220))
                    surface.blit(as_, as_.get_rect(center=r.center))
                self._click_rects.append((up_r,   "sim_scroll_up"))
                self._click_rects.append((down_r, "sim_scroll_down"))

            surface.set_clip(old_clip)

        # ── Start Simulation button ───────────────────────────────────────
        btn_y     = sh - 68
        has_any   = any(v > 0 for v in self._sim_counts.values())
        has_agents = self.game_manager.fighter_agent is not None
        ready     = has_any and has_agents
        bg_col    = (30, 70, 25) if ready else (30, 30, 30)
        bd_col    = (60, 200, 70) if ready else (60, 60, 60)
        txt_col   = (140, 255, 150) if ready else (80, 80, 80)
        btn_rect  = pygame.Rect(cx - 160, btn_y, 320, 44)
        pygame.draw.rect(surface, bg_col, btn_rect, border_radius=8)
        pygame.draw.rect(surface, bd_col, btn_rect, 2, border_radius=8)
        hint = "[B] Start Simulation" if ready else "[B] Add minions to start"
        bt   = font_btn.render(hint, True, txt_col)
        surface.blit(bt, bt.get_rect(center=btn_rect.center))
        if ready:
            self._click_rects.append((btn_rect, "sim_start"))

        if not has_agents:
            info_s = font_small.render(
                "No DQN agents loaded — play a regular battle first.",
                True, (200, 80, 80))
            surface.blit(info_s, info_s.get_rect(center=(cx, btn_y - 20)))
