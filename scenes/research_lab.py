"""
Research Lab — meta-upgrade screen between battles.

Tabs:
  Fighter / Archer  — spend Coins to raise HP, Attack, Speed, Atk Speed, Stamina.
  AI Master         — upgrade MP, spells, deployment limit, replay buffer;
                      trigger memory replay training.
"""
import threading
import pygame
from engine.scene import BaseScene
from config import CFG

UPGRADE_COSTS = [50, 100, 150, 200, 250]
MAX_LEVEL     = 5

STATS = [
    ("HP",          "hp"),
    ("Attack",      "attack"),
    ("Move Speed",  "move_speed"),
    ("Atk Speed",   "attack_speed"),
    ("Stamina",     "stamina"),
]

_MINIONS       = ["fighter", "archer"]
_MINION_LABELS = ["Fighter", "Archer"]
_MINION_COLORS = [(100, 160, 255), (100, 255, 140)]

_STAT_EFFECTS = [
    "+20 max HP per level",
    "+5 attack damage per level",
    "+20 move speed per level",
    "-0.05 s attack cooldown per level",
    "+20 max stamina per level",
]

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

_ROW_START_Y = 148
_ROW_H       = 58


class ResearchLabScene(BaseScene):
    def __init__(self, game_manager):
        super().__init__(game_manager)
        self._font_title  = pygame.font.SysFont("arial", 44, bold=True)
        self._font_header = pygame.font.SysFont("arial", 28, bold=True)
        self._font_stat   = pygame.font.SysFont("arial", 22)
        self._font_small  = pygame.font.SysFont("arial", 17)
        self._font_stats  = pygame.font.SysFont("arial", 14)
        self._font_btn    = pygame.font.SysFont("arial", 26, bold=True)
        self._font_tab    = pygame.font.SysFont("arial", 20, bold=True)

        # Active tab: 0 = AI Minions, 1 = AI Master
        self._tab    = 0
        self._col    = 0   # used for fighter/archer tabs
        self._row    = 0   # selected row
        self._aim_row = 0  # selected row in AI Master tab

        self._msg       = ""
        self._msg_timer = 0.0

        self._click_rects = []   # [(rect, info...)]

        # Memory replay training
        self._replay_iters        = 100
        self._replay_running      = False
        self._replay_result_lines = []   # list of strings shown after training completes
        self._replay_saving       = False
        self._replay_progress     = 0.0  # 0.0–1.0, updated live from the training thread
        self._replay_thread : threading.Thread | None = None

    # ── Accessors ─────────────────────────────────────────────────────────

    def _research(self):
        return self.game_manager.save_data["research"]

    def _ai_master(self):
        return self.game_manager.save_data.setdefault("ai_master", {})

    def _stats(self):
        return self.game_manager.save_data.get("stats", {})

    def _level(self, col: int, row: int) -> int:
        return self._research()[_MINIONS[col]].get(STATS[row][1], 0)

    def _set_level(self, col: int, row: int, val: int):
        self._research()[_MINIONS[col]][STATS[row][1]] = val

    def _aim_level(self, row: int) -> int:
        _, key, _, _, _ = _AI_MASTER_ROWS[row]
        return self._ai_master().get(key, 0)

    # ── Events ────────────────────────────────────────────────────────────

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            self._process_key(event.key)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._process_click(event.pos)

    def _process_key(self, key: int):
        if key == pygame.K_ESCAPE:
            self.game_manager.pop_scene()
        elif key == pygame.K_TAB:
            self._tab = (self._tab + 1) % 2
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
            if key == pygame.K_UP:
                self._row = (self._row - 1) % len(STATS)
            elif key == pygame.K_DOWN:
                self._row = (self._row + 1) % len(STATS)
            elif key in (pygame.K_LEFT, pygame.K_RIGHT):
                self._col = 1 - self._col
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
        self._flash(f"{_MINION_LABELS[self._col]} {STATS[self._row][0]} → Lv.{level + 1}!")

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
        if fa is None or aa is None:
            self._flash("No agents loaded — start a game first.")
            return
        if self.game_manager.coins < cost:
            self._flash(f"Need {cost} coins  (have {self.game_manager.coins})")
            return
        if fa.tree.size < fa.min_buffer_size and aa.tree.size < aa.min_buffer_size:
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
            total_f_loss, total_a_loss = 0.0, 0.0
            total_f_rew,  total_a_rew  = 0.0, 0.0
            steps = 0
            for i in range(n):
                rf = fa.train_step()
                ra = aa.train_step()
                if rf.get("steps", 0) > 0:
                    total_f_loss += rf["loss"]
                    total_f_rew  += rf.get("avg_reward", 0.0)
                    steps        += 1
                if ra.get("steps", 0) > 0:
                    total_a_loss += ra["loss"]
                    total_a_rew  += ra.get("avg_reward", 0.0)
                self._replay_progress = (i + 1) / n
            avg_f_loss = total_f_loss / max(1, steps)
            avg_a_loss = total_a_loss / max(1, steps)
            avg_f_rew  = total_f_rew  / max(1, steps)
            avg_a_rew  = total_a_rew  / max(1, steps)
            self._replay_running = False

            # Save model checkpoints only (buffer saved at session end)
            name = gm.player_name
            if name:
                self._replay_saving = True
                try:
                    fa.save_checkpoint(gm.brain_path(name, "fighter"))
                    aa.save_checkpoint(gm.brain_path(name, "archer"))
                finally:
                    self._replay_saving = False

            self._replay_result_lines = [
                f"F  loss:{avg_f_loss:.4f}  avg rew:{avg_f_rew:.3f}",
                f"A  loss:{avg_a_loss:.4f}  avg rew:{avg_a_rew:.3f}",
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
        tab_labels = ["AI Minions", "AI Master"]
        tab_colors = [(100, 220, 150), (200, 130, 255)]
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
        else:
            self._draw_ai_master_tab(surface, sw, sh, cx, content_top)

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
            "Tab  Switch tab     Up/Down  Select     Enter/Click  Upgrade     [B] Battle     ESC  Back",
            True, (60, 60, 90))
        surface.blit(ctrl, ctrl.get_rect(center=(cx, sh - 16)))

    # ── Fighter / Archer tab ──────────────────────────────────────────────

    def _draw_minion_tab(self, surface, sw, sh, cx, top_y):
        col_xs     = [sw // 4, 3 * sw // 4]
        rows_bottom = top_y + len(STATS) * _ROW_H

        for ci in range(2):
            self._draw_minion_column(surface, ci, col_xs[ci], top_y)

        stats_top = rows_bottom + 12
        pygame.draw.line(surface, (40, 40, 68), (60, stats_top), (sw - 60, stats_top), 1)
        hdr = self._font_stats.render("— Battle Statistics —", True, (120, 130, 160))
        surface.blit(hdr, hdr.get_rect(center=(cx, stats_top + 10)))
        for ci in range(2):
            self._draw_stats_column(surface, ci, col_xs[ci], stats_top + 22)

        effect_y = stats_top + 22 + 5 * 17 + 10
        eff_surf = self._font_small.render(
            f"Effect: {_STAT_EFFECTS[self._row]}", True, (140, 160, 200))
        surface.blit(eff_surf, eff_surf.get_rect(center=(cx, effect_y)))

        btn_top  = effect_y + 26
        btn_rect = pygame.Rect(cx - 150, btn_top, 300, 48)
        pygame.draw.rect(surface, (25, 70, 30), btn_rect, border_radius=8)
        pygame.draw.rect(surface, (60, 180, 70), btn_rect, 2, border_radius=8)
        btn_t = self._font_btn.render("[B] Start Battle", True, (140, 255, 150))
        surface.blit(btn_t, btn_t.get_rect(center=btn_rect.center))
        self._click_rects.append((btn_rect, "battle"))

    def _draw_minion_column(self, surface, ci, col_x, top_y):
        lc   = _MINION_COLORS[ci]
        hdr  = self._font_header.render(_MINION_LABELS[ci], True, lc)
        hdr_r = hdr.get_rect(center=(col_x, top_y - 22 + _ROW_START_Y - 148))
        # Adjust to content_top
        hdr_r.centery = top_y + 14
        surface.blit(hdr, hdr_r)
        self._click_rects.append((hdr_r.inflate(20, 8), "header", ci))

        for ri, (sl, _) in enumerate(STATS):
            level = self._level(ci, ri)
            sel   = (ci == self._col and ri == self._row and self._tab == ci)
            ry    = top_y + 32 + ri * _ROW_H

            row_rect = pygame.Rect(col_x - 230, ry - 4, 460, _ROW_H - 4)
            self._click_rects.append((row_rect, "stat", ci, ri))

            if sel:
                pygame.draw.rect(surface, (35, 45, 70), row_rect, border_radius=6)
                pygame.draw.rect(surface, (70, 90, 170), row_rect, 2, border_radius=6)

            sc = (255, 255, 255) if sel else (170, 170, 190)
            ss = self._font_stat.render(sl, True, sc)
            surface.blit(ss, ss.get_rect(midleft=(col_x - 220, ry + _ROW_H // 2 - 8)))

            for p in range(MAX_LEVEL):
                px      = col_x - 20 + p * 26
                py      = ry + _ROW_H // 2 - 8
                pc      = (60, 200, 90) if p < level else (40, 42, 58)
                pygame.draw.rect(surface, pc, (px, py, 20, 20), border_radius=4)
                if p < level:
                    pygame.draw.rect(surface, (90, 230, 110), (px, py, 20, 20), 1, border_radius=4)

            if level < MAX_LEVEL:
                cost = UPGRADE_COSTS[level]
                ca   = self.game_manager.coins >= cost
                cc   = (255, 220, 60) if (sel and ca) else (180, 140, 40) if ca else (160, 80, 80)
                cs   = self._font_small.render(f"{cost} coins", True, cc)
            else:
                cs = self._font_small.render("MAX", True, (80, 200, 90))
            surface.blit(cs, cs.get_rect(midright=(col_x + 220, ry + _ROW_H // 2 - 8)))

    def _draw_stats_column(self, surface, ci, col_x, top_y):
        mk  = _MINIONS[ci]
        s   = self._stats().get(mk, {})
        col = (140, 150, 175)
        if mk == "fighter":
            rows = [
                f"Kills: {s.get('total_kills', 0):,}",
                f"Damage: {s.get('total_damage', 0):,}",
                f"Max Waves: {s.get('max_waves_survived', 0)}",
                f"Training Waves: {s.get('training_waves', 0)}",
                f"Swing Acc: {s.get('attacks_hit', 0) / max(1, s.get('attacks_attempted', 1)) * 100:.1f}%",
            ]
        else:
            f = s.get("shots_fired", 0)
            rows = [
                f"Kills: {s.get('total_kills', 0):,}",
                f"Damage: {s.get('total_damage', 0):,}",
                f"Max Waves: {s.get('max_waves_survived', 0)}",
                f"Shots Fired: {f:,}",
                f"Shot Acc: {s.get('shots_hit', 0) / max(1, f) * 100:.1f}%",
            ]
        for i, text in enumerate(rows):
            surf = self._font_stats.render(text, True, col)
            surface.blit(surf, surf.get_rect(midleft=(col_x - 120, top_y + i * 17 + 8)))

    # ── AI Master tab ─────────────────────────────────────────────────────

    def _draw_ai_master_tab(self, surface, sw, sh, cx, top_y):
        surface.blit(self._font_header.render("AI Master Upgrades", True, (200, 130, 255)),
                     pygame.Rect(0, 0, sw, 30).move(0, top_y).inflate(-2, 0)
                     .move(cx - self._font_header.size("AI Master Upgrades")[0] // 2 - sw // 2 + cx // 2, 0))

        row_h = 25
        col_w = sw // 2 - 30
        n_rows = len(_AI_MASTER_ROWS)

        fa = self.game_manager.fighter_agent
        aa = self.game_manager.archer_agent
        f_buf = fa.tree.size if fa else 0
        a_buf = aa.tree.size if aa else 0

        # Header
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

            # Pip dots
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

            # Effect hint (only for selected row)
            if sel:
                es = self._font_stats.render(f"  {effect}", True, (160, 140, 200))
                surface.blit(es, (row_rect.x + ts.get_width() + 16, ry + row_h // 2 + 6))

        # ── Memory Replay Training ─────────────────────────────────────────
        mr_top = row_start + n_rows * row_h + 18
        pygame.draw.line(surface, (60, 40, 80), (cx - 300, mr_top), (cx + 300, mr_top), 1)

        mr_title = self._font_header.render("Memory Replay Training", True, (200, 160, 255))
        surface.blit(mr_title, mr_title.get_rect(center=(cx, mr_top + 20)))

        # Buffer status
        bs_text = self._font_small.render(
            f"Fighter buffer: {f_buf:,}   Archer buffer: {a_buf:,}", True, (150, 140, 170))
        surface.blit(bs_text, bs_text.get_rect(center=(cx, mr_top + 48)))

        # Iterations selector
        cost_total = self._replay_cost()
        iter_text  = self._font_stat.render(
            f"Iterations: {self._replay_iters:,}   Cost: {cost_total:,} coins",
            True, (220, 200, 255))
        surface.blit(iter_text, iter_text.get_rect(center=(cx, mr_top + 76)))

        btn_y = mr_top + 106
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

        # Progress bar (while training), saving indicator, or result text (when done)
        if self._replay_running:
            bar_w = 280
            bar_h = 10
            bar_x = cx - bar_w // 2
            bar_y = btn_y + 46
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
            surface.blit(sv, sv.get_rect(center=(cx, btn_y + 48)))
        elif self._replay_result_lines:
            for j, line in enumerate(self._replay_result_lines):
                rs = self._font_small.render(line, True, (180, 220, 180))
                surface.blit(rs, rs.get_rect(center=(cx, btn_y + 44 + j * 18)))

        # Start battle button
        bb_y     = btn_y + 74
        btn_rect = pygame.Rect(cx - 150, bb_y, 300, 48)
        pygame.draw.rect(surface, (25, 70, 30), btn_rect, border_radius=8)
        pygame.draw.rect(surface, (60, 180, 70), btn_rect, 2, border_radius=8)
        btn_t = self._font_btn.render("[B] Start Battle", True, (140, 255, 150))
        surface.blit(btn_t, btn_t.get_rect(center=btn_rect.center))
        self._click_rects.append((btn_rect, "battle"))
