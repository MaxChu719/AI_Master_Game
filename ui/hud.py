from __future__ import annotations
"""
HUD — battle scene overlay.

Additions vs. original:
  - Wave display: "Wave X/100" + BOSS WAVE banner.
  - MP bar + two spell icons (Healing / Fireball) with cooldown overlays.
  - Multi-minion right panel (lists all fighters and archers).
  - HUD now exposes hit_test_spell_panel() so BattleScene can detect spell clicks.
  - DQN panel shows Rainbow info (no epsilon, shows noise/steps).
"""
import math
import pygame
from systems.wave_system import WaveState
from config import CFG

_SPAWN_CFG = CFG.get("spawning", {})
_MAX_WAVES = int(CFG["wave"]["max_waves"])
_SP        = CFG["spells"]
_MP_CFG    = CFG["mp"]
_SUMMON_CFG = {
    "fighter":   float(_SP.get("summon_fighter",   {}).get("mp_cost", 50)),
    "archer":    float(_SP.get("summon_archer",    {}).get("mp_cost", 40)),
    "fire_mage": float(_SP.get("summon_fire_mage", {}).get("mp_cost", 60)),
    "ice_mage":  float(_SP.get("summon_ice_mage",  {}).get("mp_cost", 55)),
}


class HUD:
    def __init__(self, scene):
        self.scene = scene
        self.font_wave    = pygame.font.SysFont("arial", 24)
        self.font_coins   = pygame.font.SysFont("arial", 24)
        self.font_ai_hdr  = pygame.font.SysFont("arial", 20)
        self.font_ai      = pygame.font.SysFont("arial", 18)
        self.font_enemies = pygame.font.SysFont("arial", 20)
        self.font_big     = pygame.font.SysFont("arial", 48)
        self.font_sub     = pygame.font.SysFont("arial", 24)
        self.font_small   = pygame.font.SysFont("arial", 18)
        self.font_overlay = pygame.font.SysFont("arial", 36, bold=True)
        self.font_ctrl    = pygame.font.SysFont("arial", 16)
        self.font_boss    = pygame.font.SysFont("arial", 30, bold=True)
        self.font_spell   = pygame.font.SysFont("arial", 13, bold=True)

        self._ctrl_rects  = []   # [(pygame.Rect, key)]
        self._spell_rects = []   # [(pygame.Rect, spell_name)]

    # ── Master draw ─────────────────────────��─────────────────────────────

    def draw(self, surface: pygame.Surface):
        sw, sh = surface.get_size()
        state = self.scene.wave_system.state

        self._draw_wave_banner(surface, sw)
        self._draw_coins(surface, sw)
        self._draw_control_panel(surface, sw)
        self._draw_ai_panel(surface, sh)
        self._draw_mp_and_spells(surface, sw, sh)
        self._draw_right_panel(surface, sw, sh)

        if state == WaveState.INTERMISSION:
            self._draw_intermission(surface, sw, sh)
        elif state == WaveState.GAME_OVER:
            self._draw_game_over(surface, sw, sh)
        elif state == WaveState.VICTORY:
            self._draw_victory(surface, sw, sh)

        if self.scene.paused and state not in (WaveState.GAME_OVER, WaveState.VICTORY):
            self._draw_paused(surface, sw, sh)

        if self.scene.brain_reset_timer > 0:
            self._draw_brain_reset(surface, sw, sh)

        if self.scene.spell_mode:
            self._draw_spell_hint(surface, sw, sh)

    # ── Wave banner ─────���─────────────────────────────��───────────────────

    def _draw_wave_banner(self, surface, sw):
        wave_num = self.scene.wave_system.wave_number
        is_boss  = self.scene.wave_system.is_boss_wave and \
                   self.scene.wave_system.state in (WaveState.SPAWNING, WaveState.ACTIVE)

        if is_boss:
            col = (255, 80, 80)
            txt = f"★ BOSS WAVE {wave_num}/{_MAX_WAVES} ★"
        else:
            col = (255, 255, 255)
            txt = f"Wave {wave_num}/{_MAX_WAVES}"

        surf = self.font_wave.render(txt, True, col)
        surface.blit(surf, (16, 16))

    def _draw_coins(self, surface, sw):
        coins = self.scene.game_manager.coins
        name  = self.scene.game_manager.player_name
        label = f"{name}   Coins: {coins}" if name else f"Coins: {coins}"
        surf  = self.font_coins.render(label, True, (255, 215, 0))
        surface.blit(surf, (sw - surf.get_width() - 16, 16))

    # ── Control panel ───────────────────────────��─────────────────────────

    def _draw_control_panel(self, surface, sw):
        paused = self.scene.paused
        speed  = self.scene.speed_multiplier
        segments = [
            ("[P] " + ("Resume" if paused else "Pause"),
             (100, 255, 100) if paused else (200, 200, 200), pygame.K_p),
            ("[R] Reset Brain", (200, 200, 200), pygame.K_r),
            (f"[+/-] Speed: {speed}x", (160, 200, 255), pygame.K_PLUS),
            ("[ESC] Menu", (200, 200, 200), pygame.K_ESCAPE),
        ]
        gap     = 18
        total_w = sum(self.font_ctrl.size(t)[0] for t, _, _k in segments) + gap * (len(segments) - 1)
        x       = (sw - total_w) // 2
        y       = 14
        pad_x, pad_y = 10, 5
        bg = pygame.Surface((total_w + pad_x * 2, self.font_ctrl.get_height() + pad_y * 2),
                             pygame.SRCALPHA)
        bg.fill((0, 0, 0, 130))
        surface.blit(bg, (x - pad_x, y - pad_y))
        self._ctrl_rects = []
        for text, color, key in segments:
            s = self.font_ctrl.render(text, True, color)
            r = pygame.Rect(x, y - pad_y, s.get_width(), self.font_ctrl.get_height() + pad_y * 2)
            self._ctrl_rects.append((r, key))
            surface.blit(s, (x, y))
            x += s.get_width() + gap

    def hit_test_control_panel(self, pos):
        for rect, key in self._ctrl_rects:
            if rect.collidepoint(pos):
                return key
        return None

    # ── AI / DQN panel ────────────────────────────────────────────────────

    @staticmethod
    def _agent_mode_loss(agent, steps, loss_val, buf) -> tuple[str, str]:
        """Return (mode_str, loss_str) for an agent."""
        preset_only = agent.preset_only if agent else False
        min_buf     = agent.min_buffer_size if agent else 200
        if preset_only:
            if steps > 0:
                return "Preset+Train", f"{loss_val:.4f}"
            elif buf > 0:
                return "Preset+Warmup", f"{buf}/{min_buf}"
            else:
                return "Preset+Train", "N/A"
        else:
            if steps > 0:
                return "DQN", f"{loss_val:.4f}"
            elif buf > 0:
                return "Warmup", f"{buf}/{min_buf}"
            else:
                return "DQN", "N/A"

    def _draw_ai_panel(self, surface, sh):
        fa  = self.scene.fighter_agent
        aa  = self.scene.archer_agent
        fma = getattr(self.scene, "fire_mage_agent", None)
        ima = getattr(self.scene, "ice_mage_agent",  None)

        # ── Per-agent stats ────────────────────────────────────────────────
        f_steps  = self.scene.latest_steps
        f_buf    = getattr(self.scene, "latest_buffer_size",        0)
        f_loss   = getattr(self.scene, "latest_loss",               0.0)
        f_buf_max = fa.buffer_size if fa else 1
        f_mode_str, f_loss_str = self._agent_mode_loss(fa, f_steps, f_loss, f_buf)

        a_steps  = getattr(self.scene, "latest_archer_steps",       0)
        a_buf    = getattr(self.scene, "latest_archer_buffer_size",  0)
        a_loss   = getattr(self.scene, "latest_archer_loss",         0.0)
        a_buf_max = aa.buffer_size if aa else 1
        a_mode_str, a_loss_str = self._agent_mode_loss(aa, a_steps, a_loss, a_buf)

        fm_steps = getattr(self.scene, "latest_fm_steps",           0)
        fm_buf   = getattr(self.scene, "latest_fm_buffer_size",      0)
        fm_loss  = getattr(self.scene, "latest_fm_loss",             0.0)
        fm_buf_max = fma.buffer_size if fma else 1
        fm_mode_str, fm_loss_str = self._agent_mode_loss(fma, fm_steps, fm_loss, fm_buf)

        im_steps = getattr(self.scene, "latest_im_steps",           0)
        im_buf   = getattr(self.scene, "latest_im_buffer_size",      0)
        im_loss  = getattr(self.scene, "latest_im_loss",             0.0)
        im_buf_max = ima.buffer_size if ima else 1
        im_mode_str, im_loss_str = self._agent_mode_loss(ima, im_steps, im_loss, im_buf)

        f_avg_rew  = getattr(self.scene, "latest_avg_reward",           0.0)
        a_avg_rew  = getattr(self.scene, "latest_archer_avg_reward",    0.0)
        fm_avg_rew = getattr(self.scene, "latest_fm_avg_reward",        0.0)
        im_avg_rew = getattr(self.scene, "latest_im_avg_reward",        0.0)
        speed      = self.scene.speed_multiplier

        f_buf_pct  = int(100 * f_buf  / max(1, f_buf_max))
        a_buf_pct  = int(100 * a_buf  / max(1, a_buf_max))
        fm_buf_pct = int(100 * fm_buf / max(1, fm_buf_max))
        im_buf_pct = int(100 * im_buf / max(1, im_buf_max))

        f_mode_col  = (120, 200, 120) if (fa  and fa.preset_only)  else (160, 200, 255)
        a_mode_col  = (120, 200, 120) if (aa  and aa.preset_only)  else (100, 220, 120)
        fm_mode_col = (120, 200, 120) if (fma and fma.preset_only) else (255, 160, 80)
        im_mode_col = (120, 200, 120) if (ima and ima.preset_only) else (80,  180, 255)

        lines = [
            (f"Fighter Brain [{f_mode_str}]",             self.font_ai_hdr, f_mode_col),
            (f"Loss:{f_loss_str} Steps:{f_steps}",        self.font_ai, (180, 180, 180)),
            (f"Rew:{f_avg_rew:.3f} Buf:{f_buf_pct}%",    self.font_ai, (160, 160, 160)),
            (f"Archer Brain [{a_mode_str}]",               self.font_ai_hdr, a_mode_col),
            (f"Loss:{a_loss_str} Steps:{a_steps}",        self.font_ai, (180, 180, 180)),
            (f"Rew:{a_avg_rew:.3f} Buf:{a_buf_pct}%",    self.font_ai, (160, 160, 160)),
            (f"FireMage Brain [{fm_mode_str}]",            self.font_ai_hdr, fm_mode_col),
            (f"Loss:{fm_loss_str} Steps:{fm_steps}",      self.font_ai, (180, 180, 180)),
            (f"Rew:{fm_avg_rew:.3f} Buf:{fm_buf_pct}%",  self.font_ai, (160, 160, 160)),
            (f"IceMage Brain [{im_mode_str}]",             self.font_ai_hdr, im_mode_col),
            (f"Loss:{im_loss_str} Steps:{im_steps}",      self.font_ai, (180, 180, 180)),
            (f"Rew:{im_avg_rew:.3f} Buf:{im_buf_pct}%",  self.font_ai, (160, 160, 160)),
            (f"Speed: {speed}x",                           self.font_ai, (180, 180, 180)),
        ]
        if getattr(self.scene, "_saving", False):
            lines.append(("Saving...", self.font_ai, (255, 200, 60)))
        line_h  = 19
        panel_w = 260
        panel_h = len(lines) * line_h + 10
        panel_x = 10
        panel_y = sh - panel_h - 10

        bg = pygame.Surface((panel_w, panel_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 150))
        surface.blit(bg, (panel_x, panel_y))

        y = panel_y + 5
        for text, font, color in lines:
            s = font.render(text, True, color)
            surface.blit(s, (panel_x + 6, y))
            y += line_h


    # ── MP bar + spell icons ──────────────────────────────────────────────

    def _draw_mp_and_spells(self, surface, sw, sh):
        """
        Two-row spell panel at the bottom centre of the screen.
        Row 1: MP bar (full panel width).
        Row 2: Four spell icons — [Heal] [Fireball] [Summon F] [Summon A].
        Summon spells cost MP and are only available during an active wave
        when the minion cap has not been reached.
        """
        self._spell_rects = []

        scene  = self.scene
        mp     = scene.mp
        max_mp = scene.max_mp
        heal_cd = scene._heal_cd
        fb_cd   = scene._fb_cd
        am = scene.game_manager.save_data.get("ai_master", {}) \
             if scene.game_manager.save_data else {}

        from scenes.battle import _compute_heal_stats, _compute_fireball_stats
        _, _, heal_base_cd = _compute_heal_stats(am)
        _, _, fb_base_cd   = _compute_fireball_stats(am)
        heal_cost = int(_SP["healing"]["mp_cost"])
        fb_cost   = int(_SP["fireball"]["mp_cost"])
        sf_cost   = int(_SUMMON_CFG["fighter"])
        sa_cost   = int(_SUMMON_CFG["archer"])
        sfm_cost  = int(_SUMMON_CFG["fire_mage"])
        sim_cost  = int(_SUMMON_CFG["ice_mage"])

        # Wave / cap availability for summon icons (global shared cap)
        from systems.wave_system import WaveState as _WS
        wave_active  = scene.wave_system.state == _WS.ACTIVE
        _default_caps = _SPAWN_CFG.get("deployment_caps_global", [2, 5, 8, 12, 16, 20])
        global_cap   = getattr(scene, "spawn_cap_total", int(_default_caps[-1]))
        cur_f        = len(scene.fighters)
        cur_a        = len(scene.archers)
        cur_fm       = len(getattr(scene, "fire_mages", []))
        cur_im       = len(getattr(scene, "ice_mages",  []))
        total_minions = cur_f + cur_a + cur_fm + cur_im
        summon_f_ok  = wave_active and total_minions < global_cap
        summon_a_ok  = wave_active and total_minions < global_cap
        summon_fm_ok = wave_active and total_minions < global_cap
        summon_im_ok = wave_active and total_minions < global_cap

        # Layout constants — 6 icons wide
        cx      = sw // 2
        icon_sz = 44
        gap     = 8
        bar_h   = 18
        bar_w   = 6 * icon_sz + 5 * gap
        bar_x   = cx - bar_w // 2

        icon_row_y = sh - icon_sz - 8
        bar_y      = icon_row_y - bar_h - 6
        icon_row_x = bar_x

        # Background panel
        pad = 6
        bg_x = bar_x - pad
        bg_y = bar_y - pad
        bg_w = bar_w + pad * 2
        bg_h = bar_h + 6 + icon_sz + pad * 2
        bg = pygame.Surface((bg_w, bg_h), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 120))
        surface.blit(bg, (bg_x, bg_y))

        # ── Row 1: MP bar ─────────────────────────────────────────────────
        pygame.draw.rect(surface, (20, 20, 60),
                         (bar_x, bar_y, bar_w, bar_h), border_radius=6)
        fill_w = int(bar_w * mp / max(1, max_mp))
        if fill_w > 0:
            pygame.draw.rect(surface, (80, 120, 255),
                             (bar_x, bar_y, fill_w, bar_h), border_radius=6)
        pygame.draw.rect(surface, (100, 140, 255),
                         (bar_x, bar_y, bar_w, bar_h), 2, border_radius=6)
        mp_txt = self.font_spell.render(f"MP  {int(mp)} / {int(max_mp)}", True, (180, 200, 255))
        surface.blit(mp_txt, mp_txt.get_rect(center=(cx, bar_y + bar_h // 2)))

        # ── Row 2: Six spell icons ────────────────────────────────────────
        spell_defs = [
            ("healing",          "Heal",      heal_cost, heal_cd, heal_base_cd,
             (50, 200, 100),  True),
            ("fireball",         "Fireball",  fb_cost,   fb_cd,  fb_base_cd,
             (220, 80, 20),   True),
            ("summon_fighter",   "Summon F",  sf_cost,   0.0,    1.0,
             (80, 130, 230),  summon_f_ok),
            ("summon_archer",    "Summon A",  sa_cost,   0.0,    1.0,
             (60, 200, 90),   summon_a_ok),
            ("summon_fire_mage", "Summon FM", sfm_cost,  0.0,    1.0,
             (220, 80, 20),   summon_fm_ok),
            ("summon_ice_mage",  "Summon IM", sim_cost,  0.0,    1.0,
             (60, 180, 255),  summon_im_ok),
        ]

        for i, (name, label, cost, cd, max_cd, color, extra_cond) in enumerate(spell_defs):
            ix = icon_row_x + i * (icon_sz + gap)
            self._draw_spell_icon(
                surface, ix, icon_row_y, icon_sz,
                name=name, label=label, icon_color=color,
                cost=cost, cooldown=cd, max_cooldown=max_cd,
                mp=mp, spell_mode=scene.spell_mode,
                extra_cond=extra_cond,
            )
            self._spell_rects.append((pygame.Rect(ix, icon_row_y, icon_sz, icon_sz), name))

    def _draw_spell_icon(self, surface, x, y, sz, name, label, icon_color,
                         cost, cooldown, max_cooldown, mp, spell_mode,
                         extra_cond=True):
        selected  = (spell_mode == name)
        available = (mp >= cost and cooldown <= 0) and extra_cond

        # Background colour per spell family
        _BG = {
            "healing":          (30, 55, 35),
            "fireball":         (55, 25, 15),
            "summon_fighter":   (20, 30, 60),
            "summon_archer":    (20, 50, 25),
            "summon_fire_mage": (55, 18, 10),
            "summon_ice_mage":  (10, 28, 60),
        }
        bg_col = _BG.get(name, (30, 30, 50))
        bord   = icon_color if available else (55, 55, 70)
        if selected:
            bord = (255, 255, 100)
        pygame.draw.rect(surface, bg_col, (x, y, sz, sz), border_radius=8)
        pygame.draw.rect(surface, bord,   (x, y, sz, sz), 2, border_radius=8)

        cx_i, cy_i = x + sz // 2, y + sz // 2
        c_on  = icon_color
        c_off = tuple(max(0, v // 3) for v in icon_color)

        if name == "healing":
            c = c_on if available else c_off
            # Glowing cross
            pygame.draw.rect(surface, c, (cx_i - 3, cy_i - 12, 6, 24), border_radius=2)
            pygame.draw.rect(surface, c, (cx_i - 12, cy_i - 3, 24, 6), border_radius=2)
            if available:
                t = pygame.time.get_ticks() / 800.0
                for k in range(4):
                    a  = k * math.pi / 2 + t
                    sx = cx_i + int(math.cos(a) * 16)
                    sy = cy_i + int(math.sin(a) * 16)
                    pygame.draw.circle(surface, (150, 255, 180), (sx, sy), 2)

        elif name == "fireball":
            c = c_on if available else c_off
            # Fireball orb + flame tongues
            pygame.draw.circle(surface, c, (cx_i, cy_i + 3), 11)
            pygame.draw.circle(surface, (255, 200, 50) if available else (60, 50, 30),
                                (cx_i, cy_i - 1), 7)
            if available:
                t = pygame.time.get_ticks() / 600.0
                for k in range(3):
                    a  = -math.pi / 2 + (k - 1) * 0.5 + math.sin(t + k) * 0.3
                    ex = cx_i + int(math.cos(a) * 18)
                    ey = cy_i + int(math.sin(a) * 18)
                    pygame.draw.line(surface, (255, 150, 20), (cx_i, cy_i - 1), (ex, ey), 2)

        elif name == "summon_fighter":
            c = c_on if available else c_off
            # Sword: blade + crossguard + pommel
            pygame.draw.rect(surface, c,           (cx_i - 2, cy_i - 13, 4, 18), border_radius=1)
            pygame.draw.rect(surface, c,           (cx_i - 9, cy_i + 2,  18, 3), border_radius=1)
            pygame.draw.circle(surface, c,         (cx_i, cy_i + 9), 3)
            # Glowing tip when available
            if available:
                tip_col = (160, 210, 255)
                pygame.draw.circle(surface, tip_col, (cx_i, cy_i - 13), 3)
                t = pygame.time.get_ticks() / 700.0
                pulse = int(abs(math.sin(t)) * 80)
                pygame.draw.circle(surface, (80 + pulse, 120 + pulse, 255),
                                   (cx_i, cy_i - 13), 5, 1)
            # Count badge: shows "F:n | total/cap" to reflect shared global pool
            scene = self.scene
            _caps = _SPAWN_CFG.get("deployment_caps_global", [2, 5, 8, 12, 16, 20])
            gcap  = getattr(scene, "spawn_cap_total", int(_caps[-1]))
            cur_f  = len(scene.fighters)
            cur_a  = len(scene.archers)
            cur_fm = len(getattr(scene, "fire_mages", []))
            cur_im = len(getattr(scene, "ice_mages",  []))
            total  = cur_f + cur_a + cur_fm + cur_im
            badge = self.font_spell.render(
                f"F:{cur_f}  {total}/{gcap}", True,
                (160, 200, 255) if available else (70, 70, 90))
            surface.blit(badge, badge.get_rect(midbottom=(cx_i, y + sz - 1)))

        elif name == "summon_archer":
            c = c_on if available else c_off
            # Bow arc + arrow
            bow_rect = pygame.Rect(cx_i - 6, cy_i - 13, 12, 26)
            pygame.draw.arc(surface, c, bow_rect, -math.pi / 2.2, math.pi / 2.2, 2)
            # Arrow shaft + head
            pygame.draw.line(surface, c, (cx_i - 13, cy_i), (cx_i + 10, cy_i), 2)
            pygame.draw.polygon(surface, c,
                                [(cx_i + 10, cy_i),
                                 (cx_i + 5,  cy_i - 3),
                                 (cx_i + 5,  cy_i + 3)])
            if available:
                t = pygame.time.get_ticks() / 700.0
                pulse = int(abs(math.sin(t)) * 80)
                tip = (60, 200 + pulse // 2, 80 + pulse)
                pygame.draw.circle(surface, tip, (cx_i + 10, cy_i), 3)
            # Count badge
            scene = self.scene
            _caps = _SPAWN_CFG.get("deployment_caps_global", [2, 5, 8, 12, 16, 20])
            gcap  = getattr(scene, "spawn_cap_total", int(_caps[-1]))
            cur_f  = len(scene.fighters)
            cur_a  = len(scene.archers)
            cur_fm = len(getattr(scene, "fire_mages", []))
            cur_im = len(getattr(scene, "ice_mages",  []))
            total  = cur_f + cur_a + cur_fm + cur_im
            badge = self.font_spell.render(
                f"A:{cur_a}  {total}/{gcap}", True,
                (100, 220, 130) if available else (70, 90, 70))
            surface.blit(badge, badge.get_rect(midbottom=(cx_i, y + sz - 1)))

        elif name == "summon_fire_mage":
            c = c_on if available else c_off
            # Fire orb with flame lines
            pygame.draw.circle(surface, c, (cx_i, cy_i + 2), 9)
            pygame.draw.circle(surface, (255, 200, 60) if available else (60, 50, 20),
                               (cx_i, cy_i - 1), 5)
            if available:
                t = pygame.time.get_ticks() / 500.0
                for k in range(4):
                    a  = k * math.pi / 2 + t
                    ex = cx_i + int(math.cos(a) * 14)
                    ey = cy_i + int(math.sin(a) * 14)
                    pygame.draw.line(surface, (255, 120, 20), (cx_i, cy_i - 1), (ex, ey), 2)
            scene = self.scene
            _caps = _SPAWN_CFG.get("deployment_caps_global", [2, 5, 8, 12, 16, 20])
            gcap  = getattr(scene, "spawn_cap_total", int(_caps[-1]))
            cur_f  = len(scene.fighters)
            cur_a  = len(scene.archers)
            cur_fm = len(getattr(scene, "fire_mages", []))
            cur_im = len(getattr(scene, "ice_mages",  []))
            total  = cur_f + cur_a + cur_fm + cur_im
            badge  = self.font_spell.render(
                f"FM:{cur_fm} {total}/{gcap}", True,
                (255, 160, 60) if available else (80, 50, 30))
            surface.blit(badge, badge.get_rect(midbottom=(cx_i, y + sz - 1)))

        elif name == "summon_ice_mage":
            c = c_on if available else c_off
            # Ice crystal diamond
            pygame.draw.circle(surface, c, (cx_i, cy_i), 9)
            hs = 5
            pts = [(cx_i, cy_i - hs - 4), (cx_i + hs, cy_i),
                   (cx_i, cy_i + hs + 4), (cx_i - hs, cy_i)]
            pygame.draw.polygon(surface, c, pts)
            pygame.draw.polygon(surface, (200, 240, 255) if available else (50, 60, 80),
                               pts, 1)
            if available:
                t = pygame.time.get_ticks() / 700.0
                pulse = int(abs(math.sin(t)) * 60)
                pygame.draw.circle(surface, (80, 180, min(255, 200 + pulse)), (cx_i, cy_i), 11, 1)
            scene = self.scene
            _caps = _SPAWN_CFG.get("deployment_caps_global", [2, 5, 8, 12, 16, 20])
            gcap  = getattr(scene, "spawn_cap_total", int(_caps[-1]))
            cur_f  = len(scene.fighters)
            cur_a  = len(scene.archers)
            cur_fm = len(getattr(scene, "fire_mages", []))
            cur_im = len(getattr(scene, "ice_mages",  []))
            total  = cur_f + cur_a + cur_fm + cur_im
            badge  = self.font_spell.render(
                f"IM:{cur_im} {total}/{gcap}", True,
                (100, 200, 255) if available else (30, 50, 80))
            surface.blit(badge, badge.get_rect(midbottom=(cx_i, y + sz - 1)))

        # MP cost label at top (skip badge-label spells' bottom)
        cost_s = self.font_spell.render(f"{cost}MP", True,
                                        (160, 200, 255) if available else (70, 70, 90))
        surface.blit(cost_s, cost_s.get_rect(midtop=(cx_i, y + 2)))

        # Spell name label at bottom for Heal / Fireball
        if name in ("healing", "fireball"):
            label_s = self.font_spell.render(label, True,
                                             (220, 220, 220) if available else (90, 90, 90))
            surface.blit(label_s, label_s.get_rect(midbottom=(cx_i, y + sz - 1)))

        # Cooldown overlay
        if cooldown > 0:
            overlay = pygame.Surface((sz, sz), pygame.SRCALPHA)
            ratio   = cooldown / max(0.01, max_cooldown)
            overlay.fill((0, 0, 0, int(160 * ratio)))
            surface.blit(overlay, (x, y))
            cd_s = self.font_spell.render(f"{cooldown:.1f}s", True, (200, 200, 200))
            surface.blit(cd_s, cd_s.get_rect(center=(cx_i, cy_i)))

    def hit_test_spell_panel(self, pos) -> str | None:
        """Return spell name if pos hits a spell icon, else None."""
        for rect, spell in self._spell_rects:
            if rect.collidepoint(pos):
                return spell
        return None

    # ── Right panel (minion status) ───────────────────────────────────────

    def _draw_right_panel(self, surface, sw, sh):
        alive_swarms = sum(1 for e in self.scene.enemies if e.is_alive)
        boss         = self.scene.wave_system.boss
        boss_alive   = (boss is not None and boss.is_alive)

        enemy_str = f"Enemies: {alive_swarms}"
        if boss_alive:
            enemy_str += f"  +BOSS {boss.hp}/{boss.max_hp}"
        enemy_surf = self.font_enemies.render(enemy_str, True,
                                               (255, 80, 80) if boss_alive else (255, 255, 255))
        ey = sh - enemy_surf.get_height() - 92   # above two-row spell panel
        surface.blit(enemy_surf, (sw - enemy_surf.get_width() - 16, ey))

        stack_y = ey - 4
        fire_mages = list(reversed(getattr(self.scene, "fire_mages", [])))
        ice_mages  = list(reversed(getattr(self.scene, "ice_mages",  [])))
        all_minions = list(reversed(self.scene.archers + self.scene.fighters)) + fire_mages + ice_mages
        fighters_set   = set(self.scene.fighters)
        archers_set    = set(self.scene.archers)
        fire_mages_set = set(getattr(self.scene, "fire_mages", []))
        for m in all_minions:
            if m in fighters_set:
                role_lbl  = "F"
                alive_col = (100, 160, 255)
            elif m in archers_set:
                role_lbl  = "A"
                alive_col = (100, 220, 120)
            elif m in fire_mages_set:
                role_lbl  = "FM"
                alive_col = (255, 140, 40)
            else:
                role_lbl  = "IM"
                alive_col = (80, 180, 255)

            if m.is_alive:
                if role_lbl in ("FM", "IM"):
                    txt = f"{role_lbl}: {m.hp}/{m.max_hp}HP  {m.stamina:.0f}MP"
                else:
                    txt = f"{role_lbl}: {m.hp}/{m.max_hp} HP"
                lbl_c = alive_col
            else:
                txt   = f"{role_lbl}: DEAD"
                lbl_c = (160, 80, 80)
            s = self.font_small.render(txt, True, lbl_c)
            stack_y -= s.get_height() + 4
            surface.blit(s, (sw - s.get_width() - 16, stack_y))

    # ── Overlays ──────────────────────────���──────────────────────────���────

    def _draw_intermission(self, surface, sw, sh):
        cx, cy  = sw // 2, sh // 2
        wave    = self.scene.wave_system.wave_number
        cd      = self.scene.wave_system.intermission_seconds_left
        is_boss = self.scene.wave_system.is_boss_wave
        col     = (255, 80, 80) if is_boss else (255, 240, 100)
        prefix  = "★ BOSS " if is_boss else ""
        txt     = self.font_big.render(f"{prefix}Wave {wave} in {cd}…", True, col)
        surface.blit(txt, txt.get_rect(center=(cx, cy)))

    def _draw_game_over(self, surface, sw, sh):
        cx, cy   = sw // 2, sh // 2
        wave_num = self.scene.wave_system.wave_number
        title    = self.font_big.render(f"GAME OVER — Reached Wave {wave_num}", True, (220, 60, 60))
        surface.blit(title, title.get_rect(center=(cx, cy - 80)))
        stats = [
            f"Waves Survived: {self.scene.waves_survived}",
            f"Enemies Killed: {self.scene.total_kills}",
            f"Damage Dealt: {int(self.scene.total_damage_dealt)}",
            f"Fighter Steps: {self.scene.latest_steps}",
            f"Archer Steps: {self.scene.latest_archer_steps}",
        ]
        for i, line in enumerate(stats):
            s = self.font_small.render(line, True, (200, 160, 160))
            surface.blit(s, s.get_rect(center=(cx, cy - 10 + i * 26)))
        if self.scene._session_saving:
            sv = self.font_sub.render("Saving session...", True, (255, 200, 60))
            surface.blit(sv, sv.get_rect(center=(cx, cy + 140)))
        else:
            sub = self.font_sub.render("Press ENTER to return to Research Lab", True, (200, 160, 160))
            surface.blit(sub, sub.get_rect(center=(cx, cy + 140)))

    def _draw_victory(self, surface, sw, sh):
        cx, cy = sw // 2, sh // 2
        coins  = self.scene.game_manager.coins
        title  = self.font_big.render("VICTORY — All 100 waves cleared!", True, (255, 215, 0))
        surface.blit(title, title.get_rect(center=(cx, cy - 80)))
        stats = [
            f"Total Coins: {coins}",
            f"Enemies Killed: {self.scene.total_kills}",
            f"Damage Dealt: {int(self.scene.total_damage_dealt)}",
            f"Fighter Steps: {self.scene.latest_steps}",
            f"Archer Steps: {self.scene.latest_archer_steps}",
        ]
        for i, line in enumerate(stats):
            s = self.font_small.render(line, True, (255, 230, 100))
            surface.blit(s, s.get_rect(center=(cx, cy - 10 + i * 26)))
        if self.scene._session_saving:
            sv = self.font_sub.render("Saving session...", True, (255, 200, 60))
            surface.blit(sv, sv.get_rect(center=(cx, cy + 140)))
        else:
            sub = self.font_sub.render("Press ENTER to return to Research Lab", True, (200, 200, 160))
            surface.blit(sub, sub.get_rect(center=(cx, cy + 140)))

    def _draw_paused(self, surface, sw, sh):
        overlay = pygame.Surface((sw, sh), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        surface.blit(overlay, (0, 0))
        text = self.font_overlay.render("PAUSED", True, (255, 255, 100))
        surface.blit(text, text.get_rect(center=(sw // 2, sh // 2)))

    def _draw_brain_reset(self, surface, sw, sh):
        alpha = min(255, int(self.scene.brain_reset_timer / 2.0 * 255))
        text  = self.font_overlay.render("BRAIN RESET", True, (255, 80, 80))
        surf  = pygame.Surface(text.get_size(), pygame.SRCALPHA)
        surf.blit(text, (0, 0))
        surf.set_alpha(alpha)
        surface.blit(surf, surf.get_rect(center=(sw // 2, 60)))

    def _draw_spell_hint(self, surface, sw, sh):
        mode = self.scene.spell_mode
        _cancel = "  (Right-click or ESC to cancel)"
        if mode == "healing":
            txt = "Click on the arena to place Healing Circle" + _cancel
            col = (80, 255, 120)
        elif mode == "fireball":
            txt = "Click on the arena to launch Fireball Meteor" + _cancel
            col = (255, 160, 40)
        elif mode == "summon_fighter":
            txt = "Click on the arena to place Fighter Summon Portal" + _cancel
            col = (100, 160, 255)
        elif mode == "summon_archer":
            txt = "Click on the arena to place Archer Summon Portal" + _cancel
            col = (100, 255, 140)
        elif mode == "summon_fire_mage":
            txt = "Click on the arena to place Fire Mage Summon Portal" + _cancel
            col = (255, 140, 60)
        elif mode == "summon_ice_mage":
            txt = "Click on the arena to place Ice Mage Summon Portal" + _cancel
            col = (80, 200, 255)
        else:
            return
        s   = self.font_small.render(txt, True, col)
        bg  = pygame.Surface((s.get_width() + 20, s.get_height() + 8), pygame.SRCALPHA)
        bg.fill((0, 0, 0, 160))
        bx  = sw // 2 - bg.get_width() // 2
        by  = sh - 122
        surface.blit(bg, (bx, by))
        surface.blit(s, (bx + 10, by + 4))
