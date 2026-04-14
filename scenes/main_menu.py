import pygame
from engine.scene import BaseScene

_STATE_MAIN     = "MAIN"
_STATE_NAME     = "NAME_INPUT"
_STATE_CONFIRM  = "CONFIRM_OVERRIDE"
_STATE_LOAD     = "LOAD_SELECT"

_MAIN_ITEMS = ["Start New Game", "Load Saved Game", "Quit"]


class MainMenuScene(BaseScene):
    def __init__(self, game_manager):
        super().__init__(game_manager)
        self._font_title = pygame.font.SysFont("arial", 72)
        self._font_item  = pygame.font.SysFont("arial", 36)
        self._font_input = pygame.font.SysFont("arial", 32)
        self._font_sub   = pygame.font.SysFont("arial", 22)

        self._state    = _STATE_MAIN
        self._selected = 0          # main menu cursor

        self._name_input    = ""    # in-progress typed name
        self._pending_name  = ""    # confirmed name awaiting override check

        self._saves          = []   # list of save names for load screen
        self._save_selected  = 0

        # Cursor blink
        self._cursor_timer = 0.0
        self._cursor_visible = True

        # Clickable rects built each draw call
        self._item_rects = []   # [(rect, index)] for MAIN and LOAD states

    # ------------------------------------------------------------------

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            if self._state == _STATE_MAIN:
                self._on_main(event)
            elif self._state == _STATE_NAME:
                self._on_name(event)
            elif self._state == _STATE_CONFIRM:
                self._on_confirm(event)
            elif self._state == _STATE_LOAD:
                self._on_load(event)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self._on_click(event.pos)

    def _on_click(self, pos):
        for rect, idx in self._item_rects:
            if not rect.collidepoint(pos):
                continue
            if self._state == _STATE_MAIN:
                if idx == self._selected:
                    # Clicking already-highlighted item activates it
                    fake = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)
                    self._on_main(fake)
                else:
                    self._selected = idx
            elif self._state == _STATE_LOAD:
                if idx == self._save_selected:
                    fake = pygame.event.Event(pygame.KEYDOWN, key=pygame.K_RETURN)
                    self._on_load(fake)
                else:
                    self._save_selected = idx
            break

    # ── State handlers ─────────────────────────────────────────────────

    def _on_main(self, ev):
        if ev.key == pygame.K_UP:
            self._selected = (self._selected - 1) % len(_MAIN_ITEMS)
        elif ev.key == pygame.K_DOWN:
            self._selected = (self._selected + 1) % len(_MAIN_ITEMS)
        elif ev.key == pygame.K_RETURN:
            item = _MAIN_ITEMS[self._selected]
            if item == "Start New Game":
                self._name_input = ""
                self._state = _STATE_NAME
            elif item == "Load Saved Game":
                self._saves = self.game_manager.list_saves()
                self._save_selected = 0
                self._state = _STATE_LOAD
            elif item == "Quit":
                self.game_manager.running = False
        elif ev.key == pygame.K_ESCAPE:
            self.game_manager.running = False

    def _on_name(self, ev):
        if ev.key == pygame.K_ESCAPE:
            self._state = _STATE_MAIN
        elif ev.key == pygame.K_RETURN:
            name = self._name_input.strip()
            if not name:
                return
            if self.game_manager.save_exists(name):
                self._pending_name = name
                self._state = _STATE_CONFIRM
            else:
                self._begin_new_game(name)
        elif ev.key == pygame.K_BACKSPACE:
            self._name_input = self._name_input[:-1]
        elif ev.unicode and ev.unicode.isprintable() and len(self._name_input) < 20:
            self._name_input += ev.unicode

    def _on_confirm(self, ev):
        if ev.key in (pygame.K_y, pygame.K_RETURN):
            self._begin_new_game(self._pending_name)
        elif ev.key in (pygame.K_n, pygame.K_ESCAPE):
            # Return to name input, pre-fill with conflicting name
            self._name_input = self._pending_name
            self._state = _STATE_NAME

    def _on_load(self, ev):
        if ev.key == pygame.K_ESCAPE:
            self._state = _STATE_MAIN
        elif ev.key == pygame.K_UP and self._saves:
            self._save_selected = (self._save_selected - 1) % len(self._saves)
        elif ev.key == pygame.K_DOWN and self._saves:
            self._save_selected = (self._save_selected + 1) % len(self._saves)
        elif ev.key == pygame.K_RETURN and self._saves:
            name = self._saves[self._save_selected]
            from scenes.loading import LoadingScene
            def task(n=name):
                self.game_manager.load_save(n)
            self.game_manager.push_scene(
                LoadingScene(self.game_manager, task, self._go_to_research_lab,
                             "Loading saved game..."))

    # ── Transitions ────────────────────────────────────────────────────

    def _begin_new_game(self, name: str):
        from scenes.loading import LoadingScene
        def task():
            self.game_manager.new_game(name)
        self.game_manager.push_scene(
            LoadingScene(self.game_manager, task, self._go_to_research_lab,
                         "Creating new game..."))

    def _go_to_research_lab(self):
        from scenes.research_lab import ResearchLabScene
        self.game_manager.push_scene(ResearchLabScene(self.game_manager))

    # ------------------------------------------------------------------

    def update(self, dt: float):
        self._cursor_timer += dt
        if self._cursor_timer >= 0.5:
            self._cursor_timer = 0.0
            self._cursor_visible = not self._cursor_visible

    def draw(self, surface: pygame.Surface):
        surface.fill((20, 20, 30))
        cx = surface.get_width() // 2

        title = self._font_title.render("AI MASTER", True, (240, 240, 255))
        surface.blit(title, title.get_rect(center=(cx, 110)))

        if self._state == _STATE_MAIN:
            self._draw_main(surface, cx)
        elif self._state == _STATE_NAME:
            self._item_rects = []
            self._draw_name(surface, cx)
        elif self._state == _STATE_CONFIRM:
            self._item_rects = []
            self._draw_confirm(surface, cx)
        elif self._state == _STATE_LOAD:
            self._draw_load(surface, cx)

    # ── Draw helpers ───────────────────────────────────────────────────

    def _draw_main(self, surface: pygame.Surface, cx: int):
        sh = surface.get_height()
        start_y = sh // 2 - 30
        self._item_rects = []
        for i, item in enumerate(_MAIN_ITEMS):
            sel = (i == self._selected)
            color = (255, 220, 60) if sel else (180, 180, 200)
            prefix = "> " if sel else "  "
            s = self._font_item.render(prefix + item, True, color)
            r = s.get_rect(center=(cx, start_y + i * 58))
            surface.blit(s, r)
            self._item_rects.append((r.inflate(40, 10), i))
        hint = self._font_sub.render("Up/Down  Navigate     Enter/Click  Select", True, (80, 80, 100))
        surface.blit(hint, hint.get_rect(center=(cx, sh - 35)))

    def _draw_name(self, surface: pygame.Surface, cx: int):
        sh = surface.get_height()
        cy = sh // 2

        prompt = self._font_item.render("Enter Your Name:", True, (200, 200, 240))
        surface.blit(prompt, prompt.get_rect(center=(cx, cy - 60)))

        box_w, box_h = 420, 52
        box = pygame.Rect(cx - box_w // 2, cy - box_h // 2 + 10, box_w, box_h)
        pygame.draw.rect(surface, (35, 35, 55), box, border_radius=6)
        pygame.draw.rect(surface, (100, 110, 200), box, 2, border_radius=6)

        cursor = "|" if self._cursor_visible else " "
        inp = self._font_input.render(self._name_input + cursor, True, (240, 240, 255))
        surface.blit(inp, inp.get_rect(center=box.center))

        hint = self._font_sub.render("Enter  Confirm     ESC  Back", True, (80, 80, 100))
        surface.blit(hint, hint.get_rect(center=(cx, sh - 35)))

    def _draw_confirm(self, surface: pygame.Surface, cx: int):
        sh = surface.get_height()
        cy = sh // 2

        line1 = self._font_item.render(
            f'Save  "{self._pending_name}"  already exists.', True, (220, 160, 60))
        surface.blit(line1, line1.get_rect(center=(cx, cy - 40)))

        line2 = self._font_item.render("Override?    [Y] Yes     [N] No", True, (200, 200, 200))
        surface.blit(line2, line2.get_rect(center=(cx, cy + 30)))

    def _draw_load(self, surface: pygame.Surface, cx: int):
        sh = surface.get_height()
        cy = sh // 2

        title = self._font_item.render("Select Save File", True, (200, 210, 255))
        surface.blit(title, title.get_rect(center=(cx, cy - 140)))

        self._item_rects = []
        if not self._saves:
            msg = self._font_sub.render("No saved games found.", True, (140, 100, 100))
            surface.blit(msg, msg.get_rect(center=(cx, cy)))
        else:
            visible = self._saves[:8]   # show at most 8 entries
            for i, name in enumerate(visible):
                sel = (i == self._save_selected)
                color = (255, 220, 60) if sel else (180, 180, 200)
                prefix = "> " if sel else "  "
                s = self._font_item.render(prefix + name, True, color)
                r = s.get_rect(center=(cx, cy - 80 + i * 48))
                surface.blit(s, r)
                self._item_rects.append((r.inflate(40, 8), i))

        hint = self._font_sub.render(
            "Up/Down  Navigate     Enter/Click  Load     ESC  Back", True, (80, 80, 100))
        surface.blit(hint, hint.get_rect(center=(cx, sh - 35)))
