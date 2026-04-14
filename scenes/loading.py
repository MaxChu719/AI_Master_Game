"""
LoadingScene — shown while a blocking task (new_game / load_save) runs in a
background thread.  Prevents the UI from appearing frozen during the few
seconds it takes to initialise agents (loading PyTorch checkpoints, etc.).

Usage:
    from scenes.loading import LoadingScene

    def task():
        game_manager.new_game(name)

    def on_done():
        from scenes.research_lab import ResearchLabScene
        game_manager.push_scene(ResearchLabScene(game_manager))

    game_manager.push_scene(LoadingScene(game_manager, task, on_done, "Creating new game..."))
"""
import threading
import math
import pygame
from engine.scene import BaseScene


class LoadingScene(BaseScene):
    def __init__(self, game_manager, task_fn, on_done_fn, message: str = "Loading..."):
        super().__init__(game_manager)
        self._message   = message
        self._on_done   = on_done_fn
        self._done      = False
        self._error     = None
        self._elapsed   = 0.0  # for dot animation

        self._font_title = pygame.font.SysFont("arial", 36, bold=True)
        self._font_msg   = pygame.font.SysFont("arial", 24)
        self._font_hint  = pygame.font.SysFont("arial", 16)

        # Launch the blocking task on a daemon thread
        self._thread = threading.Thread(target=self._run_task, args=(task_fn,), daemon=True)
        self._thread.start()

    def _run_task(self, task_fn):
        try:
            task_fn()
        except Exception as exc:
            self._error = str(exc)
        finally:
            self._done = True

    # ── Scene interface ───────────────────────────────────────────────────────

    def handle_event(self, event: pygame.event.Event):
        pass  # Block all input while loading

    def update(self, dt: float):
        self._elapsed += dt
        if self._done:
            # Always pop LoadingScene first so on_done() pushes onto the correct stack top.
            self.game_manager.pop_scene()
            if self._error:
                print(f"[LoadingScene] Task failed: {self._error}")
                return
            self._on_done()

    def draw(self, surface: pygame.Surface):
        surface.fill((20, 20, 30))
        sw, sh = surface.get_size()
        cx, cy = sw // 2, sh // 2

        # Title
        title = self._font_title.render("AI MASTER", True, (160, 170, 230))
        surface.blit(title, title.get_rect(center=(cx, cy - 90)))

        # Animated dots  (cycles through "", ".", "..", "...")
        dot_count = int(self._elapsed * 2) % 4
        dots = "." * dot_count
        msg_text = self._message.rstrip(".") + dots
        msg = self._font_msg.render(msg_text, True, (200, 200, 240))
        surface.blit(msg, msg.get_rect(center=(cx, cy - 20)))

        # Spinning arc
        angle = (self._elapsed * 220) % 360
        radius = 28
        arc_rect = pygame.Rect(cx - radius, cy + 20, radius * 2, radius * 2)
        pygame.draw.arc(surface, (100, 140, 220), arc_rect,
                        math.radians(angle), math.radians(angle + 260), 4)

        # Hint
        hint = self._font_hint.render("Please wait...", True, (70, 70, 100))
        surface.blit(hint, hint.get_rect(center=(cx, cy + 80)))
