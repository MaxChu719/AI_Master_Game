import copy
import json
import os
import pygame

# Resolved relative to this file: ai_master/data/saves/
_SAVES_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "saves")

_DEFAULT_RESEARCH = {
    "fighter": {"hp": 0, "attack": 0, "move_speed": 0, "attack_speed": 0, "stamina": 0},
    "archer":  {"hp": 0, "attack": 0, "move_speed": 0, "attack_speed": 0, "stamina": 0},
}

_DEFAULT_AI_MASTER = {
    "max_mp":          0,   # level 0-5, +20 MP each
    "mp_regen":        0,   # level 0-5, +2/s each
    "heal_amount":     0,   # level 0-5, +15 HP each
    "heal_radius":     0,   # level 0-5, +20 px each
    "heal_cooldown":   0,   # level 0-4, -1.0 s each
    "fb_damage":       0,   # level 0-5, +20 dmg each
    "fb_radius":       0,   # level 0-5, +15 px each
    "fb_cooldown":     0,   # level 0-4, -1.5 s each
    "deployment":      0,   # level 0-4, +2 minions each (starts at 2)
    "buffer_size":     0,   # level 0-5, +10 000 transitions each
}

_DEFAULT_STATS = {
    "fighter": {
        "total_kills": 0,
        "total_damage": 0,
        "training_waves": 0,
        "max_waves_survived": 0,
        "attacks_attempted": 0,
        "attacks_hit": 0,
    },
    "archer": {
        "total_kills": 0,
        "total_damage": 0,
        "shots_fired": 0,
        "shots_hit": 0,
        "max_waves_survived": 0,
    },
}


class GameManager:
    def __init__(self, screen: pygame.Surface):
        self.screen = screen
        self.running = True
        self._scene_stack = []
        # Battle-session state (reset each run)
        self.coins = 0
        self.wave_number = 0
        # Player / save state
        self.player_name = ""
        self.save_data = None   # populated by new_game() or load_save()
        # Shared DQN agents (persist across battles; none until init_agents())
        self.fighter_agent = None
        self.archer_agent  = None
        os.makedirs(_SAVES_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # Scene stack

    def push_scene(self, scene):
        self._scene_stack.append(scene)

    def pop_scene(self):
        if self._scene_stack:
            self._scene_stack.pop()

    def _top(self):
        return self._scene_stack[-1] if self._scene_stack else None

    def handle_event(self, event: pygame.event.Event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        scene = self._top()
        if scene:
            scene.handle_event(event)

    def update(self, dt: float):
        scene = self._top()
        if scene:
            scene.update(dt)

    def draw(self, surface: pygame.Surface):
        scene = self._top()
        if scene:
            scene.draw(surface)

    # ------------------------------------------------------------------
    # Agent initialisation (called after new_game / load_save)

    def init_agents(self):
        """Create (or recreate) shared DQN agents and load any saved checkpoints."""
        from ai.dqn import DQNAgent
        from config import CFG

        buf_base    = int(CFG["dqn"]["replay_buffer_size"])
        buf_per_lvl = int(CFG["memory_replay"]["buffer_size_per_upgrade"])
        buf_level   = self.save_data.get("ai_master", {}).get("buffer_size", 0) if self.save_data else 0
        buf_size    = buf_base + buf_level * buf_per_lvl

        # Fighter: 16 actions (8 move + 8 attack directions)
        # Archer:  24 actions (8 move + 16 attack directions for higher precision)
        # Both use image observation (obs_type="image") by default.
        self.fighter_agent = DQNAgent(action_dim=16, role="fighter",
                                      buffer_size=buf_size, obs_type="image")
        self.archer_agent  = DQNAgent(action_dim=24, role="archer",
                                      buffer_size=buf_size, obs_type="image")
        if self.player_name:
            self.fighter_agent.load_checkpoint(self.brain_path(self.player_name, "fighter"))
            self.archer_agent.load_checkpoint( self.brain_path(self.player_name, "archer"))
            self.fighter_agent.load_buffer(self.buffer_path(self.player_name, "fighter"))
            self.archer_agent.load_buffer( self.buffer_path(self.player_name, "archer"))

    # ------------------------------------------------------------------
    # Save / Load helpers

    def _save_path(self, name: str) -> str:
        return os.path.join(_SAVES_DIR, f"{name}.json")

    def brain_path(self, name: str, role: str) -> str:
        """Return path to the brain checkpoint (model weights only) for the given save name and role."""
        return os.path.join(_SAVES_DIR, f"{name}_{role}.pt")

    def buffer_path(self, name: str, role: str) -> str:
        """Return path to the replay buffer file for the given save name and role."""
        return os.path.join(_SAVES_DIR, f"{name}_{role}_buffer.pt")

    def save_exists(self, name: str) -> bool:
        return os.path.isfile(self._save_path(name))

    def list_saves(self) -> list:
        """Return save names (without .json) sorted alphabetically."""
        if not os.path.isdir(_SAVES_DIR):
            return []
        return sorted(f[:-5] for f in os.listdir(_SAVES_DIR) if f.endswith(".json"))

    def new_game(self, name: str):
        """Create a fresh save and set it as current."""
        self.player_name = name
        self.coins = 0
        self.wave_number = 0
        self.save_data = {
            "name": name,
            "coins": 0,
            "waves_completed": 0,
            "research":   copy.deepcopy(_DEFAULT_RESEARCH),
            "ai_master":  copy.deepcopy(_DEFAULT_AI_MASTER),
            "stats":      copy.deepcopy(_DEFAULT_STATS),
        }
        self._write_save()
        self.init_agents()

    def load_save(self, name: str):
        """Load save from disk and set as current."""
        with open(self._save_path(name), "r") as f:
            data = json.load(f)

        # Ensure research structure is complete (handles old saves missing keys)
        data.setdefault("research", copy.deepcopy(_DEFAULT_RESEARCH))
        for minion in ("fighter", "archer"):
            data["research"].setdefault(minion, copy.deepcopy(_DEFAULT_RESEARCH[minion]))
            for stat in _DEFAULT_RESEARCH[minion]:
                data["research"][minion].setdefault(stat, 0)

        # AI Master upgrades
        data.setdefault("ai_master", copy.deepcopy(_DEFAULT_AI_MASTER))
        for key, val in _DEFAULT_AI_MASTER.items():
            data["ai_master"].setdefault(key, val)

        # Stats
        data.setdefault("stats", copy.deepcopy(_DEFAULT_STATS))
        for minion in ("fighter", "archer"):
            data["stats"].setdefault(minion, copy.deepcopy(_DEFAULT_STATS[minion]))
            for stat in _DEFAULT_STATS[minion]:
                data["stats"][minion].setdefault(stat, 0)

        self.player_name = name
        self.save_data   = data
        self.coins       = data.get("coins", 0)
        self.wave_number = data.get("waves_completed", 0)
        self.init_agents()

    def save_game(self):
        """Write current save_data (with live coins) to disk."""
        if self.save_data is None:
            return
        self.save_data["coins"] = self.coins
        self._write_save()

    def _write_save(self):
        with open(self._save_path(self.player_name), "w") as f:
            json.dump(self.save_data, f, indent=2)
