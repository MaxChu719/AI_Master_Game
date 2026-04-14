from concurrent.futures import ThreadPoolExecutor
from config import CFG

_TCFG = CFG.get("training", {})


class TrainingSystem:
    """
    Runs one DQN training step in a background thread, throttled to every
    `train_interval` frames (default 10, configurable in config.json →
    training.train_interval).

    mode="rainbow"        → calls agent.train_step()                (default, in-game)
    mode="expected_sarsa" → calls agent.train_step_expected_sarsa() (optional, in-game)

    The mode is read from config.json → training.ingame_mode at construction
    time and can be overridden by passing mode= explicitly (e.g. ResearchLab
    always uses "rainbow" regardless of the config setting).
    """

    def __init__(self, mode: str | None = None, train_interval: int | None = None):
        self._executor = ThreadPoolExecutor(max_workers=1)
        self._future = None
        self.latest_result = None

        # Default mode comes from config; caller may override (e.g. "rainbow"
        # for Memory Replay training that always uses Rainbow DQN).
        if mode is None:
            mode = str(_TCFG.get("ingame_mode", "rainbow"))
        self.mode = mode   # "rainbow" | "expected_sarsa"

        if train_interval is None:
            train_interval = int(_TCFG.get("train_interval", 10))
        self.train_interval = train_interval
        self._frame = 0    # internal frame counter

    def schedule_training(self, agent):
        """Call every game frame; training is only dispatched every train_interval frames."""
        self._frame += 1
        if self._frame % self.train_interval != 0:
            return
        if self._future is None or self._future.done():
            if self.mode == "expected_sarsa":
                self._future = self._executor.submit(agent.train_step_expected_sarsa)
            else:
                self._future = self._executor.submit(agent.train_step)

    def collect_result(self):
        if self._future is not None and self._future.done():
            result = self._future.result()
            self._future = None
            if "steps" in result:   # real training occurred (buffer was full)
                self.latest_result = result
            return result
        return None
