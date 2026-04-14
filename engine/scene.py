from abc import ABC, abstractmethod
import pygame


class BaseScene(ABC):
    def __init__(self, game_manager):
        self.game_manager = game_manager

    @abstractmethod
    def handle_event(self, event: pygame.event.Event):
        pass

    @abstractmethod
    def update(self, dt: float):
        pass

    @abstractmethod
    def draw(self, surface: pygame.Surface):
        pass
