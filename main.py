import pygame
from engine.game_manager import GameManager
from scenes.main_menu import MainMenuScene


def main():
    pygame.mixer.pre_init(44100, -16, 2, 512)
    pygame.init()
    screen = pygame.display.set_mode((1280, 720))
    pygame.display.set_caption("AI Master")
    clock = pygame.time.Clock()

    gm = GameManager(screen)
    gm.push_scene(MainMenuScene(gm))

    while gm.running:
        dt = clock.tick(60) / 1000.0

        for event in pygame.event.get():
            gm.handle_event(event)

        gm.update(dt)
        gm.draw(screen)
        pygame.display.flip()

    pygame.quit()


if __name__ == "__main__":
    main()
