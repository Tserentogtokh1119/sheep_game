from librr import *

class SmokeParticle:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.radius = random.uniform(3, 6)
        self.alpha = 255
        self.vy = random.uniform(-1, -0.5)
        self.vx = random.uniform(-0.2, 0.2)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        self.alpha -= 2
        self.radius += 0.05

    def draw(self, screen):
        if self.alpha > 0:
            surface = pygame.Surface((self.radius * 2, self.radius * 2), pygame.SRCALPHA)
            pygame.draw.circle(surface, (200, 200, 200, int(self.alpha)), (self.radius, self.radius), self.radius)
            screen.blit(surface, (self.x - self.radius, self.y - self.radius))
