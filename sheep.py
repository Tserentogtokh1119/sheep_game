from librr import *

class Sheep:
    def __init__(self, x, y, images, speed=2.0):
        self.base_images = images
        self.image = self.base_images[0]
        self.rect = self.image.get_rect(center=(x, y))
        self.speed = speed
        self.direction = [random.choice([-1, 1]), random.choice([-1, 1])]
        self.direction_change_timer = 0
        self.direction_change_interval = 1.0
        magnitude = (self.direction[0]**2 + self.direction[1]**2)**0.5
        if magnitude > 0:
            self.direction[0] /= magnitude
            self.direction[1] /= magnitude

    def move(self, screen_width, screen_height, collidable_rects, fence_left, fence_right, player1, player2):
        new_rect = self.rect.copy()
        self.direction_change_timer += 1.0 / 60
        if self.direction_change_timer >= self.direction_change_interval:
            self.direction = [random.uniform(-1, 1), random.uniform(-1, 1)]
            magnitude = (self.direction[0]**2 + self.direction[1]**2)**0.5
            if magnitude > 0:
                self.direction[0] /= magnitude
                self.direction[1] /= magnitude
            self.direction_change_timer = 0

        new_rect.x += self.direction[0] * self.speed
        new_rect.y += self.direction[1] * self.speed

        for fence in [fence_left, fence_right]:
            dx = new_rect.centerx - fence.centerx
            dy = new_rect.centery - fence.centery
            distance = (dx**2 + dy**2)**0.5
            if distance < 200:
                self.direction[0] = dx / max(distance, 1)
                self.direction[1] = dy / max(distance, 1)
                magnitude = (self.direction[0]**2 + self.direction[1]**2)**0.5
                if magnitude > 0:
                    self.direction[0] /= magnitude
                    self.direction[1] /= magnitude
                new_rect.x += self.direction[0] * self.speed
                new_rect.y += self.direction[1] * self.speed
                break

        if new_rect.left <= 0:
            self.direction[0] = abs(self.direction[0])
            new_rect.x = 1
        elif new_rect.right >= screen_width:
            self.direction[0] = -abs(self.direction[0])
            new_rect.x = screen_width - new_rect.width - 1
        if new_rect.top <= 0:
            self.direction[1] = abs(self.direction[1])
            new_rect.y = 1
        elif new_rect.bottom >= screen_height:
            self.direction[1] = -abs(self.direction[1])
            new_rect.y = screen_height - new_rect.height - 1

        for collidable in collidable_rects:
            if new_rect.colliderect(collidable):
                dx = new_rect.centerx - collidable.centerx
                dy = new_rect.centery - collidable.centery
                distance = (dx**2 + dy**2)**0.5
                if distance < 200:
                    self.direction[0] = dx / distance
                    self.direction[1] = dy / distance
                    magnitude = (self.direction[0]**2 + self.direction[1]**2)**0.5
                    if magnitude > 0:
                        self.direction[0] /= magnitude
                        self.direction[1] /= magnitude
                    new_rect.x += self.direction[0] * self.speed
                    new_rect.y += self.direction[1] * self.speed
                break

        self.rect = new_rect

    def update_animation(self, delta_time):
        if self.direction[0] < 0:
            self.image = pygame.transform.flip(self.base_images[0], True, False)
        else:
            self.image = self.base_images[0]

    def draw(self, screen):
        screen.blit(self.image, self.rect)
