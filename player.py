from librr import *

class Player:
    def __init__(self, x, y, images, fence_rect, speed=5, sheep_offset_x=0, is_ai=False):
        self.base_images = images
        self.image = self.base_images[0]
        self.rect = self.image.get_rect(center=(x, y))
        self.collision_rect = pygame.Rect(0, 0, 50, 50)
        self.collision_rect.center = (x, y)
        self.speed = speed
        self.fence_rect = fence_rect
        self.carrying_sheep = None
        self.score = 0
        self.sheep_in_fence = []
        self.sheep_offset_x = sheep_offset_x
        self.direction = "idle"
        self.is_moving = False
        self.is_ai = is_ai
        self.ai_controller = None
        
    def set_ai_controller(self, ai_controller):
        self.ai_controller = ai_controller
        
    def move(self, keys, key_up, key_down, key_left, key_right, screen_width, screen_height, collidable_rects):
        new_rect = self.collision_rect.copy() 
        moving = False
        if keys[key_up] and new_rect.top > 0:
            new_rect.y -= self.speed
            moving = True
        if keys[key_down] and new_rect.bottom < screen_height:
            new_rect.y += self.speed
            moving = True
        if keys[key_left] and new_rect.left > 0:
            new_rect.x -= self.speed
            self.direction = "left"
            moving = True
        if keys[key_right] and new_rect.right < screen_width:
            new_rect.x += self.speed
            self.direction = "right"
            moving = True

        self.is_moving = moving
        if not moving:
            self.direction = "idle"

        for collidable in collidable_rects:
            if new_rect.colliderect(collidable):
                return
        self.collision_rect = new_rect
        self.rect.center = self.collision_rect.center

    def ai_move(self, screen_width, screen_height, collidable_rects):
        if self.ai_controller:
            dx, dy = self.ai_controller.update(collidable_rects)
            self.move_by_vector(dx, dy, screen_width, screen_height, collidable_rects)
    
    def move_by_vector(self, dx, dy, screen_width, screen_height, collidable_rects):
        new_rect = self.collision_rect.copy()
        
        if abs(dx) > 0.1 or abs(dy) > 0.1:
            self.is_moving = True
            
            if dx < 0:
                self.direction = "left"
            elif dx > 0:
                self.direction = "right"
                
            new_rect.x += dx
            new_rect.y += dy
        else:
            self.is_moving = False
            self.direction = "idle"
            
        # Хил хязгаар шалгах
        if new_rect.left < 0:
            new_rect.left = 0
        if new_rect.right > screen_width:
            new_rect.right = screen_width
        if new_rect.top < 0:
            new_rect.top = 0
        if new_rect.bottom > screen_height:
            new_rect.bottom = screen_height
            
        # Саадтай мөргөлдөх шалгах
        collision = False
        for collidable in collidable_rects:
            if new_rect.colliderect(collidable):
                collision = True
                break
                
        if not collision:
            self.collision_rect = new_rect
            
        self.rect.center = self.collision_rect.center

    def update_animation(self, delta_time, is_player2=False):
        if self.direction == "left":
            self.image = pygame.transform.flip(self.base_images[0], True, False)
            self.sheep_offset_x = -abs(self.sheep_offset_x)
        else:
            self.image = self.base_images[0]
            self.sheep_offset_x = abs(self.sheep_offset_x)

        self.rect = self.image.get_rect(center=self.rect.center)

    def draw(self, screen):
        screen.blit(self.image, self.rect)
        if self.carrying_sheep:
            sheep_rect = self.carrying_sheep.image.get_rect(
                center=(self.rect.centerx + self.sheep_offset_x, self.rect.top + 25)
            )
            screen.blit(self.carrying_sheep.image, sheep_rect)