from librr import *

class SimpleAI:
    def __init__(self, player, sheep_list, fence_rect):
        self.player = player
        self.sheep_list = sheep_list
        self.fence_rect = fence_rect
        self.state = "seek_sheep" #ehnii tolow ni honi haih
        self.target_sheep = None
        self.obstacle_avoid_timer = 0
    #collidable_rects = toirch garah yostoi saaduud
    def update(self, collidable_rects):
        if self.player.carrying_sheep:
            self.state = "return_to_fence"
        else:
            self.state = "seek_sheep"
        
        #saadtai morgoldson eshiig shalgah
        for obstacle in collidable_rects:
            if self.player.collision_rect.colliderect(obstacle.inflate(50, 50)):
                self.state = "avoid_obstacle"
                self.obstacle_avoid_timer = 30
        
        if self.obstacle_avoid_timer > 0:
            self.obstacle_avoid_timer -= 1
            return self.avoid_obstacle(collidable_rects)
        
        if self.state == "seek_sheep":
            return self.seek_sheep()
        elif self.state == "return_to_fence":
            return self.return_to_fence()
            
    def seek_sheep(self):
        if not self.sheep_list:
            return 0, 0
            
        # Hamgiin oir honiig shalgah
        closest_sheep = None
        closest_distance = float('inf')
        
        for sheep in self.sheep_list:
            dx = sheep.rect.centerx - self.player.collision_rect.centerx
            dy = sheep.rect.centery - self.player.collision_rect.centery
            distance = (dx**2 + dy**2)**0.5
            
            if distance < closest_distance:
                closest_distance = distance
                closest_sheep = sheep
                
        if closest_sheep:
            dx = closest_sheep.rect.centerx - self.player.collision_rect.centerx
            dy = closest_sheep.rect.centery - self.player.collision_rect.centery
            
            #Normalization
            distance = max(1, (dx**2 + dy**2)**0.5)
            dx = dx / distance * self.player.speed
            dy = dy / distance * self.player.speed
            
            return dx, dy
            
        return 0, 0
    
    def return_to_fence(self):
        #back to fence
        fence_inner = self.fence_rect.inflate(-50, -50)
        
        dx = fence_inner.centerx - self.player.collision_rect.centerx
        dy = fence_inner.centery - self.player.collision_rect.centery
        
        #Normalization
        distance = max(1, (dx**2 + dy**2)**0.5)
        dx = dx / distance * self.player.speed
        dy = dy / distance * self.player.speed
        
        return dx, dy
    
    def avoid_obstacle(self, collidable_rects):
        # saadnaas zailshiih
        escape_vector = [0, 0]
        
        for obstacle in collidable_rects:
            if self.player.collision_rect.colliderect(obstacle.inflate(50, 50)):
                dx = self.player.collision_rect.centerx - obstacle.centerx
                dy = self.player.collision_rect.centery - obstacle.centery
                
                # Normalization
                distance = max(1, (dx**2 + dy**2)**0.5)
                escape_vector[0] += dx / distance * self.player.speed * 2
                escape_vector[1] += dy / distance * self.player.speed * 2
                
        return escape_vector[0], escape_vector[1]