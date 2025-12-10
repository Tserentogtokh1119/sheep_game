from librr import *

class GameState(Enum):
    PLAYING = 0
    WINNER = 1
    QUIT = 2
    PAUSED = 3

class SheepCollectGame:
    def __init__(self):
        pygame.init()
        self.X, self.Y = 1550, 840
        self.screen = pygame.display.set_mode((self.X, self.Y))
        pygame.display.set_caption('Хонио хашцгаая - AI vs Hand Gesture')
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 36)

        self.smoke_particles = []
        self.smoke_timer = 0
        self.smoke_interval = 0.2

        try:
            self.background = pygame.image.load('D:/semester5/AI/project/hg_rec/honi/env\grass.png').convert()
            self.background = pygame.transform.scale(self.background, (self.X//5, self.Y//5))
        except Exception as e:
            print(f"Failed to load background.png: {e}")
            self.background = pygame.Surface((self.X, self.Y))
            self.background.fill((150, 255, 150))
            pygame.draw.rect(self.background, (100, 200, 100), (0, 0, self.X, self.Y//3))
            pygame.draw.circle(self.background, (100, 200, 100), (self.X//4, self.Y//3), 150)
            pygame.draw.circle(self.background, (100, 200, 100), (3*self.X//4, self.Y//3), 200)

        try:
            self.yurt = pygame.image.load('D:/semester5/AI/project/hg_rec/honi/env\ger.png').convert_alpha()
            self.yurt = pygame.transform.scale(self.yurt, (220, 220))
        except Exception as e:
            print(f"Failed to load yurt.png: {e}")
            self.yurt = pygame.Surface((220, 220), pygame.SRCALPHA)
            pygame.draw.circle(self.yurt, (200, 200, 200), (110, 110), 110)
            pygame.draw.circle(self.yurt, (139, 69, 19), (110, 110), 110, 3)
            pygame.draw.rect(self.yurt, (139, 69, 19), (95, 29, 30, 47))
            pygame.draw.circle(self.yurt, (139, 69, 19), (110, 110), 15)
        self.yurt_rect = self.yurt.get_rect(center=(self.X//2, 120))
        self.collision_yurt_rect = pygame.Rect(0, 0, 90, 90)
        self.collision_yurt_rect.center = (self.X//2, 120)

        try:
            self.fence = pygame.image.load('D:/semester5/AI/project/hg_rec/honi/env\hashaa.png').convert_alpha()
            self.fence = pygame.transform.scale(self.fence, (200, 300))
        except Exception as e:
            print(f"Failed to load fence.png: {e}")
            self.fence = pygame.Surface((150, 225), pygame.SRCALPHA)
            for i in range(5):
                pygame.draw.rect(self.fence, (139, 69, 19), (0, i*45, 150, 30))
            pygame.draw.rect(self.fence, (139, 69, 19), (0, 0, 7.5, 225))
            pygame.draw.rect(self.fence, (139, 69, 19), (142.5, 0, 7.5, 225))
        self.fence_left = self.fence.get_rect(midleft=(50, self.Y//2))
        self.fence_right = self.fence.get_rect(midright=(self.X-50, self.Y//2))
        self.fence_left_collision = pygame.Rect(0, 0, 100, 150)
        self.fence_left_collision = pygame.Rect(0, 0, 100, 150)
        self.fence_left_collision.center = self.fence_left.center
        self.fence_right_collision = pygame.Rect(0, 0, 100, 150)
        self.fence_right_collision.center = self.fence_right.center

        try:
            sheep_img = pygame.image.load('D:/semester5/AI/project/hg_rec/honi/env\honi.png').convert_alpha()
            sheep_img = pygame.transform.scale(sheep_img, (60, 60))
            self.sheep_images = [
                sheep_img,
                pygame.transform.rotate(sheep_img, 5),
                pygame.transform.rotate(sheep_img, -5),
            ]
        except Exception as e:
            print(f"Failed to load sheep.png: {e}")
            sheep_img = pygame.Surface((30, 30), pygame.SRCALPHA)
            pygame.draw.circle(sheep_img, (255, 255, 255), (15, 15), 11)
            pygame.draw.circle(sheep_img, (50, 50, 50), (15, 8), 4)
            pygame.draw.polygon(sheep_img, (50, 50, 50), [(11, 4), (15, 8), (19, 4)])
            pygame.draw.polygon(sheep_img, (50, 50, 50), [(11, 11), (15, 8), (19, 11)])
            pygame.draw.rect(sheep_img, (50, 50, 50), (11, 22, 4, 8))
            self.sheep_images = [
                sheep_img,
                pygame.transform.rotate(sheep_img, 5),
                pygame.transform.rotate(sheep_img, -5),
            ]

        try:
            player1_img = pygame.image.load('D:/semester5/AI/project/hg_rec/honi/env\huu.png').convert_alpha()
            player1_img = pygame.transform.scale(player1_img, (90, 90))
            self.player1_images = [
                player1_img,
                pygame.transform.rotate(player1_img, 5),
                pygame.transform.rotate(player1_img, -5),
            ]
        except Exception as e:
            print(f"Failed to load boy_blue.png: {e}")
            player1_img = pygame.Surface((50, 50), pygame.SRCALPHA)
            pygame.draw.circle(player1_img, (0, 0, 255), (25, 25), 20)
            pygame.draw.circle(player1_img, (255, 200, 150), (25, 15), 10)
            pygame.draw.rect(player1_img, (0, 0, 255), (15, 5, 20, 10))
            pygame.draw.rect(player1_img, (0, 0, 255), (5, 15, 10, 20))
            pygame.draw.rect(player1_img, (0, 0, 255), (35, 15, 10, 20))
            pygame.draw.rect(player1_img, (255, 200, 150), (15, 25, 5, 10))
            pygame.draw.rect(player1_img, (255, 200, 150), (30, 25, 5, 10))
            pygame.draw.rect(player1_img, (255, 165, 0), (20, 35, 10, 5))
            self.player1_images = [
                player1_img,
                pygame.transform.rotate(player1_img, 5),
                pygame.transform.rotate(player1_img, -5),
            ]

        try:
            player2_img = pygame.image.load('D:/semester5/AI/project/hg_rec/honi/env\ohin.png').convert_alpha()
            player2_img = pygame.transform.scale(player2_img, (90, 90))
            self.player2_images = [
                player2_img,
                pygame.transform.rotate(player2_img, 5),
                pygame.transform.rotate(player2_img, -5),
            ]
        except Exception as e:
            print(f"Failed to load girl_red.png: {e}")
            player2_img = pygame.Surface((50, 50), pygame.SRCALPHA)
            pygame.draw.circle(player2_img, (255, 0, 0), (25, 25), 20)
            pygame.draw.circle(player2_img, (255, 200, 150), (25, 15), 10)
            pygame.draw.rect(player2_img, (255, 0, 0), (15, 5, 20, 10))
            pygame.draw.rect(player2_img, (255, 0, 0), (5, 15, 10, 20))
            pygame.draw.rect(player2_img, (255, 0, 0), (35, 15, 10, 20))
            pygame.draw.rect(player2_img, (255, 200, 150), (15, 25, 5, 10))
            pygame.draw.rect(player2_img, (255, 200, 150), (30, 25, 5, 10))
            pygame.draw.rect(player2_img, (0, 128, 0), (20, 35, 10, 5))
            pygame.draw.rect(player2_img, (0, 0, 0), (15, 5, 5, 10))
            pygame.draw.rect(player2_img, (0, 0, 0), (30, 5, 5, 10))
            self.player2_images = [
                player2_img,
                pygame.transform.rotate(player2_img, 5),
                pygame.transform.rotate(player2_img, -5),
            ]

        # player1 iig ai bolgoh
        self.player1 = Player(200, self.Y//2, self.player1_images, self.fence_left, 
                             speed=5, sheep_offset_x=15, is_ai=True)
        
        # Player 2 iin hand dohio
        self.player2 = Player(self.X-200, self.Y//2, self.player2_images, self.fence_right, 
                             speed=5, sheep_offset_x=15, is_ai=False)
        
        # ai bolon garni dohioni hynagch
        self.hand_controller = HandGestureController(self.X, self.Y)
        
        # ai controller iig tohiruulah 
        self.player1_ai = SimpleAI(self.player1, [], self.fence_left_collision)
        self.player1.set_ai_controller(self.player1_ai)

        self.sheep_list = []
        for _ in range(14):
            self.spawn_sheep_initial()

        # ai - d sheep list ogoh
        self.player1_ai.sheep_list = self.sheep_list

        self.game_state = GameState.PLAYING
        self.winner = None

    def spawn_sheep_initial(self):
        for _ in range(100):
            x = random.randint(50, self.X - 50)
            y = random.randint(50, self.Y - 50)
            new_sheep_rect = self.sheep_images[0].get_rect(center=(x, y))
            too_close = (new_sheep_rect.colliderect(self.collision_yurt_rect) or
                         new_sheep_rect.colliderect(self.fence_left_collision) or
                         new_sheep_rect.colliderect(self.fence_right_collision))
            if not too_close:
                dx_yurt = x - self.collision_yurt_rect.centerx
                dy_yurt = y - self.collision_yurt_rect.centery
                dist_yurt = (dx_yurt**2 + dy_yurt**2)**0.5
                dx_fence_left = x - self.fence_left_collision.centerx
                dy_fence_left = y - self.fence_left_collision.centery
                dist_fence_left = (dx_fence_left**2 + dy_fence_left**2)**0.5
                dx_fence_right = x - self.fence_right_collision.centerx
                dy_fence_right = y - self.fence_right_collision.centery
                dist_fence_right = (dx_fence_right**2 + dy_fence_right**2)**0.5
                if dist_yurt > 200 and dist_fence_left > 200 and dist_fence_right > 200:
                    self.sheep_list.append(Sheep(x, y, self.sheep_images))
                    return
        print("Warning: Could not find suitable spawn location for sheep after 100 attempts.")

    def spawn_sheep_center(self):
        x, y = self.X//2, self.Y//2
        new_sheep_rect = self.sheep_images[0].get_rect(center=(x, y))
        too_close = (new_sheep_rect.colliderect(self.collision_yurt_rect) or
                     new_sheep_rect.colliderect(self.fence_left_collision) or
                     new_sheep_rect.colliderect(self.fence_right_collision))
        if not too_close:
            dx_yurt = x - self.collision_yurt_rect.centerx
            dy_yurt = y - self.collision_yurt_rect.centery
            dist_yurt = (dx_yurt**2 + dy_yurt**2)**0.5
            dx_fence_left = x - self.fence_left_collision.centerx
            dy_fence_left = y - self.fence_left_collision.centery
            dist_fence_left = (dx_fence_left**2 + dy_fence_left**2)**0.5
            dx_fence_right = x - self.fence_right_collision.centerx
            dy_fence_right = y - self.fence_right_collision.centery
            dist_fence_right = (dx_fence_right**2 + dy_fence_right**2)**0.5
            if dist_yurt > 200 and dist_fence_left > 200 and dist_fence_right > 200:
                self.sheep_list.append(Sheep(x, y, self.sheep_images))
                return
        print("Warning: Could not spawn sheep at center due to proximity constraints.")

    def update_smoke(self, delta_time):
        self.smoke_timer += delta_time
        if self.smoke_timer >= self.smoke_interval:
            self.smoke_timer = 0
            smoke_x = self.yurt_rect.centerx
            smoke_y = self.yurt_rect.top + 10
            self.smoke_particles.append(SmokeParticle(smoke_x, smoke_y))

        self.smoke_particles = [p for p in self.smoke_particles if p.alpha > 0]
        for particle in self.smoke_particles:
            particle.update()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.game_state = GameState.QUIT
                return False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.game_state = GameState.PAUSED if self.game_state == GameState.PLAYING else GameState.PLAYING
                elif event.key == pygame.K_ESCAPE:
                    self.game_state = GameState.QUIT
                    return False
        return True

    def update(self):
        keys = pygame.key.get_pressed()

        # Always check hand gesture input for pause/resume, even when paused
        self.hand_controller.update_keyboard_input(keys)
        dx, dy, pause_signal, exit_signal = self.hand_controller.get_movement()
        
        # Handle exit signal (5+ fingers) - pause entire game
        if exit_signal:
            self.game_state = GameState.PAUSED
            return
        
        # Sync game pause state with hand gesture controller pause state
        # If hand gesture controller shows pause_signal, sync the game state
        if pause_signal and self.game_state == GameState.PLAYING:
            # Hand gesture controller wants pause
            self.game_state = GameState.PAUSED
            return
        elif not pause_signal and self.game_state == GameState.PAUSED:
            # Hand gesture controller resumed - resume game
            self.game_state = GameState.PLAYING
        
        # Only update game if PLAYING
        if self.game_state != GameState.PLAYING:
            return
        
        # Player 1: AI хөдөлгөөн
        self.player1.ai_move(self.X, self.Y, [self.collision_yurt_rect])
        
        # Player 2: Movement from hand gesture
        self.player2.move_by_vector(dx, dy, self.X, self.Y, [self.collision_yurt_rect])

        delta_time = 1.0 / 60
        self.player1.update_animation(delta_time)
        self.player2.update_animation(delta_time)
        for sheep in self.sheep_list:
            sheep.update_animation(delta_time)
            sheep.move(self.X, self.Y, [self.collision_yurt_rect, self.fence_left, self.fence_right], 
                       self.fence_left, self.fence_right, self.player1, self.player2)

        self.update_smoke(delta_time)

        if not self.player1.carrying_sheep:
            for sheep in self.sheep_list[:]:
                if self.player1.collision_rect.colliderect(sheep.rect):
                    self.player1.carrying_sheep = sheep
                    self.sheep_list.remove(sheep)
                    break

        if not self.player2.carrying_sheep:
            for sheep in self.sheep_list[:]:
                if self.player2.collision_rect.colliderect(sheep.rect):
                    self.player2.carrying_sheep = sheep
                    self.sheep_list.remove(sheep)
                    break

        if self.player1.carrying_sheep:
            fence_inner = self.fence_left_collision.inflate(-50, -50)
            if fence_inner.colliderect(self.player1.collision_rect):
                self.player1.score += 1
                self.player1.sheep_in_fence.append(self.player1.carrying_sheep)
                self.player1.carrying_sheep = None
                self.spawn_sheep_center()
                if self.player1.score >= 9:
                    self.winner = "AI yallaa."
                    self.game_state = GameState.WINNER

        if self.player2.carrying_sheep:
            fence_inner = self.fence_right_collision.inflate(-50, -50)
            if fence_inner.colliderect(self.player2.collision_rect):
                self.player2.score += 1
                self.player2.sheep_in_fence.append(self.player2.carrying_sheep)
                self.player2.carrying_sheep = None
                self.spawn_sheep_center()
                if self.player2.score >= 9:
                    self.winner = "Ta yallaa."
                    self.game_state = GameState.WINNER

    def draw(self):
        for i in range(0, self.X, self.background.get_width()):
            for j in range(0, self.Y, self.background.get_height()):
                self.screen.blit(self.background, (i, j))
        for i, sheep in enumerate(self.player1.sheep_in_fence[:9]):
            x = self.fence_left.centerx - 30 + (i % 3) * 30
            y = self.fence_left.centery + (i // 3) * 40
            sheep_rect = sheep.image.get_rect(center=(x, y))
            self.screen.blit(sheep.image, sheep_rect)
        self.screen.blit(self.fence, self.fence_left)

        for i, sheep in enumerate(self.player2.sheep_in_fence[:9]):
            x = self.fence_right.centerx - 30 + (i % 3) * 30
            y = self.fence_right.centery + (i // 3) * 40
            sheep_rect = sheep.image.get_rect(center=(x, y))
            self.screen.blit(sheep.image, sheep_rect)
        self.screen.blit(self.fence, self.fence_right)

        self.screen.blit(self.yurt, self.yurt_rect)

        for particle in self.smoke_particles:
            particle.draw(self.screen)

        for sheep in self.sheep_list:
            sheep.draw(self.screen)

        self.player1.draw(self.screen)
        self.player2.draw(self.screen)

        score_text = self.font.render(f'Player1: {self.player1.score}   |   Player2: {self.player2.score}', True, (255, 255, 255)) 
        self.screen.blit(score_text, (self.X//2 - score_text.get_width()//2, 20))

        control_text = self.font.render('P: Togloomih zogsooh | ESC: Garah', True, (255, 255, 255))
        self.screen.blit(control_text, (self.X//2 - control_text.get_width()//2, 60))
        
        if self.hand_controller.use_keyboard:
            keyboard_help = self.font.render('Hand gesture recognition ii orond: I(deesh), K(doosh), J(zuun), L(baruun)', True, (255, 255, 200))
            self.screen.blit(keyboard_help, (self.X//2 - keyboard_help.get_width()//2, self.Y - 40))
        else:
            camera_status = self.font.render('Camera ajillaj baina doloowor huruugaa hodolgon toglono uu.', True, (200, 255, 200))
            self.screen.blit(camera_status, (self.X//2 - camera_status.get_width()//2, self.Y - 40))

        if self.game_state == GameState.WINNER:
            text = self.font.render(f"{self.winner}", True, (255, 255, 0))
            self.screen.blit(text, (self.X//2 - text.get_width()//2, self.Y//2))
            restart_text = self.font.render("Dahin ehluulhiin tuld program - iig ahin achaallana uu", True, (255, 200, 200))
            self.screen.blit(restart_text, (self.X//2 - restart_text.get_width()//2, self.Y//2 + 50))
        elif self.game_state == GameState.PAUSED:
            text = self.font.render("Tur zogsooson. Dahin P darna uu.", True, (255, 255, 247))
            self.screen.blit(text, (self.X//2 - text.get_width()//2, self.Y//2))

        pygame.display.flip()

    async def run(self):
        running = True
        try:
            while running and self.game_state != GameState.QUIT:
                running = self.handle_events()
                self.update()
                self.draw()
                self.clock.tick(60)
                await asyncio.sleep(1.0 / 60)
        finally:
            self.hand_controller.release()
            cv2.destroyAllWindows()
            pygame.quit()

if platform.system() == "Emscripten":
    game = SheepCollectGame()
    asyncio.ensure_future(game.run())
else:
    if __name__ == "__main__":
        try:
            game = SheepCollectGame()
            asyncio.run(game.run())
        except KeyboardInterrupt:
            print("\nTogloom duuslaa.")
        except Exception as e:
            print(f"Aldaa garlaa: {e}")