from librr import *

class HandGestureController:
    def __init__(self, screen_width=800, screen_height=600):
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("camera neej chadsangui.")
            self.use_keyboard = True
        else:
            self.cap.set(3, 640)
            self.cap.set(4, 480)
            self.use_keyboard = False
        
        # Smooth hudulguun
        self.dx_buffer = deque(maxlen=6)
        self.dy_buffer = deque(maxlen=6)
        self.prev_cx = None
        self.prev_cy = None
        
        # Keyboard fallback
        self.keyboard_dx = 0
        self.keyboard_dy = 0

        # Default utga (aldaa garhaas sergiileh)
        self.raised_count = 0
        self.is_paused = False
        self.pause_cooldown = 0

    def is_finger_up(self, tip, pip):
        return tip.y < pip.y - 0.02

    def count_raised_fingers(self, landmarks):
        wrist = landmarks.landmark[self.mp_hands.HandLandmark.WRIST]
        tips = [4, 8, 12, 16, 20]  # thumb, index, middle, ring, pinky
        pips = [3, 6, 10, 14, 18]  # thumb_ip, others_pip

        raised = 0
        # Erhii huruu (ontsgoi)
        if abs(landmarks.landmark[4].x - landmarks.landmark[2].x) > 0.05:
            raised += 1
            
        # Busad 4 huruu
        for tip_id, pip_id in zip(tips[1:], pips[1:]):
            tip = landmarks.landmark[tip_id]
            pip = landmarks.landmark[pip_id]
            if tip.y < pip.y - 0.02:
                raised += 1
        return raised

    def get_movement(self):
        # Keyboard fallback
        if self.use_keyboard:
            return self.keyboard_dx, self.keyboard_dy, self.is_paused, False

        dx = dy = 0
        exit_signal = False
        self.raised_count = 0
        
        # Cooldown for gesture recognition
        if self.pause_cooldown > 0:
            self.pause_cooldown -= 1

        ret, frame = self.cap.read()
        if not ret:
            return 0, 0, self.is_paused, False

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb)

        if results.multi_hand_landmarks:
            hand = results.multi_hand_landmarks[0]
            self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

            # Gariin tow heseg (wrist)
            index_tip = hand.landmark[8]
            cx = index_tip.x * self.screen_width
            cy = index_tip.y * self.screen_height

            # Huruu tooloh
            self.raised_count = self.count_raised_fingers(hand)

            # === Gesture logic ===
            if self.raised_count >= 5 and self.pause_cooldown == 0:
                # 5+ huruu - Exit
                exit_signal = True
                cv2.putText(frame, "EXIT SIGNAL", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 4)
                self.pause_cooldown = 30
                
            elif self.raised_count == 4 and self.pause_cooldown == 0:
                # 4 huruu - Resume
                if self.is_paused:
                    self.is_paused = False
                    cv2.putText(frame, "RESUME", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,0), 4)
                    self.pause_cooldown = 30
                    
            elif self.raised_count == 3 and self.pause_cooldown == 0:
                # 3 huruu - Pause
                if not self.is_paused:
                    self.is_paused = True
                    cv2.putText(frame, "PAUSE", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 2, (0,255,255), 4)
                    self.pause_cooldown = 30
                    self.prev_cx = self.prev_cy = None
                
            elif self.raised_count == 1 or self.raised_count == 2:
                if not self.is_paused:
                    # 1-2 huruu -> hudulguun
                    index_tip = hand.landmark[8]
                    ix = int(index_tip.x * frame.shape[1])
                    iy = int(index_tip.y * frame.shape[0])
                    cv2.circle(frame, (ix, iy), 15, (0,255,0), -1)

                    if self.prev_cx is not None:
                        dx = (cx - self.prev_cx) * 3.0
                        dy = (cy - self.prev_cy) * 3.0

                    self.prev_cx, self.prev_cy = cx, cy

                    # Chigiin sum
                    if abs(dx) > 1 or abs(dy) > 1:
                        cv2.arrowedLine(frame, (ix-30, iy-30), (ix, iy), (255,0,255), 5, tipLength=0.4)
                else:
                    self.prev_cx = self.prev_cy = None
            else:
                self.prev_cx = self.prev_cy = None

            # Status display
            status_color = (0,255,0) if not self.is_paused else (0,255,255)
            status_text = "RUNNING" if not self.is_paused else "PAUSED"
            cv2.putText(frame, f"Fingers: {self.raised_count} | {status_text}", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        else:
            cv2.putText(frame, "HAND NOT DETECTED", (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1.5, (0,0,255), 3)
            self.prev_cx = self.prev_cy = None
            self.raised_count = 0

        # Smooth hudulguun zuvhun 1-2 huruutai uyd 
        if (self.raised_count == 1 or self.raised_count == 2) and not self.is_paused:
            self.dx_buffer.append(dx)
            self.dy_buffer.append(dy)
            smooth_dx = sum(self.dx_buffer) / len(self.dx_buffer)
            smooth_dy = sum(self.dy_buffer) / len(self.dy_buffer)
            smooth_dx = max(-12, min(12, smooth_dx))
            smooth_dy = max(-12, min(12, smooth_dy))
        else:
            smooth_dx = smooth_dy = 0
            self.dx_buffer.clear()
            self.dy_buffer.clear()

        # Delgets deer haragdah medeelel
        cv2.putText(frame, f"DX: {smooth_dx:+.1f} | DY: {smooth_dy:+.1f}", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(frame, "1-2 finger = MOVE | 3 = PAUSE | 4 = RESUME | 5 = EXIT", (50, 480), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200,255,0), 2)

        cv2.imshow("Mongolian Hand Gesture Game - Player 2", frame)
        cv2.waitKey(1)

        return smooth_dx, smooth_dy, self.is_paused, exit_signal

    def update_keyboard_input(self, keys):
        if not self.use_keyboard:
            return
        self.keyboard_dx = self.keyboard_dy = 0
        speed = 8
        if keys[pygame.K_i]: self.keyboard_dy = -speed
        if keys[pygame.K_k]: self.keyboard_dy = speed
        if keys[pygame.K_j]: self.keyboard_dx = -speed
        if keys[pygame.K_l]: self.keyboard_dx = speed

    def release(self):
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()