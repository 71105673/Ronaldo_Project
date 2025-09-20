import pygame
import sys
import cv2
import numpy as np
import random
import serial
import os
import time
from typing import Dict, List

import Photofunia
from Button import ImageButton, MenuButton
from Config import *

class Game:
    """Penalty Kick Challenge 게임의 메인 클래스 (원본 로직 완벽 복원)"""
    
    def __init__(self):
        """게임 초기화"""
        pygame.init(); pygame.mixer.init()
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.NOFRAME)
        pygame.display.set_caption("Penalty Kick Challenge")
        self.clock = pygame.time.Clock()
        self.running = True

        self.screen_state = "menu"
        self.game_mode = "single"
        
        self._load_fonts()
        self._load_assets()

        self.ser_goalkeeper = self._init_serial(GOALKEEPER_SERIAL_PORT, "골키퍼")
        self.ser_attacker = self._init_serial(ATTACKER_SERIAL_PORT, "공격수")
        self.cap_goalkeeper = self._init_camera(GOALKEEPER_CAM_INDEX, "골키퍼")
        self.cap_attacker = self._init_camera(ATTACKER_CAM_INDEX, "공격수")

        self.buttons = self._create_buttons()
        self.reset_game_state(full_reset=True)

        self.transition_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT)); self.transition_surface.fill(BLACK)
        self.transition_alpha = 0; self.fading_out = False; self.fading_in = False; self.transition_target = None
        
        if self.sounds.get("bg_thumbnail"): self.sounds["bg_thumbnail"].play(-1)

    def _load_fonts(self):
        self.fonts = {
            "default": self._safe_load_font(FONT_PATH, 40, 50), "small": self._safe_load_font(FONT_PATH, 30, 40),
            "description": self._safe_load_font(FONT_PATH, 50, 60), "title": self._safe_load_font(FONT_BOLD_PATH, 120, 130),
            "countdown": self._safe_load_font(FONT_PATH, 200, 250), "score": self._safe_load_font(FONT_BOLD_PATH, 60, 70),
            "rank": self._safe_load_font(FONT_BOLD_PATH, 100, 110),
        }

    def _load_assets(self):
        self.images = {}; self.sounds = {}; self.videos = {}; self.gif_frames = {}
        try:
            self.sounds.update({
                "button": pygame.mixer.Sound(os.path.join(SOUND_PATH, "button_click.wav")),
                "siu": pygame.mixer.Sound(os.path.join(SOUND_PATH, "SIUUUUU.wav")),
                "success": pygame.mixer.Sound(os.path.join(SOUND_PATH, "야유.mp3")),
                "bg_thumbnail": pygame.mixer.Sound(os.path.join(SOUND_PATH, "Time_Bomb.mp3"))
            }); self.sounds["failed"] = self.sounds["siu"]
            
            ball_img = pygame.image.load(os.path.join(IMAGE_PATH, "final_ronaldo", "Ball.png")).convert_alpha()
            self.images["scoreboard_ball"] = pygame.transform.scale(ball_img, (80, 80))
            self.images["ball"] = pygame.transform.scale(ball_img, (200, 200))
            self.images["info_bg"] = pygame.image.load(os.path.join(IMAGE_PATH, "info", "info_back2.jpg")).convert()

            self.gif_frames['success'] = self._load_gif_frames(os.path.join(IMAGE_PATH, "final_ronaldo", "pk.gif"))
            self.gif_frames['failure'] = self._load_gif_frames(os.path.join(IMAGE_PATH, "G.O.A.T", "siuuu.gif"))

            self.videos = {name: cv2.VideoCapture(os.path.join(IMAGE_PATH, f"{filename}.{ext}")) for name, filename, ext in [
                ("lose", "lose_keeper", "gif"), ("victory", "victory", "gif"), ("defeat", "defeat", "gif"), 
                ("game_bg", "Ground1", "mp4"), ("menu_bg", "game_thumbnail", "mp4")
            ]}
            self.shoot_video = cv2.VideoCapture(os.path.join(IMAGE_PATH, "shoot.gif"))
            if self.shoot_video.isOpened():
                self.shoot_video_total_frames = int(self.shoot_video.get(cv2.CAP_PROP_FRAME_COUNT))
                self.shoot_video_interval = 7000 / self.shoot_video_total_frames if self.shoot_video_total_frames > 0 else 0
        except (pygame.error, FileNotFoundError) as e: print(f"에셋 로딩 중 오류 발생: {e}")

    def _create_buttons(self) -> Dict[str, List]:
        btn_sound = self.sounds.get("button")
        return {
            "game_select": [
                MenuButton("1인 플레이", MAIN_START_X + 50, 400, 350, 100, self.fonts["default"], lambda: self.start_game("single"), sound=btn_sound),
                MenuButton("2인 플레이", MAIN_START_X + 50, 500, 350, 100, self.fonts["default"], lambda: self.start_game("multi"), sound=btn_sound),
                MenuButton("게임 설명", MAIN_START_X + 50, 600, 350, 100, self.fonts["default"], lambda: self.start_transition("info"), sound=btn_sound),
                ImageButton(os.path.join(IMAGE_PATH, "btn_back.png"), 150, 150, 100, 100, self.go_to_menu, sound=btn_sound)
            ],
            "face_capture": [ImageButton(os.path.join(IMAGE_PATH, "btn_back.png"), 150, 150, 100, 100, self.go_to_game_select, sound=btn_sound)],
            "play": [ImageButton(os.path.join(IMAGE_PATH, "btn_back.png"), 150, 150, 100, 100, self.go_to_game_select, sound=btn_sound)],
            "info": [ImageButton(os.path.join(IMAGE_PATH, "btn_exit.png"), MAIN_MONITOR_WIDTH - 150, 150, 100, 100, self.go_to_game_select, sound=btn_sound)],
            "end": [
                ImageButton(os.path.join(IMAGE_PATH, "btn_restart.png"), MAIN_MONITOR_CENTER_X - 300, SCREEN_HEIGHT - 250, 400, 250, self.go_to_game_select, sound=btn_sound),
                ImageButton(os.path.join(IMAGE_PATH, "btn_main_menu.png"), MAIN_MONITOR_CENTER_X + 300, SCREEN_HEIGHT - 250, 400, 250, self.go_to_menu, sound=btn_sound)
            ]
        }

    def run(self):
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        self.cleanup()

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): self.running = False
            if self.fading_in or self.fading_out: continue
            
            if self.screen_state == "menu" and event.type == pygame.KEYDOWN: self.start_transition("game_select")
            elif self.screen_state == "play" and self.waiting_for_start and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.countdown_start, self.waiting_for_start = pygame.time.get_ticks(), False
            
            for button in self.buttons.get(self.screen_state, []): button.handle_event(event)

    def update(self):
        if self.fading_out or self.fading_in:
            if self.fading_out:
                self.transition_alpha = min(255, self.transition_alpha + TRANSITION_SPEED)
                if self.transition_alpha == 255: self.fading_out, self.fading_in, self.screen_state = False, True, self.transition_target
            else:
                self.transition_alpha = max(0, self.transition_alpha - TRANSITION_SPEED)
                if self.transition_alpha == 0: self.fading_in = False
        else:
             for button in self.buttons.get(self.screen_state, []): button.update()
        
        if self.screen_state == 'play': self._update_play_state()
        elif self.screen_state == 'face_capture': self._update_face_capture_state()

    def draw(self):
        self.screen.fill(BLACK)
        draw_function = getattr(self, f"_draw_{self.screen_state}_screen")
        draw_function()
        
        if not (self.fading_in or self.fading_out):
            for button in self.buttons.get(self.screen_state, []): button.draw(self.screen)
        
        if self.fading_in or self.fading_out:
            self.transition_surface.set_alpha(self.transition_alpha)
            self.screen.blit(self.transition_surface, (0, 0))
        pygame.display.flip()

    def _draw_menu_screen(self):
        self._draw_video_background(self.videos.get("menu_bg"), MAIN_START_X, MAIN_MONITOR_WIDTH)
        self._draw_text("Penalty Kick Challenge", self.fonts["title"], WHITE, MAIN_MONITOR_CENTER_X, SCREEN_HEIGHT * 0.3)
        self._draw_text("PRESS ANY KEY", self.fonts["description"], WHITE, MAIN_MONITOR_CENTER_X, SCREEN_HEIGHT * 0.75)

    def _draw_game_select_screen(self): self._draw_video_background(self.videos.get("game_bg"), MAIN_START_X, MAIN_MONITOR_WIDTH)
    
    def _update_face_capture_state(self):
        if not self.is_capturing_face:
            self._send_uart_command(self.ser_goalkeeper, 'face')
            self.is_capturing_face = True

        if not self.captured_goalkeeper_face:
            self._handle_face_uart(self.ser_goalkeeper, "goalkeeper")
        elif self.game_mode == "multi" and not self.captured_attacker_face:
            if not self.attacker_capture_sent:
                self._send_uart_command(self.ser_attacker, 'face')
                self.attacker_capture_sent = True
            self._handle_face_uart(self.ser_attacker, "attacker")

    def _draw_face_capture_screen(self):
        self.screen.fill(BLACK)
        self._draw_player_capture_panel(self.cap_goalkeeper, self.captured_goalkeeper_face, "골키퍼", GOALKEEPER_START_X, GOALKEEPER_MONITOR_WIDTH, self.last_goalkeeper_face_coords)
        if self.game_mode == "multi":
            if not self.captured_goalkeeper_face:
                self._draw_placeholder_panel("골키퍼 대기 중...", ATTACKER_START_X, ATTACKER_MONITOR_WIDTH)
            else:
                self._draw_player_capture_panel(self.cap_attacker, self.captured_attacker_face, "공격수", ATTACKER_START_X, ATTACKER_MONITOR_WIDTH, self.last_attacker_face_coords)
        else: self._draw_placeholder_panel("", ATTACKER_START_X, ATTACKER_MONITOR_WIDTH)

    def _update_play_state(self):
        current_time = pygame.time.get_ticks()
        if self.countdown_start and not self.result_time:
            if current_time - self.countdown_start < 5000:
                self._handle_grid_uart(self.ser_goalkeeper, "goalkeeper")
                if self.game_mode == "multi": self._handle_grid_uart(self.ser_attacker, "attacker")
            else:
                self.final_col = self.selected_col if self.selected_col is not None else random.randint(0, 4)
                self.ball_col = self.attacker_selected_col if self.game_mode == "multi" and self.attacker_selected_col is not None else random.randint(0, 4)
                self.is_success = (self.final_col == self.ball_col)
                self.result_time = current_time
                self.chances_left -= 1
                
                if self.is_success:
                    self.score += 1; self.sounds.get("success", pygame.mixer.Sound).play()
                else:
                    if self.game_mode == 'multi': self.attacker_score += 1
                    self.sounds.get("failed", pygame.mixer.Sound).play()
                self.countdown_start = None

        if self.result_time and not self.gif_start_time and (current_time - self.result_time > RESULT_DELAY_MS):
             self.gif_start_time = current_time
        
        if self.gif_start_time and (current_time - self.gif_start_time > GIF_PLAY_DURATION_MS):
            if self.chances_left > 0: self.start_new_round()
            else: self._prepare_for_end_screen()

    def _draw_play_screen(self):
        self._draw_shoot_video()
        self._draw_player_play_panel("goalkeeper")
        if self.game_mode == "multi": self._draw_player_play_panel("attacker")
        else: self._draw_placeholder_panel("", ATTACKER_START_X, ATTACKER_MONITOR_WIDTH)

        if self.waiting_for_start:
            overlay = pygame.Surface((MAIN_MONITOR_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (MAIN_START_X, 0))
            self._draw_text("시작하시겠습니까?", self.fonts["title"], WHITE, MAIN_MONITOR_CENTER_X, SCREEN_HEIGHT/2 - 60)
            self._draw_text("(Press Space Bar)", self.fonts["default"], WHITE, MAIN_MONITOR_CENTER_X, SCREEN_HEIGHT/2 + 40)
        
        if self.countdown_start and not self.result_time:
            countdown_val = 5 - (pygame.time.get_ticks() - self.countdown_start) // 1000
            if countdown_val > 0:
                self._draw_text(str(countdown_val), self.fonts["countdown"], WHITE, GOALKEEPER_MONITOR_CENTER_X, SCREEN_HEIGHT / 2)
                if self.game_mode == "multi": self._draw_text(str(countdown_val), self.fonts["countdown"], WHITE, ATTACKER_MONITOR_CENTER_X, SCREEN_HEIGHT / 2)
        
        if self.gif_start_time: self._draw_result_gif()
        
    def _draw_info_screen(self):
        pygame.draw.rect(self.screen, BLACK, (GOALKEEPER_START_X, 0, GOALKEEPER_MONITOR_WIDTH, SCREEN_HEIGHT))
        pygame.draw.rect(self.screen, BLACK, (ATTACKER_START_X, 0, ATTACKER_MONITOR_WIDTH, SCREEN_HEIGHT))
        
        scaled_bg = pygame.transform.scale(self.images["info_bg"], (MAIN_MONITOR_WIDTH, SCREEN_HEIGHT))
        self.screen.blit(scaled_bg, (MAIN_START_X, 0))
        
        self._draw_text("게임 방법", self.fonts["title"], WHITE, MAIN_MONITOR_CENTER_X, 150)
        text_1p = ["[1인 플레이]", "", "1. 스페이스 바를 누르면 5초 카운트 다운 시작", "2. 5개의 영역 중 한 곳을 선택", "3. 5번의 기회동안 최대한 많은 공을 막으세요!"]
        text_2p = ["[2인 플레이]", "", "1. 공격수와 골키퍼로 나뉩니다.", "2. 공격수는 공을 찰 방향을 선택", "3. 골키퍼는 공을 막을 방향을 선택", "4. 더 많은 득점을 한 쪽이 승리합니다!"]
        
        font, line_height, start_y = self.fonts["description"], 60, 300
        for i, line in enumerate(text_1p): self._draw_text(line, font, WHITE, MAIN_MONITOR_WIDTH * 0.25, start_y + i * line_height)
        for i, line in enumerate(text_2p): self._draw_text(line, font, WHITE, MAIN_MONITOR_WIDTH * 0.75, start_y + i * line_height)

    def _draw_synthesizing_screen(self):
        self.screen.fill(BLACK)
        if not self.synthesized_frames and self.synthesis_info:
            pygame.display.flip()
            info = self.synthesis_info
            self.synthesized_frames = Photofunia.create_synthesized_gif_frames(info["face_path"], info["gif_path"], info["monitor_size"])
            if self.end_video: self.end_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.start_transition("end")
        self._draw_text("얼굴 합성 중...", self.fonts["title"], WHITE, SCREEN_WIDTH/2, SCREEN_HEIGHT/2)
    
    def _draw_end_screen(self):
        self.screen.fill(BLACK)
        if self.end_video: self._draw_video_background(self.end_video, MAIN_START_X, MAIN_MONITOR_WIDTH)
        
        if self.synthesized_frames:
            frame_idx = int((pygame.time.get_ticks() / SYNTHESIZED_GIF_FRAME_DURATION) % len(self.synthesized_frames))
            self.screen.blit(self.synthesized_frames[frame_idx], (GOALKEEPER_START_X, 0))
            if self.game_mode == "multi": self.screen.blit(self.synthesized_frames[frame_idx], (ATTACKER_START_X, 0))

        self._draw_text(self.final_rank, self.fonts["rank"], GOLD_COLOR, MAIN_MONITOR_CENTER_X, SCREEN_HEIGHT/2 - 150)
        if self.game_mode == "multi":
            self._draw_text(f"{self.score} : {self.attacker_score}", self.fonts["score"], BLACK, MAIN_MONITOR_CENTER_X, SCREEN_HEIGHT/2)
        else:
            self._draw_text(f"FINAL SCORE: {self.score}", self.fonts["score"], BLACK, MAIN_MONITOR_CENTER_X, SCREEN_HEIGHT/2)
            self._draw_text(f"HIGH SCORE: {self.highscore}", self.fonts["score"], GOLD_COLOR, MAIN_MONITOR_CENTER_X, SCREEN_HEIGHT/2 + 80)
    
    def reset_game_state(self, full_reset: bool = True):
        self.countdown_start=None; self.selected_col=None; self.final_col=None; self.ball_col=None; self.is_success=False
        self.result_time=None; self.gif_start_time=None; self.waiting_for_start=False; self.synthesized_frames=None
        self.synthesis_info=None; self.end_video=None; self.final_rank=""; self.is_capturing_face=False; self.attacker_capture_sent = False
        self.goalkeeper_face_data_buffer=[]; self.last_goalkeeper_face_coords=None
        self.attacker_face_data_buffer=[]; self.last_attacker_face_coords=None; self.attacker_selected_col = None
        if full_reset:
            self.chances_left=TOTAL_CHANCES; self.score=0; self.attacker_score=0; self.highscore=self._load_highscore()
            self.captured_goalkeeper_face=None; self.captured_attacker_face=None
    
    def start_new_round(self):
        self.reset_game_state(full_reset=False)
        if self.shoot_video: self.shoot_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.waiting_for_start = True
        self.start_transition("play")
    
    def start_game(self, mode: str):
        if self.sounds.get("button"): self.sounds["button"].play()
        self.game_mode = mode; self.reset_game_state(full_reset=True); self.start_transition("face_capture")
        
    def go_to_menu(self): self.reset_game_state(); self.start_transition("menu")
    def go_to_game_select(self): self.reset_game_state(); self.start_transition("game_select")

    def _prepare_for_end_screen(self):
        face_path, gif_path, monitor_size = None, None, (GOALKEEPER_MONITOR_WIDTH, SCREEN_HEIGHT)
        if self.game_mode == 'multi':
            if self.score > self.attacker_score:
                self.final_rank, self.end_video, face_path, gif_path = "GOALKEEPER WINS!", self.videos.get("victory"), self.captured_goalkeeper_face, os.path.join(IMAGE_PATH, "final_ronaldo", "goalkeeper_win.gif")
            elif self.attacker_score > self.score:
                self.final_rank, self.end_video, face_path, gif_path = "ATTACKER WINS!", self.videos.get("defeat"), self.captured_attacker_face, os.path.join(IMAGE_PATH, "final_ronaldo", "attacker_win.gif")
            else: self.final_rank, self.end_video = "DRAW", self.videos.get("defeat")
        else: # 1인 플레이
            if self.score > self.highscore: self.highscore = self.score; self._save_highscore(self.score)
            score = self.score
            if score >= 3: self.final_rank, self.end_video, gif_path = "Pro Keeper", self.videos.get("victory"), os.path.join(IMAGE_PATH, "final_ronaldo", "goalkeeper_win.gif")
            elif score >= 1: self.final_rank, self.end_video, gif_path = "Rookie Keeper", self.videos.get("defeat"), os.path.join(IMAGE_PATH, "lose_goalkeeper.gif")
            else: self.final_rank, self.end_video, gif_path = "Human Sieve", self.videos.get("defeat"), os.path.join(IMAGE_PATH, "lose_goalkeeper.gif")
            face_path = self.captured_goalkeeper_face

        if face_path and gif_path:
            self.synthesis_info = {"face_path": face_path, "gif_path": gif_path, "monitor_size": monitor_size}
            self.start_transition("synthesizing")
        else:
            if self.end_video: self.end_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.start_transition("end")

    def _safe_load_font(self, path, size, default_size):
        try: return pygame.font.Font(path, size)
        except (pygame.error, FileNotFoundError): return pygame.font.Font(None, default_size)

    def _init_serial(self, port, name):
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=0)
            print(f"{name} 보드({port}) 연결 성공"); return ser
        except serial.SerialException as e: print(f"오류: {name} 보드({port}) 연결 실패 - {e}"); return None

    def _init_camera(self, index, name):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened(): print(f"경고: 카메라 {index}({name}용) 열 수 없음")
        return cap
    
    def _load_gif_frames(self, path):
        frames = []
        try:
            cap = cv2.VideoCapture(path)
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1)))
            cap.release()
        except Exception as e: print(f"GIF 로드 오류 '{path}': {e}")
        return frames

    def _load_highscore(self):
        try:
            with open(HIGHSCORE_FILE, "r") as f: return int(f.read())
        except (FileNotFoundError, ValueError): return 0

    def _save_highscore(self, score):
        try:
            with open(HIGHSCORE_FILE, "w") as f: f.write(str(score))
        except IOError as e: print(f"최고 점수 저장 실패: {e}")
    
    def _send_uart_command(self, ser, command):
        byte = UART_COMMANDS.get(command)
        if byte and ser and ser.is_open:
            try: ser.write(bytes([byte]))
            except Exception as e: print(f"UART({command}) 송신 오류: {e}")

    def _draw_video_background(self, cap, start_x, width):
        if not cap or not cap.isOpened(): return
        ret, frame = cap.read()
        if not ret: cap.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, SCREEN_HEIGHT))
            self.screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (start_x, 0))
    
    def _draw_text(self, text, font, color, center_x, center_y):
        surf = font.render(text, True, color); self.screen.blit(surf, surf.get_rect(center=(center_x, center_y)))

    def _capture_and_crop_face(self, frame, raw_coords, filename):
        if frame is None: return None
        try:
            h, w, _ = frame.shape
            cx, cy, radius = raw_coords[0], raw_coords[1], 150
            
            bgra_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            bgra_frame[:, :, 3] = mask
            
            x1, y1 = max(0, cx - radius), max(0, cy - radius)
            x2, y2 = min(w, cx + radius), min(h, cy + radius)
            cropped_bgra = bgra_frame[y1:y2, x1:x2]
            
            pygame.image.save(pygame.image.frombuffer(cropped_bgra.tobytes(), cropped_bgra.shape[1::-1], "RGBA"), filename)
            print(f"얼굴 캡처 성공: {filename}"); return filename
        except Exception as e: print(f"얼굴 캡처/저장 오류: {e}"); return None

    def _handle_face_uart(self, ser, player_type):
        if not ser or ser.in_waiting <= 0: return
        
        buffer = getattr(self, f"{player_type}_face_data_buffer")
        uart_bytes = ser.read(ser.in_waiting)
        for byte in uart_bytes:
            if (byte >> 5) == 2: buffer.append(byte & 31)

        while len(buffer) >= 4:
            chunks = buffer[:4]
            full_data = (chunks[0] << 15) | (chunks[1] << 10) | (chunks[2] << 5) | chunks[3]
            x_raw, y_raw = (full_data >> 10) & 0x3FF, full_data & 0x3FF
            
            start_x = GOALKEEPER_START_X if player_type == "goalkeeper" else ATTACKER_START_X
            width = GOALKEEPER_MONITOR_WIDTH if player_type == "goalkeeper" else ATTACKER_MONITOR_WIDTH
            x_scaled = start_x + (width - int(x_raw * (width / 640)))
            y_scaled = int(y_raw * (screen_height / 480))

            coords_attr = f"last_{player_type}_face_coords"
            setattr(self, coords_attr, {"raw": (x_raw, y_raw), "scaled": (x_scaled, y_scaled)})
            
            center_x = GOALKEEPER_MONITOR_CENTER_X if player_type == "goalkeeper" else ATTACKER_MONITOR_CENTER_X
            capture_area = pygame.Rect(center_x - 100, screen_height // 2 - 350, 200, 200)

            if capture_area.collidepoint(x_scaled, y_scaled):
                frame = self.last_goalkeeper_frame if player_type == "goalkeeper" else self.last_attacker_frame
                filename = self._capture_and_crop_face(frame, (x_raw, y_raw), f"captured_{player_type}_face.png")
                if filename:
                    setattr(self, f"captured_{player_type}_face", filename)
                    if self.game_mode == "single" or (self.captured_goalkeeper_face and self.captured_attacker_face):
                        self.start_new_round()
            buffer = buffer[4:]
        setattr(self, f"{player_type}_face_data_buffer", buffer)

    def _handle_grid_uart(self, ser, player_type):
        if not ser or ser.in_waiting <= 0: return
        attr = "selected_col" if player_type == "goalkeeper" else "attacker_selected_col"
        header_val = 1 if player_type == "goalkeeper" else 3
        try:
            uart_bytes = ser.read(ser.in_waiting)
            for byte in uart_bytes:
                if (byte >> 5) == header_val:
                    value = byte & 31
                    if 1 <= value <= 5: setattr(self, attr, 5 - value)
        except Exception as e: print(f"UART({player_type} Grid) 수신 오류: {e}")

    def _draw_player_capture_panel(self, cap, captured_face, name, start_x, width, last_coords):
        center_x = start_x + width / 2
        frame_attr = "last_goalkeeper_frame" if name == "골키퍼" else "last_attacker_frame"
        
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                setattr(self, frame_attr, frame)
                frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                cam_surf = pygame.transform.scale(pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1)), (width, SCREEN_HEIGHT))
                self.screen.blit(cam_surf, (start_x, 0))
        
        if not captured_face:
            overlay = pygame.Surface((width, screen_height), pygame.SRCALPHA); overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (start_x, 0))
            self._draw_text(f"{name} 얼굴 캡처", self.fonts["title"], WHITE, center_x, SCREEN_HEIGHT/2 - 80)
            self._draw_text("얼굴을 중앙 사각형에 맞춰주세요", self.fonts["default"], WHITE, center_x, SCREEN_HEIGHT/2 + 40)
            pygame.draw.rect(self.screen, GRID_COLOR, (center_x - 100, SCREEN_HEIGHT // 2 - 350, 200, 200), 3, border_radius=15)
            if last_coords: pygame.draw.circle(self.screen, RED, last_coords["scaled"], 20, 4)
        else:
            self._draw_placeholder_panel(f"{name}\n캡처 완료!", start_x, width)

    def _draw_player_play_panel(self, player_type):
        is_gk = player_type == "goalkeeper"
        cap = self.cap_goalkeeper if is_gk else self.cap_attacker
        start_x = GOALKEEPER_START_X if is_gk else ATTACKER_START_X
        width = GOALKEEPER_MONITOR_WIDTH if is_gk else ATTACKER_MONITOR_WIDTH
        score = self.score if is_gk else self.attacker_score
        selected_col_val = self.selected_col if is_gk else self.attacker_selected_col
        
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                cam_surf = pygame.transform.scale(pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1)), (width, SCREEN_HEIGHT))
                self.screen.blit(cam_surf, (start_x, 0))
        
        self._draw_text(f"SCORE: {score}", self.fonts["score"], WHITE, start_x + width - 150, 50)
        self._draw_text("CHANCES", self.fonts["small"], WHITE, start_x + width - 150, 120)
        for i in range(self.chances_left): self.screen.blit(self.images["scoreboard_ball"], (start_x + width - 250 + i * 40, 150))

        cell_w = width / 5
        for i in range(1, 5): pygame.draw.line(self.screen, GRID_COLOR, (start_x + i * cell_w, 0), (start_x + i * cell_w, SCREEN_HEIGHT), 2)
        
        if self.countdown_start and selected_col_val is not None:
             pygame.draw.rect(self.screen, GOLD_COLOR, (start_x + selected_col_val * cell_w, 0, cell_w, SCREEN_HEIGHT), 10)

        if self.final_col is not None and is_gk:
            highlight = pygame.Surface((cell_w, SCREEN_HEIGHT), pygame.SRCALPHA); highlight.fill(HIGHLIGHT_COLOR)
            self.screen.blit(highlight, (start_x + self.final_col * cell_w, 0))
        
        if self.ball_col is not None:
            ball_rect = self.images["ball"].get_rect(center=(start_x + self.ball_col * cell_w + cell_w / 2, SCREEN_HEIGHT / 2))
            self.screen.blit(self.images["ball"], ball_rect)

    def _draw_placeholder_panel(self, text, start_x, width):
        center_x = start_x + width/2
        pygame.draw.rect(self.screen, (20,20,20), (start_x, 0, width, SCREEN_HEIGHT))
        if text:
            for i, line in enumerate(text.split('\n')):
                self._draw_text(line, self.fonts["description"], WHITE, center_x, SCREEN_HEIGHT/2 + (i * 60))

    def _draw_shoot_video(self):
        if self.shoot_video and (self.waiting_for_start or self.countdown_start):
            if self.waiting_for_start: self.shoot_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                elapsed = pygame.time.get_ticks() - self.countdown_start
                frame_pos = int(elapsed / self.shoot_video_interval)
                if frame_pos < self.shoot_video_total_frames: self.shoot_video.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = self.shoot_video.read()
            if ret:
                self._draw_video_background(self.shoot_video, MAIN_START_X, MAIN_MONITOR_WIDTH, read_new=False, pre_read_frame=frame)

    def _draw_result_gif(self):
        gif_key = 'success' if self.is_success else 'failure'
        frames = self.gif_frames.get(gif_key)
        if frames:
            frame_idx = int(((pygame.time.get_ticks() - self.gif_start_time) / GIF_FRAME_DURATION) % len(frames))
            frame_surf = pygame.transform.scale(frames[frame_idx], (MAIN_MONITOR_WIDTH, SCREEN_HEIGHT))
            self.screen.blit(frame_surf, (MAIN_START_X, 0))

    def cleanup(self):
        print("게임을 종료하며 리소스를 해제합니다...")
        if self.cap_goalkeeper: self.cap_goalkeeper.release()
        if self.cap_attacker: self.cap_attacker.release()
        if self.ser_goalkeeper: self.ser_goalkeeper.close()
        if self.ser_attacker: self.ser_attacker.close()
        if self.shoot_video: self.shoot_video.release()
        for video in self.videos.values():
            if video: video.release()
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    game = Game()
    game.run()