# main.py
import pygame
import sys
import cv2
import numpy as np
import random
import serial
import os
import time
from typing import Dict, List, Optional, Tuple

import Photofunia
from Button import ImageButton, MenuButton
from Config import *

class Game:
    """Penalty Kick Challenge 게임의 메인 클래스"""
    
    def __init__(self):
        """게임 초기화"""
        # Pygame 및 화면 설정
        pygame.init()
        pygame.mixer.init()
        os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.NOFRAME)
        pygame.display.set_caption("Penalty Kick Challenge")
        self.clock = pygame.time.Clock()
        self.running = True

        # 게임 상태 변수
        self.screen_state = "menu"
        self.game_mode = "single"
        self.reset_game_state(full_reset=True)

        # 리소스 로딩
        self._load_fonts()
        self._load_assets()
        
        # 하드웨어 연결
        self.ser_goalkeeper = self._init_serial(GOALKEEPER_SERIAL_PORT, "골키퍼")
        self.ser_attacker = self._init_serial(ATTACKER_SERIAL_PORT, "공격수")
        self.cap_goalkeeper = self._init_camera(GOALKEEPER_CAM_INDEX, "골키퍼")
        self.cap_attacker = self._init_camera(ATTACKER_CAM_INDEX, "공격수")

        # 버튼 생성
        self.buttons = self._create_buttons()
        
        # 화면 전환 효과
        self.transition_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT)); self.transition_surface.fill(BLACK)
        self.transition_alpha = 0
        self.fading_out = False
        self.fading_in = False
        self.transition_target = None
        
        if self.sounds.get("bg_thumbnail"):
            self.sounds["bg_thumbnail"].play(-1)

    # --- 초기화 메서드들 ---
    
    def _load_fonts(self):
        """폰트 로딩"""
        self.fonts = {
            "default": self._safe_load_font(FONT_PATH, 40, 50), "small": self._safe_load_font(FONT_PATH, 30, 40),
            "description": self._safe_load_font(FONT_PATH, 50, 60), "title": self._safe_load_font(FONT_BOLD_PATH, 120, 130),
            "countdown": self._safe_load_font(FONT_PATH, 200, 250), "score": self._safe_load_font(FONT_BOLD_PATH, 60, 70),
            "rank": self._safe_load_font(FONT_BOLD_PATH, 100, 110),
        }

    def _load_assets(self):
        """이미지, 사운드, 비디오 등 모든 에셋 로딩"""
        self.images = {}; self.sounds = {}; self.videos = {}; self.gif_frames = {}
        try:
            # 사운드
            self.sounds["button"] = pygame.mixer.Sound(os.path.join(SOUND_PATH, "button_click.wav"))
            self.sounds["siu"] = pygame.mixer.Sound(os.path.join(SOUND_PATH, "SIUUUUU.wav"))
            self.sounds["success"] = pygame.mixer.Sound(os.path.join(SOUND_PATH, "야유.mp3"))
            self.sounds["bg_thumbnail"] = pygame.mixer.Sound(os.path.join(SOUND_PATH, "Time_Bomb.mp3"))
            self.sounds["failed"] = self.sounds["siu"]
            
            # 이미지
            ball_img = pygame.image.load(os.path.join(IMAGE_PATH, "final_ronaldo", "Ball.png")).convert_alpha()
            self.images["scoreboard_ball"] = pygame.transform.scale(ball_img, (80, 80))
            self.images["ball"] = pygame.transform.scale(ball_img, (200, 200))
            self.images["info_bg"] = pygame.transform.scale(pygame.image.load(os.path.join(IMAGE_PATH, "info", "info_back2.jpg")).convert(), (MAIN_MONITOR_WIDTH, SCREEN_HEIGHT))

            # GIF 프레임
            self.gif_frames['success'] = self._load_gif_frames(os.path.join(IMAGE_PATH, "final_ronaldo", "pk.gif"), (MAIN_MONITOR_WIDTH, SCREEN_HEIGHT))
            self.gif_frames['failure'] = self._load_gif_frames(os.path.join(IMAGE_PATH, "G.O.A.T", "siuuu.gif"), (MAIN_MONITOR_WIDTH, SCREEN_HEIGHT))

            # 비디오
            self.videos = {name: cv2.VideoCapture(os.path.join(IMAGE_PATH, f"{filename}.{ext}")) for name, filename, ext in [
                ("lose", "lose_keeper", "gif"), ("victory", "victory", "gif"), ("defeat", "defeat", "gif"), 
                ("game_bg", "Ground1", "mp4"), ("menu_bg", "game_thumbnail", "mp4"), ("shoot", "shoot", "gif")
            ]}
        except (pygame.error, FileNotFoundError) as e: print(f"에셋 로딩 중 오류 발생: {e}")

    def _create_buttons(self) -> Dict[str, List]:
        """화면 상태별 버튼 생성"""
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
            "info": [ImageButton(os.path.join(IMAGE_PATH, "btn_exit.png"), MAIN_MONITOR_CENTER_X*2 - 150, 150, 100, 100, self.go_to_game_select, sound=btn_sound)],
            "end": [
                ImageButton(os.path.join(IMAGE_PATH, "btn_restart.png"), MAIN_MONITOR_CENTER_X - 300, SCREEN_HEIGHT - 250, 400, 250, self.go_to_game_select, sound=btn_sound),
                ImageButton(os.path.join(IMAGE_PATH, "btn_main_menu.png"), MAIN_MONITOR_CENTER_X + 300, SCREEN_HEIGHT - 250, 400, 250, self.go_to_menu, sound=btn_sound)
            ]
        }

    # --- 메인 루프 및 이벤트 처리 ---

    def run(self):
        """메인 게임 루프"""
        while self.running:
            self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(FPS)
        self.cleanup()

    def handle_events(self):
        """사용자 입력(이벤트) 처리"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): self.running = False
            if self.fading_in or self.fading_out: continue
            
            if self.screen_state == "menu" and event.type == pygame.KEYDOWN: self.start_transition("game_select")
            elif self.screen_state == "play" and self.waiting_for_start and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                self.countdown_start, self.waiting_for_start = pygame.time.get_ticks(), False
            
            for button in self.buttons.get(self.screen_state, []): button.handle_event(event)

    def update(self):
        """게임 상태 업데이트"""
        if self.fading_out or self.fading_in:
            if self.fading_out:
                self.transition_alpha = min(255, self.transition_alpha + TRANSITION_SPEED)
                if self.transition_alpha == 255: self.fading_out, self.fading_in, self.screen_state = False, True, self.transition_target
            else:
                self.transition_alpha = max(0, self.transition_alpha - TRANSITION_SPEED)
                if self.transition_alpha == 0: self.fading_in = False
        else:
             for button in self.buttons.get(self.screen_state, []): button.update()
        
        if self.screen_state == 'play': self._update_play_state(pygame.time.get_ticks())
        elif self.screen_state == 'face_capture': self._update_face_capture_state()

    def draw(self):
        """화면 그리기"""
        self.screen.fill(BLACK)
        draw_function = getattr(self, f"_draw_{self.screen_state}_screen", self._draw_placeholder_screen)
        draw_function()
        
        if not (self.fading_in or self.fading_out):
            for button in self.buttons.get(self.screen_state, []): button.draw(self.screen)
        
        if self.fading_in or self.fading_out:
            self.transition_surface.set_alpha(self.transition_alpha)
            self.screen.blit(self.transition_surface, (0, 0))

        pygame.display.flip()

    # --- 화면별 그리기 및 업데이트 메서드 ---
    
    def _draw_menu_screen(self):
        self._draw_video_background(self.videos["menu_bg"], MAIN_START_X, MAIN_MONITOR_WIDTH)
        self._draw_text("Penalty Kick Challenge", self.fonts["title"], WHITE, MAIN_MONITOR_CENTER_X, SCREEN_HEIGHT * 0.3)
        self._draw_text("PRESS ANY KEY", self.fonts["description"], WHITE, MAIN_MONITOR_CENTER_X, SCREEN_HEIGHT * 0.75)

    def _draw_game_select_screen(self): self._draw_video_background(self.videos["game_bg"], MAIN_START_X, MAIN_MONITOR_WIDTH)
    
    def _update_face_capture_state(self):
        # (이 부분은 UART 신호에 따라 얼굴 좌표를 받고 처리하는 복잡한 로직 대신, 버튼 클릭으로 캡처하는 방식으로 단순화했습니다.)
        # (기존 UART 로직을 사용하려면, 원본 코드의 UART 데이터 파싱 부분을 여기에 통합해야 합니다.)
        if not self.is_capturing_face:
            self._send_uart_command(self.ser_goalkeeper, 'face')
            self.is_capturing_face = True
        
        if not self.captured_goalkeeper_face:
            if self.ser_goalkeeper and self.ser_goalkeeper.in_waiting > 0:
                if self.ser_goalkeeper.read(1) == b'\x01': # 버튼 클릭 신호 (예시)
                    self._capture_face(self.cap_goalkeeper, "goalkeeper")
        
        elif self.game_mode == "multi" and not self.captured_attacker_face:
            if not self.attacker_capture_sent:
                self._send_uart_command(self.ser_attacker, 'face')
                self.attacker_capture_sent = True
            if self.ser_attacker and self.ser_attacker.in_waiting > 0:
                if self.ser_attacker.read(1) == b'\x01':
                    self._capture_face(self.cap_attacker, "attacker")

        if self.captured_goalkeeper_face and (self.game_mode == "single" or self.captured_attacker_face):
            self.start_new_round()

    def _draw_face_capture_screen(self):
        self.screen.fill(BLACK)
        self._draw_player_capture_panel(self.cap_goalkeeper, self.captured_goalkeeper_face, "골키퍼", GOALKEEPER_START_X, GOALKEEPER_MONITOR_WIDTH, GOALKEEPER_MONITOR_CENTER_X)
        if self.game_mode == "multi":
            if not self.captured_goalkeeper_face:
                self._draw_placeholder_panel("골키퍼 대기 중...", ATTACKER_START_X, ATTACKER_MONITOR_WIDTH, ATTACKER_MONITOR_CENTER_X)
            else:
                self._draw_player_capture_panel(self.cap_attacker, self.captured_attacker_face, "공격수", ATTACKER_START_X, ATTACKER_MONITOR_WIDTH, ATTACKER_MONITOR_CENTER_X)
        else: self._draw_placeholder_panel("", ATTACKER_START_X, ATTACKER_MONITOR_WIDTH, ATTACKER_MONITOR_CENTER_X)

    def _update_play_state(self, current_time):
        if self.countdown_start and not self.result_time:
            self._send_uart_command(self.ser_goalkeeper, 'grid')
            # (UART로 그리드 값 받는 로직 추가)
            if current_time - self.countdown_start > COUNTDOWN_SECONDS * 1000:
                self.final_col = self.selected_col if self.selected_col is not None else random.randint(0, 4)
                self.ball_col = random.randint(0, 4) # 2인 모드일 경우 공격수 선택 값으로 변경
                self.is_success = (self.final_col == self.ball_col) # 성공 로직 수정
                self.result_time = current_time
                self.chances_left -= 1
                
                if self.is_success:
                    self.score += 1
                    if self.sounds.get("success"): self.sounds["success"].play()
                else:
                    if self.game_mode == 'multi': self.attacker_score += 1
                    if self.sounds.get("failed"): self.sounds["failed"].play()
                self.countdown_start = None

        if self.result_time and not self.gif_start_time and (current_time - self.result_time > RESULT_DELAY_MS): self.gif_start_time = current_time
        if self.gif_start_time and (current_time - self.gif_start_time > GIF_PLAY_DURATION_MS):
            if self.chances_left > 0: self.start_new_round()
            else: self._prepare_for_end_screen()

    def _draw_play_screen(self):
        self._draw_video_background(self.videos["shoot"], MAIN_START_X, MAIN_MONITOR_WIDTH)
        self._draw_player_play_panel(self.cap_goalkeeper, "goalkeeper", GOALKEEPER_START_X, GOALKEEPER_MONITOR_WIDTH)
        if self.game_mode == "multi": self._draw_player_play_panel(self.cap_attacker, "attacker", ATTACKER_START_X, ATTACKER_MONITOR_WIDTH)
        else: self._draw_placeholder_panel("", ATTACKER_START_X, ATTACKER_MONITOR_WIDTH, ATTACKER_MONITOR_CENTER_X)

        if self.waiting_for_start:
            self._draw_text("PRESS SPACE BAR", self.fonts["description"], WHITE, MAIN_MONITOR_CENTER_X, SCREEN_HEIGHT / 2)
        
        if self.countdown_start and not self.result_time:
            num_str = str(COUNTDOWN_SECONDS - (pygame.time.get_ticks() - self.countdown_start) // 1000)
            self._draw_text(num_str, self.fonts["countdown"], WHITE, GOALKEEPER_MONITOR_CENTER_X, SCREEN_HEIGHT / 2)
            if self.game_mode == "multi": self._draw_text(num_str, self.fonts["countdown"], WHITE, ATTACKER_MONITOR_CENTER_X, SCREEN_HEIGHT / 2)

        if self.gif_start_time:
            gif_key = 'success' if self.is_success else 'failure'
            frames = self.gif_frames.get(gif_key)
            if frames:
                frame_idx = int(((pygame.time.get_ticks() - self.gif_start_time) / GIF_FRAME_DURATION) % len(frames))
                self.screen.blit(frames[frame_idx], (MAIN_START_X, 0)) # 중앙 모니터에 결과 GIF
    
    def _draw_info_screen(self): self.screen.blit(self.images["info_bg"], (0,0))

    def _draw_synthesizing_screen(self):
        self.screen.fill(BLACK)
        if not self.synthesized_frames and self.synthesis_info:
            pygame.display.flip() # "합성 중" 텍스트 먼저 표시
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
            if self.game_mode == "multi" and self.synthesis_info.get("face_path_attacker"):
                 self.screen.blit(self.synthesized_frames[frame_idx], (ATTACKER_START_X, 0))

        self._draw_text(self.final_rank, self.fonts["rank"], GOLD_COLOR, MAIN_MONITOR_CENTER_X, SCREEN_HEIGHT/2 - 150)
        if self.game_mode == "multi":
            score_str = f"{self.score} : {self.attacker_score}"
            self._draw_text(score_str, self.fonts["score"], BLACK, MAIN_MONITOR_CENTER_X, SCREEN_HEIGHT/2)
        else:
            self._draw_text(f"FINAL SCORE: {self.score}", self.fonts["score"], BLACK, MAIN_MONITOR_CENTER_X, SCREEN_HEIGHT/2)
            self._draw_text(f"HIGH SCORE: {self.highscore}", self.fonts["score"], GOLD_COLOR, MAIN_MONITOR_CENTER_X, SCREEN_HEIGHT/2 + 80)
    
    def _draw_placeholder_screen(self): self._draw_text(f"Unknown State: {self.screen_state}", self.fonts["default"], RED, SCREEN_WIDTH/2, SCREEN_HEIGHT/2)

    # --- 상태 관리 및 전환 메서드 ---
    
    def start_transition(self, target_state: str): self.transition_target, self.fading_out, self.fading_in = target_state, True, False
            
    def reset_game_state(self, full_reset: bool = True):
        self.countdown_start=None; self.selected_col=None; self.final_col=None; self.ball_col=None; self.is_success=False
        self.result_time=None; self.gif_start_time=None; self.waiting_for_start=False; self.synthesized_frames=None
        self.synthesis_info=None; self.end_video=None; self.final_rank=""; self.is_capturing_face=False; self.attacker_capture_sent = False
        if full_reset:
            self.chances_left=TOTAL_CHANCES; self.score=0; self.attacker_score=0; self.highscore=self._load_highscore()
            self.captured_goalkeeper_face=None; self.captured_attacker_face=None
    
    def start_new_round(self):
        self.reset_game_state(full_reset=False)
        self.waiting_for_start = True
        self.start_transition("play")
    
    def start_game(self, mode: str):
        if self.sounds.get("button"): self.sounds["button"].play()
        self.game_mode = mode; self.reset_game_state(full_reset=True); self.start_transition("face_capture")
        
    def go_to_menu(self): self.start_transition("menu")
        
    def go_to_game_select(self): self.reset_game_state(full_reset=True); self.start_transition("game_select")

    def _prepare_for_end_screen(self):
        face_path, gif_path, monitor_size = None, None, (GOALKEEPER_MONITOR_WIDTH, SCREEN_HEIGHT)
        if self.game_mode == 'multi':
            if self.score > self.attacker_score:
                self.final_rank, self.end_video = "GOALKEEPER WINS!", self.videos["victory"]
                face_path, gif_path = self.captured_goalkeeper_face, os.path.join(IMAGE_PATH, "final_ronaldo", "goalkeeper_win.gif")
            elif self.attacker_score > self.score:
                self.final_rank, self.end_video = "ATTACKER WINS!", self.videos["defeat"]
                face_path, gif_path = self.captured_attacker_face, os.path.join(IMAGE_PATH, "final_ronaldo", "attacker_win.gif")
            else: self.final_rank, self.end_video = "DRAW", self.videos["defeat"]
        else: # 1인 플레이
            if self.score > self.highscore: self.highscore = self.score; self._save_highscore(self.score)
            if self.score >= 3: self.final_rank, self.end_video, gif_path = "Pro Keeper", self.videos["victory"], os.path.join(IMAGE_PATH, "final_ronaldo", "goalkeeper_win.gif")
            elif self.score >= 1: self.final_rank, self.end_video, gif_path = "Rookie Keeper", self.videos["defeat"], os.path.join(IMAGE_PATH, "lose_goalkeeper.gif")
            else: self.final_rank, self.end_video, gif_path = "Human Sieve", self.videos["defeat"], os.path.join(IMAGE_PATH, "lose_goalkeeper.gif")
            face_path = self.captured_goalkeeper_face

        if face_path and gif_path:
            self.synthesis_info = {"face_path": face_path, "gif_path": gif_path, "monitor_size": monitor_size}
            self.start_transition("synthesizing")
        else: self.start_transition("end")

    # --- 유틸리티 및 헬퍼 메서드 ---

    def _safe_load_font(self, path, size, default_size):
        try: return pygame.font.Font(path, size)
        except (pygame.error, FileNotFoundError): return pygame.font.Font(None, default_size)

    def _init_serial(self, port, name):
        try:
            ser = serial.Serial(port, BAUD_RATE, timeout=0)
            print(f"{name} 보드({port})가 성공적으로 연결되었습니다."); return ser
        except serial.SerialException as e: print(f"오류: {name} 보드({port})를 열 수 없습니다 - {e}"); return None

    def _init_camera(self, index, name):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened(): print(f"경고: 카메라 {index}({name}용)를 열 수 없습니다.")
        return cap
    
    def _load_gif_frames(self, path, size):
        frames = []
        try:
            cap = cv2.VideoCapture(path)
            while True:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                frames.append(pygame.surfarray.make_surface(frame.swapaxes(0, 1)))
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
        byte_to_send = UART_COMMANDS.get(command)
        if byte_to_send and ser and ser.is_open:
            try: ser.write(bytes([byte_to_send]))
            except Exception as e: print(f"UART({command}) 데이터 송신 오류: {e}")

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

    def _capture_face(self, cap, player_type):
        if not cap or not cap.isOpened(): return
        ret, frame = cap.read()
        if ret:
            filename = f"captured_{player_type}_{int(time.time())}.png"
            cv2.imwrite(filename, frame) # 전체 프레임 저장
            if player_type == "goalkeeper": self.captured_goalkeeper_face = filename
            else: self.captured_attacker_face = filename
            print(f"{player_type} 얼굴 캡처 완료: {filename}")

    def _draw_player_capture_panel(self, cap, captured_face_path, name, start_x, width, center_x):
        if captured_face_path:
            self._draw_placeholder_panel("캡처 완료!", start_x, width, center_x)
        else:
            if cap and cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                    cam_surf = pygame.transform.scale(pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1)), (width, SCREEN_HEIGHT))
                    self.screen.blit(cam_surf, (start_x, 0))
            self._draw_text(f"{name} 얼굴 캡처", self.fonts["title"], WHITE, center_x, SCREEN_HEIGHT/2 - 80)
            self._draw_text("버튼을 눌러주세요", self.fonts["default"], WHITE, center_x, SCREEN_HEIGHT/2 + 40)
            pygame.draw.rect(self.screen, GRID_COLOR, (center_x - 100, SCREEN_HEIGHT // 2 - 350, 200, 200), 3, border_radius=15)

    def _draw_player_play_panel(self, cap, player_type, start_x, width):
        if cap and cap.isOpened():
            ret, frame = cap.read()
            if ret:
                frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                cam_surf = pygame.transform.scale(pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1)), (width, SCREEN_HEIGHT))
                self.screen.blit(cam_surf, (start_x, 0))
        
        score = self.score if player_type == "goalkeeper" else self.attacker_score
        self._draw_text(f"SCORE: {score}", self.fonts["score"], WHITE, start_x + width - 150, 50)
        self._draw_text(f"CHANCES", self.fonts["small"], WHITE, start_x + width - 150, 120)
        for i in range(self.chances_left): self.screen.blit(self.images["scoreboard_ball"], (start_x + width - 250 + i * 40, 150))

        cell_w = width / 5
        for i in range(1, 5): pygame.draw.line(self.screen, GRID_COLOR, (start_x + i * cell_w, 0), (start_x + i * cell_w, SCREEN_HEIGHT), 2)
        
        if self.final_col is not None and player_type == "goalkeeper":
            highlight_surf = pygame.Surface((cell_w, SCREEN_HEIGHT), pygame.SRCALPHA); highlight_surf.fill(HIGHLIGHT_COLOR)
            self.screen.blit(highlight_surf, (start_x + self.final_col * cell_w, 0))
        
        if self.ball_col is not None:
            ball_rect = self.images["ball"].get_rect(center=(start_x + self.ball_col * cell_w + cell_w / 2, SCREEN_HEIGHT / 2))
            self.screen.blit(self.images["ball"], ball_rect)

    def _draw_placeholder_panel(self, text, start_x, width, center_x):
        pygame.draw.rect(self.screen, (20,20,20), (start_x, 0, width, SCREEN_HEIGHT))
        if text: self._draw_text(text, self.fonts["description"], WHITE, center_x, SCREEN_HEIGHT/2)

    def cleanup(self):
        print("게임을 종료하며 리소스를 해제합니다...")
        if self.cap_goalkeeper: self.cap_goalkeeper.release()
        if self.cap_attacker: self.cap_attacker.release()
        if self.ser_goalkeeper: self.ser_goalkeeper.close()
        if self.ser_attacker: self.ser_attacker.close()
        for video in self.videos.values():
            if video: video.release()
        pygame.quit()
        sys.exit()

if __name__ == '__main__':
    game = Game()
    game.run()