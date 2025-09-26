import pygame
import sys
import cv2
import numpy as np
import random
import os
import serial  

from button import ImageButton

# ========================================================== #
# 클래스에 속하지 않는 전역 유틸리티 함수
# ========================================================== #
def send_uart_command(serial_port, command):
    """FPGA로 정해진 UART 명령어를 전송하는 함수."""
    if not serial_port or not serial_port.is_open:
        return
    commands = {
        'grid': 225,  
        'face': 226,  
        'kick': 227,  
        'stop': 0     
    }
    byte_to_send = commands.get(command)
    if byte_to_send is not None:
        try:
            serial_port.write(bytes([byte_to_send]))
        except Exception as e:
            print(f"UART({command}) 데이터 송신 오류: {e}")

def load_highscore():
    if not os.path.exists("highscore.txt"): return 0
    try:
        with open("highscore.txt", "r") as f: return int(f.read())
    except (ValueError, IOError): return 0

def save_highscore(new_score):
    try:
        with open("highscore.txt", "w") as f: f.write(str(new_score))
    except Exception as e: print(f"최고 기록 저장 오류: {e}")

def get_scaled_rect(original_w, original_h, container_w, container_h):
    if original_h == 0 or container_h == 0: return (0, 0)
    aspect_ratio = original_w / original_h
    container_aspect_ratio = container_w / container_h
    if aspect_ratio > container_aspect_ratio:
        new_w, new_h = container_w, int(container_w / aspect_ratio)
    else:
        new_w, new_h = int(container_h * aspect_ratio), container_h
    return new_w, new_h

# ========================================================== #
# 게임의 모든 것을 관리하는 Game 클래스
# ========================================================== #
class PenaltyKickGame:
    """페널티킥 챌린지 게임의 모든 상태, 리소스, 로직을 관리하는 메인 클래스."""
    def __init__(self):
        """ 게임 객체 초기화: 모든 설정, 리소스 로딩, 변수 선언 """
        pygame.init()
        try:
            pygame.mixer.init()
        except Exception as e:
            print(f"Mixer 초기화 실패: {e}")

        # 1. 설정값 중앙 관리 (Data-Driven Design)
        self.CONFIG = {
            "screen": {"pos": "0,0"},
            "colors": {
                'BLACK': (0, 0, 0), 'WHITE': (255, 255, 255), 'GRID': (0, 255, 0),
                'RED': (255, 0, 0), 'HIGHLIGHT': (255, 0, 0, 100), 'GOLD': (255, 215, 0)
            },
            "paths": {
                "font_main": "../fonts/netmarbleM.ttf",
                "font_bold": "../fonts/netmarbleB.ttf",
                "img_ball": "../image/final_ronaldo/Ball.png",
                "img_info_bg": "../image/info/info_back2.jpg",
                "sound_button": "../sound/button_click.wav",
                "sound_siu": "../sound/SIUUUUU.wav",
                "sound_success": "../sound/야유.mp3",
                "vid_failure": "../image/son.mp4",
                "vid_success": "../image/final_ronaldo/pk.gif",
                "vid_victory": "../image/victory.gif",
                "vid_defeat": "../image/defeat.gif",
                "vid_menu_bg": "../image/game_thumbnail.mp4",
                "vid_bg": "../image/shoot.gif"
            },
            "serial": {
                "goalkeeper_port": "COM17",
                "attacker_port": "COM13",
                "baudrate": 9600
            },
            "camera_indices": {
                "goalkeeper": 0,
                "attacker": 2
            }
        }
        self.COLORS = self.CONFIG['colors']

        self._setup_screen()
        
        self.state = {}
        self.resources = {"sounds": {}, "images": {}, "videos": {}, "gif_frames": {}, "cached_surfaces": {}}
        
        self._load_fonts()
        self._load_images()
        self._load_sounds()
        self._load_videos()
        self._connect_hardware()
        self._create_buttons()
        self._cache_static_surfaces()
        
        self._reset_game_state(full_reset=True, initial_screen="menu")

        self.clock = pygame.time.Clock()
        self.is_running = True
        self.transition_surface = pygame.Surface((self.screen_width, self.screen_height))
        self.transition_surface.fill(self.COLORS['BLACK'])

    # ---------------------------------------------------------- #
    # 1. 초기 설정 및 로딩 메서드들
    # ---------------------------------------------------------- #
    def _setup_screen(self):
        """모니터 해상도를 감지하고 Pygame 화면 및 레이아웃을 설정합니다."""
        try:
            desktop_sizes = pygame.display.get_desktop_sizes()
            total_width = sum(w for w, h in desktop_sizes)
            max_height = max(h for w, h in desktop_sizes)
        except AttributeError:
            info = pygame.display.Info()
            total_width, max_height = info.current_w, info.current_h

        os.environ['SDL_VIDEO_WINDOW_POS'] = self.CONFIG['screen']['pos']
        self.screen = pygame.display.set_mode((total_width, max_height), pygame.NOFRAME)
        pygame.display.set_caption("Penalty Kick Challenge")

        self.screen_width = self.screen.get_width()
        self.screen_height = self.screen.get_height()
        
        self.goalkeeper_monitor_width = self.screen_width // 3 
        self.main_monitor_width = self.screen_width // 3
        self.attacker_monitor_width = self.screen_width - self.goalkeeper_monitor_width - self.main_monitor_width

        self.main_start_x = 0
        self.attacker_start_x = self.attacker_monitor_width
        self.goalkeeper_start_x = self.goalkeeper_monitor_width + self.main_monitor_width

        self.goalkeeper_monitor_center_x = self.goalkeeper_start_x + (self.goalkeeper_monitor_width // 2)
        self.main_monitor_center_x = self.main_start_x + (self.main_monitor_width // 2)
        self.attacker_monitor_center_x = self.attacker_start_x + (self.attacker_monitor_width // 2)

    def _load_fonts(self):
        """폰트 파일을 로드합니다."""
        def load_font_safe(path, size, default_size):
            try: return pygame.font.Font(path, size)
            except: return pygame.font.Font(None, default_size)
        
        paths = self.CONFIG['paths']
        self.fonts = {
            'font': load_font_safe(paths['font_main'], 40, 50),
            'small_font': load_font_safe(paths['font_main'], 30, 40),
            'description_font': load_font_safe(paths['font_main'], 50, 60),
            'title_font': load_font_safe(paths['font_bold'], 120, 130),
            'countdown_font': load_font_safe(paths['font_main'], 200, 250),
            'score_font': load_font_safe(paths['font_bold'], 60, 70),
            'rank_font': load_font_safe(paths['font_bold'], 100, 110)
        }

    def _load_images(self):
        try:
            paths = self.CONFIG['paths']
            ball_img = pygame.image.load(paths['img_ball']).convert_alpha()
            self.resources["images"]["scoreboard_ball"] = pygame.transform.scale(ball_img, (80, 80))
            self.resources["images"]["ball"] = pygame.transform.scale(ball_img, (200, 200))
            self.resources["images"]["info_bg"] = pygame.transform.scale(pygame.image.load(paths['img_info_bg']).convert(), (self.screen_width, self.screen_height))
        except Exception as e:
            print(f"이미지 로딩 오류: {e}")
    
    def _load_sounds(self):
        try:
            paths = self.CONFIG['paths']
            self.resources["sounds"]["button"] = pygame.mixer.Sound(paths['sound_button'])
            self.resources["sounds"]["siu"] = pygame.mixer.Sound(paths['sound_siu'])
            self.resources["sounds"]["success"] = pygame.mixer.Sound(paths['sound_success'])
            self.resources["sounds"]["failed"] = self.resources["sounds"]["siu"]
        except Exception as e:
            print(f"사운드 로딩 오류: {e}")

    def _load_videos(self):
        paths = self.CONFIG['paths']
        self.resources["videos"] = {
            "victory": cv2.VideoCapture(paths['vid_victory']),
            "defeat": cv2.VideoCapture(paths['vid_defeat']),
            "menu_bg": cv2.VideoCapture(paths['vid_menu_bg']),
            "bg_video": cv2.VideoCapture(paths['vid_bg'])
        }
        
        gif_size = (self.main_monitor_width, self.screen_height)
        self.resources['gif_frames']['success'] = self._load_gif_frames(paths['vid_success'], gif_size)
        self.resources['gif_frames']['failure'] = self._load_gif_frames(paths['vid_failure'], gif_size)

        bg_video = self.resources["videos"]["bg_video"]
        if bg_video and bg_video.isOpened():
            self.bg_video_total_frames = int(bg_video.get(cv2.CAP_PROP_FRAME_COUNT))
            self.bg_video_w = int(bg_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.bg_video_h = int(bg_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.bg_video_interval = 7000 / self.bg_video_total_frames if self.bg_video_total_frames > 0 else 0
        else:
            self.resources["videos"]["bg_video"] = None

    def _load_gif_frames(self, video_path, size):
        """비디오 파일의 모든 프레임을 Pygame Surface 객체로 미리 변환합니다."""
        frames = []
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"경고: 비디오 파일을 열 수 없습니다: {video_path}")
            return frames
        while True:
            ret, frame = cap.read()
            if not ret: break
            
            frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            frame_pygame = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            frames.append(frame_pygame)
        
        cap.release()
        print(f"{video_path}에서 {len(frames)}개의 프레임을 로드했습니다.")
        return frames

    def _connect_hardware(self):
        """카메라와 시리얼 포트를 연결합니다."""
        indices = self.CONFIG['camera_indices']
        serial_conf = self.CONFIG['serial']
        
        self.resources['cap_goalkeeper'] = cv2.VideoCapture(indices['goalkeeper'])
        self.resources['cap_attacker'] = cv2.VideoCapture(indices['attacker'])
        if not self.resources['cap_attacker'].isOpened():
            print(f"경고: 카메라 {indices['attacker']}(공격수용)를 열 수 없습니다.")
        
        try:
            self.resources["ser_goalkeeper"] = serial.Serial(serial_conf['goalkeeper_port'], serial_conf['baudrate'], timeout=0)
            print(f"골키퍼 보드({serial_conf['goalkeeper_port']})가 성공적으로 연결되었습니다.")
        except serial.SerialException as e:
            self.resources["ser_goalkeeper"] = None
            print(f"오류: 골키퍼 보드({serial_conf['goalkeeper_port']}) - {e}")

        try:
            self.resources["ser_attacker"] = serial.Serial(serial_conf['attacker_port'], serial_conf['baudrate'], timeout=0)
            print(f"공격수 보드({serial_conf['attacker_port']})가 성공적으로 연결되었습니다.")
        except serial.SerialException as e:
            self.resources["ser_attacker"] = None
            print(f"오류: 공격수 보드({serial_conf['attacker_port']}) - {e}")
    
    def _create_buttons(self):
        """게임 화면별 버튼들을 생성합니다."""
        sound = self.resources["sounds"].get("button")
        self.buttons = {
            "menu": [ImageButton("../image/btn_desc.png", self.main_monitor_center_x*2 - 150, 150, 100, 100, lambda: self._start_transition("info"), sound=sound)],
            "game": [ImageButton("../image/btn_single.png", self.main_monitor_center_x - 300, self.screen_height//2 + 200, 550, 600, lambda: self._start_game("single")),
                     ImageButton("../image/btn_multi.png", self.main_monitor_center_x + 300, self.screen_height//2 + 200, 550, 600, lambda: self._start_game("multi")),
                     ImageButton("../image/btn_back.png", 150, 150, 100, 100, self._go_to_menu, sound=sound)],
            "face_capture": [ImageButton("../image/btn_back.png", 150, 150, 100, 100, self._go_to_game_select, sound=sound)],
            "webcam_view": [ImageButton("../image/btn_back.png", 150, 150, 100, 100, self._go_to_game_select, sound=sound)],
            "info": [ImageButton("../image/btn_exit.png", self.main_monitor_center_x*2 - 150, 150, 100, 100, self._go_to_menu, sound=sound)],
            "end": [ImageButton("../image/btn_restart.png", self.main_monitor_center_x - 300, self.screen_height - 250, 400, 250, self._go_to_game_select, sound=sound),
                    ImageButton("../image/btn_main_menu.png", self.main_monitor_center_x + 300, self.screen_height - 250, 400, 250, self._go_to_menu, sound=sound)]
        }

    def _cache_static_surfaces(self):
        """변하지 않는 UI 요소들을 미리 Surface 객체로 만들어 캐싱합니다."""
        overlay = pygame.Surface((self.goalkeeper_monitor_width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        self.resources['cached_surfaces']['player_overlay'] = overlay
        
        chances_text = self.fonts['font'].render("CHANCES", True, self.COLORS['WHITE'])
        self.resources['cached_surfaces']['chances_text'] = chances_text

    def _reset_game_state(self, full_reset=True, initial_screen=""):
        """게임 상태를 초기화합니다."""
        round_reset_state = {
            "countdown_start": None, "selected_col": None, "final_col": None, "ball_col": None,
            "is_failure": False, "is_success": False, "result_time": None, "gif_start_time": None,
            "waiting_for_start": False, "is_capturing_face": False, "attacker_selected_col": None,
            "goalkeeper_face_data_buffer": [], "last_goalkeeper_face_coords": None,
            "attacker_face_data_buffer": [], "last_attacker_face_coords": None,
            "gif_frame_index": 0
        }
        self.state.update(round_reset_state)

        if full_reset:
            full_reset_state = {
                "screen_state": initial_screen, "chances_left": 5, "score": 0, "highscore": load_highscore(),
                "attacker_score": 0, "final_rank": "", "end_video": None, "last_end_frame": None,
                "game_mode": None, "captured_goalkeeper_face_filename": None, "captured_attacker_face_filename": None,
                "transition_alpha": 0, "transition_target": None, "transition_speed": 15,
                "fading_out": False, "fading_in": False
            }
            self.state.update(full_reset_state)

    def _start_transition(self, target_state):
        if not self.state.get('fading_out') and not self.state.get('fading_in'):
            self.state['transition_target'] = target_state
            self.state['fading_out'] = True
            
    def _start_new_round(self):
        self._reset_game_state(full_reset=False)
        bg_video = self.resources["videos"].get("bg_video")
        if bg_video: bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self.state["waiting_for_start"] = True
        
    def _start_game(self, mode):
        if self.resources["sounds"].get("button"): self.resources["sounds"]["button"].play()
        self.state["game_mode"] = mode
        self._reset_game_state(full_reset=True, initial_screen="face_capture")
        self._start_transition("face_capture")
        
    def _go_to_menu(self):
        self._reset_game_state(full_reset=True, initial_screen="menu")
        self._start_transition("menu")
        
    def _go_to_game_select(self):
        self._reset_game_state(full_reset=True, initial_screen="game")
        self._start_transition("game")

    def _end_game(self):
        """게임을 종료하고 결과 화면으로 전환합니다."""
        if self.state["game_mode"] == 'multi':
            if self.state["score"] > self.state["attacker_score"]:
                self.state["winner"], self.state["final_rank"], self.state["end_video"] = "goalkeeper", "GOALKEEPER WINS!", self.resources["videos"]["victory"]
            elif self.state["attacker_score"] > self.state["score"]:
                self.state["winner"], self.state["final_rank"], self.state["end_video"] = "attacker", "ATTACKER WINS!", self.resources["videos"]["victory"]
            else:
                self.state["winner"], self.state["final_rank"], self.state["end_video"] = "draw", "DRAW", self.resources["videos"]["defeat"]
        else: # Single Player
            self.state["winner"] = "goalkeeper"
            if self.state["score"] > self.state["highscore"]:
                self.state["highscore"] = self.state["score"]
                save_highscore(self.state["score"])
            score = self.state["score"]
            if score == 5: self.state["final_rank"], self.state["end_video"] = "THE WALL", self.resources["videos"]["victory"]
            elif score >= 3: self.state["final_rank"], self.state["end_video"] = "Pro Keeper", self.resources["videos"]["victory"]
            elif score >= 1: self.state["final_rank"], self.state["end_video"] = "Rookie Keeper", self.resources["videos"]["defeat"]
            else: self.state["final_rank"], self.state["end_video"] = "Human Sieve", self.resources["videos"]["defeat"]
        
        end_video = self.state.get("end_video")
        if end_video: 
            end_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        self._start_transition("end")

    # ---------------------------------------------------------- #
    # 3. 이벤트 처리 및 업데이트
    # ---------------------------------------------------------- #
    def _handle_events(self):
        """Pygame 이벤트를 처리합니다. (키보드, 마우스 등)"""
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.is_running = False
                return

            if self.state["screen_state"] == "menu" and event.type == pygame.KEYDOWN:
                self._start_transition("game")

            elif self.state["screen_state"] == "webcam_view" and self.state["waiting_for_start"]:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.state["countdown_start"] = pygame.time.get_ticks()
                    self.state["waiting_for_start"] = False
            
            if not (self.state['fading_in'] or self.state['fading_out']):
                for button in self.buttons.get(self.state["screen_state"], []):
                    button.handle_event(event)
    
    def _update(self):
        """게임의 논리적 상태를 매 프레임 업데이트합니다."""
        if not (self.state['fading_in'] or self.state['fading_out']):
            for button in self.buttons.get(self.state["screen_state"], []):
                button.update()
        
        if self.state['screen_state'] == 'face_capture':
             self._update_face_capture_logic()

        elif self.state['screen_state'] == 'webcam_view':
            self._update_gameplay_logic()

    def _update_face_capture_logic(self):
        """얼굴 캡쳐 화면의 UART 통신 및 자동 캡쳐 로직을 처리합니다."""
        if not self.state["captured_goalkeeper_face_filename"]:
            if not self.state["is_capturing_face"]:
                send_uart_command(self.resources["ser_goalkeeper"], 'face')
                self.state["is_capturing_face"] = True

            ser = self.resources.get("ser_goalkeeper")
            if ser and ser.in_waiting > 0:
                uart_bytes = ser.read(ser.in_waiting)
                for byte in uart_bytes: self.state["goalkeeper_face_data_buffer"].append(byte & 31)

            if len(self.state["goalkeeper_face_data_buffer"]) >= 4:
                chunks = self.state["goalkeeper_face_data_buffer"]
                full_data = (chunks[0] << 15) | (chunks[1] << 10) | (chunks[2] << 5) | chunks[3]
                y_coord_raw, x_coord_raw = (full_data >> 10) & 0x3FF, full_data & 0x3FF
                self.state["last_goalkeeper_face_coords"] = {
                    "raw": (x_coord_raw, y_coord_raw),
                    "scaled": (self.goalkeeper_start_x + int(x_coord_raw * (self.goalkeeper_monitor_width / 640)), int(y_coord_raw * (self.screen_height / 480)))
                }

                coords = self.state["last_goalkeeper_face_coords"]
                capture_area = pygame.Rect(self.goalkeeper_monitor_center_x - 100, self.screen_height // 2 - 350, 200, 200)
                if capture_area.collidepoint(coords["scaled"]):
                    filename = self._capture_and_save_face(self.resources["last_cam_frame"], coords["raw"], "captured_goalkeeper_face.png")
                    if filename:
                        self.state["captured_goalkeeper_face_filename"] = filename
                        send_uart_command(self.resources["ser_goalkeeper"], 'stop')
                        self.state["is_capturing_face"] = False
                        if self.state["game_mode"] == "multi":
                            send_uart_command(self.resources["ser_attacker"], 'face')
                        else:
                            self._start_new_round()
                            self._start_transition("webcam_view")
                self.state["goalkeeper_face_data_buffer"] = chunks[4:]

        elif self.state["game_mode"] == "multi" and not self.state["captured_attacker_face_filename"]:
            ser = self.resources.get("ser_attacker")
            if ser and ser.in_waiting > 0:
                uart_bytes = ser.read(ser.in_waiting)
                for byte in uart_bytes: self.state["attacker_face_data_buffer"].append(byte & 31)
            
            if len(self.state["attacker_face_data_buffer"]) >= 4:
                chunks = self.state["attacker_face_data_buffer"]
                full_data = (chunks[0] << 15) | (chunks[1] << 10) | (chunks[2] << 5) | chunks[3]
                y_coord_raw, x_coord_raw = (full_data >> 10) & 0x3FF, full_data & 0x3FF
                self.state["last_attacker_face_coords"] = {
                    "raw": (x_coord_raw, y_coord_raw),
                    "scaled": (self.attacker_start_x + int(x_coord_raw * (self.attacker_monitor_width / 640)), int(y_coord_raw * (self.screen_height / 480)))
                }

                coords = self.state["last_attacker_face_coords"]
                capture_area = pygame.Rect(self.attacker_monitor_center_x - 100, self.screen_height // 2 - 350, 200, 200)
                if capture_area.collidepoint(coords["scaled"]):
                    filename = self._capture_and_save_face(self.resources["last_cam2_frame"], coords["raw"], "captured_attacker_face.png")
                    if filename:
                        self.state["captured_attacker_face_filename"] = filename
                        send_uart_command(self.resources["ser_attacker"], 'stop')
                        self._start_new_round()
                        self._start_transition("webcam_view")
                self.state["attacker_face_data_buffer"] = chunks[4:]

    def _update_gameplay_logic(self):
        """게임 플레이 중의 시간 기반 상태 변화를 처리합니다."""
        if self.state["gif_start_time"] and (pygame.time.get_ticks() - self.state["gif_start_time"] > 3000):
            if self.state["chances_left"] > 0:
                self._start_new_round()
            else:
                self._end_game()

    # ---------------------------------------------------------- #
    # 4. 렌더링(그리기) 메서드들
    # ---------------------------------------------------------- #
    def _draw(self):
        """현재 게임 상태에 맞는 주 화면과 UI 요소들을 그립니다."""
        screen_state = self.state['screen_state']
        
        # getattr을 사용하여 상태 이름에 맞는 _draw_..._screen 메서드를 동적으로 호출
        draw_function = getattr(self, f'_draw_{screen_state}_screen', self._draw_default_screen)
        
        if screen_state in ['menu', 'game']:
            draw_function(screen_state)
        else:
            draw_function()
            
        for button in self.buttons.get(screen_state, []): 
            button.draw(self.screen)
        
        self._draw_transition()
        pygame.display.flip()
    
    def _draw_default_screen(self):
        """일치하는 그리기 함수가 없을 경우 검은 화면을 그립니다."""
        self.screen.fill(self.COLORS['BLACK'])

    def _draw_menu_screen(self, state): self._draw_menu_or_game_screen(state)
    def _draw_game_screen(self, state): self._draw_menu_or_game_screen(state)

    def _draw_menu_or_game_screen(self, state):
        self.screen.fill(self.COLORS['BLACK'])
        pygame.draw.rect(self.screen, self.COLORS['BLACK'], (self.goalkeeper_start_x, 0, self.goalkeeper_monitor_width, self.screen_height))
        pygame.draw.rect(self.screen, self.COLORS['BLACK'], (self.attacker_start_x, 0, self.attacker_monitor_width, self.screen_height))
        
        menu_bg = self.resources["videos"]["menu_bg"]
        if not menu_bg: return
        ret, frame = menu_bg.read()
        if not ret:
            menu_bg.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = menu_bg.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized_main = cv2.resize(frame_rgb, (self.main_monitor_width, self.screen_height))
            self.screen.blit(pygame.surfarray.make_surface(frame_resized_main.swapaxes(0, 1)), (self.main_start_x, 0))

        if state == "game":
            text_surf = self.fonts['font'].render("플레이어 수를 선택하세요", True, self.COLORS['WHITE'])
            self.screen.blit(text_surf, text_surf.get_rect(center=(self.main_monitor_center_x, self.screen_height//2 - 200)))
        elif state == "menu":
            self.fonts['font'].set_bold(True)
            start_text_l1 = self.fonts['font'].render("게임을 시작하려면 아무 키나 누르세요", True, self.COLORS['WHITE'])
            self.fonts['font'].set_bold(False)
            self.fonts['description_font'].set_bold(True)
            start_text_l2 = self.fonts['description_font'].render("PRESS ANY KEY", True, self.COLORS['WHITE'])
            self.fonts['description_font'].set_bold(False)
            y_pos_l1 = self.screen_height * 0.75
            y_pos_l2 = y_pos_l1 + 80
            self.screen.blit(start_text_l1, start_text_l1.get_rect(center=(self.main_monitor_center_x, y_pos_l1)))
            self.screen.blit(start_text_l2, start_text_l2.get_rect(center=(self.main_monitor_center_x, y_pos_l2)))

    def _draw_face_capture_screen(self):
        self.screen.fill(self.COLORS['BLACK'])
    
        def draw_capture_ui(surface, start_x, width, center_x, captured_filename, player_name):
            overlay = pygame.Surface((width, self.screen_height), pygame.SRCALPHA)
            surface.blit(overlay, (start_x, 0))
            if not captured_filename:
                overlay.fill((0, 0, 0, 128))
                title_surf = self.fonts['title_font'].render(f"{player_name} 얼굴 캡처", True, self.COLORS['WHITE'])
                desc_surf = self.fonts['font'].render("얼굴을 중앙의 사각형에 맞춰주세요", True, self.COLORS['WHITE'])
                surface.blit(title_surf, title_surf.get_rect(center=(center_x, self.screen_height/2 - 80)))
                surface.blit(desc_surf, desc_surf.get_rect(center=(center_x, self.screen_height/2 + 40)))
                capture_area_rect = pygame.Rect(center_x - 100, self.screen_height // 2 - 350, 200, 200)
                pygame.draw.rect(surface, self.COLORS['GRID'], capture_area_rect, 3, border_radius=15)
            else:
                overlay.fill((0, 0, 0, 200))
                captured_text = self.fonts['title_font'].render("캡처 완료!", True, self.COLORS['GOLD'])
                surface.blit(captured_text, captured_text.get_rect(center=(center_x, self.screen_height / 2)))

        # 공격수 화면
        if self.state["game_mode"] == "multi":
            cap2 = self.resources['cap_attacker']
            if cap2.isOpened():
                ret_cam2, frame_cam2 = cap2.read()
                if ret_cam2:
                    self.resources["last_cam2_frame"] = frame_cam2
                    frame_cam2_flipped = cv2.flip(frame_cam2, 1)
                    frame_cam2_rgb = cv2.cvtColor(frame_cam2_flipped, cv2.COLOR_BGR2RGB)
                    cam2_surf = pygame.surfarray.make_surface(frame_cam2_rgb.swapaxes(0, 1))
                    cam2_surf_scaled = pygame.transform.scale(cam2_surf, (self.attacker_monitor_width, self.screen_height))
                    self.screen.blit(cam2_surf_scaled, (self.attacker_start_x, 0))

            if not self.state["captured_goalkeeper_face_filename"]:
                overlay = pygame.Surface((self.attacker_monitor_width, self.screen_height), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 200))
                wait_text = self.fonts['title_font'].render("대기 중...", True, self.COLORS['WHITE'])
                overlay.blit(wait_text, wait_text.get_rect(center=(self.attacker_monitor_width/2, self.screen_height/2)))
                self.screen.blit(overlay, (self.attacker_start_x, 0))
            else:
                draw_capture_ui(self.screen, self.attacker_start_x, self.attacker_monitor_width, self.attacker_monitor_center_x, self.state["captured_attacker_face_filename"], "공격수")
        else:
            pygame.draw.rect(self.screen, self.COLORS['BLACK'], (self.attacker_start_x, 0, self.attacker_monitor_width, self.screen_height))

        # 골키퍼 화면
        ret_cam, frame_cam = self.resources['cap_goalkeeper'].read()
        if ret_cam:
            self.resources["last_cam_frame"] = frame_cam
            frame_cam_flipped = cv2.flip(frame_cam, 1)
            frame_cam_rgb = cv2.cvtColor(frame_cam_flipped, cv2.COLOR_BGR2RGB)
            cam_surf = pygame.surfarray.make_surface(frame_cam_rgb.swapaxes(0, 1))
            cam_surf_scaled = pygame.transform.scale(cam_surf, (self.goalkeeper_monitor_width, self.screen_height))
            self.screen.blit(cam_surf_scaled, (self.goalkeeper_start_x, 0))
        draw_capture_ui(self.screen, self.goalkeeper_start_x, self.goalkeeper_monitor_width, self.goalkeeper_monitor_center_x, self.state["captured_goalkeeper_face_filename"], "골키퍼")

        if self.state["last_goalkeeper_face_coords"]:
            pygame.draw.circle(self.screen, self.COLORS['RED'], self.state["last_goalkeeper_face_coords"]["scaled"], 20, 4)
        if self.state["last_attacker_face_coords"]:
            pygame.draw.circle(self.screen, self.COLORS['RED'], self.state["last_attacker_face_coords"]["scaled"], 20, 4)

    def _draw_webcam_view_screen(self):
        # GIF 재생 여부를 먼저 확인
        should_play_gif = (self.state["is_failure"] or self.state["is_success"]) and self.state["result_time"] and (pygame.time.get_ticks() - self.state["result_time"] > 1000)
        
        gif_key = None # active_gif 대신 사용할 키
        if should_play_gif:
            if self.state["is_failure"]: gif_key = 'failure'
            elif self.state["is_success"]: gif_key = 'success'
        
        if gif_key and not self.state["gif_start_time"]:
            self.state["gif_start_time"] = pygame.time.get_ticks()
            self.state["gif_frame_index"] = 0 # GIF가 시작될 때 인덱스를 0으로 초기화
            if self.state["is_success"] and self.resources["sounds"].get("success"): self.resources["sounds"]["success"].play()
            elif self.state["is_failure"] and self.resources["sounds"].get("failed"): self.resources["sounds"]["failed"].play()

        # GIF가 재생 중이면, GIF만 그리고 함수를 종료
        if self.state["gif_start_time"] and gif_key:
            self._draw_result_gif_fullscreen(gif_key) # <<-- 이 부분 수정
            return

        # GIF가 재생 중이 아닐 때만, 일반 게임 화면을 그림
        self.screen.fill(self.COLORS['BLACK'])
        
        bg_video = self.resources["videos"].get("bg_video")
        if bg_video and (self.state["waiting_for_start"] or self.state["countdown_start"]):
            if self.state["waiting_for_start"]: bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            elif self.state["countdown_start"]:
                elapsed = pygame.time.get_ticks() - self.state["countdown_start"]
                current_frame_pos = int(elapsed / self.bg_video_interval)
                if current_frame_pos < self.bg_video_total_frames: bg_video.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
            
            ret_vid, frame_vid = bg_video.read()
            if ret_vid:
                new_w, new_h = get_scaled_rect(self.bg_video_w, self.bg_video_h, self.main_monitor_width, self.screen_height)
                pos_x = self.main_start_x + (self.main_monitor_width - new_w) // 2
                pos_y = (self.screen_height - new_h) // 2
                frame_vid_rgb = cv2.cvtColor(frame_vid, cv2.COLOR_BGR2RGB)
                frame_vid_resized = cv2.resize(frame_vid_rgb, (new_w, new_h))
                self.screen.blit(pygame.surfarray.make_surface(frame_vid_resized.swapaxes(0, 1)), (pos_x, pos_y))

        # 골키퍼 화면
        ret_cam, frame_cam = self.resources['cap_goalkeeper'].read()
        if ret_cam:
            frame_cam_flipped = cv2.flip(frame_cam, 1)
            frame_cam_rgb = cv2.cvtColor(frame_cam_flipped, cv2.COLOR_BGR2RGB)
            frame_cam_resized = cv2.resize(frame_cam_rgb, (self.goalkeeper_monitor_width, self.screen_height))
            self.screen.blit(pygame.surfarray.make_surface(frame_cam_resized.swapaxes(0, 1)), (self.goalkeeper_start_x, 0))
        
        cell_w_gk = self.goalkeeper_monitor_width / 5
        for i in range(1, 5): pygame.draw.line(self.screen, self.COLORS['GRID'], (self.goalkeeper_start_x + i * cell_w_gk, 0), (self.goalkeeper_start_x + i * cell_w_gk, self.screen_height), 2)
        self._draw_player_info(self.screen, self.goalkeeper_start_x, self.goalkeeper_monitor_width, "goalkeeper")

        # 공격수 화면
        cell_w_atk = self.attacker_monitor_width / 5
        if self.state["game_mode"] == "multi":
            cap2 = self.resources['cap_attacker']
            if cap2.isOpened():
                ret_cam2, frame_cam2 = cap2.read()
                if ret_cam2:
                    frame_cam2_flipped = cv2.flip(frame_cam2, 1)
                    frame_cam2_rgb = cv2.cvtColor(frame_cam2_flipped, cv2.COLOR_BGR2RGB)
                    cam2_surf_scaled = pygame.transform.scale(pygame.surfarray.make_surface(frame_cam2_rgb.swapaxes(0, 1)), (self.attacker_monitor_width, self.screen_height))
                    self.screen.blit(cam2_surf_scaled, (self.attacker_start_x, 0))
            for i in range(1, 5): pygame.draw.line(self.screen, self.COLORS['GRID'], (self.attacker_start_x + i * cell_w_atk, 0), (self.attacker_start_x + i * cell_w_atk, self.screen_height), 2)
            if self.state["attacker_selected_col"] is not None: pygame.draw.rect(self.screen, self.COLORS['RED'], (self.attacker_start_x + self.state["attacker_selected_col"] * cell_w_atk, 0, cell_w_atk, self.screen_height), 10)
            if self.state["ball_col"] is not None and self.resources["images"].get("ball"):
                ball_rect_atk = self.resources["images"]["ball"].get_rect(center=(self.attacker_start_x + self.state["ball_col"] * cell_w_atk + cell_w_atk / 2, self.screen_height / 2))
                self.screen.blit(self.resources["images"]["ball"], ball_rect_atk)
            self._draw_player_info(self.screen, self.attacker_start_x, self.attacker_monitor_width, "attacker")
        else:
            pygame.draw.rect(self.screen, self.COLORS['BLACK'], (self.attacker_start_x, 0, self.attacker_monitor_width, self.screen_height))

        # 중앙 및 공통 UI
        if self.state["waiting_for_start"]:
            overlay = pygame.Surface((self.main_monitor_width, self.screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            self.screen.blit(overlay, (self.main_start_x, 0))
            start_text_l1 = self.fonts['title_font'].render("시작하시겠습니까?", True, self.COLORS['WHITE'])
            start_text_l2 = self.fonts['font'].render("(Press Space Bar)", True, self.COLORS['WHITE'])
            self.screen.blit(start_text_l1, start_text_l1.get_rect(center=(self.main_monitor_center_x, self.screen_height/2 - 60)))
            self.screen.blit(start_text_l2, start_text_l2.get_rect(center=(self.main_monitor_center_x, self.screen_height/2 + 40)))
        elif self.state["countdown_start"]:
            elapsed = pygame.time.get_ticks() - self.state["countdown_start"]
            if elapsed < 5000:
                self._handle_countdown_input()
                if self.state["selected_col"] is not None:
                    pygame.draw.rect(self.screen, self.COLORS['GOLD'], (self.goalkeeper_start_x + self.state["selected_col"] * cell_w_gk, 0, cell_w_gk, self.screen_height), 10)
                
                num_str = str(5 - (elapsed // 1000))
                text_surf = self.fonts['countdown_font'].render(num_str, True, self.COLORS['WHITE'])
                self.screen.blit(text_surf, text_surf.get_rect(center=(self.goalkeeper_monitor_center_x, self.screen_height/2)))
                if self.state["game_mode"] == "multi":
                    self.screen.blit(text_surf, text_surf.get_rect(center=(self.attacker_monitor_center_x, self.screen_height/2)))
            else:
                if self.state["final_col"] is None:
                    self._process_round_result()
        
        if self.state["final_col"] is not None:
            highlight_surf = pygame.Surface((cell_w_gk, self.screen_height), pygame.SRCALPHA); highlight_surf.fill(self.COLORS['HIGHLIGHT'])
            self.screen.blit(highlight_surf, (self.goalkeeper_start_x + self.state["final_col"] * cell_w_gk, 0))
        if self.state["ball_col"] is not None and self.resources["images"].get("ball"):
            ball_rect_gk = self.resources["images"]["ball"].get_rect(center=(self.goalkeeper_start_x + self.state["ball_col"] * cell_w_gk + cell_w_gk / 2, self.screen_height / 2))
            self.screen.blit(self.resources["images"]["ball"], ball_rect_gk)

    def _draw_result_gif_fullscreen(self, gif_key):
        """미리 로드된 GIF 프레임을 화면에 그립니다."""
        self.screen.fill(self.COLORS['BLACK'])
        
        frame_list = self.resources['gif_frames'].get(gif_key)
        if not frame_list:
            return
        
        current_index = self.state['gif_frame_index']
        frame_surface = frame_list[current_index]

        self.screen.blit(frame_surface, (self.goalkeeper_start_x, 0))
        if self.state["game_mode"] == "multi":
            self.screen.blit(frame_surface, (self.attacker_start_x, 0))
            
        # 프레임 속도를 조절하고 싶으면 `... % 2 == 0` 와 같은 조건을 추가
        if pygame.time.get_ticks() % 2 == 0:
            next_index = (current_index + 1) % len(frame_list)
            self.state['gif_frame_index'] = next_index

    def _handle_countdown_input(self):
        send_uart_command(self.resources.get("ser_goalkeeper"), 'grid')
        ser_gk = self.resources.get("ser_goalkeeper")
        if ser_gk and ser_gk.in_waiting > 0:
            try:
                uart_bytes = ser_gk.read(ser_gk.in_waiting)
                if uart_bytes:
                    valid_values = [b for b in uart_bytes if b in [1, 2, 3, 4, 5]]
                    if valid_values: self.state["selected_col"] = 5 - valid_values[-1]
            except Exception as e: print(f"UART(Grid) 수신 오류: {e}")

        if self.state["game_mode"] == "multi":
            send_uart_command(self.resources.get("ser_attacker"), 'kick')
            ser_atk = self.resources.get("ser_attacker")
            if ser_atk and ser_atk.in_waiting > 0:
                try:
                    uart_bytes_attacker = ser_atk.read(ser_atk.in_waiting)
                    if uart_bytes_attacker:
                        valid_values_attacker = [b for b in uart_bytes_attacker if b in [1, 2, 3, 4, 5]]
                        if valid_values_attacker: self.state["attacker_selected_col"] = 5 - valid_values_attacker[-1]
                except Exception as e: print(f"UART(Attacker Kick) 수신 오류: {e}")

    def _process_round_result(self):
        send_uart_command(self.resources.get("ser_goalkeeper"), 'stop')
        if self.state["game_mode"] == "multi": send_uart_command(self.resources.get("ser_attacker"), 'stop')
        self.state["final_col"] = self.state["selected_col"]
        self.state["chances_left"] -= 1
        
        if self.state["game_mode"] == 'single':
            self.state["ball_col"] = random.randint(0, 4)
        else: 
            self.state["ball_col"] = self.state["attacker_selected_col"] if self.state["attacker_selected_col"] is not None else random.randint(0, 4)
        
        if self.state["final_col"] is not None and self.state["ball_col"] is not None:
             self.state["is_success"] = (self.state["final_col"] == self.state["ball_col"])
        else: # A player didn't make a choice
            self.state["is_success"] = False # Or handle as a draw/miss
        
        self.state["is_failure"] = not self.state["is_success"]

        if self.state["is_success"]: 
            self.state["score"] += 1
        elif self.state["is_failure"] and self.state["game_mode"] == "multi":
            self.state["attacker_score"] += 1
            
        self.state["result_time"] = pygame.time.get_ticks()
        self.state["countdown_start"] = None

    def _draw_player_info(self, surface, start_x, width, player_type):
        overlay = pygame.Surface((width, self.screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        surface.blit(overlay, (start_x, 0))
        
        display_score = self.state['score'] if player_type == 'goalkeeper' else self.state['attacker_score']

        score_text = self.fonts['score_font'].render(f"SCORE: {display_score}", True, self.COLORS['WHITE'])
        score_rect = score_text.get_rect(topright=(start_x + width - 20, 20))
        surface.blit(score_text, score_rect)

        chances_text = self.fonts['font'].render("CHANCES", True, self.COLORS['WHITE'])
        chances_rect = chances_text.get_rect(topright=(start_x + width - 20, score_rect.bottom + 10))
        surface.blit(chances_text, chances_rect)
        
        scoreboard_ball = self.resources["images"].get("scoreboard_ball")
        if scoreboard_ball:
            ball_width = scoreboard_ball.get_width()
            total_balls_width = self.state["chances_left"] * (ball_width + 10) - 10
            start_ball_x = (start_x + width - 20) - total_balls_width

            for i in range(self.state["chances_left"]):
                surface.blit(scoreboard_ball, (start_ball_x + i * (ball_width + 10), chances_rect.bottom + 10))

    def _draw_info_screen(self):
        self.screen.fill(self.COLORS['BLACK'])
        pygame.draw.rect(self.screen, self.COLORS['BLACK'], (self.goalkeeper_start_x, 0, self.goalkeeper_monitor_width, self.screen_height))
        pygame.draw.rect(self.screen, self.COLORS['BLACK'], (self.attacker_start_x, 0, self.attacker_monitor_width, self.screen_height))

        info_bg = self.resources["images"].get("info_bg")
        if info_bg:
            scaled_info_bg = pygame.transform.scale(info_bg, (self.main_monitor_width, self.screen_height))
            self.screen.blit(scaled_info_bg, (self.main_start_x, 0))

        title_surf = self.fonts['title_font'].render("게임 방법", True, self.COLORS['WHITE'])
        self.screen.blit(title_surf, title_surf.get_rect(center=(self.main_monitor_center_x, 200)))
        
        text_1p = ["[1인 플레이]", "", "1. 스페이스 바를 누르면 5초 카운트 다운이 시작됩니다.", "", "2. 5개의 영역 중 한 곳을 선택합니다.", "", "3. 5번의 기회동안 최대한 많은 공을 막으세요!"]
        text_2p = ["[2인 플레이]", "", "1. 스페이스 바를 누르면 5초 카운트 다운이 시작됩니다.", "", "2. 공격수와 골키퍼로 나뉩니다.", "", "3. 공격수는 공을 찰 방향을 정합니다.", "", "4. 골키퍼는 공을 막을 방향을 정합니다.", "", "5. 5번의 기회동안 더 많은 득점을 한 쪽이 승리합니다!"]
        font = self.fonts['description_font']
        for i, line in enumerate(text_1p): self.screen.blit(font.render(line, True, self.COLORS['WHITE']), (self.main_monitor_width/4 - 550, 475 + i*75))
        for i, line in enumerate(text_2p): self.screen.blit(font.render(line, True, self.COLORS['WHITE']), (self.main_monitor_width*3/4 - 500, 475 + i*75))

    def _draw_end_screen(self):
        self.screen.fill(self.COLORS['BLACK'])
        
        end_video = self.state.get("end_video")
        if end_video:
            read_new_frame = not (end_video == self.resources["videos"]["defeat"] and pygame.time.get_ticks() % 2 == 0)
            if read_new_frame:
                ret, frame = end_video.read()
                if not ret:
                    end_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = end_video.read()
                if ret: self.state["last_end_frame"] = frame
            if self.state["last_end_frame"] is not None:
                frame_rgb = cv2.cvtColor(self.state["last_end_frame"], cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (self.main_monitor_width, self.screen_height))
                self.screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (self.main_start_x, 0))

        winner = self.state.get("winner")
        face_filename = self.state.get("captured_goalkeeper_face_filename")
        if self.state["game_mode"] == "multi" and winner == "attacker":
            face_filename = self.state.get("captured_attacker_face_filename")

        face_img_scaled = None
        if face_filename and os.path.exists(face_filename):
            try:
                face_img = pygame.image.load(face_filename)
                face_img_scaled = pygame.transform.scale(face_img, (face_img.get_width(), face_img.get_height()))
                face_rect = face_img_scaled.get_rect(center=(self.main_monitor_center_x, self.screen_height / 2))
                self.screen.blit(face_img_scaled, face_rect)
            except Exception as e: print(f"이미지 파일 불러오기 오류: {e}")

        rank_y_pos = self.screen_height/2 - 150
        score_y_pos = self.screen_height/2
        if face_img_scaled:
            face_rect = face_img_scaled.get_rect(center=(self.main_monitor_center_x, self.screen_height / 2))
            rank_y_pos, score_y_pos = face_rect.top - 100, face_rect.bottom + 80

        rank_surf = self.fonts['rank_font'].render(self.state["final_rank"], True, self.COLORS['GOLD'])
        self.screen.blit(rank_surf, rank_surf.get_rect(center=(self.main_monitor_center_x, rank_y_pos)))
        
        score_font = self.fonts['score_font']
        if self.state["game_mode"] == "multi":
            score_str = f"{self.state['score']} : {self.state['attacker_score']}"
            goalkeeper_text = score_font.render("Goalkeeper", True, self.COLORS['WHITE'])
            attacker_text = score_font.render("Attacker", True, self.COLORS['WHITE'])
            score_surf = score_font.render(score_str, True, self.COLORS['WHITE'])
            total_width = goalkeeper_text.get_width() + score_surf.get_width() + attacker_text.get_width() + 100
            start_x = self.main_monitor_center_x - total_width / 2
            self.screen.blit(goalkeeper_text, (start_x, score_y_pos))
            self.screen.blit(score_surf, (start_x + goalkeeper_text.get_width() + 50, score_y_pos))
            self.screen.blit(attacker_text, (start_x + goalkeeper_text.get_width() + score_surf.get_width() + 100, score_y_pos))
        else:
            score_surf = score_font.render(f"FINAL SCORE: {self.state['score']}", True, self.COLORS['WHITE'])
            self.screen.blit(score_surf, score_surf.get_rect(center=(self.main_monitor_center_x, score_y_pos)))
            highscore_surf = score_font.render(f"HIGH SCORE: {self.state['highscore']}", True, self.COLORS['GOLD'])
            self.screen.blit(highscore_surf, highscore_surf.get_rect(center=(self.main_monitor_center_x, score_y_pos + 80)))

    def _draw_transition(self):
        """화면 전환 시 페이드 효과를 화면에 그립니다."""
        if self.state['fading_out'] or self.state['fading_in']:
            if self.state['fading_out']:
                self.state['transition_alpha'] = min(255, self.state['transition_alpha'] + self.state['transition_speed'])
                if self.state['transition_alpha'] == 255:
                    self.state['fading_out'], self.state['fading_in'] = False, True
                    self.state["screen_state"] = self.state['transition_target']
            else: # fading_in
                self.state['transition_alpha'] = max(0, self.state['transition_alpha'] - self.state['transition_speed'])
                if self.state['transition_alpha'] == 0: 
                    self.state['fading_in'] = False
            self.transition_surface.set_alpha(self.state['transition_alpha'])
            self.screen.blit(self.transition_surface, (0, 0))

    # ---------------------------------------------------------- #
    # 5. 헬퍼 메서드
    # ---------------------------------------------------------- #
    def _capture_and_save_face(self, original_frame, raw_coords, output_filename):
        """웹캠 프레임에서 얼굴 부분을 원형으로 잘라 투명 배경 이미지로 저장합니다."""
        if original_frame is None: return None
        try:
            h, w, _ = original_frame.shape
            cx, cy = raw_coords[0], raw_coords[1]
            radius = 150
            
            bgra_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2BGRA)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            bgra_frame[:, :, 3] = mask
            
            x1, y1 = max(0, cx - radius), max(0, cy - radius)
            x2, y2 = min(w, cx + radius), min(h, cy + radius)
            cropped_bgra = bgra_frame[y1:y2, x1:x2]
            
            final_rgba = cv2.cvtColor(cropped_bgra, cv2.COLOR_BGRA2RGBA)
            face_surface = pygame.image.frombuffer(final_rgba.tobytes(), final_rgba.shape[1::-1], "RGBA")
            pygame.image.save(face_surface, output_filename)
            print(f"얼굴 캡처 성공! ({output_filename}으로 저장됨)")
            return output_filename
        except Exception as e:
            print(f"이미지 저장/변환 오류: {e}")
            return None

    # ---------------------------------------------------------- #
    # 6. 메인 루프 및 정리
    # ---------------------------------------------------------- #
    def run(self):
        """메인 게임 루프를 실행합니다."""
        while self.is_running:
            self._handle_events()
            self._update()
            self._draw()
            self.clock.tick(60)
        self._cleanup()

    def _cleanup(self):
        """게임 종료 시 모든 리소스를 해제합니다."""
        print("게임을 종료하고 리소스를 정리합니다.")
        for cap_key in ['cap_goalkeeper', 'cap_attacker']:
            cap = self.resources.get(cap_key)
            if cap and cap.isOpened(): cap.release()
        for ser_key in ['ser_goalkeeper', 'ser_attacker']:
            ser = self.resources.get(ser_key)
            if ser and ser.is_open: ser.close()
        for video in self.resources["videos"].values():
            if video: video.release()
        cv2.destroyAllWindows()
        pygame.quit()
        sys.exit()

# ========================================================== #
# 게임 실행
# ========================================================== #
if __name__ == '__main__':
    try:
        game = PenaltyKickGame()
        game.run()
    except Exception as e:
        print(f"게임 실행 중 치명적인 오류 발생: {e}")
        pygame.quit()
        sys.exit()