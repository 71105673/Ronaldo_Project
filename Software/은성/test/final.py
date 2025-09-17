# ===================================================================
# 1. 라이브러리 임포트
# ===================================================================
import pygame
import sys
import cv2
import numpy as np
import random
import os
import serial
from PIL import Image
import imageio
from Button import ImageButton, MenuButton
from config import *

# ===================================================================
# 2. 데이터 관리 클래스
# ===================================================================
class GameState:
    """게임의 모든 상태를 관리하는 클래스"""
    def __init__(self):
        self.screen_state = "menu"
        self.highscore = load_highscore()
        self.game_mode = None
        self.chances_left = CHANCES_PER_GAME
        self.score = 0
        self.attacker_score = 0
        
        self.final_rank = ""
        self.end_video = None
        
        self.countdown_start = None
        self.selected_col = None
        self.final_col = None
        self.ball_col = None
        self.is_failure = False
        self.is_success = False
        self.result_time = None
        
        self.gif_start_time = None
        self.gif_frame_index = 0
        self.gif_last_frame_time = 0
        
        self.waiting_for_start = False
        self.is_capturing_face = False
        
        self.synthesized_frames = []
        self.synthesized_frame_index = 0
        self.synthesized_last_update = 0
        self.synthesis_info = None

    def start_countdown(self):
        self.countdown_start = pygame.time.get_ticks()
        self.waiting_for_start = False

    def reset_round(self):
        """한 라운드가 끝났을 때 상태를 초기화"""
        self.countdown_start = None
        self.selected_col = None
        self.final_col = None
        self.ball_col = None
        self.is_failure = False
        self.is_success = False
        self.result_time = None
        self.gif_start_time = None
        self.gif_frame_index = 0
        self.waiting_for_start = False
        self.is_capturing_face = False
        self.synthesized_frames = []
        self.synthesis_info = None

    def reset_full(self):
        """게임 전체를 처음부터 다시 시작할 때 상태를 초기화"""
        self.reset_round()
        self.chances_left = CHANCES_PER_GAME
        self.score = 0
        self.attacker_score = 0

class Player:
    """플레이어 데이터와 하드웨어를 관리하는 클래스"""
    def __init__(self, name, camera_id, serial_port_name, mon_start_x, mon_width):
        self.name = name
        self.score = 0
        self.selected_col = None
        self.face_filename = None
        self.last_face_coords = None
        self.uart_buffer = []
        self.last_cam_frame = None

        self.start_x = mon_start_x
        self.width = mon_width
        self.center_x = mon_start_x + mon_width // 2

        self.cap = cv2.VideoCapture(camera_id)
        if not self.cap.isOpened():
            print(f"경고: {self.name}용 카메라(ID: {camera_id})를 열 수 없습니다.")
        
        try:
            self.serial_port = serial.Serial(serial_port_name, 9600, timeout=0)
            print(f"{self.name} 보드({serial_port_name})가 성공적으로 연결되었습니다.")
        except serial.SerialException as e:
            self.serial_port = None
            print(f"오류: {self.name} 보드({serial_port_name})를 열 수 없습니다 - {e}")

    def release(self):
        """플레이어 관련 리소스를 해제"""
        if self.cap: self.cap.release()
        if self.serial_port and self.serial_port.is_open: self.serial_port.close()

# ===================================================================
# 3. 유틸리티 함수
# ===================================================================
def send_uart_command(serial_port, command):
    commands = {'grid': 225, 'face': 226, 'kick': 227, 'stop': 0}
    byte_to_send = commands.get(command)
    if byte_to_send is not None and serial_port and serial_port.is_open:
        try:
            serial_port.write(bytes([byte_to_send]))
        except Exception as e:
            print(f"UART({command}) 데이터 송신 오류: {e}")

def get_camera_surface(player, flip=True):
    """카메라에서 프레임을 읽어 Pygame Surface 객체로 변환하여 반환"""
    if not player.cap or not player.cap.isOpened():
        return None
    
    ret, frame = player.cap.read()
    if not ret:
        return None
    
    player.last_cam_frame = frame
    
    if flip:
        frame = cv2.flip(frame, 1)
    
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
    scaled_surface = pygame.transform.scale(surface, (player.width, screen_height))
    
    return scaled_surface

def parse_uart_data(serial_port, buffer, num_bytes=4):
    """시리얼 포트에서 데이터를 읽고 파싱 (얼굴 좌표용)"""
    if not serial_port or not serial_port.in_waiting > 0:
        return None, buffer

    uart_bytes = serial_port.read(serial_port.in_waiting)
    for byte in uart_bytes:
        buffer.append(byte & 31)

    if len(buffer) >= num_bytes:
        full_data = (buffer[0] << 15) | (buffer[1] << 10) | (buffer[2] << 5) | buffer[3]
        y_coord_raw = (full_data >> 10) & 0x3FF
        x_coord_raw = full_data & 0x3FF
        return (x_coord_raw, y_coord_raw), buffer[num_bytes:]
    
    return None, buffer

def capture_and_save_face(original_frame, raw_coords, output_filename):
    """카메라 프레임에서 얼굴 부분을 캡처하여 원형으로 자른 뒤 파일로 저장"""
    if original_frame is None: return None
    try:
        h, w, _ = original_frame.shape
        cx, cy, radius = raw_coords[0], raw_coords[1], 150
        
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

def preprocess_gif(gif_path):
    print("GIF 파일을 미리 분석 중입니다...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    try:
        gif_reader = imageio.get_reader(gif_path)
    except FileNotFoundError:
        print(f"오류: '{gif_path}' 파일을 찾을 수 없습니다.")
        return None, None
    frames, face_locations = [], []
    for frame_data in gif_reader:
        frame_pil = Image.fromarray(frame_data).convert("RGBA")
        frames.append(pygame.image.fromstring(frame_pil.tobytes(), frame_pil.size, "RGBA"))
        frame_cv = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        if len(faces) > 0:
            main_face = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face_locations.append(main_face)
        else:
            face_locations.append(None)
    print("GIF 분석 완료!")
    return frames, face_locations

def create_synthesized_gif_frames(face_image_path, gif_path, target_size):
    if not face_image_path or not os.path.exists(face_image_path):
        print(f"합성할 얼굴 이미지 파일을 찾을 수 없습니다: {face_image_path}")
        return []
    gif_frames, gif_face_locations = preprocess_gif(gif_path)
    if not gif_frames: return []
    try:
        overlay_face_pil = Image.open(face_image_path).convert("RGBA")
    except Exception as e:
        print(f"얼굴 이미지 로드 오류: {e}")
        return []
    synthesized_frames = []
    for i, base_frame_surface in enumerate(gif_frames):
        new_frame = base_frame_surface.copy()
        face_loc = gif_face_locations[i]
        if face_loc is not None:
            gx, gy, gw, gh = face_loc
            resized_face_pil = overlay_face_pil.resize((gw, gh), Image.Resampling.LANCZOS)
            face_surface = pygame.image.fromstring(resized_face_pil.tobytes(), resized_face_pil.size, "RGBA")
            new_frame.blit(face_surface, (gx, gy))
        scaled_frame = pygame.transform.scale(new_frame, target_size)
        synthesized_frames.append(scaled_frame)
    print("얼굴 합성 GIF 프레임 생성 완료!")
    return synthesized_frames
    
# ===================================================================
# 4. 메인 게임 클래스
# ===================================================================
class Game:
    def __init__(self):
        self.clock = pygame.time.Clock()
        self.is_running = True
        self.game_state = GameState()
        
        self.goalkeeper = Player("골키퍼", 0, 'COM17', goalkeeper_start_x, goalkeeper_monitor_width)
        self.attacker = Player("공격수", 2, 'COM13', attacker_start_x, attacker_monitor_width)
        
        self.resources = self.load_resources()
        self.buttons = self.create_buttons()
        
        self.transition_surface = pygame.Surface((screen_width, screen_height))
        self.transition_surface.fill(BLACK)
        self.transition_alpha = 0
        self.transition_target = None
        self.transition_speed = 15
        self.fading_out = False
        self.fading_in = False

        self.current_game_bg_surface = None
        self.game_bg_last_update_time = 0

    def load_resources(self):
        res = {"sounds": {}, "images": {}, "videos": {}, "gif_frames": {}}
        try:
            res["sounds"]["button"] = pygame.mixer.Sound("../sound/button_click.wav")
            res["sounds"]["siu"] = pygame.mixer.Sound("../sound/SIUUUUU.wav")
            res["sounds"]["success"] = pygame.mixer.Sound("../sound/야유.mp3")
            res["sounds"]["bg_thumbnail"] = pygame.mixer.Sound("../sound/Time_Bomb.mp3")
            res["sounds"]["failed"] = res["sounds"]["siu"]
            
            ball_img = pygame.image.load("../image/final_ronaldo/Ball.png").convert_alpha()
            res["images"]["scoreboard_ball"] = pygame.transform.scale(ball_img, (80, 80))
            res["images"]["ball"] = pygame.transform.scale(ball_img, (200, 200))
            res["images"]["info_bg"] = pygame.transform.scale(pygame.image.load("../image/info/info_back2.jpg").convert(), (screen_width, screen_height))

            res["gif_frames"]['success'] = load_gif_frames("../image/final_ronaldo/pk.gif", (main_monitor_width, screen_height))
            res["gif_frames"]['failure'] = load_gif_frames("../image/G.O.A.T/siuuu.gif", (main_monitor_width, screen_height))
        
            res["videos"]["lose"] = cv2.VideoCapture("../image/lose_keeper.gif")
            res["videos"]["victory"] = cv2.VideoCapture("../image/victory.gif")
            res["videos"]["defeat"] = cv2.VideoCapture("../image/defeat.gif")
            res["videos"]["game_bg"] = cv2.VideoCapture("../image/Ground1.mp4")
            res["videos"]["menu_bg"] = cv2.VideoCapture("../image/game_thumbnail.mp4")

            if res["sounds"].get("bg_thumbnail"):
                res["sounds"]["bg_thumbnail"].play(-1)
        except Exception as e:
            print(f"리소스 로딩 중 오류 발생: {e}")
        return res

    def create_buttons(self):
        btn_sound = self.resources["sounds"].get("button")
        return {
            "game": [
                MenuButton("1인 플레이", main_start_x + 50, 400, 350, 100, font, lambda: self.start_game("single"), sound=btn_sound),
                MenuButton("2인 플레이", main_start_x + 50, 500, 350, 100, font, lambda: self.start_game("multi"), sound=btn_sound),
                MenuButton("게임 설명", main_start_x + 50, 600, 350, 100, font, lambda: self.start_transition("info"), sound=btn_sound),
                ImageButton("../image/btn_back.png", 150, 150, 100, 100, self.go_to_menu, sound=btn_sound)
            ],
            "face_capture": [ImageButton("../image/btn_back.png", 150, 150, 100, 100, self.go_to_game_select, sound=btn_sound)],
            "webcam_view": [ImageButton("../image/btn_back.png", 150, 150, 100, 100, self.go_to_game_select, sound=btn_sound)],
            "info": [ImageButton("../image/btn_exit.png", main_monitor_center_x*2 - 150, 150, 100, 100, self.go_to_game_select, sound=btn_sound)],
            "end": [
                ImageButton("../image/btn_restart.png", main_monitor_center_x - 300, screen_height - 250, 400, 250, self.go_to_game_select, sound=btn_sound),
                ImageButton("../image/btn_main_menu.png", main_monitor_center_x + 300, screen_height - 250, 400, 250, self.go_to_menu, sound=btn_sound)
            ]
        }

    def start_transition(self, target_state):
        self.transition_target = target_state
        self.fading_out = True
        self.fading_in = False
    
    def start_new_round(self):
        self.game_state.reset_round()
        self.game_state.waiting_for_start = True
    
    def start_game(self, mode):
        if self.resources["sounds"].get("button"): self.resources["sounds"]["button"].play()
        self.game_state.game_mode = mode
        self.game_state.reset_full()
        self.goalkeeper.face_filename = None
        self.attacker.face_filename = None
        self.start_transition("face_capture")
        
    def go_to_menu(self):
        self.game_state.reset_full()
        self.start_transition("menu")
        
    def go_to_game_select(self):
        self.game_state.reset_full()
        self.start_transition("game")

    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                self.is_running = False
                return

            current_state = self.game_state.screen_state
            if current_state == "menu" and event.type == pygame.KEYDOWN:
                self.start_transition("game")
            elif current_state == "webcam_view" and self.game_state.waiting_for_start:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    self.game_state.start_countdown()
            
            if not (self.fading_in or self.fading_out):
                for button in self.buttons.get(current_state, []):
                    button.handle_event(event)
    
    def run(self):
        """메인 게임 루프"""
        while self.is_running:
            self.handle_events()

            if not (self.fading_in or self.fading_out):
                for button in self.buttons.get(self.game_state.screen_state, []):
                    button.update()
            
            self.update_state()
            self.draw_screen()
            
            for button in self.buttons.get(self.game_state.screen_state, []):
                button.draw(screen)

            self.draw_transition()
            
            pygame.display.flip()
            self.clock.tick(60)
        
        self.cleanup()

    def update_state(self):
        state = self.game_state.screen_state
        if state == "webcam_view":
            self.update_webcam_view()
        elif state == "synthesizing":
            self.update_synthesizing()

    def draw_screen(self):
        state = self.game_state.screen_state
        if state in ["menu", "game"]:
            self.draw_menu_or_game_screen(state)
        elif state == "face_capture":
            self.draw_face_capture_screen()
        elif state == "webcam_view":
            self.draw_webcam_view()
        elif state == "info":
            self.draw_info_screen()
        elif state == "synthesizing":
            self.draw_synthesizing_screen()
        elif state == "end":
            self.draw_end_screen()

    def cleanup(self):
        """게임 종료 시 모든 리소스 해제"""
        self.goalkeeper.release()
        self.attacker.release()
        for video in self.resources["videos"].values():
            if video: video.release()
        pygame.quit()
        sys.exit()

    # ===================================================================
    # 화면별 업데이트 및 그리기 함수들
    # ===================================================================

    def draw_transition(self):
        if self.fading_out or self.fading_in:
            if self.fading_out:
                self.transition_alpha = min(255, self.transition_alpha + self.transition_speed)
                if self.transition_alpha == 255:
                    self.fading_out, self.fading_in = False, True
                    self.game_state.screen_state = self.transition_target
            else: # fading_in
                self.transition_alpha = max(0, self.transition_alpha - self.transition_speed)
                if self.transition_alpha == 0:
                    self.fading_in = False
            
            self.transition_surface.set_alpha(self.transition_alpha)
            screen.blit(self.transition_surface, (0, 0))

    def draw_menu_or_game_screen(self, state):
        screen.fill(BLACK)
        video = self.resources["videos"].get("menu_bg" if state == "menu" else "game_bg")
        if video:
            ret, frame = video.read()
            if not ret:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = video.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (main_monitor_width, screen_height))
                screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (main_start_x, 0))

        if state == "menu":
            start_text_l1 = font.render("게임을 시작하려면 아무 키나 누르세요", True, WHITE)
            start_text_l2 = description_font.render("PRESS ANY KEY", True, WHITE)
            screen.blit(start_text_l1, start_text_l1.get_rect(center=(main_monitor_center_x, screen_height * 0.75)))
            screen.blit(start_text_l2, start_text_l2.get_rect(center=(main_monitor_center_x, screen_height * 0.75 + 80)))

    def draw_face_capture_screen(self):
        screen.fill(BLACK)
        
        # 골키퍼 화면
        gk_surface = get_camera_surface(self.goalkeeper)
        if gk_surface:
            screen.blit(gk_surface, (self.goalkeeper.start_x, 0))
        self.draw_capture_ui(self.goalkeeper)

        # 공격수 화면
        if self.game_state.game_mode == "multi":
            atk_surface = get_camera_surface(self.attacker)
            if atk_surface:
                screen.blit(atk_surface, (self.attacker.start_x, 0))

            if not self.goalkeeper.face_filename: # 골키퍼가 먼저 캡처
                overlay = pygame.Surface((self.attacker.width, screen_height), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 200))
                wait_text = title_font.render("대기 중...", True, WHITE)
                overlay.blit(wait_text, wait_text.get_rect(center=(self.attacker.width / 2, screen_height / 2)))
                screen.blit(overlay, (self.attacker.start_x, 0))
            else:
                self.draw_capture_ui(self.attacker)
        else:
            pygame.draw.rect(screen, BLACK, (self.attacker.start_x, 0, self.attacker.width, screen_height))

        # 얼굴 캡처 로직
        self.handle_face_capture()

    def draw_capture_ui(self, player):
        if not player.face_filename:
            overlay = pygame.Surface((player.width, screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            screen.blit(overlay, (player.start_x, 0))
            
            title_surf = title_font.render(f"{player.name} 얼굴 캡처", True, WHITE)
            desc_surf = font.render("얼굴을 중앙의 사각형에 맞춰주세요", True, WHITE)
            screen.blit(title_surf, title_surf.get_rect(center=(player.center_x, screen_height / 2 - 80)))
            screen.blit(desc_surf, desc_surf.get_rect(center=(player.center_x, screen_height / 2 + 40)))
            
            capture_area_rect = pygame.Rect(player.center_x - 100, screen_height // 2 - 350, 200, 200)
            pygame.draw.rect(screen, GRID_COLOR, capture_area_rect, 3, border_radius=15)
        else:
            overlay = pygame.Surface((player.width, screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 200))
            screen.blit(overlay, (player.start_x, 0))
            captured_text = title_font.render("캡처 완료!", True, GOLD_COLOR)
            screen.blit(captured_text, captured_text.get_rect(center=(player.center_x, screen_height / 2)))

    def handle_face_capture(self):
        if not self.game_state.is_capturing_face:
            send_uart_command(self.goalkeeper.serial_port, 'face')
            self.game_state.is_capturing_face = True

        # 현재 캡처 대상 플레이어 결정
        if not self.goalkeeper.face_filename:
            player_to_capture = self.goalkeeper
        elif self.game_state.game_mode == "multi" and not self.attacker.face_filename:
            player_to_capture = self.attacker
        else:
            return # 모두 캡처 완료

        # 데이터 파싱 및 캡처
        raw_coords, player_to_capture.uart_buffer = parse_uart_data(player_to_capture.serial_port, player_to_capture.uart_buffer)
        if raw_coords:
            scaled_coords = (player_to_capture.start_x + int(raw_coords[0] * (player_to_capture.width / 640)), int(raw_coords[1] * (screen_height / 480)))
            pygame.draw.circle(screen, RED, scaled_coords, 20, 4)
            
            capture_area = pygame.Rect(player_to_capture.center_x - 100, screen_height // 2 - 350, 200, 200)
            if capture_area.collidepoint(scaled_coords):
                filename = capture_and_save_face(player_to_capture.last_cam_frame, raw_coords, f"captured_{player_to_capture.name}_face.png")
                if filename:
                    player_to_capture.face_filename = filename
                    send_uart_command(player_to_capture.serial_port, 'stop')
                    
                    # 다음 단계로 전환
                    if player_to_capture == self.goalkeeper and self.game_state.game_mode == "multi":
                        send_uart_command(self.attacker.serial_port, 'face') # 공격수 캡처 시작
                    else: # 1인이거나 공격수까지 완료
                        self.game_state.is_capturing_face = False
                        self.start_new_round()
                        self.start_transition("webcam_view")
    
    def update_webcam_view(self):
        gs = self.game_state
        if gs.countdown_start and not (gs.is_success or gs.is_failure):
            elapsed = pygame.time.get_ticks() - gs.countdown_start
            if elapsed >= COUNTDOWN_MS:
                send_uart_command(self.goalkeeper.serial_port, 'stop')
                if gs.game_mode == "multi": send_uart_command(self.attacker.serial_port, 'stop')
                
                gs.final_col = self.goalkeeper.selected_col
                gs.chances_left -= 1
                
                if gs.game_mode == 'single':
                    gs.ball_col = random.randint(0, 4)
                else:
                    gs.ball_col = self.attacker.selected_col if self.attacker.selected_col is not None else random.randint(0, 4)

                gs.is_success = (gs.final_col == gs.ball_col)
                gs.is_failure = not gs.is_success

                if gs.is_success: gs.score += 1
                elif gs.is_failure and gs.game_mode == "multi": gs.attacker_score += 1
                
                gs.result_time = pygame.time.get_ticks()
                gs.countdown_start = None

        if gs.gif_start_time and (pygame.time.get_ticks() - gs.gif_start_time > GIF_SHOW_DURATION_MS):
            if gs.chances_left > 0:
                self.start_new_round()
            else:
                self.end_game()
    
    def end_game(self):
        gs = self.game_state
        face_path, gif_path, monitor_size = None, None, None

        if gs.game_mode == 'multi':
            if gs.score > gs.attacker_score:
                gs.final_rank, gs.end_video = "GOALKEEPER WINS!", self.resources["videos"]["victory"]
                face_path, gif_path = self.goalkeeper.face_filename, "../image/final_ronaldo/goalkeeper_win.gif"
            elif gs.attacker_score > gs.score:
                gs.final_rank, gs.end_video = "ATTACKER WINS!", self.resources["videos"]["defeat"]
                face_path, gif_path = self.attacker.face_filename, "../image/final_ronaldo/attacker_win.gif"
            else:
                gs.final_rank, gs.end_video = "DRAW", self.resources["videos"]["defeat"]
        else: # 1인 플레이
            if gs.score > gs.highscore:
                gs.highscore = gs.score
                save_highscore(gs.score)
            
            if gs.score >= 3:
                gs.final_rank, gs.end_video, gif_path = "Pro Keeper", self.resources["videos"]["victory"], "../image/final_ronaldo/goalkeeper_win.gif"
            else:
                gs.final_rank, gs.end_video, gif_path = "Rookie Keeper" if gs.score >= 1 else "Human Sieve", self.resources["videos"]["defeat"], "../image/lose_goalkeeper.gif"
            face_path = self.goalkeeper.face_filename

        monitor_size = (goalkeeper_monitor_width, screen_height)
        if face_path and gif_path and monitor_size:
            gs.synthesis_info = {"face_path": face_path, "gif_path": gif_path, "monitor_size": monitor_size}
            self.start_transition("synthesizing")
        else:
            if gs.end_video: gs.end_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.start_transition("end")

    def draw_webcam_view(self):
        screen.fill(BLACK)
        gs = self.game_state
        
        # 중앙 모니터 (슈팅 모션)
        # (슈팅 모션 비디오 로직은 길어서 생략, 기존 코드 참조하여 추가 가능)
        
        # 골키퍼/공격수 화면
        gk_surface = get_camera_surface(self.goalkeeper)
        if gk_surface: screen.blit(gk_surface, (self.goalkeeper.start_x, 0))
        self.draw_grid_lines(self.goalkeeper)
        
        if gs.game_mode == "multi":
            atk_surface = get_camera_surface(self.attacker)
            if atk_surface: screen.blit(atk_surface, (self.attacker.start_x, 0))
            self.draw_grid_lines(self.attacker)
        else:
            pygame.draw.rect(screen, BLACK, (self.attacker.start_x, 0, self.attacker.width, screen_height))
        
        # UI 그리기
        if gs.waiting_for_start:
            self.draw_waiting_ui()
        elif gs.countdown_start:
            self.handle_countdown_and_input()
        
        # 결과 표시
        self.draw_result_on_webcam()

    def draw_grid_lines(self, player):
        cell_w = player.width / 5
        for i in range(1, 5):
            pygame.draw.line(screen, GRID_COLOR, (player.start_x + i * cell_w, 0), (player.start_x + i * cell_w, screen_height), 2)

    def draw_waiting_ui(self):
        overlay = pygame.Surface((main_monitor_width, screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        screen.blit(overlay, (main_start_x, 0))
        start_text_l1 = title_font.render("시작하시겠습니까?", True, WHITE)
        start_text_l2 = font.render("(Press Space Bar)", True, WHITE)
        screen.blit(start_text_l1, start_text_l1.get_rect(center=(main_monitor_center_x, screen_height / 2 - 60)))
        screen.blit(start_text_l2, start_text_l2.get_rect(center=(main_monitor_center_x, screen_height / 2 + 40)))

    def handle_countdown_and_input(self):
        gs = self.game_state
        elapsed = pygame.time.get_ticks() - gs.countdown_start
        if elapsed < COUNTDOWN_MS:
            # 골키퍼 입력
            send_uart_command(self.goalkeeper.serial_port, 'grid')
            if self.goalkeeper.serial_port and self.goalkeeper.serial_port.in_waiting > 0:
                uart_bytes = self.goalkeeper.serial_port.read(self.goalkeeper.serial_port.in_waiting)
                valid_values = [b for b in uart_bytes if b in [1, 2, 3, 4, 5]]
                if valid_values: self.goalkeeper.selected_col = 5 - valid_values[-1]

            # 공격수 입력
            if gs.game_mode == "multi":
                send_uart_command(self.attacker.serial_port, 'kick')
                if self.attacker.serial_port and self.attacker.serial_port.in_waiting > 0:
                    uart_bytes = self.attacker.serial_port.read(self.attacker.serial_port.in_waiting)
                    valid_values = [b for b in uart_bytes if b in [1, 2, 3, 4, 5]]
                    if valid_values: self.attacker.selected_col = 5 - valid_values[-1]
            
            # 카운트다운 숫자 표시
            num_str = str(COUNTDOWN_SECONDS - (elapsed // 1000))
            text_surf = countdown_font.render(num_str, True, WHITE)
            screen.blit(text_surf, text_surf.get_rect(center=(self.goalkeeper.center_x, screen_height / 2)))
            if gs.game_mode == "multi":
                screen.blit(text_surf, text_surf.get_rect(center=(self.attacker.center_x, screen_height / 2)))

    def draw_result_on_webcam(self):
        gs = self.game_state
        
        # 결과 GIF 재생 로직
        should_play_gif = (gs.is_failure or gs.is_success) and gs.result_time and (pygame.time.get_ticks() - gs.result_time > RESULT_DELAY_MS)
        gif_key = 'failure' if gs.is_failure else ('success' if gs.is_success else None)
        
        if should_play_gif and gif_key:
            if not gs.gif_start_time:
                gs.gif_start_time = pygame.time.get_ticks()
                sound_key = "failed" if gs.is_failure else "success"
                if self.resources["sounds"].get(sound_key): self.resources["sounds"][sound_key].play()

            frame_list = self.resources['gif_frames'].get(gif_key, [])
            if frame_list:
                screen.fill(BLACK)
                current_time = pygame.time.get_ticks()
                if current_time - gs.gif_last_frame_time > GIF_FRAME_DURATION_MS:
                    gs.gif_frame_index = (gs.gif_frame_index + 1) % len(frame_list)
                    gs.gif_last_frame_time = current_time
                
                frame_surface = frame_list[gs.gif_frame_index]
                screen.blit(frame_surface, (goalkeeper_start_x, 0)) # 중앙 모니터에 표시
                # 양쪽 모니터에도 표시하려면 아래 코드 주석 해제
                # if gs.game_mode == "multi": screen.blit(frame_surface, (attacker_start_x, 0))

    def update_synthesizing(self):
        gs = self.game_state
        if not gs.synthesized_frames and gs.synthesis_info:
            pygame.display.flip() # "합성 중" 메시지를 먼저 보여줌
            info = gs.synthesis_info
            gs.synthesized_frames = create_synthesized_gif_frames(
                info["face_path"], info["gif_path"], info["monitor_size"]
            )
            if gs.end_video: gs.end_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.start_transition("end")

    def draw_synthesizing_screen(self):
        screen.fill(BLACK)
        loading_text = title_font.render("얼굴 합성 중...", True, WHITE)
        screen.blit(loading_text, loading_text.get_rect(center=(goalkeeper_monitor_center_x, screen_height / 2)))
        if self.game_state.game_mode == "multi":
            screen.blit(loading_text, loading_text.get_rect(center=(attacker_monitor_center_x, screen_height / 2)))
    
    def draw_info_screen(self):
        screen.fill(BLACK)
        if self.resources["images"].get("info_bg"):
            scaled_info_bg = pygame.transform.scale(self.resources["images"]["info_bg"], (main_monitor_width, screen_height))
            screen.blit(scaled_info_bg, (main_start_x, 0))
        
        title_surf = title_font.render("게임 방법", True, WHITE)
        screen.blit(title_surf, title_surf.get_rect(center=(main_monitor_center_x, 200)))
        
        text_1p = ["[1인 플레이]", "1. 스페이스 바를 누르면 5초 카운트가 시작됩니다.", "2. 5개의 영역 중 한 곳을 선택하여 공을 막습니다.", "3. 5번의 기회동안 최대한 많은 공을 막으세요!"]
        text_2p = ["[2인 플레이]", "1. 공격수는 공을 찰 방향을, 골키퍼는 막을 방향을 정합니다.", "2. 5번의 기회동안 더 많은 득점을 한 쪽이 승리합니다!"]
        
        y_pos = 450
        for line in text_1p:
            screen.blit(description_font.render(line, True, WHITE), (main_monitor_width/4 - 400, y_pos))
            y_pos += 90
        
        y_pos = 450
        for line in text_2p:
            screen.blit(description_font.render(line, True, WHITE), (main_monitor_width * 3/4 - 350, y_pos))
            y_pos += 90

    def draw_end_screen(self):
        screen.fill(BLACK)
        gs = self.game_state

        if gs.end_video:
            ret, frame = gs.end_video.read()
            if not ret:
                gs.end_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = gs.end_video.read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (main_monitor_width, screen_height))
                screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (main_start_x, 0))
        
        if gs.synthesized_frames:
            current_time = pygame.time.get_ticks()
            if current_time - gs.synthesized_last_update > 90:
                gs.synthesized_frame_index = (gs.synthesized_frame_index + 1) % len(gs.synthesized_frames)
                gs.synthesized_last_update = current_time
            current_frame = gs.synthesized_frames[gs.synthesized_frame_index]
            screen.blit(current_frame, (goalkeeper_start_x, 0))
            if gs.game_mode == "multi":
                screen.blit(current_frame, (attacker_start_x, 0))

        rank_surf = rank_font.render(gs.final_rank, True, GOLD_COLOR)
        screen.blit(rank_surf, rank_surf.get_rect(center=(main_monitor_center_x, screen_height / 2 - 150)))

        if gs.game_mode == "multi":
            score_str = f"{gs.score} : {gs.attacker_score}"
            score_surf = score_font.render(score_str, True, BLACK)
            screen.blit(score_surf, score_surf.get_rect(center=(main_monitor_center_x, screen_height/2)))
        else:
            score_surf = score_font.render(f"FINAL SCORE: {gs.score}", True, BLACK)
            screen.blit(score_surf, score_surf.get_rect(center=(main_monitor_center_x, screen_height/2)))
            highscore_surf = score_font.render(f"HIGH SCORE: {gs.highscore}", True, GOLD_COLOR)
            screen.blit(highscore_surf, highscore_surf.get_rect(center=(main_monitor_center_x, screen_height/2 + 80)))


if __name__ == '__main__':
    game = Game()
    game.run()