import pygame
import sys
import cv2
import numpy as np
import random
import os
import serial

from Button import ImageButton, MenuButton

# ========================================================== #
# UART 통신 함수
# ========================================================== #
def send_uart_command(serial_port, command):

    commands = {
        'grid': 225, 
        'face': 226,  
        'kick': 227,  
        'stop': 0     
    }
    byte_to_send = commands.get(command)
    if byte_to_send is not None and serial_port and serial_port.is_open:
        try:
            serial_port.write(bytes([byte_to_send]))
        except Exception as e:
            print(f"UART({command}) 데이터 송신 오류: {e}")

# =========================================
# 초기 설정 및 상수
# =========================================
pygame.init()

# 전체 화면 크기 감지
try:
    # Pygame 2.0.2+
    desktop_sizes = pygame.display.get_desktop_sizes()
    total_width = sum(w for w, h in desktop_sizes)
    max_height = max(h for w, h in desktop_sizes)
except AttributeError:
    # Older Pygame
    info = pygame.display.Info()
    total_width = info.current_w
    max_height = info.current_h

os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
screen = pygame.display.set_mode((total_width, max_height), pygame.NOFRAME)

screen_width = screen.get_width()
screen_height = screen.get_height()
pygame.display.set_caption("Penalty Kick Challenge")
 
# ================================================================= #
# *** 3-모니터 레이아웃 설정 (왼쪽 골키퍼 | 중앙 메인 | 오른쪽 공격수) *** #
# ================================================================= #
goalkeeper_monitor_width = screen_width // 3 
main_monitor_width = screen_width // 3
attacker_monitor_width = screen_width - goalkeeper_monitor_width - main_monitor_width

main_start_x = 0
attacker_start_x = attacker_monitor_width
goalkeeper_start_x = goalkeeper_monitor_width + main_monitor_width

goalkeeper_monitor_center_x = goalkeeper_start_x + (goalkeeper_monitor_width // 2)
main_monitor_center_x = main_start_x + (main_monitor_width // 2)
attacker_monitor_center_x = attacker_start_x + (attacker_monitor_width // 2)


# 색상 및 폰트 등
BLACK, WHITE, GRID_COLOR, RED = (0, 0, 0), (255, 255, 255), (0, 255, 0), (255, 0, 0)
HIGHLIGHT_COLOR, GOLD_COLOR = (255, 0, 0, 100), (255, 215, 0)
try: pygame.mixer.init()
except: pass
def load_font(path, size, default_size):
    try: return pygame.font.Font(path, size)
    except: return pygame.font.Font(None, default_size)
font = load_font("../fonts/netmarbleM.ttf", 40, 50)
small_font = load_font("../fonts/netmarbleM.ttf", 30, 40)
description_font = load_font("../fonts/netmarbleM.ttf", 50, 60)
title_font = load_font("../fonts/netmarbleB.ttf", 120, 130)
countdown_font = load_font("../fonts/netmarbleM.ttf", 200, 250)
score_font = load_font("../fonts/netmarbleB.ttf", 60, 70)
rank_font = load_font("../fonts/netmarbleB.ttf", 100, 110)

# =========================================
# 유틸리티 함수
# =========================================
def load_highscore():
    if not os.path.exists("highscore.txt"): return 0
    try:
        with open("highscore.txt", "r") as f: return int(f.read())
    except: return 0

def save_highscore(new_score):
    try:
        with open("highscore.txt", "w") as f: f.write(str(new_score))
    except Exception as e: print(f"최고 기록 저장 오류: {e}")

def get_scaled_rect(original_w, original_h, container_w, container_h):
    if original_h == 0 or container_h == 0: return (0,0)
    aspect_ratio = original_w / original_h
    container_aspect_ratio = container_w / container_h
    if aspect_ratio > container_aspect_ratio:
        new_w, new_h = container_w, int(container_w / aspect_ratio)
    else:
        new_w, new_h = int(container_h * aspect_ratio), container_h
    return new_w, new_h

# <<< [수정됨] GIF 프레임을 미리 로드하는 함수 추가 >>>
def load_gif_frames(video_path, size):
    """
    비디오 파일의 모든 프레임을 Pygame Surface 객체로 미리 변환하여 리스트로 반환합니다.
    이 함수는 게임 시작 시 한 번만 호출되어야 합니다.
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video file: {video_path}")
        return frames
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_pygame = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        frames.append(frame_pygame)
    
    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}.")
    return frames

# ==========================
# 메인 함수
# ==========================
def main():
    game_state = {
        "screen_state": "menu", "chances_left": 5, "score": 0, "highscore": load_highscore(),
        "attacker_score": 0, # (추가) 공격수 점수
        "final_rank": "", "end_video": None, "last_end_frame": None, "countdown_start": None,
        "selected_col": None, "final_col": None, "ball_col": None, "is_failure": False,
        "is_success": False, "result_time": None, "gif_start_time": None,
        "gif_frame_index": 0, "gif_last_frame_time": 0,
        "waiting_for_start": False, "game_mode": None, "is_capturing_face": False,
        
        "goalkeeper_face_data_buffer": [],
        "last_goalkeeper_face_coords": None,
        "captured_goalkeeper_face_filename": None,
        
        "attacker_face_data_buffer": [],
        "last_attacker_face_coords": None,
        "captured_attacker_face_filename": None,

        "attacker_selected_col": None 
    }
    transition_surface = pygame.Surface((screen_width, screen_height)); transition_surface.fill(BLACK)
    transition_alpha, transition_target, transition_speed = 0, None, 15
    fading_out, fading_in = False, False
    resources = {
        "cap": cv2.VideoCapture(0), # 골키퍼 카메라
        "cap2": cv2.VideoCapture(2),# 공격수 카메라
        "ser_goalkeeper": None, 
        "ser_attacker": None,   
        "sounds": {}, "images": {}, "videos": {},
        "gif_frames": {}, # <<< [수정됨] 미리 로드된 GIF 프레임을 저장할 공간 >>>
        "last_cam_frame": None,
        "last_cam2_frame": None 
    }
    
    if not resources["cap2"].isOpened():
        print("경고: 카메라 2(공격수용)를 열 수 없습니다. 오른쪽 모니터는 검은색으로 표시됩니다.")
        
    try:
        resources["ser_goalkeeper"] = serial.Serial('COM17', 9600, timeout=0)
        print("골키퍼 보드(COM17)가 성공적으로 연결되었습니다.")
    except serial.SerialException as e:
        print(f"오류: 골키퍼 보드(COM17)를 열 수 없습니다 - {e}")

    try:
        resources["ser_attacker"] = serial.Serial('COM13', 9600, timeout=0)
        print("공격수 보드(COM13)가 성공적으로 연결되었습니다.")
    except serial.SerialException as e:
        print(f"오류: 공격수 보드(COM13)를 열 수 없습니다 - {e}")

    try:
        resources["sounds"]["button"] = pygame.mixer.Sound("../sound/button_click.wav")
        resources["sounds"]["siu"] = pygame.mixer.Sound("../sound/SIUUUUU.wav")
        resources["sounds"]["success"] = pygame.mixer.Sound("../sound/야유.mp3")
        resources["sounds"]["failed"] = resources["sounds"]["siu"]
    except: pass
    try:
        ball_img = pygame.image.load("../image/final_ronaldo/Ball.png").convert_alpha()
        resources["images"]["scoreboard_ball"] = pygame.transform.scale(ball_img, (80, 80))
        resources["images"]["ball"] = pygame.transform.scale(ball_img, (200, 200))
        resources["images"]["info_bg"] = pygame.transform.scale(pygame.image.load("../image/info/info_back2.jpg").convert(), (screen_width, screen_height))
        resources["images"]["game_bg"] = pygame.transform.scale(pygame.image.load("../image/Ground.jpg").convert(), (screen_width, screen_height))
    except: pass
    
    # <<< [수정됨] GIF 파일을 프레임 단위로 미리 로드 >>>
    resources["gif_frames"] = {
        'success': load_gif_frames("../image/final_ronaldo/pk.gif", (main_monitor_width, screen_height)),
        'failure': load_gif_frames("../image/G.O.A.T/siuuu.gif", (main_monitor_width, screen_height))
    }
    
    # 일반 비디오는 기존 방식대로 로드
    resources["videos"]["victory"] = cv2.VideoCapture("../image/victory.gif")
    resources["videos"]["defeat"] = cv2.VideoCapture("../image/defeat.gif")
    resources["videos"]["menu_bg"] = cv2.VideoCapture("../image/game_thumbnail.mp4")
    resources["videos"]["attacker_win"] = cv2.VideoCapture("../image/final_ronaldo/attacker_win.gif")
    resources["videos"]["goalkeeper_win"] = cv2.VideoCapture("../image/final_ronaldo/goalkeeper_win.gif")
    bg_video = cv2.VideoCapture("../image/shoot.gif")
    if bg_video.isOpened():
        bg_video_total_frames = int(bg_video.get(cv2.CAP_PROP_FRAME_COUNT))
        bg_video_w = int(bg_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        bg_video_h = int(bg_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        bg_video_interval = 7000 / bg_video_total_frames if bg_video_total_frames > 0 else 0
    else:
        bg_video = None

    def start_transition(target_state):
        nonlocal transition_target, fading_out
        if not fading_out and not fading_in:
            transition_target, fading_out = target_state, True
            
    def reset_game_state(full_reset=True):
        game_state.update({
            "countdown_start": None, "selected_col": None, "final_col": None, "ball_col": None,
            "is_failure": False, "is_success": False, "result_time": None, "gif_start_time": None,
            "gif_frame_index": 0, # GIF 인덱스도 초기화
            "waiting_for_start": False, "is_capturing_face": False, 
            "attacker_selected_col": None,
            "goalkeeper_face_data_buffer": [], "last_goalkeeper_face_coords": None,
            "attacker_face_data_buffer": [], "last_attacker_face_coords": None,
        })
        if full_reset:
            game_state.update({
                "chances_left": 5, "score": 0,
                "attacker_score": 0, # (추가) 공격수 점수 초기화
                "captured_goalkeeper_face_filename": None,
                "captured_attacker_face_filename": None,
            })
            
    def start_new_round():
        reset_game_state(full_reset=False)
        if bg_video: bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        game_state["waiting_for_start"] = True
        
    def start_game(mode):
        if resources["sounds"].get("button"): resources["sounds"]["button"].play()
        game_state["game_mode"] = mode
        reset_game_state(full_reset=True)
        start_transition("face_capture")
        
    def go_to_menu():
        reset_game_state(full_reset=True)
        start_transition("menu")
        
    def go_to_game_select():
        reset_game_state(full_reset=True)
        start_transition("game")
        
    buttons = {
        "game": [MenuButton("1인 플레이", main_start_x + 50, 400, 350, 100, font, lambda: start_game("single"), sound=resources["sounds"].get("button")),
                 MenuButton("2인 플레이", main_start_x + 50, 500, 350, 100, font, lambda: start_game("multi"), sound=resources["sounds"].get("button")),
                 MenuButton("게임 설명", main_start_x + 50, 600, 350, 100, font, lambda: start_transition("info"), sound=resources["sounds"].get("button")),
                 ImageButton("../image/btn_back.png", 150, 150, 100, 100, go_to_menu, sound=resources["sounds"].get("button"))],
        "face_capture": [ImageButton("../image/btn_back.png", 150, 150, 100, 100, go_to_game_select, sound=resources["sounds"].get("button"))],
        "webcam_view": [ImageButton("../image/btn_back.png", 150, 150, 100, 100, go_to_game_select, sound=resources["sounds"].get("button"))],
        "info": [ImageButton("../image/btn_exit.png", main_monitor_center_x*2 - 150, 150, 100, 100, go_to_game_select, sound=resources["sounds"].get("button"))],
        "end": [ImageButton("../image/btn_restart.png", main_monitor_center_x - 300, screen_height - 250, 400, 250, go_to_game_select, sound=resources["sounds"].get("button")),
                ImageButton("../image/btn_main_menu.png", main_monitor_center_x + 300, screen_height - 250, 400, 250, go_to_menu, sound=resources["sounds"].get("button"))]
    }
    clock = pygame.time.Clock()

    # ================================================================= #
    # *** (수정됨) draw_player_info 함수 *** #
    # ================================================================= #
    def draw_player_info(surface, start_x, width, player_type): # (수정) player_type 인자 추가
        # 반투명 오버레이
        overlay = pygame.Surface((width, screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        surface.blit(overlay, (start_x, 0))

        # --- 얼굴 이미지 표시 부분 삭제 ---
        
        # (수정) 표시할 점수를 결정
        display_score = 0
        if player_type == 'goalkeeper':
            display_score = game_state['score']
        elif player_type == 'attacker':
            display_score = game_state['attacker_score']

        # 점수 텍스트 (오른쪽 상단)
        score_text = score_font.render(f"SCORE: {display_score}", True, WHITE) # (수정) display_score 사용
        score_rect = score_text.get_rect(topright=(start_x + width - 20, 20))
        surface.blit(score_text, score_rect)

        # 남은 기회 텍스트 (점수 아래)
        chances_text = font.render("CHANCES", True, WHITE)
        chances_rect = chances_text.get_rect(topright=(start_x + width - 20, score_rect.bottom + 10))
        surface.blit(chances_text, chances_rect)
        
        # 남은 기회 공 이미지 (CHANCES 텍스트 아래)
        if resources["images"].get("scoreboard_ball"):
            ball_width = resources["images"]["scoreboard_ball"].get_width()
            total_balls_width = game_state["chances_left"] * (ball_width + 10) - 10
            start_ball_x = (start_x + width - 20) - total_balls_width

            for i in range(game_state["chances_left"]):
                surface.blit(resources["images"]["scoreboard_ball"], (start_ball_x + i * (ball_width + 10), chances_rect.bottom + 10))

    def draw_menu_or_game_screen(state):
        pygame.draw.rect(screen, BLACK, (goalkeeper_start_x, 0, goalkeeper_monitor_width, screen_height))
        pygame.draw.rect(screen, BLACK, (attacker_start_x, 0, attacker_monitor_width, screen_height))
        
        ret, frame = resources["videos"]["menu_bg"].read()
        if not ret:
            resources["videos"]["menu_bg"].set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = resources["videos"]["menu_bg"].read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized_main = cv2.resize(frame_rgb, (main_monitor_width, screen_height))
            screen.blit(pygame.surfarray.make_surface(frame_resized_main.swapaxes(0, 1)), (main_start_x, 0))
        else:
            pygame.draw.rect(screen, BLACK, (main_start_x, 0, main_monitor_width, screen_height))

        if state == "menu":
            font.set_bold(True)
            start_text_l1 = font.render("게임을 시작하려면 아무 키나 누르세요", True, WHITE)
            font.set_bold(False)
            description_font.set_bold(True)
            start_text_l2 = description_font.render("PRESS ANY KEY", True, WHITE)
            description_font.set_bold(False)
            y_pos_l1 = screen_height * 0.75
            y_pos_l2 = y_pos_l1 + 80
            screen.blit(start_text_l1, start_text_l1.get_rect(center=(main_monitor_center_x, y_pos_l1)))
            screen.blit(start_text_l2, start_text_l2.get_rect(center=(main_monitor_center_x, y_pos_l2)))


    def draw_face_capture_screen():
        screen.fill(BLACK)
    
        def draw_capture_ui(surface, start_x, width, center_x, captured_filename, player_name):
            overlay = pygame.Surface((width, screen_height), pygame.SRCALPHA)
            surface.blit(overlay, (start_x, 0))
            if not captured_filename:
                overlay.fill((0, 0, 0, 128))
                title_surf = title_font.render(f"{player_name} 얼굴 캡처", True, WHITE)
                desc_surf = font.render("얼굴을 중앙의 사각형에 맞춰주세요", True, WHITE) # 설명 수정
                surface.blit(title_surf, title_surf.get_rect(center=(center_x, screen_height/2 - 80)))
                surface.blit(desc_surf, desc_surf.get_rect(center=(center_x, screen_height/2 + 40)))
                # 캡처 영역 정의
                capture_area_rect = pygame.Rect(center_x - 100, screen_height // 2- 350, 200, 200)
                pygame.draw.rect(surface, GRID_COLOR, capture_area_rect, 3, border_radius=15)
            else:
                overlay.fill((0, 0, 0, 200))
                captured_text = title_font.render("캡처 완료!", True, GOLD_COLOR)
                surface.blit(captured_text, captured_text.get_rect(center=(center_x, screen_height / 2)))

        # 공격수(웹캠1) 화면 표시
        if game_state["game_mode"] == "multi":
            if resources["cap2"].isOpened():
                ret_cam2, frame_cam2 = resources["cap2"].read()
                if ret_cam2:
                    resources["last_cam2_frame"] = frame_cam2
                    frame_cam2_flipped = cv2.flip(frame_cam2, 1)
                    frame_cam2_rgb = cv2.cvtColor(frame_cam2_flipped, cv2.COLOR_BGR2RGB)
                    cam2_surf = pygame.surfarray.make_surface(frame_cam2_rgb.swapaxes(0, 1))
                    cam2_surf_scaled = pygame.transform.scale(cam2_surf, (attacker_monitor_width, screen_height))
                    screen.blit(cam2_surf_scaled, (attacker_start_x, 0))

            if not game_state["captured_goalkeeper_face_filename"]:
                overlay = pygame.Surface((attacker_monitor_width, screen_height), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 200))
                wait_text = title_font.render("대기 중...", True, WHITE)
                overlay.blit(wait_text, wait_text.get_rect(center=(attacker_monitor_width/2, screen_height/2)))
                screen.blit(overlay, (attacker_start_x, 0))
            else:
                draw_capture_ui(screen, attacker_start_x, attacker_monitor_width, attacker_monitor_center_x, game_state["captured_attacker_face_filename"], "공격수")
        else:
            pygame.draw.rect(screen, BLACK, (attacker_start_x, 0, attacker_monitor_width, screen_height))

        # 골키퍼(웹캠2) 화면 표시
        ret_cam, frame_cam = resources["cap"].read()
        if ret_cam:
            resources["last_cam_frame"] = frame_cam
            frame_cam_flipped = cv2.flip(frame_cam, 1)
            frame_cam_rgb = cv2.cvtColor(frame_cam_flipped, cv2.COLOR_BGR2RGB)
            cam_surf = pygame.surfarray.make_surface(frame_cam_rgb.swapaxes(0, 1))
            cam_surf_scaled = pygame.transform.scale(cam_surf, (goalkeeper_monitor_width, screen_height))
            screen.blit(cam_surf_scaled, (goalkeeper_start_x, 0))
        draw_capture_ui(screen, goalkeeper_start_x, goalkeeper_monitor_width, goalkeeper_monitor_center_x, game_state["captured_goalkeeper_face_filename"], "골키퍼")

        # UART 신호 전송 및 데이터 수신/처리
        if not game_state["is_capturing_face"]:
            send_uart_command(resources["ser_goalkeeper"], 'face')
            game_state["is_capturing_face"] = True

        # 골키퍼 캡처 로직
        if not game_state["captured_goalkeeper_face_filename"]:
            if resources["ser_goalkeeper"] and resources["ser_goalkeeper"].in_waiting > 0:
                uart_bytes = resources["ser_goalkeeper"].read(resources["ser_goalkeeper"].in_waiting)
                for byte in uart_bytes: game_state["goalkeeper_face_data_buffer"].append(byte & 31)

            if len(game_state["goalkeeper_face_data_buffer"]) >= 4:
                chunks = game_state["goalkeeper_face_data_buffer"]
                full_data = (chunks[0] << 15) | (chunks[1] << 10) | (chunks[2] << 5) | chunks[3]
                y_coord_raw, x_coord_raw = (full_data >> 10) & 0x3FF, full_data & 0x3FF
                game_state["last_goalkeeper_face_coords"] = {"raw": (x_coord_raw, y_coord_raw),"scaled": (goalkeeper_start_x + int(x_coord_raw * (goalkeeper_monitor_width / 640)), int(y_coord_raw * (screen_height / 480)))}

                # [수정] 자동 캡처 로직
                coords = game_state["last_goalkeeper_face_coords"]
                capture_area = pygame.Rect(goalkeeper_monitor_center_x - 100, screen_height // 2 - 350, 200, 200)
                if capture_area.collidepoint(coords["scaled"]):
                    filename = capture_and_save_face(resources["last_cam_frame"], coords["raw"], "captured_goalkeeper_face.png")
                    if filename:
                        game_state["captured_goalkeeper_face_filename"] = filename
                        send_uart_command(resources["ser_goalkeeper"], 'stop')
                        if game_state["game_mode"] == "multi":
                            send_uart_command(resources["ser_attacker"], 'face')
                        else:
                            game_state["is_capturing_face"] = False
                            start_new_round()
                            start_transition("webcam_view")

                game_state["goalkeeper_face_data_buffer"] = chunks[4:]

        # 공격수 캡처 로직 (멀티플레이)
        elif game_state["game_mode"] == "multi" and not game_state["captured_attacker_face_filename"]:
            if resources["ser_attacker"] and resources["ser_attacker"].in_waiting > 0:
                uart_bytes = resources["ser_attacker"].read(resources["ser_attacker"].in_waiting)
                for byte in uart_bytes: game_state["attacker_face_data_buffer"].append(byte & 31)

            if len(game_state["attacker_face_data_buffer"]) >= 4:
                chunks = game_state["attacker_face_data_buffer"]
                full_data = (chunks[0] << 15) | (chunks[1] << 10) | (chunks[2] << 5) | chunks[3]
                y_coord_raw, x_coord_raw = (full_data >> 10) & 0x3FF, full_data & 0x3FF
                game_state["last_attacker_face_coords"] = {"raw": (x_coord_raw, y_coord_raw), "scaled": (attacker_start_x + int(x_coord_raw * (attacker_monitor_width / 640)), int(y_coord_raw * (screen_height / 480)))}

                # [수정] 자동 캡처 로직
                coords = game_state["last_attacker_face_coords"]
                capture_area = pygame.Rect(attacker_monitor_center_x - 100, screen_height // 2 - 350, 200, 200)
                if capture_area.collidepoint(coords["scaled"]):
                    filename = capture_and_save_face(resources["last_cam2_frame"], coords["raw"], "captured_attacker_face.png")
                    if filename:
                        game_state["captured_attacker_face_filename"] = filename
                        send_uart_command(resources["ser_attacker"], 'stop')
                        game_state["is_capturing_face"] = False
                        start_new_round()
                        start_transition("webcam_view")

                game_state["attacker_face_data_buffer"] = chunks[4:]

        if game_state["last_goalkeeper_face_coords"]:
            pygame.draw.circle(screen, RED, game_state["last_goalkeeper_face_coords"]["scaled"], 20, 4)
        if game_state["last_attacker_face_coords"]:
            pygame.draw.circle(screen, RED, game_state["last_attacker_face_coords"]["scaled"], 20, 4)

    def draw_webcam_view():
        screen.fill(BLACK)
        
        # 중앙 메인 모니터 배경 영상 처리
        if bg_video and (game_state["waiting_for_start"] or game_state["countdown_start"]):
            if game_state["waiting_for_start"]: bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                elapsed = pygame.time.get_ticks() - game_state["countdown_start"]
                current_frame_pos = int(elapsed / bg_video_interval)
                if current_frame_pos < bg_video_total_frames: bg_video.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
            ret_vid, frame_vid = bg_video.read()
            if ret_vid:
                new_w, new_h = get_scaled_rect(bg_video_w, bg_video_h, main_monitor_width, screen_height)
                pos_x = main_start_x + (main_monitor_width - new_w) // 2
                pos_y = (screen_height - new_h) // 2
                frame_vid_rgb = cv2.cvtColor(frame_vid, cv2.COLOR_BGR2RGB)
                frame_vid_resized = cv2.resize(frame_vid_rgb, (new_w, new_h))
                screen.blit(pygame.surfarray.make_surface(frame_vid_resized.swapaxes(0, 1)), (pos_x, pos_y))
        
        # --- 골키퍼 화면 그리기 (모드와 상관없이 항상 실행) ---
        ret_cam, frame_cam = resources["cap"].read()
        if ret_cam:
            frame_cam_flipped = cv2.flip(frame_cam, 1)
            frame_cam_rgb = cv2.cvtColor(frame_cam_flipped, cv2.COLOR_BGR2RGB)
            frame_cam_resized = cv2.resize(frame_cam_rgb, (goalkeeper_monitor_width, screen_height))
            screen.blit(pygame.surfarray.make_surface(frame_cam_resized.swapaxes(0, 1)), (goalkeeper_start_x, 0))

        cell_w_gk = goalkeeper_monitor_width / 5
        for i in range(1, 5):
            pygame.draw.line(screen, GRID_COLOR, (goalkeeper_start_x + i * cell_w_gk, 0), (goalkeeper_start_x + i * cell_w_gk, screen_height), 2)
        
        # 골키퍼 정보(스코어보드) 그리기
        draw_player_info(screen, goalkeeper_start_x, goalkeeper_monitor_width, "goalkeeper") # (수정) "goalkeeper" 타입 전달

        # --- 공격수 화면 그리기 (모드에 따라 분기) ---
        cell_w_atk = attacker_monitor_width / 5
        if game_state["game_mode"] == "multi":
            # [멀티플레이 모드] 공격수 화면에 모든 요소 그리기
            if resources["cap2"].isOpened():
                ret_cam2, frame_cam2 = resources["cap2"].read()
                if ret_cam2:
                    frame_cam2_flipped = cv2.flip(frame_cam2, 1)
                    frame_cam2_rgb = cv2.cvtColor(frame_cam2_flipped, cv2.COLOR_BGR2RGB)
                    cam2_surf = pygame.surfarray.make_surface(frame_cam2_rgb.swapaxes(0, 1))
                    cam2_surf_scaled = pygame.transform.scale(cam2_surf, (attacker_monitor_width, screen_height))
                    screen.blit(cam2_surf_scaled, (attacker_start_x, 0))
            
            for i in range(1, 5):
                pygame.draw.line(screen, GRID_COLOR, (attacker_start_x + i * cell_w_atk, 0), (attacker_start_x + i * cell_w_atk, screen_height), 2)
            
            if game_state["attacker_selected_col"] is not None:
                pygame.draw.rect(screen, RED, (attacker_start_x + game_state["attacker_selected_col"] * cell_w_atk, 0, cell_w_atk, screen_height), 10)

            if game_state["ball_col"] is not None and resources["images"]["ball"]:
                ball_rect_atk = resources["images"]["ball"].get_rect(center=(attacker_start_x + game_state["ball_col"] * cell_w_atk + cell_w_atk / 2, screen_height / 2))
                screen.blit(resources["images"]["ball"], ball_rect_atk)
            
            draw_player_info(screen, attacker_start_x, attacker_monitor_width, "attacker") # (수정) "attacker" 타입 전달
        else:
            # [싱글플레이 모드] 공격수 화면을 검은색으로 채움
            pygame.draw.rect(screen, BLACK, (attacker_start_x, 0, attacker_monitor_width, screen_height))

        # --- 중앙 및 공통 UI 요소 처리 ---
        if game_state["waiting_for_start"]:
            overlay = pygame.Surface((main_monitor_width, screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            screen.blit(overlay, (main_start_x, 0))
            start_text_l1 = title_font.render("시작하시겠습니까?", True, WHITE)
            start_text_l2 = font.render("(Press Space Bar)", True, WHITE)
            screen.blit(start_text_l1, start_text_l1.get_rect(center=(main_monitor_center_x, screen_height/2 - 60)))
            screen.blit(start_text_l2, start_text_l2.get_rect(center=(main_monitor_center_x, screen_height/2 + 40)))
        
        elif game_state["countdown_start"]:
            elapsed = pygame.time.get_ticks() - game_state["countdown_start"]
            if elapsed < 5000:
                send_uart_command(resources["ser_goalkeeper"], 'grid')
                if resources["ser_goalkeeper"] and resources["ser_goalkeeper"].in_waiting > 0:
                    try:
                        uart_bytes = resources["ser_goalkeeper"].read(resources["ser_goalkeeper"].in_waiting)
                        if uart_bytes:
                            valid_values = [b for b in uart_bytes if b in [1, 2, 3, 4, 5]]
                            if valid_values: game_state["selected_col"] = 5 - valid_values[-1]
                    except Exception as e: print(f"UART(Grid) 데이터 수신 오류: {e}")

                if game_state["game_mode"] == "multi":
                    send_uart_command(resources["ser_attacker"], 'kick')
                    if resources["ser_attacker"] and resources["ser_attacker"].in_waiting > 0:
                        try:
                            uart_bytes_attacker = resources["ser_attacker"].read(resources["ser_attacker"].in_waiting)
                            if uart_bytes_attacker:
                                valid_values_attacker = [b for b in uart_bytes_attacker if b in [1, 2, 3, 4, 5]]
                                if valid_values_attacker: game_state["attacker_selected_col"] = 5 - valid_values_attacker[-1]
                        except Exception as e: print(f"UART(Attacker Kick) 데이터 수신 오류: {e}")

                if game_state["selected_col"] is not None:
                    pygame.draw.rect(screen, GOLD_COLOR, (goalkeeper_start_x + game_state["selected_col"] * cell_w_gk, 0, cell_w_gk, screen_height), 10)
                
                num_str = str(5 - (elapsed // 1000))
                text_surf = countdown_font.render(num_str, True, WHITE)
                screen.blit(text_surf, text_surf.get_rect(center=(goalkeeper_monitor_center_x, screen_height/2)))

                if game_state["game_mode"] == "multi":
                    screen.blit(text_surf, text_surf.get_rect(center=(attacker_monitor_center_x, screen_height/2)))

            else:
                if game_state["final_col"] is None:
                    send_uart_command(resources["ser_goalkeeper"], 'stop')
                    if game_state["game_mode"] == "multi": send_uart_command(resources["ser_attacker"], 'stop')
                    game_state["final_col"] = game_state["selected_col"]
                    game_state["chances_left"] -= 1
                    if game_state["game_mode"] == 'single':
                        game_state["ball_col"] = random.randint(0, 4)
                    else: 
                        game_state["ball_col"] = game_state["attacker_selected_col"] if game_state["attacker_selected_col"] is not None else random.randint(0, 4)
                    
                    game_state["is_success"] = (game_state["final_col"] == game_state["ball_col"])
                    game_state["is_failure"] = not game_state["is_success"]

                    # (수정) 점수 계산 로직
                    if game_state["is_success"]: 
                        game_state["score"] += 1
                    elif game_state["is_failure"] and game_state["game_mode"] == "multi":
                        game_state["attacker_score"] += 1
                        
                    game_state["result_time"] = pygame.time.get_ticks()
                    game_state["countdown_start"] = None
        
        # 골키퍼 최종 선택 및 공 위치 그리기
        if game_state["final_col"] is not None:
            highlight_surf = pygame.Surface((cell_w_gk, screen_height), pygame.SRCALPHA); highlight_surf.fill(HIGHLIGHT_COLOR)
            screen.blit(highlight_surf, (goalkeeper_start_x + game_state["final_col"] * cell_w_gk, 0))
            
        if game_state["ball_col"] is not None and resources["images"]["ball"]:
            ball_rect_gk = resources["images"]["ball"].get_rect(center=(goalkeeper_start_x + game_state["ball_col"] * cell_w_gk + cell_w_gk / 2, screen_height / 2))
            screen.blit(resources["images"]["ball"], ball_rect_gk)

    def draw_game_screen():
        pygame.draw.rect(screen, BLACK, (goalkeeper_start_x, 0, goalkeeper_monitor_width, screen_height))
        pygame.draw.rect(screen, BLACK, (attacker_start_x, 0, attacker_monitor_width, screen_height))

        if resources["images"].get("game_bg"):
            scaled_game_bg = pygame.transform.scale(resources["images"]["game_bg"], (main_monitor_width, screen_height))
            screen.blit(scaled_game_bg, (main_start_x, 0))
        
        else:
            # 이미지가 없는 경우를 대비해 검은색으로 채웁니다.
            pygame.draw.rect(screen, BLACK, (main_start_x, 0, main_monitor_width, screen_height))

    def draw_info_screen():
        pygame.draw.rect(screen, BLACK, (goalkeeper_start_x, 0, goalkeeper_monitor_width, screen_height))
        pygame.draw.rect(screen, BLACK, (attacker_start_x, 0, attacker_monitor_width, screen_height))

        if resources["images"].get("info_bg"):
            scaled_info_bg = pygame.transform.scale(resources["images"]["info_bg"], (main_monitor_width, screen_height))
            screen.blit(scaled_info_bg, (main_start_x, 0))
        
        else:
            # 이미지가 없는 경우를 대비해 검은색으로 채웁니다.
            pygame.draw.rect(screen, BLACK, (main_start_x, 0, main_monitor_width, screen_height))

        # 3. "게임 방법" 타이틀을 중앙 모니터의 중앙에 맞게 그립니다.
        title_surf = title_font.render("게임 방법", True, WHITE)
        screen.blit(title_surf, title_surf.get_rect(center=(main_monitor_center_x, 200)))
        
        text_1p = ["[1인 플레이]", "", "1. 스페이스 바를 누르면 5초 카운트 다운이 시작됩니다.", "", "2. 5개의 영역 중 한 곳을 선택합니다.", "", "3. 5번의 기회동안 최대한 많은 공을 막으세요!"]
        text_2p = ["[2인 플레이]", "", "1. 스페이스 바를 누르면 5초 카운트 다운이 시작됩니다.", "", "2. 공격수와 골키퍼로 나뉩니다.", "", "3. 공격수는 공을 찰 방향을 정합니다.", "", "4. 골키퍼는 공을 막을 방향을 정합니다.", "", "5. 5번의 기회동안 더 많은 득점을 한 쪽이 승리합니다!"]
        for i, line in enumerate(text_1p): screen.blit(description_font.render(line, True, WHITE), (main_monitor_width/4 - 550, 475 + i*75))
        for i, line in enumerate(text_2p): screen.blit(description_font.render(line, True, WHITE), (main_monitor_width*3/4 - 500, 475 + i*75))

    # (수정) draw_end_screen 함수
    def draw_end_screen():
        screen.fill(BLACK)
        
        # --- 1. 배경 영상 재생 ---
        if game_state["end_video"]:
            read_new_frame = not (game_state["end_video"] == resources["videos"]["defeat"] and pygame.time.get_ticks() % 2 == 0)
            if read_new_frame:
                ret, frame = game_state["end_video"].read()
                if not ret:
                    game_state["end_video"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = game_state["end_video"].read()
                if ret: game_state["last_end_frame"] = frame
            if game_state["last_end_frame"] is not None:
                frame_rgb = cv2.cvtColor(game_state["last_end_frame"], cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (main_monitor_width, screen_height))
                screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (main_start_x, 0))
            
                # --- 2. [추가된 부분] 골키퍼 및 공격수 모니터 배경 영상 ---
        winner = game_state.get("winner")

        # 기본적으로 양쪽 플레이어 화면을 검은색으로 채웁니다.
        pygame.draw.rect(screen, BLACK, (attacker_start_x, 0, attacker_monitor_width, screen_height))
        pygame.draw.rect(screen, BLACK, (goalkeeper_start_x, 0, goalkeeper_monitor_width, screen_height))

        # 승리한 플레이어의 화면에 맞는 영상을 재생합니다.
        video_to_play = None
        target_monitor_width = 0

        if winner == "attacker":
            video_to_play = resources["videos"]["attacker_win"]
            target_monitor_width = attacker_monitor_width
            last_frame_key = "last_attacker_win_frame" # 프레임 저장을 위한 고유 키

        elif winner == "goalkeeper":
            video_to_play = resources["videos"]["goalkeeper_win"]
            target_monitor_width = goalkeeper_monitor_width
            last_frame_key = "last_goalkeeper_win_frame" # 프레임 저장을 위한 고유 키

        # 재생할 영상이 있다면 화면에 그립니다.
        if video_to_play:
            ret, frame = video_to_play.read()
            if not ret:
                video_to_play.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = video_to_play.read()
            if ret:
                game_state[last_frame_key] = frame

            if game_state.get(last_frame_key) is not None:
                frame_rgb = cv2.cvtColor(game_state[last_frame_key], cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (target_monitor_width, screen_height))
                video_surface = pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1))
                screen.blit(video_surface, (goalkeeper_start_x, 0))
                if game_state["game_mode"] == "multi":
                    screen.blit(video_surface, (attacker_start_x, 0))
                
        

        # --- 2. 승자에 따라 표시할 정보 결정 ---
        winner = game_state.get("winner")
        face_filename_to_show = None
        
        if game_state["game_mode"] == "multi":
            if winner == "goalkeeper":
                face_filename_to_show = game_state["captured_goalkeeper_face_filename"]
            elif winner == "attacker":
                face_filename_to_show = game_state["captured_attacker_face_filename"]
            # 무승부일 경우, 두 얼굴을 모두 표시하거나 한 명을 기본값으로 할 수 있습니다. 여기서는 골키퍼를 기본으로 합니다.
            else: 
                face_filename_to_show = game_state["captured_goalkeeper_face_filename"]
        else: # 싱글 플레이
            face_filename_to_show = game_state["captured_goalkeeper_face_filename"]

        # --- 3. 결정된 얼굴 이미지 표시 ---
        face_img_scaled = None
        if face_filename_to_show and os.path.exists(face_filename_to_show):
            try:
                face_img = pygame.image.load(face_filename_to_show)
                face_img_scaled = pygame.transform.scale(face_img, (int(face_img.get_width()), int(face_img.get_height())))
                face_rect = face_img_scaled.get_rect(center=(main_monitor_center_x, screen_height / 2))
                screen.blit(face_img_scaled, face_rect)
            except Exception as e: print(f"이미지 파일 불러오기 오류: {e}")

        # --- 4. 텍스트 위치 계산 ---
        if face_img_scaled:
            face_rect = face_img_scaled.get_rect(center=(main_monitor_center_x, screen_height / 2))
            rank_y_pos, score_y_pos = face_rect.top - 100, face_rect.bottom + 80
        else:
            rank_y_pos, score_y_pos = screen_height/2 - 150, screen_height/2

        # --- 5. 최종 랭크(승패 결과) 표시 ---
        rank_surf = rank_font.render(game_state["final_rank"], True, GOLD_COLOR)
        screen.blit(rank_surf, rank_surf.get_rect(center=(main_monitor_center_x, rank_y_pos)))
        
        # --- 6. 최종 점수 표시 ---
        if game_state["game_mode"] == "multi":
            # 멀티플레이: "골키퍼 점수 : 공격수 점수" 형태로 표시
            score_str = f"{game_state['score']} : {game_state['attacker_score']}"
            goalkeeper_text = score_font.render("Goalkeeper", True, WHITE)
            attacker_text = score_font.render("Attacker", True, WHITE)
            score_surf = score_font.render(score_str, True, WHITE)
            
            total_width = goalkeeper_text.get_width() + score_surf.get_width() + attacker_text.get_width() + 100
            start_x = main_monitor_center_x - total_width / 2

            screen.blit(goalkeeper_text, (start_x, score_y_pos))
            screen.blit(score_surf, (start_x + goalkeeper_text.get_width() + 50, score_y_pos))
            screen.blit(attacker_text, (start_x + goalkeeper_text.get_width() + score_surf.get_width() + 100, score_y_pos))
        else:
            # 싱글플레이: 기존 방식대로 표시
            score_surf = score_font.render(f"FINAL SCORE: {game_state['score']}", True, WHITE)
            screen.blit(score_surf, score_surf.get_rect(center=(main_monitor_center_x, score_y_pos)))
            
            highscore_surf = score_font.render(f"HIGH SCORE: {game_state['highscore']}", True, GOLD_COLOR)
            highscore_y_pos = score_y_pos + 80
            screen.blit(highscore_surf, highscore_surf.get_rect(center=(main_monitor_center_x, highscore_y_pos)))
    
    def capture_and_save_face(original_frame, raw_coords, output_filename):
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
    
    def handle_events():
        nonlocal running
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                return
            if game_state["screen_state"] == "menu" and event.type == pygame.KEYDOWN:
                start_transition("game")

            elif game_state["screen_state"] == "webcam_view" and game_state["waiting_for_start"]:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    game_state["countdown_start"] = pygame.time.get_ticks()
                    game_state["waiting_for_start"] = False

            if not (fading_in or fading_out):
                for button in buttons.get(game_state["screen_state"], []):
                    button.handle_event(event)

    running = True
    while running:
        handle_events()
        if not (fading_in or fading_out):
            for button in buttons.get(game_state["screen_state"], []): button.update()
            
        current_screen = game_state["screen_state"]
        
        if current_screen in ["menu"]:
            draw_menu_or_game_screen(current_screen)
        elif current_screen == "face_capture":
            draw_face_capture_screen()
        elif current_screen == "webcam_view":
            # 1. 배경(웹캠, 그리드, 점수 등)을 항상 한 번만 그립니다.
            draw_webcam_view()

            # (수정) 2. 라운드 종료 및 다음 라운드 전환 로직을 처리합니다.
            if game_state["gif_start_time"] and (pygame.time.get_ticks() - game_state["gif_start_time"] > 3000):
                if game_state["chances_left"] > 0:
                    start_new_round()
                else:
                    # 게임 모드에 따라 최종 결과 처리
                    if game_state["game_mode"] == 'multi':
                        if game_state["score"] > game_state["attacker_score"]:
                            game_state["winner"] = "goalkeeper"
                            game_state["final_rank"] = "GOALKEEPER WINS!"
                            game_state["end_video"] = resources["videos"]["victory"]
                        elif game_state["attacker_score"] > game_state["score"]:
                            game_state["winner"] = "attacker"
                            game_state["final_rank"] = "ATTACKER WINS!"
                            game_state["end_video"] = resources["videos"]["victory"]
                        else:
                            game_state["winner"] = "draw"
                            game_state["final_rank"] = "DRAW"
                            game_state["end_video"] = resources["videos"]["defeat"]
                    else: # 싱글 플레이 모드
                        game_state["winner"] = "goalkeeper" # 싱글플레이에서는 항상 골키퍼가 기준
                        if game_state["score"] > game_state["highscore"]:
                            game_state["highscore"] = game_state["score"]
                            save_highscore(game_state["score"])
                        score = game_state["score"]
                        if score == 5: game_state["final_rank"], game_state["end_video"] = "THE WALL", resources["videos"]["victory"]
                        elif score >= 3: game_state["final_rank"], game_state["end_video"] = "Pro Keeper", resources["videos"]["victory"]
                        elif score >= 1: game_state["final_rank"], game_state["end_video"] = "Rookie Keeper", resources["videos"]["defeat"]
                        else: game_state["final_rank"], game_state["end_video"] = "Human Sieve", resources["videos"]["defeat"]
                    
                    if game_state["end_video"]: 
                        game_state["end_video"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                    start_transition("end")
            
            # <<< [수정됨] 시작: 최적화된 GIF 재생 로직 >>>
            should_play_gif = (game_state["is_failure"] or game_state["is_success"]) and game_state["result_time"] and (pygame.time.get_ticks() - game_state["result_time"] > 2000)
            GIF_FRAME_DURATION = 70

            gif_key = None
            if should_play_gif:
                gif_key = 'failure' if game_state["is_failure"] else 'success'
            
            if gif_key:
                # GIF 재생 시작 시점 초기화
                if not game_state["gif_start_time"]:
                    game_state["gif_start_time"] = pygame.time.get_ticks()
                    game_state["gif_frame_index"] = 0  # 인덱스 초기화
                    game_state["gif_last_frame_time"] = pygame.time.get_ticks() 
                    if game_state["is_success"] and resources["sounds"].get("success"):
                        resources["sounds"]["success"].play()
                    elif game_state["is_failure"] and resources["sounds"].get("failed"):
                        resources["sounds"]["failed"].play()
                
                # 최적화된 그리기 로직
                screen.fill(BLACK)
                frame_list = resources['gif_frames'].get(gif_key)

                if frame_list:
                    current_index = game_state['gif_frame_index']
                    frame_surface = frame_list[current_index]
                    screen.blit(frame_surface, (goalkeeper_start_x, 0))

                    if game_state["game_mode"] == "multi":
                        screen.blit(frame_surface, (attacker_start_x, 0))

                    current_time = pygame.time.get_ticks()
                    if current_time - game_state["gif_last_frame_time"] > GIF_FRAME_DURATION:
                        game_state['gif_frame_index'] = (current_index + 1) % len(frame_list)
                        game_state["gif_last_frame_time"] = current_time # 마지막 업데이트 시간 갱신

        elif current_screen == "info":
            draw_info_screen()
        elif current_screen == "game":
            draw_game_screen()
        elif current_screen == "end":
            draw_end_screen()

        for button in buttons.get(current_screen, []): 
            button.draw(screen)

        if fading_out or fading_in:
            if fading_out:
                transition_alpha = min(255, transition_alpha + transition_speed)
                if transition_alpha == 255:
                    fading_out, fading_in = False, True
                    game_state["screen_state"] = transition_target
            else:
                transition_alpha = max(0, transition_alpha - transition_speed)
                if transition_alpha == 0: fading_in = False
            transition_surface.set_alpha(transition_alpha); screen.blit(transition_surface, (0, 0))

        pygame.display.flip()
        clock.tick(60)

    if resources["cap"]: resources["cap"].release()
    if resources["cap2"]: resources["cap2"].release()
    if resources.get("ser_goalkeeper"): resources["ser_goalkeeper"].close()
    if resources.get("ser_attacker"): resources["ser_attacker"].close()
    if bg_video: bg_video.release()
    for video in resources["videos"].values():
        if video: video.release()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()