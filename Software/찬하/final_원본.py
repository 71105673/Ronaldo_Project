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
from config import (
    screen, screen_width, screen_height,
    main_monitor_width, main_start_x, main_monitor_center_x,
    goalkeeper_monitor_width, goalkeeper_start_x, goalkeeper_monitor_center_x,
    attacker_monitor_width, attacker_start_x, attacker_monitor_center_x,
    BLACK, WHITE, GRID_COLOR, RED, HIGHLIGHT_COLOR, GOLD_COLOR,
    font, small_font, description_font, title_font, countdown_font, score_font, rank_font,
    load_highscore, save_highscore, get_scaled_rect, load_gif_frames
)

# ===================================================================
# 2. 하드웨어 통신 함수
# ===================================================================

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

# ===================================================================
# 3. GIF 처리 및 얼굴 합성 함수
# ===================================================================

# GIF 파일을 분석하여 각 프레임과 프레임별 얼굴 위치를 반환하는 함수
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
        # Pygame에서 사용하기 위해 이미지 데이터 형식 변환
        frame_pil = Image.fromarray(frame_data).convert("RGBA")
        frames.append(pygame.image.fromstring(frame_pil.tobytes(), frame_pil.size, "RGBA"))
        
        # OpenCV에서 얼굴 인식을 위해 형식 변환
        frame_cv = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)
        
        # 성능 최적화를 위해 이미지를 작은 크기로 변환하여 얼굴 인식 수행
        scale = 0.5
        small_frame = cv2.resize(frame_cv, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(20, 20))
        
        if len(faces) > 0:
            # 인식된 얼굴 좌표를 원본 이미지 크기에 맞게 복원
            main_face = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face_locations.append((main_face / scale).astype(int))
        else:
            # 얼굴이 인식되지 않은 경우 None 추가
            face_locations.append(None)
            
    print("GIF 분석 완료!")
    return frames, face_locations

# 캡처된 얼굴 이미지를 GIF 프레임에 합성하여 Pygame Surface 리스트를 생성하는 함수
def create_synthesized_gif_frames(face_image_path, gif_path, target_size):
    if not face_image_path or not os.path.exists(face_image_path):
        print(f"합성할 얼굴 이미지 파일을 찾을 수 없습니다: {face_image_path}")
        return []

    # 1. GIF 분석 함수를 호출하여 프레임과 얼굴 위치 정보 가져오기
    gif_frames, gif_face_locations = preprocess_gif(gif_path)
    if not gif_frames:
        return []

    # 2. 합성할 얼굴 이미지를 불러오기
    try:
        overlay_face_pil = Image.open(face_image_path).convert("RGBA")
    except Exception as e:
        print(f"얼굴 이미지 로드 오류: {e}")
        return []

    synthesized_frames = []
    # 3. 각 프레임을 순회하며 얼굴 합성 작업 수행
    for i, base_frame_surface in enumerate(gif_frames):
        new_frame = base_frame_surface.copy()
        face_loc = gif_face_locations[i]

        # 해당 프레임에서 얼굴이 인식되었을 경우에만 합성
        if face_loc is not None:
            gx, gy, gw, gh = face_loc
            # 인식된 얼굴 크기에 맞춰 캡처된 얼굴 이미지 리사이즈
            resized_face_pil = overlay_face_pil.resize((gw, gh), Image.Resampling.LANCZOS)
            face_surface = pygame.image.fromstring(resized_face_pil.tobytes(), resized_face_pil.size, "RGBA")
            # 원본 프레임 위에 얼굴 이미지 합성
            new_frame.blit(face_surface, (gx, gy))
        
        # 최종적으로 화면에 표시될 크기로 프레임 조정
        scaled_frame = pygame.transform.scale(new_frame, target_size)
        synthesized_frames.append(scaled_frame)
        
    print("얼굴 합성 GIF 프레임 생성 완료!")
    return synthesized_frames


# ===================================================================
# 4. 메인 게임 함수
# ===================================================================
def main():
    # 게임의 모든 상태 변수를 저장하는 딕셔너리
    game_state = {
        "screen_state": "menu",                    # 현재 화면 상태 (e.g., "menu", "game", "end")
        "chances_left": 5,                         # 남은 기회
        "score": 0,                                # 골키퍼 점수
        "highscore": load_highscore(),             # 최고 점수
        "attacker_score": 0,                       # 공격수 점수
        "final_rank": "",                          # 최종 결과 랭크
        "end_video": None,                         # 종료 화면 배경 비디오
        "countdown_start": None,                   # 카운트다운 시작 시간
        "selected_col": None,                      # 골키퍼가 선택한 영역
        "final_col": None,                         # 최종 선택된 영역
        "ball_col": None,                          # 공이 날아갈 영역
        "is_failure": False,                       # 막기 실패 여부
        "is_success": False,                       # 막기 성공 여부
        "result_time": None,                       # 결과가 나온 시간
        "gif_start_time": None,                    # 결과 GIF 재생 시작 시간
        "gif_frame_index": 0,                      # 현재 GIF 프레임 인덱스
        "waiting_for_start": False,                # 스페이스바 입력 대기 상태
        "game_mode": None,                         # 게임 모드 ("single" or "multi")
        "is_capturing_face": False,                # 얼굴 캡처 진행 중 여부
        "captured_goalkeeper_face_filename": None, # 캡처된 골키퍼 얼굴 파일 경로
        "captured_attacker_face_filename": None,   # 캡처된 공격수 얼굴 파일 경로
        "synthesized_frames": [],                  # 합성된 GIF 프레임들을 저장하는 리스트
        "synthesized_frame_index": 0,              # 합성된 GIF의 현재 프레임 인덱스
        "synthesized_last_update": 0,              # 합성된 GIF의 마지막 업데이트 시간
        "synthesis_info": None                     # 합성에 필요한 정보(얼굴, GIF 경로) 임시 저장
    }

    # 화면 전환 효과에 사용될 Surface 및 변수
    transition_surface = pygame.Surface((screen_width, screen_height)); transition_surface.fill(BLACK)
    transition_alpha, transition_target, transition_speed = 0, None, 15
    fading_out, fading_in = False, False

    # 게임 리소스(카메라, 시리얼 포트, 사운드, 이미지 등)를 저장하는 딕셔너리
    resources = {
        "cap": cv2.VideoCapture(0),                                 # 골키퍼용 카메라
        "cap2": cv2.VideoCapture(2),                                # 공격수용 카메라
        "ser_goalkeeper": None,                                     # 골키퍼용 시리얼 포트
        "ser_attacker": None,                                       # 공격수용 시리얼 포트
        "sounds": {}, "images": {}, "videos": {}, "gif_frames": {},
        "last_cam_frame": None,                                     # 마지막으로 읽은 골키퍼 카메라 프레임
        "last_cam2_frame": None                                     # 마지막으로 읽은 공격수 카메라 프레임
    }
    
    # -------------------------------------------------------------------
    # 리소스 로딩 및 초기 설정
    # -------------------------------------------------------------------

    # 카메라 및 시리얼 포트 연결 시도 및 예외 처리
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

    # 사운드 및 이미지 파일 로딩
    try:
        resources["sounds"]["button"] = pygame.mixer.Sound("../sound/button_click.wav")
        resources["sounds"]["siu"] = pygame.mixer.Sound("../sound/SIUUUUU.wav")
        resources["sounds"]["success"] = pygame.mixer.Sound("../sound/야유.mp3")
        resources["sounds"]["bg_thumbnail"] = pygame.mixer.Sound("../sound/Time_Bomb.mp3")
        resources["sounds"]["failed"] = resources["sounds"]["siu"]
    except: pass
    try:
        ball_img = pygame.image.load("../image/final_ronaldo/Ball.png").convert_alpha()
        resources["images"]["scoreboard_ball"] = pygame.transform.scale(ball_img, (80, 80))
        resources["images"]["ball"] = pygame.transform.scale(ball_img, (200, 200))
        resources["images"]["info_bg"] = pygame.transform.scale(pygame.image.load("../image/info/info_back2.jpg").convert(), (screen_width, screen_height))
    except: pass

    if resources["sounds"].get("bg_thumbnail"):
        resources["sounds"]["bg_thumbnail"].play(-1)
    
    # 게임 결과에 따른 GIF 프레임 미리 로딩
    resources["gif_frames"] = {
        'success': load_gif_frames("../image/final_ronaldo/pk.gif", (main_monitor_width, screen_height)),
        'failure': load_gif_frames("../image/G.O.A.T/siuuu.gif", (main_monitor_width, screen_height))
    }
    
    # 배경 비디오 파일 로딩
    resources["videos"]["lose"] = cv2.VideoCapture("../image/lose_keeper.gif")
    resources["videos"]["victory"] = cv2.VideoCapture("../image/victory.gif")
    resources["videos"]["defeat"] = cv2.VideoCapture("../image/defeat.gif")
    resources["videos"]["game_bg"] = cv2.VideoCapture("../image/Ground1.mp4")
    resources["videos"]["menu_bg"] = cv2.VideoCapture("../image/game_thumbnail.mp4")
    
    # 슈팅 모션 비디오 로딩 및 정보 저장
    bg_video = cv2.VideoCapture("../image/shoot.gif")
    if bg_video.isOpened():
        bg_video_total_frames = int(bg_video.get(cv2.CAP_PROP_FRAME_COUNT))
        bg_video_w = int(bg_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        bg_video_h = int(bg_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        bg_video_interval = 7000 / bg_video_total_frames if bg_video_total_frames > 0 else 0
    else:
        bg_video = None

    # 게임 배경 비디오 FPS 정보 저장
    game_bg_fps = 0
    game_bg_interval = 0
    if resources["videos"]["game_bg"].isOpened():
        game_bg_fps = resources["videos"]["game_bg"].get(cv2.CAP_PROP_FPS)
        game_bg_interval = 1000 / game_bg_fps if game_bg_fps > 0 else 0
    game_bg_last_update_time = 0
    current_game_bg_surface = None

    # -------------------------------------------------------------------
    # 게임 상태 관리 및 화면 전환 함수
    # -------------------------------------------------------------------

    # 화면을 부드럽게 전환(페이드인/아웃)시키는 함수
    def start_transition(target_state):
        nonlocal transition_target, fading_out, fading_in
        transition_target = target_state
        fading_out = True
        fading_in = False
            
    # 게임 상태 변수들을 초기화하는 함수
    def reset_game_state(full_reset=True):
        game_state.update({
            "countdown_start": None, "selected_col": None, "final_col": None, "ball_col": None,
            "is_failure": False, "is_success": False, "result_time": None, "gif_start_time": None,
            "gif_frame_index": 0,
            "waiting_for_start": False, "is_capturing_face": False, 
            "attacker_selected_col": None,
            "goalkeeper_face_data_buffer": [], "last_goalkeeper_face_coords": None,
            "attacker_face_data_buffer": [], "last_attacker_face_coords": None,
            "synthesized_frames": [], "synthesized_frame_index": 0, 
            "synthesis_info": None,
        })
        if full_reset:
            game_state.update({
                "chances_left": 5, "score": 0, "attacker_score": 0,
                "captured_goalkeeper_face_filename": None,
                "captured_attacker_face_filename": None,
            })
            
    # 새로운 라운드를 시작하기 위해 상태를 초기화하는 함수
    def start_new_round():
        reset_game_state(full_reset=False)
        if bg_video: bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        game_state["waiting_for_start"] = True
        
    # 게임 모드를 설정하고 게임을 시작하는 함수
    def start_game(mode):
        if resources["sounds"].get("button"): resources["sounds"]["button"].play()
        game_state["game_mode"] = mode
        reset_game_state(full_reset=True)
        start_transition("face_capture")
        
    # 메인 메뉴로 돌아가는 함수
    def go_to_menu():
        reset_game_state(full_reset=True)
        start_transition("menu")
        
    # 게임 선택 화면으로 돌아가는 함수
    def go_to_game_select():
        reset_game_state(full_reset=True)
        start_transition("game")
        
    # 각 화면에 표시될 버튼들을 정의하고, 클릭 시 실행될 함수를 연결
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
    
    # Pygame의 시계 객체 생성 (FPS 제어용)
    clock = pygame.time.Clock()

    # -------------------------------------------------------------------
    # 각 화면을 그리는 함수들
    # -------------------------------------------------------------------

    # 플레이어의 점수와 남은 기회를 화면에 그리는 함수
    def draw_player_info(surface, start_x, width, player_type):
        overlay = pygame.Surface((width, screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        surface.blit(overlay, (start_x, 0))
        
        display_score = game_state['score'] if player_type == 'goalkeeper' else game_state['attacker_score']

        score_text = score_font.render(f"SCORE: {display_score}", True, WHITE)
        score_rect = score_text.get_rect(topright=(start_x + width - 20, 20))
        surface.blit(score_text, score_rect)

        chances_text = font.render("CHANCES", True, WHITE)
        chances_rect = chances_text.get_rect(topright=(start_x + width - 20, score_rect.bottom + 10))
        surface.blit(chances_text, chances_rect)
        
        if resources["images"].get("scoreboard_ball"):
            ball_width = resources["images"]["scoreboard_ball"].get_width()
            total_balls_width = game_state["chances_left"] * (ball_width + 10) - 10
            start_ball_x = (start_x + width - 20) - total_balls_width

            for i in range(game_state["chances_left"]):
                surface.blit(resources["images"]["scoreboard_ball"], (start_ball_x + i * (ball_width + 10), chances_rect.bottom + 10))

    # 게임 선택 화면을 그리는 함수
    def draw_game_screen():
        nonlocal game_bg_last_update_time, current_game_bg_surface
        pygame.draw.rect(screen, BLACK, (goalkeeper_start_x, 0, goalkeeper_monitor_width, screen_height))
        pygame.draw.rect(screen, BLACK, (attacker_start_x, 0, attacker_monitor_width, screen_height))
        current_time = pygame.time.get_ticks()

        if current_time - game_bg_last_update_time > game_bg_interval:
            game_bg_last_update_time = current_time
            ret, frame = resources["videos"]["game_bg"].read()
            if not ret:
                resources["videos"]["game_bg"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = resources["videos"]["game_bg"].read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized_main = cv2.resize(frame_rgb, (main_monitor_width, screen_height))
                current_game_bg_surface = pygame.surfarray.make_surface(frame_resized_main.swapaxes(0, 1))
        if current_game_bg_surface:
            screen.blit(current_game_bg_surface, (main_start_x, 0))
        else:
            pygame.draw.rect(screen, BLACK, (main_start_x, 0, main_monitor_width, screen_height))

    # 메인 메뉴 화면을 그리는 함수
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
            y_pos_l1, y_pos_l2 = screen_height * 0.75, screen_height * 0.75 + 80
            screen.blit(start_text_l1, start_text_l1.get_rect(center=(main_monitor_center_x, y_pos_l1)))
            screen.blit(start_text_l2, start_text_l2.get_rect(center=(main_monitor_center_x, y_pos_l2)))

    # 얼굴 캡처 화면을 그리는 함수
    def draw_face_capture_screen():
        screen.fill(BLACK)
    
        # 캡처 UI(안내 문구, 사각형 등)를 그리는 내부 함수
        def draw_capture_ui(surface, start_x, width, center_x, captured_filename, player_name):
            overlay = pygame.Surface((width, screen_height), pygame.SRCALPHA)
            surface.blit(overlay, (start_x, 0))
            if not captured_filename:
                overlay.fill((0, 0, 0, 128))
                title_surf = title_font.render(f"{player_name} 얼굴 캡처", True, WHITE)
                desc_surf = font.render("얼굴을 중앙의 사각형에 맞춰주세요", True, WHITE)
                surface.blit(title_surf, title_surf.get_rect(center=(center_x, screen_height/2 - 80)))
                surface.blit(desc_surf, desc_surf.get_rect(center=(center_x, screen_height/2 + 40)))
                capture_area_rect = pygame.Rect(center_x - 100, screen_height // 2- 350, 200, 200)
                pygame.draw.rect(surface, GRID_COLOR, capture_area_rect, 3, border_radius=15)
            else:
                overlay.fill((0, 0, 0, 200))
                captured_text = title_font.render("캡처 완료!", True, GOLD_COLOR)
                surface.blit(captured_text, captured_text.get_rect(center=(center_x, screen_height / 2)))

        # 2인 플레이 시 공격수 카메라 화면 표시
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

        # 골키퍼 카메라 화면 표시
        ret_cam, frame_cam = resources["cap"].read()
        if ret_cam:
            resources["last_cam_frame"] = frame_cam
            frame_cam_flipped = cv2.flip(frame_cam, 1)
            frame_cam_rgb = cv2.cvtColor(frame_cam_flipped, cv2.COLOR_BGR2RGB)
            cam_surf = pygame.surfarray.make_surface(frame_cam_rgb.swapaxes(0, 1))
            cam_surf_scaled = pygame.transform.scale(cam_surf, (goalkeeper_monitor_width, screen_height))
            screen.blit(cam_surf_scaled, (goalkeeper_start_x, 0))
        draw_capture_ui(screen, goalkeeper_start_x, goalkeeper_monitor_width, goalkeeper_monitor_center_x, game_state["captured_goalkeeper_face_filename"], "골키퍼")

        # 시리얼 통신을 통해 얼굴 좌표를 받고, 조건에 맞으면 얼굴 캡처
        if not game_state["is_capturing_face"]:
            send_uart_command(resources["ser_goalkeeper"], 'face')
            game_state["is_capturing_face"] = True

        if not game_state["captured_goalkeeper_face_filename"]:
            if resources["ser_goalkeeper"] and resources["ser_goalkeeper"].in_waiting > 0:
                uart_bytes = resources["ser_goalkeeper"].read(resources["ser_goalkeeper"].in_waiting)
                for byte in uart_bytes: game_state["goalkeeper_face_data_buffer"].append(byte & 31)

            if len(game_state["goalkeeper_face_data_buffer"]) >= 4:
                chunks = game_state["goalkeeper_face_data_buffer"]
                full_data = (chunks[0] << 15) | (chunks[1] << 10) | (chunks[2] << 5) | chunks[3]
                y_coord_raw, x_coord_raw = (full_data >> 10) & 0x3FF, full_data & 0x3FF
                game_state["last_goalkeeper_face_coords"] = {"raw": (x_coord_raw, y_coord_raw),"scaled": (goalkeeper_start_x + int(x_coord_raw * (goalkeeper_monitor_width / 640)), int(y_coord_raw * (screen_height / 480)))}

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

        elif game_state["game_mode"] == "multi" and not game_state["captured_attacker_face_filename"]:
            if resources["ser_attacker"] and resources["ser_attacker"].in_waiting > 0:
                uart_bytes = resources["ser_attacker"].read(resources["ser_attacker"].in_waiting)
                for byte in uart_bytes: game_state["attacker_face_data_buffer"].append(byte & 31)

            if len(game_state["attacker_face_data_buffer"]) >= 4:
                chunks = game_state["attacker_face_data_buffer"]
                full_data = (chunks[0] << 15) | (chunks[1] << 10) | (chunks[2] << 5) | chunks[3]
                y_coord_raw, x_coord_raw = (full_data >> 10) & 0x3FF, full_data & 0x3FF
                game_state["last_attacker_face_coords"] = {"raw": (x_coord_raw, y_coord_raw), "scaled": (attacker_start_x + int(x_coord_raw * (attacker_monitor_width / 640)), int(y_coord_raw * (screen_height / 480)))}

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

    # 메인 게임 플레이 화면을 그리는 함수
    def draw_webcam_view():
        screen.fill(BLACK)
        
        # 슈팅 모션 비디오 재생
        if bg_video and (game_state["waiting_for_start"] or game_state["countdown_start"]):
            if game_state["waiting_for_start"]: bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                elapsed = pygame.time.get_ticks() - game_state["countdown_start"]
                current_frame_pos = int(elapsed / bg_video_interval)
                if current_frame_pos < bg_video_total_frames: bg_video.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
            ret_vid, frame_vid = bg_video.read()
            if ret_vid:
                new_w, new_h = get_scaled_rect(bg_video_w, bg_video_h, main_monitor_width, screen_height)
                pos_x, pos_y = main_start_x + (main_monitor_width - new_w) // 2, (screen_height - new_h) // 2
                frame_vid_rgb = cv2.cvtColor(frame_vid, cv2.COLOR_BGR2RGB)
                frame_vid_resized = cv2.resize(frame_vid_rgb, (new_w, new_h))
                screen.blit(pygame.surfarray.make_surface(frame_vid_resized.swapaxes(0, 1)), (pos_x, pos_y))
        
        # 골키퍼 카메라 화면 및 UI 표시
        ret_cam, frame_cam = resources["cap"].read()
        if ret_cam:
            frame_cam_flipped = cv2.flip(frame_cam, 1)
            frame_cam_rgb = cv2.cvtColor(frame_cam_flipped, cv2.COLOR_BGR2RGB)
            frame_cam_resized = cv2.resize(frame_cam_rgb, (goalkeeper_monitor_width, screen_height))
            screen.blit(pygame.surfarray.make_surface(frame_cam_resized.swapaxes(0, 1)), (goalkeeper_start_x, 0))

        cell_w_gk = goalkeeper_monitor_width / 5
        for i in range(1, 5): pygame.draw.line(screen, GRID_COLOR, (goalkeeper_start_x + i * cell_w_gk, 0), (goalkeeper_start_x + i * cell_w_gk, screen_height), 2)
        draw_player_info(screen, goalkeeper_start_x, goalkeeper_monitor_width, "goalkeeper")

        # 2인 플레이 시 공격수 카메라 화면 및 UI 표시
        cell_w_atk = attacker_monitor_width / 5
        if game_state["game_mode"] == "multi":
            if resources["cap2"].isOpened():
                ret_cam2, frame_cam2 = resources["cap2"].read()
                if ret_cam2:
                    frame_cam2_flipped = cv2.flip(frame_cam2, 1)
                    frame_cam2_rgb = cv2.cvtColor(frame_cam2_flipped, cv2.COLOR_BGR2RGB)
                    cam2_surf = pygame.surfarray.make_surface(frame_cam2_rgb.swapaxes(0, 1))
                    cam2_surf_scaled = pygame.transform.scale(cam2_surf, (attacker_monitor_width, screen_height))
                    screen.blit(cam2_surf_scaled, (attacker_start_x, 0))
            
            for i in range(1, 5): pygame.draw.line(screen, GRID_COLOR, (attacker_start_x + i * cell_w_atk, 0), (attacker_start_x + i * cell_w_atk, screen_height), 2)
            if game_state["attacker_selected_col"] is not None: pygame.draw.rect(screen, RED, (attacker_start_x + game_state["attacker_selected_col"] * cell_w_atk, 0, cell_w_atk, screen_height), 10)
            if game_state["ball_col"] is not None and resources["images"]["ball"]:
                ball_rect_atk = resources["images"]["ball"].get_rect(center=(attacker_start_x + game_state["ball_col"] * cell_w_atk + cell_w_atk / 2, screen_height / 2))
                screen.blit(resources["images"]["ball"], ball_rect_atk)
            draw_player_info(screen, attacker_start_x, attacker_monitor_width, "attacker")
        else:
            pygame.draw.rect(screen, BLACK, (attacker_start_x, 0, attacker_monitor_width, screen_height))

        # 게임 시작 전 대기 상태 UI 표시
        if game_state["waiting_for_start"]:
            overlay = pygame.Surface((main_monitor_width, screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            screen.blit(overlay, (main_start_x, 0))
            start_text_l1, start_text_l2 = title_font.render("시작하시겠습니까?", True, WHITE), font.render("(Press Space Bar)", True, WHITE)
            screen.blit(start_text_l1, start_text_l1.get_rect(center=(main_monitor_center_x, screen_height/2 - 60)))
            screen.blit(start_text_l2, start_text_l2.get_rect(center=(main_monitor_center_x, screen_height/2 + 40)))
        
        # 카운트다운 진행 및 입력 처리
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

                if game_state["selected_col"] is not None: pygame.draw.rect(screen, GOLD_COLOR, (goalkeeper_start_x + game_state["selected_col"] * cell_w_gk, 0, cell_w_gk, screen_height), 10)
                
                num_str = str(5 - (elapsed // 1000))
                text_surf = countdown_font.render(num_str, True, WHITE)
                screen.blit(text_surf, text_surf.get_rect(center=(goalkeeper_monitor_center_x, screen_height/2)))
                if game_state["game_mode"] == "multi": screen.blit(text_surf, text_surf.get_rect(center=(attacker_monitor_center_x, screen_height/2)))
            else:
                # 카운트다운 종료 후 결과 판정
                if game_state["final_col"] is None:
                    send_uart_command(resources["ser_goalkeeper"], 'stop')
                    if game_state["game_mode"] == "multi": send_uart_command(resources["ser_attacker"], 'stop')
                    game_state["final_col"], game_state["chances_left"] = game_state["selected_col"], game_state["chances_left"] - 1
                    game_state["ball_col"] = random.randint(0, 4) if game_state["game_mode"] == 'single' else (game_state["attacker_selected_col"] if game_state["attacker_selected_col"] is not None else random.randint(0, 4))
                    game_state["is_success"] = (game_state["final_col"] == game_state["ball_col"])
                    game_state["is_failure"] = not game_state["is_success"]

                    if game_state["is_success"]: game_state["score"] += 1
                    elif game_state["is_failure"] and game_state["game_mode"] == "multi": game_state["attacker_score"] += 1
                    game_state["result_time"], game_state["countdown_start"] = pygame.time.get_ticks(), None
        
        # 최종 선택 영역 및 공 위치 표시
        if game_state["final_col"] is not None:
            highlight_surf = pygame.Surface((cell_w_gk, screen_height), pygame.SRCALPHA); highlight_surf.fill(HIGHLIGHT_COLOR)
            screen.blit(highlight_surf, (goalkeeper_start_x + game_state["final_col"] * cell_w_gk, 0))
        if game_state["ball_col"] is not None and resources["images"]["ball"]:
            ball_rect_gk = resources["images"]["ball"].get_rect(center=(goalkeeper_start_x + game_state["ball_col"] * cell_w_gk + cell_w_gk / 2, screen_height / 2))
            screen.blit(resources["images"]["ball"], ball_rect_gk)
        
    # 게임 설명 화면을 그리는 함수
    def draw_info_screen():
        pygame.draw.rect(screen, BLACK, (goalkeeper_start_x, 0, goalkeeper_monitor_width, screen_height))
        pygame.draw.rect(screen, BLACK, (attacker_start_x, 0, attacker_monitor_width, screen_height))
        if resources["images"].get("info_bg"):
            scaled_info_bg = pygame.transform.scale(resources["images"]["info_bg"], (main_monitor_width, screen_height))
            screen.blit(scaled_info_bg, (main_start_x, 0))
        else:
            pygame.draw.rect(screen, BLACK, (main_start_x, 0, main_monitor_width, screen_height))
        
        title_surf = title_font.render("게임 방법", True, WHITE)
        screen.blit(title_surf, title_surf.get_rect(center=(main_monitor_center_x, 200)))
        
        text_1p = ["[1인 플레이]", "", "1. 스페이스 바를 누르면 5초 카운트 다운이 시작됩니다.", "", "2. 5개의 영역 중 한 곳을 선택합니다.", "", "3. 5번의 기회동안 최대한 많은 공을 막으세요!"]
        text_2p = ["[2인 플레이]", "", "1. 스페이스 바를 누르면 5초 카운트 다운이 시작됩니다.", "", "2. 공격수와 골키퍼로 나뉩니다.", "", "3. 공격수는 공을 찰 방향을 정합니다.", "", "4. 골키퍼는 공을 막을 방향을 정합니다.", "", "5. 5번의 기회동안 더 많은 득점을 한 쪽이 승리합니다!"]
        for i, line in enumerate(text_1p): screen.blit(description_font.render(line, True, WHITE), (main_monitor_width/4 - 550, 475 + i*75))
        for i, line in enumerate(text_2p): screen.blit(description_font.render(line, True, WHITE), (main_monitor_width*3/4 - 500, 475 + i*75))

    # 게임 종료 화면을 그리는 함수
    def draw_end_screen():
        screen.fill(BLACK)
        
        # 1. 중앙 모니터에 승리/패배 배경 영상 재생
        if game_state["end_video"]:
            ret, frame = game_state["end_video"].read()
            if not ret:
                game_state["end_video"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = game_state["end_video"].read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (main_monitor_width, screen_height))
                screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (main_start_x, 0))
        
        # 2. 양쪽 사이드 모니터에 얼굴이 합성된 GIF 재생
        synthesized_frames = game_state.get("synthesized_frames")
        if synthesized_frames:
            current_time = pygame.time.get_ticks()
            if current_time - game_state["synthesized_last_update"] > 90:
                game_state["synthesized_frame_index"] = (game_state["synthesized_frame_index"] + 1) % len(synthesized_frames)
                game_state["synthesized_last_update"] = current_time

            current_frame_surface = synthesized_frames[game_state["synthesized_frame_index"]]
            
            screen.blit(current_frame_surface, (goalkeeper_start_x, 0))
            if game_state["game_mode"] == "multi":
                screen.blit(current_frame_surface, (attacker_start_x, 0))
        else:
            pygame.draw.rect(screen, BLACK, (goalkeeper_start_x, 0, goalkeeper_monitor_width, screen_height))
            pygame.draw.rect(screen, BLACK, (attacker_start_x, 0, attacker_monitor_width, screen_height))

        # 3. 중앙 모니터에 최종 랭크 및 점수 표시
        rank_y_pos, score_y_pos = screen_height/2 - 150, screen_height/2
        
        rank_surf = rank_font.render(game_state["final_rank"], True, GOLD_COLOR)
        screen.blit(rank_surf, rank_surf.get_rect(center=(main_monitor_center_x, rank_y_pos)))
        
        if game_state["game_mode"] == "multi":
            score_str = f"{game_state['score']} : {game_state['attacker_score']}"
            goalkeeper_text, attacker_text = score_font.render("Goalkeeper", True, BLACK), score_font.render("Attacker", True, BLACK)
            score_surf = score_font.render(score_str, True, BLACK)
            
            total_width = goalkeeper_text.get_width() + score_surf.get_width() + attacker_text.get_width() + 100
            start_x = main_monitor_center_x - total_width / 2

            screen.blit(goalkeeper_text, (start_x, score_y_pos))
            screen.blit(score_surf, (start_x + goalkeeper_text.get_width() + 50, score_y_pos))
            screen.blit(attacker_text, (start_x + goalkeeper_text.get_width() + score_surf.get_width() + 100, score_y_pos))
        else:
            score_surf = score_font.render(f"FINAL SCORE: {game_state['score']}", True, BLACK)
            screen.blit(score_surf, score_surf.get_rect(center=(main_monitor_center_x, score_y_pos)))
            
            highscore_surf = score_font.render(f"HIGH SCORE: {game_state['highscore']}", True, GOLD_COLOR)
            highscore_y_pos = score_y_pos + 80
            screen.blit(highscore_surf, highscore_surf.get_rect(center=(main_monitor_center_x, highscore_y_pos)))

    # 얼굴 합성 로딩 화면을 그리는 함수
    def draw_synthesizing_screen():
        screen.fill(BLACK)
        loading_text = title_font.render("얼굴 합성 중...", True, WHITE)
        
        text_rect_gk = loading_text.get_rect(center=(goalkeeper_monitor_center_x, screen_height / 2))
        screen.blit(loading_text, text_rect_gk)

        if game_state["game_mode"] == "multi":
            text_rect_atk = loading_text.get_rect(center=(attacker_monitor_center_x, screen_height / 2))
            screen.blit(loading_text, text_rect_atk)
    
    # 카메라 프레임에서 얼굴 부분을 캡처하여 원형으로 자른 뒤 파일로 저장하는 함수
    def capture_and_save_face(original_frame, raw_coords, output_filename):
        if original_frame is None: return None
        try:
            h, w, _ = original_frame.shape
            cx, cy, radius = raw_coords[0], raw_coords[1], 150
            
            bgra_frame, mask = cv2.cvtColor(original_frame, cv2.COLOR_BGR2BGRA), np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            bgra_frame[:, :, 3] = mask
            
            x1, y1, x2, y2 = max(0, cx - radius), max(0, cy - radius), min(w, cx + radius), min(h, cy + radius)
            cropped_bgra = bgra_frame[y1:y2, x1:x2]
            
            final_rgba = cv2.cvtColor(cropped_bgra, cv2.COLOR_BGRA2RGBA)
            face_surface = pygame.image.frombuffer(final_rgba.tobytes(), final_rgba.shape[1::-1], "RGBA")
            pygame.image.save(face_surface, output_filename)
            print(f"얼굴 캡처 성공! ({output_filename}으로 저장됨)")
            return output_filename
        except Exception as e:
            print(f"이미지 저장/변환 오류: {e}")
            return None
    
    # 키보드, 마우스 등 모든 사용자 입력을 처리하는 함수
    def handle_events():
        nonlocal running
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                return
            if game_state["screen_state"] == "menu" and event.type == pygame.KEYDOWN: start_transition("game")
            elif game_state["screen_state"] == "webcam_view" and game_state["waiting_for_start"]:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    game_state["countdown_start"], game_state["waiting_for_start"] = pygame.time.get_ticks(), False
            if not (fading_in or fading_out):
                for button in buttons.get(game_state["screen_state"], []): button.handle_event(event)

    # -------------------------------------------------------------------
    # 메인 게임 루프
    # -------------------------------------------------------------------
    running = True
    while running:
        # 1. 사용자 입력 처리
        handle_events()

        # 2. 버튼 상태 업데이트 (마우스 호버 등)
        if not (fading_in or fading_out):
            for button in buttons.get(game_state["screen_state"], []): button.update()
            
        # 3. 현재 게임 상태에 맞는 화면 그리기 및 로직 수행
        current_screen = game_state["screen_state"]
        
        if current_screen in ["menu"]:
            draw_menu_or_game_screen(current_screen)
        elif current_screen == "face_capture":
            draw_face_capture_screen()
        elif current_screen == "webcam_view":
            draw_webcam_view()

            # 결과 GIF 재생이 끝나면 다음 라운드로 넘어가거나 게임을 종료
            if game_state["gif_start_time"] and (pygame.time.get_ticks() - game_state["gif_start_time"] > 3000):
                if game_state["chances_left"] > 0:
                    start_new_round()
                else:
                    # 게임 종료: 최종 승패 판정 및 합성 정보 설정
                    face_path, gif_path, monitor_size = None, None, None
                    if game_state["game_mode"] == 'multi':
                        if game_state["score"] > game_state["attacker_score"]:
                            game_state.update({"winner": "goalkeeper", "final_rank": "GOALKEEPER WINS!", "end_video": resources["videos"]["victory"]})
                            face_path = game_state["captured_goalkeeper_face_filename"]
                            gif_path = "../image/final_ronaldo/goalkeeper_win.gif"
                            monitor_size = (goalkeeper_monitor_width, screen_height)
                        elif game_state["attacker_score"] > game_state["score"]:
                            game_state.update({"winner": "attacker", "final_rank": "ATTACKER WINS!", "end_video": resources["videos"]["defeat"]})
                            face_path = game_state["captured_attacker_face_filename"]
                            gif_path = "../image/final_ronaldo/attacker_win.gif"
                            monitor_size = (attacker_monitor_width, screen_height)
                        else:
                            game_state.update({"winner": "draw", "final_rank": "DRAW", "end_video": resources["videos"]["defeat"]})
                    else: # 1인 플레이
                        game_state["winner"] = "goalkeeper"
                        if game_state["score"] > game_state["highscore"]:
                            game_state["highscore"] = game_state["score"]
                            save_highscore(game_state["score"])
                        score = game_state["score"]
                        if score >= 3: 
                            game_state.update({"final_rank": "Pro Keeper", "end_video": resources["videos"]["victory"]})
                            gif_path = "../image/final_ronaldo/goalkeeper_win.gif"
                        elif score >= 1: 
                            game_state.update({"final_rank": "Rookie Keeper", "end_video": resources["videos"]["defeat"]})
                            gif_path = "../image/lose_keeper.gif"
                        else: 
                            game_state.update({"final_rank": "Human Sieve", "end_video": resources["videos"]["defeat"]})
                            gif_path = "../image/lose_keeper.gif"
                        
                        face_path = game_state["captured_goalkeeper_face_filename"]
                        monitor_size = (goalkeeper_monitor_width, screen_height)

                    # 합성할 정보가 있다면 '합성 중' 화면으로 전환
                    if face_path and gif_path and monitor_size:
                        game_state["synthesis_info"] = {"face_path": face_path, "gif_path": gif_path, "monitor_size": monitor_size}
                        start_transition("synthesizing")
                    else:
                        if game_state["end_video"]: game_state["end_video"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                        start_transition("end")

            # 성공/실패 판정 후 2초 뒤에 결과 GIF 재생
            should_play_gif = (game_state["is_failure"] or game_state["is_success"]) and game_state["result_time"] and (pygame.time.get_ticks() - game_state["result_time"] > 2000)
            GIF_FRAME_DURATION = 70
            gif_key = 'failure' if game_state["is_failure"] else ('success' if game_state["is_success"] else None)
            
            if should_play_gif and gif_key:
                if not game_state["gif_start_time"]:
                    game_state.update({"gif_start_time": pygame.time.get_ticks(), "gif_frame_index": 0, "gif_last_frame_time": pygame.time.get_ticks()})
                    if game_state["is_success"] and resources["sounds"].get("success"): resources["sounds"]["success"].play()
                    elif game_state["is_failure"] and resources["sounds"].get("failed"): resources["sounds"]["failed"].play()
                
                screen.fill(BLACK)
                frame_list = resources['gif_frames'].get(gif_key)
                if frame_list:
                    current_index = game_state['gif_frame_index']
                    frame_surface = frame_list[current_index]
                    screen.blit(frame_surface, (goalkeeper_start_x, 0))
                    if game_state["game_mode"] == "multi": screen.blit(frame_surface, (attacker_start_x, 0))
                    
                    current_time = pygame.time.get_ticks()
                    if current_time - game_state["gif_last_frame_time"] > GIF_FRAME_DURATION:
                        game_state['gif_frame_index'] = (current_index + 1) % len(frame_list)
                        game_state["gif_last_frame_time"] = current_time

        elif current_screen == "info":
            draw_info_screen()
        elif current_screen == "game":
            draw_game_screen()
        
        # '합성 중' 상태일 때의 로직
        elif current_screen == "synthesizing":
            draw_synthesizing_screen()
            
            # 합성이 아직 수행되지 않았다면, 단 한 번만 수행
            if not game_state["synthesized_frames"] and game_state["synthesis_info"]:
                # 중요: "합성 중" 화면을 먼저 표시한 후, 무거운 합성 작업을 시작
                pygame.display.flip()

                info = game_state["synthesis_info"]
                game_state["synthesized_frames"] = create_synthesized_gif_frames(
                    info["face_path"], info["gif_path"], info["monitor_size"]
                )
                
                # 합성이 끝나면 최종 화면으로 전환
                if game_state["end_video"]: 
                    game_state["end_video"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                start_transition("end")
        
        elif current_screen == "end":
            draw_end_screen()

        # 4. 현재 화면에 맞는 버튼들 그리기
        for button in buttons.get(current_screen, []): button.draw(screen)

        # 5. 화면 전환(페이드) 효과 그리기
        if fading_out or fading_in:
            if fading_out:
                transition_alpha = min(255, transition_alpha + transition_speed)
                if transition_alpha == 255: fading_out, fading_in, game_state["screen_state"] = False, True, transition_target
            else:
                transition_alpha = max(0, transition_alpha - transition_speed)
                if transition_alpha == 0: fading_in = False
            transition_surface.set_alpha(transition_alpha); screen.blit(transition_surface, (0, 0))

        # 6. 지금까지 그린 모든 것을 실제 화면에 업데이트
        pygame.display.flip()

        # 7. FPS를 60으로 제한
        clock.tick(60)

    # -------------------------------------------------------------------
    # 게임 종료 시 리소스 해제
    # -------------------------------------------------------------------
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