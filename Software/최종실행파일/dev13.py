import pygame
import sys
import cv2
import numpy as np
import random
import os
import serial  # pyserial 라이브러리 추가

from button import ImageButton

# ========================================================== #
# UART 통신 함수
# ========================================================== #
def send_uart_command(serial_port, command):
    """
    FPGA로 정해진 UART 명령어를 전송하는 함수.
    새로운 프로토콜(헤더 111)을 따릅니다.
    
    Args:
        serial_port: Pyserial의 serial.Serial 객체.
        command (str): 보낼 명령어 ('grid', 'face', 'kick', 'stop').
    """
    commands = {
        'grid': 225,  # 8'b11100001
        'face': 226,  # 8'b11100010
        'kick': 227,  # 8'b11100011
        'stop': 0     # 8'b00000000 (헤더가 111이 아니므로 중지 명령으로 인식됨)
    }
    byte_to_send = commands.get(command)
    if byte_to_send is not None and serial_port:
        try:
            serial_port.write(bytes([byte_to_send]))
        except Exception as e:
            print(f"UART({command}) 데이터 송신 오류: {e}")

# =========================================
# 초기 설정 및 상수
# =========================================
pygame.init()

try:
    desktop_sizes = pygame.display.get_desktop_sizes()
    total_width = sum(w for w, h in desktop_sizes)
    max_height = max(h for w, h in desktop_sizes)
except AttributeError:
    info = pygame.display.Info()
    total_width = info.current_w
    max_height = info.current_h

os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
screen = pygame.display.set_mode((total_width, max_height), pygame.NOFRAME)

screen_width = screen.get_width()
screen_height = screen.get_height()
pygame.display.set_caption("Penalty Kick Challenge")

if 'desktop_sizes' in locals() and len(desktop_sizes) > 1:
    main_monitor_width = desktop_sizes[0][0]
    sub_monitor_width = desktop_sizes[1][0]
else:
    main_monitor_width = screen_width // 2
    sub_monitor_width = screen_width // 2

main_monitor_center_x = main_monitor_width // 2
sub_monitor_center_x = main_monitor_width + (sub_monitor_width // 2)

# 색상 및 폰트 등 ...
BLACK, WHITE, GRID_COLOR = (0, 0, 0), (255, 255, 255), (0, 255, 0)
HIGHLIGHT_COLOR, GOLD_COLOR = (255, 0, 0, 100), (255, 215, 0)
try: pygame.mixer.init()
except: pass
def load_font(path, size, default_size):
    try: return pygame.font.Font(path, size)
    except: return pygame.font.Font(None, default_size)
font = load_font("../fonts/netmarbleM.ttf", 40, 50)
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

# ==========================
# 메인 함수
# ==========================
def main():
    game_state = {
        "screen_state": "menu", "chances_left": 5, "score": 0, "highscore": load_highscore(),
        "final_rank": "", "end_video": None, "last_end_frame": None, "countdown_start": None,
        "selected_col": None, "final_col": None, "ball_col": None, "is_failure": False,
        "is_success": False, "result_time": None, "gif_start_time": None, "uart_ball_col": None,
        "waiting_for_start": False, "game_mode": None,
        "waiting_for_face_data": False, "face_data_buffer": {}, "kicker_face_img": None
    }
    transition_surface = pygame.Surface((screen_width, screen_height)); transition_surface.fill(BLACK)
    transition_alpha, transition_target, transition_speed = 0, None, 15
    fading_out, fading_in = False, False
    resources = {
        "cap": cv2.VideoCapture(1), "ser": None, "sounds": {}, "images": {}, "videos": {}
    }
    try:
        resources["ser"] = serial.Serial('COM11', 9600, timeout=0)
        print("Basys3 보드가 성공적으로 연결되었습니다.")
    except serial.SerialException as e:
        print(f"오류: 시리얼 포트를 열 수 없습니다 - {e}")

    # 리소스 로딩 (효과음, 이미지, 비디오 등) ...
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
    except: pass
    resources["videos"]["failure_gif"] = cv2.VideoCapture("../image/G.O.A.T/siuuu.gif")
    resources["videos"]["success_gif"] = cv2.VideoCapture("../image/final_ronaldo/pk.gif")
    resources["videos"]["victory"] = cv2.VideoCapture("../image/victory.gif")
    resources["videos"]["defeat"] = cv2.VideoCapture("../image/defeat.gif")
    resources["videos"]["menu_bg"] = cv2.VideoCapture("../image/game_thumbnail.mp4")
    bg_video = cv2.VideoCapture("../image/shoot.gif")
    if bg_video.isOpened():
        bg_video_total_frames = int(bg_video.get(cv2.CAP_PROP_FRAME_COUNT))
        bg_video_w = int(bg_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        bg_video_h = int(bg_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        bg_video_interval = 7000 / bg_video_total_frames if bg_video_total_frames > 0 else 0
    else:
        bg_video = None
    
    # 트랜지션, 게임 상태 리셋 등 함수 ...
    def start_transition(target_state):
        nonlocal transition_target, fading_out
        if not fading_out and not fading_in:
            transition_target, fading_out = target_state, True
    def reset_game_state(full_reset=True):
        game_state.update({
            "countdown_start": None, "selected_col": None, "final_col": None, "ball_col": None,
            "is_failure": False, "is_success": False, "result_time": None, "gif_start_time": None,
            "uart_ball_col": None, "waiting_for_start": False,
            "waiting_for_face_data": False, "face_data_buffer": {}, "kicker_face_img": None
        })
        if full_reset:
            game_state["chances_left"], game_state["score"] = 5, 0
    def start_new_round():
        reset_game_state(full_reset=False)
        if bg_video: bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        game_state["waiting_for_start"] = True
    def start_game(mode):
        if resources["sounds"].get("button"): resources["sounds"]["button"].play()
        game_state["game_mode"] = mode
        reset_game_state(full_reset=True)
        start_new_round()
        start_transition("webcam_view")
    def go_to_menu():
        reset_game_state(full_reset=True)
        start_transition("menu")
    def go_to_game_select():
        reset_game_state(full_reset=True)
        start_transition("game")
    buttons = {
        "menu": [ImageButton("../image/btn_desc.png", screen_width - 150, 150, 100, 100, lambda: start_transition("info"), sound=resources["sounds"].get("button"))],
        "game": [ImageButton("../image/btn_single.png", main_monitor_center_x - 300, screen_height//2 + 200, 550, 600, lambda: start_game("single")),
                 ImageButton("../image/btn_multi.png", main_monitor_center_x + 300, screen_height//2 + 200, 550, 600, lambda: start_game("multi")),
                 ImageButton("../image/btn_back.png", 150, 150, 100, 100, go_to_menu, sound=resources["sounds"].get("button"))],
        "webcam_view": [ImageButton("../image/btn_back.png", 150, 150, 100, 100, go_to_game_select, sound=resources["sounds"].get("button"))],
        "info": [ImageButton("../image/btn_exit.png", screen_width - 150, 150, 100, 100, go_to_menu, sound=resources["sounds"].get("button"))],
        "end": [ImageButton("../image/btn_restart.png", main_monitor_center_x - 300, screen_height - 250, 400, 250, go_to_game_select, sound=resources["sounds"].get("button")),
                ImageButton("../image/btn_main_menu.png", main_monitor_center_x + 300, screen_height - 250, 400, 250, go_to_menu, sound=resources["sounds"].get("button"))]
    }
    clock = pygame.time.Clock()
    
    # ==========================
    # 렌더링 함수 (화면 그리기)
    # ==========================
    def draw_menu_or_game_screen(state):
        ret, frame = resources["videos"]["menu_bg"].read()
        if not ret: 
            resources["videos"]["menu_bg"].set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = resources["videos"]["menu_bg"].read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized_main = cv2.resize(frame_rgb, (main_monitor_width, screen_height))
            screen.blit(pygame.surfarray.make_surface(frame_resized_main.swapaxes(0, 1)), (0, 0))
            pygame.draw.rect(screen, BLACK, (main_monitor_width, 0, sub_monitor_width, screen_height))
        else:
            screen.fill(BLACK)
        if state == "game":
            text_surf = font.render("플레이어 수를 선택하세요", True, WHITE)
            screen.blit(text_surf, text_surf.get_rect(center=(main_monitor_center_x, screen_height//2 - 200)))
        elif state == "menu":
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

    def draw_webcam_view():
        screen.fill(BLACK)
        # 배경 비디오 처리
        if bg_video and (game_state["waiting_for_start"] or game_state["countdown_start"]):
            if game_state["waiting_for_start"]: bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else: 
                elapsed = pygame.time.get_ticks() - game_state["countdown_start"]
                current_frame_pos = int(elapsed / bg_video_interval)
                if current_frame_pos < bg_video_total_frames: bg_video.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
            ret_vid, frame_vid = bg_video.read()
            if ret_vid:
                new_w, new_h = get_scaled_rect(bg_video_w, bg_video_h, main_monitor_width, screen_height)
                pos_x, pos_y = (main_monitor_width - new_w) // 2, (screen_height - new_h) // 2
                frame_vid_rgb = cv2.cvtColor(frame_vid, cv2.COLOR_BGR2RGB)
                frame_vid_resized = cv2.resize(frame_vid_rgb, (new_w, new_h))
                screen.blit(pygame.surfarray.make_surface(frame_vid_resized.swapaxes(0, 1)), (pos_x, pos_y))
        
        # 웹캠 화면 처리
        ret_cam, frame_cam = resources["cap"].read()
        if not ret_cam: return
        frame_cam = cv2.flip(frame_cam, 1)
        frame_cam_rgb = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
        frame_cam_resized = cv2.resize(frame_cam_rgb, (sub_monitor_width, screen_height))
        screen.blit(pygame.surfarray.make_surface(frame_cam_resized.swapaxes(0, 1)), (main_monitor_width, 0))
        
        # 그리드 라인 그리기
        cell_w = sub_monitor_width / 5
        for i in range(1, 5):
            pygame.draw.line(screen, GRID_COLOR, (main_monitor_width + i * cell_w, 0), (main_monitor_width + i * cell_w, screen_height), 2)
        
        # 게임 시작 대기 화면
        if game_state["waiting_for_start"]:
            overlay = pygame.Surface((sub_monitor_width, screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            screen.blit(overlay, (main_monitor_width, 0))
            start_text_l1 = title_font.render("시작하시겠습니까?", True, WHITE)
            start_text_l2 = font.render("(Press Space Bar)", True, WHITE)
            screen.blit(start_text_l1, start_text_l1.get_rect(center=(sub_monitor_center_x, screen_height/2 - 60)))
            screen.blit(start_text_l2, start_text_l2.get_rect(center=(sub_monitor_center_x, screen_height/2 + 40)))
        
        # 카운트다운 진행
        elif game_state["countdown_start"]:
            elapsed = pygame.time.get_ticks() - game_state["countdown_start"]
            if elapsed < 5000:
                # ========================================================== #
                # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 코드 수정 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
                # ========================================================== #
                # 이제부터 모든 모드에서 UART 입력으로만 그리드 위치를 판단합니다.
                
                # 1. FPGA에 grid 데이터 요청
                send_uart_command(resources["ser"], 'grid')

                # 2. UART 포트에서 데이터 읽기
                if resources["ser"] and resources["ser"].in_waiting > 0:
                    try:
                        uart_bytes = resources["ser"].read(resources["ser"].in_waiting)
                        if uart_bytes:
                            # 수신된 데이터 중 유효한 값(1~5)을 찾음
                            valid_values = [b for b in uart_bytes if b in [1, 2, 3, 4, 5]]
                            if valid_values:
                                # 마지막으로 수신된 값을 사용 (FPGA 좌표계에 맞춰 변환)
                                game_state["selected_col"] = 5 - valid_values[-1]
                    except Exception as e:
                        print(f"UART(Grid) 데이터 수신 오류: {e}")
                
                # ========================================================== #
                # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 코드 수정 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ #
                # ========================================================== #
                
                # 공통 로직: 선택된 영역 하이라이트
                if game_state["selected_col"] is not None:
                    pygame.draw.rect(screen, GOLD_COLOR, (main_monitor_width + game_state["selected_col"] * cell_w, 0, cell_w, screen_height), 10)
                
                # 공통 로직: 카운트다운 숫자 표시
                num_str = str(5 - (elapsed // 1000))
                text_surf = countdown_font.render(num_str, True, WHITE)
                screen.blit(text_surf, text_surf.get_rect(center=(sub_monitor_center_x, screen_height/2)))
            
            # 5초 경과 후
            else:
                if game_state["final_col"] is None:
                    # ========================================================== #
                    # ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ 코드 수정 ▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼▼ #
                    # ========================================================== #
                    # 모든 모드에서 공통으로 grid 데이터 전송 중지 요청
                    send_uart_command(resources["ser"], 'stop')
                    # ========================================================== #
                    # ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ 코드 수정 ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲ #
                    # ========================================================== #

                    game_state["final_col"] = game_state["selected_col"]
                    print("\n" + "="*40)
                    print(f"라운드 {6 - game_state['chances_left']} 결과:")
                    print(f"  - 최종 방어 위치 (final_col): {game_state['final_col']}")
                    print("="*40)

                    # 멀티 모드일 때만 얼굴 좌표 요청
                    if game_state['game_mode'] == 'multi' and not game_state["waiting_for_face_data"]:
                        send_uart_command(resources["ser"], 'face')
                        game_state["waiting_for_face_data"] = True
                        game_state["face_data_buffer"] = {} # 딕셔너리로 초기화

                    game_state["chances_left"] -= 1
                    game_state["ball_col"] = random.randint(0, 4)
                    if game_state["final_col"] == game_state["ball_col"]:
                        game_state["is_success"], game_state["score"] = True, game_state["score"] + 1
                    else:
                        game_state["is_failure"] = True
                    game_state["result_time"] = pygame.time.get_ticks()
                    game_state["countdown_start"] = None
        
        # 얼굴 좌표 수신 및 처리 (딕셔너리 방식)
        if game_state["waiting_for_face_data"]:
            if resources["ser"] and resources["ser"].in_waiting > 0:
                try:
                    uart_bytes = resources["ser"].read(resources["ser"].in_waiting)
                    for byte in uart_bytes:
                        header = byte >> 5
                        payload = byte & 31
                        if header in [2, 3, 4, 5]:
                            game_state["face_data_buffer"][header] = payload
                except: pass

            if len(game_state["face_data_buffer"]) == 4:
                buffer = game_state["face_data_buffer"]
                chunks = [buffer[2], buffer[3], buffer[4], buffer[5]]
                full_data = (chunks[0] << 15) | (chunks[1] << 10) | (chunks[2] << 5) | chunks[3]
                
                img_h, img_w, _ = frame_cam.shape
                y_coord = (full_data >> 10) & 0x3FF
                x_coord = full_data & 0x3FF
                
                print("\n" + "="*40)
                print("얼굴 좌표 데이터 수신 완료:")
                print(f"  - 수신된 Raw 데이터 (20bit): {full_data} (0b{full_data:020b})")
                print(f"  - 계산된 (x, y) 좌표: ({x_coord}, {y_coord})")
                print("="*40)
                
                x = int(x_coord * (img_w / 640))
                y = int(y_coord * (img_h / 480))
                
                w, h = 150, 150 
                x1, y1 = max(0, x - w//2), max(0, y - h//2)
                x2, y2 = min(img_w, x + w//2), min(img_h, y + h//2)
                
                if x2 > x1 and y2 > y1:
                    face_crop_rgb = frame_cam_rgb[y1:y2, x1:x2]
                    game_state["kicker_face_img"] = pygame.surfarray.make_surface(face_crop_rgb.swapaxes(0, 1))
                
                game_state["waiting_for_face_data"] = False

        # 최종 선택 영역 및 공 위치 그리기
        if game_state["final_col"] is not None:
            highlight_surf = pygame.Surface((cell_w, screen_height), pygame.SRCALPHA); highlight_surf.fill(HIGHLIGHT_COLOR)
            screen.blit(highlight_surf, (main_monitor_width + game_state["final_col"] * cell_w, 0))
        if game_state["ball_col"] is not None and resources["images"]["ball"]:
            ball_rect = resources["images"]["ball"].get_rect(center=(main_monitor_width + game_state["ball_col"] * cell_w + cell_w / 2, screen_height / 2))
            screen.blit(resources["images"]["ball"], ball_rect)

        # 잘라낸 얼굴 이미지 그리기
        if game_state["kicker_face_img"]:
            face_img_scaled = pygame.transform.scale(game_state["kicker_face_img"], (200, 200))
            screen.blit(face_img_scaled, (200, 200))
            pygame.draw.rect(screen, GOLD_COLOR, (200, 200, 200, 200), 5)

    def draw_info_screen():
        screen.blit(resources["images"]["info_bg"], (0, 0))
        title_surf = title_font.render("게임 방법", True, WHITE)
        screen.blit(title_surf, (screen_width/2 - title_surf.get_width()/2, 200))
        text_1p = ["[1인 플레이]", "", "1. 스페이스 바를 누르면 5초 카운트 다운이 시작됩니다.", "", "2. UART 컨트롤러로 5개의 영역 중 한 곳을 선택합니다.", ""]
        text_2p = ["[2인 플레이]", "", "1. 골키퍼는 웹캠 또는 UART로 방어할 영역을 선택합니다.", "", "2. 공격수는 UART 컨트롤러로 공격할 영역을 선택합니다.", "", "3. 5번의 기회동안 더 많은 득점을 한 쪽이 승리합니다!"]
        y_start_1p = 450
        for i, line in enumerate(text_1p):
            text_surface = description_font.render(line, True, WHITE)
            screen.blit(text_surface, (main_monitor_center_x - text_surface.get_width() / 2, y_start_1p + i*75))
        y_start_2p = 450
        for i, line in enumerate(text_2p):
            text_surface = description_font.render(line, True, WHITE)
            screen.blit(text_surface, (sub_monitor_center_x - text_surface.get_width() / 2, y_start_2p + i*75))

    def draw_end_screen():
        screen.fill(BLACK)
        if game_state["end_video"]:
            read_new_frame = not (game_state["end_video"] == resources["videos"]["defeat"] and pygame.time.get_ticks() % 2 == 0)
            if read_new_frame:
                ret, frame = game_state["end_video"].read()
                if not ret: 
                    game_state["end_video"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = game_state["end_video"].read()
                if ret: 
                    game_state["last_end_frame"] = frame
            if game_state["last_end_frame"] is not None:
                frame_rgb = cv2.cvtColor(game_state["last_end_frame"], cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (main_monitor_width, screen_height))
                screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (0, 0))
        rank_surf = rank_font.render(game_state["final_rank"], True, GOLD_COLOR)
        screen.blit(rank_surf, rank_surf.get_rect(center=(main_monitor_center_x, screen_height/2 - 150)))
        score_surf = score_font.render(f"FINAL SCORE: {game_state['score']}", True, WHITE)
        screen.blit(score_surf, score_surf.get_rect(center=(main_monitor_center_x, screen_height/2)))
        highscore_surf = score_font.render(f"HIGH SCORE: {game_state['highscore']}", True, GOLD_COLOR)
        screen.blit(highscore_surf, highscore_surf.get_rect(center=(main_monitor_center_x, screen_height/2 + 80)))
        
    def handle_events():
        nonlocal running
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                return
            if game_state["screen_state"] == "menu" and event.type == pygame.KEYDOWN:
                start_transition("game")
            if game_state["screen_state"] == "webcam_view" and game_state["waiting_for_start"]:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    game_state["countdown_start"] = pygame.time.get_ticks()
                    game_state["waiting_for_start"] = False
            if not (fading_in or fading_out):
                for button in buttons.get(game_state["screen_state"], []):
                    button.handle_event(event)
    
    # ==========================
    # 메인 루프
    # ==========================
    running = True
    gif_frame = None
    while running:
        handle_events()
        if not (fading_in or fading_out):
            for button in buttons.get(game_state["screen_state"], []): button.update()
        current_screen = game_state["screen_state"]
        if current_screen in ["menu", "game"]:
            draw_menu_or_game_screen(current_screen)
        elif current_screen == "webcam_view":
            # 결과 GIF 처리
            if game_state["gif_start_time"] and (pygame.time.get_ticks() - game_state["gif_start_time"] > 2000):
                if game_state["chances_left"] > 0: start_new_round()
                else:
                    if game_state["score"] > game_state["highscore"]:
                        game_state["highscore"] = game_state["score"]
                        save_highscore(game_state["score"])
                    score = game_state["score"]
                    if score == 5: game_state["final_rank"], game_state["end_video"] = "THE WALL", resources["videos"]["victory"]
                    elif score >= 3: game_state["final_rank"], game_state["end_video"] = "Pro Keeper", resources["videos"]["victory"]
                    elif score >= 1: game_state["final_rank"], game_state["end_video"] = "Rookie Keeper", resources["videos"]["defeat"]
                    else: game_state["final_rank"], game_state["end_video"] = "Human Sieve", resources["videos"]["defeat"]
                    if game_state["end_video"]: game_state["end_video"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                    start_transition("end")
            should_play_gif = (game_state["is_failure"] or game_state["is_success"]) and game_state["result_time"] and (pygame.time.get_ticks() - game_state["result_time"] > 1000)
            active_gif = resources["videos"]["failure_gif"] if game_state["is_failure"] else resources["videos"]["success_gif"] if should_play_gif else None
            if active_gif and not game_state["gif_start_time"]:
                game_state["gif_start_time"] = pygame.time.get_ticks()
                if game_state["is_success"] and resources["sounds"].get("success"): resources["sounds"]["success"].play()
                elif game_state["is_failure"] and resources["sounds"].get("failed"): resources["sounds"]["failed"].play()
            
            if game_state["gif_start_time"] and active_gif:
                screen.fill(BLACK)
                if pygame.time.get_ticks() % (4 if game_state["is_failure"] else 2) == 0:
                    ret, frame = active_gif.read()
                    if not ret: active_gif.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = active_gif.read()
                    if ret: gif_frame = frame
                if gif_frame is not None:
                    gif_display_size = (sub_monitor_width, screen_height)
                    frame_resized = cv2.resize(gif_frame, gif_display_size, interpolation=cv2.INTER_AREA)
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    gif_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                    screen.blit(gif_surface, (main_monitor_width, 0))
            else:
                draw_webcam_view()

        elif current_screen == "info":
            draw_info_screen()
        elif current_screen == "end":
            draw_end_screen()

        # 점수판 및 버튼 그리기
        if current_screen in ["webcam_view"]:
            if resources["images"].get("scoreboard_ball"):
                for i in range(game_state["chances_left"]):
                    screen.blit(resources["images"]["scoreboard_ball"], (screen_width - 100 - i*90, 50))
            score_text = score_font.render(f"SCORE: {game_state['score']}", True, WHITE)
            screen.blit(score_text, (screen_width - 300, 150))
        for button in buttons.get(current_screen, []): button.draw(screen)
        
        # 화면 전환 효과
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

    # 종료 처리
    if resources["cap"]: resources["cap"].release()
    if resources["ser"]: resources["ser"].close()
    if bg_video: bg_video.release()
    for video in resources["videos"].values():
        if video: video.release()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()