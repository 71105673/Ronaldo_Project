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
# *** 3-모니터 레이아웃 설정 (왼쪽 서브 | 중앙 메인 | 오른쪽 서브) *** #
# ================================================================= #
# 화면을 1/4 (왼쪽), 1/2 (중앙), 1/4 (오른쪽) 비율로 분할합니다.
sub_monitor_left_width = screen_width // 4
main_monitor_width = screen_width // 2
sub_monitor_right_width = screen_width - sub_monitor_left_width - main_monitor_width # 나머지 공간 모두 사용

# 각 모니터 영역의 시작 X 좌표
sub_left_start_x = 0
main_start_x = sub_monitor_left_width
sub_right_start_x = sub_monitor_left_width + main_monitor_width

# 각 모니터 영역의 중앙 X 좌표
sub_monitor_left_center_x = sub_left_start_x + (sub_monitor_left_width // 2)
main_monitor_center_x = main_start_x + (main_monitor_width // 2)
sub_monitor_right_center_x = sub_right_start_x + (sub_monitor_right_width // 2)


# 색상 및 폰트 등 ...
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
        "is_capturing_face": False, "face_data_buffer": [], "last_face_coords": None,
        "captured_face_filename": None
    }
    transition_surface = pygame.Surface((screen_width, screen_height)); transition_surface.fill(BLACK)
    transition_alpha, transition_target, transition_speed = 0, None, 15
    fading_out, fading_in = False, False
    resources = {
        # *** NEW: 카메라 2개 초기화 ***
        "cap": cv2.VideoCapture(1),  # 오른쪽 모니터용 카메라 (골키퍼)
        "cap2": cv2.VideoCapture(2), # 왼쪽 모니터용 카메라 (공격수)
        "ser": None, "sounds": {}, "images": {}, "videos": {},
        "last_cam_frame": None
    }
    
    # *** NEW: 두 번째 카메라 연결 확인 ***
    if not resources["cap2"].isOpened():
        print("경고: 카메라 2를 열 수 없습니다. 왼쪽 모니터는 검은색으로 표시됩니다.")
        
    try:
        resources["ser"] = serial.Serial('COM13', 9600, timeout=0)
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
        # 라운드마다 초기화되는 상태들
        game_state.update({
            "countdown_start": None, "selected_col": None, "final_col": None, "ball_col": None,
            "is_failure": False, "is_success": False, "result_time": None, "gif_start_time": None,
            "uart_ball_col": None, "waiting_for_start": False,
            "is_capturing_face": False, "face_data_buffer": [], "last_face_coords": None
        })
        # 게임을 완전히 새로 시작할 때만 초기화되는 상태들
        if full_reset:
            game_state["chances_left"], game_state["score"] = 5, 0
            game_state["captured_face_filename"] = None
            
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
        "menu": [ImageButton("../image/btn_desc.png", screen_width - 150, 150, 100, 100, lambda: start_transition("info"), sound=resources["sounds"].get("button"))],
        "game": [ImageButton("../image/btn_single.png", main_monitor_center_x - 300, screen_height//2 + 200, 550, 600, lambda: start_game("single")),
                 ImageButton("../image/btn_multi.png", main_monitor_center_x + 300, screen_height//2 + 200, 550, 600, lambda: start_game("multi")),
                 ImageButton("../image/btn_back.png", 150, 150, 100, 100, go_to_menu, sound=resources["sounds"].get("button"))],
        "face_capture": [ImageButton("../image/btn_back.png", 150, 150, 100, 100, go_to_game_select, sound=resources["sounds"].get("button"))],
        "webcam_view": [ImageButton("../image/btn_back.png", 150, 150, 100, 100, go_to_game_select, sound=resources["sounds"].get("button"))],
        "info": [ImageButton("../image/btn_exit.png", screen_width - 150, 150, 100, 100, go_to_menu, sound=resources["sounds"].get("button"))],
        "end": [ImageButton("../image/btn_restart.png", main_monitor_center_x - 300, screen_height - 250, 400, 250, go_to_game_select, sound=resources["sounds"].get("button")),
                ImageButton("../image/btn_main_menu.png", main_monitor_center_x + 300, screen_height - 250, 400, 250, go_to_menu, sound=resources["sounds"].get("button"))]
    }
    clock = pygame.time.Clock()

    # ==========================
    # 렌더링 함수 (화면 그리기)
    # ==========================

    # *** NEW: 재사용 가능한 스코어보드 그리기 함수 ***
    def draw_scoreboard(surface, start_x, center_x, width):
        # 텍스트 가독성을 위해 반투명 오버레이를 그립니다.
        overlay = pygame.Surface((width, screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        surface.blit(overlay, (start_x, 0))

        # 캡처된 얼굴 이미지 (작게 표시)
        if game_state["captured_face_filename"] and os.path.exists(game_state["captured_face_filename"]):
            try:
                face_img = pygame.image.load(game_state["captured_face_filename"])
                face_img_scaled = pygame.transform.scale(face_img, (200, 200))
                face_rect = face_img_scaled.get_rect(center=(center_x, screen_height * 0.20))
                surface.blit(face_img_scaled, face_rect)
            except Exception as e:
                print(f"스코어보드 얼굴 이미지 로딩 오류: {e}")
        
        # 점수 및 남은 기회
        score_text = score_font.render(f"SCORE: {game_state['score']}", True, WHITE)
        surface.blit(score_text, score_text.get_rect(center=(center_x, screen_height / 2)))

        chances_text = font.render("CHANCES", True, WHITE)
        surface.blit(chances_text, chances_text.get_rect(center=(center_x, screen_height * 0.65)))
        
        if resources["images"].get("scoreboard_ball"):
            total_balls_width = game_state["chances_left"] * 90 - 10
            start_ball_x = center_x - total_balls_width / 2
            for i in range(game_state["chances_left"]):
                surface.blit(resources["images"]["scoreboard_ball"], (start_ball_x + i * 90, screen_height * 0.7))

    def draw_menu_or_game_screen(state):
        # 양쪽 서브 모니터는 검은색으로 채웁니다.
        pygame.draw.rect(screen, BLACK, (sub_left_start_x, 0, sub_monitor_left_width, screen_height))
        pygame.draw.rect(screen, BLACK, (sub_right_start_x, 0, sub_monitor_right_width, screen_height))
        
        ret, frame = resources["videos"]["menu_bg"].read()
        if not ret:
            resources["videos"]["menu_bg"].set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = resources["videos"]["menu_bg"].read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 배경 비디오를 중앙 메인 모니터에 맞게 크기 조절 및 표시
            frame_resized_main = cv2.resize(frame_rgb, (main_monitor_width, screen_height))
            screen.blit(pygame.surfarray.make_surface(frame_resized_main.swapaxes(0, 1)), (main_start_x, 0))
        else:
            # 비디오 로드 실패 시 중앙도 검은색으로
            pygame.draw.rect(screen, BLACK, (main_start_x, 0, main_monitor_width, screen_height))

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

    def draw_face_capture_screen():
        screen.fill(BLACK)
        
        # *** NEW: 게임 모드에 따라 왼쪽 화면 분기 처리 ***
        if game_state["game_mode"] == "multi":
            # 멀티 모드일 때만 왼쪽 카메라(공격수) 표시
            if resources["cap2"].isOpened():
                ret_cam2, frame_cam2 = resources["cap2"].read()
                if ret_cam2:
                    frame_cam2_flipped = cv2.flip(frame_cam2, 1)
                    frame_cam2_rgb = cv2.cvtColor(frame_cam2_flipped, cv2.COLOR_BGR2RGB)
                    cam2_surf = pygame.surfarray.make_surface(frame_cam2_rgb.swapaxes(0, 1))
                    cam2_surf_scaled = pygame.transform.scale(cam2_surf, (sub_monitor_left_width, screen_height))
                    screen.blit(cam2_surf_scaled, (sub_left_start_x, 0))
        else:
            # 싱글 모드일 때는 왼쪽을 검은색으로 채움
            pygame.draw.rect(screen, BLACK, (sub_left_start_x, 0, sub_monitor_left_width, screen_height))

        # 오른쪽 모니터는 기존과 동일하게 주 카메라(cap1) 사용
        ret_cam, frame_cam = resources["cap"].read()
        if not ret_cam: return
        resources["last_cam_frame"] = frame_cam
        frame_cam_flipped = cv2.flip(frame_cam, 1)
        
        frame_cam_rgb = cv2.cvtColor(frame_cam_flipped, cv2.COLOR_BGR2RGB)
        cam_surf = pygame.surfarray.make_surface(frame_cam_rgb.swapaxes(0, 1))
        cam_surf_scaled = pygame.transform.scale(cam_surf, (sub_monitor_right_width, screen_height))
        screen.blit(cam_surf_scaled, (sub_right_start_x, 0))

        overlay = pygame.Surface((sub_monitor_right_width, screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 128))
        screen.blit(overlay, (sub_right_start_x, 0))

        title_surf = title_font.render("얼굴 캡처", True, WHITE)
        desc_surf = font.render("얼굴을 맞추고 스페이스바를 눌러 좌표를 저장하세요", True, WHITE)
        screen.blit(title_surf, title_surf.get_rect(center=(sub_monitor_right_center_x, screen_height/2 - 80)))
        screen.blit(desc_surf, desc_surf.get_rect(center=(sub_monitor_right_center_x, screen_height/2 + 40)))

        capture_area_width, capture_area_height = 400, 400
        capture_area_rect = pygame.Rect(
            sub_monitor_right_center_x - capture_area_width // 2,
            screen_height // 2 - capture_area_height // 2,
            capture_area_width,
            capture_area_height
        )
        pygame.draw.rect(screen, GRID_COLOR, capture_area_rect, 3, border_radius=15)

        if not game_state["is_capturing_face"]:
            send_uart_command(resources["ser"], 'face')
            game_state["is_capturing_face"] = True
            game_state["face_data_buffer"] = []

        if resources["ser"] and resources["ser"].in_waiting > 0:
            try:
                uart_bytes = resources["ser"].read(resources["ser"].in_waiting)
                for byte in uart_bytes:
                    payload = byte & 31
                    game_state["face_data_buffer"].append(payload)
            except Exception as e:
                print(f"UART(Face) 데이터 수신 오류: {e}")

        if len(game_state["face_data_buffer"]) >= 4:
            chunks = game_state["face_data_buffer"]
            full_data = (chunks[0] << 15) | (chunks[1] << 10) | (chunks[2] << 5) | chunks[3]
            y_coord_raw = (full_data >> 10) & 0x3FF
            x_coord_raw = full_data & 0x3FF
            
            game_state["last_face_coords"] = {
                "raw": (x_coord_raw, y_coord_raw),
                "sub_monitor_scaled": (
                    sub_right_start_x + int(x_coord_raw * (sub_monitor_right_width / 640)),
                    int(y_coord_raw * (screen_height / 480))
                )
            }
            game_state["face_data_buffer"] = game_state["face_data_buffer"][4:]

        if game_state["last_face_coords"]:
            coords_to_draw = game_state["last_face_coords"]["sub_monitor_scaled"]
            pygame.draw.circle(screen, RED, coords_to_draw, 20, 4)
        else:
            wait_surf = font.render("좌표 수신 대기 중...", True, WHITE)
            screen.blit(wait_surf, wait_surf.get_rect(center=(sub_monitor_right_center_x, screen_height - 100)))

    def draw_webcam_view():
        # 중앙 모니터: 슈팅 애니메이션 (gif)
        screen.fill(BLACK)
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
        
        # *** MODIFIED: 왼쪽 모니터에 두 번째 카메라 영상 표시 ***
        if resources["cap2"].isOpened():
            ret_cam2, frame_cam2 = resources["cap2"].read()
            if ret_cam2:
                frame_cam2_flipped = cv2.flip(frame_cam2, 1)
                frame_cam2_rgb = cv2.cvtColor(frame_cam2_flipped, cv2.COLOR_BGR2RGB)
                cam2_surf = pygame.surfarray.make_surface(frame_cam2_rgb.swapaxes(0, 1))
                cam2_surf_scaled = pygame.transform.scale(cam2_surf, (sub_monitor_left_width, screen_height))
                screen.blit(cam2_surf_scaled, (sub_left_start_x, 0))

        # 오른쪽 모니터: 주 카메라(골키퍼) 뷰 및 그리드
        ret_cam, frame_cam = resources["cap"].read()
        if ret_cam:
            frame_cam_flipped = cv2.flip(frame_cam, 1)
            frame_cam_rgb = cv2.cvtColor(frame_cam_flipped, cv2.COLOR_BGR2RGB)
            frame_cam_resized = cv2.resize(frame_cam_rgb, (sub_monitor_right_width, screen_height))
            screen.blit(pygame.surfarray.make_surface(frame_cam_resized.swapaxes(0, 1)), (sub_right_start_x, 0))

        cell_w = sub_monitor_right_width / 5
        for i in range(1, 5):
            pygame.draw.line(screen, GRID_COLOR, (sub_right_start_x + i * cell_w, 0), (sub_right_start_x + i * cell_w, screen_height), 2)

        # 오른쪽 모니터 게임 로직 (카운트다운, 선택 등)
        if game_state["waiting_for_start"]:
            overlay = pygame.Surface((sub_monitor_right_width, screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            screen.blit(overlay, (sub_right_start_x, 0))
            start_text_l1 = title_font.render("시작하시겠습니까?", True, WHITE)
            start_text_l2 = font.render("(Press Space Bar)", True, WHITE)
            screen.blit(start_text_l1, start_text_l1.get_rect(center=(sub_monitor_right_center_x, screen_height/2 - 60)))
            screen.blit(start_text_l2, start_text_l2.get_rect(center=(sub_monitor_right_center_x, screen_height/2 + 40)))
        elif game_state["countdown_start"]:
            elapsed = pygame.time.get_ticks() - game_state["countdown_start"]
            if elapsed < 5000:
                send_uart_command(resources["ser"], 'grid')
                if resources["ser"] and resources["ser"].in_waiting > 0:
                    try:
                        uart_bytes = resources["ser"].read(resources["ser"].in_waiting)
                        if uart_bytes:
                            valid_values = [b for b in uart_bytes if b in [1, 2, 3, 4, 5]]
                            if valid_values:
                                game_state["selected_col"] = 5 - valid_values[-1]
                    except Exception as e:
                        print(f"UART(Grid) 데이터 수신 오류: {e}")

                if game_state["selected_col"] is not None:
                    pygame.draw.rect(screen, GOLD_COLOR, (sub_right_start_x + game_state["selected_col"] * cell_w, 0, cell_w, screen_height), 10)

                num_str = str(5 - (elapsed // 1000))
                text_surf = countdown_font.render(num_str, True, WHITE)
                screen.blit(text_surf, text_surf.get_rect(center=(sub_monitor_right_center_x, screen_height/2)))
            else:
                if game_state["final_col"] is None:
                    send_uart_command(resources["ser"], 'stop')
                    game_state["final_col"] = game_state["selected_col"]
                    game_state["chances_left"] -= 1
                    game_state["ball_col"] = random.randint(0, 4)
                    if game_state["final_col"] == game_state["ball_col"]:
                        game_state["is_success"], game_state["score"] = True, game_state["score"] + 1
                    else:
                        game_state["is_failure"] = True
                    game_state["result_time"] = pygame.time.get_ticks()
                    game_state["countdown_start"] = None
        
        # 결과 표시 (하이라이트, 공)
        if game_state["final_col"] is not None:
            highlight_surf = pygame.Surface((cell_w, screen_height), pygame.SRCALPHA); highlight_surf.fill(HIGHLIGHT_COLOR)
            screen.blit(highlight_surf, (sub_right_start_x + game_state["final_col"] * cell_w, 0))
        if game_state["ball_col"] is not None and resources["images"]["ball"]:
            ball_rect = resources["images"]["ball"].get_rect(center=(sub_right_start_x + game_state["ball_col"] * cell_w + cell_w / 2, screen_height / 2))
            screen.blit(resources["images"]["ball"], ball_rect)

        # *** MODIFIED: 양쪽 서브 모니터에 동일한 스코어보드 그리기 ***
        draw_scoreboard(screen, sub_left_start_x, sub_monitor_left_center_x, sub_monitor_left_width)
        draw_scoreboard(screen, sub_right_start_x, sub_monitor_right_center_x, sub_monitor_right_width)


    def draw_info_screen():
        screen.blit(resources["images"]["info_bg"], (0, 0))
        pygame.draw.rect(screen, BLACK, (main_start_x, 0, main_monitor_width, screen_height))
        title_surf = title_font.render("게임 방법", True, WHITE)
        screen.blit(title_surf, title_surf.get_rect(center=(screen_width/2, 200)))
        
        text_1p = ["[1인 플레이]", "", "1. 스페이스 바를 누르면 5초 카운트 다운이 시작됩니다.", "", "2. UART 컨트롤러로 5개의 영역 중 한 곳을 선택합니다.", "", "3. 5번의 기회동안 더 많은 득점을 한 쪽이 승리합니다!"]
        text_2p = ["[2인 플레이]", "", "1. 골키퍼는 웹캠 또는 UART로 방어할 영역을 선택합니다.", "", "2. 공격수는 UART 컨트롤러로 공격할 영역을 선택합니다.", "", "3. 5번의 기회동안 더 많은 득점을 한 쪽이 승리합니다!"]
        
        y_start_1p = 450
        for i, line in enumerate(text_1p):
            text_surface = description_font.render(line, True, WHITE)
            screen.blit(text_surface, text_surface.get_rect(center=(sub_monitor_left_center_x, y_start_1p + i*75)))
            
        y_start_2p = 450
        for i, line in enumerate(text_2p):
            text_surface = description_font.render(line, True, WHITE)
            screen.blit(text_surface, text_surface.get_rect(center=(sub_monitor_right_center_x, y_start_2p + i*75)))

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
                screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (main_start_x, 0))

        face_img_scaled = None
        if game_state["captured_face_filename"] and os.path.exists(game_state["captured_face_filename"]):
            try:
                face_img = pygame.image.load(game_state["captured_face_filename"])
                face_img_scaled = pygame.transform.scale(face_img, (int(face_img.get_width() * 2.5), int(face_img.get_height() * 2.5)))
                face_rect = face_img_scaled.get_rect(center=(main_monitor_center_x, screen_height / 2))
                screen.blit(face_img_scaled, face_rect)
            except Exception as e:
                print(f"이미지 파일 불러오기 오류: {e}")

        if face_img_scaled:
            face_rect = face_img_scaled.get_rect(center=(main_monitor_center_x, screen_height / 2))
            rank_y_pos = face_rect.top - 100
            score_y_pos = face_rect.bottom + 80
            highscore_y_pos = score_y_pos + 80
        else:
            rank_y_pos = screen_height/2 - 150
            score_y_pos = screen_height/2
            highscore_y_pos = screen_height/2 + 80

        rank_surf = rank_font.render(game_state["final_rank"], True, GOLD_COLOR)
        screen.blit(rank_surf, rank_surf.get_rect(center=(main_monitor_center_x, rank_y_pos)))
        
        score_surf = score_font.render(f"FINAL SCORE: {game_state['score']}", True, WHITE)
        screen.blit(score_surf, score_surf.get_rect(center=(main_monitor_center_x, score_y_pos)))
        
        highscore_surf = score_font.render(f"HIGH SCORE: {game_state['highscore']}", True, GOLD_COLOR)
        screen.blit(highscore_surf, highscore_surf.get_rect(center=(main_monitor_center_x, highscore_y_pos)))

    def handle_events():
        nonlocal running
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                return
            if game_state["screen_state"] == "menu" and event.type == pygame.KEYDOWN:
                start_transition("game")
            elif game_state["screen_state"] == "face_capture":
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE and game_state["last_face_coords"]:
                    scaled_coords = game_state["last_face_coords"]["sub_monitor_scaled"]
                    capture_area_width, capture_area_height = 400, 400
                    safe_zone = pygame.Rect(
                        sub_monitor_right_center_x - capture_area_width // 2,
                        screen_height // 2 - capture_area_height // 2,
                        capture_area_width,
                        capture_area_height
                    )
                    if safe_zone.collidepoint(scaled_coords):
                        raw_coords = game_state["last_face_coords"]["raw"]
                        original_frame = resources.get("last_cam_frame")
                        if original_frame is not None:
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
                            try:
                                final_rgba = cv2.cvtColor(cropped_bgra, cv2.COLOR_BGRA2RGBA)
                                face_surface = pygame.image.frombuffer(
                                    final_rgba.tobytes(), final_rgba.shape[1::-1], "RGBA"
                                )
                                output_filename = "captured_face.png"
                                pygame.image.save(face_surface, output_filename)
                                game_state["captured_face_filename"] = output_filename
                                print(f"얼굴 캡처 성공! ({output_filename}으로 저장됨)")
                            except Exception as e:
                                print(f"이미지 저장/변환 오류: {e}")
                        send_uart_command(resources["ser"], 'stop')
                        game_state["is_capturing_face"] = False
                        start_new_round()
                        start_transition("webcam_view")
                    else:
                        print("캡처 실패: 목표가 중앙에 위치하지 않았습니다.")
            elif game_state["screen_state"] == "webcam_view" and game_state["waiting_for_start"]:
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
        elif current_screen == "face_capture":
            draw_face_capture_screen()
        elif current_screen == "webcam_view":
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
                    # 결과 GIF는 중앙 모니터에 표시합니다.
                    gif_display_size = (main_monitor_width, screen_height)
                    frame_resized = cv2.resize(gif_frame, gif_display_size, interpolation=cv2.INTER_AREA)
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    gif_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                    screen.blit(gif_surface, (main_start_x, 0))
            else:
                draw_webcam_view()
        elif current_screen == "info":
            draw_info_screen()
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

    # 종료 처리
    if resources["cap"]: resources["cap"].release()
    # *** NEW: 두 번째 카메라 리소스 해제 ***
    if resources["cap2"]: resources["cap2"].release()
    if resources["ser"]: resources["ser"].close()
    if bg_video: bg_video.release()
    for video in resources["videos"].values():
        if video: video.release()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()