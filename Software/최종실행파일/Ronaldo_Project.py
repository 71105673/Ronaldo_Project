import pygame
import sys
import cv2
import numpy as np
import random
import serial
import Photofunia 
from Button import ImageButton, MenuButton
from Config import *

def main():
    """
    메인 게임 실행 함수
    - 게임 상태 초기화, 리소스 로딩, 메인 루프 실행
    """
    
    # -------------------------------------------------------------------------
    # 1. 게임 상태 및 변수 초기화
    # -------------------------------------------------------------------------

    game_state = {
        "screen_state": "menu",         # 현재 화면 상태 (menu, game, face_capture, webcam_view, info, end, synthesizing)
        "chances_left": 5,              # 남은 기회
        "score": 0,                     # 골키퍼 점수
        "highscore": load_highscore(),  # 최고 점수 (파일에서 로드)
        "attacker_score": 0,            # 공격수 점수 (2인 플레이용)
        "final_rank": "",               # 최종 결과 랭크
        "end_video": None,              # 게임 종료 시 재생될 비디오 객체
        "countdown_start": None,        # 카운트다운 시작 시간
        "selected_col": None,           # 골키퍼가 선택한 수비 위치 (0~4)
        "final_col": None,              # 카운트다운 종료 후 확정된 수비 위치
        "ball_col": None,               # 공이 날아오는 위치 (0~4)
        "is_failure": False,            # 실점 여부
        "is_success": False,            # 선방 성공 여부
        "attacker_did_not_kick": False, # 공격수가 킥을 안했는지 여부 (2인 플레이용)
        "synthesized_frame_index": 0,   # 얼굴 합성 GIF의 현재 프레임 인덱스
        "synthesized_last_update": 0,   # 얼굴 합성 GIF의 마지막 프레임 업데이트 시간
        "synthesis_info": None          # 얼굴 합성에 필요한 정보 (얼굴 이미지 경로, 원본 GIF 경로 등)
    }

    # 화면 전환(페이드 인/아웃)
    transition_surface = pygame.Surface((screen_width, screen_height)); transition_surface.fill(BLACK)
    transition_alpha, transition_target, transition_speed = 0, None, 15
    fading_out, fading_in = False, False

    # 게임 리소스(카메라, 시리얼 포트, 사운드, 이미지 등)를 관리 딕셔너리
    resources = {
        "cap": cv2.VideoCapture(2),             # 골키퍼용 카메라 (웹캠)
        "cap2": cv2.VideoCapture(0),            # 공격수용 카메라 (웹캠)
        "ser_goalkeeper": None,                 # 골키퍼 아두이노와의 시리얼 통신 객체
        "ser_attacker": None,                   # 공격수 아두이노와의 시리얼 통신 객체
        "sounds": {}, "images": {}, "videos": {}, "gif_frames": {}, # 사운드, 이미지, 비디오, GIF 프레임 저장
        "last_cam_frame": None,                 # 마지막으로 캡처된 골키퍼 카메라 프레임
        "last_cam2_frame": None                 # 마지막으로 캡처된 공격수 카메라 프레임
    }

    # -------------------------------------------------------------------------
    # 2. 리소스 로딩 (카메라, 시리얼, 사운드, 이미지, 비디오/GIF)
    # -------------------------------------------------------------------------

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
        
    # 사운드 파일 로딩 (오류 발생 시 패스)
    try:
        resources["sounds"]["button"] = pygame.mixer.Sound("../sound/button_click.wav")
        resources["sounds"]["siu"] = pygame.mixer.Sound("../sound/SIUUUUU.wav")
        resources["sounds"]["success"] = pygame.mixer.Sound("../sound/야유.mp3")
        resources["sounds"]["bg_thumbnail"] = pygame.mixer.Sound("../sound/Time_Bomb.mp3")
        resources["sounds"]["failed"] = resources["sounds"]["siu"]
    except: pass

    try:
        ball_img = pygame.image.load("../image/final_ronaldo/Ball.png").convert_alpha()
        glove_img = pygame.image.load("../image/glove.png").convert_alpha()
        resources["images"]["glove"] = pygame.transform.scale(glove_img, (200, 200))
        resources["images"]["scoreboard_ball"] = pygame.transform.scale(ball_img, (80, 80))
        resources["images"]["ball"] = pygame.transform.scale(ball_img, (200, 200))
        resources["images"]["info_bg"] = pygame.transform.scale(pygame.image.load("../image/info/info_back2.jpg").convert(), (screen_width, screen_height))
    except: pass

    # 배경음악 
    if resources["sounds"].get("bg_thumbnail"):
        resources["sounds"]["bg_thumbnail"].play(-1)

    # 결과 연출용 GIF 파일들을 미리 프레임별로 잘라 리스트로 저장
    resources["gif_frames"] = {
        'success': load_gif_frames("../image/final_ronaldo/pk.gif", (main_monitor_width, screen_height)),
        'failure': load_gif_frames("../image/G.O.A.T/siuuu.gif", (main_monitor_width, screen_height)),
        'miss_kick': load_gif_frames("../image/missed_kick.gif", (main_monitor_width, screen_height)) 
    }

    # 게임 종료 및 배경 비디오 로딩
    resources["videos"]["lose"] = cv2.VideoCapture("../image/lose_goalkeeper.gif")
    resources["videos"]["victory"] = cv2.VideoCapture("../image/victory.gif")
    resources["videos"]["defeat"] = cv2.VideoCapture("../image/defeat.gif")
    resources["videos"]["game_bg"] = cv2.VideoCapture("../image/Ground1.mp4")
    resources["videos"]["menu_bg"] = cv2.VideoCapture("../image/game_thumbnail.mp4")

    # 슈팅 모션 비디오(GIF) 정보 로딩
    bg_video = cv2.VideoCapture("../image/shoot.gif")
    if bg_video.isOpened():
        bg_video_total_frames = int(bg_video.get(cv2.CAP_PROP_FRAME_COUNT))
        bg_video_w = int(bg_video.get(cv2.CAP_PROP_FRAME_WIDTH))
        bg_video_h = int(bg_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        bg_video_interval = 7000 / bg_video_total_frames if bg_video_total_frames > 0 else 0
    else:
        bg_video = None
        
    game_bg_fps = 0
    game_bg_interval = 0
    if resources["videos"]["game_bg"].isOpened():
        game_bg_fps = resources["videos"]["game_bg"].get(cv2.CAP_PROP_FPS)
        game_bg_interval = 1000 / game_bg_fps if game_bg_fps > 0 else 0
    game_bg_last_update_time = 0
    current_game_bg_surface = None

    # -------------------------------------------------------------------------
    # 3. 헬퍼(Helper) 함수 정의
    # -------------------------------------------------------------------------

    def start_transition(target_state):
        """화면 전환(페이드 아웃)을 시작하는 함수"""
        nonlocal transition_target, fading_out, fading_in
        transition_target = target_state
        fading_out = True
        fading_in = False

    def reset_game_state(full_reset=True):
        # 라운드마다 초기화되는 상태
        game_state.update({
            "countdown_start": None, "selected_col": None, "final_col": None, "ball_col": None,
            "is_failure": False, "is_success": False, "result_time": None, "gif_start_time": None,
            "gif_frame_index": 0, "attacker_did_not_kick": False,
            "waiting_for_start": False, "is_capturing_face": False,
            "attacker_selected_col": None,
            "goalkeeper_face_data_buffer": [], "last_goalkeeper_face_coords": None,
            "attacker_face_data_buffer": [], "last_attacker_face_coords": None,
            "synthesized_frames": [], "synthesized_frame_index": 0,
            "synthesis_info": None,
        })
        # 게임이 완전히 새로 시작될 때만 초기화되는 상태
        if full_reset:
            game_state.update({
                "chances_left": 5, "score": 0, "attacker_score": 0,
                "captured_goalkeeper_face_filename": None,
                "captured_attacker_face_filename": None,
            })

    def start_new_round():
        """새로운 라운드를 시작하는 함수"""
        reset_game_state(full_reset=False)
        if bg_video: bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0) # 슈팅 비디오 초기화
        game_state["waiting_for_start"] = True # 스페이스바 입력 대기 상태로 변경

    def start_game(mode):
        """게임을 시작하고 얼굴 캡처 화면으로 전환하는 함수"""
        if resources["sounds"].get("button"): resources["sounds"]["button"].play()
        game_state["game_mode"] = mode
        reset_game_state(full_reset=True)
        start_transition("face_capture")

    def go_to_menu():
        """메뉴 화면으로 돌아가는 함수"""
        reset_game_state(full_reset=True)
        start_transition("menu")

    def go_to_game_select():
        """게임 선택 화면으로 돌아가는 함수"""
        reset_game_state(full_reset=True)
        start_transition("game")

    # -------------------------------------------------------------------------
    # 4. UI 요소(버튼) 생성
    # -------------------------------------------------------------------------

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

    # -------------------------------------------------------------------------
    # 5. 화면 그리기(Draw) 함수 정의
    # -------------------------------------------------------------------------

    def draw_player_info(surface, start_x, width, player_type):
        """플레이어 정보(점수, 남은 기회)를 화면에 그리는 함수"""
        # 반투명 검은색 배경
        overlay = pygame.Surface((width, screen_height), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 100))
        surface.blit(overlay, (start_x, 0))

        # 점수 표시
        display_score = game_state['score'] if player_type == 'goalkeeper' else game_state['attacker_score']
        score_text = score_font.render(f"SCORE: {display_score}", True, WHITE)
        score_rect = score_text.get_rect(topright=(start_x + width - 20, 20))
        surface.blit(score_text, score_rect)
        
        # 남은 기회(CHANCES) 텍스트 표시
        chances_text = font.render("CHANCES", True, WHITE)
        chances_rect = chances_text.get_rect(topright=(start_x + width - 20, score_rect.bottom + 10))
        surface.blit(chances_text, chances_rect)

        # 남은 기회만큼 공 이미지 표시
        if resources["images"].get("scoreboard_ball"):
            ball_width = resources["images"]["scoreboard_ball"].get_width()
            total_balls_width = game_state["chances_left"] * (ball_width + 10) - 10
            start_ball_x = (start_x + width - 20) - total_balls_width
            for i in range(game_state["chances_left"]):
                surface.blit(resources["images"]["scoreboard_ball"], (start_ball_x + i * (ball_width + 10), chances_rect.bottom + 10))

    def draw_game_screen():
        """게임 선택 화면을 그리는 함수"""
        nonlocal game_bg_last_update_time, current_game_bg_surface
        # 양쪽 모니터 검은색으로 초기화
        pygame.draw.rect(screen, BLACK, (goalkeeper_start_x, 0, goalkeeper_monitor_width, screen_height))
        pygame.draw.rect(screen, BLACK, (attacker_start_x, 0, attacker_monitor_width, screen_height))
        
        # 메인 모니터에 배경 비디오 재생
        current_time = pygame.time.get_ticks()
        if current_time - game_bg_last_update_time > game_bg_interval:
            game_bg_last_update_time = current_time
            ret, frame = resources["videos"]["game_bg"].read()
            if not ret: # 비디오가 끝나면 처음으로 되돌리기
                resources["videos"]["game_bg"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = resources["videos"]["game_bg"].read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized_main = cv2.resize(frame_rgb, (main_monitor_width, screen_height))
                current_game_bg_surface = pygame.surfarray.make_surface(frame_resized_main.swapaxes(0, 1))
        
        if current_game_bg_surface:
            screen.blit(current_game_bg_surface, (main_start_x, 0))
        else: # 비디오 로드 실패 시 검은색 배경
            pygame.draw.rect(screen, BLACK, (main_start_x, 0, main_monitor_width, screen_height))
    
    def draw_menu_or_game_screen(state):
        """메인 메뉴 또는 게임 선택 화면을 그리는 함수"""
        # 양쪽 모니터 검은색으로 초기화
        pygame.draw.rect(screen, BLACK, (goalkeeper_start_x, 0, goalkeeper_monitor_width, screen_height))
        pygame.draw.rect(screen, BLACK, (attacker_start_x, 0, attacker_monitor_width, screen_height))
        
        # 메인 모니터에 메뉴 배경 비디오 재생
        ret, frame = resources["videos"]["menu_bg"].read()
        if not ret: # 비디오가 끝나면 처음으로 되돌리기
            resources["videos"]["menu_bg"].set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = resources["videos"]["menu_bg"].read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized_main = cv2.resize(frame_rgb, (main_monitor_width, screen_height))
            screen.blit(pygame.surfarray.make_surface(frame_resized_main.swapaxes(0, 1)), (main_start_x, 0))
        else: # 비디오 로드 실패 시 검은색 배경
            pygame.draw.rect(screen, BLACK, (main_start_x, 0, main_monitor_width, screen_height))
        
        # 'menu' 상태일 때 "PRESS ANY KEY" 텍스트 표시
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

    def draw_face_capture_screen():
        """얼굴 캡처 화면을 그리는 함수"""
        screen.fill(BLACK)
        
        # 얼굴 캡처 UI를 그리는 내부 함수
        def draw_capture_ui(surface, start_x, width, center_x, captured_filename, player_name):
            overlay = pygame.Surface((width, screen_height), pygame.SRCALPHA)
            surface.blit(overlay, (start_x, 0))
            if not captured_filename: # 캡처 전
                overlay.fill((0, 0, 0, 128))
                title_surf = title_font.render(f"{player_name} 얼굴 캡처", True, WHITE)
                desc_surf = font.render("얼굴을 중앙의 사각형에 맞춰주세요", True, WHITE)
                surface.blit(title_surf, title_surf.get_rect(center=(center_x, screen_height/2 - 80)))
                surface.blit(desc_surf, desc_surf.get_rect(center=(center_x, screen_height/2 + 40)))
                capture_area_rect = pygame.Rect(center_x - 100, screen_height // 2- 300, 150, 150)
                pygame.draw.rect(surface, GRID_COLOR, capture_area_rect, 3, border_radius=9)
            else: # 캡처 후
                overlay.fill((0, 0, 0, 200))
                captured_text = title_font.render("캡처 완료!", True, GOLD_COLOR)
                surface.blit(captured_text, captured_text.get_rect(center=(center_x, screen_height / 2)))

        # 2인 플레이 모드일 때 공격수 화면 처리
        if game_state["game_mode"] == "multi":
            if resources["cap2"].isOpened():
                ret_cam2, frame_cam2 = resources["cap2"].read()
                if ret_cam2:
                    resources["last_cam2_frame"] = frame_cam2 # 마지막 프레임 저장
                    frame_cam2_flipped = cv2.flip(frame_cam2, 1) # 좌우 반전
                    frame_cam2_rgb = cv2.cvtColor(frame_cam2_flipped, cv2.COLOR_BGR2RGB)
                    cam2_surf = pygame.surfarray.make_surface(frame_cam2_rgb.swapaxes(0, 1))
                    cam2_surf_scaled = pygame.transform.scale(cam2_surf, (attacker_monitor_width, screen_height))
                    screen.blit(cam2_surf_scaled, (attacker_start_x, 0))
            
            # 골키퍼 캡처가 끝나기 전까지 공격수 화면은 '대기 중' 표시
            if not game_state["captured_goalkeeper_face_filename"]:
                overlay = pygame.Surface((attacker_monitor_width, screen_height), pygame.SRCALPHA)
                overlay.fill((0, 0, 0, 200))
                wait_text = title_font.render("대기 중...", True, WHITE)
                overlay.blit(wait_text, wait_text.get_rect(center=(attacker_monitor_width/2, screen_height/2)))
                screen.blit(overlay, (attacker_start_x, 0))
            else: # 골키퍼 캡처 후 공격수 캡처 UI 표시
                draw_capture_ui(screen, attacker_start_x, attacker_monitor_width, attacker_monitor_center_x, game_state["captured_attacker_face_filename"], "공격수")
        else: # 1인 플레이 모드일 때 공격수 화면은 검은색
            pygame.draw.rect(screen, BLACK, (attacker_start_x, 0, attacker_monitor_width, screen_height))

        # 골키퍼 화면 처리
        ret_cam, frame_cam = resources["cap"].read()
        if ret_cam:
            resources["last_cam_frame"] = frame_cam # 마지막 프레임 저장
            frame_cam_flipped = cv2.flip(frame_cam, 1) # 좌우 반전
            frame_cam_rgb = cv2.cvtColor(frame_cam_flipped, cv2.COLOR_BGR2RGB)
            cam_surf = pygame.surfarray.make_surface(frame_cam_rgb.swapaxes(0, 1))
            cam_surf_scaled = pygame.transform.scale(cam_surf, (goalkeeper_monitor_width, screen_height))
            screen.blit(cam_surf_scaled, (goalkeeper_start_x, 0))
        draw_capture_ui(screen, goalkeeper_start_x, goalkeeper_monitor_width, goalkeeper_monitor_center_x, game_state["captured_goalkeeper_face_filename"], "골키퍼")

        # UART 통신으로 얼굴 좌표 수신 및 캡처 로직
        if not game_state["is_capturing_face"]:
            send_uart_command(resources["ser_goalkeeper"], 'face') 
            game_state["is_capturing_face"] = True

        # 골키퍼 얼굴 캡처 로직
        if not game_state["captured_goalkeeper_face_filename"]:
            if resources["ser_goalkeeper"] and resources["ser_goalkeeper"].in_waiting > 0:
                uart_bytes = resources["ser_goalkeeper"].read(resources["ser_goalkeeper"].in_waiting)
                for byte in uart_bytes:
                    header = byte >> 5 
                    if header == 2: 
                        game_state["goalkeeper_face_data_buffer"].append(byte & 31) 
            
            if len(game_state["goalkeeper_face_data_buffer"]) >= 4:
                chunks = game_state["goalkeeper_face_data_buffer"]
                # 5비트씩 4개를 합쳐 20비트 데이터 생성 -> x좌표 10비트, y좌표 10비트
                full_data = (chunks[0] << 15) | (chunks[1] << 10) | (chunks[2] << 5) | chunks[3]
                x_coord_raw, y_coord_raw = (full_data >> 10) & 0x3FF, full_data & 0x3FF
                game_state["last_goalkeeper_face_coords"] = {
                    "raw": (x_coord_raw, y_coord_raw), 
                    "scaled": (goalkeeper_start_x + (goalkeeper_monitor_width - int(x_coord_raw * (goalkeeper_monitor_width / 640))), int(y_coord_raw * (screen_height / 480)))
                }
                coords = game_state["last_goalkeeper_face_coords"]
                capture_area = pygame.Rect(goalkeeper_monitor_center_x - 100, screen_height // 2 - 300, 150, 150)
                
                # 얼굴 좌표가 캡처 영역 안에 들어오면 캡처 실행
                if capture_area.collidepoint(coords["scaled"]):
                    filename = capture_and_save_face(resources["last_cam_frame"], coords["raw"], "captured_goalkeeper_face.png")
                    if filename:
                        game_state["captured_goalkeeper_face_filename"] = filename
                        if game_state["game_mode"] == "multi":
                            send_uart_command(resources["ser_attacker"], 'face') # 공격수 보드에 좌표 요청
                        else: # 1인 플레이면 바로 게임 시작
                            game_state["is_capturing_face"] = False
                            start_new_round()
                            start_transition("webcam_view")
                game_state["goalkeeper_face_data_buffer"] = chunks[4:] # 처리된 데이터는 버퍼에서 제거

        # 공격수 얼굴 캡처 로직 (2인 플레이, 골키퍼 캡처 완료 후)
        elif game_state["game_mode"] == "multi" and not game_state["captured_attacker_face_filename"]:
            # 골키퍼와 동일한 로직으로 공격수 얼굴 좌표 수신 및 처리
            if resources["ser_attacker"] and resources["ser_attacker"].in_waiting > 0:
                uart_bytes = resources["ser_attacker"].read(resources["ser_attacker"].in_waiting)
                for byte in uart_bytes:
                    header = byte >> 5
                    if header == 2:
                        game_state["attacker_face_data_buffer"].append(byte & 31)
            
            if len(game_state["attacker_face_data_buffer"]) >= 4:
                chunks = game_state["attacker_face_data_buffer"]
                full_data = (chunks[0] << 15) | (chunks[1] << 10) | (chunks[2] << 5) | chunks[3]
                x_coord_raw, y_coord_raw = (full_data >> 10) & 0x3FF, full_data & 0x3FF
                game_state["last_attacker_face_coords"] = {"raw": (x_coord_raw, y_coord_raw),
                                                            "scaled": (attacker_start_x + (attacker_monitor_width - int(x_coord_raw * (attacker_monitor_width / 640))), int(y_coord_raw * (screen_height / 480)))}
                coords = game_state["last_attacker_face_coords"]
                capture_area = pygame.Rect(attacker_monitor_center_x - 100, screen_height // 2 - 300, 150, 150)

                if capture_area.collidepoint(coords["scaled"]):
                    filename = capture_and_save_face(resources["last_cam2_frame"], coords["raw"], "captured_attacker_face.png")
                    if filename:
                        game_state["captured_attacker_face_filename"] = filename
                        game_state["is_capturing_face"] = False
                        start_new_round() 
                        start_transition("webcam_view")
                game_state["attacker_face_data_buffer"] = chunks[4:]

        # 수신된 얼굴 좌표를 화면에 빨간 원으로 표시 (디버깅용)
        if game_state["last_goalkeeper_face_coords"]:
            coords = game_state["last_goalkeeper_face_coords"]["scaled"]
            pygame.draw.circle(screen, RED, (coords[0], coords[1]), 20, 4)
        if game_state["last_attacker_face_coords"]:
            coords = game_state["last_attacker_face_coords"]["scaled"]
            pygame.draw.circle(screen, RED, (coords[0], coords[1]), 20, 4)
            
    def draw_webcam_view():
        """메인 게임 플레이(웹캠) 화면을 그리는 함수"""
        screen.fill(BLACK)

        # (라운드 시작 전 또는 카운트다운 중) 중앙 모니터에 슈팅 비디오 재생
        if bg_video and (game_state["waiting_for_start"] or game_state["countdown_start"]):
            if game_state["waiting_for_start"]: 
                bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
            else:
                elapsed = pygame.time.get_ticks() - game_state["countdown_start"]
                current_frame_pos = int(elapsed / bg_video_interval)
                if current_frame_pos < bg_video_total_frames: 
                    bg_video.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
            
            ret_vid, frame_vid = bg_video.read()
            if ret_vid:
                new_w, new_h = get_scaled_rect(bg_video_w, bg_video_h, main_monitor_width, screen_height)
                pos_x, pos_y = main_start_x + (main_monitor_width - new_w) // 2, (screen_height - new_h) // 2
                frame_vid_rgb = cv2.cvtColor(frame_vid, cv2.COLOR_BGR2RGB)
                frame_vid_resized = cv2.resize(frame_vid_rgb, (new_w, new_h))
                screen.blit(pygame.surfarray.make_surface(frame_vid_resized.swapaxes(0, 1)), (pos_x, pos_y))

        # 골키퍼 모니터에 웹캠 영상과 그리드 표시
        ret_cam, frame_cam = resources["cap"].read()
        if ret_cam:
            frame_cam_flipped = cv2.flip(frame_cam, 1)
            frame_cam_rgb = cv2.cvtColor(frame_cam_flipped, cv2.COLOR_BGR2RGB)
            frame_cam_resized = cv2.resize(frame_cam_rgb, (goalkeeper_monitor_width, screen_height))
            screen.blit(pygame.surfarray.make_surface(frame_cam_resized.swapaxes(0, 1)), (goalkeeper_start_x, 0))
        
        cell_w_gk = goalkeeper_monitor_width / 5
        for i in range(1, 5): # 5개 영역으로 나누는 세로선
            pygame.draw.line(screen, GRID_COLOR, (goalkeeper_start_x + i * cell_w_gk, 0), (goalkeeper_start_x + i * cell_w_gk, screen_height), 2)
        draw_player_info(screen, goalkeeper_start_x, goalkeeper_monitor_width, "goalkeeper")

        # 공격수 모니터 처리 (2인 플레이 시)
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

            for i in range(1, 5): 
                pygame.draw.line(screen, GRID_COLOR, (attacker_start_x + i * cell_w_atk, 0), (attacker_start_x + i * cell_w_atk, screen_height), 2)
            
            # 공격수가 선택한 방향 표시
            if game_state["attacker_selected_col"] is not None:
                pygame.draw.rect(screen, RED, (attacker_start_x + game_state["attacker_selected_col"] * cell_w_atk, 0, cell_w_atk, screen_height), 10)
            
            # 골키퍼가 막은 위치 (장갑 이미지) 표시
            if game_state["final_col"] is not None and resources["images"].get("glove"):
                glove_rect_atk = resources["images"]["glove"].get_rect(center=(attacker_start_x + game_state["final_col"] * cell_w_atk + cell_w_atk / 2, screen_height / 2))
                screen.blit(resources["images"]["glove"], glove_rect_atk)
                
            draw_player_info(screen, attacker_start_x, attacker_monitor_width, "attacker")
        else: 
            pygame.draw.rect(screen, BLACK, (attacker_start_x, 0, attacker_monitor_width, screen_height))
        
        # 라운드 시작 전 대기 상태 UI
        if game_state["waiting_for_start"]:
            overlay = pygame.Surface((main_monitor_width, screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            screen.blit(overlay, (main_start_x, 0))
            start_text_l1 = title_font.render("시작하시겠습니까?", True, WHITE)
            start_text_l2 = font.render("(Press Space Bar)", True, WHITE)
            screen.blit(start_text_l1, start_text_l1.get_rect(center=(main_monitor_center_x, screen_height/2 - 60)))
            screen.blit(start_text_l2, start_text_l2.get_rect(center=(main_monitor_center_x, screen_height/2 + 40)))
            
        # 카운트다운 중 UI 및 로직
        elif game_state["countdown_start"]:
            elapsed = pygame.time.get_ticks() - game_state["countdown_start"]
            if elapsed < 5000: 
                # 골키퍼 입력 처리
                send_uart_command(resources["ser_goalkeeper"], 'grid')
                if resources["ser_goalkeeper"] and resources["ser_goalkeeper"].in_waiting > 0:
                    try:
                        uart_bytes = resources["ser_goalkeeper"].read(resources["ser_goalkeeper"].in_waiting)
                        for byte in uart_bytes:
                            header = byte >> 5
                            if header == 1: 
                                value = byte & 31
                                if 1 <= value <= 5:
                                    game_state["selected_col"] = 5 - value 
                    except Exception as e:
                        print(f"UART(Grid) 데이터 수신 오류: {e}")
                        
                # 공격수 입력 처리 (2인 플레이 시)
                if game_state["game_mode"] == "multi":
                    send_uart_command(resources["ser_attacker"], 'kick')
                    if resources["ser_attacker"] and resources["ser_attacker"].in_waiting > 0:
                        try:
                            uart_bytes_attacker = resources["ser_attacker"].read(resources["ser_attacker"].in_waiting)
                            for byte in uart_bytes_attacker:
                                header = byte >> 5
                                if header == 3:
                                    value = byte & 31
                                    if 1 <= value <= 5:
                                        game_state["attacker_selected_col"] = 5 - value
                        except Exception as e:
                            print(f"UART(Attacker Kick) 데이터 수신 오류: {e}")
                
                # 골키퍼가 선택한 영역 실시간 표시
                if game_state["selected_col"] is not None:
                    pygame.draw.rect(screen, GOLD_COLOR, (goalkeeper_start_x + game_state["selected_col"] * cell_w_gk, 0, cell_w_gk, screen_height), 10)
                
                # 카운트다운 숫자 표시
                num_str = str(5 - (elapsed // 1000))
                text_surf = countdown_font.render(num_str, True, WHITE)
                screen.blit(text_surf, text_surf.get_rect(center=(goalkeeper_monitor_center_x, screen_height/2)))
                if game_state["game_mode"] == "multi":
                    screen.blit(text_surf, text_surf.get_rect(center=(attacker_monitor_center_x, screen_height/2)))
            
            # 카운트다운 종료 후 결과 판정
            else:
                if game_state["final_col"] is None: 
                    game_state["final_col"] = game_state["selected_col"] 
                    game_state["chances_left"] -= 1

                    # 공격수가 킥을 선택하지 않은 경우 (2인 플레이)
                    if game_state["game_mode"] == 'multi' and game_state["attacker_selected_col"] is None:
                        game_state["attacker_did_not_kick"] = True
                        game_state["is_success"] = True 
                        game_state["is_failure"] = False
                    else: # 1인 플레이 또는 공격수가 킥을 선택한 경우
                        # 공 위치 결정 (1인플: 랜덤, 2인플: 공격수 선택)
                        game_state["ball_col"] = random.randint(0, 4) if game_state["game_mode"] == 'single' else game_state["attacker_selected_col"]
                        game_state["is_success"] = (game_state["final_col"] == game_state["ball_col"])
                        game_state["is_failure"] = not game_state["is_success"]
                        
                        if game_state["is_success"]:
                            game_state["score"] += 1 
                        elif game_state["is_failure"] and game_state["game_mode"] == "multi":
                            game_state["attacker_score"] += 1 
                            
                    game_state["result_time"] = pygame.time.get_ticks()
                    game_state["countdown_start"] = None 

        # 결과 판정 후 최종 선택 영역과 공 위치 표시
        if game_state["final_col"] is not None:
            highlight_surf = pygame.Surface((cell_w_gk, screen_height), pygame.SRCALPHA)
            highlight_surf.fill(HIGHLIGHT_COLOR)
            screen.blit(highlight_surf, (goalkeeper_start_x + game_state["final_col"] * cell_w_gk, 0))
        
        if game_state["ball_col"] is not None and resources["images"]["ball"]:
            ball_rect_gk = resources["images"]["ball"].get_rect(center=(goalkeeper_start_x + game_state["ball_col"] * cell_w_gk + cell_w_gk / 2, screen_height / 2))
            screen.blit(resources["images"]["ball"], ball_rect_gk)

    def draw_info_screen():
        """게임 설명 화면을 그리는 함수"""
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
        
        for i, line in enumerate(text_1p):
            screen.blit(description_font.render(line, True, WHITE), (main_monitor_width/4 - 550, 475 + i*75))
        for i, line in enumerate(text_2p):
            screen.blit(description_font.render(line, True, WHITE), (main_monitor_width*3/4 - 500, 475 + i*75))

    def draw_end_screen():
        """게임 종료 화면을 그리는 함수"""
        screen.fill(BLACK)
        
        # 메인 모니터에 승리/패배 비디오 재생
        if game_state["end_video"]:
            ret, frame = game_state["end_video"].read()
            if not ret: # 비디오 끝나면 처음으로
                game_state["end_video"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = game_state["end_video"].read()
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (main_monitor_width, screen_height))
                screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (main_start_x, 0))
        
        # 양쪽 모니터에 얼굴 합성 GIF 애니메이션 재생
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

        # 최종 랭크 및 점수 표시
        rank_y_pos, score_y_pos = screen_height/2 - 150, screen_height/2
        rank_surf = rank_font.render(game_state["final_rank"], True, GOLD_COLOR)
        screen.blit(rank_surf, rank_surf.get_rect(center=(main_monitor_center_x, rank_y_pos)))
        
        if game_state["game_mode"] == "multi": # 2인 플레이 점수
            score_str = f"{game_state['score']} : {game_state['attacker_score']}"
            goalkeeper_text, attacker_text = score_font.render("Goalkeeper", True, BLACK), score_font.render("Attacker", True, BLACK)
            score_surf = score_font.render(score_str, True, BLACK)
            total_width = goalkeeper_text.get_width() + score_surf.get_width() + attacker_text.get_width() + 100
            start_x = main_monitor_center_x - total_width / 2
            screen.blit(goalkeeper_text, (start_x, score_y_pos))
            screen.blit(score_surf, (start_x + goalkeeper_text.get_width() + 50, score_y_pos))
            screen.blit(attacker_text, (start_x + goalkeeper_text.get_width() + score_surf.get_width() + 100, score_y_pos))
        else: # 1인 플레이 점수
            score_surf = score_font.render(f"FINAL SCORE: {game_state['score']}", True, BLACK)
            screen.blit(score_surf, score_surf.get_rect(center=(main_monitor_center_x, score_y_pos)))
            highscore_surf = score_font.render(f"HIGH SCORE: {game_state['highscore']}", True, GOLD_COLOR)
            highscore_y_pos = score_y_pos + 80
            screen.blit(highscore_surf, highscore_surf.get_rect(center=(main_monitor_center_x, highscore_y_pos)))

    def draw_synthesizing_screen():
        """얼굴 합성 중 로딩 화면을 그리는 함수"""
        screen.fill(BLACK)
        loading_text = title_font.render("얼굴 합성 중...", True, WHITE)
        text_rect_gk = loading_text.get_rect(center=(goalkeeper_monitor_center_x, screen_height / 2))
        screen.blit(loading_text, text_rect_gk)
        if game_state["game_mode"] == "multi":
            text_rect_atk = loading_text.get_rect(center=(attacker_monitor_center_x, screen_height / 2))
            screen.blit(loading_text, text_rect_atk)
            
    def capture_and_save_face(original_frame, raw_coords, output_filename):
        """카메라 프레임과 좌표를 받아 얼굴 부분을 원형으로 잘라 저장하는 함수"""
        if original_frame is None: return None
        try:
            h, w, _ = original_frame.shape
            cx, cy, radius = raw_coords[0], raw_coords[1], 150 
            
            # 원형 마스크 생성
            bgra_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2BGRA)
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(mask, (cx, cy), radius, 255, -1)
            
            # 마스크를 알파 채널에 적용
            bgra_frame[:, :, 3] = mask
            
            # 필요한 부분만 잘라내기(cropping)
            x1, y1 = max(0, cx - radius), max(0, cy - radius)
            x2, y2 = min(w, cx + radius), min(h, cy + radius)
            cropped_bgra = bgra_frame[y1:y2, x1:x2]
            
            # Pygame Surface로 변환하여 저장
            final_rgba = cv2.cvtColor(cropped_bgra, cv2.COLOR_BGRA2RGBA)
            face_surface = pygame.image.frombuffer(final_rgba.tobytes(), final_rgba.shape[1::-1], "RGBA")
            pygame.image.save(face_surface, output_filename)
            print(f"얼굴 캡처 성공! ({output_filename}으로 저장됨)")
            return output_filename
        except Exception as e:
            print(f"이미지 저장/변환 오류: {e}")
            return None

    def handle_events():
        """Pygame 이벤트를 처리하는 함수 (키보드, 마우스 등)"""
        nonlocal running
        for event in pygame.event.get():
            # 종료 이벤트
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                return
            
            # 메뉴 화면에서 아무 키나 누르면 게임 선택 화면으로
            if game_state["screen_state"] == "menu" and event.type == pygame.KEYDOWN:
                start_transition("game")
            # 라운드 대기 중 스페이스바를 누르면 카운트다운 시작
            elif game_state["screen_state"] == "webcam_view" and game_state["waiting_for_start"]:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    game_state["countdown_start"] = pygame.time.get_ticks()
                    game_state["waiting_for_start"] = False
                    
            # 화면 전환 중이 아닐 때만 버튼 이벤트 처리
            if not (fading_in or fading_out):
                for button in buttons.get(game_state["screen_state"], []):
                    button.handle_event(event)
    
    # -------------------------------------------------------------------------
    # 6. 메인 게임 루프
    # -------------------------------------------------------------------------

    running = True
    while running:
        # 6-1. 이벤트 처리
        handle_events()
        
        # 6-2. 버튼 상태 업데이트 (마우스 호버 등)
        if not (fading_in or fading_out):
            for button in buttons.get(game_state["screen_state"], []):
                button.update()
        
        # 6-3. 현재 게임 상태에 따라 적절한 화면 그리기
        current_screen = game_state["screen_state"]
        
        if current_screen in ["menu"]:
            draw_menu_or_game_screen(current_screen)
            
        elif current_screen == "face_capture":
            draw_face_capture_screen()
            
        elif current_screen == "webcam_view":
            draw_webcam_view()
            
            # 결과(선방/실점) 연출 후 다음 라운드 또는 게임 종료 처리
            if game_state["gif_start_time"] and (pygame.time.get_ticks() - game_state["gif_start_time"] > 3000): # 3초 후
                if game_state["chances_left"] > 0:
                    start_new_round() # 기회가 남았으면 새 라운드 시작
                else: # 기회가 없으면 게임 종료
                    # 승패 판정 및 최종 랭크, 비디오 설정
                    face_path, gif_path, monitor_size = None, None, None
                    if game_state["game_mode"] == 'multi':
                        if game_state["score"] > game_state["attacker_score"]: # 골키퍼 승
                            game_state.update({"winner": "goalkeeper", "final_rank": "GOALKEEPER WINS!", "end_video": resources["videos"]["victory"]})
                            face_path = game_state["captured_goalkeeper_face_filename"]
                            gif_path = "../image/final_ronaldo/goalkeeper_win.gif"
                            monitor_size = (goalkeeper_monitor_width, screen_height)
                        elif game_state["attacker_score"] > game_state["score"]: # 공격수 승
                            game_state.update({"winner": "attacker", "final_rank": "ATTACKER WINS!", "end_video": resources["videos"]["defeat"]})
                            face_path = game_state["captured_attacker_face_filename"]
                            gif_path = "../image/final_ronaldo/attacker_win.gif"
                            monitor_size = (attacker_monitor_width, screen_height)
                        else: 
                            game_state.update({"winner": "draw", "final_rank": "DRAW", "end_video": resources["videos"]["defeat"]})
                    else: # 1인 플레이
                        game_state["winner"] = "goalkeeper"
                        if game_state["score"] > game_state["highscore"]: # 최고 기록 갱신
                            game_state["highscore"] = game_state["score"]
                            save_highscore(game_state["score"])
                        score = game_state["score"]
                        if score >= 3:
                            game_state.update({"final_rank": "Pro Keeper", "end_video": resources["videos"]["victory"]})
                            gif_path = "../image/final_ronaldo/goalkeeper_win.gif"
                        elif score >= 1:
                            game_state.update({"final_rank": "Rookie Keeper", "end_video": resources["videos"]["defeat"]})
                            gif_path = "../image/lose_goalkeeper.gif"
                        else:
                            game_state.update({"final_rank": "Human Sieve", "end_video": resources["videos"]["defeat"]})
                            gif_path = "../image/lose_goalkeeper.gif"
                        face_path = game_state["captured_goalkeeper_face_filename"]
                        monitor_size = (goalkeeper_monitor_width, screen_height)
                    
                    if face_path and gif_path and monitor_size:
                        game_state["synthesis_info"] = {"face_path": face_path, "gif_path": gif_path, "monitor_size": monitor_size}
                        start_transition("synthesizing")
                    else:
                        if game_state["end_video"]: game_state["end_video"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                        start_transition("end")

            should_play_gif = (game_state["is_failure"] or game_state["is_success"] or game_state["attacker_did_not_kick"]) and \
                              game_state["result_time"] and (pygame.time.get_ticks() - game_state["result_time"] > 2000)
            
            gif_key = None # 재생할 GIF 키
            if game_state.get("attacker_did_not_kick", False): 
                gif_key = 'miss_kick'
            elif game_state["is_failure"]: # 실점
                gif_key = 'failure'
            elif game_state["is_success"]: # 선방
                gif_key = 'success'

            if should_play_gif and gif_key:
                if not game_state["gif_start_time"]: # GIF 재생 시작
                    game_state.update({"gif_start_time": pygame.time.get_ticks(), "gif_frame_index": 0, "gif_last_frame_time": pygame.time.get_ticks()})
                    if game_state["is_success"] and resources["sounds"].get("success"): resources["sounds"]["success"].play()
                    elif game_state["is_failure"] and resources["sounds"].get("failed"): resources["sounds"]["failed"].play()

                screen.fill(BLACK) # 화면을 지우고 GIF만 표시
                frame_list = resources['gif_frames'].get(gif_key)
                if frame_list:
                    # GIF 애니메이션 처리
                    current_index = game_state['gif_frame_index']
                    frame_surface = frame_list[current_index]
                    screen.blit(frame_surface, (goalkeeper_start_x, 0))
                    if game_state["game_mode"] == "multi": screen.blit(frame_surface, (attacker_start_x, 0))
                    
                    current_time = pygame.time.get_ticks()
                    if current_time - game_state["gif_last_frame_time"] > 10: # 프레임 간격
                        game_state['gif_frame_index'] = (current_index + 1) % len(frame_list)
                        game_state["gif_last_frame_time"] = current_time
                        
        elif current_screen == "info":
            draw_info_screen()
            
        elif current_screen == "game":
            draw_game_screen()
            
        elif current_screen == "synthesizing":
            draw_synthesizing_screen() 
            if not game_state["synthesized_frames"] and game_state["synthesis_info"]:
                pygame.display.flip() # 로딩 화면을 먼저 보여주기 위해 화면 업데이트
                info = game_state["synthesis_info"]
                # Photofunia 모듈을 사용하여 얼굴 합성 GIF 프레임 생성
                game_state["synthesized_frames"] = Photofunia.create_synthesized_gif_frames(
                    info["face_path"], info["gif_path"], info["monitor_size"]
                )
                if game_state["end_video"]:
                    game_state["end_video"].set(cv2.CAP_PROP_POS_FRAMES, 0)
                start_transition("end") 
                
        elif current_screen == "end":
            draw_end_screen()
            
        # 6-4. 현재 화면에 맞는 버튼들 그리기
        for button in buttons.get(current_screen, []):
            button.draw(screen)
            
        # 6-5. 화면 전환(페이드) 효과 처리
        if fading_out or fading_in:
            if fading_out:
                transition_alpha = min(255, transition_alpha + transition_speed)
                if transition_alpha == 255:
                    fading_out, fading_in = False, True
                    game_state["screen_state"] = transition_target # 목표 화면으로 상태 변경
            else: 
                transition_alpha = max(0, transition_alpha - transition_speed)
                if transition_alpha == 0:
                    fading_in = False
            transition_surface.set_alpha(transition_alpha)
            screen.blit(transition_surface, (0, 0))
            
        pygame.display.flip()
        
        clock.tick(60)
        
    # -------------------------------------------------------------------------
    # 7. 게임 종료 시 리소스 해제
    # -------------------------------------------------------------------------
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