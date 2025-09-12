import pygame
import sys
import cv2
import numpy as np
import random
import os
import serial  # pyserial 라이브러리 추가

from button import ImageButton

# =========================================
# 초기 설정 및 상수
# =========================================
pygame.init()
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width = screen.get_width()
screen_height = screen.get_height()
pygame.display.set_caption("Penalty Kick Challenge")

# 색상
BLACK, WHITE, GRID_COLOR = (0, 0, 0), (255, 255, 255), (0, 255, 0)
HIGHLIGHT_COLOR, GOLD_COLOR = (255, 0, 0, 100), (255, 215, 0)

# 오디오 초기화
try:
    pygame.mixer.init()
except pygame.error as e:
    print(f"오디오 초기화 오류: {e}. 소리 없이 게임을 계속합니다.")

# 폰트 로드 함수
def load_font(path, size, default_size):
    try:
        return pygame.font.Font(path, size)
    except FileNotFoundError:
        return pygame.font.Font(None, default_size)

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
    except (IOError, ValueError): return 0

def save_highscore(new_score):
    try:
        with open("highscore.txt", "w") as f: f.write(str(new_score))
    except IOError as e: print(f"최고 기록 저장 오류: {e}")

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
    # 게임 상태 관리 딕셔너리
    game_state = {
        "screen_state": "menu", "chances_left": 5, "score": 0, "highscore": load_highscore(),
        "final_rank": "", "end_video": None, "last_end_frame": None, "countdown_start": None,
        "selected_col": None, "final_col": None, "ball_col": None, "is_failure": False,
        "is_success": False, "result_time": None, "gif_start_time": None, "uart_ball_col": None,
        "waiting_for_start": False, "game_mode": None
    }

    # 화면 전환 변수
    transition_surface = pygame.Surface((screen_width, screen_height)); transition_surface.fill(BLACK)
    transition_alpha, transition_target, transition_speed = 0, None, 15
    fading_out, fading_in = False, False

    # 리소스 로드
    resources = {
        "cap": cv2.VideoCapture(1),
        "ser": None,
        "sounds": {},
        "images": {},
        "videos": {}
    }

    # 웹캠 초기화 (외부캠 1번, QVGA 해상도로 설정)
    resources["cap"] = cv2.VideoCapture(1, cv2.CAP_DSHOW) 
    if resources["cap"].isOpened():
        resources["cap"].set(cv2.CAP_PROP_FRAME_WIDTH, 320)
        resources["cap"].set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
        print("외부 카메라(1번)를 QVGA(320x240)로 열었습니다.")
    else:
        print("오류: 외부 카메라(1번)를 열 수 없습니다.")
        resources["cap"] = None

    try:
        resources["ser"] = serial.Serial('COM13', 9600, timeout=0)
        print("Basys3 보드가 성공적으로 연결되었습니다.")
    except serial.SerialException as e:
        print(f"오류: 시리얼 포트를 열 수 없습니다 - {e}")
    
    # 사운드 로드
    try:
        resources["sounds"]["button"] = pygame.mixer.Sound("../sound/button_click.wav")
        resources["sounds"]["siu"] = pygame.mixer.Sound("../sound/SIUUUUU.wav")
        resources["sounds"]["success"] = pygame.mixer.Sound("../sound/야유.mp3")
        resources["sounds"]["failed"] = resources["sounds"]["siu"]
    except pygame.error as e:
        print(f"효과음 로드 오류: {e}")

    # 이미지 로드
    try:
        ball_img = pygame.image.load("../image/final_ronaldo/Ball.png").convert_alpha()
        resources["images"]["scoreboard_ball"] = pygame.transform.scale(ball_img, (80, 80))
        resources["images"]["ball"] = pygame.transform.scale(ball_img, (200, 200))
        resources["images"]["info_bg"] = pygame.transform.scale(pygame.image.load("../image/info/info_back2.jpg").convert(), (screen_width, screen_height))
    except pygame.error as e:
        print(f"이미지 로드 오류: {e}")

    # 비디오/GIF 로드
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

    # ==========================
    # 액션 함수 (상태 변경)
    # ==========================
    def start_transition(target_state):
        nonlocal transition_target, fading_out
        if not fading_out and not fading_in:
            transition_target, fading_out = target_state, True

    def reset_game_state(full_reset=True):
        game_state.update({
            "countdown_start": None, "selected_col": None, "final_col": None, "ball_col": None,
            "is_failure": False, "is_success": False, "result_time": None, "gif_start_time": None,
            "uart_ball_col": None, "waiting_for_start": False
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

    # [수정 1] 'menu'에서 시작 버튼 제거
    buttons = {
        "menu": [ImageButton("../image/btn_desc.png", screen_width - 150, 150, 100, 100, lambda: start_transition("info"), sound=resources["sounds"].get("button"))],
        "game": [ImageButton("../image/btn_single.png", screen_width//2 - 280, screen_height//2 + 200, 550, 600, lambda: start_game("single")),
                 ImageButton("../image/btn_multi.png", screen_width//2 + 430, screen_height//2 + 200, 550, 600, lambda: start_game("multi")),
                 ImageButton("../image/btn_back.png", 150, 150, 100, 100, go_to_menu, sound=resources["sounds"].get("button"))],
        "webcam_view": [ImageButton("../image/btn_back.png", 150, 150, 100, 100, go_to_game_select, sound=resources["sounds"].get("button"))],
        "info": [ImageButton("../image/btn_exit.png", screen_width - 150, 150, 100, 100, go_to_menu, sound=resources["sounds"].get("button"))],
        "end": [ImageButton("../image/btn_restart.png", screen_width//2 - 300, screen_height - 250, 400, 250, go_to_game_select, sound=resources["sounds"].get("button")),
                ImageButton("../image/btn_main_menu.png", screen_width//2 + 300, screen_height - 250, 400, 250, go_to_menu, sound=resources["sounds"].get("button"))]
    }

    clock = pygame.time.Clock()
    
    # ==========================
    # 렌더링 함수 (화면 그리기)
    # ==========================
    def draw_menu_or_game_screen(state):
        ret, frame = resources["videos"]["menu_bg"].read()
        if not ret: resources["videos"]["menu_bg"].set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = resources["videos"]["menu_bg"].read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame_resized = cv2.resize(frame_rgb, (screen_width, screen_height))
            screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (0, 0))
        
        if state == "game":
            text_surf = font.render("플레이어 수를 선택하세요", True, WHITE)
            screen.blit(text_surf, (screen_width//2 - text_surf.get_width()//2, screen_height//2 - 200))
        
        elif state == "menu":
            # [수정] 폰트를 굵게 설정하여 렌더링
            font.set_bold(True)
            start_text_l1 = font.render("게임을 시작하려면 아무 키나 누르세요", True, WHITE)
            font.set_bold(False) # 다른 곳에 영향을 주지 않도록 원래대로 복구

            description_font.set_bold(True)
            start_text_l2 = description_font.render("PRESS ANY KEY", True, WHITE)
            description_font.set_bold(False) # 다른 곳에 영향을 주지 않도록 원래대로 복구
            
            y_pos_l1 = screen_height * 0.75 
            y_pos_l2 = y_pos_l1 + 80

            screen.blit(start_text_l1, start_text_l1.get_rect(center=(screen_width/2, y_pos_l1)))
            screen.blit(start_text_l2, start_text_l2.get_rect(center=(screen_width/2, y_pos_l2)))

    def draw_webcam_view():
        half_width = screen_width // 2
        pygame.draw.rect(screen, BLACK, (half_width, 0, half_width, screen_height))

        ret_cam, frame_cam = resources["cap"].read()
        if not ret_cam: return

        frame_cam = cv2.flip(frame_cam, 1)
        frame_cam_rgb = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
         # [수정] interpolation 옵션을 추가하여 픽셀을 복제하는 방식으로 확대합니다.
        frame_cam_resized = cv2.resize(frame_cam_rgb, (half_width, screen_height), interpolation=cv2.INTER_NEAREST)
        screen.blit(pygame.surfarray.make_surface(frame_cam_resized.swapaxes(0, 1)), (0, 0))

        cell_w = half_width / 5
        for i in range(1, 5): pygame.draw.line(screen, GRID_COLOR, (i * cell_w, 0), (i * cell_w, screen_height), 2)

        if game_state["waiting_for_start"]:
            if bg_video:
                bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret_vid, frame_vid = bg_video.read()
                if ret_vid:
                    new_w, new_h = get_scaled_rect(bg_video_w, bg_video_h, half_width, screen_height)
                    pos_x = half_width + (half_width - new_w) // 2
                    pos_y = (screen_height - new_h) // 2
                    frame_vid_rgb = cv2.cvtColor(frame_vid, cv2.COLOR_BGR2RGB)
                    frame_vid_resized = cv2.resize(frame_vid_rgb, (new_w, new_h))
                    screen.blit(pygame.surfarray.make_surface(frame_vid_resized.swapaxes(0, 1)), (pos_x, pos_y))
            
            overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
            overlay.fill((0, 0, 0, 128))
            screen.blit(overlay, (0,0))
            
            start_text_l1 = title_font.render("시작하시겠습니까?", True, WHITE)
            start_text_l2 = font.render("(Press Space Bar)", True, WHITE)
            screen.blit(start_text_l1, start_text_l1.get_rect(center=(screen_width/2, screen_height/2 - 60)))
            screen.blit(start_text_l2, start_text_l2.get_rect(center=(screen_width/2, screen_height/2 + 40)))

        elif game_state["countdown_start"]:
            elapsed = pygame.time.get_ticks() - game_state["countdown_start"]
            if elapsed < 5000:
                hsv = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2HSV)
                mask = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255])) + cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))
                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 500:
                    largest = max(contours, key=cv2.contourArea)
                    x, _, w, _ = cv2.boundingRect(largest)
                    game_state["selected_col"] = int((x + w/2) / (frame_cam.shape[1]/5))
                else:
                    game_state["selected_col"] = None

                if game_state['game_mode'] == 'multi' and resources["ser"] and resources["ser"].in_waiting > 0:
                    try:
                        uart_stream = resources["ser"].read(resources["ser"].in_waiting).decode('utf-8', errors='ignore')
                        valid_chars = [c for c in uart_stream if c in '12345']
                        if valid_chars: game_state["uart_ball_col"] = int(valid_chars[-1]) - 1
                    except Exception as e: print(f"UART 데이터 수신 오류: {e}")
                
                num_str = str(5 - (elapsed // 1000))
                text_surf = countdown_font.render(num_str, True, WHITE)
                screen.blit(text_surf, text_surf.get_rect(center=(screen_width/4, screen_height/2)))
            else:
                if game_state["final_col"] is None:
                    game_state["final_col"] = game_state["selected_col"]
                    game_state["chances_left"] -= 1
                    game_state["ball_col"] = (game_state["uart_ball_col"] if game_state["uart_ball_col"] is not None else random.randint(0, 4)) if game_state['game_mode'] == 'multi' else random.randint(0, 4)
                    if game_state["final_col"] == game_state["ball_col"]:
                        game_state["is_success"], game_state["score"] = True, game_state["score"] + 1
                    else:
                        game_state["is_failure"] = True
                    game_state["result_time"] = pygame.time.get_ticks()
                    game_state["countdown_start"] = None

        if game_state["final_col"] is not None:
            highlight_surf = pygame.Surface((cell_w, screen_height), pygame.SRCALPHA); highlight_surf.fill(HIGHLIGHT_COLOR)
            screen.blit(highlight_surf, (game_state["final_col"] * cell_w, 0))

        if game_state["ball_col"] is not None and resources["images"]["ball"]:
            ball_rect = resources["images"]["ball"].get_rect(center=(game_state["ball_col"] * cell_w + cell_w / 2, screen_height / 2))
            screen.blit(resources["images"]["ball"], ball_rect)

        if bg_video and game_state["countdown_start"] and not game_state["waiting_for_start"]:
            elapsed = pygame.time.get_ticks() - game_state["countdown_start"]
            current_frame_pos = int(elapsed / bg_video_interval)
            if current_frame_pos < bg_video_total_frames:
                bg_video.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                ret_vid, frame_vid = bg_video.read()
                if ret_vid:
                    new_w, new_h = get_scaled_rect(bg_video_w, bg_video_h, half_width, screen_height)
                    pos_x = half_width + (half_width - new_w) // 2
                    pos_y = (screen_height - new_h) // 2
                    frame_vid_rgb = cv2.cvtColor(frame_vid, cv2.COLOR_BGR2RGB)
                    frame_vid_resized = cv2.resize(frame_vid_rgb, (new_w, new_h))
                    screen.blit(pygame.surfarray.make_surface(frame_vid_resized.swapaxes(0, 1)), (pos_x, pos_y))

    def draw_info_screen():
        screen.blit(resources["images"]["info_bg"], (0, 0))
        title_surf = title_font.render("게임 방법", True, WHITE)
        screen.blit(title_surf, (screen_width/2 - title_surf.get_width()/2, 200))
        text_1p = ["[1인 플레이]", "", "1. 스페이스 바를 누르면 5초 카운트 다운이 시작됩니다.", "", "2. 5개의 영역 중 한 곳을 선택합니다.", "", "3. 5번의 기회동안 최대한 많은 공을 막으세요!"]
        text_2p = ["[2인 플레이]", "", "1. 스페이스 바를 누르면 5초 카운트 다운이 시작됩니다.", "", "2. 공격수와 골키퍼로 나뉩니다.", "", "3. 공격수는 공을 찰 방향을 정합니다.", "", "4. 골키퍼는 공을 막을 방향을 정합니다.", "", "5. 5번의 기회동안 더 많은 득점을 한 쪽이 승리합니다!"]
        for i, line in enumerate(text_1p): screen.blit(description_font.render(line, True, WHITE), (screen_width/4 - 550, 475 + i*75))
        for i, line in enumerate(text_2p): screen.blit(description_font.render(line, True, WHITE), (screen_width*3/4 - 500, 475 + i*75))

    def draw_end_screen():
        if game_state["end_video"]:
            read_new_frame = not (game_state["end_video"] == resources["videos"]["defeat"] and pygame.time.get_ticks() % 2 == 0)
            if read_new_frame:
                ret, frame = game_state["end_video"].read()
                if not ret: game_state["end_video"].set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = game_state["end_video"].read()
                if ret: game_state["last_end_frame"] = frame
            if game_state["last_end_frame"] is not None:
                frame_rgb = cv2.cvtColor(game_state["last_end_frame"], cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (screen_width, screen_height))
                screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (0, 0))
        else:
            screen.fill(BLACK)
        
        rank_surf = rank_font.render(game_state["final_rank"], True, GOLD_COLOR)
        screen.blit(rank_surf, rank_surf.get_rect(center=(screen_width/2, screen_height/2 - 150)))
        score_surf = score_font.render(f"FINAL SCORE: {game_state['score']}", True, WHITE)
        screen.blit(score_surf, score_surf.get_rect(center=(screen_width/2, screen_height/2)))
        highscore_surf = score_font.render(f"HIGH SCORE: {game_state['highscore']}", True, GOLD_COLOR)
        screen.blit(highscore_surf, highscore_surf.get_rect(center=(screen_width/2, screen_height/2 + 80)))
        
    def handle_events():
        nonlocal running
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
                return
            
            # [수정 3] menu 상태에서 키 입력 감지
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
            
            active_gif = None
            if should_play_gif:
                active_gif = resources["videos"]["failure_gif"] if game_state["is_failure"] else resources["videos"]["success_gif"]
            
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
                    gif_display_size = (screen_width, screen_height)
                    frame_resized = cv2.resize(gif_frame, gif_display_size, interpolation=cv2.INTER_AREA)
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    gif_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                    screen.blit(gif_surface, gif_surface.get_rect(center=(screen_width/2, screen_height/2)))
            else:
                draw_webcam_view()
                
        elif current_screen == "info":
            draw_info_screen()
        elif current_screen == "end":
            draw_end_screen()

        if current_screen in ["webcam_view"]:
            if resources["images"].get("scoreboard_ball"):
                for i in range(game_state["chances_left"]):
                    screen.blit(resources["images"]["scoreboard_ball"], (screen_width - 100 - i*90, 50))
            score_text = score_font.render(f"SCORE: {game_state['score']}", True, WHITE)
            screen.blit(score_text, (screen_width - 300, 150))
        
        for button in buttons.get(current_screen, []): button.draw(screen)

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

    # 리소스 해제
    if resources["cap"]: resources["cap"].release()
    if resources["ser"]: resources["ser"].close()
    if bg_video: bg_video.release()
    for video in resources["videos"].values():
        if video: video.release()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()