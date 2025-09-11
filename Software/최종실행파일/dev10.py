import pygame
import sys
import cv2
import numpy as np
import random
import os
import serial  # pyserial 라이브러리 추가

from button import ImageButton


pygame.init()

# 자동 전체 화면 설정
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width = screen.get_width()
screen_height = screen.get_height()

pygame.display.set_caption("Penalty Kick Challenge")

# 색상 정의
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BUTTON_COLOR = (100, 100, 100)
GRID_COLOR = (0, 255, 0)
HIGHLIGHT_COLOR = (255, 0, 0, 100)
GOLD_COLOR = (255, 215, 0)

# 오디오 초기화
try:
    pygame.mixer.init()
except pygame.error as e:
    print(f"오디오 초기화 오류: {e}. 소리 없이 게임을 계속합니다.")

# Font setup
try:
    font = pygame.font.Font("../fonts/netmarbleM.ttf", 40)
    description_font = pygame.font.Font("../fonts/netmarbleM.ttf", 50)
    title_font = pygame.font.Font("../fonts/netmarbleB.ttf", 120)
    countdown_font = pygame.font.Font("../fonts/netmarbleM.ttf", 200)
    score_font = pygame.font.Font("../fonts/netmarbleB.ttf", 60)
    rank_font = pygame.font.Font("../fonts/netmarbleB.ttf", 100)
except FileNotFoundError:
    print("폰트 파일을 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    font = pygame.font.Font(None, 50)
    description_font = pygame.font.Font(None, 60)
    title_font = pygame.font.Font(None, 130)
    countdown_font = pygame.font.Font(None, 250)
    score_font = pygame.font.Font(None, 70)
    rank_font = pygame.font.Font(None, 110)

# =========================================
# 최고 기록 관리 함수
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

# =========================================
# 화면 비율 유지 리사이즈 함수
# =========================================
def get_scaled_rect(original_w, original_h, container_w, container_h):
    aspect_ratio = original_w / original_h
    container_aspect_ratio = container_w / container_h

    if aspect_ratio > container_aspect_ratio:
        new_w = container_w
        new_h = int(new_w / aspect_ratio)
    else:
        new_h = container_h
        new_w = int(new_h * aspect_ratio)

    return new_w, new_h

# ==========================
# 메인 함수
# ==========================
def main():
    screen_state = {"current": "menu"}
    
    # 게임 상태 변수
    chances_left, score = 5, 0
    highscore = load_highscore()
    final_rank, end_video_to_play = "", None
    last_end_frame = None

    countdown_start_time, selected_grid_col, final_selected_col, ball_col = None, None, None, None
    is_failure, is_success, result_display_time, gif_start_time = False, False, None, None
    uart_ball_col = None
    waiting_for_start = False 

    # 화면 전환 변수
    transition_surface = pygame.Surface((screen_width, screen_height)); transition_surface.fill(BLACK)
    transition_alpha, transition_target, transition_speed = 0, None, 15
    fading_out, fading_in = False, False

    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("오류: 웹캠을 열 수 없습니다."); cap = None

    # 시리얼 포트 초기화
    ser = None
    try:
        ser = serial.Serial('COM11', 9600, timeout=0) 
        print("Basys3 보드가 성공적으로 연결되었습니다.")
    except serial.SerialException as e:
        print(f"오류: 시리얼 포트를 열 수 없습니다 - {e}")
        print("멀티플레이어 모드는 Basys3 보드 연결이 필요합니다.")

    # 효과음 로드
    try:
        button_sound = pygame.mixer.Sound("../sound/button_click.wav")
        siu_sound = pygame.mixer.Sound("../sound/SIUUUUU.wav")
        success_sound = pygame.mixer.Sound("../sound/야유.mp3") 
        failed_sound = siu_sound
    except pygame.error as e:
        print(f"효과음 로드 오류: {e}"); button_sound=siu_sound=success_sound=failed_sound=None
    
    # 게임 이미지 및 GIF/영상 로드
    try:
        ball_image = pygame.image.load("../image/final_ronaldo/Ball.png").convert_alpha()
        scoreboard_ball_image = pygame.transform.scale(ball_image, (80, 80)) 
        ball_image = pygame.transform.scale(ball_image, (200, 200))
    except pygame.error as e: print(f"이미지 로드 오류: Ball.png - {e}"); ball_image=scoreboard_ball_image=None

    try: failure_gif = cv2.VideoCapture("../image/G.O.A.T/siuuu.gif")
    except Exception as e: print(f"GIF 로드 오류: siuuu.gif - {e}"); failure_gif = None
    try: success_gif = cv2.VideoCapture("../image/final_ronaldo/pk.gif")
    except Exception as e: print(f"GIF 로드 오류: pk.gif - {e}"); success_gif = None
    try: victory_video = cv2.VideoCapture("../image/victory.gif")
    except Exception as e: print(f"영상 로드 오류: victory.gif - {e}"); victory_video = None
    try: defeat_video = cv2.VideoCapture("../image/defeat.gif")
    except Exception as e: print(f"영상 로드 오류: defeat.gif - {e}"); defeat_video = None
    
    bg_video = None
    bg_video_total_frames, bg_video_frame_interval = 0, 0
    bg_video_original_w, bg_video_original_h = 0, 0
    try:
        bg_video = cv2.VideoCapture("../image/shoot.gif")
        if bg_video.isOpened():
            bg_video_total_frames = int(bg_video.get(cv2.CAP_PROP_FRAME_COUNT))
            bg_video_original_w = int(bg_video.get(cv2.CAP_PROP_FRAME_WIDTH))
            bg_video_original_h = int(bg_video.get(cv2.CAP_PROP_FRAME_HEIGHT))
            if bg_video_total_frames > 0:
                bg_video_frame_interval = 7000 / bg_video_total_frames
    except Exception as e:
        print(f"배경 영상 로드 오류: {e}"); bg_video = None

    # ==========================
    # 액션 함수
    # ==========================
    def start_transition(target_state):
        nonlocal transition_target, fading_out
        if not fading_out and not fading_in: transition_target, fading_out = target_state, True

    def reset_game_state(full_reset=True):
        nonlocal countdown_start_time, selected_grid_col, final_selected_col, ball_col
        nonlocal is_failure, is_success, result_display_time, gif_start_time, uart_ball_col
        nonlocal chances_left, score, waiting_for_start
        countdown_start_time, selected_grid_col, final_selected_col, ball_col = None, None, None, None
        is_failure, is_success, result_display_time, gif_start_time = False, False, None, None
        uart_ball_col = None
        waiting_for_start = False
        if full_reset:
            chances_left, score = 5, 0
    
    def start_new_round():
        nonlocal waiting_for_start
        reset_game_state(full_reset=False)
        if bg_video: bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        waiting_for_start = True 

    def go_to_game_select():
        reset_game_state(full_reset=True)
        start_transition("game")

    def start_game(mode):
        nonlocal game_mode
        if button_sound: button_sound.play()
        game_mode["mode"] = mode
        reset_game_state(full_reset=True)
        start_new_round()
        start_transition("webcam_view")

    def go_to_menu():
        reset_game_state(full_reset=True)
        start_transition("menu")

    game_mode = {"mode": None}
    buttons = {
        "menu": [ImageButton("../image/btn_start.png", screen_width - 300, screen_height - 175, 400, 250, lambda: start_transition("game"), sound=button_sound),
                 ImageButton("../image/btn_desc.png", screen_width - 150, 150, 100, 100, lambda: start_transition("info"), sound=button_sound)],
        "game": [ImageButton("../image/btn_single.png", screen_width//2 - 280, screen_height//2 + 200, 550, 600, lambda: start_game("single")),
                 ImageButton("../image/btn_multi.png", screen_width//2 + 430, screen_height//2 + 200, 550, 600, lambda: start_game("multi")),
                 ImageButton("../image/btn_back.png", 150, 150, 100, 100, go_to_menu, sound=button_sound)],
        "webcam_view": [ImageButton("../image/btn_back.png", 150, 150, 100, 100, go_to_game_select, sound=button_sound)],
        "info": [ImageButton("../image/btn_exit.png", screen_width - 150, 150, 100, 100, go_to_menu, sound=button_sound)],
        "end": [ImageButton("../image/btn_restart.png", screen_width//2 - 300, screen_height - 250, 400, 250, go_to_game_select, sound=button_sound),
                ImageButton("../image/btn_main_menu.png", screen_width//2 + 300, screen_height - 250, 400, 250, go_to_menu, sound=button_sound)]
    }

    video = cv2.VideoCapture("../image/game_thumbnail.mp4")
    info_bg = pygame.image.load("../image/info/info_back2.jpg").convert()
    info_bg = pygame.transform.scale(info_bg, (screen_width, screen_height))
    clock = pygame.time.Clock()
    loop_counter, gif_frame = 0, None

    # ==========================
    # 메인 루프
    # ==========================
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): running = False
            
            # 스페이스바 이벤트 처리 로직
            if screen_state["current"] == "webcam_view" and waiting_for_start:
                if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    countdown_start_time = pygame.time.get_ticks()
                    waiting_for_start = False

            if not (fading_in or fading_out):
                for button in buttons.get(screen_state["current"], []): button.handle_event(event)
        
        if not (fading_in or fading_out):
            for button in buttons.get(screen_state["current"], []): button.update()

        if screen_state["current"] in ["menu", "game"]:
            ret, frame = video.read()
            if not ret: video.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame = cv2.resize(frame, (screen_width, screen_height))
                screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))

        if screen_state["current"] == "game":
            text_surf = font.render("플레이어 수를 선택하세요", True, WHITE)
            screen.blit(text_surf, (screen_width//2 - text_surf.get_width()//2, screen_height//2 - 200))
        
        elif screen_state["current"] == "webcam_view":
            if gif_start_time and (pygame.time.get_ticks() - gif_start_time > 2000):
                if chances_left > 0: start_new_round()
                else:
                    if score > highscore: highscore, _ = score, save_highscore(score)
                    if score == 5: final_rank, end_video_to_play = "THE WALL", victory_video
                    elif score >= 3: final_rank, end_video_to_play = "Pro Keeper", victory_video
                    elif score >= 1: final_rank, end_video_to_play = "Rookie Keeper", defeat_video
                    else: final_rank, end_video_to_play = "Human Sieve", defeat_video
                    if end_video_to_play: end_video_to_play.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    start_transition("end")

            should_play_gif = (is_failure or is_success) and result_display_time and (pygame.time.get_ticks() - result_display_time > 1000)
            
            active_gif = None
            if should_play_gif:
                active_gif = failure_gif if is_failure else success_gif
            
            if active_gif and not gif_start_time:
                gif_start_time = pygame.time.get_ticks()
                if is_success and success_sound: success_sound.play()
                elif is_failure and failed_sound: failed_sound.play()

            if gif_start_time and active_gif:
                screen.fill(BLACK) 
                if loop_counter % (4 if is_failure else 2) == 0:
                    ret, frame = active_gif.read()
                    if not ret: active_gif.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = active_gif.read()
                    if ret: gif_frame = frame
                
                if gif_frame is not None:
                    gif_display_size = (screen_width, screen_height) if is_failure else (screen_width // 3, screen_height // 3)
                    frame_resized = cv2.resize(gif_frame, gif_display_size, interpolation=cv2.INTER_AREA)
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    gif_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                    screen.blit(gif_surface, gif_surface.get_rect(center=(screen_width/2, screen_height/2)))
            
            elif cap:
                if game_mode['mode'] in ['single', 'multi']:
                    half_width = screen_width // 2
                    
                    pygame.draw.rect(screen, BLACK, (half_width, 0, half_width, screen_height))

                    ret_cam, frame_cam = cap.read()
                    if ret_cam:
                        frame_cam = cv2.flip(frame_cam, 1)
                        frame_cam_rgb = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2RGB)
                        frame_cam_resized = cv2.resize(frame_cam_rgb, (half_width, screen_height))
                        screen.blit(pygame.surfarray.make_surface(frame_cam_resized.swapaxes(0, 1)), (0, 0))

                        cell_w = half_width / 5
                        for i in range(1, 5): pygame.draw.line(screen, GRID_COLOR, (i * cell_w, 0), (i * cell_w, screen_height), 2)
                        
                        # [수정] 스페이스바 대기 상태와 카운트다운 로직 분리
                        if waiting_for_start:
                            # 오른쪽 화면에 shoot.gif의 첫 프레임을 고정해서 보여주기
                            if bg_video:
                                bg_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                                ret_vid, frame_vid = bg_video.read()
                                if ret_vid:
                                    new_w, new_h = get_scaled_rect(bg_video_original_w, bg_video_original_h, half_width, screen_height)
                                    pos_x = half_width + (half_width - new_w) // 2
                                    pos_y = (screen_height - new_h) // 2
                                    frame_vid_rgb = cv2.cvtColor(frame_vid, cv2.COLOR_BGR2RGB)
                                    frame_vid_resized = cv2.resize(frame_vid_rgb, (new_w, new_h))
                                    screen.blit(pygame.surfarray.make_surface(frame_vid_resized.swapaxes(0, 1)), (pos_x, pos_y))
                            
                            # 화면 중앙에 안내 문구 표시
                            overlay = pygame.Surface((screen_width, screen_height), pygame.SRCALPHA)
                            overlay.fill((0, 0, 0, 128))
                            screen.blit(overlay, (0,0))
                            
                            start_text_l1 = title_font.render("시작하시겠습니까?", True, WHITE)
                            start_text_l2 = font.render("(Press Space Bar)", True, WHITE)
                            screen.blit(start_text_l1, start_text_l1.get_rect(center=(screen_width/2, screen_height/2 - 60)))
                            screen.blit(start_text_l2, start_text_l2.get_rect(center=(screen_width/2, screen_height/2 + 40)))

                        elif countdown_start_time:
                            elapsed = pygame.time.get_ticks() - countdown_start_time
                            if elapsed < 5000:
                                hsv = cv2.cvtColor(frame_cam, cv2.COLOR_BGR2HSV)
                                mask = cv2.inRange(hsv, np.array([0,120,70]), np.array([10,255,255])) + cv2.inRange(hsv, np.array([170,120,70]), np.array([180,255,255]))
                                contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 500:
                                    largest = max(contours, key=cv2.contourArea)
                                    x, _, w, _ = cv2.boundingRect(largest)
                                    selected_grid_col = int((x + w/2) / (frame_cam.shape[1]/5))
                                else: selected_grid_col = None

                                if game_mode['mode'] == 'multi' and ser and ser.is_open and ser.in_waiting > 0:
                                    try:
                                        uart_stream = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                                        valid_chars = [c for c in uart_stream if c in '12345']
                                        if valid_chars: uart_ball_col = int(valid_chars[-1]) - 1
                                    except Exception as e: print(f"UART 데이터 수신 오류: {e}")
                                
                                num_str = str(5 - (elapsed // 1000))
                                text_surf = countdown_font.render(num_str, True, WHITE)
                                screen.blit(text_surf, text_surf.get_rect(center=(screen_width/4, screen_height/2)))
                            else:
                                if final_selected_col is None:
                                    final_selected_col, chances_left = selected_grid_col, chances_left - 1
                                    ball_col = (uart_ball_col if uart_ball_col is not None else random.randint(0, 4)) if game_mode['mode'] == 'multi' else random.randint(0, 4)
                                    if final_selected_col == ball_col: is_success, score = True, score + 1
                                    else: is_failure = True
                                    result_display_time = pygame.time.get_ticks()
                                    countdown_start_time = None
                        
                        if final_selected_col is not None:
                            highlight_surf = pygame.Surface((cell_w, screen_height), pygame.SRCALPHA); highlight_surf.fill(HIGHLIGHT_COLOR)
                            screen.blit(highlight_surf, (final_selected_col * cell_w, 0))

                        if ball_col is not None and ball_image:
                            ball_rect = ball_image.get_rect(center=(ball_col * cell_w + cell_w / 2, screen_height / 2))
                            screen.blit(ball_image, ball_rect)

                    # [수정] shoot.gif는 카운트다운이 시작되었을 때만 재생
                    if bg_video and countdown_start_time and bg_video_original_w > 0 and not waiting_for_start:
                        elapsed = pygame.time.get_ticks() - countdown_start_time
                        current_frame_pos = int(elapsed / bg_video_frame_interval)
                        
                        if current_frame_pos < bg_video_total_frames:
                            bg_video.set(cv2.CAP_PROP_POS_FRAMES, current_frame_pos)
                            ret_vid, frame_vid = bg_video.read()
                            if ret_vid:
                                new_w, new_h = get_scaled_rect(bg_video_original_w, bg_video_original_h, half_width, screen_height)
                                pos_x = half_width + (half_width - new_w) // 2
                                pos_y = (screen_height - new_h) // 2
                                frame_vid_rgb = cv2.cvtColor(frame_vid, cv2.COLOR_BGR2RGB)
                                frame_vid_resized = cv2.resize(frame_vid_rgb, (new_w, new_h))
                                screen.blit(pygame.surfarray.make_surface(frame_vid_resized.swapaxes(0, 1)), (pos_x, pos_y))
                
                else: 
                    ret, frame = cap.read()
                    if ret:
                        frame = cv2.flip(frame, 1)
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame_resized = cv2.resize(frame_rgb, (screen_width, screen_height))
                        screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (0, 0))
                        for i in range(1, 5): pygame.draw.line(screen, GRID_COLOR, (i*screen_width//5, 0), (i*screen_width//5, screen_height), 2)

            if scoreboard_ball_image:
                for i in range(chances_left): screen.blit(scoreboard_ball_image, (screen_width - 100 - i*90, 50))
            score_text = score_font.render(f"SCORE: {score}", True, WHITE)
            screen.blit(score_text, (screen_width - 300, 150))

        elif screen_state["current"] == "info":
            screen.blit(info_bg, (0, 0))
            title_surf = title_font.render("게임 방법", True, WHITE)
            screen.blit(title_surf, (screen_width/2 - title_surf.get_width()/2, 200))
            text_1p = ["[1인 플레이]", "", "1. 스페이스 바를 누르면 5초 카운트 다운이 시작됩니다.", "", "2. 5개의 영역 중 한 곳을 선택합니다.", "", "3. 5번의 기회동안 최대한 많은 공을 막으세요!"]
            text_2p = ["[2인 플레이]", "", "1. 스페이스 바를 누르면 5초 카운트 다운이 시작됩니다.", "", "2. 공격수와 골키퍼로 나뉩니다.", "", "3. 공격수는 공을 찰 방향을 정합니다.", "", "4. 골키퍼는 공을 막을 방향을 정합니다.", "", "5. 5번의 기회동안 더 많은 득점을 한 쪽이 승리합니다!"]
            for i, line in enumerate(text_1p): screen.blit(description_font.render(line, True, WHITE), (screen_width/4 - 550, 475 + i*75))
            for i, line in enumerate(text_2p): screen.blit(description_font.render(line, True, WHITE), (screen_width*3/4 - 500, 475 + i*75))

        elif screen_state["current"] == "end":
            if end_video_to_play:
                read_new_frame = not (end_video_to_play == defeat_video and loop_counter % 2 != 0)
                if read_new_frame:
                    ret, frame = end_video_to_play.read()
                    if not ret: end_video_to_play.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = end_video_to_play.read()
                    if ret: last_end_frame = frame
                if last_end_frame is not None:
                    frame_rgb = cv2.cvtColor(last_end_frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (screen_width, screen_height))
                    screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (0, 0))
            else:
                screen.fill(BLACK)
            
            rank_surf = rank_font.render(final_rank, True, GOLD_COLOR)
            screen.blit(rank_surf, rank_surf.get_rect(center=(screen_width/2, screen_height/2 - 150)))
            score_surf = score_font.render(f"FINAL SCORE: {score}", True, WHITE)
            screen.blit(score_surf, score_surf.get_rect(center=(screen_width/2, screen_height/2)))
            highscore_surf = score_font.render(f"HIGH SCORE: {highscore}", True, GOLD_COLOR)
            screen.blit(highscore_surf, highscore_surf.get_rect(center=(screen_width/2, screen_height/2 + 80)))
        
        for button in buttons.get(screen_state["current"], []): button.draw(screen)

        if fading_out or fading_in:
            if fading_out:
                transition_alpha = min(255, transition_alpha + transition_speed)
                if transition_alpha == 255: fading_out, fading_in, screen_state["current"] = False, True, transition_target
            else: # fading_in
                transition_alpha = max(0, transition_alpha - transition_speed)
                if transition_alpha == 0: fading_in = False
            transition_surface.set_alpha(transition_alpha); screen.blit(transition_surface, (0, 0))

        pygame.display.flip()
        clock.tick(60)
        loop_counter += 1

    # 리소스 해제
    if cap: cap.release()
    if bg_video: bg_video.release()
    video.release()
    if failure_gif: failure_gif.release()
    if success_gif: success_gif.release()
    if victory_video: victory_video.release()
    if defeat_video: defeat_video.release()
    if ser and ser.is_open: ser.close()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()