import pygame
import sys
import cv2
import numpy as np
import random
import serial # pyserial 라이브러리
import os

pygame.init()
pygame.mixer.init()

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
GOLD_COLOR = (255, 215, 0) # [추가] 최고 기록용 색상

# Font setup
try:
    font = pygame.font.Font("./fonts/netmarbleM.ttf", 40)
    description_font = pygame.font.Font("./fonts/netmarbleM.ttf", 50)
    title_font = pygame.font.Font("./fonts/netmarbleB.ttf", 120)
    countdown_font = pygame.font.Font("./fonts/netmarbleM.ttf", 200)
    score_font = pygame.font.Font("./fonts/netmarbleB.ttf", 60)
    rank_font = pygame.font.Font("./fonts/netmarbleB.ttf", 100)
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
# ImageButton 클래스
# =========================================
class ImageButton:
    def __init__(self, image_path, x, y, width=None, height=None, action=None, sound=None):
        self.action, self.sound, self.is_hovered = action, sound, False
        try:
            self.original_image = pygame.image.load(image_path).convert_alpha()
            scale_factor = 1.05
            self.image = pygame.transform.scale(self.original_image, (width, height)) if width and height else self.original_image
            hover_width = int(self.image.get_width() * scale_factor)
            hover_height = int(self.image.get_height() * scale_factor)
            self.hover_image = pygame.transform.scale(self.original_image, (hover_width, hover_height))
            self.rect = self.image.get_rect(center=(x, y))
        except pygame.error as e:
            print(f"이미지 로드 오류: {image_path} - {e}")
            self.image = pygame.Surface((width or 100, height or 50)); self.image.fill(BUTTON_COLOR)
            self.hover_image = pygame.Surface((width or 100, height or 50)); self.hover_image.fill((150,150,150))
            self.rect = self.image.get_rect(center=(x, y))

    def update(self):
        self.is_hovered = self.rect.collidepoint(pygame.mouse.get_pos())

    def draw(self, screen):
        current_image = self.hover_image if self.is_hovered else self.image
        screen.blit(current_image, current_image.get_rect(center=self.rect.center))

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.is_hovered:
            if self.sound: self.sound.play()
            if self.action: self.action()

# ==========================
# 메인 함수
# ==========================
def main():
    screen_state = {"current": "menu"}
    
    # 게임 상태 변수
    chances_left, score = 5, 0
    highscore = load_highscore() # [추가] 최고 기록 로드
    final_rank, end_video_to_play = "", None # [추가] 종료 화면용 변수
    countdown_start_time, selected_grid_col, final_selected_col, ball_col = None, None, None, None
    is_failure, is_success, result_display_time, gif_start_time = False, False, None, None
    uart_ball_col = None

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
        ser = serial.Serial('COM13', 9600, timeout=0) 
        print("Basys3 보드가 성공적으로 연결되었습니다.")
    except serial.SerialException as e:
        print(f"오류: 시리얼 포트를 열 수 없습니다 - {e}")
        
    # 효과음 로드
    try:
        button_sound = pygame.mixer.Sound("./sound/button_click.wav")
        siu_sound = pygame.mixer.Sound("./sound/SIUUUUU.wav")
        success_sound = pygame.mixer.Sound("./sound/SIUUUUU.wav")
    except pygame.error as e:
        print(f"효과음 로드 오류: {e}"); button_sound, siu_sound, success_sound = None, None, None
    
    # 게임 이미지 및 GIF/영상 로드
    try:
        ball_image = pygame.image.load("./image/final_ronaldo/Ball.png").convert_alpha()
        scoreboard_ball_image = pygame.transform.scale(ball_image, (80, 80)) 
        ball_image = pygame.transform.scale(ball_image, (200, 200))
    except pygame.error as e: print(f"이미지 로드 오류: Ball.png - {e}"); ball_image = None; scoreboard_ball_image = None

    try: failure_gif = cv2.VideoCapture("./image/G.O.A.T/siuuu.gif")
    except Exception as e: print(f"GIF 로드 오류: siuuu.gif - {e}"); failure_gif = None
    try: success_gif = cv2.VideoCapture("./image/final_ronaldo/pk.gif")
    except Exception as e: print(f"GIF 로드 오류: pk.gif - {e}"); success_gif = None
    # [추가] 승리/패배 영상 로드
    try: victory_video = cv2.VideoCapture("./image/victory.gif")
    except Exception as e: print(f"영상 로드 오류: victory.gif - {e}"); victory_video = None
    try: defeat_video = cv2.VideoCapture("./image/defeat.gif")
    except Exception as e: print(f"영상 로드 오류: defeat.gif - {e}"); defeat_video = None

    # ==========================
    # 액션 함수
    # ==========================
    def start_transition(target_state):
        nonlocal transition_target, fading_out
        if not fading_out and not fading_in: transition_target, fading_out = target_state, True

    def reset_game_state(full_reset=True): # [수정] full_reset 파라미터 추가
        nonlocal countdown_start_time, selected_grid_col, final_selected_col, ball_col
        nonlocal is_failure, is_success, result_display_time, gif_start_time, uart_ball_col
        nonlocal chances_left, score
        countdown_start_time, selected_grid_col, final_selected_col, ball_col = None, None, None, None
        is_failure, is_success, result_display_time, gif_start_time = False, False, None, None
        uart_ball_col = None
        if full_reset:
            chances_left, score = 5, 0

    def start_new_round():
        nonlocal countdown_start_time
        reset_game_state(full_reset=False) # [수정] 라운드 리셋 시 점수와 기회는 유지
        countdown_start_time = pygame.time.get_ticks()

    # [추가] test_ver15_3의 재시작/메뉴 이동 함수
    def restart_game():
        reset_game_state(full_reset=True)
        start_new_round()
        target_screen = "webcam_view" if game_mode["mode"] == "single" else "multi"
        start_transition(target_screen)

    def go_to_menu():
        reset_game_state(full_reset=True)
        start_transition("menu")

    def set_game_mode(mode):
        nonlocal game_mode
        if siu_sound: siu_sound.play()
        game_mode["mode"] = mode
        restart_game() # [수정] restart_game 함수로 통합하여 코드 간소화

    # [수정] 버튼 설정 변경
    game_mode = {"mode": None}
    buttons = {
        "menu": [ImageButton("./image/btn_start.png", screen_width - 300, screen_height - 175, 400, 250, lambda: start_transition("game"), sound=button_sound),
                 ImageButton("./image/btn_desc.png", screen_width - 150, 150, 100, 100, lambda: start_transition("info"), sound=button_sound)],
        "game": [ImageButton("./image/btn_single.png", screen_width//2 - 280, screen_height//2 + 200, 550, 600, lambda: set_game_mode("single")),
                 ImageButton("./image/btn_multi.png", screen_width//2 + 430, screen_height//2 + 200, 550, 600, lambda: set_game_mode("multi")),
                 ImageButton("./image/btn_back.png", 150, 150, 100, 100, go_to_menu, sound=button_sound)],
        "webcam_view": [ImageButton("./image/btn_back.png", 150, 150, 100, 100, go_to_menu, sound=button_sound)],
        "multi": [ImageButton("./image/btn_back.png", 150, 150, 100, 100, go_to_menu, sound=button_sound)],
        "info": [ImageButton("./image/btn_exit.png", screen_width - 150, 150, 100, 100, go_to_menu, sound=button_sound)],
        "end": [ImageButton("./image/btn_restart.png", screen_width//2 - 300, screen_height - 250, 400, 250, restart_game, sound=button_sound),
                ImageButton("./image/btn_main_menu.png", screen_width//2 + 300, screen_height - 250, 400, 250, go_to_menu, sound=button_sound)]
    }

    video = cv2.VideoCapture("./image/game_thumbnail.mp4")
    clock = pygame.time.Clock()
    info_bg = pygame.image.load("./image/info/info_back2.jpg").convert()
    info_bg = pygame.transform.scale(info_bg, (screen_width, screen_height))
    loop_counter, gif_frame = 0, None
    
    def handle_game_logic():
        nonlocal gif_start_time, final_selected_col, ball_col, chances_left, is_success, score, is_failure, highscore, final_rank, end_video_to_play
        nonlocal result_display_time, countdown_start_time, selected_grid_col, gif_frame, uart_ball_col

        should_play_gif = (is_failure or is_success) and result_display_time and (pygame.time.get_ticks() - result_display_time > 1000)
        
        # [수정] 게임 종료 로직
        if gif_start_time and (pygame.time.get_ticks() - gif_start_time > 2000):
            if chances_left > 0:
                start_new_round()
            else:
                # 최고 기록 갱신
                if score > highscore:
                    highscore = score
                    save_highscore(highscore)
                
                # 점수에 따른 칭호 및 승/패 영상 결정
                if score == 5: final_rank, end_video_to_play = "THE WALL", victory_video
                elif score >= 3: final_rank, end_video_to_play = "Pro Keeper", victory_video
                elif score >= 1: final_rank, end_video_to_play = "Rookie Keeper", defeat_video
                else: final_rank, end_video_to_play = "Human Sieve", defeat_video
                
                if end_video_to_play: end_video_to_play.set(cv2.CAP_PROP_POS_FRAMES, 0)
                start_transition("end")

        active_gif = None
        if should_play_gif:
            if is_failure: active_gif = failure_gif
            elif is_success: active_gif = success_gif

        if active_gif and not gif_start_time:
            gif_start_time = pygame.time.get_ticks()

        if gif_start_time:
            screen.fill(BLACK) 
            if loop_counter % 2 == 0:
                ret, frame = active_gif.read()
                if not ret: active_gif.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = active_gif.read()
                if ret: gif_frame = frame
            
            if gif_frame is not None:
                gif_display_size = (screen_width // 3, screen_height // 3) if is_success else (screen_width, screen_height)
                frame_resized = cv2.resize(gif_frame, gif_display_size, interpolation=cv2.INTER_AREA)
                frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                gif_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                gif_rect = gif_surface.get_rect(center=(screen_width // 2, screen_height // 2))
                screen.blit(gif_surface, gif_rect)
        
        elif cap: 
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1)
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_resized = cv2.resize(frame_rgb, (screen_width, screen_height))
                screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (0, 0))

                for i in range(1, 5): 
                    pygame.draw.line(screen, GRID_COLOR, (i * (screen_width // 5), 0), (i * (screen_width // 5), screen_height), 2)
                
                if countdown_start_time is not None:
                    elapsed_time = pygame.time.get_ticks() - countdown_start_time
                    if elapsed_time < 5000:
                        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
                        lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
                        mask = cv2.inRange(hsv_frame, lower_red1, upper_red1) + cv2.inRange(hsv_frame, lower_red2, upper_red2)
                        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 500:
                                x, _, w, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
                                _, cam_w, _ = frame.shape
                                selected_grid_col = int((x + w / 2) / (cam_w / 5))
                        else: 
                            selected_grid_col = None
                        
                        if game_mode["mode"] == 'multi' and ser and ser.is_open and ser.in_waiting > 0:
                            try:
                                uart_data_stream = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                                valid_chars = [c for c in uart_data_stream if c in '12345']
                                if valid_chars:
                                    last_valid_char = valid_chars[-1]
                                    uart_ball_col = int(last_valid_char) - 1
                            except Exception as e: print(f"UART 데이터 수신/처리 중 오류: {e}")

                        num = str(5 - (elapsed_time // 1000))
                        text_surf = countdown_font.render(num, True, WHITE)
                        screen.blit(text_surf, text_surf.get_rect(center=(screen_width // 2, screen_height // 2)))

                    else: # 5초가 지났을 때
                        if final_selected_col is None:
                            final_selected_col = selected_grid_col
                            chances_left -= 1
                            
                            if game_mode["mode"] == 'multi':
                                ball_col = uart_ball_col if uart_ball_col is not None else random.randint(0, 4)
                            else: # 싱글플레이 모드
                                ball_col = random.randint(0, 4)
                            
                            if ball_col is not None and final_selected_col == ball_col:
                                is_success, score = True, score + 1
                                if success_sound: success_sound.play()
                            else: is_failure = True
                            
                            result_display_time = pygame.time.get_ticks()
                            countdown_start_time = None
                
                if final_selected_col is not None:
                    cell_w = screen_width / 5
                    highlight_surf = pygame.Surface((cell_w, screen_height), pygame.SRCALPHA)
                    highlight_surf.fill(HIGHLIGHT_COLOR)
                    screen.blit(highlight_surf, (final_selected_col * cell_w, 0))

                if ball_col is not None and ball_image:
                    cell_w = screen_width / 5
                    ball_rect = ball_image.get_rect(center=(ball_col * cell_w + cell_w / 2, screen_height / 2))
                    screen.blit(ball_image, ball_rect)
        
        # 스코어보드 그리기
        if scoreboard_ball_image:
            for i in range(chances_left):
                screen.blit(scoreboard_ball_image, (screen_width - 100 - i*90, 50))
        score_text = score_font.render(f"SCORE: {score}", True, WHITE)
        screen.blit(score_text, (screen_width - 300, 150))

    # ==========================
    # 메인 루프
    # ==========================
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): running = False
            if not (fading_in or fading_out):
                for button in buttons.get(screen_state["current"], []): button.handle_event(event)
        
        if not (fading_in or fading_out):
            for button in buttons.get(screen_state["current"], []): button.update()

        # 화면 그리기
        if screen_state["current"] in ["menu", "game"]:
            ret, frame = video.read()
            if not ret: video.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame = cv2.resize(frame, (screen_width, screen_height))
                screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))

        if screen_state["current"] == "game":
            text_surf = font.render("플레이어 수를 선택하세요", True, WHITE)
            screen.blit(text_surf, (screen_width//2 - text_surf.get_width()//2, screen_height//2 - 200))
        
        elif screen_state["current"] in ["webcam_view", "multi"]:
            handle_game_logic()

        elif screen_state["current"] == "info":
            screen.blit(info_bg, (0, 0))
            title_surf = title_font.render("게임 방법", True, WHITE)
            screen.blit(title_surf, (screen_width / 2 - title_surf.get_width() / 2, 150))
            text_lines_1p = ["[1인 플레이]", "1. 5초의 카운트 다운이 시작됩니다.", "2. 카메라에 비치는 빨간색", "   물체를 인식합니다.", "3. 5개의 영역 중 하나를 선택합니다.", "4. 공을 막으면 성공!"]
            text_lines_2p = ["[2인 플레이]", "1. 플레이어는 공격수가 되어", "   몸으로 방향을 정합니다.", "2. 골키퍼(Basys3)는 스위치로", "   막을 방향을 정합니다.", "3. 더 많은 득점을 한 플레이어가", "   승리합니다."]
            x_offset_1p, x_offset_2p, y_start = screen_width / 4 - 150, screen_width * 3 / 4 - 300, 400
            for i, line in enumerate(text_lines_1p): screen.blit(description_font.render(line, True, WHITE), (x_offset_1p, y_start + i*75))
            for i, line in enumerate(text_lines_2p): screen.blit(description_font.render(line, True, WHITE), (x_offset_2p, y_start + i*75))
        
        # [수정] 종료 화면 그리기 로직
        elif screen_state["current"] == "end":
            if end_video_to_play:
                ret, frame = end_video_to_play.read()
                if not ret: end_video_to_play.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = end_video_to_play.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame = cv2.resize(frame, (screen_width, screen_height))
                    screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
            else:
                screen.fill(BLACK)
            
            # 칭호, 최종 점수, 최고 점수 표시
            rank_surf = rank_font.render(final_rank, True, GOLD_COLOR)
            screen.blit(rank_surf, rank_surf.get_rect(center=(screen_width/2, screen_height/2 - 150)))
            
            score_surf = score_font.render(f"FINAL SCORE: {score}", True, WHITE)
            screen.blit(score_surf, score_surf.get_rect(center=(screen_width/2, screen_height/2)))
            
            highscore_surf = score_font.render(f"HIGH SCORE: {highscore}", True, GOLD_COLOR)
            screen.blit(highscore_surf, highscore_surf.get_rect(center=(screen_width/2, screen_height/2 + 80)))
        
        for button in buttons.get(screen_state["current"], []): button.draw(screen)

        # 화면 전환 효과
        if fading_out:
            transition_alpha = min(255, transition_alpha + transition_speed)
            if transition_alpha == 255: fading_out, fading_in, screen_state["current"] = False, True, transition_target
            transition_surface.set_alpha(transition_alpha); screen.blit(transition_surface, (0, 0))
        elif fading_in:
            transition_alpha = max(0, transition_alpha - transition_speed)
            if transition_alpha == 0: fading_in = False
            transition_surface.set_alpha(transition_alpha); screen.blit(transition_surface, (0, 0))

        pygame.display.flip()
        clock.tick(60)
        loop_counter += 1

    # 프로그램 종료 시 모든 리소스 해제
    if cap: cap.release()
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