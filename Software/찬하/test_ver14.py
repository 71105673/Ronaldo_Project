import pygame
import sys
import cv2
import numpy as np
import random

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

# Font setup
try:
    font = pygame.font.Font("./fonts/netmarbleM.ttf", 40)
    description_font = pygame.font.Font("./fonts/netmarbleM.ttf", 50)
    title_font = pygame.font.Font("./fonts/netmarbleB.ttf", 120)
    countdown_font = pygame.font.Font("./fonts/netmarbleM.ttf", 200)
    score_font = pygame.font.Font("./fonts/netmarbleB.ttf", 60) # 스코어보드 폰트
except FileNotFoundError:
    print("폰트 파일을 찾을 수 없습니다. 기본 폰트를 사용합니다.")
    font = pygame.font.Font(None, 50)
    description_font = pygame.font.Font(None, 60)
    title_font = pygame.font.Font(None, 130)
    countdown_font = pygame.font.Font(None, 250)
    score_font = pygame.font.Font(None, 70)

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
            hover_width = int(width * scale_factor) if width else int(self.original_image.get_width() * scale_factor)
            hover_height = int(height * scale_factor) if height else int(self.original_image.get_height() * scale_factor)
            self.hover_image = pygame.transform.scale(self.original_image, (hover_width, hover_height))
            self.rect = self.image.get_rect(center=(x, y))
        except pygame.error as e:
            print(f"이미지 로드 오류: {image_path} - {e}")
            self.image = pygame.Surface((width, height)); self.image.fill(BUTTON_COLOR)
            self.hover_image = pygame.Surface((width, height)); self.hover_image.fill((150,150,150))
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
    countdown_start_time, selected_grid_col, final_selected_col, ball_col = None, None, None, None
    is_failure, is_success, result_display_time, gif_start_time = False, False, None, None

    # 화면 전환 변수
    transition_surface = pygame.Surface((screen_width, screen_height)); transition_surface.fill(BLACK)
    transition_alpha, transition_target, transition_speed = 0, None, 15
    fading_out, fading_in = False, False

    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("오류: 웹캠을 열 수 없습니다."); cap = None

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
    try: success_gif = cv2.VideoCapture("./image/final_ronaldo/success.gif")
    except Exception as e: print(f"GIF 로드 오류: success.gif - {e}"); success_gif = None
    try: end_video = cv2.VideoCapture("./video/ending.mp4") # 종료 영상
    except Exception as e: print(f"영상 로드 오류: ending.mp4 - {e}"); end_video = None

    # ==========================
    # 액션 함수
    # ==========================
    def start_transition(target_state):
        nonlocal transition_target, fading_out
        if not fading_out and not fading_in: transition_target, fading_out = target_state, True

    def reset_game_state():
        nonlocal countdown_start_time, selected_grid_col, final_selected_col, ball_col, is_failure, is_success, result_display_time, gif_start_time, chances_left, score
        countdown_start_time, selected_grid_col, final_selected_col, ball_col = None, None, None, None
        is_failure, is_success, result_display_time, gif_start_time = False, False, None, None
        chances_left, score = 5, 0

    def start_new_round():
        nonlocal countdown_start_time, selected_grid_col, final_selected_col, ball_col, is_failure, is_success, result_display_time, gif_start_time
        final_selected_col, ball_col = None, None
        is_failure, is_success, result_display_time, gif_start_time = False, False, None, None
        countdown_start_time = pygame.time.get_ticks()

    def go_back():
        current = screen_state["current"]
        target = "game" if current in ["single", "multi", "webcam_view", "end"] else "menu"
        if target == "game" or target == "menu": reset_game_state()
        start_transition(target)

    def set_game_mode(mode):
        nonlocal countdown_start_time
        if siu_sound: siu_sound.play()
        game_mode["mode"] = mode
        if mode == "single":
            countdown_start_time = pygame.time.get_ticks() 
            start_transition("webcam_view")
        else: start_transition(mode)

    # 버튼 생성
    game_mode = {"mode": None}
    buttons = {
        "menu": [ImageButton("./image/btn_start.png", screen_width - 300, screen_height - 175, 400, 250, lambda: start_transition("game"), sound=button_sound),
                 ImageButton("./image/btn_desc.png", screen_width - 150, 150, 100, 100, lambda: start_transition("info"), sound=button_sound)],
        "game": [ImageButton("./image/btn_single.png", screen_width//2 - 280, screen_height//2 + 200, 550, 600, lambda: set_game_mode("single")),
                 ImageButton("./image/btn_multi.png", screen_width//2 + 430, screen_height//2 + 200, 550, 600, lambda: set_game_mode("multi")),
                 ImageButton("./image/btn_back.png", 150, 150, 100, 100, go_back, sound=button_sound)],
        "webcam_view": [ImageButton("./image/btn_back.png", 150, 150, 100, 100, go_back, sound=button_sound)],
        "multi": [ImageButton("./image/btn_back.png", 150, 150, 100, 100, go_back, sound=button_sound)],
        "info": [ImageButton("./image/btn_exit.png", screen_width - 150, 150, 100, 100, go_back, sound=button_sound)],
        "end": [ImageButton("./image/btn_exit.png", screen_width - 150, 150, 100, 100, go_back, sound=button_sound)]
    }

    video = cv2.VideoCapture("./image/game_thumbnail.mp4")
    clock = pygame.time.Clock()
    info_bg = pygame.image.load("./image/info/info_back2.jpg").convert()
    info_bg = pygame.transform.scale(info_bg, (screen_width, screen_height))
    loop_counter, gif_frame = 0, None

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
        if screen_state["current"] in ["menu", "game", "multi"]:
            ret, frame = video.read();
            if not ret: video.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame = cv2.resize(frame, (screen_width, screen_height))
                screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))

        if screen_state["current"] == "game":
            text_surf = font.render("플레이어 수를 선택하세요", True, WHITE)
            screen.blit(text_surf, (screen_width//2 - text_surf.get_width()//2, screen_height//2 - 200))
        
        elif screen_state["current"] == "webcam_view":
            should_play_gif = (is_failure or is_success) and result_display_time and (pygame.time.get_ticks() - result_display_time > 1000)
            
            if gif_start_time and (pygame.time.get_ticks() - gif_start_time > 2000): # GIF 2초 재생 후
                if chances_left > 0:
                    start_new_round()
                else:
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
                    gif_display_size = (0,0)
                    if is_success: gif_display_size = (screen_width // 3, screen_height // 3) 
                    elif is_failure: gif_display_size = (screen_width, screen_height)
                    frame_resized = cv2.resize(gif_frame, gif_display_size, interpolation=cv2.INTER_AREA)
                    frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                    gif_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                    gif_rect = gif_surface.get_rect(center=(screen_width // 2, screen_height // 2))
                    screen.blit(gif_surface, gif_rect)
            
            elif cap: 
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    if countdown_start_time is not None and (pygame.time.get_ticks() - countdown_start_time) < 5000:
                        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
                        lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
                        mask = cv2.inRange(hsv_frame, lower_red1, upper_red1) + cv2.inRange(hsv_frame, lower_red2, upper_red2)
                        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            if cv2.contourArea(largest_contour) > 500:
                                x, y, w, h = cv2.boundingRect(largest_contour); cam_h, cam_w, _ = frame.shape
                                selected_grid_col = int((x + w / 2) / (cam_w / 5))
                        else: selected_grid_col = None
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame_resized = cv2.resize(frame_rgb, (screen_width, screen_height))
                    screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (0, 0))
                    for i in range(1, 5): pygame.draw.line(screen, GRID_COLOR, (i * (screen_width // 5), 0), (i * (screen_width // 5), screen_height), 2)
                    
                    if countdown_start_time is not None:
                        elapsed_time = pygame.time.get_ticks() - countdown_start_time
                        if elapsed_time < 5000:
                            num = str(5 - (elapsed_time // 1000))
                            text_surf = countdown_font.render(num, True, WHITE)
                            screen.blit(text_surf, text_surf.get_rect(center=(screen_width // 2, screen_height // 2)))
                        else:
                            if final_selected_col is None:
                                final_selected_col = selected_grid_col
                                ball_col = random.randint(0, 4)
                                chances_left -= 1
                                
                                if final_selected_col == ball_col:
                                    is_success = True; score += 1
                                    if success_sound: success_sound.play()
                                    print("SUCCESS!")
                                else:
                                    is_failure = True
                                    print(f"FAILURE! Player: {final_selected_col}, Ball: {ball_col}")
                                
                                result_display_time = pygame.time.get_ticks()
                                if final_selected_col is not None: print(f"FPGA Signal: Cell {final_selected_col}")
                            countdown_start_time = None
                    
                    if final_selected_col is not None:
                        cell_w, cell_h = screen_width / 5, screen_height
                        highlight_surf = pygame.Surface((cell_w, cell_h), pygame.SRCALPHA)
                        highlight_surf.fill(HIGHLIGHT_COLOR)
                        screen.blit(highlight_surf, (final_selected_col * cell_w, 0))

                    if ball_col is not None and ball_image:
                        cell_w, cell_h = screen_width / 5, screen_height
                        ball_rect = ball_image.get_rect(center=(ball_col * cell_w + cell_w / 2, cell_h / 2))
                        screen.blit(ball_image, ball_rect)

                else: screen.fill(BLACK)
            else: screen.fill(BLACK)
            
            # 스코어보드 그리기
            if scoreboard_ball_image:
                for i in range(chances_left):
                    screen.blit(scoreboard_ball_image, (screen_width - 100 - i*90, 50))
            score_text = score_font.render(f"SCORE: {score}", True, WHITE)
            screen.blit(score_text, (screen_width - 300, 150))


        elif screen_state["current"] == "info":
            screen.blit(info_bg, (0, 0))
            title_surf = title_font.render("게임 방법", True, WHITE)
            screen.blit(title_surf, (screen_width / 2 - title_surf.get_width() / 2, 150))
            text_lines_1p = ["[1인 플레이]", "1. 5초의 카운트 다운이 시작됩니다.", "2. 카메라에 비치는 빨간색", "   물체를 인식합니다.", "3. 5개의 영역 중 하나를 선택합니다.", "4. 공을 막으면 성공!"]
            text_lines_2p = ["[2인 플레이]", "1. COM과 번갈아가며 공격과", "   수비를 합니다.", "2. 5번의 기회가 주어집니다.", "3. 더 많은 득점을 한 플레이어가", "   승리합니다."]
            x_offset_1p, x_offset_2p, y_start = screen_width / 4 - 150, screen_width * 3 / 4 - 300, 400
            for i, line in enumerate(text_lines_1p): screen.blit(description_font.render(line, True, WHITE), (x_offset_1p, y_start + i*75))
            for i, line in enumerate(text_lines_2p): screen.blit(description_font.render(line, True, WHITE), (x_offset_2p, y_start + i*75))
        
        elif screen_state["current"] == "end":
            if end_video:
                ret, frame = end_video.read()
                if not ret: end_video.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = end_video.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame = cv2.resize(frame, (screen_width, screen_height))
                    screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
            else:
                screen.fill(BLACK)
                end_text = title_font.render("GAME OVER", True, WHITE)
                screen.blit(end_text, end_text.get_rect(center=(screen_width/2, screen_height/2)))
        
        for button in buttons.get(screen_state["current"], []): button.draw(screen)

        if fading_out:
            transition_alpha += transition_speed
            if transition_alpha >= 255: transition_alpha = 255; fading_out = False; fading_in = True; screen_state["current"] = transition_target; transition_target = None
            transition_surface.set_alpha(transition_alpha); screen.blit(transition_surface, (0, 0))
        elif fading_in:
            transition_alpha -= transition_speed
            if transition_alpha <= 0: transition_alpha = 0; fading_in = False
            transition_surface.set_alpha(transition_alpha); screen.blit(transition_surface, (0, 0))

        pygame.display.flip()
        clock.tick(60)
        loop_counter += 1

    if cap: cap.release()
    video.release()
    if failure_gif: failure_gif.release()
    if success_gif: success_gif.release()
    if end_video: end_video.release()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()

