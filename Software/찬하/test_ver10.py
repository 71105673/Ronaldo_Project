import pygame
import sys
import cv2
import numpy as np
import random # --- 추가: 랜덤 기능 사용을 위해 import ---

pygame.init()
pygame.mixer.init() # 사운드 시스템 초기화

# 자동 전체 화면 설정
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width = screen.get_width()
screen_height = screen.get_height()

pygame.display.set_caption("Pygame with Webcam")

# 색상 정의
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BUTTON_COLOR = (100, 100, 100)
GRID_COLOR = (0, 255, 0)
HIGHLIGHT_COLOR = (255, 0, 0, 100) # 하이라이트 색상 (R, G, B, Alpha)

# 한글 폰트
try:
    font = pygame.font.Font("C:/Windows/Fonts/malgun.ttf", 40)
    countdown_font = pygame.font.Font("C:/Windows/Fonts/malgun.ttf", 200)
except FileNotFoundError:
    font = pygame.font.Font(None, 50)
    countdown_font = pygame.font.Font(None, 250)

# =========================================
# 버튼 호버 효과가 추가된 ImageButton 클래스
# =========================================
class ImageButton:
    def __init__(self, image_path, x, y, width=None, height=None, action=None, sound=None):
        self.action = action
        self.sound = sound
        self.is_hovered = False
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
        mouse_pos = pygame.mouse.get_pos()
        self.is_hovered = self.rect.collidepoint(mouse_pos)

    def draw(self, screen):
        current_image = self.hover_image if self.is_hovered else self.image
        draw_rect = current_image.get_rect(center=self.rect.center)
        screen.blit(current_image, draw_rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.is_hovered:
            if self.sound:
                self.sound.play()
            if self.action:
                self.action()

# ==========================
# 메인 함수
# ==========================
def main():
    screen_state = {"current": "menu"}
    
    # 웹캠 및 게임 상태 변수
    countdown_start_time = None
    selected_grid_cell = None 
    final_selected_cell = None 
    ball_cell = None
    is_failure = False # --- 수정: 'match_success'에서 'is_failure'로 이름 변경하여 명확성 확보 ---
    result_display_time = None # --- 추가: 결과 화면 표시 시작 시간을 기록할 타이머 ---

    # 화면 전환 변수
    transition_surface = pygame.Surface((screen_width, screen_height))
    transition_surface.fill(BLACK)
    transition_alpha = 0
    transition_target = None
    transition_speed = 15
    fading_out = False
    fading_in = False

    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다."); cap = None

    # 효과음 로드
    try:
        button_sound = pygame.mixer.Sound("./sound/button_click.wav")
        siu_sound = pygame.mixer.Sound("./sound/SIUUUUU.wav")
    except pygame.error as e:
        print(f"효과음 로드 오류: {e}")
        button_sound = None
        siu_sound = None
    
    # 게임 이미지 및 GIF 로드
    try:
        ball_image = pygame.image.load("./image/final_ronaldo/Ball.png").convert_alpha()
        ball_image = pygame.transform.scale(ball_image, (200, 200)) 
    except pygame.error as e:
        print(f"이미지 로드 오류: Ball.png - {e}"); ball_image = None

    try:
        failure_gif = cv2.VideoCapture("./image/G.O.A.T/siuuu.gif")
    except Exception as e:
        print(f"GIF 로드 오류: siuuu.gif - {e}"); failure_gif = None
        
    # ==========================
    # 버튼 액션 함수
    # ==========================
    def start_transition(target_state):
        nonlocal transition_target, fading_out
        if not fading_out and not fading_in:
            transition_target = target_state
            fading_out = True

    def go_back():
        nonlocal countdown_start_time, selected_grid_cell, final_selected_cell, ball_cell, is_failure, result_display_time
        current = screen_state["current"]
        target = "menu"
        if current in ["single", "multi", "webcam_view"]:
            target = "game"
            # 모든 게임 상태 변수 초기화
            countdown_start_time = None
            selected_grid_cell = None
            final_selected_cell = None
            ball_cell = None
            is_failure = False
            result_display_time = None
        elif current == "game" or current == "info":
            target = "menu"
        start_transition(target)

    def set_game_mode(mode):
        nonlocal countdown_start_time
        if siu_sound:
            siu_sound.play()
        game_mode["mode"] = mode
        if mode == "single":
            countdown_start_time = pygame.time.get_ticks() 
            start_transition("webcam_view")
        else:
            start_transition(mode)

    # ==========================
    # 버튼 생성
    # ==========================
    game_mode = {"mode": None}
    buttons = {
        "menu": [
            ImageButton("./image/btn_start.png", screen_width - 300, screen_height - 175, 400, 250, lambda: start_transition("game"), sound=button_sound),
            ImageButton("./image/btn_desc.png", screen_width - 150, 150, 100, 100, lambda: start_transition("info"), sound=button_sound)
        ],
        "game": [
            ImageButton("./image/btn_single.png", screen_width//2 - 280, screen_height//2 + 200, 550, 600, lambda: set_game_mode("single")),
            ImageButton("./image/btn_multi.png", screen_width//2 + 430, screen_height//2 + 200, 550, 600, lambda: set_game_mode("multi")),
            ImageButton("./image/btn_back.png", 150, 150, 100, 100, go_back, sound=button_sound)
        ], "webcam_view": [ ImageButton("./image/btn_back.png", 150, 150, 100, 100, go_back, sound=button_sound) ],
        "multi": [ ImageButton("./image/btn_back.png", 150, 150, 100, 100, go_back, sound=button_sound) ],
        "info": [ ImageButton("./image/btn_exit.png", screen_width - 150, 150, 100, 100, lambda: start_transition("menu"), sound=button_sound) ]
    }

    video = cv2.VideoCapture("./image/game_thumbnail.mp4")
    clock = pygame.time.Clock()
    info_bg = pygame.image.load("./image/des_back.jpg").convert()
    info_bg = pygame.transform.scale(info_bg, (screen_width, screen_height))
    
    # --- 추가: GIF 재생 속도 제어를 위한 변수 ---
    loop_counter = 0
    gif_frame = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            if not (fading_in or fading_out):
                for button in buttons.get(screen_state["current"], []):
                    button.handle_event(event)
        
        if not (fading_in or fading_out):
            for button in buttons.get(screen_state["current"], []):
                button.update()

        # ==========================
        # 화면 그리기
        # ==========================
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
            # --- 수정: GIF 재생 조건 및 로직 변경 ---
            should_play_gif = is_failure and result_display_time and (pygame.time.get_ticks() - result_display_time > 1000) # 1초 후

            if should_play_gif and failure_gif:
                # 2프레임마다 한 번씩만 새 프레임을 읽어와 GIF 속도를 절반으로 줄임
                if loop_counter % 2 == 0:
                    ret, frame = failure_gif.read()
                    if not ret: failure_gif.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = failure_gif.read()
                    if ret: gif_frame = frame
                
                if gif_frame is not None:
                    frame = cv2.cvtColor(gif_frame, cv2.COLOR_BGR2RGB); frame = cv2.resize(frame, (screen_width, screen_height))
                    screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
            
            elif cap: 
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    
                    if countdown_start_time is not None and (pygame.time.get_ticks() - countdown_start_time) < 5000:
                        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        lower_red1 = np.array([0, 120, 70]); upper_red1 = np.array([10, 255, 255])
                        lower_red2 = np.array([170, 120, 70]); upper_red2 = np.array([180, 255, 255])
                        mask = cv2.inRange(hsv_frame, lower_red1, upper_red1) + cv2.inRange(hsv_frame, lower_red2, upper_red2)
                        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            if cv2.contourArea(largest_contour) > 500:
                                x, y, w, h = cv2.boundingRect(largest_contour)
                                cam_h, cam_w, _ = frame.shape
                                col = int((x + w / 2) / (cam_w / 5)); row = int((y + h / 2) / (cam_h / 2))
                                selected_grid_cell = (row, col)
                        else: selected_grid_cell = None
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame_resized = cv2.resize(frame_rgb, (screen_width, screen_height))
                    screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (0, 0))
                    
                    pygame.draw.line(screen, GRID_COLOR, (0, screen_height // 2), (screen_width, screen_height // 2), 2)
                    for i in range(1, 5): pygame.draw.line(screen, GRID_COLOR, (i * (screen_width // 5), 0), (i * (screen_width // 5), screen_height), 2)
                    
                    if countdown_start_time is not None:
                        elapsed_time = pygame.time.get_ticks() - countdown_start_time
                        if elapsed_time < 5000:
                            num = str(5 - (elapsed_time // 1000))
                            text_surf = countdown_font.render(num, True, WHITE)
                            screen.blit(text_surf, text_surf.get_rect(center=(screen_width // 2, screen_height // 2)))
                        else:
                            if final_selected_cell is None:
                                final_selected_cell = selected_grid_cell
                                ball_cell = (random.randint(0, 1), random.randint(0, 4))
                                
                                # --- 수정: 실패했을 때 is_failure를 True로 설정 ---
                                if final_selected_cell != ball_cell:
                                    is_failure = True
                                    print(f"FAILURE! Player: {final_selected_cell}, Ball: {ball_cell}")
                                else: print("SUCCESS!")
                                
                                # --- 추가: 결과 표시 타이머 시작 ---
                                result_display_time = pygame.time.get_ticks()

                                if final_selected_cell is not None: print(f"FPGA Signal: Cell {final_selected_cell}")
                            countdown_start_time = None
                    
                    if final_selected_cell is not None:
                        row, col = final_selected_cell
                        cell_w, cell_h = screen_width / 5, screen_height / 2
                        highlight_surf = pygame.Surface((cell_w, cell_h), pygame.SRCALPHA)
                        highlight_surf.fill(HIGHLIGHT_COLOR)
                        screen.blit(highlight_surf, (col * cell_w, row * cell_h))

                    if ball_cell is not None and ball_image:
                        row, col = ball_cell
                        cell_w, cell_h = screen_width / 5, screen_height / 2
                        ball_rect = ball_image.get_rect(center=(col * cell_w + cell_w / 2, row * cell_h + cell_h / 2))
                        screen.blit(ball_image, ball_rect)

                else: screen.fill(BLACK)
            else: screen.fill(BLACK)

        elif screen_state["current"] == "multi":
            text_surf = font.render("Multi 플레이 화면!", True, WHITE)
            screen.blit(text_surf, (screen_width//2 - text_surf.get_width()//2, 300))
            
        elif screen_state["current"] == "info":
            screen.blit(info_bg, (0, 0))
            text_lines = ["게임 설명:", "- 화살표 키로 움직입니다.", "- 장애물을 피하세요.", "- 점수를 모으세요."]
            for i, line in enumerate(text_lines):
                screen.blit(font.render(line, True, BLACK), (200, 200 + i * 60))
        
        for button in buttons.get(screen_state["current"], []):
            button.draw(screen)

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
        loop_counter += 1 # GIF 속도 제어를 위해 루프 카운터 증가

    if cap: cap.release()
    video.release()
    if failure_gif: failure_gif.release()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()

