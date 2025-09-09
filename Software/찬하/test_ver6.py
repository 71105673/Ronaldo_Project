import pygame
import sys
import cv2
import numpy as np

pygame.init()

# 자동 전체 화면 설정
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width = screen.get_width()
screen_height = screen.get_height()

pygame.display.set_caption("Pygame with Webcam")

# 색상 정의
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BUTTON_COLOR = (100, 100, 100)
BUTTON_HOVER_COLOR = (150, 150, 150)
GRID_COLOR = (0, 255, 0)
HIGHLIGHT_COLOR = (255, 0, 0, 100) # 하이라이트 색상 (R, G, B, Alpha)

# 한글 폰트
try:
    font = pygame.font.Font("C:/Windows/Fonts/malgun.ttf", 40)
    countdown_font = pygame.font.Font("C:/Windows/Fonts/malgun.ttf", 200)
except FileNotFoundError:
    font = pygame.font.Font(None, 50)
    countdown_font = pygame.font.Font(None, 250)

# ==========================
# 이미지 버튼 클래스
# ==========================
class ImageButton:
    def __init__(self, image_path, x, y, width=None, height=None, action=None):
        try:
            self.original_image = pygame.image.load(image_path).convert_alpha()
            self.image = pygame.transform.scale(self.original_image, (width, height)) if width and height else self.original_image
            self.rect = self.image.get_rect(topleft=(x, y))
            self.action = action
        except pygame.error as e:
            print(f"이미지 로드 오류: {image_path} - {e}")
            self.image = pygame.Surface((width, height))
            self.image.fill(BUTTON_COLOR)
            self.rect = self.image.get_rect(topleft=(x, y))
            self.action = action

    def draw(self, screen):
        screen.blit(self.image, self.rect)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(pygame.mouse.get_pos()):
            if self.action:
                self.action()

# ==========================
# 메인 함수
# ==========================
def main():
    screen_state = {"current": "menu"}
    
    countdown_start_time = None
    selected_grid_cell = None # 실시간으로 감지되는 셀 위치
    final_selected_cell = None # 카운트다운 후 확정된 셀 위치

    # 웹캠 초기화
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다.")
        cap = None
        
    # ==========================
    # 버튼 액션 함수
    # ==========================
    def go_to_game(): screen_state["current"] = "game"
    def go_to_menu(): screen_state["current"] = "menu"
    def go_to_info(): screen_state["current"] = "info"
    def go_back():
        nonlocal countdown_start_time, selected_grid_cell, final_selected_cell
        if screen_state["current"] in ["single", "multi", "webcam_view"]:
            screen_state["current"] = "game"
            countdown_start_time = None # 카운트다운 리셋
            selected_grid_cell = None # 실시간 셀 리셋
            final_selected_cell = None # 확정된 셀 리셋
        elif screen_state["current"] == "game":
            screen_state["current"] = "menu"
        elif screen_state["current"] == "info":
             screen_state["current"] = "menu"

    def quit_game(): pygame.quit(); sys.exit()

    # ==========================
    # 버튼 생성
    # ==========================
    button_menu = ImageButton("./image/btn_start.png", screen_width - 500, screen_height - 300, 400, 250, go_to_game)
    button_info = ImageButton("./image/btn_desc.png", screen_width - 200, 100, 100, 100, go_to_info)
    button_back = ImageButton("./image/btn_back.png", 100, 100, 100, 100, go_back)
    button_quit = ImageButton("./image/btn_exit.png", screen_width - 200, 100, 100, 100, go_to_menu)
    
    game_mode = {"mode": None}
    def set_game_mode(mode):
        nonlocal countdown_start_time
        game_mode["mode"] = mode
        if mode == "single":
            screen_state["current"] = "webcam_view"
            countdown_start_time = pygame.time.get_ticks()
        else:
            screen_state["current"] = mode
            
    button_single = ImageButton("./image/btn_single.png", screen_width//2 - 550, screen_height//2 - 100, 550, 600, lambda: set_game_mode("single"))
    button_multi = ImageButton("./image/btn_multi.png", screen_width//2 + 150, screen_height//2 - 100, 550, 600, lambda: set_game_mode("multi"))

    # 배경 영상 및 이미지
    video = cv2.VideoCapture("./image/game_thumbnail.mp4")
    clock = pygame.time.Clock()
    info_bg = pygame.image.load("./image/des_back.jpg").convert()
    info_bg = pygame.transform.scale(info_bg, (screen_width, screen_height))

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

            if screen_state["current"] == "menu":
                button_menu.handle_event(event)
                button_info.handle_event(event)
            elif screen_state["current"] == "game":
                button_back.handle_event(event)
                button_single.handle_event(event)
                button_multi.handle_event(event)
            elif screen_state["current"] == "info":
                button_quit.handle_event(event)
            elif screen_state["current"] in ["webcam_view", "multi"]:
                button_back.handle_event(event)

        # ==========================
        # 화면 그리기
        # ==========================
        if screen_state["current"] in ["menu", "game", "multi"]:
            ret, frame = video.read()
            if not ret:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (screen_width, screen_height))
                frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(frame_surface, (0, 0))

        if screen_state["current"] == "menu":
            button_menu.draw(screen)
            button_info.draw(screen)

        elif screen_state["current"] == "game":
            text_surf = font.render("플레이어 수를 선택하세요", True, WHITE)
            screen.blit(text_surf, (screen_width//2 - text_surf.get_width()//2, screen_height//2 - 200))
            button_single.draw(screen)
            button_multi.draw(screen)
            button_back.draw(screen)

        elif screen_state["current"] == "webcam_view":
            if cap:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    
                    is_countdown_active = countdown_start_time is not None and (pygame.time.get_ticks() - countdown_start_time) < 3000
                    
                    # 카운트다운 중에만 실시간으로 빨간색 감지
                    if is_countdown_active:
                        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                        
                        lower_red1 = np.array([0, 120, 70])
                        upper_red1 = np.array([10, 255, 255])
                        lower_red2 = np.array([170, 120, 70])
                        upper_red2 = np.array([180, 255, 255])
                        
                        mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
                        mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
                        mask = mask1 + mask2
                        
                        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                        
                        if contours:
                            largest_contour = max(contours, key=cv2.contourArea)
                            if cv2.contourArea(largest_contour) > 500:
                                x, y, w, h = cv2.boundingRect(largest_contour)
                                center_x, center_y = x + w // 2, y + h // 2
                                
                                cam_h, cam_w, _ = frame.shape
                                col = int(center_x / (cam_w / 5))
                                row = int(center_y / (cam_h / 2))
                                selected_grid_cell = (row, col)
                        else:
                            selected_grid_cell = None
                    
                    # Pygame Surface로 변환하여 그리기
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_resized = cv2.resize(frame_rgb, (screen_width, screen_height))
                    frame_surface = pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1))
                    screen.blit(frame_surface, (0, 0))
                    
                    # 그리드 그리기
                    pygame.draw.line(screen, GRID_COLOR, (0, screen_height // 2), (screen_width, screen_height // 2), 2)
                    for i in range(1, 5):
                        x = i * (screen_width // 5)
                        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, screen_height), 2)
                    
                    # --- 수정된 카운트다운 및 하이라이트 로직 ---
                    if countdown_start_time is not None:
                        elapsed_time = pygame.time.get_ticks() - countdown_start_time
                        
                        if elapsed_time < 3000: # 카운트다운 진행 중
                            countdown_num = 3 - (elapsed_time // 1000)
                            text_surf = countdown_font.render(str(countdown_num), True, WHITE)
                            text_rect = text_surf.get_rect(center=(screen_width // 2, screen_height // 2))
                            screen.blit(text_surf, text_rect)
                        else: # 카운트다운 종료
                            # 종료되는 시점에 단 한 번만 final_selected_cell에 값을 저장
                            if final_selected_cell is None:
                                final_selected_cell = selected_grid_cell
                                if final_selected_cell is not None:
                                    row, col = final_selected_cell
                                    print(f"FPGA Signal Triggered: Red object detected at grid cell ({row}, {col})")

                            # 타이머 종료 (이 블록이 다시 실행되지 않도록)
                            countdown_start_time = None
                    
                    # 최종 선택된 셀을 붉게 칠함 (카운트다운이 끝난 후)
                    if final_selected_cell is not None:
                        row, col = final_selected_cell
                        cell_width = screen_width / 5
                        cell_height = screen_height / 2
                        
                        highlight_surface = pygame.Surface((cell_width, cell_height), pygame.SRCALPHA)
                        highlight_surface.fill(HIGHLIGHT_COLOR)
                        screen.blit(highlight_surface, (col * cell_width, row * cell_height))
                else:
                    screen.fill(BLACK)
                    err_text = font.render("Webcam frame could not be read.", True, WHITE)
                    screen.blit(err_text, (50, 50))
            else:
                screen.fill(BLACK)
                err_text = font.render("Webcam not found.", True, WHITE)
                screen.blit(err_text, (50, 50))
            
            button_back.draw(screen)

        elif screen_state["current"] == "multi":
            text_surf = font.render(f"Multi 플레이 화면!", True, WHITE)
            screen.blit(text_surf, (screen_width//2 - text_surf.get_width()//2, 300))
            button_back.draw(screen)

        elif screen_state["current"] == "info":
            screen.blit(info_bg, (0, 0))
            text_lines = ["게임 설명:", "- 화살표 키로 움직입니다.", "- 장애물을 피하세요.", "- 점수를 모으세요."]
            for i, line in enumerate(text_lines):
                screen.blit(font.render(line, True, BLACK), (200, 200 + i * 60))
            button_quit.draw(screen)

        pygame.display.flip()
        clock.tick(30)

    if cap:
        cap.release()
    video.release()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()

