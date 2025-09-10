import pygame
import sys
import cv2

pygame.init()

# --- 수정된 부분: 자동 전체 화면 설정 ---
# (0, 0)과 FULLSCREEN 플래그를 사용해 현재 해상도에 맞는 전체 화면을 생성합니다.
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
# 생성된 화면의 너비와 높이 정보를 가져와 변수에 저장합니다.
screen_width = screen.get_width()
screen_height = screen.get_height()
# ------------------------------------

pygame.display.set_caption("Pygame with Webcam")

# 색상 정의
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BUTTON_COLOR = (100, 100, 100)
BUTTON_HOVER_COLOR = (150, 150, 150)
GRID_COLOR = (0, 255, 0) # 격자 색상 정의

# 한글 폰트 (경로가 다를 경우 수정 필요)
try:
    font = pygame.font.Font("C:/Windows/Fonts/malgun.ttf", 40)
except FileNotFoundError:
    font = pygame.font.Font(None, 50) # 대체 폰트

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

    # --- 추가: 웹캠 초기화 ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다.")
        # 웹캠 없이도 실행되도록 처리
        cap = None
        
    # ==========================
    # 버튼 액션 함수
    # ==========================
    def go_to_game(): screen_state["current"] = "game"
    def go_to_menu(): screen_state["current"] = "menu"
    def go_to_info(): screen_state["current"] = "info"
    def go_back():
        # --- 수정: webcam_view 상태에서 뒤로가기 추가 ---
        if screen_state["current"] in ["single", "multi", "webcam_view"]:
            screen_state["current"] = "game"
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
    
    # 플레이어 선택 버튼
    game_mode = {"mode": None}
    def set_game_mode(mode):
        game_mode["mode"] = mode
        # --- 수정: 'single'일 때 'webcam_view'로 상태 변경 ---
        if mode == "single":
            screen_state["current"] = "webcam_view"
        else:
            screen_state["current"] = mode
            
    button_single = ImageButton("./image/btn_single.png", screen_width//2 - 550, screen_height//2 - 100, 550, 600, lambda: set_game_mode("single"))
    button_multi = ImageButton("./image/btn_multi.png", screen_width//2 + 150, screen_height//2 - 100, 550, 600, lambda: set_game_mode("multi"))

    # 배경 영상
    video = cv2.VideoCapture("./image/game_thumbnail.mp4")
    clock = pygame.time.Clock()

    # 게임 설명 배경
    info_bg = pygame.image.load("./image/des_back.jpg").convert()
    info_bg = pygame.transform.scale(info_bg, (screen_width, screen_height))

    running = True
    while running:
        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

            # 버튼 이벤트 처리 (상태에 따라)
            if screen_state["current"] == "menu":
                button_menu.handle_event(event)
                button_info.handle_event(event)
            elif screen_state["current"] == "game":
                button_back.handle_event(event)
                button_single.handle_event(event)
                button_multi.handle_event(event)
            elif screen_state["current"] == "info":
                button_quit.handle_event(event)
            # --- 수정: 'webcam_view'와 'multi' 상태의 이벤트 처리 통합 ---
            elif screen_state["current"] in ["webcam_view", "multi"]:
                button_back.handle_event(event)

        # ==========================
        # 화면 그리기
        # ==========================
        # 공통 영상 배경
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

        # --- 상태별 화면 ---
        if screen_state["current"] == "menu":
            button_menu.draw(screen)
            button_info.draw(screen)

        elif screen_state["current"] == "game":
            text_surf = font.render("플레이어 수를 선택하세요", True, WHITE)
            screen.blit(text_surf, (screen_width//2 - text_surf.get_width()//2, screen_height//2 - 200))
            button_single.draw(screen)
            button_multi.draw(screen)
            button_back.draw(screen)

        # --- 추가: 웹캠 뷰 그리기 ---
        elif screen_state["current"] == "webcam_view":
            if cap:
                ret, frame = cap.read()
                if ret:
                    # 좌우 반전
                    frame = cv2.flip(frame, 1)
                    
                    # Pygame Surface로 변환하여 배경으로 그리기
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (screen_width, screen_height))
                    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    screen.blit(frame_surface, (0, 0))
                    
                    # --- 수정: Pygame을 사용하여 화면에 직접 격자 그리기 ---
                    # 가로선 (세로를 2등분)
                    pygame.draw.line(screen, GRID_COLOR, (0, screen_height // 2), (screen_width, screen_height // 2), 2)
                    # 세로선 (가로를 5등분)
                    for i in range(1, 5):
                        x = i * (screen_width // 5)
                        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, screen_height), 2)
                else:
                    # 웹캠 프레임 읽기 실패 시
                    screen.fill(BLACK)
                    err_text = font.render("Webcam frame could not be read.", True, WHITE)
                    screen.blit(err_text, (50, 50))
            else:
                # 웹캠이 없을 경우
                screen.fill(BLACK)
                err_text = font.render("Webcam not found.", True, WHITE)
                screen.blit(err_text, (50, 50))
            
            # 뒤로가기 버튼은 항상 위에 그림
            button_back.draw(screen)

        elif screen_state["current"] == "multi":
            text_surf = font.render(f"Multi 플레이 화면!", True, WHITE)
            screen.blit(text_surf, (screen_width//2 - text_surf.get_width()//2, 300))
            button_back.draw(screen)

        elif screen_state["current"] == "info":
            screen.blit(info_bg, (0, 0))
            text_lines = [
                "게임 설명:",
                "- 화살표 키로 움직입니다.",
                "- 장애물을 피하세요.",
                "- 점수를 모으세요."
            ]
            for i, line in enumerate(text_lines):
                screen.blit(font.render(line, True, BLACK), (200, 200 + i * 60))
            button_quit.draw(screen)

        pygame.display.flip()
        clock.tick(30)

    # --- 추가: 자원 해제 ---
    if cap:
        cap.release()
    video.release()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()