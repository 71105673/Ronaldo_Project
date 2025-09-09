# 화면전환
import pygame
import sys
import cv2

pygame.init()

# --- 자동 전체 화면 설정 ---
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width = screen.get_width()
screen_height = screen.get_height()

pygame.display.set_caption("Pygame with Webcam")

# 색상 정의
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BUTTON_COLOR = (100, 100, 100)
GRID_COLOR = (0, 255, 0)

# 한글 폰트
try:
    font = pygame.font.Font("C:/Windows/Fonts/malgun.ttf", 40)
except FileNotFoundError:
    font = pygame.font.Font(None, 50)

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

    # --- 웹캠 초기화 ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 웹캠을 열 수 없습니다.")
        cap = None

    # ### ========================== ###
    # ### 화면 전환 효과 관련 변수 ###
    # ### ========================== ###
    transition_surface = pygame.Surface((screen_width, screen_height))
    transition_surface.fill(BLACK)
    transition_alpha = 0      # 0 (투명) ~ 255 (불투명)
    transition_target = None  # 전환될 목표 상태
    transition_speed = 15     # 전환 속도 (값이 클수록 빠름)
    fading_out = False        # 페이드 아웃 진행 여부
    fading_in = False         # 페이드 인 진행 여부

    # ### ========================== ###
    # ### 화면 전환 시작 함수 ###
    # ### ========================== ###
    def start_transition(target_state):
        nonlocal transition_target, fading_out
        # 다른 전환이 진행 중일 때는 새로 시작하지 않음
        if not fading_out and not fading_in:
            transition_target = target_state
            fading_out = True

    # ==========================
    # 버튼 액션 함수
    # ==========================
    # ### 기존 함수들을 start_transition을 사용하도록 수정 ###
    def go_back():
        current = screen_state["current"]
        target = "menu"  # 기본값
        if current in ["single", "multi", "webcam_view"]:
            target = "game"
        elif current == "game":
            target = "menu"
        elif current == "info":
            target = "menu"
        start_transition(target)

    def set_game_mode(mode):
        game_mode["mode"] = mode
        if mode == "single":
            start_transition("webcam_view")
        else:
            start_transition(mode)

    # ==========================
    # 버튼 생성
    # ==========================
    # ### 버튼 액션을 start_transition 또는 수정된 함수로 변경 ###
    button_menu = ImageButton("./image/btn_start.png", screen_width - 500, screen_height - 300, 400, 250, lambda: start_transition("game"))
    button_info = ImageButton("./image/btn_desc.png", screen_width - 200, 100, 100, 100, lambda: start_transition("info"))
    button_back = ImageButton("./image/btn_back.png", 100, 100, 100, 100, go_back)
    button_quit = ImageButton("./image/btn_exit.png", screen_width - 200, 100, 100, 100, lambda: start_transition("menu"))
    
    game_mode = {"mode": None}
    button_single = ImageButton("./image/btn_single.png", screen_width//2 - 550, screen_height//2 - 100, 550, 600, lambda: set_game_mode("single"))
    button_multi = ImageButton("./image/btn_multi.png", screen_width//2 + 150, screen_height//2 - 100, 550, 600, lambda: set_game_mode("multi"))

    # 배경 영상 및 이미지
    video = cv2.VideoCapture("./image/game_thumbnail.mp4")
    clock = pygame.time.Clock()
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

            # ### 화면 전환 중에는 버튼 입력을 받지 않음 ###
            if not fading_in and not fading_out:
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

        elif screen_state["current"] == "webcam_view":
            if cap:
                ret, frame = cap.read()
                if ret:
                    frame = cv2.flip(frame, 1)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame = cv2.resize(frame, (screen_width, screen_height))
                    frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                    screen.blit(frame_surface, (0, 0))
                    
                    pygame.draw.line(screen, GRID_COLOR, (0, screen_height // 2), (screen_width, screen_height // 2), 2)
                    for i in range(1, 5):
                        x = i * (screen_width // 5)
                        pygame.draw.line(screen, GRID_COLOR, (x, 0), (x, screen_height), 2)
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
            text_surf = font.render("Multi 플레이 화면!", True, WHITE)
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

        # ### ========================== ###
        # ### 화면 전환 효과 처리 ###
        # ### ========================== ###
        if fading_out:
            transition_alpha += transition_speed
            # 알파값이 255를 넘으면 페이드 아웃을 끝내고 페이드 인 시작
            if transition_alpha >= 255:
                transition_alpha = 255
                fading_out = False
                fading_in = True
                # ### 화면이 완전히 어두워졌을 때 실제 상태를 변경 ###
                screen_state["current"] = transition_target
                transition_target = None
            transition_surface.set_alpha(transition_alpha)
            screen.blit(transition_surface, (0, 0))

        elif fading_in:
            transition_alpha -= transition_speed
            # 알파값이 0보다 작아지면 페이드 인 종료
            if transition_alpha <= 0:
                transition_alpha = 0
                fading_in = False
            transition_surface.set_alpha(transition_alpha)
            screen.blit(transition_surface, (0, 0))

        pygame.display.flip()
        clock.tick(60) 

    # 자원 해제
    if cap:
        cap.release()
    video.release()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()