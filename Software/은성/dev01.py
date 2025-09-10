import pygame
import sys
import cv2
import time

pygame.init()

# 화면 설정
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width, screen_height = screen.get_width(), screen.get_height()
pygame.display.set_caption("축구 PK 게임")

# 색상 정의
BLACK = (0, 0, 0)
BUTTON_COLOR = (100, 100, 100)
BUTTON_HOVER_COLOR = (150, 150, 150)

# 한글 폰트
font = pygame.font.Font("C:/Windows/Fonts/malgun.ttf", 40)

# ==========================
# 버튼 클래스
# ==========================
class ImageButton:
    def __init__(self, image_path, x, y, width=None, height=None, action=None):
        self.original_image = pygame.image.load(image_path).convert_alpha()
        self.image = pygame.transform.scale(self.original_image, (width, height)) if width and height else self.original_image
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
    game_mode = {"mode": None}

    # 멀티 상태용 변수
    multi_start_time = None
    countdown_done = False
    round_count = 0
    max_rounds = 5

    # ==========================
    # 버튼 액션 함수
    # ==========================
    def go_to_game(): 
        nonlocal multi_start_time, countdown_done
        # 게임 선택 화면으로 돌아가면 카운트다운 초기화해두는게 안전
        multi_start_time, countdown_done = None, False
        screen_state["current"] = "game"

    def go_to_menu():
        nonlocal round_count, multi_start_time, countdown_done
        # 메뉴로 완전히 나갈 때는 라운드 초기화
        round_count = 0
        multi_start_time, countdown_done = None, False
        game_mode["mode"] = None
        screen_state["current"] = "menu"

    def go_to_info(): 
        screen_state["current"] = "info"

    def go_back():
        # 여기서 round_count도 실제로 초기화하려면 nonlocal로 선언해야 합니다.
        nonlocal multi_start_time, countdown_done, round_count
        if screen_state["current"] in ["single", "multi", "wait"]:
            # single/multi/wait 상태에서 뒤로가면 'game' 선택 화면으로
            screen_state["current"] = "game"
            multi_start_time, countdown_done = None, False
            round_count = 0  # <<< 라운드 초기화 (이전에는 nonlocal 누락으로 효력이 없었음)
            game_mode["mode"] = None
        elif screen_state["current"] == "game":
            # game에서 뒤로가면 메뉴로
            screen_state["current"] = "menu"
            # 필요하면 여기도 초기화
            multi_start_time, countdown_done = None, False
            round_count = 0
            game_mode["mode"] = None

    def set_game_mode(mode):
        nonlocal multi_start_time, countdown_done, round_count
        game_mode["mode"] = mode
        screen_state["current"] = mode
        if mode == "multi":
            # multi로 들어갈 때 카운트다운 시작 (새 라운드 준비)
            multi_start_time = time.time()
            countdown_done = False
            # round_count는 라운드 진행 로직에서 증가하므로 여기서는 증가시키지 않음

    # ==========================
    # 버튼 생성
    # ==========================
    button_menu = ImageButton("./image/btn_start.png", screen_width - 500, screen_height - 300, 400, 250, go_to_game)
    button_info = ImageButton("./image/btn_desc.png", screen_width - 200, 100, 100, 100, go_to_info)
    button_back = ImageButton("./image/btn_back.png", 100, 100, 100, 100, go_back)
    button_quit = ImageButton("./image/btn_exit.png", screen_width - 200, 100, 100, 100, go_to_menu)
    button_single = ImageButton("./image/btn_single.png", screen_width//2 - 550, screen_height//2 - 100, 550, 600, lambda: set_game_mode("single"))
    button_multi = ImageButton("./image/btn_multi.png", screen_width//2 + 150, screen_height//2 - 100, 550, 600, lambda: set_game_mode("multi"))

    # ==========================
    # 영상 (메뉴/게임/싱글/멀티 배경)
    # ==========================
    video = cv2.VideoCapture("./image/game_thumbnail.mp4")
    webcam = cv2.VideoCapture(0)
    clock = pygame.time.Clock()

    # ==========================
    # 게임 설명 배경
    # ==========================
    info_bg = pygame.image.load("./image/des_back.jpg").convert()
    info_bg = pygame.transform.scale(info_bg, (screen_width, screen_height))

    # ==========================
    # 배경 그리기 함수
    # ==========================
    def draw_background():
        if screen_state["current"] == "multi":
            ret, frame = webcam.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (screen_width, screen_height))
                frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
                screen.blit(frame_surface, (0, 0))
            else:
                screen.fill((50, 50, 50))

            # --- 5칸 그리드 ---
            cell_w, cell_h = screen_width // 5, screen_height // 5
            for i in range(1, 5):
                pygame.draw.line(screen, (0, 255, 0), (i * cell_w, 0), (i * cell_w, screen_height), 2)
                pygame.draw.line(screen, (0, 255, 0), (0, i * cell_h), (screen_width, i * cell_h), 2)

        elif screen_state["current"] in ["menu", "game", "single"]:
            ret, frame = video.read()
            if not ret:
                video.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = video.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (screen_width, screen_height))
            frame_surface = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            screen.blit(frame_surface, (0, 0))

    # ==========================
    # 메인 루프
    # ==========================
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE: running = False

            # wait 상태에서 스페이스 키 입력 → 다음 라운드 시작
            if screen_state["current"] == "wait" and event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if round_count < max_rounds:
                    screen_state["current"] = "multi"
                    multi_start_time = time.time()
                    countdown_done = False
                else:
                    screen_state["current"] = "winner"

            # 버튼 이벤트 처리
            if screen_state["current"] == "menu":
                button_menu.handle_event(event)
                button_info.handle_event(event)
            elif screen_state["current"] == "game":
                button_back.handle_event(event)
                button_single.handle_event(event)
                button_multi.handle_event(event)
            elif screen_state["current"] == "info":
                button_quit.handle_event(event)
            elif screen_state["current"] in ["single", "multi", "wait"]:
                button_back.handle_event(event)

        # --- 배경 ---
        if screen_state["current"] != "info":
            draw_background()

        # --- 상태별 화면 ---
        if screen_state["current"] == "menu":
            button_menu.draw(screen)
            button_info.draw(screen)

        elif screen_state["current"] == "game":
            text_surf = font.render("플레이어 수를 선택하세요", True, BLACK)
            screen.blit(text_surf, (screen_width//2 - 250, screen_height//2 - 200))
            button_single.draw(screen)
            button_multi.draw(screen)
            button_back.draw(screen)

        elif screen_state["current"] == "single":
            screen.blit(font.render("Single 플레이 화면!", True, BLACK), (500, 300))
            button_back.draw(screen)

        elif screen_state["current"] == "multi":
            # --- 카운트다운 ---
            if not countdown_done:
                elapsed = time.time() - multi_start_time
                remaining = 5 - int(elapsed) 
                if remaining > 0:
                    countdown_text = font.render(str(remaining), True, (255, 0, 0))
                    text_rect = countdown_text.get_rect(center=(screen_width//2, screen_height//2))
                    screen.blit(countdown_text, text_rect)
                else:
                    countdown_done = True
                    # 카운트다운 끝난 시점 초기화(게임 시작 기준 시간)
                    # multi_start_time를 재설정하면 라운드 경과 측정이 더 명확합니다.
                    multi_start_time = time.time()
            else:
                screen.blit(font.render(f"Multi 플레이 {round_count+1} 라운드!", True, BLACK), (500, 300))

                # TODO: 실제 PK 판정 로직 넣기
                # 여기서는 임시로 5초 경과하면 라운드 종료 처리
                if time.time() - multi_start_time > 5:
                    round_count += 1
                    if round_count < max_rounds:
                        screen_state["current"] = "wait"
                    else:
                        screen_state["current"] = "winner"

            button_back.draw(screen)

        elif screen_state["current"] == "wait":
            screen.fill((200, 200, 200))
            text = font.render(f"{round_count}/{max_rounds} 라운드 완료!", True, BLACK)
            screen.blit(text, (screen_width//2 - 200, screen_height//2 - 100))
            text2 = font.render("스페이스 키를 눌러 다음 라운드 시작", True, BLACK)
            screen.blit(text2, (screen_width//2 - 300, screen_height//2))
            button_back.draw(screen)

        elif screen_state["current"] == "winner":
            screen.fill((255, 220, 220))
            text = font.render("게임 종료! 승자를 확인하세요!", True, BLACK)
            screen.blit(text, (screen_width//2 - 200, screen_height//2 - 50))
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

    video.release()
    webcam.release()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()
