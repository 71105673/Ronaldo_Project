import pygame
import imageio
import sys
import os

# --- 초기 설정 ---
pygame.init()

# 화면 크기 설정
screen_width = 1280
screen_height = 720
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("FC Online Style GUI")

# 색상 정의
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (150, 150, 150)
HOVER_COLOR = (200, 200, 200)

# 폰트 설정
try:
    font_title = pygame.font.SysFont('malgungothic', 80, bold=True)
    font_large = pygame.font.SysFont('malgungothic', 40)
    font_medium = pygame.font.SysFont('malgungothic', 30)
    font_small = pygame.font.SysFont('arial', 25)
except pygame.error:
    print("오류: 지정된 폰트를 찾을 수 없습니다. 기본 폰트로 대체합니다.")
    font_title = pygame.font.Font(None, 90)
    font_large = pygame.font.Font(None, 50)
    font_medium = pygame.font.Font(None, 40)
    font_small = pygame.font.Font(None, 30)

# --- 파일 경로 설정 ---
script_dir = os.path.dirname(__file__)
image_dir = os.path.join(script_dir, "../image")

video_path = os.path.join(image_dir, "game_thumbnail.mp4")
ground_image_path = os.path.join(image_dir, "Ground.jpg")

# --- 게임 상태 정의 ---
GAME_STATE_START_SCREEN = 0
GAME_STATE_MAIN_MENU = 1
GAME_STATE_1PLAYER = 2
GAME_STATE_2PLAYER = 3
GAME_STATE_DESCRIPTION = 4 # ★ 게임설명 상태 추가

current_game_state = GAME_STATE_START_SCREEN

# --- 배경 (비디오 및 이미지) 로드 ---
video_frames = []
try:
    vid_reader = imageio.get_reader(video_path, loop=True)
    video_frames = [pygame.image.fromstring(frame.tobytes(), frame.shape[1::-1], "RGB") for frame in vid_reader]
    video_frames = [pygame.transform.scale(frame, (screen_width, screen_height)) for frame in video_frames]
    video_frame_count = len(video_frames)
except FileNotFoundError:
    print(f"오류: 시작 비디오 '{video_path}' 파일을 찾을 수 없습니다.")
    sys.exit()
except Exception as e:
    print(f"시작 비디오 파일을 불러오는 중 오류가 발생했습니다: {e}")
    sys.exit()

main_menu_bg = None
try:
    main_menu_bg = pygame.image.load(ground_image_path).convert()
    main_menu_bg = pygame.transform.scale(main_menu_bg, (screen_width, screen_height))
except pygame.error:
    print(f"오류: 메뉴 배경 이미지 '{ground_image_path}'를 찾을 수 없거나 로드할 수 없습니다.")
    sys.exit()

# --- 시작 화면 텍스트 설정 ---
start_text_kor_surf = font_medium.render("게임을 시작하려면 아무 키나 누르세요", True, WHITE)
start_text_eng_surf = font_small.render("PRESS ANY KEY", True, WHITE)

start_text_kor_rect = start_text_kor_surf.get_rect(center=(screen_width / 2, screen_height * 0.8))
start_text_eng_rect = start_text_eng_surf.get_rect(center=(screen_width / 2, start_text_kor_rect.bottom + 20))

# --- 제목 텍스트 설정 ---
title_surf = font_title.render("FC반도체", True, WHITE)
title_rect = title_surf.get_rect(center=(screen_width / 2, 120))

# --- 메뉴 버튼 클래스 ---
class MenuButton:
    def __init__(self, text, x, y, width, height, font, action=None):
        self.text = text
        self.rect = pygame.Rect(x, y, width, height)
        self.font = font
        self.action = action
        self.text_surf = self.font.render(self.text, True, WHITE)
        self.text_rect = self.text_surf.get_rect(center=self.rect.center)

    def draw(self, surface):
        mouse_pos = pygame.mouse.get_pos()
        is_hover = self.rect.collidepoint(mouse_pos)

        color = HOVER_COLOR if is_hover else WHITE
        self.text_surf = self.font.render(self.text, True, color)
        surface.blit(self.text_surf, self.text_rect)

        line_start = (self.rect.left, self.rect.bottom)
        line_end = (self.rect.right, self.rect.bottom)
        pygame.draw.line(surface, WHITE, line_start, line_end, 1)

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                if self.action:
                    self.action()
                    return True
        return False

# --- 메뉴 액션 함수 ---
def go_to_1player():
    global current_game_state
    current_game_state = GAME_STATE_1PLAYER
    print("1인 플레이 시작!")

def go_to_2player():
    global current_game_state
    current_game_state = GAME_STATE_2PLAYER
    print("2인 플레이 시작!")

# ★ 게임설명 화면으로 이동하는 함수 추가
def show_description():
    global current_game_state
    current_game_state = GAME_STATE_DESCRIPTION
    print("게임설명 화면으로 이동!")

# --- 메뉴 버튼 생성 (메인 메뉴 화면용) ---
menu_button_width = 300
menu_button_height = 50
menu_padding_top = 250
menu_margin_left = 50
menu_item_spacing = 60

menu_buttons = [
    MenuButton("1인 플레이", menu_margin_left, menu_padding_top,
               menu_button_width, menu_button_height, font_large, go_to_1player),
    MenuButton("2인 플레이", menu_margin_left, menu_padding_top + menu_item_spacing,
               menu_button_width, menu_button_height, font_large, go_to_2player),
    # ★ 게임설명 버튼 추가
    MenuButton("게임설명", menu_margin_left, menu_padding_top + menu_item_spacing * 2,
               menu_button_width, menu_button_height, font_large, show_description),
]


# --- 메인 게임 루프 ---
running = True
video_current_frame_idx = 0
clock = pygame.time.Clock()
FPS = 30

while running:
    clock.tick(FPS)

    # --- 이벤트 처리 ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        if current_game_state == GAME_STATE_START_SCREEN:
            if event.type == pygame.KEYDOWN or (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1):
                current_game_state = GAME_STATE_MAIN_MENU
        elif current_game_state == GAME_STATE_MAIN_MENU:
            for button in menu_buttons:
                button.handle_event(event)

    # --- 화면 그리기 ---
    if current_game_state == GAME_STATE_START_SCREEN:
        screen.blit(video_frames[video_current_frame_idx], (0, 0))
        video_current_frame_idx = (video_current_frame_idx + 1) % video_frame_count

        screen.blit(start_text_kor_surf, start_text_kor_rect)
        screen.blit(start_text_eng_surf, start_text_eng_rect)

    elif current_game_state == GAME_STATE_MAIN_MENU:
        screen.blit(main_menu_bg, (0, 0))
        screen.blit(title_surf, title_rect)
        for button in menu_buttons:
            button.draw(screen)

    elif current_game_state == GAME_STATE_1PLAYER:
        screen.fill(BLACK)
        text_surf = font_large.render("1인 플레이 화면입니다!", True, WHITE)
        text_rect = text_surf.get_rect(center=(screen_width / 2, screen_height / 2))
        screen.blit(text_surf, text_rect)

    elif current_game_state == GAME_STATE_2PLAYER:
        screen.fill(GRAY)
        text_surf = font_large.render("2인 플레이 화면입니다!", True, WHITE)
        text_rect = text_surf.get_rect(center=(screen_width / 2, screen_height / 2))
        screen.blit(text_surf, text_rect)
        
    # ★ 게임설명 화면 그리기 로직 추가
    elif current_game_state == GAME_STATE_DESCRIPTION:
        screen.fill(BLACK)
        text_surf = font_large.render("게임설명 화면입니다!", True, WHITE)
        text_rect = text_surf.get_rect(center=(screen_width / 2, screen_height / 2))
        screen.blit(text_surf, text_rect)

    pygame.display.flip()

# --- 프로그램 종료 ---
pygame.quit()
sys.exit()