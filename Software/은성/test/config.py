import pygame
import os
import cv2

# =========================================
# 초기 설정
# =========================================
pygame.init()
pygame.mixer.init()

# 전체 화면 크기 감지
try:
    desktop_sizes = pygame.display.get_desktop_sizes()
    total_width = sum(w for w, h in desktop_sizes)
    max_height = max(h for w, h in desktop_sizes)
except AttributeError:
    info = pygame.display.Info()
    total_width = info.current_w
    max_height = info.current_h

os.environ['SDL_VIDEO_WINDOW_POS'] = "0,0"
screen = pygame.display.set_mode((total_width, max_height), pygame.NOFRAME)

screen_width = screen.get_width()
screen_height = screen.get_height()
pygame.display.set_caption("Penalty Kick Challenge")

# ================================================================= #
# *** 3-모니터 레이아웃 설정 (왼쪽 골키퍼 | 중앙 메인 | 오른쪽 공격수) *** #
# ================================================================= #
goalkeeper_monitor_width = screen_width // 3
main_monitor_width = screen_width // 3
attacker_monitor_width = screen_width - goalkeeper_monitor_width - main_monitor_width

main_start_x = 0
attacker_start_x = attacker_monitor_width
goalkeeper_start_x = goalkeeper_monitor_width + main_monitor_width

goalkeeper_monitor_center_x = goalkeeper_start_x + (goalkeeper_monitor_width // 2)
main_monitor_center_x = main_start_x + (main_monitor_width // 2)
attacker_monitor_center_x = attacker_start_x + (attacker_monitor_width // 2)

# =========================================
# 색상 및 폰트
# =========================================
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRID_COLOR = (0, 255, 0)
RED = (255, 0, 0)
HIGHLIGHT_COLOR = (255, 0, 0, 100)
GOLD_COLOR = (255, 215, 0)

def load_font(path, size, default_size):
    try:
        return pygame.font.Font(path, size)
    except (pygame.error, FileNotFoundError):
        print(f"폰트 로드 실패: {path}. 기본 폰트를 사용합니다.")
        return pygame.font.Font(None, default_size)

font = load_font("../fonts/netmarbleM.ttf", 40, 50)
small_font = load_font("../fonts/netmarbleM.ttf", 30, 40)
description_font = load_font("../fonts/netmarbleM.ttf", 50, 60)
title_font = load_font("../fonts/netmarbleB.ttf", 120, 130)
countdown_font = load_font("../fonts/netmarbleM.ttf", 200, 250)
score_font = load_font("../fonts/netmarbleB.ttf", 60, 70)
rank_font = load_font("../fonts/netmarbleB.ttf", 100, 110)

# =========================================
# 게임 플레이 상수
# =========================================
CHANCES_PER_GAME = 5
COUNTDOWN_SECONDS = 5
COUNTDOWN_MS = COUNTDOWN_SECONDS * 1000
RESULT_DELAY_MS = 2000          # 결과 판정 후 대기 시간
GIF_SHOW_DURATION_MS = 3000     # 결과 GIF 재생 시간
GIF_FRAME_DURATION_MS = 70      # 결과 GIF 프레임 간격

# =========================================
# 유틸리티 함수
# =========================================
def load_highscore():
    if not os.path.exists("highscore.txt"):
        return 0
    try:
        with open("highscore.txt", "r") as f:
            return int(f.read())
    except (IOError, ValueError):
        return 0

def save_highscore(new_score):
    try:
        with open("highscore.txt", "w") as f:
            f.write(str(new_score))
    except IOError as e:
        print(f"최고 기록 저장 오류: {e}")

def get_scaled_rect(original_w, original_h, container_w, container_h):
    if original_h == 0 or container_h == 0:
        return (0, 0)
    aspect_ratio = original_w / original_h
    container_aspect_ratio = container_w / container_h
    if aspect_ratio > container_aspect_ratio:
        new_w, new_h = container_w, int(container_w / aspect_ratio)
    else:
        new_w, new_h = int(container_h * aspect_ratio), container_h
    return new_w, new_h

def load_gif_frames(video_path, size):
    frames = []
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video file: {video_path}")
        return frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_resized = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        frame_pygame = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
        frames.append(frame_pygame)

    cap.release()
    print(f"Loaded {len(frames)} frames from {video_path}.")
    return frames