import pygame
import os

# --- 기본 경로 설정 ---
# 이 파일(Config.py)이 있는 위치를 기준으로 경로를 설정합니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# assets 폴더가 프로젝트 폴더 바로 아래에 있다고 가정합니다.
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

FONT_PATH = os.path.join(ASSETS_DIR, "fonts", "netmarbleM.ttf")
FONT_BOLD_PATH = os.path.join(ASSETS_DIR, "fonts", "netmarbleB.ttf")
IMAGE_PATH = os.path.join(ASSETS_DIR, "image")
SOUND_PATH = os.path.join(ASSETS_DIR, "sound")


# --- 화면 설정 ---
pygame.init()
try:
    # 여러 모니터를 합친 전체 크기 감지
    desktop_sizes = pygame.display.get_desktop_sizes()
    SCREEN_WIDTH = sum(w for w, h in desktop_sizes)
    SCREEN_HEIGHT = max(h for w, h in desktop_sizes)
except AttributeError:
    info = pygame.display.Info()
    SCREEN_WIDTH = info.current_w
    SCREEN_HEIGHT = info.current_h

# 모니터 분할 설정
MAIN_MONITOR_WIDTH = SCREEN_WIDTH // 3
GOALKEEPER_MONITOR_WIDTH = SCREEN_WIDTH // 3
ATTACKER_MONITOR_WIDTH = SCREEN_WIDTH - MAIN_MONITOR_WIDTH - GOALKEEPER_MONITOR_WIDTH

MAIN_START_X = 0
ATTACKER_START_X = ATTACKER_MONITOR_WIDTH
GOALKEEPER_START_X = ATTACKER_MONITOR_WIDTH + MAIN_MONITOR_WIDTH

MAIN_MONITOR_CENTER_X = MAIN_START_X + (MAIN_MONITOR_WIDTH // 2)
GOALKEEPER_MONITOR_CENTER_X = GOALKEEPER_START_X + (GOALKEEPER_MONITOR_WIDTH // 2)
ATTACKER_MONITOR_CENTER_X = ATTACKER_START_X + (ATTACKER_MONITOR_WIDTH // 2)


# --- 색상 정의 ---
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRID_COLOR = (0, 255, 0)
RED = (255, 0, 0)
GOLD_COLOR = (255, 215, 0)
HOVER_COLOR = (200, 200, 200)
HIGHLIGHT_COLOR = (255, 0, 0, 100) # (R, G, B, Alpha)
BUTTON_COLOR = (100, 100, 100)


# --- 게임 상수 ---
FPS = 60
TOTAL_CHANCES = 5
COUNTDOWN_SECONDS = 5
TRANSITION_SPEED = 15
GIF_FRAME_DURATION = 70
SYNTHESIZED_GIF_FRAME_DURATION = 90
RESULT_DELAY_MS = 2000 # 결과 판정 후 GIF 재생까지 딜레이
GIF_PLAY_DURATION_MS = 3000 # 결과 GIF 재생 시간


# --- 시리얼 포트 설정 ---
GOALKEEPER_SERIAL_PORT = 'COM17'
ATTACKER_SERIAL_PORT = 'COM13'
BAUD_RATE = 9600


# --- 카메라 설정 ---
GOALKEEPER_CAM_INDEX = 0
ATTACKER_CAM_INDEX = 2

# --- 파일 경로 ---
HIGHSCORE_FILE = "highscore.txt"

# UART 명령어 바이트
UART_COMMANDS = {
    'grid': 225,
    'face': 226,
    'kick': 227
}