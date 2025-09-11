# main.py

import pygame
import sys
import cv2
import serial

from score_manager import load_highscore
import game_logic

pygame.init()
pygame.mixer.init()

# 화면 설정
screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)
screen_width = screen.get_width()
screen_height = screen.get_height()
pygame.display.set_caption("Penalty Kick Challenge")

# 색상 및 폰트
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
BUTTON_COLOR = (100, 100, 100)
GOLD_COLOR = (255, 215, 0)
try:
    font = pygame.font.Font("./fonts/netmarbleM.ttf", 40)
    description_font = pygame.font.Font("./fonts/netmarbleM.ttf", 50)
    title_font = pygame.font.Font("./fonts/netmarbleB.ttf", 120)
except FileNotFoundError:
    font = pygame.font.Font(None, 50)
    description_font = pygame.font.Font(None, 60)
    title_font = pygame.font.Font(None, 130)

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
    clock = pygame.time.Clock()

    screen_state = {
        'current': 'menu', 'highscore': load_highscore(),
        'final_rank': '', 'end_video_to_play': None, 'mode': None
    }
    game_logic.reset_game_state(screen_state, full_reset=True)

    transition_surface = pygame.Surface((screen_width, screen_height)); transition_surface.fill(BLACK)
    transition_alpha, transition_target, transition_speed = 0, None, 15
    fading_out, fading_in = False, False

    assets = {}
    try:
        assets['button_sound'] = pygame.mixer.Sound("./sound/button_click.wav")
        assets['siu_sound'] = pygame.mixer.Sound("./sound/SIUUUUU.wav")
        assets['success_sound'] = pygame.mixer.Sound("./sound/SIUUUUU.wav")
        ball_img = pygame.image.load("./image/final_ronaldo/Ball.png").convert_alpha()
        assets['scoreboard_ball_image'] = pygame.transform.scale(ball_img, (80, 80))
        assets['ball_image'] = pygame.transform.scale(ball_img, (200, 200))
        assets['info_bg'] = pygame.transform.scale(pygame.image.load("./image/info/info_back2.jpg").convert(), (screen_width, screen_height))
        assets['countdown_font'] = pygame.font.Font("./fonts/netmarbleM.ttf", 200)
        assets['score_font'] = pygame.font.Font("./fonts/netmarbleB.ttf", 60)
        assets['rank_font'] = pygame.font.Font("./fonts/netmarbleB.ttf", 100)
        assets['main_video'] = cv2.VideoCapture("./image/game_thumbnail.mp4")
        assets['failure_gif'] = cv2.VideoCapture("./image/G.O.A.T/siuuu.gif")
        assets['success_gif'] = cv2.VideoCapture("./image/final_ronaldo/pk.gif")
        assets['victory_video'] = cv2.VideoCapture("./image/victory.gif")
        assets['defeat_video'] = cv2.VideoCapture("./image/defeat.gif")
    except Exception as e:
        print(f"리소스 로드 중 오류 발생: {e}"); pygame.quit(); sys.exit()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened(): print("오류: 웹캠을 열 수 없습니다."); cap = None
    ser = None
    try:
        ser = serial.Serial('COM13', 9600, timeout=0)
        print("Basys3 보드가 성공적으로 연결되었습니다.")
    except serial.SerialException as e: print(f"오류: 시리얼 포트를 열 수 없습니다 - {e}")

    def start_transition(target_state):
        nonlocal transition_target, fading_out
        if not fading_out and not fading_in: transition_target, fading_out = target_state, True

    buttons = {
        "menu": [ImageButton("./image/btn_start.png", screen_width - 300, screen_height - 175, 400, 250, lambda: start_transition("game"), sound=assets['button_sound']),
                 ImageButton("./image/btn_desc.png", screen_width - 150, 150, 100, 100, lambda: start_transition("info"), sound=assets['button_sound'])],
        "game": [ImageButton("./image/btn_single.png", screen_width//2 - 280, screen_height//2 + 200, 550, 600, lambda: game_logic.set_game_mode(screen_state, "single", assets['siu_sound'], start_transition)),
                 ImageButton("./image/btn_multi.png", screen_width//2 + 430, screen_height//2 + 200, 550, 600, lambda: game_logic.set_game_mode(screen_state, "multi", assets['siu_sound'], start_transition)),
                 ImageButton("./image/btn_back.png", 150, 150, 100, 100, lambda: game_logic.go_to_menu(screen_state, start_transition), sound=assets['button_sound'])],
        "webcam_view": [ImageButton("./image/btn_back.png", 150, 150, 100, 100, lambda: game_logic.go_to_menu(screen_state, start_transition), sound=assets['button_sound'])],
        "multi": [ImageButton("./image/btn_back.png", 150, 150, 100, 100, lambda: game_logic.go_to_menu(screen_state, start_transition), sound=assets['button_sound'])],
        "info": [ImageButton("./image/btn_exit.png", screen_width - 150, 150, 100, 100, lambda: game_logic.go_to_menu(screen_state, start_transition), sound=assets['button_sound'])],
        "end": [ImageButton("./image/btn_restart.png", screen_width//2 - 300, screen_height - 250, 400, 250, lambda: game_logic.restart_game(screen_state, start_transition), sound=assets['button_sound']),
                ImageButton("./image/btn_main_menu.png", screen_width//2 + 300, screen_height - 250, 400, 250, lambda: game_logic.go_to_menu(screen_state, start_transition), sound=assets['button_sound'])]
    }

    loop_counter = 0
    # ==========================
    # 메인 루프 (사용자 요청 버전)
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
            ret, frame = assets['main_video'].read()
            if not ret: assets['main_video'].set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = assets['main_video'].read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame = cv2.resize(frame, (screen_width, screen_height))
                screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))

        if screen_state["current"] == "game":
            text_surf = font.render("플레이어 수를 선택하세요", True, WHITE)
            screen.blit(text_surf, (screen_width//2 - text_surf.get_width()//2, screen_height//2 - 200))

        elif screen_state["current"] == "webcam_view" or screen_state["current"] == "multi":
            # 분리된 함수 호출 (모든 필요한 인자 전달)
            game_logic.handle_game_logic(screen, screen_state, assets, cap, ser, start_transition, loop_counter)

        elif screen_state["current"] == "info":
            screen.blit(assets['info_bg'], (0, 0))
            title_surf = title_font.render("게임 방법", True, WHITE)
            screen.blit(title_surf, (screen_width / 2 - title_surf.get_width() / 2, 150))
            text_lines_1p = ["[1인 플레이]", "1. 5초의 카운트 다운이 시작됩니다.", "2. 카메라에 비치는 빨간색", "   물체를 인식합니다.", "3. 5개의 영역 중 하나를 선택합니다.", "4. 공을 막으면 성공!"]
            text_lines_2p = ["[2인 플레이]", "1. COM과 번갈아가며 공격과", "   수비를 합니다.", "2. 5번의 기회가 주어집니다.", "3. 더 많은 득점을 한 플레이어가", "   승리합니다."] # 사용자 요청 텍스트
            x_offset_1p, x_offset_2p, y_start = screen_width / 4 - 150, screen_width * 3 / 4 - 300, 400
            for i, line in enumerate(text_lines_1p): screen.blit(description_font.render(line, True, WHITE), (x_offset_1p, y_start + i*75))
            for i, line in enumerate(text_lines_2p): screen.blit(description_font.render(line, True, WHITE), (x_offset_2p, y_start + i*75))

        elif screen_state["current"] == "end":
            end_video = screen_state.get('end_video_to_play')
            if end_video:
                ret, frame = end_video.read()
                if not ret: end_video.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = end_video.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame = cv2.resize(frame, (screen_width, screen_height))
                    screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
            else:
                screen.fill(BLACK)
            
            # 랭크, 점수 등 표시
            rank_surf = assets['rank_font'].render(screen_state['final_rank'], True, GOLD_COLOR)
            screen.blit(rank_surf, rank_surf.get_rect(center=(screen_width/2, screen_height/2 - 150)))
            score_surf = assets['score_font'].render(f"FINAL SCORE: {screen_state['score']}", True, WHITE)
            screen.blit(score_surf, score_surf.get_rect(center=(screen_width/2, screen_height/2)))
            highscore_surf = assets['score_font'].render(f"HIGH SCORE: {screen_state['highscore']}", True, GOLD_COLOR)
            screen.blit(highscore_surf, highscore_surf.get_rect(center=(screen_width/2, screen_height/2 + 80)))


        for button in buttons.get(screen_state["current"], []): button.draw(screen)

        # 화면 전환 효과 (사용자 요청 버전)
        if fading_out:
            transition_alpha += transition_speed
            if transition_alpha >= 255:
                transition_alpha = 255; fading_out = False; fading_in = True
                screen_state["current"] = transition_target; transition_target = None
            transition_surface.set_alpha(transition_alpha); screen.blit(transition_surface, (0, 0))
        elif fading_in:
            transition_alpha -= transition_speed
            if transition_alpha <= 0:
                transition_alpha = 0; fading_in = False
            transition_surface.set_alpha(transition_alpha); screen.blit(transition_surface, (0, 0))

        pygame.display.flip()
        clock.tick(60)
        loop_counter += 1

    # 리소스 해제
    if cap: cap.release()
    for key, val in assets.items():
        if isinstance(val, cv2.VideoCapture): val.release()
    if ser and ser.is_open: ser.close()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()