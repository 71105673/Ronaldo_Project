import pygame
import sys
import cv2
import serial  # pyserial 라이브러리
from button import ImageButton
import actions as game_actions # 분리된 actions.py 파일을 import

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
GRID_COLOR = (0, 255, 0)
HIGHLIGHT_COLOR = (255, 0, 0, 100)

# ==========================
# 메인 함수
# ==========================
def main():
    # 게임 상태 및 리소스를 관리하는 딕셔너리
    game_data = {
        "screen": screen, "screen_width": screen_width, "screen_height": screen_height,
        "clock": pygame.time.Clock(), "loop_counter": 0,
        "screen_state": {"current": "menu"}, "game_mode": {"mode": None},
        "chances_left": 5, "score": 0,
        "countdown_start_time": None, "selected_grid_col": None, "final_selected_col": None,
        "ball_col": None, "uart_ball_col": None, "is_failure": False, "is_success": False,
        "result_display_time": None, "gif_start_time": None, "gif_frame": None,
        "transition_surface": pygame.Surface((screen_width, screen_height)),
        "transition_alpha": 0, "transition_target": None, "transition_speed": 15,
        "fading_out": False, "fading_in": False,
        "ser": None, "fonts": {}, "sounds": {}, "images": {}, "videos": {},
        "colors": {'black': BLACK, 'white': WHITE, 'grid': GRID_COLOR, 'highlight': HIGHLIGHT_COLOR}
    }
    game_data["transition_surface"].fill(BLACK)

    # 폰트 로드
    try:
        game_data['fonts']['default'] = pygame.font.Font("./fonts/netmarbleM.ttf", 40)
        game_data['fonts']['description'] = pygame.font.Font("./fonts/netmarbleM.ttf", 50)
        game_data['fonts']['title'] = pygame.font.Font("./fonts/netmarbleB.ttf", 120)
        game_data['fonts']['countdown'] = pygame.font.Font("./fonts/netmarbleM.ttf", 200)
        game_data['fonts']['score'] = pygame.font.Font("./fonts/netmarbleB.ttf", 60)
    except FileNotFoundError:
        print("폰트 파일을 찾을 수 없습니다. 기본 폰트를 사용합니다.")
        game_data['fonts']['default'] = pygame.font.Font(None, 50)
        # ... (다른 기본 폰트 설정)
    
    # 효과음 및 리소스 로드 (이하 생략된 부분은 기존 코드와 동일)
    try:
        game_data['sounds']['button'] = pygame.mixer.Sound("./sound/button_click.wav")
        game_data['sounds']['siu'] = pygame.mixer.Sound("./sound/SIUUUUU.wav")
        game_data['sounds']['success'] = pygame.mixer.Sound("./sound/야유.mp3")
        game_data['sounds']['failed'] = pygame.mixer.Sound("./sound/SIUUUUU.wav")
    except pygame.error as e: print(f"효과음 로드 오류: {e}")

    try:
        ball_img = pygame.image.load("./image/final_ronaldo/Ball.png").convert_alpha()
        game_data['images']['scoreboard_ball'] = pygame.transform.scale(ball_img, (80, 80))
        game_data['images']['ball'] = pygame.transform.scale(ball_img, (200, 200))
        game_data['images']['info_bg'] = pygame.transform.scale(pygame.image.load("./image/info/info_back2.jpg").convert(), (screen_width, screen_height))
    except pygame.error as e: print(f"이미지 로드 오류: {e}")

    try: game_data['videos']['menu_bg'] = cv2.VideoCapture("./image/game_thumbnail.mp4")
    except: pass
    try: game_data['videos']['failure_gif'] = cv2.VideoCapture("./image/G.O.A.T/siuuu.gif")
    except: pass
    try: game_data['videos']['success_gif'] = cv2.VideoCapture("./image/final_ronaldo/pk.gif")
    except: pass
    try: game_data['videos']['end_video'] = cv2.VideoCapture("./video/ending.mp4")
    except: pass
    
    game_data['videos']['webcam'] = cv2.VideoCapture(0)
    if not game_data['videos']['webcam'].isOpened(): print("오류: 웹캠을 열 수 없습니다."); game_data['videos']['webcam'] = None
    try:
        game_data['ser'] = serial.Serial('COM13', 9600, timeout=0)
        print("Basys3 보드가 성공적으로 연결되었습니다.")
    except serial.SerialException as e: print(f"오류: 시리얼 포트를 열 수 없습니다 - {e}")

    # 버튼 생성 (람다 함수에서 game_actions의 함수를 호출)
    buttons = {
        "menu": [ImageButton("./image/btn_start.png", screen_width - 300, screen_height - 175, 400, 250, lambda: game_actions.start_transition(game_data, "game"), sound=game_data['sounds'].get('button')),
                 ImageButton("./image/btn_desc.png", screen_width - 150, 150, 100, 100, lambda: game_actions.start_transition(game_data, "info"), sound=game_data['sounds'].get('button'))],
        "game": [ImageButton("./image/btn_single.png", screen_width//2 - 280, screen_height//2 + 200, 550, 600, lambda: game_actions.set_game_mode(game_data, "single")),
                 ImageButton("./image/btn_multi.png", screen_width//2 + 430, screen_height//2 + 200, 550, 600, lambda: game_actions.set_game_mode(game_data, "multi")),
                 ImageButton("./image/btn_back.png", 150, 150, 100, 100, lambda: game_actions.go_back(game_data), sound=game_data['sounds'].get('button'))],
        "webcam_view": [ImageButton("./image/btn_back.png", 150, 150, 100, 100, lambda: game_actions.go_back(game_data), sound=game_data['sounds'].get('button'))],
        "multi": [ImageButton("./image/btn_back.png", 150, 150, 100, 100, lambda: game_actions.go_back(game_data), sound=game_data['sounds'].get('button'))],
        "info": [ImageButton("./image/btn_exit.png", screen_width - 150, 150, 100, 100, lambda: game_actions.go_back(game_data), sound=game_data['sounds'].get('button'))],
        "end": [ImageButton("./image/btn_exit.png", screen_width - 150, 150, 100, 100, lambda: game_actions.go_back(game_data), sound=game_data['sounds'].get('button'))]
    }

    # ==========================
    # 메인 루프
    # ==========================
    running = True
    while running:
        # 이벤트 처리
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            if not (game_data['fading_in'] or game_data['fading_out']):
                for button in buttons.get(game_data['screen_state']["current"], []):
                    button.handle_event(event)
        
        if not (game_data['fading_in'] or game_data['fading_out']):
            for button in buttons.get(game_data['screen_state']["current"], []):
                button.update()

        # 화면 상태에 따른 그리기
        current_screen = game_data['screen_state']['current']
        
        if current_screen in ["menu", "game"]:
            video = game_data['videos']['menu_bg']
            ret, frame = video.read()
            if not ret: video.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = video.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame = cv2.resize(frame, (screen_width, screen_height))
                screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
            if current_screen == "game":
                text_surf = game_data['fonts']['default'].render("플레이어 수를 선택하세요", True, WHITE)
                screen.blit(text_surf, (screen_width//2 - text_surf.get_width()//2, screen_height//2 - 200))
        
        elif current_screen in ["webcam_view", "multi"]:
            game_actions.handle_game_logic(game_data)

        elif current_screen == "info":
            screen.blit(game_data['images']['info_bg'], (0, 0))
            title_surf = game_data['fonts']['title'].render("게임 방법", True, WHITE)
            screen.blit(title_surf, (screen_width / 2 - title_surf.get_width() / 2, 150))
            text_lines_1p = ["[1인 플레이]", "1. 5초의 카운트 다운이 시작됩니다.", "2. 카메라에 비치는 빨간색", "   물체를 인식합니다.", "3. 5개의 영역 중 하나를 선택합니다.", "4. 공을 막으면 성공!"]
            text_lines_2p = ["[2인 플레이]", "1. 플레이어는 공격수가 되어", "   몸으로 방향을 정합니다.", "2. 골키퍼(Basys3)는 스위치로", "   막을 방향을 정합니다.", "3. 더 많은 득점을 한 플레이어가", "   승리합니다."]
            x1, x2, y_start = screen_width / 4 - 150, screen_width * 3 / 4 - 300, 400
            desc_font = game_data['fonts']['description']
            for i, line in enumerate(text_lines_1p): screen.blit(desc_font.render(line, True, WHITE), (x1, y_start + i*75))
            for i, line in enumerate(text_lines_2p): screen.blit(desc_font.render(line, True, WHITE), (x2, y_start + i*75))
            
        elif current_screen == "end":
            end_video = game_data['videos'].get('end_video')
            if end_video:
                ret, frame = end_video.read()
                if not ret: end_video.set(cv2.CAP_PROP_POS_FRAMES, 0); ret, frame = end_video.read()
                if ret:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); frame = cv2.resize(frame, (screen_width, screen_height))
                    screen.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))
            else:
                screen.fill(BLACK)
                end_text = game_data['fonts']['title'].render("GAME OVER", True, WHITE)
                screen.blit(end_text, end_text.get_rect(center=(screen_width/2, screen_height/2)))
        
        for button in buttons.get(current_screen, []):
            button.draw(screen)

        # 화면 전환 효과
        if game_data['fading_out'] or game_data['fading_in']:
            if game_data['fading_out']: game_data['transition_alpha'] += game_data['transition_speed']
            else: game_data['transition_alpha'] -= game_data['transition_speed']
            
            game_data['transition_alpha'] = max(0, min(255, game_data['transition_alpha']))
            
            if game_data['transition_alpha'] == 255:
                game_data.update({'fading_out': False, 'fading_in': True})
                game_data['screen_state']["current"] = game_data['transition_target']
                game_data['transition_target'] = None
            elif game_data['transition_alpha'] == 0:
                game_data['fading_in'] = False
                
            game_data['transition_surface'].set_alpha(game_data['transition_alpha'])
            screen.blit(game_data['transition_surface'], (0, 0))

        pygame.display.flip()
        game_data['clock'].tick(60)
        game_data['loop_counter'] += 1

    # 종료 시 리소스 해제
    for video in game_data['videos'].values():
        if video: video.release()
    if game_data['ser'] and game_data['ser'].is_open:
        game_data['ser'].close()
    pygame.quit()
    sys.exit()

if __name__ == '__main__':
    main()