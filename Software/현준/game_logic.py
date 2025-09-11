# game_logic.py

import pygame
import cv2
import numpy as np
import random

# --- 상태 및 흐름 제어 함수 ---

def reset_game_state(screen_state):
    """게임을 처음부터 다시 시작하기 위해 모든 상태를 초기화합니다."""
    screen_state.update({
        'countdown_start_time': None, 'selected_grid_col': None, 'final_selected_col': None,
        'ball_col': None, 'is_failure': False, 'is_success': False,
        'result_display_time': None, 'gif_start_time': None, 'uart_ball_col': None,
        'gif_frame': None, 'chances_left': 5, 'score': 0
    })

def start_new_round(screen_state):
    """다음 라운드를 위해 게임 상태를 부분적으로 초기화합니다."""
    screen_state.update({
        'countdown_start_time': pygame.time.get_ticks(),
        'selected_grid_col': None, 'final_selected_col': None, 'ball_col': None,
        'is_failure': False, 'is_success': False,
        'result_display_time': None, 'gif_start_time': None, 'uart_ball_col': None
    })

def go_back(screen_state, start_transition_func):
    """'뒤로가기' 버튼 로직. 현재 화면에 따라 적절한 이전 화면으로 이동합니다."""
    current = screen_state["current"]
    # 게임 플레이 관련 화면이면 'game' 선택 화면으로, 그 외에는 'menu'로 이동
    target = "game" if current in ["single", "multi", "webcam_view", "end"] else "menu"
    
    # 게임 상태를 리셋해야 하는 경우 (메뉴나 게임 선택 화면으로 돌아갈 때)
    if target == "game" or target == "menu":
        reset_game_state(screen_state)
    
    start_transition_func(target)

def set_game_mode(screen_state, mode, assets, start_transition_func):
    """게임 모드(싱글/멀티)를 설정하고 게임을 시작합니다."""
    if assets.get('siu_sound'):
        assets['siu_sound'].play()
    screen_state["mode"] = mode
    
    reset_game_state(screen_state)
    screen_state['countdown_start_time'] = pygame.time.get_ticks()
    
    target_screen = "webcam_view" if mode == "single" else "multi"
    start_transition_func(target_screen)

# --- 메인 게임 로직 ---

def handle_game_logic(screen, screen_state, assets, cap, ser, start_transition_func, loop_counter):
    """실시간 게임 플레이를 처리하고 화면에 그립니다."""
    screen_width, screen_height = screen.get_width(), screen.get_height()
    
    # GIF 재생이 끝나면 다음 라운드 또는 종료 화면으로 전환
    if screen_state['gif_start_time'] and (pygame.time.get_ticks() - screen_state['gif_start_time'] > 2000):
        if screen_state['chances_left'] > 0:
            start_new_round(screen_state)
        else:
            start_transition_func("end")
        return

    # 결과(성공/실패)가 나오고 1초가 지나면 GIF 재생 시작
    should_play_gif = (screen_state['is_failure'] or screen_state['is_success']) and \
                      screen_state['result_display_time'] and \
                      (pygame.time.get_ticks() - screen_state['result_display_time'] > 1000)

    active_gif = None
    if should_play_gif:
        active_gif = assets["failure_gif"] if screen_state['is_failure'] else assets["success_gif"]

    # GIF 재생 시작 시점에 사운드도 함께 재생
    if active_gif and not screen_state['gif_start_time']:
        screen_state['gif_start_time'] = pygame.time.get_ticks()
        if screen_state['is_success'] and assets.get('success_sound'):
            assets['success_sound'].play()
        elif screen_state['is_failure'] and assets.get('failed_sound'):
            assets['failed_sound'].play()

    # GIF 재생 중일 때의 화면 처리
    if screen_state['gif_start_time']:
        screen.fill((0, 0, 0))
        # 성공/실패에 따라 GIF 재생 속도 조절
        update_frame = False
        if screen_state['is_success'] and loop_counter % 2 == 0:
            update_frame = True
        elif screen_state['is_failure'] and loop_counter % 4 == 0:
            update_frame = True

        if update_frame:
            ret, frame = active_gif.read()
            if not ret:
                active_gif.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = active_gif.read()
            if ret:
                screen_state['gif_frame'] = frame
        
        if screen_state['gif_frame'] is not None:
            if screen_state['is_success']:
                gif_display_size = (screen_width // 3, screen_height // 3)
            else: # is_failure
                gif_display_size = (screen_width, screen_height)
            
            frame_resized = cv2.resize(screen_state['gif_frame'], gif_display_size, interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            gif_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            gif_rect = gif_surface.get_rect(center=(screen_width // 2, screen_height // 2))
            screen.blit(gif_surface, gif_rect)
        return

    # 웹캠을 이용한 실시간 플레이 화면 처리
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (screen_width, screen_height))
            screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (0, 0))

            # 그리드 그리기
            GRID_COLOR = (0, 255, 0)
            for i in range(1, 5): 
                pygame.draw.line(screen, GRID_COLOR, (i * (screen_width // 5), 0), (i * (screen_width // 5), screen_height), 2)
            
            # 카운트다운 로직
            if screen_state['countdown_start_time'] is not None:
                elapsed_time = pygame.time.get_ticks() - screen_state['countdown_start_time']
                if elapsed_time < 5000:
                    # 물체 감지, UART 수신, 카운트다운 숫자 표시 등
                    # ... (이하 로직은 이전과 동일) ...
                    num = str(5 - (elapsed_time // 1000))
                    text_surf = assets['countdown_font'].render(num, True, (255,255,255))
                    screen.blit(text_surf, text_surf.get_rect(center=(screen_width / 2, screen_height / 2)))
                else: # 5초 후 결과 처리
                    if screen_state['final_selected_col'] is None:
                        screen_state['final_selected_col'] = screen_state['selected_grid_col']
                        screen_state['chances_left'] -= 1
                        
                        if screen_state["mode"] == 'multi':
                            screen_state['ball_col'] = screen_state.get('uart_ball_col', random.randint(0, 4))
                        else: # single mode
                            screen_state['ball_col'] = random.randint(0, 4)
                        
                        if screen_state['ball_col'] == screen_state['final_selected_col']:
                            screen_state['is_success'] = True
                            screen_state['score'] += 1
                        else:
                            screen_state['is_failure'] = True
                        
                        screen_state['result_display_time'] = pygame.time.get_ticks()
                        screen_state['countdown_start_time'] = None

            # 최종 선택된 영역과 공의 위치 표시
            if screen_state['final_selected_col'] is not None:
                HIGHLIGHT_COLOR = (255, 0, 0, 100)
                cell_w = screen_width / 5
                highlight_surf = pygame.Surface((cell_w, screen_height), pygame.SRCALPHA)
                highlight_surf.fill(HIGHLIGHT_COLOR)
                screen.blit(highlight_surf, (screen_state['final_selected_col'] * cell_w, 0))

            if screen_state['ball_col'] is not None:
                cell_w = screen_width / 5
                ball_rect = assets['ball_image'].get_rect(center=(screen_state['ball_col'] * cell_w + cell_w / 2, screen_height / 2))
                screen.blit(assets['ball_image'], ball_rect)

    # 점수판 그리기
    if assets['scoreboard_ball_image']:
        for i in range(screen_state['chances_left']):
            screen.blit(assets['scoreboard_ball_image'], (screen_width - 100 - i*90, 50))
    score_text = assets['score_font'].render(f"SCORE: {screen_state['score']}", True, (255,255,255))
    screen.blit(score_text, (screen_width - 300, 150))