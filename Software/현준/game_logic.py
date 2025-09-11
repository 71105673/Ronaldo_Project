# game_logic.py

import pygame
import cv2
import numpy as np
import random
from score_manager import save_highscore

# --- 상태 관리 함수 ---

def reset_game_state(game_state, full_reset=True):
    """게임 상태 변수들을 초기화합니다."""
    game_state.update({
        'countdown_start_time': None, 'selected_grid_col': None, 
        'final_selected_col': None, 'ball_col': None,
        'is_failure': False, 'is_success': False, 
        'result_display_time': None, 'gif_start_time': None, 
        'uart_ball_col': None, 'gif_frame': None
    })
    if full_reset:
        game_state.update({'chances_left': 5, 'score': 0})

def start_new_round(game_state):
    """점수와 기회는 유지한 채 새로운 라운드를 시작합니다."""
    reset_game_state(game_state, full_reset=False)
    game_state['countdown_start_time'] = pygame.time.get_ticks()

# --- 흐름 제어 함수 ---

def restart_game(game_state, start_transition_func):
    """게임을 완전히 새로 시작합니다."""
    reset_game_state(game_state, full_reset=True)
    start_new_round(game_state)
    target_screen = "webcam_view" if game_state["mode"] == "single" else "multi"
    start_transition_func(target_screen)

def go_to_menu(game_state, start_transition_func):
    """메인 메뉴로 돌아갑니다."""
    reset_game_state(game_state, full_reset=True)
    start_transition_func("menu")

def set_game_mode(game_state, mode, siu_sound, start_transition_func):
    """게임 모드를 설정하고 게임을 시작합니다."""
    if siu_sound: siu_sound.play()
    game_state["mode"] = mode
    # restart_game 함수를 직접 호출
    restart_game(game_state, start_transition_func)

# --- 메인 게임 로직 함수 ---

def handle_game_logic(screen, game_state, assets, cap, ser, start_transition_func, loop_counter):
    """실제 게임 플레이(웹캠, 카운트다운, 결과 판정 등)를 처리하고 화면에 그립니다."""
    screen_width = screen.get_width()
    screen_height = screen.get_height()
    
    # 2초 후 다음 라운드 또는 게임 종료
    if game_state['gif_start_time'] and (pygame.time.get_ticks() - game_state['gif_start_time'] > 2000):
        if game_state['chances_left'] > 0:
            start_new_round(game_state)
        else:
            if game_state['score'] > game_state['highscore']:
                game_state['highscore'] = game_state['score']
                save_highscore(game_state['highscore'])
            
            if game_state['score'] == 5: game_state['final_rank'], game_state['end_video_to_play'] = "THE WALL", assets["victory_video"]
            elif game_state['score'] >= 3: game_state['final_rank'], game_state['end_video_to_play'] = "Pro Keeper", assets["victory_video"]
            elif game_state['score'] >= 1: game_state['final_rank'], game_state['end_video_to_play'] = "Rookie Keeper", assets["defeat_video"]
            else: game_state['final_rank'], game_state['end_video_to_play'] = "Human Sieve", assets["defeat_video"]
            
            if game_state['end_video_to_play']: game_state['end_video_to_play'].set(cv2.CAP_PROP_POS_FRAMES, 0)
            start_transition_func("end")
        return

    # 성공/실패 GIF 재생 로직
    should_play_gif = (game_state['is_failure'] or game_state['is_success']) and game_state['result_display_time'] and (pygame.time.get_ticks() - game_state['result_display_time'] > 1000)
    
    active_gif = None
    if should_play_gif:
        active_gif = assets["failure_gif"] if game_state['is_failure'] else assets["success_gif"]
    
    if active_gif and not game_state['gif_start_time']:
        game_state['gif_start_time'] = pygame.time.get_ticks()

    if game_state['gif_start_time']:
        # GIF 재생 화면
        screen.fill((0,0,0)) 
        if loop_counter % 2 == 0:
            ret, frame = active_gif.read()
            if not ret: 
                active_gif.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ret, frame = active_gif.read()
            if ret: game_state['gif_frame'] = frame
        
        if game_state['gif_frame'] is not None:
            gif_display_size = (screen_width // 3, screen_height // 3) if game_state['is_success'] else (screen_width, screen_height)
            frame_resized = cv2.resize(game_state['gif_frame'], gif_display_size, interpolation=cv2.INTER_AREA)
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            gif_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
            gif_rect = gif_surface.get_rect(center=(screen_width // 2, screen_height // 2))
            screen.blit(gif_surface, gif_rect)
        return

    # 웹캠 화면 및 게임 플레이 로직
    if cap and cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (screen_width, screen_height))
            screen.blit(pygame.surfarray.make_surface(frame_resized.swapaxes(0, 1)), (0, 0))

            # 그리드 그리기
            for i in range(1, 5): 
                pygame.draw.line(screen, (0, 255, 0), (i * (screen_width // 5), 0), (i * (screen_width // 5), screen_height), 2)
            
            # 카운트다운 로직
            if game_state['countdown_start_time'] is not None:
                elapsed_time = pygame.time.get_ticks() - game_state['countdown_start_time']
                if elapsed_time < 5000:
                    # 빨간색 물체 감지
                    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    lower_red1, upper_red1 = np.array([0, 120, 70]), np.array([10, 255, 255])
                    lower_red2, upper_red2 = np.array([170, 120, 70]), np.array([180, 255, 255])
                    mask = cv2.inRange(hsv_frame, lower_red1, upper_red1) + cv2.inRange(hsv_frame, lower_red2, upper_red2)
                    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if contours and cv2.contourArea(max(contours, key=cv2.contourArea)) > 500:
                        x, _, w, _ = cv2.boundingRect(max(contours, key=cv2.contourArea))
                        _, cam_w, _ = frame.shape
                        game_state['selected_grid_col'] = int((x + w / 2) / (cam_w / 5))
                    else: 
                        game_state['selected_grid_col'] = None
                    
                    # UART 데이터 수신
                    if game_state["mode"] == 'multi' and ser and ser.is_open and ser.in_waiting > 0:
                        try:
                            uart_data_stream = ser.read(ser.in_waiting).decode('utf-8', errors='ignore')
                            valid_chars = [c for c in uart_data_stream if c in '12345']
                            if valid_chars:
                                game_state['uart_ball_col'] = int(valid_chars[-1]) - 1
                        except Exception as e: print(f"UART 데이터 수신/처리 중 오류: {e}")

                    # 카운트다운 숫자 표시
                    num = str(5 - (elapsed_time // 1000))
                    text_surf = assets['countdown_font'].render(num, True, (255, 255, 255))
                    screen.blit(text_surf, text_surf.get_rect(center=(screen_width // 2, screen_height // 2)))

                else: # 5초 후 결과 처리
                    if game_state['final_selected_col'] is None:
                        game_state['final_selected_col'] = game_state['selected_grid_col']
                        game_state['chances_left'] -= 1
                        
                        game_state['ball_col'] = (game_state['uart_ball_col'] if game_state["mode"] == 'multi' and game_state['uart_ball_col'] is not None else random.randint(0, 4))
                        
                        if game_state['ball_col'] is not None and game_state['final_selected_col'] == game_state['ball_col']:
                            game_state['is_success'], game_state['score'] = True, game_state['score'] + 1
                            if assets['success_sound']: assets['success_sound'].play()
                        else: 
                            game_state['is_failure'] = True
                        
                        game_state['result_display_time'] = pygame.time.get_ticks()
                        game_state['countdown_start_time'] = None
            
            # 최종 선택 영역 및 공 그리기
            if game_state['final_selected_col'] is not None:
                cell_w = screen_width / 5
                highlight_surf = pygame.Surface((cell_w, screen_height), pygame.SRCALPHA)
                highlight_surf.fill((255, 0, 0, 100))
                screen.blit(highlight_surf, (game_state['final_selected_col'] * cell_w, 0))

            if game_state['ball_col'] is not None and assets['ball_image']:
                cell_w = screen_width / 5
                ball_rect = assets['ball_image'].get_rect(center=(game_state['ball_col'] * cell_w + cell_w / 2, screen_height / 2))
                screen.blit(assets['ball_image'], ball_rect)
    
    # 스코어보드 그리기
    if assets['scoreboard_ball_image']:
        for i in range(game_state['chances_left']):
            screen.blit(assets['scoreboard_ball_image'], (screen_width - 100 - i*90, 50))
    score_text = assets['score_font'].render(f"SCORE: {game_state['score']}", True, (255, 255, 255))
    screen.blit(score_text, (screen_width - 300, 150))