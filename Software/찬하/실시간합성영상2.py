import pygame
import cv2
import numpy as np
from PIL import Image, ImageDraw
import imageio
import os

def create_circular_mask(size):
    """원형 마스크를 생성합니다."""
    if size[0] <= 0 or size[1] <= 0: return None
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)
    return mask

def preprocess_gif(gif_path):
    """
    GIF을 미리 분석하여 각 프레임과 얼굴 위치 정보를 반환합니다. (최적화 적용)
    """
    print("GIF 파일을 미리 분석 중입니다...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    try:
        gif_reader = imageio.get_reader(gif_path)
    except FileNotFoundError:
        print(f"오류: '{gif_path}' 파일을 찾을 수 없습니다.")
        return None, None

    frames, face_locations = [], []

    for frame_data in gif_reader:
        frame_pil = Image.fromarray(frame_data).convert("RGBA")
        frames.append(pygame.image.fromstring(frame_pil.tobytes(), frame_pil.size, "RGBA"))
        
        frame_cv = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)
        
        # 최적화 1: GIF 분석 시에도 이미지를 줄여서 처리
        scale = 0.5
        small_frame = cv2.resize(frame_cv, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # 최적화 2: 파라미터 조절로 속도 향상
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(20, 20))
        
        if len(faces) > 0:
            # 원본 크기로 좌표 복원
            main_face = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face_locations.append((main_face / scale).astype(int))
        else:
            face_locations.append(None)
            
    print("GIF 분석 완료!")
    return frames, face_locations

def main():
    gif_path = '../image/final_ronaldo/goalkeeper_win.gif'
    gif_frames, gif_face_locations = preprocess_gif(gif_path)
    if not gif_frames: return

    pygame.init()
    screen_size = gif_frames[0].get_size()
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("실시간 영상 합성 (최적화)")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 웹캠(0)을 열 수 없습니다.")
        return
        
    face_cascade_cam = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    running = True
    clock = pygame.time.Clock()
    gif_frame_index = 0
    
    # --- 최적화를 위한 변수 ---
    frame_counter = 0
    WEBCAM_FACE_UPDATE_RATE = 5  # 5프레임마다 한 번씩만 웹캠 얼굴 위치 갱신
    last_known_face_rect = None  # 마지막으로 찾은 얼굴 위치 저장

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        ret, cam_frame = cap.read()
        if not ret: break
            
        cam_frame_flipped = cv2.flip(cam_frame, 1)
        
        # ★★★ 핵심 최적화: 프레임 건너뛰기 ★★★
        if frame_counter % WEBCAM_FACE_UPDATE_RATE == 0:
            # 이미지를 작게 만들어서 얼굴 탐색
            small_frame = cv2.resize(cam_frame_flipped, (0, 0), fx=0.5, fy=0.5)
            gray_cam = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            faces_cam = face_cascade_cam.detectMultiScale(gray_cam, 1.3, 5)
            
            if len(faces_cam) > 0:
                # 찾은 얼굴 위치를 원래 크기로 되돌리고 저장
                last_known_face_rect = (faces_cam[0] * 2).astype(int)

        cropped_face_pil = None
        if last_known_face_rect is not None:
            x, y, w, h = last_known_face_rect
            
            # 원본 고화질 프레임에서 얼굴 자르기
            face_area = cam_frame_flipped[y:y+h, x:x+w]
            if face_area.size > 0:
                face_area_rgb = cv2.cvtColor(face_area, cv2.COLOR_BGR2RGB)
                cropped_face_pil = Image.fromarray(face_area_rgb)
                
                mask = create_circular_mask((w, h))
                if mask:
                    cropped_face_pil.putalpha(mask)

        # --- GIF와 합성하여 화면에 그리기 (이 부분은 동일) ---
        current_gif_frame = gif_frames[gif_frame_index].copy()
        face_loc = gif_face_locations[gif_frame_index]
        
        if cropped_face_pil and face_loc is not None:
            gx, gy, gw, gh = face_loc
            if gw > 0 and gh > 0:
                resized_face = cropped_face_pil.resize((gw, gh), Image.Resampling.LANCZOS)
                face_surface = pygame.image.fromstring(resized_face.tobytes(), resized_face.size, "RGBA")
                current_gif_frame.blit(face_surface, (gx, gy))

        screen.blit(current_gif_frame, (0, 0))
        
        gif_frame_index = (gif_frame_index + 1) % len(gif_frames)
        frame_counter += 1

        pygame.display.flip()
        clock.tick(60) # FPS를 60으로 올려도 부드럽게 동작할 수 있습니다.

    cap.release()
    pygame.quit()

if __name__ == '__main__':
    main()