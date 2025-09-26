import pygame
import cv2
import numpy as np
from PIL import Image, ImageDraw
import imageio
import os

def create_circular_mask(size):
    """원형 마스크를 생성합니다."""
    mask = Image.new('L', size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0) + size, fill=255)
    return mask

def preprocess_gif(gif_path):
    """
    GIF을 미리 분석하여 각 프레임과 얼굴 위치 정보를 반환합니다.
    """
    print("GIF 파일을 미리 분석 중입니다...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    try:
        gif_reader = imageio.get_reader(gif_path)
    except FileNotFoundError:
        print(f"오류: '{gif_path}' 파일을 찾을 수 없습니다.")
        return None, None

    frames = []
    face_locations = []

    for frame_data in gif_reader:
        frame_pil = Image.fromarray(frame_data).convert("RGBA")
        frame_surface = pygame.image.fromstring(frame_pil.tobytes(), frame_pil.size, "RGBA")
        frames.append(frame_surface)
        
        frame_cv = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            main_face = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face_locations.append(main_face)
        else:
            face_locations.append(None)
            
    print("GIF 분석 완료!")
    return frames, face_locations

def main():
    gif_path = '../image/final_ronaldo/goalkeeper_win.gif'
    
    # 1. GIF 미리 분석하기
    gif_frames, gif_face_locations = preprocess_gif(gif_path)
    if not gif_frames:
        return

    pygame.init()
    
    # 화면 크기는 첫 번째 GIF 프레임에 맞춤
    screen_size = gif_frames[0].get_size()
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("실시간 영상 합성")
    
    # 2. 웹캠(cap0) 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 웹캠(0)을 열 수 없습니다.")
        return
        
    face_cascade_cam = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    running = True
    clock = pygame.time.Clock()
    gif_frame_index = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # --- 실시간 웹캠 영상에서 얼굴 잘라내기 ---
        ret, cam_frame = cap.read()
        if not ret:
            break
            
        cam_frame_flipped = cv2.flip(cam_frame, 1) # 좌우반전
        gray_cam = cv2.cvtColor(cam_frame_flipped, cv2.COLOR_BGR2GRAY)
        faces_cam = face_cascade_cam.detectMultiScale(gray_cam, 1.3, 5)
        
        cropped_face_pil = None
        if len(faces_cam) > 0:
            x, y, w, h = faces_cam[0] # 첫 번째로 찾은 얼굴 사용
            
            face_area = cam_frame_flipped[y:y+h, x:x+w]
            face_area_rgb = cv2.cvtColor(face_area, cv2.COLOR_BGR2RGB)
            cropped_face_pil = Image.fromarray(face_area_rgb)
            
            # 원형 마스크 적용
            mask = create_circular_mask((w, h))
            cropped_face_pil.putalpha(mask)

        # --- GIF와 합성하여 화면에 그리기 ---
        current_gif_frame = gif_frames[gif_frame_index].copy()
        face_loc = gif_face_locations[gif_frame_index]
        
        if cropped_face_pil and face_loc is not None:
            gx, gy, gw, gh = face_loc
            
            resized_face = cropped_face_pil.resize((gw, gh), Image.Resampling.LANCZOS)
            face_surface = pygame.image.fromstring(resized_face.tobytes(), resized_face.size, "RGBA")
            current_gif_frame.blit(face_surface, (gx, gy))

        screen.blit(current_gif_frame, (0, 0))
        
        gif_frame_index = (gif_frame_index + 1) % len(gif_frames)

        pygame.display.flip()
        clock.tick(30)

    cap.release()
    pygame.quit()

if __name__ == '__main__':
    main()