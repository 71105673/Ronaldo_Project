import cv2
import imageio
import numpy as np
import pygame
from PIL import Image
import os

# ===================================================================
# 3. GIF 처리 및 얼굴 합성 함수
# ===================================================================

# GIF 파일을 분석하여 각 프레임과 프레임별 얼굴 위치를 반환하는 함수
def preprocess_gif(gif_path):
    """
    GIF 파일을 분석하여 Pygame Surface 프레임 리스트와 각 프레임의 얼굴 위치를 반환합니다.
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
        # Pygame에서 사용하기 위해 이미지 데이터 형식 변환
        frame_pil = Image.fromarray(frame_data).convert("RGBA")
        frames.append(pygame.image.fromstring(frame_pil.tobytes(), frame_pil.size, "RGBA"))
        
        # OpenCV에서 얼굴 인식을 위해 형식 변환
        frame_cv = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)
        
        # 성능 최적화를 위해 이미지를 작은 크기로 변환하여 얼굴 인식 수행
        scale = 0.5
        small_frame = cv2.resize(frame_cv, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4, minSize=(20, 20))
        
        if len(faces) > 0:
            # 인식된 얼굴 좌표를 원본 이미지 크기에 맞게 복원
            main_face = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)[0]
            face_locations.append((main_face / scale).astype(int))
        else:
            # 얼굴이 인식되지 않은 경우 None 추가
            face_locations.append(None)
            
    print("GIF 분석 완료!")
    return frames, face_locations

# 캡처된 얼굴 이미지를 GIF 프레임에 합성하여 Pygame Surface 리스트를 생성하는 함수
def create_synthesized_gif_frames(face_image_path, gif_path, target_size):
    """
    얼굴 이미지를 GIF의 각 프레임에 합성하여 최종 Pygame Surface 리스트를 생성합니다.
    """
    if not face_image_path or not os.path.exists(face_image_path):
        print(f"합성할 얼굴 이미지 파일을 찾을 수 없습니다: {face_image_path}")
        return []

    # 1. GIF 분석 함수를 호출하여 프레임과 얼굴 위치 정보 가져오기
    gif_frames, gif_face_locations = preprocess_gif(gif_path)
    if not gif_frames:
        return []

    # 2. 합성할 얼굴 이미지를 불러오기
    try:
        overlay_face_pil = Image.open(face_image_path).convert("RGBA")
    except Exception as e:
        print(f"얼굴 이미지 로드 오류: {e}")
        return []

    synthesized_frames = []
    # 3. 각 프레임을 순회하며 얼굴 합성 작업 수행
    for i, base_frame_surface in enumerate(gif_frames):
        new_frame = base_frame_surface.copy()
        face_loc = gif_face_locations[i]

        # 해당 프레임에서 얼굴이 인식되었을 경우에만 합성
        if face_loc is not None:
            gx, gy, gw, gh = face_loc
            # 인식된 얼굴 크기에 맞춰 캡처된 얼굴 이미지 리사이즈
            resized_face_pil = overlay_face_pil.resize((gw, gh), Image.Resampling.LANCZOS)
            face_surface = pygame.image.fromstring(resized_face_pil.tobytes(), resized_face_pil.size, "RGBA")
            # 원본 프레임 위에 얼굴 이미지 합성
            new_frame.blit(face_surface, (gx, gy))
        
        # 최종적으로 화면에 표시될 크기로 프레임 조정
        scaled_frame = pygame.transform.scale(new_frame, target_size)
        synthesized_frames.append(scaled_frame)
        
    print("얼굴 합성 GIF 프레임 생성 완료!")
    return synthesized_frames