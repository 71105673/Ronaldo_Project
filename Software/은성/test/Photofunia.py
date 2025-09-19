# photofunia.py
import cv2
import imageio
import numpy as np
import pygame
from PIL import Image
import os
from typing import List, Tuple, Optional

def preprocess_gif(gif_path: str) -> Tuple[Optional[List[pygame.Surface]], Optional[List[Optional[np.ndarray]]]]:
    """
    GIF 파일을 분석하여 Pygame Surface 프레임 리스트와 각 프레임의 얼굴 위치를 반환합니다.
    얼굴이 감지되지 않은 프레임의 위치 정보는 None입니다.
    """
    print("GIF 파일을 미리 분석 중입니다...")
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if not os.path.exists(gif_path):
        print(f"오류: '{gif_path}' 파일을 찾을 수 없습니다.")
        return None, None
        
    try:
        gif_reader = imageio.get_reader(gif_path)
    except Exception as e:
        print(f"GIF 리더 생성 오류: {e}")
        return None, None

    frames, face_locations = [], []

    for frame_data in gif_reader:
        frame_pil = Image.fromarray(frame_data).convert("RGBA")
        frames.append(pygame.image.fromstring(frame_pil.tobytes(), frame_pil.size, "RGBA"))
        
        frame_cv = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)
        
        # 성능을 위해 그레이스케일로 변환하여 얼굴 인식
        gray = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2GRAY)
        
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # 가장 큰 얼굴을 메인 얼굴로 선택
            main_face = max(faces, key=lambda f: f[2] * f[3])
            face_locations.append(main_face)
        else:
            face_locations.append(None)
            
    print("GIF 분석 완료!")
    return frames, face_locations

def create_synthesized_gif_frames(face_image_path: str, gif_path: str, target_size: Tuple[int, int]) -> List[pygame.Surface]:
    """
    캡처된 얼굴 이미지를 GIF 각 프레임에 합성하여 최종 Pygame Surface 리스트를 생성합니다.
    """
    if not face_image_path or not os.path.exists(face_image_path):
        print(f"합성할 얼굴 이미지 파일을 찾을 수 없습니다: {face_image_path}")
        return []

    # 1. GIF 분석
    gif_frames, gif_face_locations = preprocess_gif(gif_path)
    if not gif_frames:
        return []

    # 2. 합성할 얼굴 이미지 로드
    try:
        overlay_face_pil = Image.open(face_image_path).convert("RGBA")
    except Exception as e:
        print(f"얼굴 이미지 로드 오류: {e}")
        return []

    synthesized_frames = []
    # 3. 각 프레임에 얼굴 합성
    for i, base_frame_surface in enumerate(gif_frames):
        new_frame = base_frame_surface.copy()
        face_loc = gif_face_locations[i]

        if face_loc is not None:
            gx, gy, gw, gh = face_loc
            # 인식된 얼굴 크기에 맞춰 캡처된 얼굴 이미지 리사이즈
            resized_face_pil = overlay_face_pil.resize((gw, gh), Image.Resampling.LANCZOS)
            face_surface = pygame.image.fromstring(resized_face_pil.tobytes(), resized_face_pil.size, "RGBA")
            new_frame.blit(face_surface, (gx, gy))
        
        # 최종 화면 크기로 프레임 조정
        scaled_frame = pygame.transform.scale(new_frame, target_size)
        synthesized_frames.append(scaled_frame)
        
    print("얼굴 합성 GIF 프레임 생성 완료!")
    return synthesized_frames