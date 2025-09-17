import pygame
import cv2
import numpy as np
from PIL import Image, ImageDraw
import imageio
import os

def create_circular_mask(size):
    """원형 마스크를 생성합니다."""
    # 크기가 0이하일 경우 오류를 방지합니다.
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
        
        # ★★★ 최적화 1: GIF 분석 시 이미지를 50%로 줄여서 처리 ★★★
        scale = 0.5
        small_frame = cv2.resize(frame_cv, (0, 0), fx=scale, fy=scale)
        gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
        
        # ★★★ 최적화 2: 파라미터 조절로 속도 향상 ★★★
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
    # --- 설정 ---
    gif_path = '../image/final_ronaldo/goalkeeper_win.gif'
    output_folder = 'cam'
    captured_face_path = os.path.join(output_folder, 'captured_face.png')

    pygame.init()
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 웹캠(0)을 열 수 없습니다.")
        return
    
    ret, frame = cap.read()
    if not ret: return
    cam_h, cam_w, _ = frame.shape
    screen = pygame.display.set_mode((cam_w, cam_h))
    pygame.display.set_caption("얼굴을 찍어주세요 (Press Spacebar)")
    
    face_cascade_cam = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    font = pygame.font.Font(None, 40)
    
    running = True
    clock = pygame.time.Clock()
    game_state = "capture"
    
    overlay_face_pil = None
    gif_frames, gif_face_locations = None, None
    gif_frame_index = 0

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if game_state == "capture":
                    ret, latest_frame = cap.read()
                    if ret:
                        frame_flipped = cv2.flip(latest_frame, 1)
                        
                        # ★★★ 최적화 3: 캡처 시에도 이미지를 줄여서 얼굴 탐색 ★★★
                        small_frame = cv2.resize(frame_flipped, (0, 0), fx=0.5, fy=0.5)
                        gray_cam = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade_cam.detectMultiScale(gray_cam, 1.3, 5)

                        if len(faces) > 0:
                            # 원본 크기로 좌표 복원
                            x, y, w, h = (faces[0] * 2).astype(int)
                            face_area = frame_flipped[y:y+h, x:x+w]
                            
                            if face_area.size > 0:
                                face_pil = Image.fromarray(cv2.cvtColor(face_area, cv2.COLOR_BGR2RGB))
                                mask = create_circular_mask((w, h))
                                if mask: face_pil.putalpha(mask)
                                
                                os.makedirs(output_folder, exist_ok=True)
                                face_pil.save(captured_face_path)
                                print(f"얼굴 캡처 완료! -> '{captured_face_path}'")
                                
                                # GIF 분석 및 합성 준비
                                gif_frames, gif_face_locations = preprocess_gif(gif_path)
                                if not gif_frames:
                                    running = False
                                    break
                                
                                overlay_face_pil = face_pil
                                game_state = "overlay"
                                
                                screen_size = gif_frames[0].get_size()
                                screen = pygame.display.set_mode(screen_size)
                                pygame.display.set_caption("사진 합성 중")
                        else:
                            print("얼굴을 찾지 못했습니다. 다시 시도해주세요.")

        # --- 화면 그리기 ---
        if game_state == "capture":
            ret, cam_frame = cap.read()
            if ret:
                frame_surface = pygame.surfarray.make_surface(cv2.cvtColor(cv2.flip(cam_frame, 1), cv2.COLOR_BGR2RGB).swapaxes(0, 1))
                screen.blit(frame_surface, (0, 0))
                text_surf = font.render("Spacebar를 눌러 얼굴을 촬영하세요", True, (255, 255, 0))
                screen.blit(text_surf, (10, 10))

        elif game_state == "overlay":
            current_gif_frame = gif_frames[gif_frame_index].copy()
            face_loc = gif_face_locations[gif_frame_index]
            
            if overlay_face_pil and face_loc is not None:
                gx, gy, gw, gh = face_loc
                if gw > 0 and gh > 0:
                    resized_face = overlay_face_pil.resize((gw, gh), Image.Resampling.LANCZOS)
                    face_surface = pygame.image.fromstring(resized_face.tobytes(), resized_face.size, "RGBA")
                    current_gif_frame.blit(face_surface, (gx, gy))

            screen.blit(current_gif_frame, (0, 0))
            gif_frame_index = (gif_frame_index + 1) % len(gif_frames)

        pygame.display.flip()
        clock.tick(60) # 최적화되었으므로 FPS를 60으로 올려도 좋습니다.

    cap.release()
    pygame.quit()

if __name__ == '__main__':
    main()