import pygame
import cv2
import numpy as np
from PIL import Image, ImageDraw
import imageio  # <-- 1. 오류 해결: imageio 라이브러리 import 추가
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
    # --- 설정 ---
    gif_path = '../image/final_ronaldo/goalkeeper_win.gif'
    # 2. 새로운 기능: 캡처한 얼굴을 저장할 경로 설정
    output_folder = 'cam'
    captured_face_path = os.path.join(output_folder, 'captured_face.png')

    pygame.init()
    
    # 웹캠 열기
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("오류: 웹캠(0)을 열 수 없습니다.")
        return
    
    # 웹캠 크기에 맞춰 화면 설정
    ret, frame = cap.read()
    if not ret:
        print("오류: 웹캠에서 프레임을 읽을 수 없습니다.")
        cap.release()
        return
    cam_h, cam_w, _ = frame.shape
    screen = pygame.display.set_mode((cam_w, cam_h))
    pygame.display.set_caption("얼굴을 찍어주세요 (Press Spacebar)")
    
    face_cascade_cam = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    font = pygame.font.Font(None, 40)
    
    running = True
    clock = pygame.time.Clock()
    game_state = "capture" # 초기 상태는 '촬영'

    # --- 메인 루프 ---
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False
            
            # 3. 새로운 기능: 스페이스바를 누르면 얼굴 캡처
            if event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                if game_state == "capture":
                    ret, latest_frame = cap.read()
                    if ret:
                        frame_flipped = cv2.flip(latest_frame, 1)
                        gray_cam = cv2.cvtColor(frame_flipped, cv2.COLOR_BGR2GRAY)
                        faces = face_cascade_cam.detectMultiScale(gray_cam, 1.3, 5)

                        if len(faces) > 0:
                            x, y, w, h = faces[0]
                            face_area = frame_flipped[y:y+h, x:x+w]
                            face_pil = Image.fromarray(cv2.cvtColor(face_area, cv2.COLOR_BGR2RGB))
                            
                            mask = create_circular_mask((w, h))
                            face_pil.putalpha(mask)
                            
                            # 'cam' 폴더 생성
                            os.makedirs(output_folder, exist_ok=True)
                            # 파일 저장
                            face_pil.save(captured_face_path)
                            print(f"얼굴 캡처 완료! -> '{captured_face_path}'")
                            
                            # 상태를 '합성'으로 변경하고 GIF 처리 시작
                            game_state = "overlay"
                            pygame.display.set_caption("얼굴 합성 중...")

                            # GIF 분석 및 화면 크기 재설정
                            gif_frames, gif_face_locations = preprocess_gif(gif_path)
                            if not gif_frames:
                                running = False
                                break
                            
                            screen_size = gif_frames[0].get_size()
                            screen = pygame.display.set_mode(screen_size)
                            overlay_face_pil = Image.open(captured_face_path).convert("RGBA")
                            gif_frame_index = 0
                        else:
                            print("얼굴을 찾지 못했습니다. 다시 시도해주세요.")

        # --- 화면 그리기 ---
        if game_state == "capture":
            ret, cam_frame = cap.read()
            if ret:
                cam_frame_flipped = cv2.flip(cam_frame, 1)
                frame_rgb = cv2.cvtColor(cam_frame_flipped, cv2.COLOR_BGR2RGB)
                frame_surface = pygame.surfarray.make_surface(frame_rgb.swapaxes(0, 1))
                screen.blit(frame_surface, (0, 0))
                
                # 안내 문구 표시
                text_surf = font.render("얼굴을 맞추고 Spacebar를 누르세요", True, (255, 255, 0))
                screen.blit(text_surf, (10, 10))

        elif game_state == "overlay":
            current_gif_frame = gif_frames[gif_frame_index].copy()
            face_loc = gif_face_locations[gif_frame_index]
            
            if face_loc is not None:
                gx, gy, gw, gh = face_loc
                resized_face = overlay_face_pil.resize((gw, gh), Image.Resampling.LANCZOS)
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