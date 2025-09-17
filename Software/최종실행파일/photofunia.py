import pygame
import cv2
import numpy as np
from PIL import Image, ImageDraw
import imageio
import os

def create_circular_mask(size):
    """
    원형 마스크를 생성하여 PIL 이미지로 반환합니다.
    (이 함수는 원본 코드에 없었지만, 원활한 실행을 위해 추가되었습니다.)
    """
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
        # RGBA가 아닌 경우 변환
        if frame_data.shape[2] == 3:
            frame_data = cv2.cvtColor(frame_data, cv2.COLOR_RGB2RGBA)
            
        frame_pil = Image.fromarray(frame_data).convert("RGBA")
        frame_surface = pygame.image.fromstring(frame_pil.tobytes(), frame_pil.size, "RGBA")
        frames.append(frame_surface)

        frame_cv = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGBA2BGR)
        gray = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            main_face = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)[0]
            face_locations.append(main_face)
        else:
            face_locations.append(None)

    print("GIF 분석 완료!")
    return frames, face_locations

def main():
    # --- 설정 ---
    gif_path = '../image/final_ronaldo/goalkeeper_win.gif'
    # 캡처하는 대신, 미리 준비된 얼굴 이미지 경로
    output_folder = 'cam'
    captured_face_path = os.path.join(output_folder, 'captured_face.png')

    # --- 초기화 ---
    pygame.init()

    # 얼굴 이미지가 있는지 확인
    if not os.path.exists(captured_face_path):
        print(f"오류: 합성할 얼굴 이미지 파일('{captured_face_path}')을 찾을 수 없습니다.")
        print("프로그램을 실행하기 전에 'cam' 폴더에 'captured_face.png' 파일을 준비해주세요.")
        return

    # GIF 분석 및 Pygame 화면 설정
    gif_frames, gif_face_locations = preprocess_gif(gif_path)
    if not gif_frames:
        pygame.quit()
        return

    screen_size = gif_frames[0].get_size()
    screen = pygame.display.set_mode(screen_size)
    pygame.display.set_caption("얼굴 합성 GIF")

    # 합성할 얼굴 이미지 불러오기
    try:
        overlay_face_pil = Image.open(captured_face_path).convert("RGBA")
    except Exception as e:
        print(f"오류: 얼굴 이미지 파일을 불러오는 중 문제가 발생했습니다: {e}")
        pygame.quit()
        return

    gif_frame_index = 0
    running = True
    clock = pygame.time.Clock()

    # --- 메인 루프 ---
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                running = False

        # --- 화면 그리기 (GIF 합성) ---
        current_gif_frame = gif_frames[gif_frame_index].copy()
        face_loc = gif_face_locations[gif_frame_index]

        if face_loc is not None:
            gx, gy, gw, gh = face_loc
            # 얼굴 크기에 맞춰 합성할 이미지 리사이즈
            resized_face = overlay_face_pil.resize((gw, gh), Image.Resampling.LANCZOS)
            face_surface = pygame.image.fromstring(resized_face.tobytes(), resized_face.size, "RGBA")
            # GIF 프레임 위에 얼굴 합성
            current_gif_frame.blit(face_surface, (gx, gy))

        screen.blit(current_gif_frame, (0, 0))
        
        # 다음 GIF 프레임으로 이동
        gif_frame_index = (gif_frame_index + 1) % len(gif_frames)

        pygame.display.flip()
        clock.tick(30) # GIF 속도 조절

    pygame.quit()

if __name__ == '__main__':
    main()