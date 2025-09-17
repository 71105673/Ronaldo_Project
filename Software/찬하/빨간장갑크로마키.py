import cv2
import numpy as np

def nothing(x):
    # 트랙바를 위한 빈 함수
    pass

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다.")
    exit()

# 배경 이미지 로드
try:
    background_image = cv2.imread('background.jpg')
    if background_image is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("오류: 'background.jpg' 파일을 찾을 수 없습니다. 임시 배경을 생성합니다.")
    ret, frame = cap.read()
    if ret:
        background_image = np.full(frame.shape, (20, 20, 20), dtype=np.uint8)
    else:
        exit()


# 설정 창 및 트랙바 생성 (빨간색용 초기값 설정)
cv2.namedWindow('Chroma Key Settings')
# --- 낮은 H 영역 ---
cv2.createTrackbar('H_lower1', 'Chroma Key Settings', 0, 179, nothing)   # Hue 최소값 (낮은 영역)
cv2.createTrackbar('H_upper1', 'Chroma Key Settings', 15, 179, nothing)  # Hue 최대값 (낮은 영역)
# --- 높은 H 영역 ---
cv2.createTrackbar('H_lower2', 'Chroma Key Settings', 160, 179, nothing) # Hue 최소값 (높은 영역)
cv2.createTrackbar('H_upper2', 'Chroma Key Settings', 179, 179, nothing) # Hue 최대값 (높은 영역)
# --- 채도(S)와 명도(V) ---
cv2.createTrackbar('S_lower', 'Chroma Key Settings', 100, 255, nothing) # Saturation 최소값
cv2.createTrackbar('V_lower', 'Chroma Key Settings', 100, 255, nothing) # Value 최소값

print("ESC 키를 누르면 종료됩니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 웹캠 영상 크기에 맞게 배경 이미지 리사이즈
    h, w, _ = frame.shape
    background_image_resized = cv2.resize(background_image, (w, h))

    # BGR -> HSV 색 공간으로 변환
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 트랙바에서 현재 설정값 가져오기
    h_lower1 = cv2.getTrackbarPos('H_lower1', 'Chroma Key Settings')
    h_upper1 = cv2.getTrackbarPos('H_upper1', 'Chroma Key Settings')
    h_lower2 = cv2.getTrackbarPos('H_lower2', 'Chroma Key Settings')
    h_upper2 = cv2.getTrackbarPos('H_upper2', 'Chroma Key Settings')
    s_lower = cv2.getTrackbarPos('S_lower', 'Chroma Key Settings')
    v_lower = cv2.getTrackbarPos('V_lower', 'Chroma Key Settings')
    
    # HSV에서 빨간색 범위로 2개의 마스크 생성
    lower_red1 = np.array([h_lower1, s_lower, v_lower])
    upper_red1 = np.array([h_upper1, 255, 255])
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    
    lower_red2 = np.array([h_lower2, s_lower, v_lower])
    upper_red2 = np.array([h_upper2, 255, 255])
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    
    # 2개의 마스크를 하나로 합침
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 마스크 보정 (노이즈 제거 및 부드럽게)
    kernel = np.ones((5, 5), np.uint8)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    mask_blurred = cv2.GaussianBlur(mask_closed, (5, 5), 0)

    # 마스크의 반대(인물 부분)를 구함
    inverse_mask = cv2.bitwise_not(mask_blurred)
    
    # 마스크를 이용해 영상 합성
    background = cv2.bitwise_and(background_image_resized, background_image_resized, mask=mask_blurred)
    foreground = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    result = cv2.add(background, foreground)

    # 결과 보여주기
    cv2.imshow('Original Camera', frame)
    cv2.imshow('Mask', mask_blurred) # 마스크 상태를 보고 싶을 때 유용
    cv2.imshow('Chroma Key Result (Red Glove)', result)
    
    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()