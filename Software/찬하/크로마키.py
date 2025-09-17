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

# 배경 이미지 로드 (★★ 1. 경로 수정 ★★)
try:
    # 요청하신 경로로 배경 이미지를 불러옵니다.
    background_image = cv2.imread('../image/info/info_back2.jpg')
    if background_image is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("오류: '../image/info/info_back2.jpg' 파일을 찾을 수 없습니다. 임시 배경을 생성합니다.")
    ret, frame = cap.read()
    if ret:
        background_image = np.full(frame.shape, (20, 20, 20), dtype=np.uint8)
    else:
        exit()


# 설정 창 및 트랙바 생성 (녹색용 초기값)
cv2.namedWindow('Chroma Key Settings')
cv2.createTrackbar('H_lower', 'Chroma Key Settings', 40, 179, nothing)
cv2.createTrackbar('H_upper', 'Chroma Key Settings', 80, 179, nothing)
cv2.createTrackbar('S_lower', 'Chroma Key Settings', 70, 255, nothing)
cv2.createTrackbar('S_upper', 'Chroma Key Settings', 255, 255, nothing)
cv2.createTrackbar('V_lower', 'Chroma Key Settings', 50, 255, nothing)
cv2.createTrackbar('V_upper', 'Chroma Key Settings', 255, 255, nothing)

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
    h_lower = cv2.getTrackbarPos('H_lower', 'Chroma Key Settings')
    h_upper = cv2.getTrackbarPos('H_upper', 'Chroma Key Settings')
    s_lower = cv2.getTrackbarPos('S_lower', 'Chroma Key Settings')
    s_upper = cv2.getTrackbarPos('S_upper', 'Chroma Key Settings')
    v_lower = cv2.getTrackbarPos('V_lower', 'Chroma Key Settings')
    v_upper = cv2.getTrackbarPos('V_upper', 'Chroma Key Settings')
    
    # HSV에서 녹색 범위로 마스크 생성
    lower_green = np.array([h_lower, s_lower, v_lower])
    upper_green = np.array([h_upper, s_upper, v_upper])
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    
    # ★★★★ 2. 초록 테두리 제거 (마스크 축소) ★★★★
    # Erode 연산을 이용해 마스크를 약간 축소시켜 경계선을 깔끔하게 만듭니다.
    erode_kernel = np.ones((3, 3), np.uint8)
    mask_eroded = cv2.erode(mask, erode_kernel, iterations=1)
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

    # 마스크 보정 (노이즈 제거 및 부드럽게)
    # 이제 축소된 마스크(mask_eroded)를 사용해 후처리를 진행합니다.
    morph_kernel = np.ones((5, 5), np.uint8)
    mask_opened = cv2.morphologyEx(mask_eroded, cv2.MORPH_OPEN, morph_kernel)
    mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, morph_kernel)
    mask_blurred = cv2.GaussianBlur(mask_closed, (5, 5), 0)

    # 마스크의 반대(인물 부분)를 구함
    inverse_mask = cv2.bitwise_not(mask_blurred)
    
    # 마스크를 이용해 영상 합성
    background = cv2.bitwise_and(background_image_resized, background_image_resized, mask=mask_blurred)
    foreground = cv2.bitwise_and(frame, frame, mask=inverse_mask)
    result = cv2.add(background, foreground)

    # 결과 보여주기
    cv2.imshow('Original Camera', frame)
    cv2.imshow('Mask', mask_blurred) # 보정된 마스크 확인용
    cv2.imshow('Chroma Key Result', result)
    
    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()