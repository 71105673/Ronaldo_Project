import cv2
import numpy as np

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다.")
    exit()

# 배경 이미지 로드 (★★ 경로를 실제 파일 위치에 맞게 수정하세요 ★★)
try:
    background_image = cv2.imread('../image/info/info_back2.jpg')
    if background_image is None:
        raise FileNotFoundError
except FileNotFoundError:
    print("오류: 배경 이미지를 찾을 수 없습니다. 임시 배경을 생성합니다.")
    ret, frame = cap.read()
    if ret:
        background_image = np.full(frame.shape, (20, 20, 20), dtype=np.uint8)
    else:
        # 웹캠 프레임을 읽어오지 못하면 종료
        exit()

# ★★★★ 트랙바 제거: 고정된 HSV 값 설정 ★★★★
# 크로마키로 제거할 색상 범위를 미리 지정합니다. (기본값: 녹색)
# 조명이나 환경에 따라 이 값을 직접 조절해야 할 수 있습니다.
lower_green = np.array([40, 70, 50])
upper_green = np.array([80, 255, 255])
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

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
    
    # 지정된 HSV 범위로 마스크 생성
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    
    # 초록 테두리 제거 (마스크 축소)
    erode_kernel = np.ones((3, 3), np.uint8)
    mask_eroded = cv2.erode(mask, erode_kernel, iterations=1)

    # 마스크 보정 (노이즈 제거 및 부드럽게)
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

    # 결과 보여주기 (최종 결과만 표시)
    # cv2.imshow('Original Camera', frame) # 원본 확인이 필요하면 주석 해제
    # cv2.imshow('Mask', mask_blurred)    # 마스크 확인이 필요하면 주석 해제
    cv2.imshow('Chroma Key Result', result)
    
    # ESC 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()