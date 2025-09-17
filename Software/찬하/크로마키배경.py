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

# 배경 이미지 로드 (요청하신 경로로 수정)
try:
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

# 설정 창 및 트랙바 생성
cv2.namedWindow('Chroma Key Settings')
cv2.createTrackbar('H_lower', 'Chroma Key Settings', 40, 179, nothing)
cv2.createTrackbar('H_upper', 'Chroma Key Settings', 80, 179, nothing)
cv2.createTrackbar('S_lower', 'Chroma Key Settings', 70, 255, nothing)
cv2.createTrackbar('V_lower', 'Chroma Key Settings', 50, 255, nothing)
# ★★★ 1. 테두리 보정 강도 조절용 트랙바 추가 ★★★
cv2.createTrackbar('Spill_Strength', 'Chroma Key Settings', 50, 100, nothing)

print("ESC 키를 누르면 종료됩니다.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    background_image_resized = cv2.resize(background_image, (w, h))

    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    h_lower = cv2.getTrackbarPos('H_lower', 'Chroma Key Settings')
    h_upper = cv2.getTrackbarPos('H_upper', 'Chroma Key Settings')
    s_lower = cv2.getTrackbarPos('S_lower', 'Chroma Key Settings')
    v_lower = cv2.getTrackbarPos('V_lower', 'Chroma Key Settings')
    
    lower_green = np.array([h_lower, s_lower, v_lower])
    upper_green = np.array([h_upper, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)
    
    # 마스크 후처리 (노이즈 제거, 구멍 메우기)
    kernel = np.ones((5, 5), np.uint8)
    mask_opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_final = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
    
    # ★★★ 2. 초록 테두리 색상 보정 (Green Spill 제거) ★★★
    
    # 1. 피사체(인물)의 외곽선 찾기 (테두리 영역 생성)
    inverse_mask = cv2.bitwise_not(mask_final)
    # 외곽선을 살짝 팽창(dilate)시켜 테두리 영역을 좀 더 넓게 잡음
    dilated_inverse_mask = cv2.dilate(inverse_mask, kernel, iterations=2)
    # 팽창된 외곽선에서 원래 외곽선을 빼서 순수한 '테두리' 부분만 추출
    spill_mask = cv2.subtract(dilated_inverse_mask, inverse_mask)

    # 2. 원본 프레임 복사 후, 테두리 영역에만 색상 보정 적용
    spill_corrected_frame = frame.copy()
    
    # 트랙바 값으로 보정 강도 조절 (0.0 ~ 1.0)
    strength = cv2.getTrackbarPos('Spill_Strength', 'Chroma Key Settings') / 100.0

    # spill_mask에서 0이 아닌(흰색) 픽셀들의 좌표를 가져옴
    spill_coords = np.where(spill_mask != 0)
    
    # 해당 좌표의 B, G, R 값을 가져옴
    b = spill_corrected_frame[spill_coords][:, 0].astype(np.float32)
    g = spill_corrected_frame[spill_coords][:, 1].astype(np.float32)
    r = spill_corrected_frame[spill_coords][:, 2].astype(np.float32)

    # 녹색(g) 값을 파란색(b)과 빨간색(r)의 평균값에 가깝게 보정
    # strength가 1이면 완전히 보정, 0이면 보정 안함
    new_g = g * (1 - strength) + ((b + r) / 2) * strength
    
    # 보정된 녹색 값을 원래 위치에 다시 적용
    spill_corrected_frame[spill_coords] = np.vstack((b, new_g.astype(np.uint8), r)).T
    
    # ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★

    # 최종 합성
    final_mask_blurred = cv2.GaussianBlur(mask_final, (5, 5), 0)
    final_inverse_mask = cv2.bitwise_not(final_mask_blurred)

    background = cv2.bitwise_and(background_image_resized, background_image_resized, mask=final_mask_blurred)
    # 보정된 프레임(spill_corrected_frame)을 사용해 최종 합성
    foreground = cv2.bitwise_and(spill_corrected_frame, spill_corrected_frame, mask=final_inverse_mask)
    result = cv2.add(background, foreground)

    cv2.imshow('Original Camera', frame)
    cv2.imshow('Mask', spill_mask) # 테두리(spill_mask)가 잘 잡히는지 확인용
    cv2.imshow('Chroma Key Result', result)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()