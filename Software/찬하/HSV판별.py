import cv2
import numpy as np

# 마우스 이벤트를 처리할 콜백 함수
def show_color_info(event, x, y, flags, param):
    global bgr_color, hsv_color_888, hsv_color_884
    
    if event == cv2.EVENT_MOUSEMOVE:
        # 프레임의 (y, x) 위치에서 BGR 색상 값을 가져옴
        bgr_color = frame[y, x]
        
        # BGR을 표준 HSV (OpenCV 기준)로 변환
        hsv_color_888 = cv2.cvtColor(np.uint8([[bgr_color]]), cv2.COLOR_BGR2HSV)[0][0]
        
        # ======================================================
        # ★★★ FPGA의 HSV 884 형식으로 변환하는 부분 ★★★
        # ======================================================
        # 1. Hue: OpenCV(0-179) -> Verilog(0-255) 스케일로 변환
        h_8bit = int(hsv_color_888[0] * (255.0 / 179.0))
        
        # 2. Saturation: 그대로 8비트(0-255) 사용
        s_8bit = hsv_color_888[1]
        
        # 3. Value: 8비트(0-255) -> 4비트(0-15)로 변환 (비트 쉬프트)
        v_4bit = hsv_color_888[2] >> 4
        # ======================================================
        
        hsv_color_884 = np.array([h_8bit, s_8bit, v_4bit])


# 초기 색상값 설정
bgr_color = np.array([0, 0, 0])
hsv_color_888 = np.array([0, 0, 0])
hsv_color_884 = np.array([0, 0, 0])

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("오류: 웹캠을 열 수 없습니다.")
    exit()

# 창 생성 및 마우스 콜백 함수 연결
cv2.namedWindow('HSV 884 Color Picker')
cv2.setMouseCallback('HSV 884 Color Picker', show_color_info)

print("마우스를 움직여 FPGA 환경과 동일한 HSV(8,8,4) 값을 확인하세요. ESC 키 누르면 종료.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)

    # 정보 표시를 위한 검은색 패널 생성
    info_panel = np.zeros((120, frame.shape[1], 3), dtype=np.uint8)

    # 선택된 색상 사각형
    cv2.rectangle(info_panel, (10, 20), (80, 90), (int(bgr_color[0]), int(bgr_color[1]), int(bgr_color[2])), -1)

    # BGR, HSV 텍스트 정보 표시
    bgr_text = f"BGR: [{bgr_color[0]}, {bgr_color[1]}, {bgr_color[2]}]"
    hsv_888_text = f"HSV (Standard): [{hsv_color_888[0]}, {hsv_color_888[1]}, {hsv_color_888[2]}]"
    hsv_884_text = f"HSV (8,8,4 for FPGA): [{hsv_color_884[0]}, {hsv_color_884[1]}, {hsv_color_884[2]}]" # 수정된 값 표시
    
    cv2.putText(info_panel, bgr_text, (100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(info_panel, hsv_888_text, (100, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(info_panel, hsv_884_text, (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 255, 100), 2) # FPGA용 값을 강조 표시
    
    # 원본 영상과 정보 패널을 하나로 합침
    combined_view = np.vstack((frame, info_panel))

    # 결과 보여주기
    cv2.imshow('HSV 884 Color Picker', combined_view)
    
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()