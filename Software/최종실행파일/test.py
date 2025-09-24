import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# --- Verilog 로직과 동일한 변환 함수 (변경 없음) ---
def verilog_rgb_to_ycrcb(r_8bit, g_8bit, b_8bit):
    """Verilog 모듈의 RGB to YCbCr 변환 로직을 정확히 재현합니다."""
    R = r_8bit.astype(np.int64)
    G = g_8bit.astype(np.int64)
    B = b_8bit.astype(np.int64)
    
    cb_acc = -43 * R - 85 * G + 128 * B
    cr_acc = 128 * R - 107 * G - 21 * B
    
    Cb = 128 + (cb_acc // 256)
    Cr = 128 + (cr_acc // 256)

    Cb = np.clip(Cb, 0, 255)
    Cr = np.clip(Cr, 0, 255)
    
    return Cb.astype(np.uint8), Cr.astype(np.uint8)

# --- 실시간 감지에 사용될 파라미터를 저장할 전역 변수 ---
skin_params = {}
params_calculated = False

# --- 결과 분석 및 파라미터 업데이트 함수 ---
def analyze_and_update_params(roi_cb, roi_cr):
    """추출된 Cb, Cr 값으로 최적 범위를 계산하고 전역 파라미터를 업데이트합니다."""
    global skin_params, params_calculated
    
    if roi_cb.size == 0 or roi_cr.size == 0:
        print("경고: 선택된 영역에 픽셀이 없습니다.")
        return

    # 최적 범위 계산
    skin_params['cb_min'] = np.min(roi_cb)
    skin_params['cb_max'] = np.max(roi_cb)
    skin_params['cr_min'] = np.min(roi_cr)
    skin_params['cr_max'] = np.max(roi_cr)
    params_calculated = True

    print("\n" + "✨ 최적 범위 계산 완료 (실시간 적용 시작) ✨".center(50, "-"))
    print("아래 값을 Verilog 코드의 localparam에 복사하세요:")
    print(f"localparam [7:0] CB_MIN = 8'd{skin_params['cb_min']};")
    print(f"localparam [7:0] CB_MAX = 8'd{skin_params['cb_max']};")
    print(f"localparam [7:0] CR_MIN = 8'd{skin_params['cr_min']};")
    print(f"localparam [7:0] CR_MAX = 8'd{skin_params['cr_max']};")
    print("-" * 52 + "\n")

    # (선택적) 상세 분석을 위한 그래프 표시
    plt.figure(figsize=(8, 6))
    ax = plt.gca()
    ax.scatter(roi_cb.flatten(), roi_cr.flatten(), alpha=0.5, s=8, label='Selected Skin Pixels')
    rect = patches.Rectangle((skin_params['cb_min'], skin_params['cr_min']), 
                             skin_params['cb_max'] - skin_params['cb_min'], 
                             skin_params['cr_max'] - skin_params['cr_min'],
                             linewidth=2, edgecolor='r', facecolor='none', label='Optimal Boundary')
    ax.add_patch(rect)
    ax.set_title('Cb-Cr Distribution of Selected Region')
    ax.set_xlabel('Cb Component')
    ax.set_ylabel('Cr Component')
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, 255)
    ax.set_ylim(0, 255)
    plt.show(block=False) # block=False로 설정하여 프로그램이 멈추지 않도록 함

# --- 마우스 이벤트 처리 (변경 없음) ---
drawing = False
roi_start, roi_end = (-1, -1), (-1, -1)
def draw_roi(event, x, y, flags, param):
    global roi_start, roi_end, drawing
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_start = (x, y)
        roi_end = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi_end = (x, y)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        roi_end = (x, y)

# --- 메인 실행 함수 ---
def main():
    global roi_start, roi_end
    
    cap = cv2.VideoCapture(1) # 0: 기본 카메라, 다른 카메라는 1, 2...
    if not cap.isOpened():
        print("오류: 카메라를 열 수 없습니다.")
        return

    window_name_feed = "Camera Feed - Drag ROI, 'c' to calculate, 'q' to quit"
    window_name_result = "Real-time Skin Detection"
    cv2.namedWindow(window_name_feed)
    cv2.setMouseCallback(window_name_feed, draw_roi)

    print("사용법:")
    print("1. [Camera Feed] 창에서 피부 영역을 마우스로 드래그하여 선택하세요.")
    print("2. 'c' 키를 눌러 기준 파라미터를 계산하면 실시간 감지가 시작됩니다.")
    print("3. 'q' 키를 눌러 프로그램을 종료합니다.")

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.flip(frame, 1) # 좌우 반전
        display_frame = frame.copy()

        # ROI 선택 사각형 그리기
        if roi_start != (-1, -1) and roi_end != (-1, -1):
            cv2.rectangle(display_frame, roi_start, roi_end, (0, 255, 0), 2)
        
        cv2.imshow(window_name_feed, display_frame)

        # 실시간 피부 감지 (파라미터가 계산된 후에만 동작)
        if params_calculated:
            # 전체 프레임을 RGB로 변환 후 YCbCr 계산
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            r, g, b = frame_rgb[..., 0], frame_rgb[..., 1], frame_rgb[..., 2]
            cb_plane, cr_plane = verilog_rgb_to_ycrcb(r, g, b)
            
            # Cb, Cr 범위 조건으로 마스크 생성
            mask_cb = (cb_plane >= skin_params['cb_min']) & (cb_plane <= skin_params['cb_max'])
            mask_cr = (cr_plane >= skin_params['cr_min']) & (cr_plane <= skin_params['cr_max'])
            skin_mask = (mask_cb & mask_cr)
            
            # boolean 마스크를 흑백 이미지로 변환 (True->255, False->0)
            result_image = (skin_mask.astype(np.uint8) * 255)
            cv2.imshow(window_name_result, result_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('c'):
            x1, y1 = min(roi_start[0], roi_end[0]), min(roi_start[1], roi_end[1])
            x2, y2 = max(roi_start[0], roi_end[0]), max(roi_start[1], roi_end[1])
            
            if x1 < x2 and y1 < y2:
                roi = frame[y1:y2, x1:x2]
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                r, g, b = roi_rgb[..., 0], roi_rgb[..., 1], roi_rgb[..., 2]
                cb_plane, cr_plane = verilog_rgb_to_ycrcb(r, g, b)
                analyze_and_update_params(cb_plane, cr_plane)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()