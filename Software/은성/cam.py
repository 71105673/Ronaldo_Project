import cv2

def find_camera_indices_without_opening():
    """
    시스템에 연결된 카메라의 인덱스를 찾아 출력하지만,
    실제로 카메라 스트림을 열어 데이터를 가져오지는 않습니다.
    """
    available_indices = []
    max_cameras_to_check = 10 # 일반적으로 0부터 시작하여 10번 정도까지 탐색합니다.

    print(f"시스템에 연결된 카메라 인덱스를 찾는 중 (최대 {max_cameras_to_check}개)...")
    
    for i in range(max_cameras_to_check): 
        # cv2.VideoCapture(i)를 호출하여 카메라 객체를 생성하고,
        # isOpened()를 통해 연결 가능 여부만 확인합니다.
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"카메라 인덱스 {i}: 감지됨 (사용 가능)")
            available_indices.append(i)
            cap.release() # 카메라 객체를 즉시 해제합니다.
                          # 실제 스트림이 시작되지 않도록 합니다.
        else:
            # 이 메시지는 카메라가 없거나 접근할 수 없을 때 나타납니다.
            print(f"카메라 인덱스 {i}: 감지되지 않음 또는 접근 불가")
            # 연속으로 사용 불가로 나오면 더 이상 카메라가 없을 가능성이 높습니다.
            # 하지만 모든 인덱스를 확인하는 것이 더 확실합니다.
            
    return available_indices

if __name__ == "__main__":
    camera_indices = find_camera_indices_without_opening()
    
    if camera_indices:
        print("\n--- 감지된 카메라 인덱스 요약 ---")
        for idx in camera_indices:
            print(f"- {idx}")
        print("\n이 인덱스들을 사용하여 필요할 때 카메라를 열 수 있습니다 (예: cv2.VideoCapture(0)).")
    else:
        print("\n어떤 카메라도 감지되지 않았습니다. 카메라가 제대로 연결되어 있는지 확인해주세요.")