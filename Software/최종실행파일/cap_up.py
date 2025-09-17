import cv2

def find_camera_indexes():
    """컴퓨터에 연결된 모든 카메라의 인덱스를 찾아서 리스트로 반환합니다."""
    index = 0
    available_indexes = []
    
    print("📷 연결된 카메라를 검색합니다...")
    
    # 0번부터 9번까지의 인덱스를 순차적으로 테스트합니다.
    # 대부분의 경우 10개 이상의 카메라를 연결하지 않습니다.
    while index < 10:
        # 해당 인덱스의 카메라를 열려고 시도합니다.
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        
        # 카메라가 성공적으로 열렸는지 확인합니다.
        if cap.isOpened():
            print(f"  -> {index}번 인덱스에서 카메라를 찾았습니다.")
            available_indexes.append(index)
            # 확인 후 반드시 장치를 해제해야 합니다.
            cap.release()
            
        index += 1
        
    return available_indexes

if __name__ == "__main__":
    camera_list = find_camera_indexes()
    
    if not camera_list:
        print("\n❌ 연결된 카메라를 찾을 수 없습니다.")
        print("드라이버가 올바르게 설치되었는지 확인해주세요.")
    else:
        print(f"\n✨ 사용 가능한 카메라 인덱스: {camera_list}")