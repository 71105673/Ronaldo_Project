# ⚽ Ronaldo_Project  

## 📌 Git 사용법 (Push 방법)
- git add .
- git commit -m "업데이트 할 내용 요약"
- git push

### ⚠️ 주의사항
- 원본이 훼손되지 않도록 버전을 늘려가며 수정 후 Push
- 중요한 변경사항 발생 시 README.md → 필수 확인사항 섹션에 반드시 기록
- Pull하지 않은 상태(최신화 X) 인 상태로 Push 하면 오류 발생
- 따라서 매일 아침 Pull 하여 최신화 할 수 있도록
  
### ✅필수 확인사항
**코드 변경이나 확인 사항이 있다면 꼭 적기**
**경로까지 알려주면 감사**

FPGA 보드와 PC 의 파이썬 코드로 신호를 전달하려면?
하드웨어와 소프트웨어가 통신할수 있는 인터페이스를 정해야 한다.

```py
import serial
ser = serial.Serial('COM3', 9600)  # Basys 보드 연결된 포트 확인 필요
while True:
    if ser.in_waiting > 0:
        data = ser.readline().decode().strip()
        print("받은 데이터:", data)
``` 


```
-> 은성아 test_ver3.py가 자동전체화면임
multi state 에서의 기능

캠 2개 동작 시켜야 함.
캠에 그리드 생성.
상대편 몇번 그리드에 공을 칠지 설정.
모두 설정한 후, 캠 위에 카운트 3초
각각의 디스플레이에 그리드 번호와 손이 위치한
그리드가 일치하면 막은 영상,
그리드가 일치하지 않으면 공이 들어가는 영상 출력
이거를 5번 반복한 후에
winner state 로 이동
```
---


### 09.10 할 일
🎯 목표

화면 중앙 정렬

카메라 모듈 분리

카메라 작동 시 3초 카운트다운 → 종료 후 화면 아웃

---

👥 역할 분담

찬하: Single Game 제작

은성: Multi Game 제작

유경: 게임 설명 (Info) 작성

현준: 화면 전환 시 모션 + 사운드 (버튼 시 siuuu~)