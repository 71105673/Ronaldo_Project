# DAY 1 (25.09.10)

## 업데이트 내역

FPGA 보드와 PC 의 파이썬 코드로 신호를 전달하려면?<br>
하드웨어와 소프트웨어가 통신할수 있는 인터페이스를 정해야 한다.

```py
import serial
ser = serial.Serial('COM3', 9600)  # Basys 보드 연결된 포트 확인 필요
while True:
    if ser.in_waiting > 0:
        data = ser.readline().decode().strip()
        print("받은 데이터:", data)
``` 
받은 데이터로 이것저것 이제 상태를 넘기든 영상을 재생하든 하면 될듯!

---

```
multi state 에서의 기능

캠 2개 동작 시켜야 함.
캠1,2에 그리드 생성. 캠1은 키퍼, 캠2는 킥커

모두 설정한 후, 캠 위에 카운트 3초
각각의 디스플레이에 그리드 번호와 손이 위치한
그리드가 일치하면 막은 영상,
그리드가 일치하지 않으면 공이 들어가는 영상 출력
이거를 5번 반복한 후에
winner state 로 이동
```
