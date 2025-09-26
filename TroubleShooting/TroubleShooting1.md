# 통신 프로토콜 불일치 문제 및 해결 방법

## 문제 상황
- **소프트웨어 ↔ 하드웨어 간 프로토콜 불일치 발생**
  - **PC → FPGA**
    - 소프트웨어: 명령어 전송(0x20, 0x40, 0x60) → 이후 지속적 데이터 전송
    - 하드웨어: 명령어 전송(0xE1, 0xE2, 0xE3) → 데이터 1회 전송
  - **FPGA → PC**
    - 소프트웨어: 명령에 따른 상태 유지 및 데이터 연속 전송
    - 하드웨어: 헤더 포함 단일 데이터 전송

즉, 양측 간의 **명령어 체계와 데이터 전송 방식이 달라 통신 동기화 문제 발생**.

---

## 초기 FSM 구조
### 상태 정의
- **IDLE**  
  대기 상태. 각 플래그 신호(grid_flag, face_flag, kick_flag) 감지 시 전송 준비.
- **SEND_GRID**  
  grid 데이터 전송.
- **SEND_FACE**  
  face 데이터 전송 시작.
- **FACE_CNT**  
  face 데이터 카운트 진행(`face_data_cnt` 증가).
- **SEND_KICK**  
  kick 데이터 전송.
  
  <img src="./image/sw/Uart_fsm.png" width=700 height=400>

  <img src="./image/trouble_shooting/uart_v2.gif" width="600" height="500">

### 전이 조건
- `IDLE → SEND_GRID`: `grid_flag && grid_data_reg != grid_data`
- `IDLE → SEND_FACE`: `face_flag`
- `SEND_FACE → FACE_CNT`: `face_data_cnt + 1`
- `FACE_CNT → IDLE`: `face_data_cnt == 3`
- `IDLE → SEND_KICK`: `kick_flag && kick_data_reg != kick_data`

---

## 해결 방법
- **의견 종합 후 단일 프로토콜 확립**
- **데이터 패킷에 `헤더` 포함 → 연속 스트리밍 방식으로 통합**

<img src="./image/trouble_shooting/uart_protocol.png" width=700 height=300>

---

## 요약
- 초기에는 **명령어와 데이터 포맷이 서로 달라 통신 불일치 발생**
- **의견 종합 후, 데이터에 (헤더)를 포함하여 연속으로 스트리밍하는 방식으로 통신 프로토콜을 확정.**

