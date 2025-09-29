# SCCB 전체 구조 
<div align="center">

<img src="../image/SCCB_BlockDesign.png" width=800 height=400>

</div>

### SCCB_Config_ROM을 통해 Write할 레지스터 주소 와 데이터 저장 및 불러오기 
### CCB_Config를 통해 타이밍과 전송할 데이터 제어
### SCCB_Interface를 통해 타이밍에 맞게 SCL, SDA 송신
<br>

# SCCB Protocol
<div align="center">

<img src="../image/SCCB_Protocol.png" width=800 height=100>


</div>

- # IDLE : SCL = 1, SDA =1
### SCL = 1일떄 SDA 가 하강엣지인 순간  START ID → REG_ADDR → REG_DATA 순 8비트 송신후 마지막 비트는 don’t care 처리 (0,1 상관x)
- # ID : SLAVE 모듈을 ID로 구분 (7비트)
### 8번쨰 비트로 R/W 여부 (0 : Write, 1 : Read)
- # REG_ADDR
### SLAVE 모듈속 레지스터 주소 값 전송
- # REG_DATA : 레지스터에 Write할 Data 전송
### 전송을 끝낼때는 SCL = 1일떄  상승엣지인 순간 DONE