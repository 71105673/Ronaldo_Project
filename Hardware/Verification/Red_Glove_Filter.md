### Red_Glove_Filter Verification

### 1. 검증할 Module
- 붉은색을 검출하는 Filter


<img width="600" height="538" alt="tb" src="https://github.com/user-attachments/assets/89afa166-8761-4717-a3bf-70287fee1627" />    

<p>

|       블록명       |                       핵심 역할 (Core Role)                       |                              세부 특징 및 구현 (Details & Implementation)                              |
|:---------------:|:------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| **interface** | DUT와 테스트벤치 간의 신호 연결 통로                                                       | - `virtual interface`로 Driver/Monitor에 핸들 전달<br>- DUT의 **RGB 5:6:5** 포맷에 맞춘 신호 선언 |
| **transaction** | 검증을 위한 최소 데이터 단위(패킷) 정의                                                    | - 1-Pixel에 해당하는 **RGB 5:6:5** 입력 및 `green_out` 출력 데이터 포함<br>- `randomize()`를 위한 `rand` 변수 선언 |
| **generator** | 테스트 시나리오(입력 자극) 생성                                                            | - `randomize()`를 호출하여 유효한 VGA 좌표 내에서 무작위 픽셀 데이터 생성<br>- `forever` 루프로 지속적인 데이터 생성 |
| **driver** | 생성된 Transaction을 DUT에 인가                                                            | - `posedge clk`에 동기화하여 인터페이스에 non-blocking (`<=`) 할당<br>- Mailbox를 통해 Generator로부터 데이터 수신 |
| **monitor** | DUT의 입력 및 출력 신호 감지                                                               | - `posedge clk` 이후 안정적인 시점에 신호 샘플링<br>- 감지한 데이터를 Transaction에 담아 Scoreboard로 전송 |
| **scoreboard** | DUT의 동작 정확성 검증                                                                     | - DUT와 동일한 로직의 **Reference Model (`predict_green`)** 내장<br>- Monitor로부터 받은 실제 출력과 Ref 모델의 예측값을 비교하여 **PASS/FAIL** 판정 및 집계 |
| **environment** | 검증 환경의 모든 컴포넌트 통합 및 제어                                                     | - 각 컴포넌트(Gen, Drv, Mon, Scb) 객체 생성 및 Mailbox 연결<br>- **Scoreboard의 처리 횟수**를 기준으로 시뮬레이션 시작 및 종료 제어 |
| **tb_top** | 시뮬레이션 최상위 모듈                                                                     | - 클럭(Clock) 생성<br>- DUT 및 `interface` 인스턴스화<br>- `environment` 실행 |

---

=========================== **데이터 처리 (Chromakey)** ===========================


**입력** : de, r_in (5b), g_in (6b), b_in (5b) 



**처리** : G값이 특정 임계값(G_THRESH) 이상이고, R/B값은 특정 최댓값(R/B_MAX) 미만이며, G값이 R/B값보다 일정량(OFFSET) 이상 큰지 판별 


**출력** : 녹색이면 green_out = 1, 아니면 green_out = 0 

</p>
  
<details>
    <summary>Chromakey_Verification</summary>

```verilog

```

</details>


