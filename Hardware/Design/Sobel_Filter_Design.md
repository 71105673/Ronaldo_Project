# Sobel_Filter

## 1. 모듈 개요

**Sobel_Filter**는 FPGA 상에서 **물체의 edge를 추출하는** 최상위 모듈이다.

이 모듈은 카메라에서 입력된 영상의 RGB 데이터를 분석하여, 픽셀이 가지는 luminance 값을 판별하고,
그 값이 255보다 큰 경우에는 edge로 판단한다. edge가 되면 흰색, 아니면 검은색으로 디스플레이에 출력된다.

---

## 2. 내부 블록 구성

Sobel_Filter 모듈은 다음과 같이 구성된다.

1. **Line Buffer**
    - 입력된 픽셀 값에 대해 3x3 window값을 행렬곱 연산합니다.
    - top, mid, cur값을 Line Buffer 2개를 사용해 읽기와 쓰기 동작을 반복합니다.
    
2. **Gradient X, Gradient Y**
    - Sobel 마스크를 이용해 Gradient X, Gradient Y를 계산
    - 앞서 구한 Gradient X,Y에 대해 절댓값 Ax, Ay를 구하고 절댓값의 합 mag값을 구함
    
3. **edge extraction**
    - mag값이 8비트의 최대값 255를 초과하면 edge로 판단하고 아니면 0(검정)으로 판단
    - 
---

## 3. 동작 원리

1. RGB 4비트 → 8비트 확장 및 루마 계산
2. 2라인 버퍼 토글로 3라인 확보
3. 3×3 윈도우 구축(가로 시프트 3탭)
4. Sobel 기울기 계산 (곱셈 없이 시프트·가감으로 구현)
5. 에지 크기 계산 및 8비트 포화
6. 출력

## 4. 핵심 특징

- **실시간 에지 검출**: DE와 좌표(x_in,y_in)에 따라 1클럭/픽셀 스트림 처리. 3×3 윈도우가 성립할 때마다 |Gx|+|Gy|로 에지 강도 산출.
- **경량 연산(Sobel 핵심부 무곱셈)**: Sobel 가중치의 ±2는 시프트(<<1) 로 구현, 나머지는 덧셈·뺄셈만 사용해 멀티플라이어 없이 동작(루마 변환은 상수계수 곱).
- **라인 버퍼 토글 구조**: 2개의 라인버퍼(lineA/B)를 라인 단위로 토글해 3라인(위/중/현재) 윈도우를 유지, 메모리 사용 최소화.

### 3x3 Kernel 
<div align="center">
<img width="800" height="450" alt="image" src="https://github.com/user-attachments/assets/bffad808-9cca-422f-bda8-c4b8c65d6e82" />
</div>
<br>

### Difference of Gx, Gy  
<div align="center">
<img width="800" height="250" alt="img" src="https://github.com/user-attachments/assets/d52d404f-47e9-48a9-a1e2-0da477338030" />
</div>
<br>

### Port Diagram

<div align="center">
  <img width="800" height="300" alt="image" src="https://github.com/user-attachments/assets/80e3eb82-fe2d-447b-b895-b8cc13cbc132" />
</div>
<br>

### Result Picture

| **Before** | **After** |
| :---: | :---: |
| <img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/fab10ea6-704e-4fc1-a5ae-546145277391" /> | <img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/7fca7a9f-b6a5-4ebe-b480-cca7d6aecf7f" /> |


## 5. CODE

<details>
    <summary> 🎨 Sobel Filter </summary>

```verilog
`timescale 1ns / 1ps

module SobelFilter #(
    parameter int WIDTH = 640
) (
    input logic clk,
    input logic rst,  // active-high reset
    input logic den,  // enable (pixel valid)

    input logic [9:0] x_in,
    input logic [9:0] y_in,
    input logic [3:0] r_in,
    input logic [3:0] g_in,
    input logic [3:0] b_in,

    output logic [3:0] r_out,
    output logic [3:0] g_out,
    output logic [3:0] b_out
);

    // ------------------------------------------------------------
    // 0) RGB(4b) -> 8b 확장 후 정수 루마: Y ≈ (77R + 150G + 29B) >> 8
    //    (77/150/29 ≈ 0.299/0.587/0.114)
    // ------------------------------------------------------------
    wire  [ 7:0] R8 = {r_in, r_in};
    wire  [ 7:0] G8 = {g_in, g_in};
    wire  [ 7:0] B8 = {b_in, b_in};

    logic [15:0] y_mul;  // 최대 255*(77+150+29)=65280 < 2^16
    wire  [ 7:0] Y8;

    always_comb begin
        y_mul = R8 * 8'd77 + G8 * 8'd150 + B8 * 8'd29;
    end
    assign Y8 = y_mul[15:8];

    // ------------------------------------------------------------
    // 1) 라인버퍼 2개 토글 + 프레임/라인 시작 정리
    // ------------------------------------------------------------
    logic [7:0] lineA[0:WIDTH-1];
    logic [7:0] lineB[0:WIDTH-1];
    logic sel;  // 현재 라인 write 타겟
    wire line_start = den && (x_in == 10'd0);
    wire frame_start = den && (x_in == 10'd0) && (y_in == 10'd0);

    // sel: 프레임 시작에 0으로 재동기화 → 프레임 경계 잔상/삐뚤어짐 방지
    always_ff @(posedge clk, posedge rst) begin
        if (rst) sel <= 1'b0;
        else if (frame_start) sel <= 1'b0;
        else if (line_start) sel <= ~sel;
    end

    // 비동기 읽기 (이전 두 라인: top=y-2, mid=y-1)
    logic [7:0] top_rd, mid_rd;
    always_comb begin
        if (sel == 1'b0) begin
            top_rd = lineB[x_in];
            mid_rd = lineA[x_in];
        end else begin
            top_rd = lineA[x_in];
            mid_rd = lineB[x_in];
        end
    end

    // 동기 쓰기 (현재 라인)
    always_ff @(posedge clk) begin
        if (!rst && den) begin
            if (sel == 1'b0) lineB[x_in] <= Y8;
            else lineA[x_in] <= Y8;
        end
    end

    // ------------------------------------------------------------
    // 2) 3x3 윈도우 (가로 3탭 시프트)
    // ------------------------------------------------------------
    logic [7:0] top_w0, top_w1, top_w2;
    logic [7:0] mid_w0, mid_w1, mid_w2;
    logic [7:0] cur_w0, cur_w1, cur_w2;

    always_ff @(posedge clk) begin
        if (rst) begin
            top_w0 <= 0;
            top_w1 <= 0;
            top_w2 <= 0;
            mid_w0 <= 0;
            mid_w1 <= 0;
            mid_w2 <= 0;
            cur_w0 <= 0;
            cur_w1 <= 0;
            cur_w2 <= 0;
        end else if (den) begin
            // 좌→우: [0]=x-2, [1]=x-1, [2]=x
            top_w0 <= top_w1;
            top_w1 <= top_w2;
            top_w2 <= top_rd;
            mid_w0 <= mid_w1;
            mid_w1 <= mid_w2;
            mid_w2 <= mid_rd;
            cur_w0 <= cur_w1;
            cur_w1 <= cur_w2;
            cur_w2 <= Y8;

            // 경계 초기화 (프레임/라인 시작)
            if (frame_start || line_start) begin
                top_w0 <= 0;
                top_w1 <= 0;
                top_w2 <= top_rd;
                mid_w0 <= 0;
                mid_w1 <= 0;
                mid_w2 <= mid_rd;
                cur_w0 <= 0;
                cur_w1 <= 0;
                cur_w2 <= Y8;
            end
        end
    end

    // ------------------------------------------------------------
    // 3) Sobel: |Gx|+|Gy| → 8b 포화
    //    (시프트/가감으로 ×2 구현, 곱셈 없음)
    // ------------------------------------------------------------
    logic signed [12:0] Gx, Gy;
    logic [12:0] Ax, Ay;
    logic [13:0] mag;
    logic [ 7:0] edge8;
    logic [ 7:0] edge8_d1;

    // 유효: 윈도우 성립(x>=2,y>=2) && DE
    wire         valid_now = den && ((x_in >= 10'd2) && (y_in >= 10'd2));

    // 유효 신호 2단 파이프 (데이터와 정확히 정렬)
    logic valid_d1, valid_d2, valid_d3, valid_d4, valid_d5, valid_d6;

    always_ff @(posedge clk) begin
        if (rst) begin
            Gx <= '0;
            Gy <= '0;
            Ax <= '0;
            Ay <= '0;
            mag <= '0;
            edge8 <= '0;
            valid_d1 <= 1'b0;
            valid_d2 <= 1'b0;
            valid_d3 <= 1'b0;
            valid_d4 <= 1'b0;
            valid_d5 <= 1'b0;
            valid_d6 <= 1'b0;
        end else begin
            // Gx = [-1 0 +1; -2 0 +2; -1 0 +1]
            Gx <= -$signed(
                {5'd0, top_w0}
            ) + $signed(
                {5'd0, top_w2}
            ) - $signed(
                {4'd0, mid_w0, 1'b0}
            ) + $signed(
                {4'd0, mid_w2, 1'b0}
            ) - $signed(
                {5'd0, cur_w0}
            ) + $signed(
                {5'd0, cur_w2}
            );

            // Gy = [+1 +2 +1; 0 0 0; -1 -2 -1]
            Gy <= -$signed(
                {5'd0, top_w0}
            ) - $signed(
                {4'd0, top_w1, 1'b0}
            ) - $signed(
                {5'd0, top_w2}
            ) + $signed(
                {5'd0, cur_w0}
            ) + $signed(
                {4'd0, cur_w1, 1'b0}
            ) + $signed(
                {5'd0, cur_w2}
            );

            // |Gx|, |Gy|, 합의 8비트 포화
            Ax <= Gx[12] ? (~Gx + 13'd1) : Gx;
            Ay <= Gy[12] ? (~Gy + 13'd1) : Gy;
            mag <= Ax + Ay;
            edge8 <= (mag[13] || (mag > 14'd255)) ? 8'hFF : mag[7:0];

            // ★★★ 정렬 포인트: 출력은 edge8이 입력 대비 2클럭 늦게 나오므로
            //      valid도 2클럭 지연시켜 정확히 맞춘다.
            valid_d1 <= valid_now;  // +1 clk
            valid_d2 <= valid_d1;  // +2 clk
            valid_d3 <= valid_d2;
            valid_d4 <= valid_d3;
            valid_d5 <= valid_d4;
            valid_d6 <= valid_d5;
        end
    end


    // ------------------------------------------------------------
    // 4) 출력: valid_d2에 맞춰 RGB=상위 4비트
    // ------------------------------------------------------------
    
    //always_ff @(posedge clk) begin
    //   if (rst) begin
    //       r_out <= 4'd0; g_out <= 4'd0; b_out <= 4'd0;
    //   end else if (valid_now) begin
    //       r_out <= edge8[7:4];
    //       g_out <= edge8[7:4];
    //       b_out <= edge8[7:4];
    //   end else begin
    //       r_out <= 4'd0; g_out <= 4'd0; b_out <= 4'd0;
    //   end
    //end

     assign r_out = rst ? 4'd0 : (valid_d6 ? edge8[7:4] : 4'd0);
     assign g_out = rst ? 4'd0 : (valid_d6 ? edge8[7:4] : 4'd0);
     assign b_out = rst ? 4'd0 : (valid_d6 ? edge8[7:4] : 4'd0);


endmodule
```

</details>










