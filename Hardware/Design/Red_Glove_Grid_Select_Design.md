# Red Glove Grid Selector

## 1. 모듈 개요

**Red_Glove_Grid_Selector**는 FPGA 상에서 **빨간 장갑이 선택한 영역 추출**을 수행하는 최상위 모듈이다.

이 모듈은 카메라에서 입력된 영상의 RGB 데이터를 분석하여, 픽셀이 **붉은 장갑(붉은 영역)**인지 판별하고,<br>
붉은색일 경우에는 **붉은 색의 면적**을 세어 **가장 많은 면적**을 추출한다.

즉, 붉은 장갑의 면적으로 5개의 영역 중 하나를 선택하는 **영역 선택** 역할을 한다.

전체 Block Diagram
<img width="1824" height="708" alt="image" src="https://github.com/user-attachments/assets/6f99eda9-5d0f-4aad-8a33-efa634a368bb" />

---

## 2. 내부 블록 구성

Red_Glove_Grid_Selector 모듈은 크게 세 개의 하위 블록으로 구성된다.

1. **RedGlove_Detector**
    - 입력된 RGB 값이 “붉은 영역”인지 판별하는 블록
    - R 성분이 가장 크고, 최대 최소 값이 일정 수준 이상이고, R값이 G, B값보다 일정 수준 이상이면 **detect** 신호 생성
    - **detect=1** → 붉은색 판정
2. **Grid_Partition**
    - detect 신호를 기준으로 **붉은색이 가장 많은 영역** 선택
    - Grid(영역)값 누적: 유효한 픽셀 값인 DE신호와 detect신호가 1일시 x_pixel(화면의 가로 좌표)에 따라 값 누적
    - Grid(영역)값 초기화: v_sync 신호가 rising edge(한 프레임이 끝나는 타이밍)이면 누적된 값을 Grid 값을 초기화
    - Grid(영역) 선택: v_sync 신호가 falling edge(화면 출력 끝)이면 누적된 Grid을 바탕으로 **가장 많이 누적된 Grid 선택**
                      이때 최대값이 일정 수치 이하라면 선택되지 않은 것으로 판단
3. **select_grid**
    - 선택된 영역 확인 용도
    - Grid값에 따라 해당 Grid 영역 픽셀만 노란색으로 변경
    - **선택된 Grid 영역** → 노란색으로 변경, **선택되지 않은 영역** → 기존 pixel값 출력

---

## 3. 동작 원리

1. Framebuffer로부터 RGB(4:4:4) 데이터가 입력된다.
2. RedGlove_Detector에서 해당 픽셀이 붉은색인지 판별한다.
    - R 값이 최대값인지(G, B보다 큰지)
    - G 값이 R, B보다 충분히 큰지
    - 최대값과 최소값의 차이가 충분히 큰지
3. Grid_Partition이 붉은색이 가장 많은 영역을 선택한다.
4. select_grid는 선택된 Grid에 따라 선택된 영역만 노란색으로 변경한다.
5. 최종적으로, 선택된 영역만 노란색으로 칠해진 영상이 출력된다.

---

## 4. 핵심 특징

- **실시간 감지**: DE(Data Enable) 신호와 좌표(x), v_sync 신호를 이용하여 픽셀 단위로 붉은색 감지
- **유연한 임계값 조정**: 파라미터(SATURATION, DIFFERENCE)를 통해 붉은색 검출 민감도 조정 가능
- **간단한 구조**: 곱셈 없이 덧셈과 뺄셈, 크기 비교만을 사용하여 간단한 회로로 구성성

---

## 5. CODE

<details>
    <summary>Red_Glove_Grid_Selector_Code(TOP)</summary>

```verilog
`timescale 1ns / 1ps

module Red_Glove_Grid_Selector (
    input logic pclk,
    input logic reset,
    input logic [3:0] r_in,
    input logic [3:0] g_in,
    input logic [3:0] b_in,
    input logic v_sync,
    input logic DE,
    input logic [9:0] x_pixel,
    output logic [2:0] selected_grid,
    output logic [3:0] r_out,
    output logic [3:0] g_out,
    output logic [3:0] b_out
);

    logic red_glove_detect;

    RedGlove_Detector U_RedGlove_Detector (
        .r_data(r_in),
        .g_data(g_in),
        .b_data(b_in),
        .detect(red_glove_detect)
    );

    Grid_Partition U_Grid_Partition (
        .clk             (pclk),
        .reset           (reset),
        .v_sync          (v_sync),
        .DE              (DE),
        .red_glove_detect(red_glove_detect),
        .x_pixel         (x_pixel),
        .selected_grid   (selected_grid)      // 0 = 없음, 1~5 = grid
    );

    select_grid U_Selected_Grid (
        .DE(DE),
        .selected_grid(selected_grid),  // 0 = 없음, 1~5 = Grid
        .x_pixel(x_pixel),
        .r_in(r_in),
        .g_in(g_in),
        .b_in(b_in),
        .r_out(r_out),
        .g_out(g_out),
        .b_out(b_out)
    );

endmodule

```

</details>

<details>
    <summary>RedGlove_Detector_Code</summary>

```verilog
`timescale 1ns / 1ps

module RedGlove_Detector (
    input  logic [3:0] r_data,
    input  logic [3:0] g_data,
    input  logic [3:0] b_data,
    output logic       detect
);

    localparam SATURATION = 3;
    localparam DIFFERENCE = 2;

    logic [3:0] max_val, min_val, delta;

    assign max_val = (r_data >= g_data && r_data >= b_data) ? r_data :
                     (g_data >= b_data) ? g_data : b_data;

    assign min_val = (r_data <= g_data && r_data <= b_data) ? r_data :
                     (g_data <= b_data) ? g_data : b_data;

    assign delta = max_val - min_val;

   
    assign detect = (r_data == max_val) && (delta >= SATURATION) && 
        ((r_data - g_data) >= DIFFERENCE) && ((r_data - b_data) >= DIFFERENCE);

endmodule
```

</details>

<details>
    <summary>Grid_Partition_Code</summary>

```verilog
`timescale 1ns / 1ps

module Grid_Partition (
    input  logic       clk,
    input  logic       reset,
    input  logic       v_sync,
    input  logic       DE,
    input  logic       red_glove_detect,
    input  logic [9:0] x_pixel,
    output logic [2:0] selected_grid      // 0 = 없음, 1~5 = grid
);

    localparam SELECT_GRID_MIN = 500;

    logic [13:0] Grid[0:4];
    logic [2:0] grid_pos;
    logic [13:0] max01, max23, max0123;
    logic [2:0] grid_final;
    logic v_sync_delay;

    logic [13:0] max_val;

    assign selected_grid = grid_final;

    // v_sync delay
    always_ff @(posedge clk, posedge reset) begin
        if (reset) v_sync_delay <= 1'b0;
        else v_sync_delay <= v_sync;
    end

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            grid_final <= 3'd0;
        end else if (!v_sync && v_sync_delay) begin
            // 최대값 계산
            max_val = Grid[0];
            grid_final = 3'd1;  // 기본값: Grid 0

            if (Grid[1] > max_val) begin
                max_val = Grid[1];
                grid_final = 3'd2;
            end
            if (Grid[2] > max_val) begin
                max_val = Grid[2];
                grid_final = 3'd3;
            end
            if (Grid[3] > max_val) begin
                max_val = Grid[3];
                grid_final = 3'd4;
            end
            if (Grid[4] > max_val) begin
                max_val = Grid[4];
                grid_final = 3'd5;
            end

            // 모든 Grid가 0이면 장갑 없음
            if (max_val <= SELECT_GRID_MIN) grid_final = 3'd0;
        end
    end

    always_ff @(posedge clk, posedge reset) begin
        if (reset) begin
            Grid[0] <= 0;
            Grid[1] <= 0;
            Grid[2] <= 0;
            Grid[3] <= 0;
            Grid[4] <= 0;
        end else if (v_sync && !v_sync_delay) begin
            // 한 프레임 끝나면 초기화
            Grid[0] <= 0;
            Grid[1] <= 0;
            Grid[2] <= 0;
            Grid[3] <= 0;
            Grid[4] <= 0;
        end else if (DE && red_glove_detect) begin
            case (x_pixel[9:7])
                0: Grid[0] <= Grid[0] + 1;
                1: Grid[1] <= Grid[1] + 1;
                2: Grid[2] <= Grid[2] + 1;
                3: Grid[3] <= Grid[3] + 1;
                4: Grid[4] <= Grid[4] + 1;
            endcase
        end

    end


endmodule
```

</details>

<details>
    <summary>select_grid_Code</summary>

```verilog
`timescale 1ns / 1ps

module select_grid (
    input  logic       DE,
    input  logic [2:0] selected_grid,  // 0 = 없음, 1~5 = Grid
    input  logic [9:0] x_pixel,
    input  logic [3:0] r_in,
    input  logic [3:0] g_in,
    input  logic [3:0] b_in,
    output logic [3:0] r_out,
    output logic [3:0] g_out,
    output logic [3:0] b_out
);

    always_comb begin
        r_out = r_in;
        g_out = g_in;
        b_out = b_in;

        //if (DE && selected_grid != 0) begin
        if (DE) begin
            // selected_grid에 맞춰 노란색 강조
            case (selected_grid)
                1:
                if (x_pixel < 128) {r_out, g_out, b_out} = {4'd15, 4'd15, 4'd0};
                2:
                if (x_pixel >= 128 && x_pixel < 256)
                    {r_out, g_out, b_out} = {4'd15, 4'd15, 4'd0};
                3:
                if (x_pixel >= 256 && x_pixel < 384)
                    {r_out, g_out, b_out} = {4'd15, 4'd15, 4'd0};
                4:
                if (x_pixel >= 384 && x_pixel < 512)
                    {r_out, g_out, b_out} = {4'd15, 4'd15, 4'd0};
                5:
                if (x_pixel >= 512 && x_pixel < 640)
                    {r_out, g_out, b_out} = {4'd15, 4'd15, 4'd0};
                    
            endcase
        end
    end

endmodule

```

</details>



