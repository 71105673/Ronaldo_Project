# Red_Glove_Grid_Selector Verification

## 1. 검증할 Module
- 주요 검증: Grd(영역) 선택
- 추가로 Red Glove Detector도 연결하여 rgb데이터로부터 Grid 선택까지 흐름 확인
<div align="center">
     <img width="2776" height="1131" alt="image" src="https://github.com/user-attachments/assets/17573a81-afa2-4311-a90f-e0ff6b105857" />
</div>

## 2. 검증 구조
- 검증 구조도
<div align="center">
<img width="600" height="538" alt="tb" src="https://github.com/user-attachments/assets/89afa166-8761-4717-a3bf-70287fee1627" />    
</div>
<br>

- 세부 내용
<p>

|       블록명       |                       핵심 역할 (Core Role)                       |                              세부 특징 및 구현 (Details & Implementation)                              |
|:---------------:|:------------------------------------------------------------------------------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------:|
| **interface** | DUT와 테스트벤치 간의 신호 연결 통로                                                       | - `virtual interface`로 Driver/Monitor에 핸들 전달<br>- clock과 DUT의 **RGB 4:4:4** 포맷에 맞춘 신호, 컨트롤 신호 등을 전달 |
| **transaction** | 검증을 위한 최소 데이터 단위(패킷) 정의                                                    | - 1-Pixel에 해당하는 **RGB 4:4:4** 입력 및 Selected_Grid 데이터 등을 포함<br>- `randomize()`를 위한 `rand` 변수 선언 |
| **generator** | 테스트 시나리오(입력 자극) 생성                                                            | - `randomize()`를 호출하여 무작위 픽셀 데이터 생성<br>-  gen_cnt만큼 random 데이터 생성<br>- 생성한 데이터를 transaction에 담아 mailbox로 put|
| **driver** | 생성된 Transaction을 DUT에 인가                                                            | - `posedge clk`에 동기화하여 인터페이스에 데이터 할당<br>- Mailbox를 통해 Generator로부터 데이터 수신 |
| **monitor** | DUT의 입력 및 출력 신호 감지                                                               | - `posedge clk` 이후 안정적인 시점에 신호 샘플링<br>- 감지한 데이터를 Transaction에 담아 mailbox를 이용하여 Scoreboard로 전송 |
| **scoreboard** | DUT의 동작 정확성 검증                                                                     | - DUT와 동일한 로직의 계산 결과와 Monitor로부터 받은 실제 출력을 비교하여 **PASS/FAIL** 판정 및 집계 |
| **environment** | 검증 환경의 모든 컴포넌트 통합 및 제어                                                     | - 각 컴포넌트(Gen, Drv, Mon, Scb) 객체 생성 및 Mailbox 연결<br>- 시뮬레이션 시작 및 fork된 proecss중 하나라도 종료시 종료 제어 |
| **tb_top** | 시뮬레이션 최상위 모듈                                                                     | - 클럭(Clock) 생성<br>- DUT 및 `interface` 인스턴스화<br>- `environment` 실행 |

---
<br>

## 3. 데이터 처리 (Red_Glove_Detector)


**입력** : r_in (4bit), g_in (4bit), b_in (4bit), DE, v_sync, x_pixel(10bit)


**처리** : 붉은색으로 감지된 픽셀의 수를 세어서 각 Grid마다 값을 누적 → v_sync가 falling edge일 때 가장 누적값이 큰 Grid를 선택하여 출력


**출력** : 가장 붉은 색이 많은 영역을 선택하여 1~5까지 selected_grid 데이터 출력, 붉은색의 면적이 일정 수치 이하일 시 selected_grid 0 출력

</p>
<br>

## 4. 검증 결과
- 검증 중간 출력
    - 정상적으로 가장 붉은색이 많은 Grid를 선택
      
      <img width="830" height="902" alt="image" src="https://github.com/user-attachments/assets/bbd77f02-159b-4d26-88e0-073483165db4" />

- 검증 최종 결과
    - 100개의 데이터 모두 PASS 확인
  <div align="center">
    <img width="1156" height="129" alt="image" src="https://github.com/user-attachments/assets/7aeab5d4-c1da-442d-82fe-ad437383ede3" />
  </div>

  <br>
## 5. 검증 Code
<details>
    <summary>Red_Glove_Detector_Verification_Code</summary>

```verilog

`timescale 1ns / 1ps

interface red_glove_intf;
    logic       pclk;
    logic       reset;
    logic [3:0] r_in;
    logic [3:0] g_in;
    logic [3:0] b_in;
    logic       v_sync;
    logic       DE;
    logic [9:0] x_pixel;
    logic [2:0] selected_grid;
    logic [3:0] r_out;
    logic [3:0] g_out;
    logic [3:0] b_out;
endinterface  //red_glove_intf

class transaction;
    rand bit [3:0] r_in;
    rand bit [3:0] g_in;
    rand bit [3:0] b_in;
    bit            v_sync;
    bit            DE;
    bit      [9:0] x_pixel;
    bit      [2:0] selected_grid;
    bit      [3:0] r_out;
    bit      [3:0] g_out;
    bit      [3:0] b_out;
endclass  //transaction

class generator;
    transaction tr;
    mailbox #(transaction) gen2drv_mbox;

    function new(mailbox#(transaction) gen2drv_mbox);
        this.gen2drv_mbox = gen2drv_mbox;
    endfunction  //new()

    task run(int gen_cnt);
        repeat (gen_cnt) begin
            for (int j = 0; j < 525; j++) begin  //y
                for (int i = 0; i < 800; i++) begin  //x
                    tr = new();
                    tr.randomize();
                    tr.DE      = ((i < 640) && (j < 480)) ? 1 : 0;
                    tr.v_sync  = !((j >= 490) && (j <= 492));
                    tr.x_pixel = i;
                    gen2drv_mbox.put(tr);
                    #40;  //25M
                end
            end
        end
    endtask  //
endclass  //generator

class driver;
    transaction tr;
    mailbox #(transaction) gen2drv_mbox;
    virtual red_glove_intf red_glove_if;

    function new(mailbox#(transaction) gen2drv_mbox,
                 virtual red_glove_intf red_glove_if);
        this.gen2drv_mbox = gen2drv_mbox;
        this.red_glove_if = red_glove_if;
    endfunction  //new()

    task run();
        forever begin
            gen2drv_mbox.get(tr);
            red_glove_if.r_in    = tr.r_in;
            red_glove_if.g_in    = tr.g_in;
            red_glove_if.b_in    = tr.b_in;
            red_glove_if.DE      = tr.DE;
            red_glove_if.v_sync  = tr.v_sync;
            red_glove_if.x_pixel = tr.x_pixel;
            @(posedge red_glove_if.pclk);
        end
    endtask  //
endclass  //driver

class monitor;
    transaction tr;
    mailbox #(transaction) mon2scr_mbox;
    virtual red_glove_intf red_glove_if;

    function new(mailbox#(transaction) mon2scr_mbox,
                 virtual red_glove_intf red_glove_if);
        this.mon2scr_mbox = mon2scr_mbox;
        this.red_glove_if = red_glove_if;
    endfunction  //new()

    task run();
        forever begin
            tr = new();
            @(posedge red_glove_if.pclk);
            #1;
            tr.r_in = red_glove_if.r_in;
            tr.g_in = red_glove_if.g_in;
            tr.b_in = red_glove_if.b_in;
            tr.DE = red_glove_if.DE;
            tr.v_sync = red_glove_if.v_sync;
            tr.x_pixel = red_glove_if.x_pixel;
            tr.selected_grid = red_glove_if.selected_grid;
            tr.r_out = red_glove_if.r_out;
            tr.g_out = red_glove_if.g_out;
            tr.b_out = red_glove_if.b_out;
            mon2scr_mbox.put(tr);
        end
    endtask  //
endclass  //monitor


class scoreboard;
    transaction tr;
    mailbox #(transaction) mon2scr_mbox;

    bit [3:0] max_val, min_val, delta;
    bit detect;

    bit [13:0] Grid[0:4], Grid_temp;
    bit [2:0] grid_pos;
    bit [2:0] grid_selected;

    bit [3:0] r_data, g_data, b_data;

    bit v_sync_temp;

    logic [20:0] success_cnt;

    function new(mailbox#(transaction) mon2scr_mbox);
        this.mon2scr_mbox = mon2scr_mbox;
        for (int i = 0; i < 5; i++) begin
            Grid[i] = 0;
        end
        grid_pos = 0;
        success_cnt=0;
    endfunction  //new()

    task red_detect();
        max_val = (tr.r_in >= tr.g_in && tr.r_in >= tr.b_in) ? tr.r_in :
                     (tr.g_in >= tr.b_in) ? tr.g_in : tr.b_in;
        min_val = (tr.r_in <= tr.g_in && tr.r_in <= tr.b_in) ? tr.r_in :
                     (tr.g_in <= tr.b_in) ? tr.g_in : tr.b_in;
        delta = max_val - min_val;
        detect = (tr.r_in == max_val) && (delta >= 7) && ((tr.r_in - tr.g_in) >= 6) && ((tr.r_in - tr.b_in) >= 6);
    endtask  //

    task count_grid();
        if (tr.DE) begin
            if (detect) begin
                grid_pos = (tr.x_pixel / 128);
                case (grid_pos)
                    0: Grid[0]++;
                    1: Grid[1]++;
                    2: Grid[2]++;
                    3: Grid[3]++;
                    4: Grid[4]++;
                endcase
            end
        end
    endtask  //

    task grid_selection();
        grid_selected = 1;
        Grid_temp = Grid[0];
        if (Grid[1] > Grid_temp) begin
            grid_selected = 2;
            Grid_temp = Grid[1];
        end
        if (Grid[2] > Grid_temp) begin
            grid_selected = 3;
            Grid_temp = Grid[2];
        end
        if (Grid[3] > Grid_temp) begin
            grid_selected = 4;
            Grid_temp = Grid[3];
        end
        if (Grid[4] > Grid_temp) begin
            grid_selected = 5;
            Grid_temp = Grid[4];
        end
        if (Grid_temp <= 500) begin
            grid_selected = 0;
        end
    endtask  //

    task grid_selector();
        if (tr.DE) begin
            case (grid_selected)
                1:
                if (tr.x_pixel < 128)
                    {r_data, g_data, b_data} = {4'd15, 4'd15, 4'd0};
                2:
                if (tr.x_pixel >= 128 && tr.x_pixel < 256)
                    {r_data, g_data, b_data} = {4'd15, 4'd15, 4'd0};
                3:
                if (tr.x_pixel >= 256 && tr.x_pixel < 384)
                    {r_data, g_data, b_data} = {4'd15, 4'd15, 4'd0};
                4:
                if (tr.x_pixel >= 384 && tr.x_pixel < 512)
                    {r_data, g_data, b_data} = {4'd15, 4'd15, 4'd0};
                5:
                if (tr.x_pixel >= 512 && tr.x_pixel < 640)
                    {r_data, g_data, b_data} = {4'd15, 4'd15, 4'd0};
            endcase
        end
    endtask  //

    task run();
        forever begin
            mon2scr_mbox.get(tr);
            
            // if (tr.x_pixel == 640) begin
            //     $display("Grid[4]", Grid[4]);
            // end
            

            if (!tr.v_sync && v_sync_temp) begin
                grid_selection();
                mon2scr_mbox.get(tr);

                if (tr.selected_grid == grid_selected) begin
                    $display(
                        "---------------------------------------------------------");
                    $display("PASS! : grid = %d, grid_val = %d",
                             tr.selected_grid, Grid_temp);
                    $display(
                        "---------------------------------------------------------");
                    success_cnt++;
                end else begin
                    $display(
                        "---------------------------------------------------------");
                    $display("FAIL! : grid was wrong!!!");
                    $display("filter grid = %d, tb grid = %d",
                             tr.selected_grid, grid_selected);
                    $display(
                        "Grid1 = %d,Grid2 = %d,Grid3 = %d,Grid4 = %d,Grid5 = %d",
                        Grid[0], Grid[1], Grid[2], Grid[3], Grid[4]);
                    $display(
                        "---------------------------------------------------------");
                end
            end

            if (tr.v_sync && !v_sync_temp) begin
                for (int i = 0; i < 5; i++) begin
                    Grid[i] = 0;
                end
                grid_pos = 0;
            end else begin
                count_grid();
            end

            red_detect();
            grid_selector();

            v_sync_temp = tr.v_sync;
        end
    endtask  //

    task pirnt_count();
        $display("|      Success Count : %d!!!       |", success_cnt);
    endtask //

endclass  //scoreboard



class environment;
    generator gen;
    driver drv;
    monitor mon;
    scoreboard scr;
    mailbox #(transaction) gen2drv_mbox;
    mailbox #(transaction) mon2scr_mbox;

    function new(virtual red_glove_intf red_glove_if);
        gen2drv_mbox = new();
        mon2scr_mbox = new();
        gen = new(gen2drv_mbox);
        drv = new(gen2drv_mbox, red_glove_if);
        mon = new(mon2scr_mbox, red_glove_if);
        scr = new(mon2scr_mbox);
    endfunction  //new()

    task run(int gen_cnt);
        fork
            gen.run(gen_cnt);
            drv.run();
            mon.run();
            scr.run();
        join_any
        #200;
        scr.pirnt_count();
        $finish();
    endtask  //
endclass  //environment

module tb_red_glove_grid_select ();
    environment env;
    red_glove_intf red_glove_if ();

    Red_Glove_Grid_Selector dut (
        .pclk         (red_glove_if.pclk),
        .reset        (red_glove_if.reset),
        .r_in         (red_glove_if.r_in),
        .g_in         (red_glove_if.g_in),
        .b_in         (red_glove_if.b_in),
        .v_sync       (red_glove_if.v_sync),
        .DE           (red_glove_if.DE),
        .x_pixel      (red_glove_if.x_pixel),
        .selected_grid(red_glove_if.selected_grid),
        .r_out        (red_glove_if.r_out),
        .g_out        (red_glove_if.g_out),
        .b_out        (red_glove_if.b_out)
    );

    always #20 red_glove_if.pclk = ~red_glove_if.pclk;

    initial begin
        red_glove_if.pclk  = 1;
        red_glove_if.reset = 1;
        #40;
        red_glove_if.reset = 0;
    end

    initial begin
        env = new(red_glove_if);
        #40;
        env.run(100);
    end

endmodule

```

</details>


