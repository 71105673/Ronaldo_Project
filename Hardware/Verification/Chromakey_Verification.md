<p>
<img width="600" height="538" alt="tb" src="https://github.com/user-attachments/assets/89afa166-8761-4717-a3bf-70287fee1627" />    
</p>
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


=========================== **데이터 처리 (Chromakey)** ===========================


**입력** : de, r_in (5b), g_in (6b), b_in (5b) 



**처리** : G값이 특정 임계값(G_THRESH) 이상이고, R/B값은 특정 최댓값(R/B_MAX) 미만이며, G값이 R/B값보다 일정량(OFFSET) 이상 큰지 판별 


**출력** : 녹색이면 green_out = 1, 아니면 green_out = 0 

</p>

## Top module
```verilog
`timescale 1ns / 1ps

module Chromakey_Filter (
    input  logic       DE,
    input  logic [9:0] x,
    input  logic [9:0] y,
    input  logic [4:0] i_r,
    input  logic [5:0] i_g,
    input  logic [4:0] i_b,
    output logic       green,
    output logic [3:0] r_port,
    output logic [3:0] g_port,
    output logic [3:0] b_port
);

    logic [16:0] bg_addr;
    logic [15:0] bg_data;

    ImgReader U_Chromakey_Reader (
        .DE    (DE),
        .x     (x),
        .y     (y),
        .addr  (bg_addr),
        .data  (bg_data),
        .r_port(r_port),
        .g_port(g_port),
        .b_port(b_port)
    );

    BackgroundROM U_BACKROM (
        .raddr(bg_addr),
        .data (bg_data)
    );

    GreenFilter_RGB U_ChromakeyFilter_RGB(
        .i_r(i_r),
        .i_g(i_g),
        .i_b(i_b),
        .green(green)
    );
endmodule

////////////////////////////////////////////////////////////

module BackgroundROM (
    input  logic [16:0] raddr,
    output logic [15:0] data
);
    logic [15:0] mem[0:320*240-1];

    initial begin
        $readmemh("background.mem", mem);  // QQVGA
    end

    assign data = mem[raddr];
endmodule

////////////////////////////////////////////////////////////

module GreenFilter_RGB (
    input  logic [4:0] i_r,
    input  logic [5:0] i_g,
    input  logic [4:0] i_b,
    output logic       green
);
    parameter G_THRESH           = 6'd18; // 최소 초록 인식 (검정 제외)
    parameter DOMINANCE_OFFSET_R = 6'd7;  // R 대비 G 우위
    parameter DOMINANCE_OFFSET_B = 6'd7;  // B 대비 G 우위
    parameter R_MAX              = 5'd28; // R 최대 허용치
    parameter B_MAX              = 5'd28; // B 최대 허용치

    logic [5:0] r_6bit, b_6bit;

    // R/B를 6비트로 확장
    assign r_6bit = {i_r, i_r[4]};
    assign b_6bit = {i_b, i_b[4]};

    // 녹색 판정
    assign green = (i_g >= G_THRESH) &&
                   (i_g > r_6bit + DOMINANCE_OFFSET_R) &&
                   (i_g > b_6bit + DOMINANCE_OFFSET_B) &&
                   (i_r < R_MAX) &&
                   (i_b < B_MAX);
endmodule

////////////////////////////////////////////////////////////

module ImgReader (
    input  logic        DE,
    input  logic [ 9:0] x,
    input  logic [ 9:0] y,
    output logic [16:0] addr,
    input  logic [15:0] data,
    output logic [ 3:0] r_port,
    output logic [ 3:0] g_port,
    output logic [ 3:0] b_port
);
    logic img_show;

    assign img_show = (DE && (x < 640) && (y < 480));
    assign addr = img_show ? ((y >> 1) * 320 + (x >> 1)) : 17'bz;
    assign {r_port, g_port, b_port} = img_show ? {data[15:12], data[10:7], data[4:1]} : 12'b0;
endmodule

////////////////////////////////////////////////////////////
```

## Test Bench

```verilog
`timescale 1ns / 1ps

// 1. DUT와 Testbench를 연결하는 Interface (RGB565로 수정)
interface chromakey_intf (
    input bit clk
);
    logic        DE;
    logic [9:0]  x, y;
    logic [4:0]  i_r; // DUT 입력에 맞게 5-bit
    logic [5:0]  i_g; // DUT 입력에 맞게 6-bit
    logic [4:0]  i_b; // DUT 입력에 맞게 5-bit
    logic        green_out; // DUT의 green 출력만 모니터링
endinterface

///////////////////////////////////////////////////////////////////////////

// 2. Transaction 클래스 (green 신호만 검증하도록 단순화)
class PixelTransaction;
    logic             de;
    rand logic [9:0]  x, y;
    rand logic [4:0]  r;
    rand logic [5:0]  g;
    rand logic [4:0]  b;

    logic        green_out; // Monitor가 수집할 green 신호

    constraint c_valid_vga_coords { x < 640; y < 480; }

    function new(); 
        this.de = 1'b1; 
    endfunction

    task print(string name);
        $display("[%s] DE=%b, (%0d, %0d), RGB_in=(%h,%h,%h) -> Green=%b",
                 name, de, x, y, r, g, b, green_out);
    endtask
endclass

///////////////////////////////////////////////////////////////////////////
// 3. Generator
class generator;
    PixelTransaction tr;
    mailbox #(PixelTransaction) gen2drv_mbox;
    event scb2gen_event;

    function new(mailbox#(PixelTransaction) gen2drv_mbox, event scb2gen_event);
        this.gen2drv_mbox = gen2drv_mbox;
        this.scb2gen_event = scb2gen_event;
    endfunction

    task run(int loop);
        repeat (loop) begin
            tr = new();
            if (!tr.randomize()) $error("Randomization Failed!");
            tr.print("GEN");
            gen2drv_mbox.put(tr);
            @(scb2gen_event);
        end
    endtask
endclass
///////////////////////////////////////////////////////////////////////////
// 4. Driver
class driver;
    PixelTransaction tr;
    mailbox #(PixelTransaction) gen2drv_mbox;
    virtual chromakey_intf ck_if;

    function new(mailbox #(PixelTransaction) gen2drv_mbox, virtual chromakey_intf ck_if);
        this.gen2drv_mbox = gen2drv_mbox;
        this.ck_if = ck_if;
    endfunction

    task run();
        forever begin
            gen2drv_mbox.get(tr);
            @(posedge ck_if.clk);
            ck_if.DE  <= tr.de;
            ck_if.x   <= tr.x;
            ck_if.y   <= tr.y;
            ck_if.i_r <= tr.r;
            ck_if.i_g <= tr.g;
            ck_if.i_b <= tr.b;
            tr.print("DRV");
        end
    endtask
endclass
///////////////////////////////////////////////////////////////////////////
// 5. Monitor (변경 없음)
class monitor;
    PixelTransaction tr;
    mailbox #(PixelTransaction) mon2scb_mbox;
    virtual chromakey_intf ck_if;
    function new(mailbox #(PixelTransaction) mon2scb_mbox, virtual chromakey_intf ck_if);
        this.mon2scb_mbox = mon2scb_mbox;
        this.ck_if = ck_if;
    endfunction
    task run();
        forever begin
            @(posedge ck_if.clk);
            #1;
            tr = new();
            tr.de = ck_if.DE;
            tr.x  = ck_if.x;
            tr.y  = ck_if.y;
            tr.r  = ck_if.i_r;
            tr.g  = ck_if.i_g;
            tr.b  = ck_if.i_b;
            tr.green_out = ck_if.green_out; // green 신호만 샘플링
            tr.print("MON");
            mon2scb_mbox.put(tr);
        end
    endtask
endclass
///////////////////////////////////////////////////////////////////////////

// 6. Scoreboard (★★ Reference Model 완성 ★★)
class Scoreboard;
    PixelTransaction tr;
    mailbox #(PixelTransaction) mon2scb_mbox;
    event scb2gen_event;
    int total_cnt, pass_cnt, fail_cnt;

    function new(mailbox#(PixelTransaction) mon2scb_mbox, event scb2gen_event);
        this.mon2scb_mbox = mon2scb_mbox;
        this.scb2gen_event = scb2gen_event;
        this.total_cnt = 0; this.pass_cnt = 0; this.fail_cnt = 0;
        // 배경 ROM 로드 로직 제거
    endfunction

    // DUT의 GreenFilter_RGB 로직과 정확히 동일한 레퍼런스 모델
    function bit predict_green(logic [4:0] r, logic [5:0] g, logic [4:0] b);
        localparam G_THRESH = 6'd18;
        localparam DOMINANCE_OFFSET_R = 6'd7;
        localparam DOMINANCE_OFFSET_B = 6'd7;
        localparam R_MAX = 5'd28;
        localparam B_MAX = 5'd28;

        logic [5:0] r_6bit = {r, r[4]};
        logic [5:0] b_6bit = {b, b[4]};
        
        return (g >= G_THRESH) && (g > r_6bit + DOMINANCE_OFFSET_R) && (g > b_6bit + DOMINANCE_OFFSET_B) && (r < R_MAX) && (b < B_MAX);
    endfunction

    task run();
        forever begin
            mon2scb_mbox.get(tr);
            total_cnt++;
            tr.print("SCB");

            if (tr.de) begin
                logic ref_green;
                ref_green = predict_green(tr.r, tr.g, tr.b); // 예상 green 신호 계산

                // --- 비교 ---
                if (ref_green === tr.green_out) begin
                    pass_cnt++;
                    $display("PASS! Matched Green Signal!");
                end else begin
                    fail_cnt++;
                    $display("FAIL! Mismatched Green Signal!");
                    $display("  Input   : RGB=(%h,%h,%h)", tr.r, tr.g, tr.b);
                    $display("  Expected: Green=%b", ref_green);
                    $display("  Actual  : Green=%b", tr.green_out);
                end
            end
            ->scb2gen_event;
        end
    endtask
endclass

///////////////////////////////////////////////////////////////////////////
// 7. Environment (변경 없음)
class environment;
    mailbox #(PixelTransaction) gen2drv_mbox;
    mailbox #(PixelTransaction) mon2scb_mbox;

    generator  gen;
    driver     drv;
    monitor    mon;
    Scoreboard scb; // [수정] 대문자 'S'로 시작하는 Scoreboard 타입 사용

    event scb2gen_event;

    function new(virtual chromakey_intf ck_if);
        gen2drv_mbox = new();
        mon2scb_mbox = new();
        
        gen = new(gen2drv_mbox, scb2gen_event);
        drv = new(gen2drv_mbox, ck_if);
        mon = new(mon2scb_mbox, ck_if);
        scb = new(mon2scb_mbox, scb2gen_event);
    endfunction

    task run(int loop);
        fork
            gen.run(loop);
            drv.run();
            mon.run();
            scb.run();
        join_any;

        $display("-----------------------------------------");
        $display("           Simulation Result           ");
        $display("-----------------------------------------");
        $display("Total : %0d", scb.total_cnt);
        $display("Pass  : %0d", scb.pass_cnt);
        $display("Fail  : %0d", scb.fail_cnt);
        $display("-----------------------------------------");
        #50;
    endtask
endclass
///////////////////////////////////////////////////////////////////////////

// 8. 최상위 테스트벤치 모듈
module tb_top ();
    bit clk;
    chromakey_intf ck_if (clk);
    environment    env;

    // DUT의 사용하지 않는 출력 포트를 연결하기 위한 로컬 와이어
    logic [3:0] dut_r_port, dut_g_port, dut_b_port;

    // DUT Instantiation
    Chromakey_Filter DUT (
        .DE(ck_if.DE),
        .x(ck_if.x),
        .y(ck_if.y),
        .i_r(ck_if.i_r),
        .i_g(ck_if.i_g),
        .i_b(ck_if.i_b),
        .green(ck_if.green_out),
        // 사용하지 않는 출력 포트는 테스트벤치의 로컬 와이어에 연결
        .r_port(dut_r_port),
        .g_port(dut_g_port),
        .b_port(dut_b_port)
    );

    always #5 clk = ~clk;

    initial begin
        clk = 1;
        env = new(ck_if);
        env.run(10000);
        $finish;
    end
endmodule
```