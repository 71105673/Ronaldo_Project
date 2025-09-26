# Red_Glove_Detector Verification

## 1. 검증할 Module
- 붉은색을 검출하는 Filter

<div align="center">
     <img width="523" height="274" alt="image" src="https://github.com/user-attachments/assets/6f16707e-892f-48d6-9a5b-e781ced68a79" />
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
| **interface** | DUT와 테스트벤치 간의 신호 연결 통로                                                       | - `virtual interface`로 Driver/Monitor에 핸들 전달<br>- clock과 DUT의 **RGB 4:4:4** 포맷에 맞춘 신호, detect 출력 신호 세트 전달 |
| **transaction** | 검증을 위한 최소 데이터 단위(패킷) 정의                                                    | - 1-Pixel에 해당하는 **RGB 4:4:4** 입력 및 detect 출력 데이터 포함<br>- `randomize()`를 위한 `rand` 변수 선언 |
| **generator** | 테스트 시나리오(입력 자극) 생성                                                            | - `randomize()`를 호출하여 무작위 픽셀 데이터 생성<br>-  gen_cnt만큼 random 데이터 생성<br>- 생성한 데이터를 transaction에 담아 mailbox로 put|
| **driver** | 생성된 Transaction을 DUT에 인가                                                            | - `posedge clk`에 동기화하여 인터페이스에 데이터 할당<br>- Mailbox를 통해 Generator로부터 데이터 수신 |
| **monitor** | DUT의 입력 및 출력 신호 감지                                                               | - `posedge clk` 이후 안정적인 시점에 신호 샘플링<br>- 감지한 데이터를 Transaction에 담아 mailbox를 이용하여 Scoreboard로 전송 |
| **scoreboard** | DUT의 동작 정확성 검증                                                                     | - DUT와 동일한 로직의 계산 결과와 Monitor로부터 받은 실제 출력을 비교하여 **PASS/FAIL** 판정 및 집계 |
| **environment** | 검증 환경의 모든 컴포넌트 통합 및 제어                                                     | - 각 컴포넌트(Gen, Drv, Mon, Scb) 객체 생성 및 Mailbox 연결<br>- 시뮬레이션 시작 및 fork된 proecss중 하나라도 종료시 종료 제어 |
| **tb_top** | 시뮬레이션 최상위 모듈                                                                     | - 클럭(Clock) 생성<br>- DUT 및 `interface` 인스턴스화<br>- `environment` 실행 |

---

## 3. 데이터 처리 (Red_Glove_Detector)


**입력** : r_data (4bit), g_data (4bit), b_data (4bit) 


**처리** : R값이 가장 크고, 최댓값과 최솟값의 차이가 일정량(7) 이상 크고, R값이 G/B값보다 일정량(6) 이상 큰지 판별 


**출력** : 붉은색이면 detect = 1, 아니면 detect = 0 

</p>

## 4. 검증 결과
- 검증 중간 출력
    - 붉은색이 있는 픽셀만 기준에 따라 잘 detection
      
      <img width="377" height="302" alt="image" src="https://github.com/user-attachments/assets/fb0cbd60-58e6-4e6b-ba53-efa05f4a3170" />

- 검증 최종 결과
    - 10만개의 데이터 모두 PASS 확인
  <div align="center">
    <img width="1269" height="256" alt="image" src="https://github.com/user-attachments/assets/f542e6e2-1125-4e97-bfb0-9dba5bc38710" />
  </div>

  <br>
<details>
    <summary>Red_Glove_Detector_Verification_Code</summary>

```verilog

`timescale 1ns / 1ps

interface red_glove_intf;
    logic       clk;
    logic       reset;
    logic [3:0] r_data;
    logic [3:0] g_data;
    logic [3:0] b_data;
    logic       detect;
endinterface  //red_glove_intf

class transaction;
    rand bit [3:0] r_data;
    rand bit [3:0] g_data;
    rand bit [3:0] b_data;
    bit            detect;
endclass

class generator;
    transaction tr;
    mailbox #(transaction) gen2drv_mbox;

    function new(mailbox#(transaction) gen2drv_mbox);
        this.gen2drv_mbox = gen2drv_mbox;
    endfunction  //new()

    task run(int gen_cnt);
        repeat (gen_cnt) begin
            tr = new();
            tr.randomize();
            gen2drv_mbox.put(tr);
            #10;
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
            red_glove_if.r_data = tr.r_data;
            red_glove_if.g_data = tr.g_data;
            red_glove_if.b_data = tr.b_data;
            @(posedge red_glove_if.clk);
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
            @(posedge red_glove_if.clk);
            #1;
            tr.r_data = red_glove_if.r_data;
            tr.g_data = red_glove_if.g_data;
            tr.b_data = red_glove_if.b_data;
            tr.detect = red_glove_if.detect;
            mon2scr_mbox.put(tr);
        end
    endtask  //
endclass  //monitor

class scoreboard;
    transaction tr;
    mailbox #(transaction) mon2scr_mbox;

    bit [3:0] max_val, min_val, delta;
    logic [20:0] total_count;
    logic [19:0] total_correct_count, total_incorrect_count;
    logic [18:0] detected_correct_count, no_detected_correct_count;
    logic [18:0] detected_incorrect_count, no_detected_incorrect_count;

    function new(mailbox#(transaction) mon2scr_mbox);
        this.mon2scr_mbox           = mon2scr_mbox;
        total_count                 = 0;
        total_correct_count         = 0;
        total_incorrect_count       = 0;
        detected_correct_count      = 0;
        no_detected_correct_count   = 0;
        detected_incorrect_count    = 0;
        no_detected_incorrect_count = 0;
    endfunction  //new()

    task run();
        forever begin
            mon2scr_mbox.get(tr);

            max_val = (tr.r_data >= tr.g_data && tr.r_data >= tr.b_data) ? tr.r_data :
                     (tr.g_data >= tr.b_data) ? tr.g_data : tr.b_data;
            min_val = (tr.r_data <= tr.g_data && tr.r_data <= tr.b_data) ? tr.r_data :
                     (tr.g_data <= tr.b_data) ? tr.g_data : tr.b_data;
            delta = max_val - min_val;

            if (tr.detect) begin  //red glove detect
                if ((tr.r_data == max_val) && (delta >= 7) &&  
                    ((tr.r_data - tr.g_data) >= 6) && ((tr.r_data - tr.b_data) >= 6)) begin
                    $display(
                        "PASS! : r_data = %d, g_data = %d, b_data = %d, detected!!!",
                        tr.r_data, tr.g_data, tr.b_data);
                    detected_correct_count++;
                end else begin
                    $display(
                        "---------------------------------------------------------");
                    $display(
                        "FAIL! : Filter detected red, but it is not a red glove!!!");
                    $display("r_data = %d, g_data = %d, b_data = %d",
                             tr.r_data, tr.g_data, tr.b_data);
                    $display(
                        "---------------------------------------------------------");
                    detected_incorrect_count++;
                end
            end else begin  //no detect
                if ((tr.r_data == max_val) && (delta >= 7) &&  
                    ((tr.r_data - tr.g_data) >= 6) && ((tr.r_data - tr.b_data) >= 6)) begin
                    $display(
                        "---------------------------------------------------------");
                    $display(
                        "FAIL! : Filter did not detected red, but it is a red glove!!!");
                    $display("r_data = %d, g_data = %d, b_data = %d",
                             tr.r_data, tr.g_data, tr.b_data);
                    $display(
                        "---------------------------------------------------------");
                    no_detected_incorrect_count++;
                end else begin
                    $display(
                        "PASS! : r_data = %d, g_data = %d, b_data = %d, no detected!!!",
                        tr.r_data, tr.g_data, tr.b_data);
                    no_detected_correct_count++;
                end
            end
            #1;
            total_correct_count = detected_correct_count + no_detected_correct_count;
            total_incorrect_count = detected_incorrect_count + no_detected_incorrect_count;
            total_count++;
        end
    endtask  //

    task print_socre();
        $display(
            "\n----------------------------Result----------------------------");
        $display(
            "| total count | total correct count | total incorrect count  |");
        $display("|  %d    |       %d       |     %d            |", total_count,
                 total_correct_count, total_incorrect_count);
        $display(
            "--------------------------------------------------------------\n");
    endtask  //
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

    task run(int gen_count);
        fork
            gen.run(gen_count);
            drv.run();
            mon.run();
            scr.run();
        join_any
        #10;
        scr.print_socre();
        $finish();
    endtask  //
endclass  //environment

module tb_red_glove_filter ();
    environment env;
    red_glove_intf red_glove_if ();

    RedGlove_Detector dut (
        .r_data(red_glove_if.r_data),
        .g_data(red_glove_if.g_data),
        .b_data(red_glove_if.b_data),
        .detect(red_glove_if.detect)
    );

    always #5 red_glove_if.clk = ~red_glove_if.clk;

    initial begin
        red_glove_if.clk = 1;
    end

    initial begin
        env = new(red_glove_if);
        env.run(100000);
    end

endmodule

```

</details>


