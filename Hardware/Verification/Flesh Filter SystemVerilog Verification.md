<p>
<img width="600" height="538" alt="tb" src="https://github.com/user-attachments/assets/89afa166-8761-4717-a3bf-70287fee1627" />    
</p>
<p>
<strong> 입력 : den, r_in/g_in/b_in(각 4b)
<strong> 처리 : 4b→8b 확장 → 정수형 YCbCr 근사 → Cb/Cr 범위 + (r>g,b) 규칙으로 살색 판정
<strong> 출력 : 살색이면 r_out/g_out/b_out = 4'hF(흰색), 아니면 4'h0(검정)   
</p>
```verilog

`timescale 1ns / 1ps

//============================================================
// Interface
//============================================================
interface flesh_if;
    logic clk;
    logic den;
    logic [3:0] r_in, g_in, b_in;
    logic [3:0] r_out, g_out, b_out;
endinterface

//============================================================
// Transaction
//============================================================
class transaction;
    rand bit       den;
    rand bit [3:0] r_data, g_data, b_data;
    bit      [3:0] r_out,  g_out,  b_out;  // observed
endclass

//============================================================
// Generator
//============================================================
class generator;
    mailbox #(transaction) gen2drv_mbox;

    function new(mailbox#(transaction) gen2drv_mbox);
        this.gen2drv_mbox = gen2drv_mbox;
    endfunction

    task run(int gen_cnt);
        transaction tr;
        repeat (gen_cnt) begin
            tr = new();
            tr.randomize();
            gen2drv_mbox.put(tr);
            #10;                 // 트래픽 간격
        end
    endtask
endclass

//============================================================
// Driver
//============================================================
class driver;
    mailbox #(transaction) gen2drv_mbox;
    virtual flesh_if vif;

    function new(mailbox#(transaction) gen2drv_mbox, virtual flesh_if vif);
        this.gen2drv_mbox = gen2drv_mbox;
        this.vif = vif;
    endfunction

    task run();
        transaction tr;
        forever begin
            gen2drv_mbox.get(tr);
            vif.den  <= tr.den;
            vif.r_in <= tr.r_data;
            vif.g_in <= tr.g_data;
            vif.b_in <= tr.b_data;
            @(posedge vif.clk);
        end
    endtask
endclass

//============================================================
// Monitor
//============================================================
class monitor;
    mailbox #(transaction) mon2scb_mbox;
    virtual flesh_if vif;

    function new(mailbox#(transaction) mon2scb_mbox, virtual flesh_if vif);
        this.mon2scb_mbox = mon2scb_mbox;
        this.vif = vif;
    endfunction

    task run();
        transaction tr;
        forever begin
            @(posedge vif.clk);
            #1; // 델타 지연으로 레이스 방지
            tr = new();
            tr.den = vif.den;
            tr.r_data = vif.r_in;
            tr.g_data = vif.g_in;
            tr.b_data = vif.b_in;
            tr.r_out = vif.r_out;
            tr.g_out = vif.g_out;
            tr.b_out = vif.b_out;
            mon2scb_mbox.put(tr);
        end
    endtask
endclass

//============================================================
// Scoreboard
//   - 예측이 흰색(FFF)이면 skin detect
//   - 흑색(000)이면 no-skin
//============================================================
class scoreboard;
    mailbox #(transaction) mon2scb_mbox;

    // 통계
    int unsigned total_count;
    int unsigned total_correct_count, total_incorrect_count;
    int unsigned detected_correct_count, no_detected_correct_count;
    int unsigned detected_incorrect_count, no_detected_incorrect_count;

    // 동일 상수
    static const int unsigned CB_MIN = 77;
    static const int unsigned CB_MAX = 127;
    static const int unsigned CR_MIN = 133;
    static const int unsigned CR_MAX = 173;

    function new(mailbox#(transaction) mon2scb_mbox);
        this.mon2scb_mbox           = mon2scb_mbox;
        total_count                 = 0;
        total_correct_count         = 0;
        total_incorrect_count       = 0;
        detected_correct_count      = 0;
        no_detected_correct_count   = 0;
        detected_incorrect_count    = 0;
        no_detected_incorrect_count = 0;
    endfunction

    // 참조모델: DUT와 완전히 동일 계산
    function bit ref_is_skin(bit den, bit [3:0] r, g, b);
        int unsigned R8 = {r, r};
        int unsigned G8 = {g, g};
        int unsigned B8 = {b, b};

        int signed cb_acc = -43 * $signed(
            {1'b0, R8}
        ) + -85 * $signed(
            {1'b0, G8}
        ) + 128 * $signed(
            {1'b0, B8}
        );
        int signed cr_acc = 128 * $signed(
            {1'b0, R8}
        ) + -107 * $signed(
            {1'b0, G8}
        ) + -21 * $signed(
            {1'b0, B8}
        );

        int Cb = 128 + (cb_acc >>> 8);
        int Cr = 128 + (cr_acc >>> 8);

        bit skin = den
            && (Cb >= CB_MIN) && (Cb <= CB_MAX)
            && (Cr >= CR_MIN) && (Cr <= CR_MAX)
            && (r > g) && (r > b);
        return skin;
    endfunction
    bit skin_ref;
    bit dut_detect;
    bit dut_no_detect;
    bit [3:0] exp_nib;
    bit match;
    int unsigned flesh_count;

    task run();
        transaction tr;
        forever begin
            mon2scb_mbox.get(tr);

            // 예측(REF)
            skin_ref = ref_is_skin(tr.den, tr.r_data, tr.g_data, tr.b_data);

            // DUT 판정
            dut_detect   = (tr.r_out===4'hF) && (tr.g_out===4'hF) && (tr.b_out===4'hF);
            dut_no_detect= (tr.r_out===4'h0) && (tr.g_out===4'h0) && (tr.b_out===4'h0);

            // X/Z 방어
            if (!(dut_detect || dut_no_detect)) begin
                continue;
            end
            exp_nib = skin_ref ? 4'hF : 4'h0;
            match = (tr.r_out===exp_nib) && (tr.g_out===exp_nib) && (tr.b_out===exp_nib);
     
            if (dut_detect) begin
                if (skin_ref) begin
                    detected_correct_count++;
                    flesh_count++;
                end
                else detected_incorrect_count++;
            end else begin
                if (skin_ref) no_detected_incorrect_count++;
                else no_detected_correct_count++;
            end

            total_correct_count   = detected_correct_count + no_detected_correct_count;
            total_incorrect_count = detected_incorrect_count + no_detected_incorrect_count;
            total_count++;

            if (match) begin
                $display("[%d] MATCH : den=%0d IN(R,G,B)=%1h,%1h,%1h  OUT=%1h%1h%1h  EXP=%1h%1h%1h",
                          total_count,tr.den, tr.r_data, tr.g_data, tr.b_data,
                          tr.r_out, tr.g_out, tr.b_out,
                          exp_nib,  exp_nib,  exp_nib);
            end else begin
                $display("[%d] MISMATCH : den=%0d IN(R,G,B)=%1h,%1h,%1h  OUT=%1h%1h%1h  EXP=%1h%1h%1h",
                          total_count,tr.den, tr.r_data, tr.g_data, tr.b_data,
                          tr.r_out, tr.g_out, tr.b_out,
                          exp_nib,  exp_nib,  exp_nib);
            end

        end
    endtask

    task print_score();
        $display(
            "\n---------------------------------------------------------------------------");
        $display(
            "| total count | total correct count | total incorrect count  | total flesh |");
        $display("|   %8d  |      %8d       |      %8d         |     %8d     |",
                 total_count, total_correct_count, total_incorrect_count,flesh_count);
        $display(
            "---------------------------------------------------------------------------\n");
    endtask
endclass

//============================================================
// Environment
//============================================================
class environment;
    generator              gen;
    driver                 drv;
    monitor                mon;
    scoreboard             scr;

    mailbox #(transaction) gen2drv_mbox;
    mailbox #(transaction) mon2scb_mbox;

    virtual flesh_if       vif;
    int                    expected;

    function new(virtual flesh_if vif);
        this.vif = vif;
        gen2drv_mbox = new();
        mon2scb_mbox = new();
        gen = new(gen2drv_mbox);
        drv = new(gen2drv_mbox, vif);
        mon = new(mon2scb_mbox, vif);
        scr = new(mon2scb_mbox);
    endfunction

    task run(int gen_count);
        expected = gen_count;

        fork
            gen.run(gen_count);
            drv.run();
            mon.run();
            scr.run();
        join_none

        // 모든 샘플 처리될 때까지 대기
        wait (scr.total_count == expected);
        scr.print_score();
        $finish;
    endtask
endclass

//============================================================
// TB 
//============================================================

module tb_sobel;
    flesh_if fif ();
    environment env;
    // 25 MHz
    initial begin
        fif.clk = 1'b0;
        forever #20 fif.clk = ~fif.clk;
    end

    // DUT
    flesh_color dut (
        .den  (fif.den),
        .r_in (fif.r_in),
        .g_in (fif.g_in),
        .b_in (fif.b_in),
        .r_out(fif.r_out),
        .g_out(fif.g_out),
        .b_out(fif.b_out)
    );

    initial begin
        // 초기화
        fif.den  = 1'b0;
        fif.r_in = '0;
        fif.g_in = '0;
        fif.b_in = '0;
        repeat (5) @(posedge fif.clk);

        env = new(fif);
        env.run(100000); 
    end
endmodule
```


