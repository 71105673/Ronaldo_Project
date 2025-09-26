 # Block Diagram
 
<div align="center">
<img width="600" height="538" alt="image" src="https://github.com/user-attachments/assets/327818fd-cf86-42df-936c-b162f222f28e" />
</div>
<br>
<br>

# Class Introduction

|     블록명      |                                    설명                                               |
|:-----:|---------------------------------------------------------------------------------------|
| **interface**   | TB & DUT 신호 묶음, TB 내부 공유 신호 집합<br>virtual interface로 driver/monitor가 같은 핸들을 사용 |
| **transaction** | 1픽셀 transaction할 데이터 선언 (den,r,g,b + 관찰된 out)                               |
| **generator**   | 랜덤 패턴 생성<br>randomize() 사용하여 랜덤 생성<br>gen_cnt만큼 tr 생성                |
| **driver**      | 인터페이스에 구동<br>non-blocking 할당 후 다음 posedge까지 유지                       |
| **monitor**     | 결과 샘플링<br>scoreboard로 den,r,g,b,out 전달                                       |
| **scoreboard**  | 참조모델(Ref) = DUT와 완전히 동일 수식으로 판정 → PASS/FAIL 집계<br>DUT가 FFF → detect, 000 → no-detect<br>다른 값(X/Z 등) 나오면 스킵<br>DUT 수식과 Ref 수식이 바이트 정확히 동일한지 판단 |
| **environment** | 위 블록들을 묶어 실행/종료 제어<br>scoreboard가 gen_count만큼 처리하면 종료          |
| **tb_sobel**    | 최상위 TB (25 MHz 클럭 생성, DUT 인스턴스, env 실행)                                  |

<br>
<br>

# Code
```verilog
`timescale 1ns / 1ps

interface sobel_intf;
    logic clk;
    logic reset;
    logic den;
    logic [9:0] x_in;
    logic [9:0] y_in;
    logic [3:0] r_in;
    logic [3:0] g_in;
    logic [3:0] b_in;
    logic [3:0] r_out;
    logic [3:0] g_out;
    logic [3:0] b_out;
endinterface


class transaction;
    bit      [9:0] x_in;
    bit      [9:0] y_in;
    rand bit [3:0] r_in;
    rand bit [3:0] g_in;
    rand bit [3:0] b_in;
    bit            den;
    bit      [3:0] r_out;
    bit      [3:0] g_out;
    bit      [3:0] b_out;
endclass

class generator;
    transaction tr;
    mailbox #(transaction) gen2drv_mbox;
    int W, H;

    function new(mailbox#(transaction) gen2drv_mbox, int width = 640,
                 int height = 480);
        this.gen2drv_mbox = gen2drv_mbox;
        this.W = width;
        this.H = height;
    endfunction

    task run(int gen_cnt);
        int cnt = 0;
        for (int y = 0; y < H && cnt < gen_cnt; y++) begin
            for (int x = 0; x < W && cnt < gen_cnt; x++) begin
                tr = new();
                tr.x_in = x;
                tr.y_in = y;
                tr.den = 1'b1;
                assert (tr.randomize() with {
                    r_in inside {[0 : 15]};
                    g_in inside {[0 : 15]};
                    b_in inside {[0 : 15]};
                });
                gen2drv_mbox.put(tr);
                cnt++;
                #1;
            end
        end
    endtask
endclass

class driver;
    transaction tr;
    mailbox #(transaction) gen2drv_mbox;
    virtual sobel_intf sobel_if;

    function new(mailbox#(transaction) gen2drv_mbox,
                 virtual sobel_intf sobel_if);
        this.gen2drv_mbox = gen2drv_mbox;
        this.sobel_if = sobel_if;
    endfunction

    task run();
        forever begin
            gen2drv_mbox.get(tr);
            @(negedge sobel_if.clk); 
            sobel_if.x_in <= tr.x_in;
            sobel_if.y_in <= tr.y_in;
            sobel_if.r_in <= tr.r_in;
            sobel_if.g_in <= tr.g_in;
            sobel_if.b_in <= tr.b_in;
            sobel_if.den  <= tr.den;
        end
    endtask
endclass

class monitor;
    transaction tr;
    mailbox #(transaction) mon2scr_mbox;
    virtual sobel_intf sobel_if;

    function new(mailbox#(transaction) mon2scr_mbox,
                 virtual sobel_intf sobel_if);
        this.mon2scr_mbox = mon2scr_mbox;
        this.sobel_if = sobel_if;
    endfunction

    task run();
        forever begin
            tr = new();
            @(posedge sobel_if.clk);
            #1;
            tr.x_in  = sobel_if.x_in;
            tr.y_in  = sobel_if.y_in;
            tr.den   = sobel_if.den;
            tr.r_in  = sobel_if.r_in;
            tr.g_in  = sobel_if.g_in;
            tr.b_in  = sobel_if.b_in;
            tr.r_out = sobel_if.r_out;
            tr.g_out = sobel_if.g_out;
            tr.b_out = sobel_if.b_out;
            mon2scr_mbox.put(tr);
        end
    endtask
endclass

class scoreboard;
    mailbox #(transaction) mon2scr_mbox;
    int W;
    
    byte unsigned lineA[];
    byte unsigned lineB[];
    bit sel;

    byte unsigned top_w0_tb, top_w1_tb, top_w2_tb;
    byte unsigned mid_w0_tb, mid_w1_tb, mid_w2_tb;
    byte unsigned cur_w0_tb, cur_w1_tb, cur_w2_tb;

    longint unsigned total, pass_cnt, fail_cnt;

    function new(mailbox#(transaction) mon2scr_mbox, int width);
        this.mon2scr_mbox = mon2scr_mbox;
        this.W = width;
        lineA = new[W];
        lineB = new[W];
        foreach (lineA[i]) begin
            lineA[i] = 8'h00;
            lineB[i] = 8'h00;
        end
        sel = 1'b0;
        top_w0_tb = 0;
        top_w1_tb = 0;
        top_w2_tb = 0;
        mid_w0_tb = 0;
        mid_w1_tb = 0;
        mid_w2_tb = 0;
        cur_w0_tb = 0;
        cur_w1_tb = 0;
        cur_w2_tb = 0;
        total = 0;
        pass_cnt = 0;
        fail_cnt = 0;
    endfunction

    task run();
        transaction tr;
        
        byte unsigned R8, G8, B8, Y8;
        int unsigned y_mul;
        byte unsigned top_rd, mid_rd;
        int xi;
        
        // sobel
        logic signed [12:0] Gx, Gy, Gx_temp, Gy_temp;
        int unsigned Ax, Ay, mag, mag_temp, Ax_temp, Ay_temp;
        byte unsigned edge8, edge8_temp;
        
        // a/b/c 매핑
        byte unsigned a0, a1, a2, b0, b1, b2, c0, c1, c2;
        bit
            den_now,
            line_start,
            frame_start,
            valid_now,
            valid_d1,
            valid_d2,
            valid_d3,
            valid_d4,
            valid_d5,
            valid_d6;
        logic [3:0] r_q1, r_q2, r_q3;
        logic [3:0] g_q1, g_q2, g_q3;
        logic [3:0] b_q1, b_q2, b_q3;

        logic [3:0] r_data, g_data, b_data;

        forever begin
            mon2scr_mbox.get(tr);

            den_now     = tr.den;
            line_start  = den_now && (tr.x_in == 10'd0);
            frame_start = den_now && (tr.x_in == 10'd0) && (tr.y_in == 10'd0);

            R8          = {tr.r_in, tr.r_in};
            G8          = {tr.g_in, tr.g_in};
            B8          = {tr.b_in, tr.b_in};
            y_mul       = (R8 * 8'd77) + (G8 * 8'd150) + (B8 * 8'd29);
            Y8          = y_mul[15:8];
            xi          = int'(tr.x_in);

            if (sel == 1'b0) begin
                top_rd = lineB[xi];
                mid_rd = lineA[xi];
            end else begin
                top_rd = lineA[xi];
                mid_rd = lineB[xi];
            end

            if (den_now) begin
                if (sel == 1'b0) lineB[xi] = Y8;
                else lineA[xi] = Y8;
            end

            if (den_now) begin
                if (frame_start || line_start) begin
                    top_w0_tb = 0;
                    top_w1_tb = 0;
                    top_w2_tb = top_rd;
                    mid_w0_tb = 0;
                    mid_w1_tb = 0;
                    mid_w2_tb = mid_rd;
                    cur_w0_tb = 0;
                    cur_w1_tb = 0;
                    cur_w2_tb = Y8;
                end else begin
                    top_w0_tb = top_w1_tb;
                    top_w1_tb = top_w2_tb;
                    top_w2_tb = top_rd;
                    mid_w0_tb = mid_w1_tb;
                    mid_w1_tb = mid_w2_tb;
                    mid_w2_tb = mid_rd;
                    cur_w0_tb = cur_w1_tb;
                    cur_w1_tb = cur_w2_tb;
                    cur_w2_tb = Y8;
                end
            end
            
            if (frame_start) sel = 1'b0;
            else if (line_start) sel = ~sel;

            // a/b/c 매핑
            a0 = top_w0_tb;
            a1 = top_w1_tb;
            a2 = top_w2_tb;
            b0 = mid_w0_tb;
            b1 = mid_w1_tb;
            b2 = mid_w2_tb;
            c0 = cur_w0_tb;
            c1 = cur_w1_tb;
            c2 = cur_w2_tb;

            // sobel
            Gx = -$signed({5'd0, a0}) + $signed({5'd0, a2}) -
                $signed({4'd0, b0, 1'b0}) + $signed({4'd0, b2, 1'b0}) -
                $signed({5'd0, c0}) + $signed({5'd0, c2});
            Gy = -$signed({5'd0, a0}) - $signed({4'd0, a1, 1'b0}) -
                $signed({5'd0, a2}) + $signed({5'd0, c0}) +
                $signed({4'd0, c1, 1'b0}) + $signed({5'd0, c2});


            //Ax = (Gx<0)?-Gx:Gx; Ay = (Gy<0)?-Gy:Gy; mag = Ax+Ay;
            Ax = (Gx_temp < 0) ? -Gx_temp : Gx_temp;
            Ay = (Gy_temp < 0) ? -Gy_temp : Gy_temp;
            mag = Ax_temp + Ay_temp;
            edge8 = (mag_temp[13] || (mag_temp>14'd255)) ? 8'hFF : mag_temp[7:0];
            valid_now = den_now && (((tr.x_in>=10'd2) && (tr.y_in>=10'd2)));

            valid_d6 = valid_d5;
            valid_d5 = valid_d4;
            valid_d4 = valid_d3;
            valid_d3 = valid_d2;
            valid_d2 = valid_d1;
            valid_d1 = valid_now;
            
            r_data = valid_d6 ? r_data : 4'b0;
            g_data = valid_d6 ? g_data : 4'b0;
            b_data = valid_d6 ? b_data : 4'b0;
            total++;
            
            if (
                tr.r_out===r_data && tr.g_out===g_data && tr.b_out===b_data) begin
                pass_cnt++;
                $display(
                    "MATCH (x=%0d,y=%0d): got=%0h exp=%0h  a=%0h,%0h,%0h b = %0h,%0h,%0h c = %0h,%0h,%0h edge8 = %0h",
                    tr.x_in, tr.y_in, tr.r_out, edge8[7:4], a0, a1, a2,
                    b0, b1, b2, c0, c1, c2, edge8_temp);
            end else begin
                fail_cnt++;
                $error(
                    "MISM (x=%0d,y=%0d): got R=%0h G=%0h B=%0h, exp=%0h  a=%0h,%0h,%0h top=%0h,%0h,%0h b = %0h,%0h,%0h mid=%0h,%0h,%0h cur = %0h, %0h, %0h c = %0h,%0h,%0h Gx = %0h Gy= %0h Ax = %0h Ay = %0h Mag = %0h edge8 = %0h valid_now = %d r_out = %0h g_out = %0h b_out = %0h",
                    tr.x_in, tr.y_in, tr.r_out, tr.g_out, tr.b_out, edge8[7:4],
                    a0, a1, a2, top_w0_tb, top_w1_tb, top_w2_tb, b0, b1, b2,
                    mid_w0_tb, mid_w1_tb, mid_w2_tb, cur_w0_tb, cur_w1_tb,
                    cur_w2_tb, c0, c1, c2, Gx_temp, Gy_temp, Ax_temp, Ay_temp,
                    mag_temp, edge8_temp, valid_now, r_data, g_data, b_data);
            end

            r_data = edge8[7:4];
            g_data = edge8[7:4];
            b_data = edge8[7:4];

            Gx_temp = Gx;
            Gy_temp = Gy;
            Ax_temp = Ax;
            Ay_temp = Ay;
            mag_temp = mag;
            

        end

    endtask

    task print_score();
        $display("\n---------------- Scoreboard ----------------");
        $display("Total comparisons : %0d", total);
        $display("Passes            : %0d", pass_cnt);
        $display("Mismatches        : %0d", fail_cnt);
        $display("--------------------------------------------\n");
    endtask
endclass

class environment;
    generator gen;
    driver drv;
    monitor mon;
    scoreboard scr;
    int W;
    mailbox #(transaction) gen2drv_mbox;
    mailbox #(transaction) mon2scr_mbox;

    function new(virtual sobel_intf sobel_if);
        gen2drv_mbox = new();
        mon2scr_mbox = new();
        W = 640;
        gen = new(gen2drv_mbox, W, 480);
        drv = new(gen2drv_mbox, sobel_if);
        mon = new(mon2scr_mbox, sobel_if);
        scr = new(mon2scr_mbox, W);
    endfunction

    task run(int gen_count);
        event gen_done;
        fork
            begin
                gen.run(gen_count);
                ->gen_done;
            end
            drv.run();
            mon.run();
            scr.run();
        join_none
        @(gen_done);
        wait (scr.total >= gen_count);
        scr.print_score();
        $finish();
    endtask
endclass

module tb_sobelfilter;
    environment env;
    sobel_intf sobel_if ();

    SobelFilter #(
        .WIDTH(640)
    ) dut (
        .clk  (sobel_if.clk),
        .rst  (sobel_if.reset),
        .den  (sobel_if.den),
        .x_in (sobel_if.x_in),
        .y_in (sobel_if.y_in),
        .r_in (sobel_if.r_in),
        .g_in (sobel_if.g_in),
        .b_in (sobel_if.b_in),
        .r_out(sobel_if.r_out),
        .g_out(sobel_if.g_out),
        .b_out(sobel_if.b_out)
    );

    // clock/reset
    always #20 sobel_if.clk = ~sobel_if.clk;
    initial begin
        sobel_if.clk   = 1'b1;
        sobel_if.reset = 1'b1;
        sobel_if.den   = 1'b0;
        sobel_if.x_in  = '0;
        sobel_if.y_in  = '0;
        sobel_if.r_in  = '0;
        sobel_if.g_in  = '0;
        sobel_if.b_in  = '0;
        #80;
        sobel_if.reset = 1'b0;
    end

    // run
    initial begin
        env = new(sobel_if);
        #100;
        env.run(10000); 
    end
endmodule
```
