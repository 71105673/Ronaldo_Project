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
        detect = (tr.r_in == max_val) && (delta >= 3) && ((tr.r_in - tr.g_in) >= 2) && ((tr.r_in - tr.b_in) >= 2);
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


/*
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

// ====================== GENERATOR ======================
class generator;
    transaction tr;
    mailbox #(transaction) gen2drv_mbox;
    virtual red_glove_intf red_glove_if;   // CHANGED

    function new(mailbox#(transaction) gen2drv_mbox,
                 virtual red_glove_intf red_glove_if); // CHANGED
        this.gen2drv_mbox = gen2drv_mbox;
        this.red_glove_if = red_glove_if;  // CHANGED
    endfunction  //new()

    task run(int gen_cnt);
        repeat (gen_cnt) begin
            for (int j = 0; j < 525; j++) begin  //y
                for (int i = 0; i < 800; i++) begin  //x
                    @(posedge red_glove_if.pclk);   // CHANGED: 클록 동기 페이싱
                    tr = new();
                    tr.randomize();
                    tr.DE      = ((i < 640) && (j < 480)) ? 1 : 0;
                    tr.v_sync  = !((j >= 490) && (j <= 492));
                    tr.x_pixel = i;
                    gen2drv_mbox.put(tr);
                    // #40;  // 삭제 (절대지연 제거)  // CHANGED
                end
            end
        end
    endtask
endclass  //generator

// ====================== DRIVER ======================
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
            @(negedge red_glove_if.pclk);     // CHANGED: negedge에 구동
            red_glove_if.r_in    <= tr.r_in;
            red_glove_if.g_in    <= tr.g_in;
            red_glove_if.b_in    <= tr.b_in;
            red_glove_if.DE      <= tr.DE;
            red_glove_if.v_sync  <= tr.v_sync;
            red_glove_if.x_pixel <= tr.x_pixel;
        end
    endtask
endclass  //driver

// ====================== MONITOR ======================
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
            @(posedge red_glove_if.pclk);     // CHANGED: posedge 즉시 샘플
            // #1;  // 삭제                      // CHANGED
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
    endtask
endclass  //monitor

// ====================== SCOREBOARD ======================
class scoreboard;
    // ----- Mailbox / Transaction -----
    transaction tr;
    mailbox #(transaction) mon2scr_mbox;

    // ----- Red detection (동일 임계) -----
    bit [3:0] max_val, min_val, delta;
    bit       detect;

    // ----- Grid 누적/선택 -----
    bit [15:0] Grid[0:4];     // 16비트 여유
    bit [15:0] Grid_temp;     // 현재 최대값
    bit [2:0]  grid_pos;      // x_pixel[9:7]
    bit [2:0]  grid_selected; // 0=없음, 1~5

    // ----- 색상 강조(옵션, 유지) -----
    bit [3:0]  r_data, g_data, b_data;

    // ----- v_sync 에지 검출 -----
    bit v_sync_temp;

    // ----- 비교 지연용 래치 -----
    bit        compare_pending;
    bit [2:0]  exp_grid_selected; // falling에서 저장한 기대값
    bit [15:0] dumpG[0:4];        // falling 시점 스냅샷

    localparam int unsigned SELECT_GRID_MIN = 16'd500;

    function new(mailbox#(transaction) mon2scr_mbox);
        this.mon2scr_mbox = mon2scr_mbox;
    endfunction

    // 빨강 검출 (DUT와 동일식)
    task red_detect();
        max_val = (tr.r_in >= tr.g_in && tr.r_in >= tr.b_in) ? tr.r_in
                : (tr.g_in >= tr.b_in) ? tr.g_in : tr.b_in;
        min_val = (tr.r_in <= tr.g_in && tr.r_in <= tr.b_in) ? tr.r_in
                : (tr.g_in <= tr.b_in) ? tr.g_in : tr.b_in;
        delta   = max_val - min_val;
        detect  = (tr.r_in == max_val) && (delta >= 3)
                  && ((tr.r_in - tr.g_in) >= 2) && ((tr.r_in - tr.b_in) >= 2);
    endtask

    // Grid 선택 영역 색상 강조(옵션)
    task grid_selector();
        if (tr.DE) begin
            case (grid_selected)
                3'd1: if (tr.x_pixel < 128)
                          {r_data,g_data,b_data} = {4'd15,4'd15,4'd0};
                3'd2: if (tr.x_pixel >= 128 && tr.x_pixel < 256)
                          {r_data,g_data,b_data} = {4'd15,4'd15,4'd0};
                3'd3: if (tr.x_pixel >= 256 && tr.x_pixel < 384)
                          {r_data,g_data,b_data} = {4'd15,4'd15,4'd0};
                3'd4: if (tr.x_pixel >= 384 && tr.x_pixel < 512)
                          {r_data,g_data,b_data} = {4'd15,4'd15,4'd0};
                3'd5: if (tr.x_pixel >= 512 && tr.x_pixel < 640)
                          {r_data,g_data,b_data} = {4'd15,4'd15,4'd0};
                default: ;
            endcase
        end
    endtask

    task run();
        // 초기화
        v_sync_temp     = 1'b0;
        compare_pending = 1'b0;
        for (int i=0;i<5;i++) Grid[i] = '0;
        grid_selected   = 3'd0;

        forever begin
            mon2scr_mbox.get(tr);

            // 1) v_sync Rising: 새 프레임 시작 → 카운터 클리어 (DUT와 동일 시점)
            if (tr.v_sync && !v_sync_temp) begin
                for (int i=0;i<5;i++) Grid[i] = '0;
                grid_selected = 3'd0;
            end

            // 2) v_sync Falling: 방금 끝난 프레임 종료 → 기대값만 저장 (비교는 다음 사이클)
            if (!tr.v_sync && v_sync_temp) begin
                exp_grid_selected = grid_selected;
                for (int i=0;i<5;i++) dumpG[i] = Grid[i];
                compare_pending = 1'b1;
            end

            // 3) Active 구간 누적 및 최대 추적 (DUT와 동일 인덱싱: x[9:7])
            red_detect();
            if (tr.DE) begin
                grid_pos = tr.x_pixel[9:7];
                case (grid_pos)
                    3'd0: if (detect) Grid[0]++;
                    3'd1: if (detect) Grid[1]++;
                    3'd2: if (detect) Grid[2]++;
                    3'd3: if (detect) Grid[3]++;
                    3'd4: if (detect) Grid[4]++;
                    default: ;
                endcase

                // 현재까지의 최대값/그리드 (동률 시 앞쪽 유지)
                grid_selected = 3'd1; Grid_temp = Grid[0];
                if (Grid[1] > Grid_temp) begin grid_selected = 3'd2; Grid_temp = Grid[1]; end
                if (Grid[2] > Grid_temp) begin grid_selected = 3'd3; Grid_temp = Grid[2]; end
                if (Grid[3] > Grid_temp) begin grid_selected = 3'd4; Grid_temp = Grid[3]; end
                if (Grid[4] > Grid_temp) begin grid_selected = 3'd5; Grid_temp = Grid[4]; end
                if (Grid_temp <= SELECT_GRID_MIN) grid_selected = 3'd0;
            end

            // 4) 한 사이클 늦춰 비교 (DUT의 selected_grid 갱신 완료 후)
            if (compare_pending) begin
                // exp_grid_selected는 직전 falling 시점 기준 기대값
                if (tr.selected_grid == exp_grid_selected) begin
                    $display("---------------------------------------------------------");
                    $display("PASS! : grid = %0d, grid_val = %0d",
                             tr.selected_grid,
                             dumpG[(exp_grid_selected==0)?0:(exp_grid_selected-1)]);
                    $display("---------------------------------------------------------");
                end else begin
                    $display("---------------------------------------------------------");
                    $display("FAIL! : grid was wrong!!!");
                    $display("filter grid = %0d, tb grid = %0d",
                             tr.selected_grid, exp_grid_selected);
                    $display("Grid1=%0d, Grid2=%0d, Grid3=%0d, Grid4=%0d, Grid5=%0d",
                              dumpG[0], dumpG[1], dumpG[2], dumpG[3], dumpG[4]);
                    $display("---------------------------------------------------------");
                end
                compare_pending = 1'b0;
            end

            // 5) 에지 상태 업데이트
            v_sync_temp = tr.v_sync;

            // (옵션) 색상 강조 로직 유지
            grid_selector();
        end
    endtask
endclass


// ====================== ENV ======================
class environment;
    generator  gen;
    driver     drv;
    monitor    mon;
    scoreboard scr;
    mailbox #(transaction) gen2drv_mbox;
    mailbox #(transaction) mon2scr_mbox;
    virtual red_glove_intf red_glove_if;

    function new(virtual red_glove_intf red_glove_if);
        this.red_glove_if = red_glove_if;
        gen2drv_mbox = new(1);   // CHANGED: 용량 1로 백프레셔
        mon2scr_mbox = new();
        gen = new(gen2drv_mbox, red_glove_if);    // CHANGED: IF 전달
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
        $finish();
    endtask
endclass  //environment

// ====================== TB TOP ======================
module tb_red_glove_grid_select ();
    environment   env;
    red_glove_intf red_glove_if ();

    // DUT는 네가 준 그대로 인스턴스한다고 가정
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

    // 25MHz (T=40ns)
    always #20 red_glove_if.pclk = ~red_glove_if.pclk;

    initial begin
        red_glove_if.pclk  = 1;
        red_glove_if.reset = 1;
        red_glove_if.r_in  = 0;
        red_glove_if.g_in  = 0;
        red_glove_if.b_in  = 0;
        red_glove_if.v_sync= 0;
        red_glove_if.DE    = 0;
        red_glove_if.x_pixel = 0;
        #40;
        red_glove_if.reset = 0;
    end

    initial begin
        #40;
        env = new(red_glove_if);
        env.run(7);
    end
endmodule
*/
