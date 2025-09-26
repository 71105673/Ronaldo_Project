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
                if ((tr.r_data == max_val) && (delta >= 3) &&  // 채도 조건
                    ((tr.r_data - tr.g_data) >= 2) && ((tr.r_data - tr.b_data) >= 2)) begin
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
                if ((tr.r_data == max_val) && (delta >= 3) &&  // 채도 조건
                    ((tr.r_data - tr.g_data) >= 2) && ((tr.r_data - tr.b_data) >= 2)) begin
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
