`timescale 1ns / 1ps

module top_UpDownCounter (
    input  logic        clk,
    input  logic        reset,
    input  logic [4:0]  sw,             
    input  logic        btn_mode,
    input  logic        btn_run_stop,
    input  logic        btn_clear,
    input  logic        btn_send,       
    input  logic        rx,
    output logic        tx,
    output logic [3:0]  fndCom,
    output logic [7:0]  fndFont,
    output logic [1:0]  led_mode,
    output logic [1:0]  led_run_stop
);
    logic [ 7:0]  rx_data;
    logic [13:0]  count;
    logic         btn_mode_edge, btn_run_stop_edge, btn_clear_edge;
    logic         rx_done;

    logic [7:0]   tx_data_sw;       
    logic         start_tx;        
    logic         btn_send_edge;    

    button_detector U_BTN_SEND (
        .clk         (clk),
        .reset       (reset),
        .in_button   (btn_send),
        .rising_edge (btn_send_edge),
        .falling_edge(),
        .both_edge   ()
    );

    button_detector U_BTN_MODE (
        .clk         (clk),
        .reset       (reset),
        .in_button   (btn_mode),
        .rising_edge (),
        .falling_edge(btn_mode_edge),
        .both_edge   ()
    );

    button_detector U_BTN_RUN_STOP (
        .clk         (clk),
        .reset       (reset),
        .in_button   (btn_run_stop),
        .rising_edge (btn_run_stop_edge),
        .falling_edge(),
        .both_edge   ()
    );

    button_detector U_BTN_CLEAR (
        .clk         (clk),
        .reset       (reset),
        .in_button   (btn_clear),
        .rising_edge (),
        .falling_edge(btn_clear_edge),
        .both_edge   ()
    );


    assign start_tx = btn_send_edge;

    Data_LUT U_Data_LUT (
        .sw         (sw),
        .tx_data_sw (tx_data_sw)
    );

    uart U_UART (
        .clk        (clk),
        .reset      (reset),
        .start      (start_tx),    
        .tx_data    (tx_data_sw),  
        .tx_busy    (),
        .tx_done    (),
        .tx         (tx),
 
        .rx         (rx),
        .rx_data    (rx_data),
        .rx_done    (rx_done)
    );

    UpDownCounter U_UpDownCounter (
        .clk         (clk),
        .reset       (reset),
        .btn_mode    (btn_mode_edge),
        .btn_run_stop(btn_run_stop_edge),
        .btn_clear   (btn_clear_edge),
        .rx_done     (rx_done),
        .rx_data     (rx_data),
        .led_mode    (led_mode),
        .led_run_stop(led_run_stop),
        .count       (count)
    );

    fndController U_FndController (
        .clk    (clk),
        .reset  (reset),
        .number (count),
        .fndCom (fndCom),
        .fndFont(fndFont)
    );
endmodule

module Data_LUT (
    input  logic [4:0] sw,
    output logic [7:0] tx_data_sw
);

    always_comb begin
        tx_data_sw = 8'h00;
        case (sw)
            5'b00001: tx_data_sw = 8'h31; 
            5'b00010: tx_data_sw = 8'h32; 
            5'b00100: tx_data_sw = 8'h33; 
            5'b01000: tx_data_sw = 8'h34; 
            5'b10000: tx_data_sw = 8'h35;  
        endcase
    end
    
endmodule