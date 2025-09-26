`timescale 1ns / 1ps

module btn_detector (
    input  wire       clk,
    input  wire       reset,
    input  wire [5:0] btn_in,
    output wire [2:0] selected_kick,
    output wire       o_btnU
    );

    wire [4:0] btn_out;

    kick_position U_Kick_Position (
        .i_btn        (btn_out),
        .selected_kick(selected_kick)
    );

    button_detector U_BTN_A (
        .clk         (clk),
        .reset       (reset),
        .i_btn       (btn_in[0]),
        .rising_edge (btn_out[0]),
        .falling_edge(),
        .both_edge   ()
    );

    button_detector U_BTN_B (
        .clk         (clk),
        .reset       (reset),
        .i_btn       (btn_in[1]),
        .rising_edge (btn_out[1]),
        .falling_edge(),
        .both_edge   ()
    );

    button_detector U_BTN_C (
        .clk         (clk),
        .reset       (reset),
        .i_btn       (btn_in[2]),
        .rising_edge (btn_out[2]),
        .falling_edge(),
        .both_edge   ()
    );

    button_detector U_BTN_D (
        .clk         (clk),
        .reset       (reset),
        .i_btn       (btn_in[3]),
        .rising_edge (btn_out[3]),
        .falling_edge(),
        .both_edge   ()
    );

    button_detector U_BTN_E (
        .clk         (clk),
        .reset       (reset),
        .i_btn       (btn_in[4]),
        .rising_edge (btn_out[4]),
        .falling_edge(),
        .both_edge   ()
    );

    button_detector U_BTN_U (
        .clk         (clk),
        .reset       (reset),
        .i_btn       (btn_in[5]),
        .rising_edge (o_btnU),
        .falling_edge(),
        .both_edge   ()
    );

endmodule


module kick_position (
    input  wire [4:0] i_btn,
    output reg  [2:0] selected_kick
);

    always @(*) begin
        selected_kick = 3'd0;
        case (i_btn)
            5'b10000: selected_kick = 3'd1;
            5'b01000: selected_kick = 3'd2;
            5'b00100: selected_kick = 3'd3;
            5'b00010: selected_kick = 3'd4;
            5'b00001: selected_kick = 3'd5;
        endcase
    end

endmodule

module button_detector (
    input  wire clk,
    input  wire reset,
    input  wire i_btn,
    output wire rising_edge,
    output wire falling_edge,
    output wire both_edge
);

    reg                       clk_1khz;
    wire                      debounce;
    wire [7:0]                shift_reg;
    reg [$clog2(100_000)-1:0] div_counter;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            div_counter <= 0;
            clk_1khz    <= 1'b0;
        end else begin
            if (div_counter == 100_000 - 1) begin
                div_counter <= 0;
                clk_1khz    <= 1'b1;
            end else begin
                div_counter <= div_counter + 1;
                clk_1khz    <= 1'b0;
            end
        end
    end

    shift_register U_SHIFT_REG (
        .clk   (clk_1khz),
        .reset (reset),
        .i_data(i_btn),
        .o_data(shift_reg)
    );

    assign debounce = &shift_reg;

    reg [1:0] edge_reg;

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            edge_reg <= 0;
        end else begin
            edge_reg[0] <= debounce;
            edge_reg[1] <= edge_reg[0];
        end
    end

    assign rising_edge  = edge_reg[0] & ~edge_reg[1];
    assign falling_edge = ~edge_reg[0] & edge_reg[1];
    assign both_edge    = rising_edge | falling_edge;

endmodule


module shift_register (
    input  wire       clk,
    input  wire       reset,
    input  wire       i_data,
    output reg  [7:0] o_data
);

    always @(posedge clk or posedge reset) begin
        if (reset) begin
            o_data <= 0;
        end else begin
            o_data <= {i_data, o_data[7:1]};  // right shift
        end
    end

endmodule
