
`timescale 1 ns / 1 ps

	module UART_FP_v1_0 #
	(
		// Users to add parameters here

		// User parameters ends
		// Do not modify the parameters beyond this line


		// Parameters of Axi Slave Bus Interface S00_AXI
		parameter integer C_S00_AXI_DATA_WIDTH	= 32,
		parameter integer C_S00_AXI_ADDR_WIDTH	= 4
	)
	(
		// Users to add ports here
		output tx,
		input  rx,
		// User ports ends
		// Do not modify the ports beyond this line


		// Ports of Axi Slave Bus Interface S00_AXI
		input wire  s00_axi_aclk,
		input wire  s00_axi_aresetn,
		input wire [C_S00_AXI_ADDR_WIDTH-1 : 0] s00_axi_awaddr,
		input wire [2 : 0] s00_axi_awprot,
		input wire  s00_axi_awvalid,
		output wire  s00_axi_awready,
		input wire [C_S00_AXI_DATA_WIDTH-1 : 0] s00_axi_wdata,
		input wire [(C_S00_AXI_DATA_WIDTH/8)-1 : 0] s00_axi_wstrb,
		input wire  s00_axi_wvalid,
		output wire  s00_axi_wready,
		output wire [1 : 0] s00_axi_bresp,
		output wire  s00_axi_bvalid,
		input wire  s00_axi_bready,
		input wire [C_S00_AXI_ADDR_WIDTH-1 : 0] s00_axi_araddr,
		input wire [2 : 0] s00_axi_arprot,
		input wire  s00_axi_arvalid,
		output wire  s00_axi_arready,
		output wire [C_S00_AXI_DATA_WIDTH-1 : 0] s00_axi_rdata,
		output wire [1 : 0] s00_axi_rresp,
		output wire  s00_axi_rvalid,
		input wire  s00_axi_rready
	);

	// Users to add ports here
    wire [31:0] csr;
    wire [31:0] tx_data;
    wire [31:0] rx_data;
    wire        rx_done;
    wire        tx_fifo_full;
    wire        tx_fifo_empty;
    // User ports ends

// Instantiation of Axi Bus Interface S00_AXI
	UART_FP_v1_0_S00_AXI # ( 
		.C_S_AXI_DATA_WIDTH(C_S00_AXI_DATA_WIDTH),
		.C_S_AXI_ADDR_WIDTH(C_S00_AXI_ADDR_WIDTH)
	) UART_FP_v1_0_S00_AXI_inst (
    	.csr(csr),
    	.tx_data(tx_data),
    	.rx_data(rx_data),
    	.rx_done(rx_done),
    	.tx_fifo_full(tx_fifo_full),
    	.tx_fifo_empty(tx_fifo_empty),
		.S_AXI_ACLK(s00_axi_aclk),
		.S_AXI_ARESETN(s00_axi_aresetn),
		.S_AXI_AWADDR(s00_axi_awaddr),
		.S_AXI_AWPROT(s00_axi_awprot),
		.S_AXI_AWVALID(s00_axi_awvalid),
		.S_AXI_AWREADY(s00_axi_awready),
		.S_AXI_WDATA(s00_axi_wdata),
		.S_AXI_WSTRB(s00_axi_wstrb),
		.S_AXI_WVALID(s00_axi_wvalid),
		.S_AXI_WREADY(s00_axi_wready),
		.S_AXI_BRESP(s00_axi_bresp),
		.S_AXI_BVALID(s00_axi_bvalid),
		.S_AXI_BREADY(s00_axi_bready),
		.S_AXI_ARADDR(s00_axi_araddr),
		.S_AXI_ARPROT(s00_axi_arprot),
		.S_AXI_ARVALID(s00_axi_arvalid),
		.S_AXI_ARREADY(s00_axi_arready),
		.S_AXI_RDATA(s00_axi_rdata),
		.S_AXI_RRESP(s00_axi_rresp),
		.S_AXI_RVALID(s00_axi_rvalid),
		.S_AXI_RREADY(s00_axi_rready)
	);

	// Add user logic here
	uart_intf U_uart_intf (
    .clk(s00_axi_aclk),
    .reset(~s00_axi_aresetn),
    
    .cr(csr),
    .rx_done(rx_done),
    .rx_data(rx_data),
    .tx_data(tx_data),
    .tx_fifo_full(tx_fifo_full),
    .tx_fifo_empty(tx_fifo_empty),

    .rx(rx),
    .tx(tx)
);
	// User logic ends

	endmodule



module uart_intf (
    input          clk,
    input          reset,
    
    input   [31:0] cr,
    output         rx_done,
    output [31:0] rx_data,
    input   [31:0] tx_data,
    output tx_fifo_full,
    output tx_fifo_empty,

    input   rx,
    output  tx
);
    wire start, tx_done, tx_busy, push;

    wire [7:0] rx_data_temp, tx_data_temp;
    assign rx_data = {24'b0 ,rx_data_temp};
    assign push = cr[0] && cr[8];
    //assign push = cr[0] && cr[8] && ((cr[5] && tx_data[7:5] == 3'b001) || (cr[6] && tx_data[7:5] == 3'b010) || (cr[7] && tx_data[7:5] == 3'b011));

    uart U_UART (
        .clk    (clk),
        .reset  (reset),
        .en     (cr[1:0]),
        .start  (start),
        .tx_data(tx_data_temp),
        .rx     (rx),
        .tx_busy(tx_busy),
        .tx_done(tx_done),
        .tx     (tx),
        .rx_busy(),
        .rx_done(rx_done),
        .rx_data(rx_data_temp)
    );

    fifo tx_fifo (
        .clk      (clk),
        .reset    (reset),
        .push_data(tx_data[7:0]),
        .push     (push),
        .valid    (start),
        .pop_data (tx_data_temp),
        //.pop      (tx_done || !tx_busy || !tx_fifo_empty),
        .pop((!tx_busy && !tx_fifo_empty)|| (tx_done && !tx_fifo_empty)),   // idle ?Üí Ï≤? ?ãú?ûë  // done ?Üí ?ù¥?ñ¥?Ñú ?†Ñ?Ü°
        .empty    (tx_fifo_empty),
        .full     (tx_fifo_full)
    );


endmodule

module fifo (
    input         clk,
    input         reset,
    input   [7:0] push_data,
    input         push,
    output reg       valid,
    output reg [7:0] pop_data,
    input         pop,
    output empty,
    output full
);

    reg [7:0] mem[0:3];
    reg [1:0] front, rear;

    assign empty = (front == rear);
    assign full  = ((front + 1) == rear);

    always @(posedge clk, posedge reset) begin
        if (reset) begin
            mem[0] <= 0;
            mem[1] <= 0;
            mem[2] <= 0;
            mem[3] <= 0;
            front <= 0;
            rear <= 0;
            pop_data <= 0;
			valid <= 0;
        end else begin
            valid <= 1'b0;
            if (push && !full) begin
                mem[front] <= push_data;
                front <= front + 1;
            end

            if (pop && !empty) begin
                pop_data <= mem[rear];
                rear <= rear + 1;
                valid <= 1'b1;
            end
        end
    end

endmodule

module uart (
    input         clk,
    input         reset,
    input   [1:0] en,
    input         start,
    input   [7:0] tx_data,
    input         rx,
    output        tx_busy,
    output        tx_done,
    output        tx,
    output        rx_busy,
    output        rx_done,
    output  [7:0] rx_data
);

    wire br_tick;

    baudrate_gen U_BAUD_GEN (
        .clk    (clk),
        .reset  (reset),
        .br_tick(br_tick)
    );

    transmitter U_Transmitter (
        .clk    (clk),
        .reset  (reset),
        .en     (en[0]),
        .br_tick(br_tick),
        .start  (start),
        .tx_data(tx_data),
        .tx_busy(tx_busy),
        .tx_done(tx_done),
        .tx     (tx)
    );

    receiver U_Receiver (
        .clk(clk),
        .reset(reset),
        .en     (en[1]),
        .br_tick(br_tick),
        .rx(rx),
        .rx_busy(rx_busy),
        .rx_done(rx_done),
        .rx_data(rx_data)
    );

endmodule

module baudrate_gen (
    input   clk,
    input   reset,
    output reg br_tick
);

    reg [$clog2(100_000_000 / 9600 / 16)-1:0] br_counter;
    //logic [3:0] br_counter;  //simulation

    always @(posedge clk, posedge reset) begin
        if (reset) begin
            br_counter <= 0;
            br_tick    <= 1'b0;
        end else begin
            if (br_counter == 100_000_000 / 9600 / 16 - 1) begin
                //if (br_counter == 10 - 1) begin  //simulation
                br_counter <= 0;
                br_tick <= 1'b1;
            end else begin
                br_counter <= br_counter + 1;
                br_tick <= 1'b0;
            end
        end
    end

endmodule

module transmitter (
    input         clk,
    input         reset,
    input         en,
    input         br_tick,
    input         start,
    input   [7:0] tx_data,
    output       tx_busy,
    output       tx_done,
    output       tx
);

	localparam IDLE = 0;
    localparam START = 1;
    localparam DATA = 2;
    localparam STOP = 3;

    reg [1:0] tx_state, tx_next_state;

    reg [7:0] temp_data_reg, temp_data_next;
    reg tx_reg, tx_next;
    reg [3:0] tick_cnt_reg, tick_cnt_next;
    reg [2:0] bit_cnt_reg, bit_cnt_next;
    reg tx_done_reg, tx_done_next;
    reg tx_busy_reg, tx_busy_next;

    assign tx = tx_reg;
    assign tx_busy = tx_busy_reg;
    assign tx_done = tx_done_reg;

    always @(posedge clk, posedge reset) begin
        if (reset) begin
            tx_state      <= IDLE;
            temp_data_reg <= 0;
            tx_reg        <= 1'b1;
            tick_cnt_reg  <= 0;
            bit_cnt_reg   <= 0;
            tx_done_reg   <= 0;
            tx_busy_reg   <= 0;
        end else begin
            tx_state      <= tx_next_state;
            temp_data_reg <= temp_data_next;
            tx_reg        <= tx_next;
            tick_cnt_reg  <= tick_cnt_next;
            bit_cnt_reg   <= bit_cnt_next;
            tx_done_reg   <= tx_done_next;
            tx_busy_reg   <= tx_busy_next;
        end
    end

    always@(*) begin
        tx_next_state = tx_state;
        temp_data_next = temp_data_reg; //latch Î∞©Ï?Î•? ?úÑ?ï¥ reg, next ?òï?ÉúÎ°? ?Ç¨?ö©
        tx_next = tx_reg;
        tick_cnt_next = tick_cnt_reg;
        bit_cnt_next = bit_cnt_reg;
        tx_done_next = tx_done_reg;
        tx_busy_next = tx_busy_reg;
        case (tx_state)
            IDLE: begin
                tx_next = 1'b1;
                tx_done_next = 0;
                tx_busy_next = 0;
                if (start && en) begin
                    tx_next_state  = START;
                    temp_data_next = tx_data;
                    tick_cnt_next  = 0;
                    bit_cnt_next   = 0;
                    tx_busy_next   = 1;
                end
            end
            START: begin
                tx_next = 1'b0;
                if (br_tick) begin
                    if (tick_cnt_reg == 15) begin
                        tx_next_state = DATA;
                        tick_cnt_next = 0;
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
            DATA: begin
                tx_next = temp_data_reg[0];
                if (br_tick) begin
                    if (tick_cnt_reg == 15) begin
                        tick_cnt_next = 0;
                        if (bit_cnt_reg == 7) begin
                            tx_next_state = STOP;
                            bit_cnt_next  = 0;
                        end else begin
                            temp_data_next = {
                                1'b0, temp_data_reg[7:1]
                            };  //?ïò?Çò?î© 0bit Í≤ÉÏùÑ Î≥¥ÎÇ¥Î©¥ÏÑú shift
                            bit_cnt_next = bit_cnt_reg + 1;
                        end
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
            STOP: begin
                tx_next = 1'b1;
                if (br_tick) begin
                    if (tick_cnt_reg == 15) begin
                        tx_next_state = IDLE;
                        tx_done_next  = 1;
                        tx_busy_next  = 0;
                        tick_cnt_next = 0;
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
        endcase
    end

endmodule

module receiver (
    input         clk,
    input         reset,
    input         en,
    input         br_tick,
    input         rx,
    output        rx_busy,
    output        rx_done,
    output  [7:0] rx_data
);

	localparam IDLE = 0;
    localparam START = 1;
    localparam DATA = 2;
    localparam STOP = 3;

    reg [1:0] rx_state, rx_next_state;

    reg [7:0] temp_data_reg, temp_data_next;
    reg [7:0] rx_reg, rx_next;
    reg [4:0] tick_cnt_reg, tick_cnt_next;
    reg [2:0] bit_cnt_reg, bit_cnt_next;
    reg rx_done_reg, rx_done_next;
    reg rx_busy_reg, rx_busy_next;

    assign rx_data = rx_reg;
    assign rx_busy = rx_busy_reg;
    assign rx_done = rx_done_reg;

    always @(posedge clk, posedge reset) begin
        if (reset) begin
            rx_state      <= IDLE;
            rx_reg        <= 0;
            temp_data_reg <= 0;
            tick_cnt_reg  <= 0;
            bit_cnt_reg   <= 0;
            rx_done_reg   <= 0;
            rx_busy_reg   <= 0;
        end else begin
            rx_state      <= rx_next_state;
            rx_reg        <= rx_next;
            temp_data_reg <= temp_data_next;
            tick_cnt_reg  <= tick_cnt_next;
            bit_cnt_reg   <= bit_cnt_next;
            rx_done_reg   <= rx_done_next;
            rx_busy_reg   <= rx_busy_next;
        end
    end

    always@(*) begin
        rx_next_state  = rx_state;
        rx_next        = rx_reg;
        temp_data_next = temp_data_reg;
        tick_cnt_next  = tick_cnt_reg;
        bit_cnt_next   = bit_cnt_reg;
        rx_done_next   = rx_done_reg;
        rx_busy_next   = rx_busy_reg;
        case (rx_state)
            IDLE: begin
                rx_busy_next = 0;
                rx_done_next = 0;
                if ((~rx) && en) begin
                    rx_next_state  = START;
                    temp_data_next = 0;
                    tick_cnt_next  = 0;
                    bit_cnt_next   = 0;
                    rx_busy_next   = 1;
                    rx_next        = 0;
                end
            end
            START: begin
                if (br_tick) begin
                    if (tick_cnt_reg == 7) begin
                        rx_next_state = DATA;
                        tick_cnt_next = 0;
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
            DATA: begin
                if (br_tick) begin
                    if (tick_cnt_reg == 15) begin
                        temp_data_next = {rx, temp_data_reg[7:1]};
                        tick_cnt_next  = 0;
                        if (bit_cnt_reg == 7) begin
                            rx_next_state = STOP;
                            bit_cnt_next  = 0;
                        end else begin
                            bit_cnt_next = bit_cnt_reg + 1;
                        end
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
            STOP: begin
                if (br_tick) begin
                    if (tick_cnt_reg == 23) begin
                        rx_next_state = IDLE;
                        tick_cnt_next = 0;
                        rx_busy_next  = 0;
                        rx_done_next  = 1;
                        rx_next       = temp_data_reg;
                    end else begin
                        tick_cnt_next = tick_cnt_reg + 1;
                    end
                end
            end
        endcase
    end

endmodule
