
`timescale 1 ns / 1 ps

	module SCCB_final_v1_0 #
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
		output wire sioc,
		output wire siod,
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

		wire start;
		wire done;
// Instantiation of Axi Bus Interface S00_AXI
	SCCB_final_v1_0_S00_AXI # ( 
		.C_S_AXI_DATA_WIDTH(C_S00_AXI_DATA_WIDTH),
		.C_S_AXI_ADDR_WIDTH(C_S00_AXI_ADDR_WIDTH)
	) SCCB_final_v1_0_S00_AXI_inst (
		.start(start),
		.done(done),
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
	i2c_ov7670_init U_SCCB(
    .clk(s00_axi_aclk),
    .start(start),
    .sioc(sioc),
    .siod(siod),
    .done(done)
	);
	// User logic ends

	endmodule

module i2c_ov7670_init
    #(
    parameter CLK_FREQ=100000000
    )
    (
    input wire clk,
    input wire start,
    output wire sioc,
    output wire siod,
    output wire done
    );
    
    wire [7:0] rom_addr;
    wire [15:0] rom_dout;
    wire [7:0] SCCB_addr;
    wire [7:0] SCCB_data;
    wire SCCB_start;
    wire SCCB_ready;
    wire SCCB_SIOC_oe;
    wire SCCB_SIOD_oe;
    
    
    assign sioc = SCCB_SIOC_oe ? 1'b0 : 1'b1;
    assign siod = SCCB_SIOD_oe ? 1'b0 : 1'b1;
    
    OV7670_config_rom rom1(
        .clk(clk),
        .addr(rom_addr),
        .dout(rom_dout)
        );
        
    OV7670_config #(.CLK_FREQ(CLK_FREQ)) config_1(
        .clk(clk),
        .SCCB_interface_ready(SCCB_ready),
        .rom_data(rom_dout),
        .start(start),
        .rom_addr(rom_addr),
        .done(done),
        .SCCB_interface_addr(SCCB_addr),
        .SCCB_interface_data(SCCB_data),
        .SCCB_interface_start(SCCB_start)
        );
    
    SCCB_interface #( .CLK_FREQ(CLK_FREQ)) SCCB1(
        .clk(clk),
        .start(SCCB_start),
        .address(SCCB_addr),
        .data(SCCB_data),
        .ready(SCCB_ready),
        .SIOC_oe(SCCB_SIOC_oe),
        .SIOD_oe(SCCB_SIOD_oe)
        );
    
endmodule

module OV7670_config
#(
    parameter CLK_FREQ = 100000000
)
(
    input wire clk,
    input wire SCCB_interface_ready,
    input wire [15:0] rom_data,
    input wire start,
    output reg [7:0] rom_addr,
    output reg done,
    output reg [7:0] SCCB_interface_addr,
    output reg [7:0] SCCB_interface_data,
    output reg SCCB_interface_start
    );
    
    initial begin
        rom_addr = 0;
        done = 0;
        SCCB_interface_addr = 0;
        SCCB_interface_data = 0;
        SCCB_interface_start = 0;
    end
    
    localparam FSM_IDLE = 0;
    localparam FSM_SEND_CMD = 1;
    localparam FSM_DONE = 2;
    localparam FSM_TIMER = 3;
    
    reg [2:0] FSM_state = FSM_IDLE;
    reg [2:0] FSM_return_state;
    reg [31:0] timer = 0; 
    
    always@(posedge clk) begin
    
        case(FSM_state)
            
            FSM_IDLE: begin 
                FSM_state <= start ? FSM_SEND_CMD : FSM_IDLE;
                rom_addr <= 0;
                done <= start ? 0 : done;
            end
            
            FSM_SEND_CMD: begin 
                case(rom_data)
                    16'hFFFF: begin //end of ROM
                        FSM_state <= FSM_DONE;
                    end
                    
                    16'hFFF0: begin //delay state 
                        timer <= (CLK_FREQ/100); //10 ms delay
                        FSM_state <= FSM_TIMER;
                        FSM_return_state <= FSM_SEND_CMD;
                        rom_addr <= rom_addr + 1;
                    end
                    
                    default: begin //normal rom commands
                        if (SCCB_interface_ready) begin
                            FSM_state <= FSM_TIMER;
                            FSM_return_state <= FSM_SEND_CMD;
                            timer <= 0; //one cycle delay gives ready chance to deassert
                            rom_addr <= rom_addr + 1;
                            SCCB_interface_addr <= rom_data[15:8];
                            SCCB_interface_data <= rom_data[7:0];
                            SCCB_interface_start <= 1;
                        end
                    end
                endcase
            end
                        
            FSM_DONE: begin //signal done 
                FSM_state <= FSM_IDLE;
                done <= 1;
            end
                           
                
            FSM_TIMER: begin //count down and jump to next state
                FSM_state <= (timer == 0) ? FSM_return_state : FSM_TIMER;
                timer <= (timer==0) ? 0 : timer - 1;
                SCCB_interface_start <= 0;
            end
        endcase
    end
endmodule

    module OV7670_config_rom(
    input wire clk,
    input wire [7:0] addr,
    output reg [15:0] dout
    );
    //FFFF is end of rom, FFF0 is delay
    always @(posedge clk) begin
    case(addr) 
    /*
    0:  dout <= 16'h12_80; //reset
    1:  dout <= 16'hFF_F0; //delay
    2:  dout <= 16'h12_14; // COM7,     set RGB color output
    3:  dout <= 16'h11_80; // CLKRC     internal PLL matches input clock
    4:  dout <= 16'h0C_00; // COM3,     default settings
    5:  dout <= 16'h3E_00; // COM14,    no scaling, normal pclock
    6:  dout <= 16'h04_00; // COM1,     disable CCIR656
    7:  dout <= 16'h40_d0; //COM15,     RGB565, full output range
    8:  dout <= 16'h3a_04; //TSLB       set correct output data sequence (magic)
    9:  dout <= 16'h14_18; //COM9       MAX AGC value x4
    10: dout <= 16'h4F_B5; //MTX1       all of these are magical matrix coefficients
    11: dout <= 16'h50_B5; //MTX2
    12: dout <= 16'h51_10; //MTX3
    13: dout <= 16'h52_3c; //MTX4
    14: dout <= 16'h53_A0; //MTX5
    15: dout <= 16'h54_E5; //MTX6
    16: dout <= 16'h58_9E; //MTXS
    17: dout <= 16'h3D_C0; //COM13      sets gamma enable, does not preserve reserved bits, may be wrong?
    18: dout <= 16'h17_15; //HSTART     start high 8 bits
    19: dout <= 16'h18_03; //HSTOP      stop high 8 bits //these kill the odd colored line
    20: dout <= 16'h32_80; //HREF       edge offset
    21: dout <= 16'h19_03; //VSTART     start high 8 bits
    22: dout <= 16'h1A_7B; //VSTOP      stop high 8 bits
    23: dout <= 16'h03_0A; //VREF       vsync edge offset
    24: dout <= 16'h0F_41; //COM6       reset timings
    25: dout <= 16'h1E_00; //MVFP       disable mirror / flip //might have magic value of 03
    26: dout <= 16'h33_0B; //CHLF       //magic value from the internet
    27: dout <= 16'h3C_78; //COM12      no HREF when VSYNC low
    28: dout <= 16'h69_00; //GFIX       fix gain control
    29: dout <= 16'h74_00; //REG74      Digital gain control
    30: dout <= 16'hB0_84; //RSVD       magic value from the internet *required* for good color
    31: dout <= 16'hB1_0c; //ABLC1
    32: dout <= 16'hB2_0e; //RSVD       more magic internet values
    33: dout <= 16'hB3_80; //THL_ST
    //begin mystery scaling numbers
    34: dout <= 16'h70_3a;
    35: dout <= 16'h71_35;
    36: dout <= 16'h72_11;
    37: dout <= 16'h73_f0;
    38: dout <= 16'ha2_02;
    //gamma curve values
    39: dout <= 16'h7a_20;
    40: dout <= 16'h7b_10;
    41: dout <= 16'h7c_1e;
    42: dout <= 16'h7d_35;
    43: dout <= 16'h7e_5a;
    44: dout <= 16'h7f_69;
    45: dout <= 16'h80_76;
    46: dout <= 16'h81_80;
    47: dout <= 16'h82_88;
    48: dout <= 16'h83_8f;
    49: dout <= 16'h84_96;
    50: dout <= 16'h85_a3;
    51: dout <= 16'h86_af;
    52: dout <= 16'h87_c4;
    53: dout <= 16'h88_d7;
    54: dout <= 16'h89_e8; //
    //AGC and AEC
    55: dout <= 16'h13_e0; //COM8, disable AGC / AEC
    56: dout <= 16'h00_00; //set gain reg to 0 for AGC
    57: dout <= 16'h10_00; //set ARCJ reg to 0
    58: dout <= 16'h0d_40; //magic reserved bit for COM4
    59: dout <= 16'h14_18; //COM9, 4x gain + magic bit
    60: dout <= 16'ha5_05; // BD50MAX
    61: dout <= 16'hab_07; //DB60MAX
    62: dout <= 16'h24_95; //AGC upper limit
    63: dout <= 16'h25_33; //AGC lower limit
    64: dout <= 16'h26_e3; //AGC/AEC fast mode op region
    65: dout <= 16'h9f_78; //HAECC1
    66: dout <= 16'ha0_68; //HAECC2
    67: dout <= 16'ha1_03; //magic
    68: dout <= 16'ha6_d8; //HAECC3
    69: dout <= 16'ha7_d8; //HAECC4
    70: dout <= 16'ha8_f0; //HAECC5
    71: dout <= 16'ha9_90; //HAECC6
    72: dout <= 16'haa_94; //HAECC7
    73: dout <= 16'h13_e5; //COM8, disenable AGC / AEC / AWB // 20
    74: dout <= 16'h01_b8; // blue f0
    75: dout <= 16'h02_c0; // red b0
    76: dout <= 16'h6a_68; // green 
    77: dout <= 16'h00_20; // gain 00
    78: dout <= 16'h09_03; // output drive capabillyty
    default: dout <= 16'hFF_FF;         //mark end of ROM
    */
    0:  dout <= 16'h12_80; //reset
    1:  dout <= 16'hFF_F0; //delay
    2:  dout <= 16'h12_14; // COM7,     set RGB color output
    3:  dout <= 16'h11_80; // CLKRC     internal PLL matches input clock
    4:  dout <= 16'h0C_00; // COM3,     default settings
    5:  dout <= 16'h3E_00; // COM14,    no scaling, normal pclock
    6:  dout <= 16'h04_00; // COM1,     disable CCIR656
    7:  dout <= 16'h40_d0; //COM15,     RGB565, full output range
    8:  dout <= 16'h3a_04; //TSLB       set correct output data sequence (magic)
    9:  dout <= 16'h14_18; //COM9       MAX AGC value x4
    10: dout <= 16'h4F_B5; //MTX1       all of these are magical matrix coefficients
    11: dout <= 16'h50_B5; //MTX2
    12: dout <= 16'h51_10; //MTX3
    13: dout <= 16'h52_3c; //MTX4
    14: dout <= 16'h53_A0; //MTX5
    15: dout <= 16'h54_E5; //MTX6
    16: dout <= 16'h58_9E; //MTXS
    17: dout <= 16'h3D_C0; //COM13      sets gamma enable, does not preserve reserved bits, may be wrong?
    18: dout <= 16'h17_15; //HSTART     start high 8 bits
    19: dout <= 16'h18_03; //HSTOP      stop high 8 bits //these kill the odd colored line
    20: dout <= 16'h32_80; //HREF       edge offset
    21: dout <= 16'h19_03; //VSTART     start high 8 bits
    22: dout <= 16'h1A_7B; //VSTOP      stop high 8 bits
    23: dout <= 16'h03_0A; //VREF       vsync edge offset
    24: dout <= 16'h0F_41; //COM6       reset timings
    25: dout <= 16'h1E_00; //MVFP       disable mirror / flip //might have magic value of 03
    26: dout <= 16'h33_0B; //CHLF       //magic value from the internet
    27: dout <= 16'h3C_78; //COM12      no HREF when VSYNC low
    28: dout <= 16'h69_00; //GFIX       fix gain control
    29: dout <= 16'h74_00; //REG74      Digital gain control
    30: dout <= 16'hB0_84; //RSVD       magic value from the internet *required* for good color
    31: dout <= 16'hB1_0c; //ABLC1
    32: dout <= 16'hB2_0e; //RSVD       more magic internet values
    33: dout <= 16'hB3_80; //THL_ST
    //begin mystery scaling numbers
    34: dout <= 16'h70_3a;
    35: dout <= 16'h71_35;
    36: dout <= 16'h72_11;
    37: dout <= 16'h73_f0;
    38: dout <= 16'ha2_02;
    //gamma curve values
    39: dout <= 16'h7a_20;
    40: dout <= 16'h7b_10;
    41: dout <= 16'h7c_1e;
    42: dout <= 16'h7d_35;
    43: dout <= 16'h7e_5a;
    44: dout <= 16'h7f_69;
    45: dout <= 16'h80_76;
    46: dout <= 16'h81_80;
    47: dout <= 16'h82_88;
    48: dout <= 16'h83_8f;
    49: dout <= 16'h84_96;
    50: dout <= 16'h85_a3;
    51: dout <= 16'h86_af;
    52: dout <= 16'h87_c4;
    53: dout <= 16'h88_d7;
    54: dout <= 16'h89_e8; //
    //AGC and AEC
    55: dout <= 16'h13_e0; //COM8, disable AGC / AEC
    56: dout <= 16'h00_00; //set gain reg to 0 for AGC
    57: dout <= 16'h10_00; //set ARCJ reg to 0
    58: dout <= 16'h0d_40; //magic reserved bit for COM4
    59: dout <= 16'h14_18; //COM9, 4x gain + magic bit
    60: dout <= 16'ha5_05; // BD50MAX
    61: dout <= 16'hab_07; //DB60MAX
    62: dout <= 16'h24_95; //AGC upper limit
    63: dout <= 16'h25_33; //AGC lower limit
    64: dout <= 16'h26_e3; //AGC/AEC fast mode op region
    65: dout <= 16'h9f_78; //HAECC1
    66: dout <= 16'ha0_68; //HAECC2
    67: dout <= 16'ha1_03; //magic
    68: dout <= 16'ha6_d8; //HAECC3
    69: dout <= 16'ha7_d8; //HAECC4
    70: dout <= 16'ha8_f0; //HAECC5
    71: dout <= 16'ha9_90; //HAECC6
    72: dout <= 16'haa_94; //HAECC7
    73: dout <= 16'h13_00; //COM8, disenable AGC / AEC / AWB // 20
    74: dout <= 16'h01_b8; // blue f0
    75: dout <= 16'h02_c0; // red b0
    76: dout <= 16'h6a_68; // green 
    77: dout <= 16'h00_20; // gain 00
    78: dout <= 16'h09_03; // output drive capabillyty
    79: dout <= 16'h41_08; // COM16 AWB gain disenalbe
    80: dout <= 16'h0e_01;
    81: dout <= 16'h10_a0; 
    default: dout <= 16'hFF_FF;         //mark end of ROM
    endcase
    
    end
endmodule



module SCCB_interface
#(
    parameter CLK_FREQ = 100000000,
    parameter SCCB_FREQ = 100000
)
(
    input wire clk,
    input wire start,
    input wire [7:0] address,
    input wire [7:0] data,
    output reg ready,
    output reg SIOC_oe,
    output reg SIOD_oe
    );
    
    localparam CAMERA_ADDR = 8'h42;
    localparam FSM_IDLE = 0;
    localparam FSM_START_SIGNAL = 1;
    localparam FSM_LOAD_BYTE = 2;
    localparam FSM_TX_BYTE_1 = 3;
    localparam FSM_TX_BYTE_2 = 4;
    localparam FSM_TX_BYTE_3 = 5;
    localparam FSM_TX_BYTE_4 = 6;
    localparam FSM_END_SIGNAL_1 = 7;
    localparam FSM_END_SIGNAL_2 = 8;
    localparam FSM_END_SIGNAL_3 = 9;
    localparam FSM_END_SIGNAL_4 = 10;
    localparam FSM_DONE = 11;
    localparam FSM_TIMER = 12;
    
    
    initial begin 
        SIOC_oe = 0;
        SIOD_oe = 0;
        ready = 1;
    end
    
    reg [3:0] FSM_state = 0;
    reg [3:0] FSM_return_state = 0;
    reg [31:0] timer = 0;
    reg [7:0] latched_address;
    reg [7:0] latched_data; 
    reg [1:0] byte_counter = 0;
    reg [7:0] tx_byte = 0;
    reg [3:0] byte_index = 0;
    
    
    always@(posedge clk) begin
        
        case(FSM_state) 
        
            FSM_IDLE: begin
                byte_index <= 0;
                byte_counter <= 0;
                if (start) begin 
                    FSM_state <= FSM_START_SIGNAL;
                    latched_address <= address;
                    latched_data <= data;
                    ready <= 0;
                end
                else begin
                    ready <= 1;
                end
            end
            
            FSM_START_SIGNAL: begin //comunication interface start signal, bring SIOD low
                FSM_state <= FSM_TIMER;
                FSM_return_state <= FSM_LOAD_BYTE;
                timer <= (CLK_FREQ/(4*SCCB_FREQ));
                SIOC_oe <= 0;
                SIOD_oe <= 1;
            end
            
            FSM_LOAD_BYTE: begin //load next byte to be transmitted
                FSM_state <= (byte_counter == 3) ? FSM_END_SIGNAL_1 : FSM_TX_BYTE_1;
                byte_counter <= byte_counter + 1;
                byte_index <= 0; //clear byte index
                case(byte_counter)
                    0: tx_byte <= CAMERA_ADDR;
                    1: tx_byte <= latched_address;
                    2: tx_byte <= latched_data;
                    default: tx_byte <= latched_data;
                endcase
            end
            
            FSM_TX_BYTE_1: begin //bring SIOC low and and delay for next state 
                FSM_state <= FSM_TIMER;
                FSM_return_state <= FSM_TX_BYTE_2;
                timer <= (CLK_FREQ/(4*SCCB_FREQ));
                SIOC_oe <= 1; 
            end
            
            FSM_TX_BYTE_2: begin //assign output data, 
                FSM_state <= FSM_TIMER;
                FSM_return_state <= FSM_TX_BYTE_3;
                timer <= (CLK_FREQ/(4*SCCB_FREQ)); //delay for SIOD to stabilize
                SIOD_oe <= (byte_index == 8) ? 0 : ~tx_byte[7]; //allow for 9 cycle ack, output enable signal is inverting
            end
            
            FSM_TX_BYTE_3: begin // bring SIOC high 
                FSM_state <= FSM_TIMER;
                FSM_return_state <= FSM_TX_BYTE_4;
                timer <= (CLK_FREQ/(2*SCCB_FREQ));
                SIOC_oe <= 0; //output enable is an inverting pulldown
            end
            
            FSM_TX_BYTE_4: begin //check for end of byte, incriment counter
                FSM_state <= (byte_index == 8) ? FSM_LOAD_BYTE : FSM_TX_BYTE_1;
                tx_byte <= tx_byte<<1; //shift in next data bit
                byte_index <= byte_index + 1;
            end
            
            FSM_END_SIGNAL_1: begin //state is entered with SIOC high, SIOD high. Start by bringing SIOC low
                FSM_state <= FSM_TIMER;
                FSM_return_state <= FSM_END_SIGNAL_2;
                timer <= (CLK_FREQ/(4*SCCB_FREQ));
                SIOC_oe <= 1;
            end
            
            FSM_END_SIGNAL_2: begin // while SIOC is low, bring SIOD low 
                FSM_state <= FSM_TIMER;
                FSM_return_state <= FSM_END_SIGNAL_3;
                timer <= (CLK_FREQ/(4*SCCB_FREQ));
                SIOD_oe <= 1;
            end
            
            FSM_END_SIGNAL_3: begin // bring SIOC high
                FSM_state <= FSM_TIMER;
                FSM_return_state <= FSM_END_SIGNAL_4;
                timer <= (CLK_FREQ/(4*SCCB_FREQ));
                SIOC_oe <= 0;
            end
            
            FSM_END_SIGNAL_4: begin // bring SIOD high when SIOC is high
                FSM_state <= FSM_TIMER;
                FSM_return_state <= FSM_DONE;
                timer <= (CLK_FREQ/(4*SCCB_FREQ));
                SIOD_oe <= 0;
            end
            
            FSM_DONE: begin //add delay between transactions
                FSM_state <= FSM_TIMER;
                FSM_return_state <= FSM_IDLE;
                timer <= (2*CLK_FREQ/(SCCB_FREQ));
                byte_counter <= 0;
            end
            
            FSM_TIMER: begin //count down and jump to next state
                FSM_state <= (timer == 0) ? FSM_return_state : FSM_TIMER;
                timer <= (timer==0) ? 0 : timer - 1;
            end
        endcase
    end
    
    
endmodule