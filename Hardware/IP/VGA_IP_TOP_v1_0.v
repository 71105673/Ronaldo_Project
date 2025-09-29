
`timescale 1 ns / 1 ps

	module VGA_IP_TOP_v1_0 #
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
        input  wire [1:0] sw_mode,
        output wire       ov7670_xclk,
        input  wire       ov7670_pclk,
        input  wire       ov7670_href,
        input  wire       ov7670_vsync,
        input  wire [7:0] ov7670_data,
        output wire       h_sync,
        output wire       v_sync,
        output wire [3:0] r_port,
        output wire [3:0] g_port,
        output wire [3:0] b_port,
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

   
    wire [19:0] cen_data;
    wire [ 2:0] selected_grid;
// Instantiation of Axi Bus Interface S00_AXI
	VGA_IP_TOP_v1_0_S00_AXI # ( 
		.C_S_AXI_DATA_WIDTH(C_S00_AXI_DATA_WIDTH),
		.C_S_AXI_ADDR_WIDTH(C_S00_AXI_ADDR_WIDTH)
	) VGA_IP_TOP_v1_0_S00_AXI_inst (
        .cen_data(cen_data),
        .selected_grid(selected_grid),
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
    VGA_Camera_Display U_VGA_Camera_Display (
    .clk(s00_axi_aclk),
    .reset(~s00_axi_aresetn),
    .sw_mode(sw_mode),
  // ov7670 side
    .ov7670_xclk(ov7670_xclk),
    .ov7670_pclk(ov7670_pclk),
    .ov7670_href(ov7670_href),
    .ov7670_vsync(ov7670_vsync),
    .ov7670_data(ov7670_data),
  // external port
    .h_sync(h_sync),
    .v_sync(v_sync),
    .r_port(r_port),
    .g_port(g_port),
    .b_port(b_port),
  // slave reg
    . cen_data(cen_data),
    . selected_grid(selected_grid)
);
	// User logic ends

	endmodule

