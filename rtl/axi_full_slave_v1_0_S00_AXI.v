
`timescale 1 ns / 1 ps

	module axi_full_slave_v1_0_S00_AXI #
	(
		// Users to add parameters here

		// User parameters ends
		// Do not modify the parameters beyond this line

		// Width of ID for for write address, write data, read address and read data
		parameter integer C_S_AXI_ID_WIDTH	= 1,
		// Width of S_AXI data bus
		parameter integer C_S_AXI_DATA_WIDTH	= 32,
		// Width of S_AXI address bus
		parameter integer C_S_AXI_ADDR_WIDTH	= 24,
		// Width of optional user defined signal in write address channel
		parameter integer C_S_AXI_AWUSER_WIDTH	= 0,
		// Width of optional user defined signal in read address channel
		parameter integer C_S_AXI_ARUSER_WIDTH	= 0,
		// Width of optional user defined signal in write data channel
		parameter integer C_S_AXI_WUSER_WIDTH	= 0,
		// Width of optional user defined signal in read data channel
		parameter integer C_S_AXI_RUSER_WIDTH	= 0,
		// Width of optional user defined signal in write response channel
		parameter integer C_S_AXI_BUSER_WIDTH	= 0
	)
	(
		// Users to add ports here

		// User ports ends
		// Do not modify the ports beyond this line

		// Global Clock Signal
		input wire  S_AXI_ACLK,
		// Global Reset Signal. This Signal is Active LOW
		input wire  S_AXI_ARESETN,
		// Write Address ID
		input wire [C_S_AXI_ID_WIDTH-1 : 0] S_AXI_AWID,
		// Write address
		input wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_AWADDR,
		// Burst length. The burst length gives the exact number of transfers in a burst
		input wire [7 : 0] S_AXI_AWLEN,
		// Burst size. This signal indicates the size of each transfer in the burst
		input wire [2 : 0] S_AXI_AWSIZE,
		// Burst type. The burst type and the size information, 
    // determine how the address for each transfer within the burst is calculated.
		input wire [1 : 0] S_AXI_AWBURST,
		// Lock type. Provides additional information about the
    // atomic characteristics of the transfer.
		input wire  S_AXI_AWLOCK,
		// Memory type. This signal indicates how transactions
    // are required to progress through a system.
		input wire [3 : 0] S_AXI_AWCACHE,
		// Protection type. This signal indicates the privilege
    // and security level of the transaction, and whether
    // the transaction is a data access or an instruction access.
		input wire [2 : 0] S_AXI_AWPROT,
		// Quality of Service, QoS identifier sent for each
    // write transaction.
		input wire [3 : 0] S_AXI_AWQOS,
		// Region identifier. Permits a single physical interface
    // on a slave to be used for multiple logical interfaces.
		input wire [3 : 0] S_AXI_AWREGION,
		// Optional User-defined signal in the write address channel.
		input wire [C_S_AXI_AWUSER_WIDTH-1 : 0] S_AXI_AWUSER,
		// Write address valid. This signal indicates that
    // the channel is signaling valid write address and
    // control information.
		input wire  S_AXI_AWVALID,
		// Write address ready. This signal indicates that
    // the slave is ready to accept an address and associated
    // control signals.
		output wire  S_AXI_AWREADY,
		// Write Data
		input wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_WDATA,
		// Write strobes. This signal indicates which byte
    // lanes hold valid data. There is one write strobe
    // bit for each eight bits of the write data bus.
		input wire [(C_S_AXI_DATA_WIDTH/8)-1 : 0] S_AXI_WSTRB,
		// Write last. This signal indicates the last transfer
    // in a write burst.
		input wire  S_AXI_WLAST,
		// Optional User-defined signal in the write data channel.
		input wire [C_S_AXI_WUSER_WIDTH-1 : 0] S_AXI_WUSER,
		// Write valid. This signal indicates that valid write
    // data and strobes are available.
		input wire  S_AXI_WVALID,
		// Write ready. This signal indicates that the slave
    // can accept the write data.
		output wire  S_AXI_WREADY,
		// Response ID tag. This signal is the ID tag of the
    // write response.
		output wire [C_S_AXI_ID_WIDTH-1 : 0] S_AXI_BID,
		// Write response. This signal indicates the status
    // of the write transaction.
		output wire [1 : 0] S_AXI_BRESP,
		// Optional User-defined signal in the write response channel.
		output wire [C_S_AXI_BUSER_WIDTH-1 : 0] S_AXI_BUSER,
		// Write response valid. This signal indicates that the
    // channel is signaling a valid write response.
		output wire  S_AXI_BVALID,
		// Response ready. This signal indicates that the master
    // can accept a write response.
		input wire  S_AXI_BREADY,
		// Read address ID. This signal is the identification
    // tag for the read address group of signals.
		input wire [C_S_AXI_ID_WIDTH-1 : 0] S_AXI_ARID,
		// Read address. This signal indicates the initial
    // address of a read burst transaction.
		input wire [C_S_AXI_ADDR_WIDTH-1 : 0] S_AXI_ARADDR,
		// Burst length. The burst length gives the exact number of transfers in a burst
		input wire [7 : 0] S_AXI_ARLEN,
		// Burst size. This signal indicates the size of each transfer in the burst
		input wire [2 : 0] S_AXI_ARSIZE,
		// Burst type. The burst type and the size information, 
    // determine how the address for each transfer within the burst is calculated.
		input wire [1 : 0] S_AXI_ARBURST,
		// Lock type. Provides additional information about the
    // atomic characteristics of the transfer.
		input wire  S_AXI_ARLOCK,
		// Memory type. This signal indicates how transactions
    // are required to progress through a system.
		input wire [3 : 0] S_AXI_ARCACHE,
		// Protection type. This signal indicates the privilege
    // and security level of the transaction, and whether
    // the transaction is a data access or an instruction access.
		input wire [2 : 0] S_AXI_ARPROT,
		// Quality of Service, QoS identifier sent for each
    // read transaction.
		input wire [3 : 0] S_AXI_ARQOS,
		// Region identifier. Permits a single physical interface
    // on a slave to be used for multiple logical interfaces.
		input wire [3 : 0] S_AXI_ARREGION,
		// Optional User-defined signal in the read address channel.
		input wire [C_S_AXI_ARUSER_WIDTH-1 : 0] S_AXI_ARUSER,
		// Write address valid. This signal indicates that
    // the channel is signaling valid read address and
    // control information.
		input wire  S_AXI_ARVALID,
		// Read address ready. This signal indicates that
    // the slave is ready to accept an address and associated
    // control signals.
		output wire  S_AXI_ARREADY,
		// Read ID tag. This signal is the identification tag
    // for the read data group of signals generated by the slave.
		output wire [C_S_AXI_ID_WIDTH-1 : 0] S_AXI_RID,
		// Read Data
		output wire [C_S_AXI_DATA_WIDTH-1 : 0] S_AXI_RDATA,
		// Read response. This signal indicates the status of
    // the read transfer.
		output wire [1 : 0] S_AXI_RRESP,
		// Read last. This signal indicates the last transfer
    // in a read burst.
		output wire  S_AXI_RLAST,
		// Optional User-defined signal in the read address channel.
		output wire [C_S_AXI_RUSER_WIDTH-1 : 0] S_AXI_RUSER,
		// Read valid. This signal indicates that the channel
    // is signaling the required read data.
		output wire  S_AXI_RVALID,
		// Read ready. This signal indicates that the master can
    // accept the read data and response information.
		input wire  S_AXI_RREADY
	);

	// AXI4FULL signals
	reg [C_S_AXI_ADDR_WIDTH-1 : 0] 	axi_awaddr;
	reg  	axi_awready;
	reg  	axi_wready;
	reg [1 : 0] 	axi_bresp;
	reg [C_S_AXI_BUSER_WIDTH-1 : 0] 	axi_buser;
	reg  	axi_bvalid;
	reg [C_S_AXI_ADDR_WIDTH-1 : 0] 	axi_araddr;
	reg  	axi_arready;
	wire [C_S_AXI_DATA_WIDTH-1 : 0] 	axi_rdata;
	reg [1 : 0] 	axi_rresp;
	reg [C_S_AXI_RUSER_WIDTH-1 : 0] 	axi_ruser;
	reg  	axi_rvalid;
	// aw_wrap_en determines wrap boundary and enables wrapping
	wire aw_wrap_en;
	// ar_wrap_en determines wrap boundary and enables wrapping
	wire ar_wrap_en;
	// aw_wrap_size is the size of the write transfer, the
	// write address wraps to a lower address if upper address
	// limit is reached
	wire [31:0]  aw_wrap_size ; 
	// ar_wrap_size is the size of the read transfer, the
	// read address wraps to a lower address if upper address
	// limit is reached
	wire [31:0]  ar_wrap_size ; 
	// The axi_awv_awr_flag flag marks the presence of write address valid
	reg axi_awv_awr_flag;
	//The axi_arv_arr_flag flag marks the presence of read address valid
	reg axi_arv_arr_flag; 
	// The axi_awlen_cntr internal write address counter to keep track of beats in a burst transaction
	reg [7:0] axi_awlen_cntr;
	//The axi_arlen_cntr internal read address counter to keep track of beats in a burst transaction
	reg [7:0] axi_arlen_cntr;
	reg [1:0] axi_arburst;
	reg [1:0] axi_awburst;
	reg [7:0] axi_arlen;
	reg [7:0] axi_awlen;
	//local parameter for addressing 32 bit / 64 bit C_S_AXI_DATA_WIDTH
	//ADDR_LSB is used for addressing 32/64 bit registers/memories
	//ADDR_LSB = 2 for 32 bits (n downto 2) 
	//ADDR_LSB = 3 for 42 bits (n downto 3)

	// ADDR_LSB = log2(bytes per beat): 2 for 32b, 4 for 128b. The old
	// (DATA_WIDTH/32)+1 form was only correct for 32/64b and gave 5 (wrong) at 128b.
	localparam integer ADDR_LSB = $clog2(C_S_AXI_DATA_WIDTH/8);
	// Word-address width auto-sized to the byte address space (keeps the same
	// 16 MB span and init time at any data width: 4M words @32b, 1M words @128b).
	localparam integer OPT_MEM_ADDR_BITS = C_S_AXI_ADDR_WIDTH - ADDR_LSB - 1;
	localparam integer USER_NUM_MEM = 1;
	//----------------------------------------------
	//-- Signals for user logic memory space example
	//------------------------------------------------
	wire [OPT_MEM_ADDR_BITS:0] mem_address;
	wire [USER_NUM_MEM-1:0] mem_select;
	reg [C_S_AXI_DATA_WIDTH-1:0] mem_data_out[0 : USER_NUM_MEM-1];

	genvar i;
	genvar j;
	genvar mem_byte_index;

	// I/O Connections assignments

	reg  axi_read_last;  // registered RLAST, driven by pipelined read FSM

	assign S_AXI_AWREADY	= axi_awready;
	assign S_AXI_WREADY	= axi_wready;
	assign S_AXI_BRESP	= axi_bresp;
	assign S_AXI_BUSER	= axi_buser;
	assign S_AXI_BVALID	= axi_bvalid;
	assign S_AXI_ARREADY	= axi_arready;
	assign S_AXI_RDATA	= axi_rdata;
	assign S_AXI_RRESP	= axi_rresp;
	assign S_AXI_RLAST	= axi_read_last; // back-to-back: rlast on last beat
	assign S_AXI_RUSER	= axi_ruser;
	assign S_AXI_RVALID	= axi_rvalid;
	assign S_AXI_BID = S_AXI_AWID;
	assign S_AXI_RID = S_AXI_ARID;
	assign  aw_wrap_size = (C_S_AXI_DATA_WIDTH/8 * (axi_awlen)); 
	assign  ar_wrap_size = (C_S_AXI_DATA_WIDTH/8 * (axi_arlen)); 
	assign  aw_wrap_en = ((axi_awaddr & aw_wrap_size) == aw_wrap_size)? 1'b1: 1'b0;
	assign  ar_wrap_en = ((axi_araddr & ar_wrap_size) == ar_wrap_size)? 1'b1: 1'b0;

	// Implement axi_awready generation

	// axi_awready is asserted for one S_AXI_ACLK clock cycle when both
	// S_AXI_AWVALID and S_AXI_WVALID are asserted. axi_awready is
	// de-asserted when reset is low.

	always @( posedge S_AXI_ACLK )
	begin
	  if ( S_AXI_ARESETN == 1'b0 )
	    begin
	      axi_awready <= 1'b0;
	      axi_awv_awr_flag <= 1'b0;
	    end 
	  else
	    begin    
	//wzn      if (~axi_awready && S_AXI_AWVALID && ~axi_awv_awr_flag && ~axi_arv_arr_flag)
	if (~axi_awready && S_AXI_AWVALID && ~axi_awv_awr_flag)
	        begin
	          // slave is ready to accept an address and
	          // associated control signals
	          axi_awready <= 1'b1;
	          axi_awv_awr_flag  <= 1'b1; 
	          // used for generation of bresp() and bvalid
	        end
	      else if (S_AXI_WLAST && axi_wready)          
	      // preparing to accept next address after current write burst tx completion
	        begin
	          axi_awv_awr_flag  <= 1'b0;
	        end
	      else        
	        begin
	          axi_awready <= 1'b0;
	        end
	    end 
	end       
	// Implement axi_awaddr latching

	// This process is used to latch the address when both 
	// S_AXI_AWVALID and S_AXI_WVALID are valid. 

	always @( posedge S_AXI_ACLK )
	begin
	  if ( S_AXI_ARESETN == 1'b0 )
	    begin
	      axi_awaddr <= 0;
	      axi_awlen_cntr <= 0;
	      axi_awburst <= 0;
	      axi_awlen <= 0;
	    end 
	  else
	    begin    
	      if (~axi_awready && S_AXI_AWVALID && ~axi_awv_awr_flag)
	        begin
	          // address latching 
	          axi_awaddr <= S_AXI_AWADDR[C_S_AXI_ADDR_WIDTH - 1:0];  
	           axi_awburst <= S_AXI_AWBURST; 
	           axi_awlen <= S_AXI_AWLEN;     
	          // start address of transfer
	          axi_awlen_cntr <= 0;
	        end   
	      else if((axi_awlen_cntr <= axi_awlen) && axi_wready && S_AXI_WVALID)        
	        begin

	          axi_awlen_cntr <= axi_awlen_cntr + 1;

	          case (axi_awburst)
	            2'b00: // fixed burst
	            // The write address for all the beats in the transaction are fixed
	              begin
	                axi_awaddr <= axi_awaddr;          
	                //for awsize = 4 bytes (010)
	              end   
	            2'b01: //incremental burst
	            // The write address for all the beats in the transaction are increments by awsize
	              begin
	                axi_awaddr[C_S_AXI_ADDR_WIDTH - 1:ADDR_LSB] <= axi_awaddr[C_S_AXI_ADDR_WIDTH - 1:ADDR_LSB] + 1;
	                //awaddr aligned to 4 byte boundary
	                axi_awaddr[ADDR_LSB-1:0]  <= {ADDR_LSB{1'b0}};   
	                //for awsize = 4 bytes (010)
	              end   
	            2'b10: //Wrapping burst
	            // The write address wraps when the address reaches wrap boundary 
	              if (aw_wrap_en)
	                begin
	                  axi_awaddr <= (axi_awaddr - aw_wrap_size); 
	                end
	              else 
	                begin
	                  axi_awaddr[C_S_AXI_ADDR_WIDTH - 1:ADDR_LSB] <= axi_awaddr[C_S_AXI_ADDR_WIDTH - 1:ADDR_LSB] + 1;
	                  axi_awaddr[ADDR_LSB-1:0]  <= {ADDR_LSB{1'b0}}; 
	                end                      
	            default: //reserved (incremental burst for example)
	              begin
	                axi_awaddr <= axi_awaddr[C_S_AXI_ADDR_WIDTH - 1:ADDR_LSB] + 1;
	                //for awsize = 4 bytes (010)
	              end
	          endcase              
	        end
	    end 
	end       
	// Implement axi_wready generation

	// axi_wready is asserted for one S_AXI_ACLK clock cycle when both
	// S_AXI_AWVALID and S_AXI_WVALID are asserted. axi_wready is 
	// de-asserted when reset is low. 

	always @( posedge S_AXI_ACLK )
	begin
	  if ( S_AXI_ARESETN == 1'b0 )
	    begin
	      axi_wready <= 1'b0;
	    end 
	  else
	    begin    
	      if ( ~axi_wready && S_AXI_WVALID && axi_awv_awr_flag)
	        begin
	          // slave can accept the write data
	          axi_wready <= 1'b1;
	        end
	      //else if (~axi_awv_awr_flag)
	      else if (S_AXI_WLAST && axi_wready)
	        begin
	          axi_wready <= 1'b0;
	        end
	    end 
	end       
	// Implement write response logic generation

	// The write response and response valid signals are asserted by the slave 
	// when axi_wready, S_AXI_WVALID, axi_wready and S_AXI_WVALID are asserted.  
	// This marks the acceptance of address and indicates the status of 
	// write transaction.

	always @( posedge S_AXI_ACLK )
	begin
	  if ( S_AXI_ARESETN == 1'b0 )
	    begin
	      axi_bvalid <= 0;
	      axi_bresp <= 2'b0;
	      axi_buser <= 0;
	    end 
	  else
	    begin    
	      if (axi_awv_awr_flag && axi_wready && S_AXI_WVALID && ~axi_bvalid && S_AXI_WLAST )
	        begin
	          axi_bvalid <= 1'b1;
	          axi_bresp  <= 2'b0; 
	          // 'OKAY' response 
	        end                   
	      else
	        begin
	          if (S_AXI_BREADY && axi_bvalid) 
	          //check if bready is asserted while bvalid is high) 
	          //(there is a possibility that bready is always asserted high)   
	            begin
	              axi_bvalid <= 1'b0; 
	            end  
	        end
	    end
	 end   
	// Implement pipelined registered read (BRAM-style)
	// ------------------------------------------------------------------
	// Pipelined REGISTERED read: mem_data_out <= byte_ram[araddr] is a real
	// 1-cycle synchronous read (maps to FPGA Block RAM). The address is
	// issued one cycle AHEAD of the data, so read beats stream back-to-back
	// (1 beat/clk) instead of the original template's 2-clk-per-beat toggle.
	// (S_AXI_RREADY || ~axi_rvalid) safely stalls the 1-deep pipeline if the
	// master ever deasserts RREADY (the NPU wrapper holds it high).
	// ------------------------------------------------------------------
	always @( posedge S_AXI_ACLK )
	begin
	  if ( S_AXI_ARESETN == 1'b0 )
	    begin
	      axi_arready      <= 1'b0;
	      axi_arv_arr_flag <= 1'b0;
	      axi_araddr       <= 0;
	      axi_arlen        <= 0;
	      axi_arburst      <= 0;
	      axi_arlen_cntr   <= 0;
	      axi_rvalid       <= 1'b0;
	      axi_read_last    <= 1'b0;
	      axi_rresp        <= 2'b0;
	      axi_ruser        <= 0;
	    end
	  else
	    begin
	      if (~axi_arv_arr_flag && ~axi_rvalid && S_AXI_ARVALID && ~axi_arready)
	        begin
	          // accept the read address only when no read beat is still pending
	          // (~axi_rvalid): a new burst must not clobber an unconsumed last
	          // beat if the master backpressures on it (RREADY low). first fetch
	          // happens next cycle.
	          axi_arready      <= 1'b1;
	          axi_arv_arr_flag <= 1'b1;
	          axi_araddr       <= S_AXI_ARADDR[C_S_AXI_ADDR_WIDTH-1:0];
	          axi_arlen        <= S_AXI_ARLEN;
	          axi_arburst      <= S_AXI_ARBURST;
	          axi_arlen_cntr   <= 0;
	          axi_rvalid       <= 1'b0;
	          axi_read_last    <= 1'b0;
	          axi_rresp        <= 2'b0;
	        end
	      else
	        begin
	          axi_arready <= 1'b0;
	          if (axi_arv_arr_flag && (S_AXI_RREADY || ~axi_rvalid))
	            begin
	              // a fetch (mem_rden) is happening this cycle; its data is
	              // presented next cycle, so assert rvalid for the next clock.
	              axi_rvalid    <= 1'b1;
	              axi_read_last <= (axi_arlen_cntr == axi_arlen);
	              if (axi_arlen_cntr == axi_arlen)
	                begin
	                  axi_arv_arr_flag <= 1'b0;   // last address issued
	                end
	              else
	                begin
	                  if (axi_arburst != 2'b00)   // INCR (and wrap): advance addr
	                    begin
	                      axi_araddr[C_S_AXI_ADDR_WIDTH-1:ADDR_LSB] <=
	                          axi_araddr[C_S_AXI_ADDR_WIDTH-1:ADDR_LSB] + 1;
	                      axi_araddr[ADDR_LSB-1:0] <= {ADDR_LSB{1'b0}};
	                    end
	                  axi_arlen_cntr <= axi_arlen_cntr + 1;
	                end
	            end
	          else if (~axi_arv_arr_flag && S_AXI_RREADY && axi_rvalid)
	            begin
	              axi_rvalid    <= 1'b0;          // burst done, last beat taken
	              axi_read_last <= 1'b0;
	            end
	        end
	    end
	end
    
	// ------------------------------------------
	// -- Example code to access user logic memory region
	// ------------------------------------------

	// ------------------------------------------
	// -- 爆改：全双工真双端口 RAM 独立地址线
	// ------------------------------------------
// ------------------------------------------
	// -- 爆改：全双工真双端口 RAM 独立地址线
	// ------------------------------------------

	// 为写操作准备独立的地址线
	wire [OPT_MEM_ADDR_BITS:0] mem_wr_address;
	assign mem_wr_address = axi_awaddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB];

	// 为读操作准备独立的地址线
	wire [OPT_MEM_ADDR_BITS:0] mem_rd_address;
	assign mem_rd_address = axi_araddr[ADDR_LSB+OPT_MEM_ADDR_BITS:ADDR_LSB];
	     
	// implement Block RAM(s)
	generate 
      // 🌟 补回最外层的 i 循环，因为 mem_data_out 是一个二维数组
	  for(i=0; i<= USER_NUM_MEM-1; i=i+1)
	    begin:BRAM_GEN
          // 🌟 补回读写使能信号的定义
	      wire mem_rden;
	      wire mem_wren;
	
	      assign mem_wren = axi_wready && S_AXI_WVALID ;
	      assign mem_rden = axi_arv_arr_flag && (S_AXI_RREADY || ~axi_rvalid); 
	     
	      for(mem_byte_index=0; mem_byte_index<= (C_S_AXI_DATA_WIDTH/8-1); mem_byte_index=mem_byte_index+1)
	      begin:BYTE_BRAM_GEN
	        wire [8-1:0] data_in ;
	        wire [8-1:0] data_out;
	        reg  [8-1:0] byte_ram [0 : (1<<(OPT_MEM_ADDR_BITS+1))-1];

	        // Initialize shared memory to 0 to prevent x propagation
	        integer bram_init_i;
        // One-time weight preload: model flash/DDR-resident packed weights so the
        // CPU does not pack them byte-by-byte. 8768 128-bit words at DDR word base
        // 0x10000 (CPU addr 0x4010_0000); lane k of each word -> this byte lane.
        reg [127:0] wgt_pre [0:8767];
        integer wpi;
        // RAW byte-packed input images (camera model): 10 x 49 = 490 128-bit words
        // at DDR word base 0x14000 (CPU addr 0x4014_0000); lane k -> this byte lane.
        // HW img_expand expands them into the tile-major Conv1 input in SRAM at
        // runtime (see deepnet_deploy.c / gen_act_hex.py).
        reg [127:0] act_pre [0:489];
        integer api;
        initial begin
            for (bram_init_i = 0; bram_init_i <= (1<<(OPT_MEM_ADDR_BITS+1))-1; bram_init_i = bram_init_i + 1)
                byte_ram[bram_init_i] = 8'h0;
            $readmemh("firmware/weights_ddr.hex", wgt_pre);
            for (wpi = 0; wpi < 8768; wpi = wpi + 1)
                byte_ram[65536 + wpi] = wgt_pre[wpi][mem_byte_index*8 +: 8];
            $readmemh("firmware/act_ddr.hex", act_pre);
            for (api = 0; api < 490; api = api + 1)
                byte_ram[81920 + api] = act_pre[api][mem_byte_index*8 +: 8];
        end
	     
	        assign data_in  = S_AXI_WDATA[(mem_byte_index*8+7) -: 8];
	        // 🌟 读端口：独立使用 mem_rd_address
	        assign data_out = byte_ram[mem_rd_address];
	     
	        always @( posedge S_AXI_ACLK )
	        begin
	          if (mem_wren && S_AXI_WSTRB[mem_byte_index])
	            begin
                  // synthesis translate_off
                  `ifdef DEBUG
                  if (mem_byte_index == 0)
                      $display("SHMEM_WR: addr=%0h data=%0h strb=%0h",
                               mem_wr_address, S_AXI_WDATA, S_AXI_WSTRB);
                  `endif
                  // synthesis translate_on
                  // 🌟 写端口：独立使用 mem_wr_address
	              byte_ram[mem_wr_address] <= data_in;
	            end
	        end    
	      
	        always @( posedge S_AXI_ACLK )
		begin
		  if (mem_rden)   // 1-cycle registered read (real BRAM)
		      mem_data_out[i][(mem_byte_index*8+7) -: 8] <= data_out;
		end
	      end
        end // 结束 BRAM_GEN
	endgenerate
	//Output register or memory read data

	assign axi_rdata = axi_rvalid ? mem_data_out[0] : 32'h00000000;    

	// synthesis translate_off
	`ifdef DEBUG
	always @(posedge S_AXI_ACLK) begin
	    if (axi_rvalid && S_AXI_RREADY)
	        $display("SHMEM_RD: addr=%0h data=%0h beat=%0d/%0d rlast=%0b",
	                 axi_araddr, axi_rdata, axi_arlen_cntr, axi_arlen, axi_read_last);
	    if (axi_arready && S_AXI_ARVALID)
	        $display("SHMEM_RD_ACCEPT: araddr=%0h arlen=%0d",
	                 S_AXI_ARADDR, S_AXI_ARLEN);
	end
	`endif
	// synthesis translate_on

	// Add user logic here

	// User logic ends

	endmodule
