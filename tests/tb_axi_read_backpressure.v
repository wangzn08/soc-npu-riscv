`timescale 1ns/1ps
// ---------------------------------------------------------------------------
// Directed AXI read-channel test for the shared-memory model
// (axi_full_slave_v1_0_S00_AXI). Focus: the pipelined registered read FSM
// under RREADY backpressure, especially the "last beat + new AR" conflict.
//
// The key checker is AXI4 VALID stability on the R channel: once RVALID is
// asserted, the slave must keep it asserted (and RDATA/RLAST stable) until a
// beat is accepted (RREADY high). If RVALID ever drops without RREADY, a read
// beat was silently dropped -> protocol violation.
//
//   T1  back-to-back read, RREADY always high      (data + RLAST + count)
//   T2  read with RREADY gaps mid-burst            (no beat dropped)
//   T3  RREADY held low on the LAST beat while a    (the conflict Codex flagged)
//       new AR is injected -> last beat must hold,
//       new AR must NOT be accepted until taken
// ---------------------------------------------------------------------------
module tb_axi_read_backpressure;

  parameter  DW = 32;        // override via vsim -gDW=128
  localparam AW = 24;

  reg              clk  = 1'b0;
  reg              rstn = 1'b0;
  integer          err  = 0;
  integer          beats;
  integer          k;

  // AR / R
  reg  [AW-1:0]    araddr  = 0;
  reg  [7:0]       arlen   = 0;
  reg              arvalid = 0;
  wire             arready;
  wire [DW-1:0]    rdata;
  wire [1:0]       rresp;
  wire             rlast;
  wire             rvalid;
  reg              rready  = 0;
  // AW / W / B  (used only to preload memory with a known pattern)
  reg  [AW-1:0]    awaddr  = 0;
  reg  [7:0]       awlen   = 0;
  reg              awvalid = 0;
  wire             awready;
  reg  [DW-1:0]    wdata   = 0;
  reg  [DW/8-1:0]  wstrb   = 0;
  reg              wlast   = 0;
  reg              wvalid  = 0;
  wire             wready;
  wire             bvalid;
  reg              bready  = 0;
  wire [1:0]       bresp;

  always #5 clk = ~clk;

  axi_full_slave_v1_0_S00_AXI #(
    .C_S_AXI_ID_WIDTH(1), .C_S_AXI_DATA_WIDTH(DW), .C_S_AXI_ADDR_WIDTH(AW)
  ) dut (
    .S_AXI_ACLK(clk),     .S_AXI_ARESETN(rstn),
    .S_AXI_AWID(1'b0),    .S_AXI_AWADDR(awaddr), .S_AXI_AWLEN(awlen), .S_AXI_AWSIZE(3'd2),
    .S_AXI_AWBURST(2'b01),.S_AXI_AWLOCK(1'b0),   .S_AXI_AWCACHE(4'd0),.S_AXI_AWPROT(3'd0),
    .S_AXI_AWQOS(4'd0),   .S_AXI_AWREGION(4'd0), .S_AXI_AWVALID(awvalid), .S_AXI_AWREADY(awready),
    .S_AXI_WDATA(wdata),  .S_AXI_WSTRB(wstrb),   .S_AXI_WLAST(wlast), .S_AXI_WVALID(wvalid), .S_AXI_WREADY(wready),
    .S_AXI_BRESP(bresp),  .S_AXI_BVALID(bvalid), .S_AXI_BREADY(bready),
    .S_AXI_ARID(1'b0),    .S_AXI_ARADDR(araddr), .S_AXI_ARLEN(arlen), .S_AXI_ARSIZE(3'd2),
    .S_AXI_ARBURST(2'b01),.S_AXI_ARLOCK(1'b0),   .S_AXI_ARCACHE(4'd0),.S_AXI_ARPROT(3'd0),
    .S_AXI_ARQOS(4'd0),   .S_AXI_ARREGION(4'd0), .S_AXI_ARVALID(arvalid), .S_AXI_ARREADY(arready),
    .S_AXI_RDATA(rdata),  .S_AXI_RRESP(rresp),   .S_AXI_RLAST(rlast), .S_AXI_RVALID(rvalid), .S_AXI_RREADY(rready)
  );

  // ---- AXI4 R-channel VALID-stability checker (2-stage delayed, race-free) ----
  reg rv_q, rr_q, rv_qq, rr_qq;
  always @(posedge clk) begin
    if (rstn && rv_qq && !rr_qq && !rv_q) begin
      $display("[%0t] *** AXI-R VIOLATION: RVALID 1->0 without RREADY -> read beat DROPPED", $time);
      err = err + 1;
    end
    rv_q  <= rvalid; rr_q  <= rready;
    rv_qq <= rv_q;   rr_qq <= rr_q;
  end

  // ---- preload memory: write burst of (len+1) words = seed, seed+1, ... ----
  task preload(input [AW-1:0] addr, input [7:0] len, input [DW-1:0] seed);
    integer i;
    begin
      @(posedge clk); #1; awaddr = addr; awlen = len; awvalid = 1'b1;
      @(posedge clk); while (!awready) @(posedge clk);
      #1; awvalid = 1'b0;
      for (i = 0; i <= len; i = i + 1) begin
        #1; wdata = seed + i; wstrb = {(DW/8){1'b1}}; wlast = (i == len); wvalid = 1'b1;
        @(posedge clk); while (!wready) @(posedge clk);
      end
      #1; wvalid = 1'b0; wlast = 1'b0; bready = 1'b1;
      @(posedge clk); while (!bvalid) @(posedge clk);
      #1; bready = 1'b0;
    end
  endtask

  // ---- issue AR ----
  task send_ar(input [AW-1:0] addr, input [7:0] len);
    begin
      #1; araddr = addr; arlen = len; arvalid = 1'b1;
      @(posedge clk); while (!arready) @(posedge clk);
      #1; arvalid = 1'b0;
    end
  endtask

  // ---- T1 / T2: read burst, optional RREADY gaps; check data, RLAST, count ----
  task read_check(input [AW-1:0] addr, input [7:0] len, input [DW-1:0] seed, input gaps);
    integer i;
    begin
      send_ar(addr, len);
      beats = 0; i = 0;
      while (i <= len) begin
        #1; rready = gaps ? ~rready : 1'b1;
        @(posedge clk);
        if (rvalid && rready) begin
          if (rdata !== (seed + i))
            begin $display("[%0t] T DATA beat %0d: got %h exp %h", $time, i, rdata, seed+i); err=err+1; end
          if ((i == len) && !rlast)
            begin $display("[%0t] T RLAST missing on last beat", $time); err=err+1; end
          if ((i != len) && rlast)
            begin $display("[%0t] T RLAST early on beat %0d", $time, i); err=err+1; end
          beats = beats + 1; i = i + 1;
        end
      end
      if (beats !== (len + 1))
        begin $display("[%0t] T BEAT COUNT got %0d exp %0d", $time, beats, len+1); err=err+1; end
      #1; rready = 1'b0;
    end
  endtask

  initial begin
    // reset
    repeat (4) @(posedge clk);
    #1; rstn = 1'b1;
    repeat (2) @(posedge clk);

    preload(24'h000100, 8'd7, 32'h1100_0000);   // 8 words @0x100
    preload(24'h000200, 8'd3, 32'h2200_0000);   // 4 words @0x200

    // ---- T1: back-to-back, RREADY=1 ----
    $display("---- T1: back-to-back read ----");
    read_check(24'h000100, 8'd7, 32'h1100_0000, 1'b0);

    // ---- T2: RREADY gaps mid-burst ----
    $display("---- T2: read with RREADY gaps ----");
    read_check(24'h000100, 8'd7, 32'h1100_0000, 1'b1);

    // ---- T3: hold RREADY low on LAST beat + inject a new AR (the conflict) ----
    $display("---- T3: last-beat backpressure + new AR ----");
    send_ar(24'h000100, 8'd7);
    // accept beats 0..6 (len beats), hold the last one (beat 7)
    #1; rready = 1'b1; k = 0;
    while (k < 7) begin
      @(posedge clk);
      if (rvalid && rready) k = k + 1;
    end
    #1; rready = 1'b0;                 // stop accepting just before the last beat
    @(posedge clk);
    while (!(rvalid && rlast)) @(posedge clk);   // last beat now offered, held
    // inject a NEW AR for 0x200 while the last beat of 0x100 is still pending
    #1; araddr = 24'h000200; arlen = 8'd3; arvalid = 1'b1;
    for (k = 0; k < 5; k = k + 1) begin
      @(posedge clk);
      if (!rvalid)
        begin $display("[%0t] T3 *** last beat DROPPED while RREADY low", $time); err=err+1; end
      if (arready)
        begin $display("[%0t] T3 *** new AR ACCEPTED while last beat pending", $time); err=err+1; end
    end
    // release: take the held last beat, then let burst2 proceed
    #1; rready = 1'b1;
    @(posedge clk);
    if (!(rvalid && rlast && (rdata === (32'h1100_0000 + 7))))
      begin $display("[%0t] T3 last beat wrong: rvalid=%b rlast=%b data=%h", $time, rvalid, rlast, rdata); err=err+1; end
    // now AR(0x200) should get accepted and burst2 stream out
    @(posedge clk); while (!arready) @(posedge clk);
    #1; arvalid = 1'b0;
    k = 0;
    while (k <= 3) begin
      #1; rready = 1'b1;
      @(posedge clk);
      if (rvalid && rready) begin
        if (rdata !== (32'h2200_0000 + k))
          begin $display("[%0t] T3 burst2 beat %0d: got %h exp %h", $time, k, rdata, 32'h2200_0000+k); err=err+1; end
        k = k + 1;
      end
    end
    #1; rready = 1'b0;

    repeat (4) @(posedge clk);
    if (err == 0) $display("==== TB_AXI_READ_BACKPRESSURE: ALL PASS ====");
    else          $display("==== TB_AXI_READ_BACKPRESSURE: %0d ERROR(S) ====", err);
    $finish;
  end

  // watchdog
  initial begin
    #200000;
    $display("*** WATCHDOG TIMEOUT (hang) ***");
    $finish;
  end

endmodule
