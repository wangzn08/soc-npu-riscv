`timescale 1 ns / 1 ps

module axi_sys_tb;
    reg clk = 1;
    reg resetn = 0;

    always #5 clk = ~clk;

    // VCD 波形 dump (ModelSim)
    initial begin
        if ($test$plusargs("dumpvars")) begin
            $dumpfile("axi_sys_tb.vcd");
            $dumpvars(0, axi_sys_tb);
        end
    end

    // 复位释放
    initial begin
        repeat (100) @(posedge clk);
        resetn <= 1;
    end

    // 超时
    initial begin
        repeat (700000000) @(posedge clk);  // raised for multi-block YOLO chains
        $display("TIMEOUT");
        $finish;
    end

    // DUT — NPU 现在由内部 MMIO 寄存器控制，无需外部端口
    wire txn_done, error;
    wire tests_passed;

    axi_sys #(
        .FIRMWARE_FILE("firmware/build/firmware7")
    ) dut (
        .clk           (clk),
        .resetn        (resetn),
        .txn_done      (txn_done),
        .error         (error),
        .tests_passed  (tests_passed)
    );

`ifdef DESC_RECORD
    // Reconstruct the 16 byte-lanes of the shared memory into one 128-bit word at
    // memory index IDX. ModelSim forbids a VARIABLE index into a generate-for
    // block, so the 16 lanes are unrolled with constant BYTE_BRAM_GEN indices.
    `define LANE(L,IDX) wd[(L)*8 +: 8] = dut.shared_mem.BRAM_GEN[0].BYTE_BRAM_GEN[L].byte_ram[IDX]
    `define RDW(IDX) begin \
        `LANE(0,IDX);  `LANE(1,IDX);  `LANE(2,IDX);  `LANE(3,IDX); \
        `LANE(4,IDX);  `LANE(5,IDX);  `LANE(6,IDX);  `LANE(7,IDX); \
        `LANE(8,IDX);  `LANE(9,IDX);  `LANE(10,IDX); `LANE(11,IDX); \
        `LANE(12,IDX); `LANE(13,IDX); `LANE(14,IDX); `LANE(15,IDX); \
    end
    // $writememh the pre-compiled descriptor image regions. Word bases: image 0,
    // qparam 131072, catalog 196608 (see yolo_desc.h).
    task dump_desc;
        integer w;
        reg [127:0] img [0:131071];
        reg [127:0] qp  [0:65535];
        reg [127:0] cat [0:511];
        reg [127:0] wd;
        begin
            for (w = 0; w < 131072; w = w + 1) begin `RDW(w)        img[w] = wd; end
            $writememh("firmware/desc_image.hex", img);
            for (w = 0; w < 65536; w = w + 1)  begin `RDW(131072+w) qp[w]  = wd; end
            $writememh("firmware/desc_qparam.hex", qp);
            for (w = 0; w < 512; w = w + 1)    begin `RDW(196608+w) cat[w] = wd; end
            $writememh("firmware/desc_catalog.hex", cat);
            $display("DESC DUMP DONE");
        end
    endtask
`endif

    // 检测 trap 和 tests_passed
    reg [31:0] cycle_counter;
    always @(posedge clk) begin
        cycle_counter <= resetn ? cycle_counter + 1 : 0;
        if (resetn && dut.cpu.trap) begin
            repeat (10) @(posedge clk);
            $display("\nTRAP after %1d clock cycles", cycle_counter);
            if (tests_passed) begin
                $display("ALL TESTS PASSED.");
`ifdef DESC_RECORD
                dump_desc;
`endif
                $finish;
            end else begin
                $display("ERROR!");
                $stop;
            end
        end
        if (tests_passed && !dut.cpu.trap) begin
            repeat (100) @(posedge clk);
            $display("\nTESTS PASSED (via MMIO) after %1d clock cycles", cycle_counter);
            $display("ALL TESTS PASSED.");
`ifdef DESC_RECORD
            dump_desc;
`endif
            $finish;
        end
    end

endmodule
