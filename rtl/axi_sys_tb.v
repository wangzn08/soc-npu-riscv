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
        repeat (200000000) @(posedge clk);
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

    // 检测 trap 和 tests_passed
    reg [31:0] cycle_counter;
    always @(posedge clk) begin
        cycle_counter <= resetn ? cycle_counter + 1 : 0;
        if (resetn && dut.cpu.trap) begin
            repeat (10) @(posedge clk);
            $display("\nTRAP after %1d clock cycles", cycle_counter);
            if (tests_passed) begin
                $display("ALL TESTS PASSED.");
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
            $finish;
        end
    end

endmodule
