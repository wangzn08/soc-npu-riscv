# ============================================================
# AXI-SYS 仿真文件列表 (soc/ 相对路径)
# ============================================================

# Testbench
rtl/axi_sys_tb.v

# System Top
rtl/axi_sys.v

# CPU RTL
rtl/picorv32.v

# AXI Components
rtl/axi_lite_ram.v
rtl/axi_lite_to_axi_full.v
rtl/axi_arbiter_2to1.v
rtl/axi_full_slave_v1_0_S00_AXI.v

# NPU Wrapper + NPU RTL
rtl/npu_axi_wrapper.v
rtl/npu_top.v
rtl/param_regfile.v
rtl/top_controller_fsm.v
rtl/im2col_line_buffer.v
rtl/wgt_reader.v
rtl/systolic_16x16.v
rtl/gp_4x4.v
rtl/pe_core.v
rtl/post_process_top.v
rtl/max_pooling_2x2.v
rtl/vector_alu.v
rtl/sram_models.v
rtl/axi_dma.v
rtl/sram_copy.v
rtl/img_expand.v
