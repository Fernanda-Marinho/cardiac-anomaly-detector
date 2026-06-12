// Top level - 32 bits
// Conecta FSM + datapath do pipeline Pan-Tompkins completo:
//   filtro passa-banda → derivada → quadrado → integrador (MWI) → detector
module ecg_processor_top (
    input  wire        clk,
    input  wire        rst,
    input  wire        enable,
    input  wire [31:0] spi_data_in,

    output wire        alert_out,
    output wire        done_out,
    output wire [31:0] final_output
);

    wire        read_en;
    wire        filter_en, derivator_en, squarer_en, integrator_en, peak_detector_en;
    wire [2:0]  ula_op_code_out;
    wire        ram_we;
    wire [7:0]  ram_addr_out;
    wire        peak_detected_from_datapath;
    wire [31:0] peak_value_from_datapath;
    wire [31:0] ram_data_out_from_datapath;

    fsm i_fsm (
        .clk(clk), .rst(rst), .enable(enable),
        .peak_detected_in(peak_detected_from_datapath),
        .read_en(read_en),
        .filter_en(filter_en),
        .derivator_en(derivator_en),
        .squarer_en(squarer_en),
        .integrator_en(integrator_en),
        .peak_detector_en(peak_detector_en),
        .ula_op_code_out(ula_op_code_out),
        .ram_we(ram_we),
        .ram_addr_out(ram_addr_out),
        .alert_out(alert_out),
        .done_out(done_out)
    );

    datapath i_datapath (
        .clk(clk), .rst(rst),
        .spi_data_in(spi_data_in),
        .read_en(read_en),
        .filter_en(filter_en),
        .derivator_en(derivator_en),
        .squarer_en(squarer_en),
        .integrator_en(integrator_en),
        .peak_detector_en(peak_detector_en),
        .ula_op_code_in(ula_op_code_out),
        .ram_we(ram_we),
        .ram_addr_in(ram_addr_out),
        .peak_detected_out(peak_detected_from_datapath),
        .peak_value_out(peak_value_from_datapath),
        .ram_data_out(ram_data_out_from_datapath),
        .datapath_output(final_output)
    );

endmodule
