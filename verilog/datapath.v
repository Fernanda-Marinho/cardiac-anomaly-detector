// Datapath - 32 bits, sem pipeline
// Pipeline Pan-Tompkins COMPLETO (fiel ao Python):
//   sample → filtro passa-banda → derivada → QUADRADO → integrador (MWI)
//          → detector de picos
// Cada módulo só opera quando seu sinal _en estiver ativo (FSM controla).
// Registrador de entrada captura spi_data_in no ciclo READ_DATA (read_en).
module datapath (
    input  wire        clk,
    input  wire        rst,
    input  wire [31:0] spi_data_in,

    // Sinais de controle da FSM
    input  wire        read_en,          // captura spi_data_in no estado READ_DATA
    input  wire        filter_en,
    input  wire        derivator_en,
    input  wire        squarer_en,       // NOVO: etapa de quadrado
    input  wire        integrator_en,
    input  wire        peak_detector_en,
    input  wire [2:0]  ula_op_code_in,
    input  wire        ram_we,
    input  wire [7:0]  ram_addr_in,

    // Saídas para a FSM
    output wire        peak_detected_out,
    output wire [31:0] peak_value_out,
    output wire [31:0] ram_data_out,
    output wire [31:0] datapath_output
);

    // Registrador de entrada: trava o dado quando read_en=1
    reg [31:0] sample_reg;
    always @(posedge clk or posedge rst) begin
        if (rst) sample_reg <= 32'd0;
        else if (read_en) sample_reg <= spi_data_in;
    end

    wire [31:0] filtered_data;
    wire [31:0] derived_data;
    wire [31:0] squared_data;
    wire [31:0] integrated_data;
    wire [31:0] ula_result;

    // ── Etapa 1: passa-banda 5–15 Hz ──
    filter i_filter (
        .clk(clk), .rst(rst),
        .en(filter_en),
        .data_in(sample_reg),       // usa dado registrado, não o wire direto
        .data_out(filtered_data)
    );

    // ── Etapa 2: derivada de 5 pontos (com sinal, normalizada /8) ──
    derivator i_derivator (
        .clk(clk), .rst(rst),
        .en(derivator_en),
        .data_in(filtered_data),
        .data_out(derived_data)
    );

    // ── Etapa 3: quadrado (torna positivo e amplifica picos) ──
    squarer i_squarer (
        .clk(clk), .rst(rst),
        .en(squarer_en),
        .data_in(derived_data),
        .data_out(squared_data)
    );

    // ── Etapa 4: integrador de janela móvel (MWI 150 ms) ──
    integrator i_integrator (
        .clk(clk), .rst(rst),
        .en(integrator_en),
        .data_in(squared_data),     // agora recebe o sinal AO QUADRADO
        .data_out(integrated_data)
    );

    // ── Etapa 5: detector de picos (limiar + refratário 200 ms) ──
    peak_detector i_peak_detector (
        .clk(clk), .rst(rst),
        .en(peak_detector_en),
        .data_in(integrated_data),
        .peak_detected(peak_detected_out),
        .peak_value(peak_value_out)
    );

    ula i_ula (
        .a(integrated_data),
        .b(peak_value_out),
        .op_code(ula_op_code_in),
        .result(ula_result)
    );

    ram_module i_ram (
        .clk(clk), .rst(rst),
        .we(ram_we),
        .addr(ram_addr_in),
        .data_in(ula_result),
        .data_out(ram_data_out)
    );

    assign datapath_output = ula_result;

endmodule
