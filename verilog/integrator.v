// ─────────────────────────────────────────────────────────────────────────
// Integrador de janela móvel (MWI) — etapa 4 do Pan-Tompkins
// Equivalente a etapa4_media_movel do Python (janela de 150 ms).
//
// Python: N = 0.150 * fs amostras; média = soma(janela)/N.
// Para fs = 360 Hz → N = 54 amostras. Aqui WINDOW_SIZE é parâmetro; o
// testbench/top devem instanciar com o N correto para a fs do sinal.
//
// A média é feita por divisão real (soma / WINDOW_SIZE) para casar com o
// Python, em vez do shift fixo >>5 (que só valeria para N=32). Mantida a
// interface original e a operação somente quando en=1.
//
// Implementação por soma corrente (running sum): a cada amostra soma a nova
// e subtrai a que sai da janela — O(1) por amostra.
// ─────────────────────────────────────────────────────────────────────────
module integrator #(
    parameter WINDOW_SIZE = 54           // 150 ms @ 360 Hz (ajuste p/ outra fs)
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        en,
    input  wire [31:0] data_in,
    output reg  [31:0] data_out
);

    reg [31:0] window [0:WINDOW_SIZE-1];
    reg [63:0] window_sum;               // 64 bits: soma de WINDOW_SIZE valores
    integer ptr;
    integer i;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            window_sum <= 64'd0;
            ptr        <= 0;
            data_out   <= 32'd0;
            for (i = 0; i < WINDOW_SIZE; i = i + 1)
                window[i] <= 32'd0;
        end else if (en) begin
            // soma corrente: + nova amostra  - amostra que sai da janela
            window_sum  <= window_sum - window[ptr] + data_in;
            window[ptr] <= data_in;
            ptr         <= (ptr == WINDOW_SIZE - 1) ? 0 : ptr + 1;

            // média da janela = soma / WINDOW_SIZE  (igual ao Python)
            data_out    <= (window_sum - window[ptr] + data_in) / WINDOW_SIZE;
        end
        // en=0: mantém valor anterior
    end

endmodule
