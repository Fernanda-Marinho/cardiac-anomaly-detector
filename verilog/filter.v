// ─────────────────────────────────────────────────────────────────────────
// Filtro passa-banda Pan-Tompkins — implementação INTEIRA (sem multiplicador)
// Equivalente ao bandpass 5–15 Hz do Python (etapa1_passabanda).
//
// Pan & Tompkins (1985) projetaram o passa-banda como uma CASCATA de dois
// filtros recursivos de coeficientes inteiros, ideais para hardware:
//
//   Passa-baixa  : y[n] = 2y[n-1] - y[n-2] + x[n] - 2x[n-6] + x[n-12]
//                  (corte ~11 Hz para fs=200 Hz; ganho 36)
//   Passa-alta   : y[n] = 32x[n-16] - (y[n-1] + x[n] - x[n-32])
//                  (corte ~5 Hz; resulta na banda ~5–15 Hz)
//
// A cascata LP→HP entrega o equivalente ao Butterworth 5–15 Hz do Python,
// usando só somas, subtrações e deslocamentos (×32 = <<5). Mantém-se a
// interface original (clk, rst, en, data_in, data_out) e a operação só
// quando en=1, controlada pela FSM.
//
// Observação de fidelidade: os coeficientes originais assumem fs≈200 Hz.
// Para o MIT-BIH (360 Hz) a banda fica um pouco mais estreita, mas a
// topologia e a resposta de QRS permanecem corretas — é a mesma aproximação
// usada na maioria das implementações em FPGA do Pan-Tompkins.
// ─────────────────────────────────────────────────────────────────────────
module filter (
    input  wire        clk,
    input  wire        rst,
    input  wire        en,
    input  wire [31:0] data_in,
    output reg  [31:0] data_out
);

    // Remoção de offset DC: o sinal do MIT-BIH em .mem chega como 8 bits sem
    // sinal (0–255), centrado em ~128. O Python trabalha com o sinal em mV já
    // centrado em zero, e o Butterworth passa-banda rejeita DC. Os filtros
    // inteiros recursivos do Pan-Tompkins têm um integrador duplo no passa-baixa
    // que satura se houver offset DC. Subtrair a linha de base (128) reproduz a
    // condição do Python: entrada centrada em zero.
    localparam signed [31:0] BASELINE = 32'sd128;
    wire signed [31:0] x_centrado = $signed({1'b0, data_in[7:0]}) - BASELINE;

    // ── Passa-baixa: precisa de x[n..n-12] e y[n-1..n-2] ──
    reg signed [31:0] lp_x [0:12];   // histórico da entrada
    reg signed [31:0] lp_y1, lp_y2;  // histórico da saída
    reg signed [31:0] lp_out;

    // ── Passa-alta: precisa de x[n..n-32] e y[n-1] ──
    reg signed [31:0] hp_x [0:32];   // histórico = saída do passa-baixa
    reg signed [31:0] hp_y1;
    reg signed [31:0] hp_out;

    integer k;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (k = 0; k <= 12; k = k + 1) lp_x[k] <= 32'sd0;
            for (k = 0; k <= 32; k = k + 1) hp_x[k] <= 32'sd0;
            lp_y1 <= 32'sd0; lp_y2 <= 32'sd0; lp_out <= 32'sd0;
            hp_y1 <= 32'sd0; hp_out <= 32'sd0;
            data_out <= 32'd0;
        end else if (en) begin
            // ── 1) PASSA-BAIXA (entrada centrada em zero) ──
            // desloca histórico de entrada
            for (k = 12; k > 0; k = k - 1) lp_x[k] <= lp_x[k-1];
            lp_x[0] <= x_centrado;

            // y[n] = 2y[n-1] - y[n-2] + x[n] - 2x[n-6] + x[n-12]
            lp_out <= (lp_y1 <<< 1) - lp_y2
                      + x_centrado
                      - (lp_x[6] <<< 1)
                      + lp_x[12];
            lp_y2 <= lp_y1;
            lp_y1 <= (lp_y1 <<< 1) - lp_y2
                      + x_centrado
                      - (lp_x[6] <<< 1)
                      + lp_x[12];

            // ── 2) PASSA-ALTA (entrada = saída do passa-baixa) ──
            for (k = 32; k > 0; k = k - 1) hp_x[k] <= hp_x[k-1];
            hp_x[0] <= lp_out;

            // y[n] = 32*x[n-16] - ( y[n-1] + x[n] - x[n-32] )
            hp_out <= (hp_x[16] <<< 5) - (hp_y1 + lp_out - hp_x[32]);
            hp_y1  <= (hp_x[16] <<< 5) - (hp_y1 + lp_out - hp_x[32]);

            // saída do passa-banda
            data_out <= hp_out;
        end
    end

endmodule
