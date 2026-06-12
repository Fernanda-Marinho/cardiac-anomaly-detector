// ─────────────────────────────────────────────────────────────────────────
// Detector de picos — etapa 5 do Pan-Tompkins
// Equivalente a etapa5_detecta_picos do Python (threshold adaptativo SPKI/NPKI).
//
// CORREÇÃO em relação à versão anterior:
//   A versão anterior usava uma média exponencial simples (avg >> 6) como
//   único limiar. O Python usa o threshold adaptativo DUPLO do Pan-Tompkins:
//     SPKI = nível estimado do SINAL  (média ponderada dos picos QRS aceitos)
//     NPKI = nível estimado do RUÍDO  (média ponderada dos picos rejeitados)
//     THR  = NPKI + 0.25 * (SPKI - NPKI)
//   Isso é mais robusto a variações de amplitude ao longo do registro.
//
// Mapeamento Python → Verilog (ponto fixo, fator 1/8 ≈ >> 3):
//   SPKI ← 0.125*amp + 0.875*SPKI  →  SPKI ← SPKI - (SPKI>>3) + (amp>>3)
//   NPKI ← 0.125*amp + 0.875*NPKI  →  NPKI ← NPKI - (NPKI>>3) + (amp>>3)
//   THR  ← NPKI + (SPKI-NPKI)/4    →  THR  ← NPKI + ((SPKI-NPKI)>>2)
//
// Fase de aprendizado (LEARNING = 720 = 2 s @ 360 Hz):
//   Nos primeiros 2 s o SPKI e NPKI são inicializados a partir da média do
//   MWI nessa janela (igual ao win_init=2*fs do Python). Nenhum pico é
//   emitido durante esse período — só o limiar converge.
//
// Período refratário: 200 ms = 72 amostras @ 360 Hz.
//   Após um pico aceito, nenhum outro é considerado por REFRACTORY ciclos.
//
// Saídas: peak_detected (pulso de 1 ciclo), peak_value (amplitude no MWI).
// Opera só quando en=1.
// ─────────────────────────────────────────────────────────────────────────
module peak_detector #(
    parameter REFRACTORY = 72,    // 200 ms @ 360 Hz
    parameter LEARNING   = 720    // 2 s @ 360 Hz (fase de aprendizado)
) (
    input  wire        clk,
    input  wire        rst,
    input  wire        en,
    input  wire [31:0] data_in,
    output reg         peak_detected,
    output reg  [31:0] peak_value
);

    // ── Threshold adaptativo SPKI / NPKI (fiel ao Python) ──
    reg [31:0] SPKI;          // nível estimado do sinal (picos QRS aceitos)
    reg [31:0] NPKI;          // nível estimado do ruído (picos rejeitados)
    reg [31:0] THR;           // limiar = NPKI + (SPKI-NPKI)/4

    // ── Detecção de máximo local ──
    reg [31:0] prev_data;     // amostra anterior do MWI
    reg        subindo;       // já cruzou o limiar nesta onda?

    // ── Refratário e aprendizado ──
    reg [7:0]  refr_cnt;      // contador do período refratário (8 bits: máx 255)
    reg [15:0] learn_cnt;     // conta amostras da fase de aprendizado

    // ── Acumulador para inicialização do SPKI (média dos primeiros 2 s) ──
    reg [47:0] learn_sum;     // soma do MWI durante o aprendizado (48 bits: margem)

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            SPKI          <= 32'd1;
            NPKI          <= 32'd0;
            THR           <= 32'd0;
            prev_data     <= 32'd0;
            subindo       <= 1'b0;
            refr_cnt      <= 8'd0;
            learn_cnt     <= 16'd0;
            learn_sum     <= 48'd0;
            peak_detected <= 1'b0;
            peak_value    <= 32'd0;
        end else if (en) begin
            peak_detected <= 1'b0;   // pulso de 1 ciclo por padrão

            // ── Fase de aprendizado: acumula MWI, inicializa SPKI e NPKI ──
            // Equivalente ao win_init = int(2.0 * fs) do Python:
            //   SPKI = mean(MWI[picos nos primeiros 2 s])
            //   NPKI = median(MWI) ≈ média dos vales → aproximado aqui como
            //          metade da média global, conservador e sem custo de hardware.
            if (learn_cnt < LEARNING) begin
                learn_sum <= learn_sum + {16'd0, data_in};
                learn_cnt <= learn_cnt + 16'd1;

                if (learn_cnt == LEARNING - 1) begin
                    // último ciclo do aprendizado: fixa SPKI e NPKI
                    // SPKI ← média do MWI nos 2 s iniciais
                    SPKI <= (learn_sum + LEARNING) / LEARNING;   // arredondamento
                    // NPKI ← metade de SPKI (estimativa conservadora do ruído de fundo)
                    NPKI <= ((learn_sum + LEARNING) / LEARNING) >> 1;
                    // THR ← NPKI + (SPKI-NPKI)/4 = NPKI + NPKI/4
                    //      = (learn_avg>>1) + (learn_avg>>3)  [≈ 0.625 * learn_avg]
                    THR  <= (((learn_sum + LEARNING) / LEARNING) >> 1)
                            + (((learn_sum + LEARNING) / LEARNING) >> 3);
                end

            // ── Período refratário: bloqueia detecção ──
            end else if (refr_cnt > 0) begin
                refr_cnt <= refr_cnt - 8'd1;

            // ── Detecção normal: procura máximo local acima do limiar ──
            end else begin

                if (data_in > THR) begin
                    subindo <= 1'b1;        // onda QRS cruzou o limiar
                end

                // Máximo local: estava subindo e agora começou a descer
                if (subindo && (prev_data > data_in) && (prev_data > THR)) begin
                    // ── Pico aceito: atualiza SPKI ──
                    // SPKI ← 0.875*SPKI + 0.125*amp  →  SPKI - (SPKI>>3) + (amp>>3)
                    SPKI <= SPKI - (SPKI >> 3) + (prev_data >> 3);

                    // Recalcula THR = NPKI + (SPKI_novo - NPKI) / 4
                    // Expansão: NPKI + SPKI/4 - NPKI/4 = 3/4*NPKI + 1/4*SPKI
                    //         = NPKI - (NPKI>>2) + ((SPKI - (SPKI>>3) + (prev_data>>3))>>2)
                    THR <= NPKI - (NPKI >> 2)
                           + ((SPKI - (SPKI >> 3) + (prev_data >> 3)) >> 2);

                    peak_detected <= 1'b1;
                    peak_value    <= prev_data;
                    refr_cnt      <= REFRACTORY[7:0];
                    subindo       <= 1'b0;

                end else if (!subindo) begin
                    // ── Candidato rejeitado (abaixo do limiar): atualiza NPKI ──
                    // Só atualiza em amostras que não cruzaram o limiar (ruído de fundo).
                    // NPKI ← 0.875*NPKI + 0.125*amp → NPKI - (NPKI>>3) + (amp>>3)
                    if (data_in <= THR) begin
                        NPKI <= NPKI - (NPKI >> 3) + (data_in >> 3);
                        // Recalcula THR = 3/4*NPKI_novo + 1/4*SPKI
                        THR <= (NPKI - (NPKI >> 3) + (data_in >> 3))
                               - ((NPKI - (NPKI >> 3) + (data_in >> 3)) >> 2)
                               + (SPKI >> 2);
                    end
                end
            end

            prev_data <= data_in;
        end
    end

endmodule
