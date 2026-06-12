// ─────────────────────────────────────────────────────────────────────────
// Squarer — etapa 3 do Pan-Tompkins (quadrado ponto a ponto)
// Equivalente a etapa3_quadrado do Python:  y[n] = x[n]^2
//
// Torna todas as amostras positivas e amplifica de forma não-linear os picos
// de maior amplitude (o QRS fica muito maior que o resto). Recebe a saída COM
// SINAL do derivador e entrega um valor sempre positivo.
//
// Nota de hardware: x é de 32 bits com sinal; o quadrado pode crescer até
// 64 bits. Mantém-se a interface de 32 bits do pipeline saturando o resultado
// no máximo de 32 bits sem sinal, o que é suficiente porque a etapa seguinte
// (integrador/limiar) trabalha em termos relativos. Opera só quando en=1.
// ─────────────────────────────────────────────────────────────────────────
module squarer (
    input  wire        clk,
    input  wire        rst,
    input  wire        en,
    input  wire [31:0] data_in,
    output reg  [31:0] data_out
);

    reg signed [31:0] x;
    reg        [63:0] sq;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            x        <= 32'sd0;
            data_out <= 32'd0;
        end else if (en) begin
            x  = $signed(data_in);
            sq = x * x;                       // 64 bits, sempre >= 0

            // satura em 32 bits (evita overflow silencioso na cascata)
            if (sq > 64'hFFFFFFFF)
                data_out <= 32'hFFFFFFFF;
            else
                data_out <= sq[31:0];
        end
    end

endmodule
