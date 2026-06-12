// ─────────────────────────────────────────────────────────────────────────
// Derivador Pan-Tompkins de 5 pontos — equivalente à etapa2_derivada do Python
//
// Python: kernel = [-1, -2, 0, 2, 1] / 8, via convolução.
// Para a amostra central x[n-2], a saída é:
//     y = ( -x[n] - 2x[n-1] + 2x[n-3] + x[n-4] ) / 8
//   = ( 2x[n-3] + x[n-4] - 2x[n-1] - x[n] ) / 8
//
// A normalização /8 (>>3) é mantida para casar com o Python. Diferente da
// versão anterior, NÃO se aplica valor absoluto aqui — o realce dos picos é
// feito pela etapa de QUADRADO (módulo squarer), exatamente como no pipeline
// Python (derivada → quadrado). Opera só quando en=1 (controle da FSM).
//
// O resultado é mantido COM SINAL (a etapa seguinte, o quadrado, o torna
// positivo). Sample-aligned: introduz latência de 2 amostras (centro do kernel).
// ─────────────────────────────────────────────────────────────────────────
module derivator (
    input  wire        clk,
    input  wire        rst,
    input  wire        en,
    input  wire [31:0] data_in,
    output reg  [31:0] data_out
);

    reg signed [31:0] x0, x1, x2, x3, x4;
    reg signed [34:0] deriv;   // 35 bits: margem para somas antes do shift

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            x0 <= 32'sd0; x1 <= 32'sd0; x2 <= 32'sd0;
            x3 <= 32'sd0; x4 <= 32'sd0;
            data_out <= 32'd0;
        end else if (en) begin
            // desloca a janela de 5 amostras
            x4 <= x3; x3 <= x2; x2 <= x1; x1 <= x0;
            x0 <= $signed(data_in);

            // derivada de 5 pontos: (2x0 + x1 - x3 - 2x4) / 8
            // (mesma forma e mesma normalização /8 do kernel Python)
            deriv = ( ($signed(x0) <<< 1) + $signed(x1)
                      - $signed(x3) - ($signed(x4) <<< 1) );

            // saída COM SINAL, normalizada por 8 (>>> 3 preserva o sinal)
            data_out <= deriv >>> 3;
        end
    end

endmodule
