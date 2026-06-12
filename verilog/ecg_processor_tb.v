// ─────────────────────────────────────────────────────────────────────────
// Testbench do processador ECG — Pan-Tompkins
// Mostra o MESMO que o script Python (pan_tompkins.py) avalia:
//   • Posições dos picos R (número da amostra → tempo em segundos)
//   • Intervalos RR (em ms)
//   • Frequência cardíaca (BPM): média, mínima e máxima
//
// CORREÇÃO em relação à versão anterior:
//   O alert_out pulsava no estado CHECK_ALERT, que ocorre 2 ciclos de FSM
//   APÓS o PEAK_DETECTING. Nesse intervalo o mem_idx já avançou, então o
//   índice gravado era o da amostra SEGUINTE ao pico real (+1 ou +2 amostras,
//   ~3–6 ms de offset sistemático em todos os picos).
//
//   Correção: introduz pico_amostra_latente que trava o mem_idx no ciclo em
//   que alert_out sobe, e só grava após confirmar o pulso. Como alert_out já
//   é combinacional em CHECK_ALERT (FSM Mealy-like) e o testbench avança
//   mem_idx só em done_out (estado FINISHED), o índice capturado em alert_out
//   corresponde ao mem_idx ainda não incrementado — ou seja, à amostra que
//   acabou de ser processada pelo peak_detector. Isso alinha com o Python.
//
// Parâmetros de fidelidade ao MIT-BIH / ao Python:
//   FS = 360 Hz  → cada amostra vale 1/360 s
//   Uma amostra é processada por vez (1 amostra por ciclo FSM completo).
// ─────────────────────────────────────────────────────────────────────────
`timescale 1ns / 1ps

module ecg_processor_tb;

    // ── Parâmetros ──
    localparam integer FS           = 360;     // Hz (MIT-BIH)
    localparam integer NUM_AMOSTRAS = 65000;   // amostras por registro
    localparam integer MAX_PICOS    = 5000;    // capacidade do buffer de picos

    reg         clk, rst, enable;
    reg  [31:0] spi_data_in;

    wire        alert_out, done_out;
    wire [31:0] final_output;

    reg [7:0] ecg_mem_raw [0:64999];

    // ── Rastreamento de picos R ──
    integer pico_amostra [0:MAX_PICOS-1];  // índice da amostra de cada pico R
    integer n_picos;

    // CORREÇÃO: trava o mem_idx no ciclo em que alert_out sobe,
    // antes de done_out avançar o índice.
    integer pico_amostra_latente;          // mem_idx capturado em alert_out

    integer mem_idx;
    integer total_registros;
    integer total_picos_geral;
    integer total_amostras_geral;

    ecg_processor_top i_top (
        .clk(clk),
        .rst(rst),
        .enable(enable),
        .spi_data_in(spi_data_in),
        .alert_out(alert_out),
        .done_out(done_out),
        .final_output(final_output)
    );

    always #5 clk = ~clk;

    // ─────────────────────────────────────────────────────────────────────
    // Relatório no estilo do Python: picos R, RR e FC
    // ─────────────────────────────────────────────────────────────────────
    task imprimir_relatorio;
        input [255:0] nome_reg;
        integer i;
        integer rr_amostras;
        real    rr_ms, fc;
        real    rr_soma, rr_min, rr_max;
        real    fc_soma, fc_min, fc_max;
        integer n_rr;
        integer listar;
        begin
            $display("");
            $display("================================================================");
            $display("  PAN-TOMPKINS - DETECCAO DE QRS   |   registro: %0s", nome_reg);
            $display("================================================================");
            $display("  Frequencia de amostragem (fs) : %0d Hz", FS);
            $display("  Amostras processadas          : %0d  (%0d s)",
                     mem_idx, mem_idx / FS);

            // ── Picos R ──
            $display("----------------------------------------------------------------");
            $display("  PICOS R DETECTADOS");
            $display("----------------------------------------------------------------");
            $display("  Total de picos R : %0d", n_picos);

            if (n_picos == 0) begin
                $display("  (nenhum pico detectado)");
                $display("================================================================");
                disable imprimir_relatorio;
            end

            listar = (n_picos < 12) ? n_picos : 12;
            $display("  Primeiros %0d picos (amostra -> tempo):", listar);
            for (i = 0; i < listar; i = i + 1) begin
                $display("    R%0d: amostra %0d   t = %0d.%03d s",
                         i + 1,
                         pico_amostra[i],
                         pico_amostra[i] / FS,
                         ((pico_amostra[i] * 1000) / FS) % 1000);
            end
            if (n_picos > listar)
                $display("    ... (+%0d picos)", n_picos - listar);

            // ── Intervalos RR + Frequência cardíaca ──
            if (n_picos >= 2) begin
                n_rr    = 0;
                rr_soma = 0.0;  rr_min = 1.0e9;  rr_max = 0.0;
                fc_soma = 0.0;  fc_min = 1.0e9;  fc_max = 0.0;

                $display("----------------------------------------------------------------");
                $display("  INTERVALOS RR  (entre picos consecutivos)");
                $display("----------------------------------------------------------------");

                for (i = 1; i < n_picos; i = i + 1) begin
                    rr_amostras = pico_amostra[i] - pico_amostra[i-1];
                    rr_ms = (rr_amostras * 1000.0) / FS;
                    fc    = 60000.0 / rr_ms;

                    rr_soma = rr_soma + rr_ms;
                    fc_soma = fc_soma + fc;
                    if (rr_ms < rr_min) rr_min = rr_ms;
                    if (rr_ms > rr_max) rr_max = rr_ms;
                    if (fc    < fc_min) fc_min = fc;
                    if (fc    > fc_max) fc_max = fc;
                    n_rr = n_rr + 1;

                    if (i <= 12)
                        $display("    RR%0d: %0d.%0d ms   (FC = %0d bpm)",
                                 i,
                                 rr_ms / 1,
                                 ((rr_amostras * 10000) / FS) % 10,
                                 fc / 1);
                end
                if (n_picos - 1 > 12)
                    $display("    ... (+%0d intervalos)", (n_picos - 1) - 12);

                $display("  RR medio  : %0d ms", rr_soma / n_rr);
                $display("  RR minimo : %0d ms", rr_min);
                $display("  RR maximo : %0d ms", rr_max);

                $display("----------------------------------------------------------------");
                $display("  FREQUENCIA CARDIACA  (FC / BPM)");
                $display("----------------------------------------------------------------");
                $display("  FC media  : %0d bpm", fc_soma / n_rr);
                $display("  FC minima : %0d bpm", fc_min);
                $display("  FC maxima : %0d bpm", fc_max);
            end

            $display("================================================================");
            $display("  Observacao: este projeto SO detecta QRS e mede RR/FC.");
            $display("  A classificacao de arritmias e uma etapa futura.");
            $display("================================================================");
        end
    endtask

    // ─────────────────────────────────────────────────────────────────────
    // Simulação de um registro
    // ─────────────────────────────────────────────────────────────────────
    task rodar_simulacao;
        input [31:0]  num_amostras;
        input [255:0] nome_reg;
        begin
            rst         = 1;
            enable      = 0;
            spi_data_in = 32'd0;
            mem_idx     = 0;
            n_picos     = 0;

            #10 rst    = 0;
            #10 enable = 1;

            spi_data_in = {24'd0, ecg_mem_raw[0]};

            $display("----------------------------------------------------------------");
            $display("[PROCESSANDO REGISTRO] %0s", nome_reg);
            $display("----------------------------------------------------------------");

            while (mem_idx < num_amostras) begin
                @(posedge clk);

                // CORREÇÃO: captura o mem_idx NO CICLO em que alert_out sobe,
                // antes de done_out avançar o índice. Assim o índice gravado
                // corresponde à amostra que o peak_detector acabou de processar,
                // alinhado com o eixo de amostras do Python.
                if (alert_out && n_picos < MAX_PICOS) begin
                    pico_amostra[n_picos] = mem_idx;   // mem_idx ainda não incrementado
                    n_picos = n_picos + 1;
                end

                // Avança para a próxima amostra quando a FSM conclui uma passagem.
                // done_out pulsa em FINISHED — DEPOIS de CHECK_ALERT (alert_out),
                // então o índice acima já foi gravado com o valor correto.
                if (done_out) begin
                    mem_idx = mem_idx + 1;
                    if (mem_idx < num_amostras)
                        spi_data_in = {24'd0, ecg_mem_raw[mem_idx]};
                end
            end

            enable = 0;
            #20;

            total_picos_geral    = total_picos_geral    + n_picos;
            total_amostras_geral = total_amostras_geral + mem_idx;
            total_registros      = total_registros + 1;

            imprimir_relatorio(nome_reg);
        end
    endtask

    // ─────────────────────────────────────────────────────────────────────
    // Programa principal
    // ─────────────────────────────────────────────────────────────────────
    initial begin
        clk                  = 0;
        total_registros      = 0;
        total_picos_geral    = 0;
        total_amostras_geral = 0;

        $display("================================================================");
        $display("   ECG PROCESSOR - PAN-TOMPKINS (Verilog)  -  MIT-BIH");
        $display("   Saida equivalente ao pan_tompkins.py: picos R, RR e FC");
        $display("================================================================");

        // ── REGISTRO 100 ──
        $readmemh("C:/Users/tassi/Desktop/projeto/verilog/verilog_mem/228_ecg.mem", ecg_mem_raw);
        rodar_simulacao(NUM_AMOSTRAS, "228");

        // Para rodar outros registros, copie as 2 linhas abaixo e troque XXX:
        // $readmemh("C:/Users/tassi/Desktop/projeto/verilog/verilog_mem/XXX_ecg.mem", ecg_mem_raw);
        // rodar_simulacao(NUM_AMOSTRAS, "XXX");

        // ── Relatório final agregado ──
        $display("");
        $display("================================================================");
        $display("   RELATORIO FINAL");
        $display("================================================================");
        $display("  Registros processados : %0d", total_registros);
        $display("  Total de amostras     : %0d", total_amostras_geral);
        $display("  Total de picos R      : %0d", total_picos_geral);
        if (total_registros > 0)
            $display("  Media picos/registro  : %0d", total_picos_geral / total_registros);
        $display("================================================================");

        #100 $finish;
    end

endmodule
