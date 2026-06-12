module fsm (
    input wire clk,
    input wire rst,
    input wire enable,
    input wire peak_detected_in,

    output reg read_en,
    output reg filter_en,
    output reg derivator_en,
    output reg squarer_en,         // NOVO: habilita etapa de quadrado
    output reg integrator_en,
    output reg peak_detector_en,
    output reg [2:0] ula_op_code_out,
    output reg ram_we,
    output reg [7:0] ram_addr_out,
    output reg alert_out,
    output reg done_out
);

    // Estados — agora com SQUARING/WAIT_SQUARE entre derivada e integrador
    localparam IDLE           = 5'd0;
    localparam READ_DATA      = 5'd1;
    localparam FILTERING      = 5'd2;
    localparam WAIT_FILTER    = 5'd3;
    localparam DERIVATING     = 5'd4;
    localparam WAIT_DERIV     = 5'd5;
    localparam SQUARING       = 5'd6;   // NOVO
    localparam WAIT_SQUARE    = 5'd7;   // NOVO
    localparam INTEGRATING    = 5'd8;
    localparam WAIT_INTEG     = 5'd9;
    localparam PEAK_DETECTING = 5'd10;
    localparam STORE_RESULT   = 5'd11;
    localparam CHECK_ALERT    = 5'd12;
    localparam FINISHED       = 5'd13;

    reg [4:0] current_state, next_state;
    reg [7:0] ram_addr_counter;

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            current_state    <= IDLE;
            ram_addr_counter <= 8'd0;
        end else begin
            current_state <= next_state;
            if (current_state == STORE_RESULT)
                ram_addr_counter <= ram_addr_counter + 8'd1;
        end
    end

    always @(*) begin
        next_state = current_state;
        case (current_state)
            IDLE:           if (enable)  next_state = READ_DATA;
            READ_DATA:                   next_state = FILTERING;
            FILTERING:                   next_state = WAIT_FILTER;
            WAIT_FILTER:                 next_state = DERIVATING;
            DERIVATING:                  next_state = WAIT_DERIV;
            WAIT_DERIV:                  next_state = SQUARING;
            SQUARING:                    next_state = WAIT_SQUARE;
            WAIT_SQUARE:                 next_state = INTEGRATING;
            INTEGRATING:                 next_state = WAIT_INTEG;
            WAIT_INTEG:                  next_state = PEAK_DETECTING;
            PEAK_DETECTING:              next_state = STORE_RESULT;
            STORE_RESULT:                next_state = CHECK_ALERT;
            CHECK_ALERT:                 next_state = FINISHED;
            FINISHED: begin
                if (enable) next_state = READ_DATA;
                else        next_state = IDLE;
            end
            default: next_state = IDLE;
        endcase
    end

    always @(*) begin
        read_en          = 1'b0;
        filter_en        = 1'b0;
        derivator_en     = 1'b0;
        squarer_en       = 1'b0;
        integrator_en    = 1'b0;
        peak_detector_en = 1'b0;
        ula_op_code_out  = 3'b000;
        ram_we           = 1'b0;
        ram_addr_out     = ram_addr_counter;
        alert_out        = 1'b0;
        done_out         = 1'b0;

        case (current_state)
            READ_DATA:      read_en          = 1'b1;
            FILTERING:      filter_en        = 1'b1;
            DERIVATING:     derivator_en     = 1'b1;
            SQUARING:       squarer_en       = 1'b1;
            INTEGRATING:    integrator_en    = 1'b1;
            PEAK_DETECTING: peak_detector_en = 1'b1;
            STORE_RESULT: begin
                ram_we          = 1'b1;
                ula_op_code_out = 3'b101;   // PASS_A: grava o MWI na RAM
            end
            CHECK_ALERT: begin
                if (peak_detected_in) alert_out = 1'b1;
            end
            FINISHED: begin
                done_out = 1'b1;
            end
            default: ;
        endcase
    end

endmodule
