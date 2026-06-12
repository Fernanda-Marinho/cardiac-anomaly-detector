// RAM 256 posições x 32 bits
module ram_module (
    input  wire        clk,
    input  wire        rst,
    input  wire        we,
    input  wire [7:0]  addr,
    input  wire [31:0] data_in,
    output wire [31:0] data_out
);

    reg [31:0] mem [255:0];

    assign data_out = mem[addr];

    integer i;
    always @(posedge clk or posedge rst) begin
        if (rst) begin
            for (i = 0; i < 256; i = i + 1)
                mem[i] <= 32'd0;
        end else if (we) begin
            mem[addr] <= data_in;
        end
    end

endmodule
