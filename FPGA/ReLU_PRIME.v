`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Tohoku University Neural Network Research
// Engineer: Adam Loo
//
// Create Date: 06/30/2017 06:11:30 PM
// Design Name: Optimus prime is a dime
// Module Name: ReLU_PRIME
// Project Name: JYPE FPGA implementation
// Target Devices:
// Tool Versions:
// Description: Module outputs appropreate ReLU prime f(x) value equation is
//              below.
//                          / x < 0  then 0
//              ReLU'(x) = |
//                         \ x >= 0 then 1

// Dependencies:
//
// Revision:
// Revision 0.01 - File Created
// Additional Comments: clock cycle analysis not done yet
//
//////////////////////////////////////////////////////////////////////////////////


module ReLU_PRIME(
    clk,
    prime_in,
    prime_out
    );
    input clk;
    input [15:0] prime_in;
    output [15:0] prime_out;
    reg [15:0] prime_out;

    begin always @ (posedge clk)
    if (prime_in[15]) begin
      prime_out <= 16'h0000;
    end else begin
      prime_out <= 16'h3C00;
    end
    end
endmodule
