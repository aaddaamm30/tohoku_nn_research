`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Tohoku University Neural Network Research
// Engineer: Adam Loo
//
// Create Date: 07/03/2017 01:39:50 PM
// Design Name: sigmoid prime shrine
// Module Name: SIGMOID_P1
// Project Name: JYPE FPGA implementation
// Target Devices:
// Tool Versions:
// Description: sigmoid prime function that has a linear increase between
//              -1 and 1 with slope of 1
//
// Dependencies:
//
// Revision:
// Revision 0.01 - File Created
// Additional Comments: loose sigmoid approximation
//
//////////////////////////////////////////////////////////////////////////////////


module SIGMOID_P1(
    clk,
    rst,
    SIGMOID_PRIME_IN,
    SIGMOID_PRIME_OUT
    );
    input clk, rst;
    input [15:0] SIGMOID_PRIME_IN;
    output [15:0] SIGMOID_PRIME_OUT;
    reg [15:0] SIGMOID_PRIME_OUT;

    //prime version of a sigmoind funciton for backpropogation
    begin always @ (posedge clk)
      if (rst) begin
        SIGMOID_PRIME_OUT <= 16'h0000;
      end else if (SIGMOID_PRIME_IN[15] & SIGMOID_PRIME_IN[14]) begin
        SIGMOID_PRIME_OUT <= 16'h0000;
      end else if (!SIGMOID_PRIME_IN[14]) begin
        SIGMOID_PRIME_OUT <= 16'b0011110000000000;
      end else if(!SIGMOID_PRIME_IN[15] & SIGMOID_PRIME_IN[14]) begin
        SIGMOID_PRIME_OUT <= 16'h0000;
      end
    end
endmodule
