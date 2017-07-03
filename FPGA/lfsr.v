`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Tohoku University Nerual Network Research
// Engineer: Adam Loo
//
// Create Date: 06/29/2017 03:49:33 PM
// Design Name: netty the network
// Module Name: lfsr
// Project Name: JYPE FPGA implementation
// Target Devices:
// Tool Versions:
// Description: linear feedback shift regester for psudo random number generation
//              used in weight randomizatio of matrix
// Dependencies:
//
// Revision:
// Revision 0.01 - File Created
// Additional Comments: prone to lots of editing
//
//////////////////////////////////////////////////////////////////////////////////


module lfsr(
    clk,
    rst,
    one_rand_weight
    );

    input clk, rst;
    output [15:0] one_rand_weight;
    reg [15:0] one_rand_weight;
    reg special = 1'b1;

    begin always @ (posedge clk)
      if(rst) begin
        one_rand_weight <= 16'h3E4A;
        special <= 1'b1;
      end else begin
        one_rand_weight[15] <= (one_rand_weight[8] ^ one_rand_weight[6] ^ one_rand_weight[4]);
        special <= (one_rand_weight[0] ^ one_rand_weight[2] ^ one_rand_weight[4]);
        one_rand_weight[7:0] <= one_rand_weight[8:1];
        one_rand_weight[8] <= special;
      end
    end
endmodule
