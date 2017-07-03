`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Tohoku University Neural Network Research
// Engineer: Adam Loo
//
// Create Date: 07/03/2017 01:39:50 PM
// Design Name: sigmoid activation assasination
// Module Name: SIGMOID_A1
// Project Name: JYPE FPGA implementation
// Target Devices:
// Tool Versions:
// Description: sigmoid activation function that is loose linear approximation
//
// Dependencies:
//
// Revision:
// Revision 0.01 - File Created
// Additional Comments: very loose approximationg following from floating point
//                      values
//
//////////////////////////////////////////////////////////////////////////////////


module SIGMOID_A1(
    clk,
    rst,
    NEURON_SIGNAL_IN,
    NEURON_ACTIVATED
    );
    input clk, rst;
    input [15:0] NEURON_SIGNAL_IN;
    output [15:0] NEURON_ACTIVATED;
    reg [15:0] NEURON_ACTIVATED;

    //floating pt out of range if fp[14] = 1;
    begin always @ (posedge clk)
      if (rst) begin
        NEURON_ACTIVATED <= 16'h0000;
      end else if (NEURON_SIGNAL_IN[15] & NEURON_SIGNAL_IN[14]) begin
        NEURON_ACTIVATED <= 16'b1011110000000000;
      end else if (!NEURON_SIGNAL_IN[14]) begin
        NEURON_ACTIVATED <= NEURON_SIGNAL_IN;
      end else if(!NEURON_SIGNAL_IN[15] & NEURON_SIGNAL_IN[14]) begin
        NEURON_ACTIVATED <= 16'b0011110000000000;
      end
    end
endmodule
