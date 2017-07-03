`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Tohoku University Neural Network Research
// Engineer: Adam Loo
//
// Create Date: 06/29/2017 03:49:33 PM
// Design Name: Top Level Pop Bezel
// Module Name: top
// Project Name: JYPE FPGA implementation
// Target Devices:
// Tool Versions:
// Description: top level file for module testing and simulation
//
// Dependencies:
//
// Revision:
// Revision 0.01 - File Created
// Additional Comments: Editing for different testing cases for different modules
//
//////////////////////////////////////////////////////////////////////////////////


module top(
    SIGA,
    SIGP);

    output [15:0] SIGA;
    output [15:0] SIGP;

    reg clk, rst;
    wire [15:0] SIGA;
    wire [15:0] SIGP;
    wire [15:0] w_to_sigs;
    wire clk_bar;

    assign clk_bar = ~clk;

    lfsr w(.clk(clk),
           .rst(rst),
           .one_rand_weight(w_to_sigs));
    SIGMOID_P1 SP(.clk(clk_bar),
                  .rst(rst),
                  .SIGMOID_PRIME_IN(w_to_sigs),
                  .SIGMOID_PRIME_OUT(SIGP));
    SIGMOID_A1 SA(.clk(clk_bar),
                  .rst(rst),
                  .NEURON_SIGNAL_IN(w_to_sigs),
                  .NEURON_ACTIVATED(SIGA));

    initial begin
      clk <= 1'b0;
      rst <= 1'b1;
      #1 clk <= 1'b1;
      #1 clk <= ~clk;
      #1 clk <= ~clk;
      #1 rst <=1'b0;
        forever begin
          #3 clk <= ~clk;
        end
    end

endmodule
