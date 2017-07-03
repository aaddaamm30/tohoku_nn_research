`timescale 1ns / 1ps
//////////////////////////////////////////////////////////////////////////////////
// Company: Tohoku University Neural Network Research
// Engineer: Adam Loo
//
// Create Date: 06/30/2017 06:11:30 PM
// Design Name: Rectify that Linear like a Unit
// Module Name: ReLU_ACTIVATION
// Project Name: JYPE FPGA implementation
// Target Devices:
// Tool Versions:
// Description: Activation function for the network implementation. Effectively
//              passes a 16 bit float value back based on this equation
//                         / x < 0       then 0
//              ReLU(x) = |  0 <= x > 4  then x
//                        \  x >= 4      then 4
// Dependencies:
//
// Revision:
// Revision 0.01 - File Created
// Additional Comments: May attempt to introduce idea of saturation to avoid
//                      extreme values that could mess with 16-but float limits
//
//////////////////////////////////////////////////////////////////////////////////

module ReLU_ACTIVATION(
    clk,
    activate_in,
    activate_out
    );

    input clk;
    input [15:0] activate_in;
    reg [15:0] activate_out;

    begin always @ (posedge clk)
    if (activate_in[15]) begin
      activate_out <= 16'h0000;
    end else begin
      activate_out <= activate_in;
    end
    end
endmodule
