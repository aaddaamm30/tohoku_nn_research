/****************************************************************
*
*	File		: main.cpp
*	Description	: This is the top level main program for the
*				  MNIST dataset neural network. It take flags
*				  that indicate if it is testing or training and
*				  appropreate weight file input if using specific
*				  pre trained weigths as well.
*				  
*	Author		: Adam Loo
*	Last Edited	: Fri May 26 2017
*
****************************************************************/

//includes
#include <iostream>
#include "read_mnist.h"
#include "weight_driver.h"
#include "nn_engine.h"

// Define section
#define USAGE \
	"MNIST neural Network --- Tohoku University Research\n" \
	"Author: Adam Loo\n\n" \
	"USAGE: run_network [-h] [-u <system>] [-train] [-test] <weights-file>\n" \
	"	-h ~ display help\n" \
	"	-u ~ unit test specific part of network\n" \
	"		<system> options\n" \
	"			r - mnist reader\n" \
	"			m - matrix feed forward\n" \
	"			o - file output creation\n" \
	"			g - gradient decent math\n" \
	"	-train ~ trains network from randomized\n" \
	"			 weights and makes a file with\n" \
	"			 trained weights for test input.\n" \
	"	-test ~ tests test set of MNIST data on\n" \
	"			weight values specified in the \n" \
	"			command line file.\n" \
	"\n" \
	"	<weights-file>\n" \
	"		file that either is used as weight\n" \
	"		output in the case of training or\n" \
	"		input in the case of testing the\n" \
	"		network\n\n" \
	"NOTE: Network only accepts one operation at\n" \
	"      a time and will fail otherwise.\n" \

//prototypes

/************************************************************
*
* 	Function     : main
*	Description  : The main function for the neural network that
*				   primarially addreses the command line parameters
*				   and correctly carries out the users commands
*
*	Inputs       : argc - the number of command line parameters
*				   argv - the parameters
*	Outputs      : 0 if successfull, 1 if failure
*
*************************************************************/

int main(int argc, char **argv){
	
	if(argc < 2 || argc > 4){
		std::cout << "ERROR: invalid command line argments (use [-h] for help)" << std::endl << std::endl;
		return(1);
	}
	
	//case of help
	if((std::string)argv[1] == "-h"){
		std::cout << USAGE << std::endl;
		return(0);
	}

	//case of unit test request
	if((std::string)argv[1] == "-u"){
		
		//runs unit test on a block to check correct matrix creation
		if((std::string)argv[2] == "r" && argc == 3){
			mnist_block* testBlock = new mnist_block(0);
			testBlock->run_unit();
			mnist_block* trainBlock = new mnist_block(1);
			trainBlock->run_unit();
			return(0);
		}else if((std::string)argv[2] == "r" && argc != 3){
			std::cout << "ERROR: invalid command line argments (use [-h] for help)" << std::endl << std::endl;
			return(1);
		}
			
		return(0);
	}

	//case of train
	if((std::string)argv[1] == "-train"){
		//train function here
		return(0);
	}
	
	//case of test
	if((std::string)argv[1] == "-test"){
		//test function here
		return(0);
	}
	
	//invalid input
	std::cout << "ERROR: invalid command line argments (use [-h] for help)" << std::endl << std::endl;
	return(1);
}
