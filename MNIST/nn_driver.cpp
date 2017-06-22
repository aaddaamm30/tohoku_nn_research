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
#include "nn_controller.h"

// Define section
#define USAGE \
	"MNIST neural Network --- Tohoku University Research\n" \
	"Author: Adam Loo\n\n" \
	"USAGE: run_network [-h] [-u <system>] [-train] [-test] <weights-file>\n" \
	"	-h ~ display help\n" \
	"	-u ~ unit test specific part of network\n" \
	"		<system> options\n" \
	"			r - mnist reader\n" \
	"			o - file output creation\n" \
	"			m - matrix feed forward\n" \
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
//	"NOTE: Network only accepts one operation at\n" \
//	"      a time and will fail otherwise.\n" \

//	"			g - gradient decent math\n" \//

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

	neural_controller nc;
		
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
			if(argc == 3){
				mnist_block* testBlock = new mnist_block(0);
				mnist_block* trainBlock = new mnist_block(1);
				testBlock->run_unit();
				trainBlock->run_unit();
				return(0);
			}else{
				std::cout << "ERROR: invalid command line argments (use [-h] for help)\n\n";
				return(1);
			}
		}
		
		//runs unit test on io_block class
		if((std::string)argv[2] == "o" && argc == 4){
			file_io* unit = new file_io();
			unit->run_unit((std::string)argv[3]);
			return(0);
		}else if((std::string)argv[2] == "o" && argc != 4){	
			std::cout << "ERROR: invalid command line argments (use [-h] for help)\n"
					  << "[You may be missing a file name]\n";
			return(1);
		}

		//runs unit test on feed forward
		if((std::string)argv[2] == "m" && argc == 4){
			nc.unit_fpv((std::string)argv[3]);
		}else if((std::string)argv[2] == "m" && argc == 3){
			nc.unit_fpv("");
		}
		
		return(0);
	}

	//case of train
	if((std::string)argv[1] == "-train"){
		
		//get command line arguemtns
		if(argc == 3){
			int epoc = 1, batch = 100;
			double step = .001;
			std::string fh = (std::string)argv[2];
			
			std::cout<<"\n====================================";
			std::cout<<"\n=========NEURAL NET TRAINER=========\n";
			nc.establishPath(fh);
			std::cout<<"\n====================================";
			std::cout<<"\nSet Epoch      : ";
			std::cin>>epoc;
			std::cout<<"====================================";
			std::cout<<"\nSet Batch Size : ";
			std::cin>>batch;
			std::cout<<"====================================";
			std::cout<<"\nSet Step Size  : ";
			std::cin>>step;
			std::cout<<"====================================\n";
		
			if(epoc<=0||batch<=0||step <=0){
				std::cout<<"\n\n||INVALID INPUT||\n";
				return(1);
			}

			//save values into nc
			if(nc.setEpoch(epoc)){
				std::cout<<"ERROR: failure setting Epoch value\n";
				return(1);
			}
			if(nc.setBatch(batch)){
				std::cout<<"ERROR: failure setting batch value\n";
				return(1);
			}
			if(nc.p_setStepSize(step)){
				std::cout<<"ERROR: failure setting step size\n";
				return(1);
			}
				
			//train the network (ptl)
			if(nc.train()){
				std::cout<<"ERROR: training network failure\n";
				return(1);
			}
		
		//case for invalid command line parameters
		}else{
			std::cout << "ERROR: invalid command line argments (use [-h] for help)\n";
			return(1);
		}
		return(0);
	}
	
	//case of test
	if((std::string)argv[1] == "-test"){
		
		//test for valid command line arguments
		if(argc == 3){
			std::string fh = (std::string)argv[2];
			
			//load path
			nc.establishPath(fh);

			//run network
			if(nc.test()){
				std::cout<<"ERROR: testing network failure\n";
				return(1);
			}
		}else{
			std::cout << "ERROR: invalid command line argments (use [-h] for help)\n";
			return(1);
		}		
			return(0);
	
	}
	
	//invalid input
	std::cout << "ERROR: invalid command line argments (use [-h] for help)" << std::endl << std::endl;
	return(1);
}
