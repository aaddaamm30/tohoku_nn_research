/****************************************************************
*
*	File		: nn_controller.h
*	Description	: header file to define the neural_controller
*				  class which is an inhereted class from the
*				  nerual_backbone class which provides public
*				  functions of a very high level .run_epoch(x)
*				  and unit tests. Most math abstracted from
*				  neural_backbone class
*
*	Author		: Adam Loo
*	Last Edited	: Sun June 4 2017
*
****************************************************************/
#ifndef _NEURAL_NETWORK_CONTROLLER_
#define _NEURAL_NETWORK_CONTROLLER_

#include <iostream>
#include <Eigen/Dense>
#include "read_mnist.h"
#include "weight_driver.h"
#include "nn_engine.h"

//neural_controller class inhereted from neural backbone and 
//implements high public level functions like setEpocs and 
//setbatch and run, each run iteration will run the number of 
//epocs in the batch size, between every epoch testing against
//MNIST test data and outputting error. After all epochs writing
//weights to the file name passed into command line
class neural_controller : public neural_backbone{

	public:

		//setter funcitons
		int setEpoch(int);
		int setBatch(int);
		
		//funcitons used in main
		int establishPath(std::string);
		int train(void);
		int test(void);
		int fullSend(void);
	
		//unit tests [may not get around to writing]
		int unit_fpv(void);
		int unit_backprop(void);

	private:

		//give access to mnist data blocks
		mnist_block* m_training_block = new mnist_block(1);
		mnist_block* m_testing_block = new mnist_block(0);
		
		//epoch size and batch size
		int m_numEpoch = 1;
		int m_batchSize = 100;
	
		//matrices to help remeber weights and find averages
		Eigen::MatrixXd** m_updateGradients;

		//path string for weight writing and reading
		std::string m_fh;
}

#endif
