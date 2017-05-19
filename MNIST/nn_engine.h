/****************************************************************
*
*	File		: nn_engine.h
*	Description	: header file of the main facilitating operation
*				  of the neural network. These functions are used
*				  from the main function and have access to all
*				  the neural netork matrix manipulation algoriths
*				  the mnist reader functions and file write and
*				  read functions. Will run unit tests, data
*				  piping and all training/testing.
*
*	Author		: Adam Loo
*	Last Edited	: Fri May 19 2017
*
****************************************************************/
#ifndef _NEURAL_NETWORK_ENGINE_
#define _NEURAL_NETWORK_ENGINE_

#include <iostream>
#include <Eigen/Dense>
#include "read_mnist.h"

//backbone has all fundamental components of neural network
//and inherited objects will use the individual functions
//provided in this class. The methods that this class will handle
//individual cases of feed forward, backprop and gradient decent
//and child classes will use these functions with test analysis,
//batch size, and softmax.
class neural_backbone{

	public:
	
		
	private:
		//class neural backbone
		mnist_block training_block = new mnist_block(1);
		mnist_block testing_block = new mnist_block(0);

		Eigen::MatrixXd weights;
}

#endif
