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
*	Last Edited	: Thu May 25 2017
*
****************************************************************/
#ifndef _NEURAL_NETWORK_ENGINE_
#define _NEURAL_NETWORK_ENGINE_

#include <iostream>
#include <Eigen/Dense>
#include "read_mnist.h"
#include "weight_driver.h"

//backbone has all fundamental components of neural network
//and inherited objects will use the individual functions
//provided in this class. The methods that this class will handle
//individual cases of feed forward, backprop and gradient decent
//and child classes will use these functions with test analysis,
//batch size, and softmax.
class neural_backbone{

	public:
		
	protected:
			
		int setMatrixWeights(Eigen::MatrixXd*,
							 Eigen::MatrixXd*, 
							 Eigen::MatrixXd*);
		int randomizeMatrixWeights(void);
		
		
		
	//private attributes of abstract class neural_backbone
	private:
		
		//give access to mnist data blocks
		mnist_block *training_block = new mnist_block(1);
		mnist_block *testing_block = new mnist_block(0);

		//values for network to use for stuff?
		double step_size = 0;
		double batch_size = 0;
		
		//matrix to represent weights
		Eigen::MatrixXd* w1 = NULL;
		Eigen::MatrixXd* w2 = NULL;
		Eigen::MatrixXd* w3 = NULL;
		Eigen::MatrixXd* o_w4 = NULL;

		//USED ONLY FOR TRAINING
		//forward pass vectors with and without sigmoid applied
		Eigen::VectorXd* v1_w = NULL;		//1 before sigmoid
		Eigen::VectorXd* v1_a = NULL;		//1 after sigmoid
		Eigen::VectorXd* v2_w = NULL;		//2 before sigmoid
		Eigen::VectorXd* v2_a = NULL;		//2 after sigamoid
		Eigen::VectorXd* v3_w = NULL;		//3 before sigmoid
		Eigen::VectorXd* v3_a = NULL;		//3 after sigmoid
		Eigen::VectorXd* o_v4_w = NULL;		//raw output layer
		Eigen::VectorXd* o_v4_smax = NULL;	//sofmax output
		
	
};

#endif
