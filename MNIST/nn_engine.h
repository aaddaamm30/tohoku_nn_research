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
*	Last Edited	: Wed Jul 12 2017
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
		
		//constructor
		neural_backbone(void);
		int p_setStepSize(float i);

	protected:
			
		//setMatrixWeights used by higer level class
		//to initialize weights to either random or 
		//read in from the file_io class
		int p_setMatrixWeights(Eigen::MatrixXf*,
							 Eigen::MatrixXf*, 
							 Eigen::MatrixXf*);
		int p_setInputVector(Eigen::VectorXi*);
		
		//pass functions for lots of control
		int p_l1Pass(void);
		int p_l2Pass(void);
		int p_l3Pass(void);
//		int p_l4Pass(void);
		int p_softmax(void);
		
		//getters for all vectors and weights that returns
		//all eigenvectors in an array with the structure
		//listed below
		Eigen::VectorXf** p_getFPV(void);
		Eigen::MatrixXf** p_getGradients(void);
		
		//backprop operations layer by layer
		int p_backprop(int,int);
		
		//weight updater
		int p_updateWeights();
		
		//weight getter
		Eigen::MatrixXf** p_getWeights(void);
	
		//get network analysis of number
		int p_runNetwork(void);

		//get step size 
		float p_getStep(void){return(m_step_size);}

	//private attributes of abstract class neural_backbone
	private:
	
		//important components of network
		float m_step_size;
		
		//matrix to represent weights
		Eigen::MatrixXf* m_w1 = NULL;
		Eigen::MatrixXf* m_w2 = NULL;
		Eigen::MatrixXf* m_w3 = NULL;
//		Eigen::MatrixXf* m_o_w4 = NULL;

		//overall input and output vectors
		Eigen::VectorXi* m_inputVec = new Eigen::VectorXi;
		Eigen::VectorXf* m_outVec = new Eigen::VectorXf;	
		Eigen::VectorXf* m_lblVec = new Eigen::VectorXf;
	
		//forward pass vectors with and without sigmoid applied
		Eigen::VectorXf* m_v1_w = new Eigen::VectorXf;		//1 before sigmoid
		Eigen::VectorXf* m_v1_a = new Eigen::VectorXf;		//1 after sigmoid
		Eigen::VectorXf* m_v2_w = new Eigen::VectorXf;		//2 before sigmoid
		Eigen::VectorXf* m_v2_a = new Eigen::VectorXf;		//2 after sigamoid
		Eigen::VectorXf* m_v3_w = new Eigen::VectorXf;		//3 before sigmoid
//		Eigen::VectorXf* m_v3_a = new Eigen::VectorXf;		//3 after sigmoid
//		Eigen::VectorXf* m_o_v4_w = new Eigen::VectorXf;	//raw output layer
//		Eigen::VectorXf* m_o_v4_a = new Eigen::VectorXf;	//raw output with simoid :(

		//delta error for each layer
		Eigen::MatrixXf* m_gradient_w1 = new Eigen::MatrixXf;
		Eigen::MatrixXf* m_gradient_w2 = new Eigen::MatrixXf;
		Eigen::MatrixXf* m_gradient_w3 = new Eigen::MatrixXf;
//		Eigen::MatrixXf* m_gradient_w4 = new Eigen::MatrixXf;
};

#endif
