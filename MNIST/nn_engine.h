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
*	Last Edited	: Mon Jun 19 2017
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
		int p_setStepSize(double i);

	protected:
			
		//setMatrixWeights used by higer level class
		//to initialize weights to either random or 
		//read in from the file_io class
		int p_setMatrixWeights(Eigen::MatrixXd*,
							 Eigen::MatrixXd*, 
							 Eigen::MatrixXd*);
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
		Eigen::VectorXd** p_getFPV(void);
		Eigen::MatrixXd** p_getGradients(void);
		
		//backprop operations layer by layer
		int p_backprop(int,int);
		
		//weight updater
		int p_updateWeights();
		
		//weight getter
		Eigen::MatrixXd** p_getWeights(void);
	
		//get network analysis of number
		int p_runNetwork(void);

		//get step size 
		double p_getStep(void){return(m_step_size);}

	//private attributes of abstract class neural_backbone
	private:
	
		//important components of network
		double m_step_size;
		
		//matrix to represent weights
		Eigen::MatrixXd* m_w1 = NULL;
		Eigen::MatrixXd* m_w2 = NULL;
		Eigen::MatrixXd* m_w3 = NULL;
//		Eigen::MatrixXd* m_o_w4 = NULL;

		//overall input and output vectors
		Eigen::VectorXi* m_inputVec = new Eigen::VectorXi;
		Eigen::VectorXd* m_outVec = new Eigen::VectorXd;	
		Eigen::VectorXd* m_lblVec = new Eigen::VectorXd;
	
		//forward pass vectors with and without sigmoid applied
		Eigen::VectorXd* m_v1_w = new Eigen::VectorXd;		//1 before sigmoid
		Eigen::VectorXd* m_v1_a = new Eigen::VectorXd;		//1 after sigmoid
		Eigen::VectorXd* m_v2_w = new Eigen::VectorXd;		//2 before sigmoid
		Eigen::VectorXd* m_v2_a = new Eigen::VectorXd;		//2 after sigamoid
		Eigen::VectorXd* m_v3_w = new Eigen::VectorXd;		//3 before sigmoid
//		Eigen::VectorXd* m_v3_a = new Eigen::VectorXd;		//3 after sigmoid
//		Eigen::VectorXd* m_o_v4_w = new Eigen::VectorXd;	//raw output layer
//		Eigen::VectorXd* m_o_v4_a = new Eigen::VectorXd;	//raw output with simoid :(

		//delta error for each layer
		Eigen::MatrixXd* m_gradient_w1 = new Eigen::MatrixXd;
		Eigen::MatrixXd* m_gradient_w2 = new Eigen::MatrixXd;
		Eigen::MatrixXd* m_gradient_w3 = new Eigen::MatrixXd;
//		Eigen::MatrixXd* m_gradient_w4 = new Eigen::MatrixXd;
};

#endif
