/****************************************************************
*
*	File		: nn_engine.cpp
*	Description	: header file of the main facilitating operation
*				  of the neural network. These functions are used
*				  from the main function and have access to all
*				  the neural netork matrix manipulation algoriths
*				  the mnist reader functions and file write and
*				  read functions. Will run unit tests, data
*				  piping and all training/testing.
*
*	Author		: Adam Loo
*	Last Edited	: Sun June 4 2017
*
****************************************************************/

//includes
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <unistd.h>
#include <stdlib.h>
#include "weight_driver.h"
#include "read_mnist.h"
#include "nn_engine.h"

//prototypes
Eigen::VectorXd ReLU(Eigen::VectorXd);
Eigen::VectorXd ReLU_prime(Eigen::VectorXd);

////////////////////////////////////////////////////////
//constructor
//	- resizes the vectors to the correct size
////////////////////////////////////////////////////////
neural_backbone::neural_backbone(){
	
	//FPV (forward pass vectors)
	// initialized to corrects size
	m_inputVec->resize(784);
	m_v1_w->resize(500);
	m_v1_a->resize(500);
	m_v2_w->resize(1000);
	m_v2_a->resize(1000);
	m_v3_w->resize(50);
	m_v3_a->resize(50);
	m_o_v4_w->resize(10);
	m_o_v4_w->resize(10);
	m_outVec->resize(10);
	m_lblVec->resize(10);
	
	//used for holding on to the average gradient
	//decent for each array of weights
	m_gradient_w1->resize(500,784);
	m_gradient_w2->resize(1000,500);
	m_gradient_w3->resize(50,1000);
	m_gradient_w4->resize(10,50);
}

////////////////////////////////////////////////////////
//matrix set weights
////////////////////////////////////////////////////////
int neural_backbone::p_setMatrixWeights(Eigen::MatrixXd** mx1, 
									  Eigen::MatrixXd** mx2,
					 				  Eigen::MatrixXd** mx3,
					 				  Eigen::MatrixXd** mx4){
	m_w1 = *mx1;
	m_w2 = *mx2;
	m_w3 = *mx3;
	m_o_w4 = *mx4;

	return(0);
}

//////////////////////////////////////////////////////////
//setter for a single input vector
//////////////////////////////////////////////////////////
int neural_backbone::p_setInputVector(Eigen::VectorXi** in){
	m_inputVec = *in;
	return(0);
}

//////////////////////////////////////////////////////////
//Pass functions!
//	-these are functions that pass values through the
//   network simultaniously updating the values for the 
//   FPV (forward pass vectors), they will be called one
//	 after the other most likely and make use of the
//	 ReLU function defined in this .cpp file
//////////////////////////////////////////////////////////
int neural_backbone::p_l1Pass(void){
	Eigen::VectorXd n = (*m_inputVec).cast<double>();
	(*m_v1_w) = (*m_w1) * n;
	(*m_v1_a) = ReLU(*m_v1_w);
	return(0);
}
int neural_backbone::p_l2Pass(void){
	(*m_v2_w) = (*m_w2) * (*m_v1_a);
	(*m_v2_a) = ReLU(*m_v2_w);
	return(0);
}
int neural_backbone::p_l3Pass(void){
	(*m_v3_w) = (*m_w3) * (*m_v2_a);
	(*m_v3_a) = ReLU(*m_v3_w);
	return(0);
}
int neural_backbone::p_l4Pass(void){
	(*m_o_v4_w) = (*m_w3) * (*m_v2_a);
	(*m_o_v4_a) = ReLU(*m_o_v4_w);
	return(0);
}

//////////////////////////////////////////////////////////
//softmax
//	- softmax time!
//	- normalize output to percentages. 
//////////////////////////////////////////////////////////
int neural_backbone::p_softmax(void){

	int size = m_o_v4_a->sum();
	(*m_outVec) = (*m_o_v4_a) / size;
	return(0);
}

//////////////////////////////////////////////////////////
//return intermediate matrixes
//	- returns array of pointers to matrixXd that represent
//	  [0   ,1	,2	 ,3   ,4   ,5   ,6	   ] index
//	  [v1_w,v1_a,v2_w,v2_a,v3_w,v3_a,o_v4_w] represent
//////////////////////////////////////////////////////////
Eigen::VectorXd** neural_backbone::p_getFPV(void){

	Eigen::VectorXd** FPV = new Eigen::VectorXd*[7];

	FPV[0] = m_v1_w;
	FPV[1] = m_v1_a;
	FPV[2] = m_v2_w;
	FPV[3] = m_v2_a;
	FPV[4] = m_v3_w;
	FPV[5] = m_v3_a;
	FPV[6] = m_o_v4_w;

	return(FPV);
}

//////////////////////////////////////////////////////////
//return array of pointers to gradient matrixes
//	- returns array of pointers to matrixXd that represent
//	  [0   			,1			   ,2	 		  ,3   	         ] index
//	  [m_gradient_w1, m_gradient_w2, m_gradient_w3, m_gradient_w4] represent
//////////////////////////////////////////////////////////
Eigen::MatrixXd** neural_backbone::p_getGradients(void){
	
	Eigen::MatrixXd** grads = new Eigen::MatrixXd*[4];
	grads[0] = m_gradient_w1;
	grads[1] = m_gradient_w2;
	grads[2] = m_gradient_w3;
	grads[3] = m_gradient_w4;

	return(grads);
}

//////////////////////////////////////////////////////////
//Backpropagation
//	- function runs through all layers backwords
//	  calculating the error.
//	- sets value to the m_delta_* vectors 
//	- accepts argument of label
//////////////////////////////////////////////////////////
int neural_backbone::p_backprop(int lbl){

	//tmp vectors
	Eigen::VectorXd gradient;
	Eigen::VectorXd insigmoid;
	Eigen::VectorXd delta;
	
	//reset lblvector to unit vector
	for(int i = 0; i < m_lblVec->rows(); i++)
		(*m_lblVec)(i) = 0;
	(*m_lblVec)(lbl) = 1;

	//backprop through to 50 neuron layer	
	gradient.resize(10);
	insigmoid.resize(10);
	delta.resize(10);
	//derivitave or cost
	gradient = (*m_outVec) - (*m_lblVec);
	insigmoid = ReLU_prime(*m_o_v4_w);
	//delta is derivitave of error
	delta = gradient.cwiseProduct(insigmoid);
	//apply error to each weight in m_gradient_w4
	//using the input signal as k and delta from j
	for(int j = 0; j < delta.rows(); j++){
		for(int k = 0; k < m_v3_a->rows(); k++){
			(*m_gradient_w4)(j,k) = delta(j) * (*m_v3_a)(k);
		}
	}
	
	//bckprop through to 1000 neuron layer
	// resize tmp vectors
	gradient.resize(50);
	insigmoid.resize(50);
	delta.resize(50);
	//calculate error for next layer in
	gradient = (*m_o_w4).transpose() * delta;
	insigmoid = ReLU_prime(*m_v3_w);
	delta = gradient.cwiseProduct(insigmoid);
	for(int j = 0; j < delta.rows(); j++){
		for(int k = 0; k < m_v2_a->rows(); k++){
			(*m_gradient_w3)(j,k) = delta(j) * (*m_v2_a)(k);
		}
	}

	//backprop through 500 neuron layer
	gradient.resize(1000);
	insigmoid.resize(1000);
	delta.resize(1000);
	gradient = (*m_w3).transpose() * delta;
	insigmoid = ReLU_prime(*m_v2_w);
	delta = gradient.cwiseProduct(insigmoid);
	for(int j = 0; j < delta.rows(); j++){
		for(int k = 0; k < m_v1_a->rows(); k++){
			(*m_gradient_w2)(j,k) = delta(j) * (*m_v1_a)(k);
		}
	}

	//backprop through 10 neuron layer
	gradient.resize(500);
	insigmoid.resize(500);
	delta.resize(500);
	gradient = (*m_w3).transpose() * delta;
	insigmoid = ReLU_prime(*m_v1_w);
	delta = gradient.cwiseProduct(insigmoid);
	for(int j = 0; j < delta.rows(); j++){
		for(int k = 0; k < m_inputVec->rows(); k++){
			(*m_gradient_w1)(j,k) = delta(j) * (*m_inputVec)(k);
		}
	}

	return(0);
	
}

//////////////////////////////////////////////////////////
//weight updater
//	- updates the weights but adding the gradent decent 
//	  (passed in) times the step size, to the current 
//	  matrixes of weights
//////////////////////////////////////////////////////////
int neural_backbone::p_updateWeights(Eigen::MatrixXd** gradDec){

	*m_w1 = *m_w1 + ((*gradDec[0]) * m_step_size);
	*m_w2 = *m_w2 + ((*gradDec[1]) * m_step_size);
	*m_w3 = *m_w3 + ((*gradDec[2]) * m_step_size);
	*m_o_w4 = *m_o_w4 + ((*gradDec[3]) * m_step_size);
	return(0);
}

//////////////////////////////////////////////////////////
//ReLU math function
//	- recursive linear unit activation function
//    designed to be performed on any vector
//////////////////////////////////////////////////////////
Eigen::VectorXd ReLU(Eigen::VectorXd avec){

	for(int i = 0;i < avec.rows(); i++){
		if(avec(i) < 0)
			avec(i) = 0;
	}

	return(avec);
}

//////////////////////////////////////////////////////////
//ReLU_prime math function
//	- recursive linear unit inverse function 
//////////////////////////////////////////////////////////
Eigen::VectorXd ReLU_prime(Eigen::VectorXd avec){

	for(int i = 0;i < avec.rows(); i++){
		if(avec(i) < 0)
			avec(i) = 0;
		else
			avec(i) = 1;
	}

	return(avec);
}

