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
*	Last Edited	: Mon Jun 19 2017
*
****************************************************************/

//includes
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <unistd.h>
#include <stdlib.h>
#include <fstream>
#include <cmath>
#include <iomanip>
#include "weight_driver.h"
#include "read_mnist.h"
#include "nn_engine.h"

//prototypes
Eigen::VectorXd ReLU(Eigen::VectorXd);
Eigen::VectorXd ReLU_prime(Eigen::VectorXd);
Eigen::VectorXd sigmoid(Eigen::VectorXd);
Eigen::VectorXd sigmoid_prime(Eigen::VectorXd);
double f_exp(double);

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
	m_v3_w->resize(10);
//	m_v3_a->resize(50);
//	m_o_v4_w->resize(10);
//	m_o_v4_a->resize(10);
	m_outVec->resize(10);
	m_lblVec->resize(10);
	
	//used for holding on to the average gradient
	//decent for each array of weights
	m_gradient_w1->resize(500,784);
	m_gradient_w2->resize(1000,500);
	m_gradient_w3->resize(10,1000);
//	m_gradient_w4->resize(10,50);
}

////////////////////////////////////////////////////////
//matrix set weights
////////////////////////////////////////////////////////
int neural_backbone::p_setMatrixWeights(Eigen::MatrixXd* mx1, 
									  Eigen::MatrixXd* mx2,
					 				  Eigen::MatrixXd* mx3/*,
					 				  Eigen::MatrixXd* mx4*/){
	m_w1 = mx1;
	m_w2 = mx2;
	m_w3 = mx3;
//	m_o_w4 = mx4;

	return(0);
}

//////////////////////////////////////////////////////////
//setter for step size
//////////////////////////////////////////////////////////
int neural_backbone::p_setStepSize(double i){
	
	m_step_size = i;
	return(0);
}

//////////////////////////////////////////////////////////
//setter for a single input vector
//////////////////////////////////////////////////////////
int neural_backbone::p_setInputVector(Eigen::VectorXi* in){
	m_inputVec = in;
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
	(*m_v1_a) = sigmoid(*m_v1_w);
	return(0);
}
int neural_backbone::p_l2Pass(void){
	(*m_v2_w) = (*m_w2) * (*m_v1_a);
	(*m_v2_a) = sigmoid(*m_v2_w);
	return(0);
}
int neural_backbone::p_l3Pass(void){
	(*m_v3_w) = (*m_w3) * (*m_v2_a);
	//(*m_v3_a) = sigmoid(*m_v3_w);
	return(0);
}
/*
int neural_backbone::p_l4Pass(void){
	(*m_o_v4_w) = (*m_o_w4) * (*m_v3_a);
	return(0);
}
*/

//////////////////////////////////////////////////////////
//softmax
//	- softmax time!
//	- normalize output to percentages. 
//////////////////////////////////////////////////////////
int neural_backbone::p_softmax(void){

	double size = 0;

	for(int j=0; j<10; j++){
		size += f_exp((*m_v3_w)(j));
	}
	
	for(int j=0; j<10; j++){
		(*m_outVec)(j) = f_exp((*m_v3_w)(j)) / size;
	}
	
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
	FPV[5] = m_outVec;
//	FPV[5] = m_v3_a;
//	FPV[6] = m_o_v4_w;
//	FPV[7] = m_o_v4_a;
//	FPV[8] = m_outVec;
	return(FPV);
}

//////////////////////////////////////////////////////////
//return array of pointers to gradient matrixes
//	- returns array of pointers to matrixXd that represent
//	  [0   			,1			   ,2	 		  ,3   	         ] index
//	  [m_gradient_w1, m_gradient_w2, m_gradient_w3, m_gradient_w4] represent
//////////////////////////////////////////////////////////
Eigen::MatrixXd** neural_backbone::p_getGradients(void){
	
	Eigen::MatrixXd** grads = new Eigen::MatrixXd*[3];

//	std::ofstream f;
//	f.open("debug_gradients.txt");

	grads[0] = m_gradient_w1;
	grads[1] = m_gradient_w2;
	grads[2] = m_gradient_w3;
//	grads[3] = m_gradient_w4;

	return(grads);
}

//////////////////////////////////////////////////////////
//Backpropagation
//	- function runs through all layers backwords
//	  calculating the error.
//	- sets value to the m_delta_* vectors 
//	- accepts argument of label
//////////////////////////////////////////////////////////
int neural_backbone::p_backprop(int lbl, int batchsize){
	
	//tmp vectors
	Eigen::VectorXd gradient;
	Eigen::VectorXd insigmoid;
	Eigen::VectorXd delta;
	Eigen::MatrixXd tps;

	//file help
//	std::ofstream f;
//	f.open("debug_backpropalg.txt");
	
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
	
	insigmoid = sigmoid_prime(*m_v3_w);

	//delta is derivitave of error
	for(int i=0; i<insigmoid.rows(); i++){
		delta(i) = gradient(i) * insigmoid(i);
	}


	//apply error to each weight in m_gradient_w4
	//using the input signal as k and delta from j
	for(int j = 0; j < delta.rows(); j++){
		for(int k = 0; k < m_v3_w->rows(); k++){
			if((*m_v2_a)(k) != 0 && delta(j) !=0)
				(*m_gradient_w3)(j,k) += (delta(j) * (*m_v2_a)(k)) / batchsize;
		}
	}

/*
	//bckprop through to 1000 neuron layer
	// resize tmp vectors
	gradient.resize(50);
	insigmoid.resize(50);
	//calculate error for next layer in
	tps = m_o_w4->transpose(); 
	gradient = tps * delta;

	delta.resize(50);
	insigmoid = sigmoid_prime(*m_v3_w);
	for(int i=0; i<insigmoid.rows(); i++){
		delta(i) = gradient(i) * insigmoid(i);
	}

	for(int j = 0; j < delta.rows(); j++){
		for(int k = 0; k < m_v2_a->rows(); k++){
			if((*m_v2_a)(k) != 0 && delta(j) !=0)
				(*m_gradient_w3)(j,k) += (delta(j) * (*m_v2_a)(k)) / batchsize;
		}
	}
*/

	//backprop through 500 neuron layer
	gradient.resize(1000);
	insigmoid.resize(1000);
	tps = m_w3->transpose();
	gradient = tps * delta;
	delta.resize(1000);
	insigmoid = sigmoid_prime(*m_v2_w);
	for(int i=0; i<insigmoid.rows(); i++){
		delta(i) = gradient(i) * insigmoid(i);
	}

	for(int j = 0; j < delta.rows(); j++){
		for(int k = 0; k < m_v1_a->rows(); k++){
			if((*m_v1_a)(k) != 0 && delta(j) !=0)
				(*m_gradient_w2)(j,k) += (delta(j) * (*m_v1_a)(k)) / batchsize;
		}
	}
	
	//backprop through 10 neuron layer
	gradient.resize(500);
	insigmoid.resize(500);
	tps = m_w2->transpose();
	gradient = tps * delta;
	delta.resize(500);
	insigmoid = sigmoid_prime(*m_v1_w);
	for(int i=0; i<insigmoid.rows(); i++){
		delta(i) = gradient(i) * insigmoid(i);
	}
	
	for(int j = 0; j < delta.rows(); j++){
		for(int k = 0; k < m_inputVec->rows(); k++){
			if((*m_inputVec)(k) != 0 && delta(j) !=0)
				(*m_gradient_w1)(j,k) += (delta(j) * (*m_inputVec)(k)) / batchsize;
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
int neural_backbone::p_updateWeights(void){

//	std::ofstream f;
//	f.open("debug_update.txt");
	
	*m_w1 = *m_w1 - (*m_gradient_w1 * m_step_size);
	*m_w2 = *m_w2 - (*m_gradient_w2 * m_step_size);
	*m_w3 = *m_w3 - (*m_gradient_w3 * m_step_size);
//	*m_o_w4 = *m_o_w4 - (*m_gradient_w4 * m_step_size);

	m_gradient_w1->resize(500,784);
	m_gradient_w2->resize(1000,500);
	m_gradient_w3->resize(10,1000);
//	m_gradient_w4->resize(10,50);
	
	return(0);
}

//////////////////////////////////////////////////////////
//weight getter
//	- returns weights
//////////////////////////////////////////////////////////
Eigen::MatrixXd** neural_backbone::p_getWeights(void){
	
	Eigen::MatrixXd** bruh = new Eigen::MatrixXd*[3];
	bruh[0] = m_w1;
	bruh[1] = m_w2;
	bruh[2] = m_w3;
//	bruh[3] = m_o_w4;

	return(bruh);
}

//////////////////////////////////////////////////////////
//runs network and returns number
//	- calculates softmax and evaluates sofmax value
//////////////////////////////////////////////////////////
int neural_backbone::p_runNetwork(void){

	int output = -1;
	double biggest = 0;

	//run network
	p_l1Pass();	
	p_l2Pass();	
	p_l3Pass();	
//	p_l4Pass();	
	p_softmax();
	

	for(int i = 0; i < 10; i++){
		if(((*m_outVec)(i)) > biggest){
			biggest = (*m_outVec)(i);
			output = i;
		}
	}

	return(output);
}

//////////////////////////////////////////////////////////
//ReLU math function
//	- recursive linear unit activation function
//    designed to be performed on any vector
//////////////////////////////////////////////////////////
Eigen::VectorXd ReLU(Eigen::VectorXd avec){

	for(int i = 0; i<avec.rows(); i++){
		if(avec(i) < 0)
			avec(i) = 0;
		if(avec(i) > 1)
			avec(i) = 1;
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

//////////////////////////////////////////////////////////
//sigmoid math function
//	- sigmoid function implemented across all units in an
//	  array. 
//////////////////////////////////////////////////////////
Eigen::VectorXd sigmoid(Eigen::VectorXd in){
	
	Eigen::VectorXd sig;
	sig.resize(in.rows());
	
	for(int i=0; i<in.rows(); i++){
		sig(i) = 1/(1+f_exp(-in(i)));
	}

	return(sig);
}

//////////////////////////////////////////////////////////
//sigmoid prime math function
//	- sigmoid function implemented across all units in an
//	  array returning an eigen vector 
//////////////////////////////////////////////////////////
Eigen::VectorXd sigmoid_prime(Eigen::VectorXd in){

	Eigen::VectorXd sigp;
	sigp.resize(in.rows());
	double sig;

	for(int i=0; i<in.rows(); i++){
		sig = 1/(1+f_exp(-in(i)));
		sigp(i) = sig * (1 - sig);
	}

	return(sigp);
}

//////////////////////////////////////////////////////////
//fast e^x exp function
// - improve speed for activation function by not using
//	 exp function
//////////////////////////////////////////////////////////
double f_exp(double x){

	x = 1.0 + (x/256.0);

	x *= x;
	x *= x;
	x *= x;
	x *= x;
	x *= x;
	x *= x;
	x *= x;
	x *= x;
	x *= x;
	x *= x;

	return(x);
}
