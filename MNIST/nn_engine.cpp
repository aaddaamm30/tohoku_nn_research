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
*	Last Edited	: Fri June 2 2017
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
	m_delta_500->resize(500);
	m_delta_1000->resize(1000);
	m_delta_50->resize(50);
	m_delta_10->resize(10);
	m_gradient_w1->resize(500,784);
	m_gradient_w2->resize(1000,500) ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////// HEREHERHERHEHREHRHER
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
//Backpropagation initialize
//	- function runs through all layers backwords
//	  calculating the error.
//	- sets value to the m_delta_* vectors 
//	- accepts argument of label
//////////////////////////////////////////////////////////
int neural_backbone::p_bp_initial(int lbl){

	//tmp vectors
	Eigen::VectorXd gradient;
	Eigen::VectorXd insigmoid;

	//reset lblvector to unit vector
	for(int i = 0; i < m_lblVec->rows(); i++)
		m_lblVec(i) = 0;
	m_lblVec(lbl) = 1;
	
	gradient.resize(10);
	insigmoid.resize(10);
	gradient = (*m_outVec) - m_lblVec;
	insigmoid = ReLU_prime(*m_o_v4_w);
	*m_delta_10 = gradient.cwizeProduct(insigmoid);

	gradient.resize(50);
	insigmoid.resize(50);
	gradient = ((*m_o_w4).transpose() * (*m_delta_50).matrix()).vector();
	insigmoid = ReLU_prime(*m_v3_w);
	*m_delta_50 = gradient.cwizeProduct(insigmoid);

	gradient.resize(1000);
	insigmoid.resize(1000);
	gradient = ((*m_w3).transpose() * (*m_delta_1000).matrix()).vector();
	insigmoid = ReLU_prime(*m_v2_w);
	*m_delta_1000 = gradient.cwizeProduct(insigmoid);

	gradient.resize(500);
	insigmoid.resize(500);
	gradient = ((*m_w3).transpose() * (*m_delta_500).matrix()).vector();
	insigmoid = ReLU_prime(*m_v1_w);
	*m_delta_500 = gradient.cwizeProduct(insigmoid);


	
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
	
