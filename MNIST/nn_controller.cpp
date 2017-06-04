/****************************************************************
*
*	File		: nn_controller.cpp
*	Description	: structual support for the neural_controller
*				  class. Handles reading and writing weights
*				  through the backbone to the weight driver
*				  and vice versa. Also uses reader_mnist class
*				  to pipe in values to the backbone.
*				
*
*	Author		: Adam Loo
*	Last Edited	: Sun June 4 2017
*
****************************************************************/

#include <iostream>
#include <Eigen/Dense>
#include "read_mnist.h"
#include "weight_driver.h"
#include "nn_engine.h"

/////////////////////////////////////////////////////////////
//setters
//	- setters for epoc and batch size
/////////////////////////////////////////////////////////////
int neural_controller::setEpoch(int i){
	m_numEpoch = i;
	return(0);
int neural_controller::setBatch(int i){
	m_batchSize = i;
	return(0);
}

/////////////////////////////////////////////////////////////
//funciton that checks path
//	- checks if path has a valid name and saves it to 
//	  the class
/////////////////////////////////////////////////////////////
int neural_controller::establishPath(std::string fh)

	file_io m;
	if(m.validateFileName(fh))
		return(1);
	m_fh = fh;
	return(0);
}

/////////////////////////////////////////////////////////////
//train function
//	- when called this funciton runs through the train
// 	  mnist data epoch number of times with a batch size
//	  of batch. uses m_updateGradients array of matrixes as
//	  a running sum then divides the them by batch at the end
//	  of a batch and applies that update to each weight
//	- writes weights to m_fh at the end. 
/////////////////////////////////////////////////////////////
int neural_controller::train(void){

	//mnist index counter
	int mnIdx = 0;

	//weight matrices 
	Eigen::MatrixXd* w1;
	Eigen::MatrixXd* w2;
	Eigen::MatrixXd* w3;
	Eigen::MatrixXd* w4;
	
	//array of gradient matrices
	Eigen::MatrixXd** gradients*[4];

	//file reader object
	file_io f; 
	
	//generate weights
	if(f.file_exists(m_fh){
		std::cout<<"WEIGHTS: reading in from file ["<<m_fh<<"]\n";
		if(f.readWeights(&w1,&w2,&w3,&w4, fh)){
			std::cout<<"ERROR: failure to read weights from file ["<<m_fh<<"]\n";
			return(1);
		}
	}else{
		std::cout<<"WEIGHTS: initializing to random\n";
		if(f.randomizeWeights(&w1, &w2, &w3, &w4)){
			std::cout<<"ERROR: failure to randomize weights\n";
			return(1);
		}
	}
	
	//write weights
	if(p_setMatrixWeights(&w1, &w2, &w3, &w4)){
		std::cout<<"ERROR: failure to set weights in network\n";
		return(1);
	}
	
	//test for epoch number of times
	for(int a = 0; a < m_numEpoch; a++){
		//run test batch number of times while there are still at least
		//batch number
	}								//////////////////////////////////////////////////////////////////////////HFEHERHEHREH

