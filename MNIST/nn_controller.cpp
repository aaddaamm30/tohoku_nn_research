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
*	Last Edited	: Mon June 5 2017
*
****************************************************************/

#include <iostream>
#include <Eigen/Dense>
#include "read_mnist.h"
#include "weight_driver.h"
#include "nn_engine.h"
#include "nn_controller.h"

/////////////////////////////////////////////////////////////
//setters
//	- setters for epoc and batch size
/////////////////////////////////////////////////////////////
int neural_controller::setEpoch(int i){
	m_numEpoch = i;
	return(0);
}
int neural_controller::setBatch(int i){
	m_batchSize = i;
	return(0);
}

/////////////////////////////////////////////////////////////
//funciton that checks path
//	- checks if path has a valid name and saves it to 
//	  the class
/////////////////////////////////////////////////////////////
int neural_controller::establishPath(std::string fh){

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
	Eigen::MatrixXd** endWeights = new Eigen::MatrixXd*[4];
	
	//array of gradient matrices and temp gradients 
	Eigen::MatrixXd** gradients = new Eigen::MatrixXd*[4];
	Eigen::MatrixXd** tmpgrad = new Eigen::MatrixXd*[4];

	//matrix of image vectors and vector of lables
	Eigen::MatrixXd* tmpMx = m_training_block->getImgI();
	Eigen::VectorXi* lblVecs = m_training_block->getLblI();
	Eigen::MatrixXi imgVecs(tmpMx->rows(),tmpMx->cols());
	Eigen::VectorXi oneImg(784);
	imgVecs = tmpMx->cast<int>();


	
	//file reader object
	file_io f; 

	//initialize array of matrices
	gradients[0] = new Eigen::MatrixXd;
	gradients[1] = new Eigen::MatrixXd;
	gradients[2] = new Eigen::MatrixXd;
	gradients[3] = new Eigen::MatrixXd;
	gradients[0]->resize(500,784);
	gradients[1]->resize(1000,500);
	gradients[2]->resize(50,1000);
	gradients[3]->resize(10,50);
		
	//generate weights
	if(f.file_exists(m_fh)){
		std::cout<<"WEIGHTS: reading in from file ["<<m_fh<<"]\n";
		if(f.readWeights(&w1,&w2,&w3,&w4, m_fh)){
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
	if(p_setMatrixWeights(w1, w2, w3, w4)){
		std::cout<<"ERROR: failure to set weights in network\n";
		return(1);
	}
	
	//test for epoch number of times
	for(int a=0; a<m_numEpoch; a++){
		
		//run test batch number of times while there are still at least
		//batch number input vectors left
		while((imgVecs.cols() - mnIdx) > m_batchSize){
			for(int m=0; m<m_batchSize; m++){
				
				for(int l=0; l<784; l++){
					oneImg(l) = imgVecs(l, mnIdx);	
				}
				if(p_setInputVector(&oneImg)){
					std::cout<<"ERROR: fail to set image input vector in network\n";
					return(1);
				}
				
				//pass through network
				p_l1Pass();
				p_l2Pass();
				p_l3Pass();
				p_l4Pass();
				p_softmax();

				//backpropogate with label value
				if(p_backprop((*lblVecs)(mnIdx))){
					std::cout<<"ERROR: failure at backpropogation step\n";
					return(1);
				}
					
				//add gradients to our array of matrices
				tmpgrad = p_getGradients();
				for(int i=0; i<4; i++)
					*gradients[i] += *tmpgrad[i];
				
				//increment index pointer
				mnIdx++;
			}
			
			//average gradients and apply to weights
			for(int i=0; i<4; i++)
				*gradients[i] /= m_batchSize;
			if(p_updateWeights(gradients)){
				std::cout<<"ERROR: weight update unsuccessfull\n";
				return(1);
			}
		}
		
		//complete remainder
		int numLeft = imgVecs.cols() - mnIdx;
		
		for(int i=0; i < numLeft; i++){
			for(int l=0; l<784; l++){
				oneImg(l) = imgVecs(l, mnIdx);	
			}
			if(p_setInputVector(&oneImg)){
					std::cout<<"ERROR: fail to set image input vector in network\n";
					return(1);
			}

			//pass through network
			p_l1Pass();
			p_l2Pass();
			p_l3Pass();
			p_l4Pass();
			p_softmax();

			//backpropogate with label value
			if(p_backprop((*lblVecs)(mnIdx))){
					std::cout<<"ERROR: failure at backpropogation step\n";
					return(1);
			}

			//add gradients to our array of matrices
			tmpgrad = p_getGradients();
			for(int i=0; i<4; i++)
					*gradients[i] += *tmpgrad[i];

			//increment index pointer
			mnIdx++;
		}
	
		//apply update to network weights
		for(int i=0; i<4; i++)
			*gradients[i] /= numLeft;
		if(p_updateWeights(gradients)){
			std::cout<<"ERROR: weight update unsuccessfull\n";
			return(1);
		}
	}
	
	//write matrices to m_fh using the f file reader
	endWeights = p_getWeights();
	if(f.writeWeights(endWeights[0],endWeights[1],endWeights[2],endWeights[3],m_fh)){
		std::cout<<"ERROR: failure to write weights to ["<<m_fh<<"]\n";
		return(1);
	}
	
	std::cout<<"SUCCESS: Trained weights written to ["<<m_fh<<"]\n";
	std::cout<<"		 epochs     = "<<m_numEpoch<<"\n";
	std::cout<<"		 batch size = "<<m_batchSize<<"\n";
	
	return(0);
}

/////////////////////////////////////////////////////////////
//test function
//	- when called this funciton runs throught the testing
//	  set of the MNIST data and evaluates accuracy of network
//	  weights. 
//  - validates inputted fh as well
/////////////////////////////////////////////////////////////
int neural_controller::test(void){

	//file io object
	file_io f;

	//load up weights
	Eigen::MatrixXd* w1;
	Eigen::MatrixXd* w2;
	Eigen::MatrixXd* w3;
	Eigen::MatrixXd* w4;
	
	//get data from mnist
	Eigen::MatrixXd* tmpMx = m_testing_block->getImgI();
	Eigen::VectorXi* lblVecs = m_testing_block->getLblI();
	Eigen::MatrixXi imgVecs(tmpMx->rows(),tmpMx->cols());
	Eigen::VectorXi inImg(tmpMx->rows());

	imgVecs = (*tmpMx).cast<int>();

	//score chart
	int num_imgs = tmpMx->rows();
	int num_correct = 0;
	int net_guess;

	std::cout<<"\n======================";
	std::cout<<"\n=====RUNNING TEST=====\n";
	
	//generate weights
	if(f.file_exists(m_fh)){
		std::cout<<"WEIGHTS: reading in from file ["<<m_fh<<"]\n";
		if(f.readWeights(&w1,&w2,&w3,&w4, m_fh)){
			std::cout<<"ERROR: failure to read weights from file ["<<m_fh<<"]\n";
			return(1);
		}
	}else{
		std::cout<<"WEIGHTS: reading in from file ["<<m_fh<<"]\n";
		std::cout<<"FILE-NOT-FOUND\n\nWEIGHTS: initializing to random\n";
		if(f.randomizeWeights(&w1, &w2, &w3, &w4)){
			std::cout<<"ERROR: failure to randomize weights\n";
			return(1);
		}
	}
	
	//write weights
	if(p_setMatrixWeights(w1, w2, w3, w4)){
		std::cout<<"ERROR: failure to set weights in network\n";
		return(1);
	}

	//loop though all test imgs and labels and output accuracy
	for(int i=0;i<tmpMx->cols();i++){
		
		for(int l=0; l<num_imgs; l++){
			inImg(l) = imgVecs(l,i);	
		}
		
		if(p_setInputVector(&inImg)){
				std::cout<<"ERROR: fail to set image input vector in network\n";
				return(1);
		}
		
		//get network guess
		net_guess = p_runNetwork();
		
		//itterate if correct
		if(net_guess == (*lblVecs)(i)){
			std::cout<<"Test img ["<<i<<"] CORRECT (value: "<<(*lblVecs)(i)<<")"<<std::endl;
			num_correct++;
		}else{
			std::cout<<"Test img ["<<i<<"] incorrect (value: "<<(*lblVecs)(i)<<" || NN guess: "<<net_guess<<")"<<std::endl;
		}
	}

	std::cout<<"TEST COMPLETE\n";
	std::cout<<"Number of trials = "<<num_imgs<<std::endl;
	std::cout<<"Number correct   = "<<num_correct<<std::endl;
	std::cout<<"Accuracy         = "<<num_correct / num_imgs<<"%\n";

	return(0);
}
