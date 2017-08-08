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
*	Last Edited	: Wed Jul 12 2017
*
****************************************************************/

#include <iostream>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
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
	Eigen::MatrixXf* w1;
	Eigen::MatrixXf* w2;
	Eigen::MatrixXf* w3;
//	Eigen::MatrixXf* w4;
	Eigen::MatrixXf** endWeights = new Eigen::MatrixXf*[3];
	
	//matrix of image vectors and vector of lables
	Eigen::MatrixXf* tmpMx = m_training_block->getImgI();
	Eigen::VectorXi* lblVecs = m_training_block->getLblI();
	Eigen::MatrixXi imgVecs(tmpMx->rows(),tmpMx->cols());
	Eigen::VectorXi oneImg(784);
	imgVecs = tmpMx->cast<int>();

	//manip varbs
	int num_imgs=tmpMx->cols();
	int batchIdx=0;
	
	//file reader object
	file_io f; 
		
	//generate weights
	if(f.file_exists(m_fh)){
		std::cout<<"WEIGHTS: reading in from file ["<<m_fh<<"]\n";
		if(f.readWeights(&w1,&w2,&w3,/*&w4,*/m_fh)){
			std::cout<<"ERROR: failure to read weights from file ["<<m_fh<<"]\n";
			return(1);
		}
	}else{
		std::cout<<"WEIGHTS: initializing to random\n";
		if(f.randomizeWeights(&w1, &w2, &w3/*, &w4*/)){
			std::cout<<"ERROR: failure to randomize weights\n";
			return(1);
		}
	}
	
	std::cout<<"WEIGHTS: writing into network\n";
	//write weights
	if(p_setMatrixWeights(w1, w2, w3/*, w4*/)){
		std::cout<<"ERROR: failure to set weights in network\n";
		return(1);
	}
	
	//test for epoch number of times
	for(int a=0; a<m_numEpoch; a++){

		std::cout<<"BEGINING EPOCH #"<<a+1<<std::endl;
		
		//reset index to beginning of training set
		mnIdx = 0;	
		
		//run test batch number of times while there are still at least
		//batch number input vectors left
		while((num_imgs - mnIdx) >= m_batchSize){
			for(int m=0; m<m_batchSize; m++){

				//enter img vector				
				for(int l=0; l<784; l++){
					oneImg(l) = imgVecs(l, mnIdx);	
				}
				if(p_setInputVector(&oneImg)){
					std::cout<<"ERROR: fail to set image input vector in network\n";
					return(1);
				}
				
//				std::cout<<"NETWORK [input] : forward pass img ["<<mnIdx+1<<"] with label <"<<(*lblVecs)(mnIdx)<<">\n";
				//pass through network
//				std::cout<<"NETWORK [output]: evaluated to ["<<p_runNetwork()<<"]\n";
				
				p_runNetwork();
				
				//backpropogate with label value
				if(p_backprop((*lblVecs)(mnIdx),m_batchSize)){
					std::cout<<"ERROR: failure at backpropogation step\n";
					return(1);
				}
					
				//increment index pointer
				mnIdx++;
			}

			batchIdx++;
			std::cout<<"NETWORK: batch ["<<batchIdx<<"] complete || updating weights\n";

			//average gradients and apply to weights
			if(p_updateWeights()){
				std::cout<<"ERROR: weight update unsuccessfull\n";
				return(1);
			}
		}
				
		//complete remainder
		int numLeft = num_imgs - mnIdx;
		if(numLeft > 0){

			std::cout<<"NETWORK: Less then ["<<m_batchSize<<"] images left running batch of size ["<<numLeft<<"]\n";
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
//				p_l4Pass();
				p_softmax();
		
				//backpropogate with label value
				if(p_backprop((*lblVecs)(mnIdx), numLeft)){
						std::cout<<"ERROR: failure at backpropogation step\n";
						return(1);
				}

				//increment index pointer
				mnIdx++;
			
			}
		}
		
		//apply update to network weights
		if(p_updateWeights()){
			std::cout<<"ERROR: weight update unsuccessfull\n";
			return(1);
		}
	}
	
	//write matrices to m_fh using the f file reader
	endWeights = p_getWeights();

//	dbg<<"endwrite\n"<<*endWeights[0]<<std::endl;

	if(f.writeWeights(endWeights[0],endWeights[1],endWeights[2],/*endWeights[3],*/m_fh)){
		std::cout<<"ERROR: failure to write weights to ["<<m_fh<<"]\n";
		return(1);
	}
	
	std::cout<<"SUCCESS: Trained weights written to ["<<m_fh<<"]\n";
	std::cout<<" epochs     = "<<m_numEpoch<<"\n";
	std::cout<<" batch size = "<<m_batchSize<<"\n";
	
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
	Eigen::MatrixXf* w1;
	Eigen::MatrixXf* w2;
	Eigen::MatrixXf* w3;
//	Eigen::MatrixXf* w4;
	
	//get data from mnist
	Eigen::MatrixXf* tmpMx = m_testing_block->getImgI();
	Eigen::VectorXi* lblVecs = m_testing_block->getLblI();
	Eigen::MatrixXi imgVecs(tmpMx->rows(),tmpMx->cols());
	Eigen::VectorXi inImg(tmpMx->rows());

	imgVecs = (*tmpMx).cast<int>();

	//score chart
	int num_imgs = tmpMx->cols();
	int size_img = tmpMx->rows();
	int num_correct = 0;
	int net_guess;

	std::cout<<"\n======================";
	std::cout<<"\n=====RUNNING TEST=====\n";
	
	//generate weights
	if(f.file_exists(m_fh)){
		std::cout<<"WEIGHTS: reading in from file ["<<m_fh<<"]\n";
		if(f.readWeights(&w1,&w2,&w3,/*&w4,*/ m_fh)){
			std::cout<<"ERROR: failure to read weights from file ["<<m_fh<<"]\n";
			return(1);
		}
	}else{
		std::cout<<"WEIGHTS: reading in from file ["<<m_fh<<"]\n";
		std::cout<<"FILE-NOT-FOUND\n\nWEIGHTS: initializing to random\n\n";
		if(f.randomizeWeights(&w1, &w2, &w3/*, &w4*/)){
			std::cout<<"ERROR: failure to randomize weights\n";
			return(1);
		}
	}
	
	//write weights
	if(p_setMatrixWeights(w1, w2, w3/*, w4*/)){
		std::cout<<"ERROR: failure to set weights in network\n";
		return(1);
	}

	//loop though all test imgs and labels and output accuracy
	for(int i=0;i<num_imgs;i++){
		
		for(int l=0; l<size_img; l++){
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
	std::cout<<"Accuracy         = "<<((float)num_correct / num_imgs)*100<<"%\n";

	return(0);
}

/////////////////////////////////////////////////////////////
//Full send
//	- same as train but runs a test set between each epoch
/////////////////////////////////////////////////////////////
int neural_controller::fullSend(void){

	//get user input
	int epoch, batch, mnIdx, net_guess, num_correct=0;
	float step;

	std::cout<<"\nFULL SEND BRUH\n";
	std::cout<<"Epoch : ";
	std::cin>>epoch;
	std::cout<<"\nBatch : ";
	std::cin>>batch;
	std::cout<<"\nstep  : ";
	std::cin>>step;

	if(epoch<=0||batch<=0||step <=0){
		std::cout<<"\n\n||INVALID INPUT||\n";
		return(1);
	}

	file_io f;
	m_numEpoch = 1;
	m_batchSize = batch;
	p_setStepSize(step);

	//load up info (will be a lot :O)
	//testing below
	Eigen::MatrixXf* tmpMx = m_testing_block->getImgI();
	Eigen::VectorXi* TEST_lbl = m_testing_block->getLblI();
	Eigen::MatrixXi TEST_img(tmpMx->rows(),tmpMx->cols());
	TEST_img = tmpMx->cast<int>();
	//training below
	tmpMx = m_training_block->getImgI();
	Eigen::VectorXi* TRAIN_lbl = m_training_block->getLblI();
	Eigen::MatrixXi TRAIN_img(tmpMx->rows(),tmpMx->cols());
	TRAIN_img = tmpMx->cast<int>();

	//init vector for one img
	Eigen::VectorXi INIT_img(tmpMx->rows());

	//randomize that $H!T
	Eigen::MatrixXf* w1;
	Eigen::MatrixXf* w2;
	Eigen::MatrixXf* w3;

	if(f.randomizeWeights(&w1, &w2, &w3/*, &w4*/)){
		std::cout<<"ERROR: failure to randomize weights\n";
		return(1);
	}
	std::cout<<"WEIGHTS: writing into network\n";
	//write weights
	if(p_setMatrixWeights(w1, w2, w3/*, w4*/)){
		std::cout<<"ERROR: failure to set weights in network\n";
		return(1);
	}
	
	
	//top level loop to run through all processes
	for(int loopnum = 0; loopnum < epoch; loopnum++){
		
		std::cout<<"\nEPOCH TRAIN #"<<loopnum+1<<std::endl;
		//reset index to beginning of training set
		mnIdx = 0;	
		
		//run test batch number of times while there are still at least
		//batch number input vectors left
		while((TRAIN_img.cols() - mnIdx) >= m_batchSize){
			for(int m=0; m<m_batchSize; m++){

				//enter img vector				
				for(int l=0; l<784; l++){
					INIT_img(l) = TRAIN_img(l, mnIdx);	
				}
				if(p_setInputVector(&INIT_img)){
					std::cout<<"ERROR: fail to set image input vector in network\n";
					return(1);
				}
				
				p_runNetwork();
				
				//backpropogate with label value
				if(p_backprop((*TRAIN_lbl)(mnIdx),m_batchSize)){
					std::cout<<"ERROR: failure at backpropogation step\n";
					return(1);
				}
					
				//increment index pointer
				mnIdx++;
			}

			//average gradients and apply to weights
			if(p_updateWeights()){
				std::cout<<"ERROR: weight update unsuccessfull\n";
				return(1);
			}
		}

		
		std::cout<<"COMPLETE -> TESTING\n";
		
		//one test output a accuracy to console
		mnIdx = 0;
		num_correct = 0;	
		//loop though all test imgs and labels and output accuracy
		for(int i=0;i<TEST_img.cols();i++){
		
			for(int l=0; l<TEST_img.rows(); l++){
				INIT_img(l) = TEST_img(l,i);	
			}
				
			if(p_setInputVector(&INIT_img)){
					std::cout<<"ERROR: fail to set image input vector in network\n";
					return(1);
			}
			
			//get network guess
			net_guess = p_runNetwork();
			
			//itterate if correct
			if(net_guess == (*TEST_lbl)(i)){
				num_correct++;
			}
		}
	
		std::cout<<"TEST COMPLETE\n";
		std::cout<<"Number correct   = "<<num_correct<<std::endl;
		std::cout<<"Accuracy         = "<<((float)num_correct / TEST_img.cols())*100<<"%\n";

	}

	return(0);
}

/////////////////////////////////////////////////////////////
//unit test to see intermediate values in a single forward
//pass operation
/////////////////////////////////////////////////////////////
int neural_controller::unit_fpv(std::string wdat){

	//get data
	Eigen::MatrixXf*  s = m_training_block->getImgI();
	Eigen::VectorXi*  v = m_training_block->getLblI();
	Eigen::VectorXi   in(784);
	Eigen::VectorXf** vec = new Eigen::VectorXf*[6];
	
	//weights
	Eigen::MatrixXf* w1; 
	Eigen::MatrixXf* w2;
	Eigen::MatrixXf* w3;
//	Eigen::MatrixXf* w4;
	
	//select random object
	int idx = std::rand() % 60000;
	int result;
	
	//file io object
	file_io f;

	//file control
	std::ofstream unitfile;
	
	//file name
	std::string fh = "unit_forward_pass_test.txt";	
	
	std::cout<<"========================\n";
	std::cout<<"=FORWARD PASS UNIT TEST=\n";
	
	std::cout<<"UNIT: evaluating file name\n";
	if(f.validateFileName(wdat)){
		std::cout<<"FILE: file name not valid, randomizing weights\n";
		f.randomizeWeights(&w1, &w2, &w3/*, &w4*/);
	}else if(f.file_exists(wdat) && (!f.validateFileName(wdat))){
		std::cout<<"FILE: reading weights from ["<<wdat<<"]\n";
		f.readWeights(&w1, &w2, &w3, /*&w4,*/ wdat);
	}else{
		std::cout<<"FILE: ["<<wdat<<"] not found. randomizing weigths\n";
		f.randomizeWeights(&w1, &w2, &w3/*, &w4*/);
	}
	
	std::cout<<"UNIT: loading up weights into network\n";
	p_setMatrixWeights(w1, w2, w3/*, w4*/);
	
	std::cout<<"UNIT: loading up image into network\n";
	
	//load up image
	for(int l=0; l<784; l++){
		in(l) = (*s)(l,idx);	
	}
	if(p_setInputVector(&in)){
			std::cout<<"ERROR: fail to set image input vector in network\n";
			return(1);
	}
	
	//pass image through network	
	result = p_runNetwork();
	std::cout<<"NETWORK: label ["<<(*v)(idx)<<"] network evaluation ["<<result<<"]\n";

	//getting fpv and writing into file
	unitfile.open(fh);
	vec = p_getFPV();
	
	unitfile<<"FORWARD PASS VECTOR UNIT TEST\n";
	unitfile<<"\n    m_v1_w|m_v1_a\n";
	for(int i=0;i<vec[0]->rows();i++){
		unitfile<<std::setw(10)<<(*vec[0])(i)<<"|"<<(*vec[1])(i)<<std::endl;
	}
	unitfile<<"\n    m_v2_w|m_v2_a\n";
	for(int i=0;i<vec[2]->rows();i++){
		unitfile<<std::setw(10)<<(*vec[2])(i)<<"|"<<(*vec[3])(i)<<std::endl;
	}
	unitfile<<"\n    m_v3_w\n";
	for(int i=0;i<vec[4]->rows();i++){
		unitfile<<std::setw(10)<<(*vec[4])(i)<<std::endl;
	}
/*	unitfile<<"\n    m_v4_w|m_v4_a\n";
	for(int i=0;i<vec[6]->rows();i++){
		unitfile<<std::setw(10)<<(*vec[6])(i)<<"|"<<(*vec[7])(i)<<std::endl;
	}
*/
	unitfile<<"\noutvec\n";
	for(int i=0;i<vec[5]->rows();i++)
		unitfile<<(*vec[5])(i)<<", ";
	
	unitfile.close();
	
	std::cout<<"UNIT: vector values written to ["<<fh<<"]\n";	
	return(0);
}
