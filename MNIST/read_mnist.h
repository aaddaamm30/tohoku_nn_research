/****************************************************************
*
*	File		: read_mnist.h
*	Description	: header to describe a custom mnist reader that 
*				  handles the reading for both the training set 
*				  and the testing set of mnist data. Very simple 
*				  top level functions that return a Eigen library
*				  based class with different size for both test 
*				  or train data grabs. Only offers 2 functions 
*				  that return a pointer to an eigen matrix 
*				  (double) for the images and an eigen vector (int) 
*				  for the labels.
*
*	Author		: Adam Loo
*	Last Edited	: Tue May 16 2017
*
****************************************************************/
#ifndef _MNIST_ILLITERATE_
#define _MNIST_ILLITERATE_

#include <iostream>
#include <vector>
#include <Eigen/Dense>

class mnist_block{

public:
	
	//constructor for train and test
	mnist_block(int);
	
	//getters
	Eigen::MatrixXd getImgI(int);
	Eigen::VectorXi getLblI(int);
	std::string getImgPath(){
		return(this->pImgData);
	}
	std::string getLblPath(){
		return(this->pLblData);
	}
	
private:

	//functions used just by constructor
	int readData(void);
	int loadUpImgs(void);
	int loadUpLbls(void);
	int setImgVec(Eigen::MatrixXd*);
	int setLblVec(Eigen::VectorXi*);
	int setPaths(std::string, std::string);
	
	//parallel vectors of images and labels
	Eigen::MatrixXd *img;
	Eigen::VectorXi *lbl;

	std::string pImgData = "";
	std::string pLblData = "";

};

#endif
