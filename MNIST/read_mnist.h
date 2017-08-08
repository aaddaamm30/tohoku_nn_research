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
*				  (double) for the images and an eigen vector 
*				  (int) for the labels.
*
*	Author		: Adam Loo
*	Last Edited	: Wed Jul 12 2017
*
****************************************************************/
#ifndef _MNIST_ILLITERATE_
#define _MNIST_ILLITERATE_

#include <iostream>
#include <Eigen/Dense>

class mnist_block{

public:
	
	//constructor for train and test
	mnist_block(int);
	
	//getters
	Eigen::MatrixXf* getImgI(void);
	Eigen::VectorXi* getLblI(void);
	

	//unit test that outputs valuable information to the console
	//and also creates a txt file with vector information for one
	//training vector/label and one testing vector/label
	int run_unit(void);

private:

	std::string _getImgPath(void){
		return(_pImgData);
	}
	std::string _getLblPath(void){
		return(_pLblData);
	}
	std::string _getFilePath(void){
		return(_unit_file_name);
	}
	
	//functions used just by constructor
	int _readData(void);
	int _loadUpImgs(void);
	int _loadUpLbls(void);
	int _setImgVec(Eigen::MatrixXf*);
	int _setLblVec(Eigen::VectorXi*);
	int _setPaths(std::string, std::string);
	int _set_UnitTest(std::string);
	
	//parallel vectors of images and labels
	Eigen::MatrixXf *_img;
	Eigen::VectorXi *_lbl;

	std::string _pImgData = "";
	std::string _pLblData = "";
	std::string _unit_file_name = "";
};

#endif
