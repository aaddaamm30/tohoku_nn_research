/************************************************
*
*	read_mnist.h
*
*	last updated: 5/15/2017 Adam (sky dream) Loo
*
*	description:
*	header to describe a custom mnist reader
*	that handles the reading for both the
*	training set and the testing set of mnist
*	data. Very simple top level functions that
*	return a vector based class with different 
*	size for either test or train data grabs
*	only offers 2 functions that both return
*	a pointer to a vector of vectors. Each
*	vector represents a an image and each
*	vector vector respresents the data and 
*	label for said image.
*
************************************************/

#ifndef _MNIST_ILLITERATE_
#define _MNIST_ILLITERATE_

#include <iostream>
#include <vector>
#include <Eigen/Dense>

class mnist_block{

public:
	
	//constructor for train and test
	mnist_block(int);

	//setters
	int setImgVec(*std::vector<Eigen::MatrixXd>);
	int setLblVec(*std::vector<int>);
	
	//getters
	Eigen::MatrixXd getImgI(int);
	int getLblI(int);
	
private:
	
	//parallel vectors of images and labels
	std::vector<Eigen::MatrixXd> *img;
	std::vector<int> *lbl;
};

#endif
