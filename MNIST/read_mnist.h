/************************************************
*
*	read_mnist.h
*
*	last updated: 5/9/2017 Adam (sky dream) Loo
*
*	description:
*	header to describe a custom mnist reader
*	that handles the reading for both the
*	training set and the testing set of mnist
*	data. Very simple top level functions that
*	return a vector based class with different 
*	size for either test or train data grabs
*	only offers 2 functions that both return
*	a pointer to the same type of class
*
************************************************/

#ifndef _MNIST_ILLITERATE_
#define _MNIST_ILLITERATE_

#include <iostream>
using namespace std;

// this is a class declaration that 
// will be used to navigate the MNIST
// dataset

class mnist_set{
public:
	

private:

	//private constructor as only pointer is passed
	//to main network structure
}

//function to return a mnist_set pointer for training
*mnist_set read_mnist_train();

//function to return a mnist_set pointer for testing
*mnist_set read_mnist_test(); 

#endif
