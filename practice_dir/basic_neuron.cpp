/****************************************
* 
* basic_neuron.cpp
*
* last updated 5/8/2017 Adam Loo
*
* basic_neuron cpp file
*
* purpose: creates appropreat methods
*	   and constructor for the nn_Neuron
*	   class. Header can be referenced
*	   for higher level analysis of
*	   method purposes.
*
*****************************************/

#include "./basic_neuron.h"
#include <cmath>

// constructor
nn_Neuron::nn_Neuron(double val){
	this->n_Val = val;
	sig_activate();
	sig_derive();
}

// sigmoid activation function
void nn_Neuron::sig_activate(){

	//function defined in header
	//setting the fireval using n_val
	this->n_fireVal = this->n_Val / (1 + abs(this->n_Val));
}

//sigmoid derivative function
void nn_Neuron::sig_derive(){
	
	//function defined in header
	//setting the defVal using n_fireVal
	this->n_dirVal = this->n_fireVal * (1 - this->n_fireVal);
}

// ReLU activation function
void nn_Neuron::ReLU_activate(){

	//function defined in header
	//setting fireVal with ReLU function
	if(this->n_Val > 0)
		this->n_fireVal = this->n_Val;
	else
		this->n_fireVal = 0;
}

// ReLU derivateive function
void nn_Neuron::ReLU_derive(){

	//setting dirval to 1 if x>0 and 0 otherwise
	if(this->n_Val > 0)
		this->n_dirVal = 1;
	else
		this->n_dirVal = 0;
}
