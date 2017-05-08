/****************************************
* 
* basic_neuron.hpp
* 
* last updated 5/8/2017 Adam Loo
*
* basic_neuron header file
*
* defines: defines the baseline neuron
*	   with value, activated value
*	   and derived value. Also
*	   declairs getters and deriving
*	   methods.
*
*****************************************/

#ifndef _BASIC_NEURON_
#define _BASIC_NEURON_

#include<iostream>
#include<cmath>
using namespace std;

//class to be used in main.cpp
class nn_Neuron{

public:

	//constructor
	nn_Neuron(double val);

	//fast sigmoid function
	// 
	//	f(x) = x/(1-|x|)
	//
	//source: Raphael B. Alampay 2/20/2017
	void sig_activate();
	
	//sigmoid derivative function
	//
	// f'(x) = f(x) * (1 - f(x))
	//
	//source: Raphael B. Alampay 2/20/2017
	void sig_derive();

	//ReLU implementation 
	//
	// f(x) = MAX(0,x)
	//
	//source: Stanford CS231n Spring 2017 Lecture notes
	void ReLU_activate();

	//ReLu derivitive function
	//
	//	   | 1, if x>0
	// f'(x) = |
	//	   | 0, otherwise
	//
	//source: Jeremy Kawahara 5/17/2016
	void ReLU_derive();

	//getters
 	double get_n_Val(){
		return this->n_Val;
	}
	double get_n_fireVal(){
		return this->n_fireVal;
	}
	double get_n_dirVal(){
		return this->n_dirVal;
	}

//values that gives neuron its power and 
//usefullness
private:

   double n_Val;	//value of neuron ([addrange])
   double n_fireVal;	//value when fired ([addrange sigmoid so -1 ~ 1?])
   double n_dirVal;	//derived value from firedval and val
	
};

#endif
