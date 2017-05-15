/**********************************
*
* last updated : 5/8/2017 Adam Loo
*
* Basic MNIST neural networks
*
* Purpose: implement a basic neural
*	   network for the purpose 
*	   of education and practic
*
* File: main.cpp - calls and
*	constructs neural network 
*	learning and testing
**********************************/

#include <iostream>
#include "basic_neuron.h"

using namespace std;

int main(int argc, char **argv){

    nn_Neuron *test1 = new nn_Neuron(0.3);
	nn_Neuron *test2 = new nn_Neuron(0.1);
	nn_Neuron *test3 = new nn_Neuron(-0.4);

	cout<<"test 1 n_val: " << test1->get_n_Val() << endl;
	cout<<"       n_fireVal: " << test1->get_n_fireVal() << endl;
	cout<<"       n_dirVal: " << test1->get_n_dirVal() << endl;
	
	cout<<"test 2 n_val: " << test2->get_n_Val() << endl;
	cout<<"       n_fireVal: " << test2->get_n_fireVal() << endl;
	cout<<"       n_dirVal: " << test2->get_n_dirVal() << endl;
	
	cout<<"test 1 n_val: " << test3->get_n_Val() << endl;
	cout<<"       n_fireVal: " << test3->get_n_fireVal() << endl;
	cout<<"       n_dirVal: " << test3->get_n_dirVal() << endl;

	//return 0 :)
    return 0;
}
