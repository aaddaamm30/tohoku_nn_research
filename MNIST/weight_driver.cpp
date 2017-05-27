/****************************************************************
*
*	File		: weight_driver.cpp
*	Description	: Methods for the file_io class defined in 
*				  weight_driver.h with full support
*
*	Author		: Adam Loo
*	Last Edited	: Sat May 27 2017
*
****************************************************************/

//includes
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include "weight_driver.h"

/////////////////////////////////////////////////////////////////////
//identifies if there is the appropreate .txt ending to a string
/////////////////////////////////////////////////////////////////////
int validateFileName(std::string f_name){
	
	int i = f_name.length();
	char tmp[5];
	
	//error check file name size
	if(i < 5){
		std::cout << "\nERROR: file name too short. May be "
				  << "missing \".txt\" file descriptor.\n" << std::endl;
		return(1);
	}

	std::size_t length = f_name.copy(tmp, (i - 4));
	tmp[length] = '\0';
	if(strncmp(tmp, ".txt", 4) != 0){
		std::cout << "ERROR: file input string not \".txt\" type. " 
				  << "Must be of \".txt\" type.\n" << std::endl;
		return(1);
	}
	
	return(0);
}

/////////////////////////////////////////////////////////////////////
//creates three 784 X 784 matrices and on 10 X 784
//and initializes them all to random values between [.3, .7]
//(this value may be fun to play with in the future)
/////////////////////////////////////////////////////////////////////
int file_io::randomizeWeights(Eigen::MatrixXd* w1,
							  Eigen::MatrixXd* w2,
							  Eigen::MatrixXd* w3,
							  Eigen::MatrixXd* o4){

	Eigen::MatrixXd a(500, 784);
	Eigen::MatrixXd b(1000, 500);
	Eigen::MatrixXd c(5000, 1000);
	Eigen::MatrixXd d(50, 5000);
	Eigen::MatrixXd o(10, 50);

	//randomizing values between -.5 and .5
	for(int i = 0; i < a.rows(); i++){
		for(int j = 0; j < a.cols(); j++){
			a(i,j) = ((((double)std::rand() / RAND_MAX) / 2.0) - .3);
		}
	}
	for(int i = 0; i < b.rows(); i++){
		for(int j = 0; j < b.cols(); j++){
			b(i,j) = ((((double)std::rand() / RAND_MAX) / 2.0) - .3);
		}
	}
	for(int i = 0; i < c.rows(); i++){
		for(int j = 0; j < c.cols(); j++){
			c(i,j) = ((((double)std::rand() / RAND_MAX) / 2.0) - .3);
		}
	}
	for(int i = 0; i < d.rows(); i++){
		for(int j = 0; j < d.cols(); j++){
			d(i,j) = ((((double)std::rand() / RAND_MAX) / 2.0) - .3);
		}
	}

	for(int i = 0; i < o.rows(); i++){
		for(int j = 0; j < o.cols(); j++){
			o(i,j) = ((((double)std::rand() / RAND_MAX) / 2.0) - .3);
		}
	}

	//setting the passed pointers to the new matrices we just randomized	
	w1 = &a;
	w2 = &b;
	w3 = &c;
	o4 = &o;

	return(0);
}

/////////////////////////////////////////////////////////////////////
//file writes data to .txt file according to example.txt doc specs
/////////////////////////////////////////////////////////////////////
int file_io::writeWeights(Eigen::MatrixXd* w1,
						  Eigen::MatrixXd* w2,
						  Eigen::MatrixXd* w3,
						  Eigen::MatrixXd* o4,
						  std::string f_name){
	
	//first open the file
	std::ofstream weights;
	weights.open(f_name);
	
	//inputting header description
	weights << "Matrix Weigths\n";
	
	//first matrix input
	weights << "Hidden Layer 1\nsize: ";
	weights << w1->rows();
	weights << " X ";
	weights << w1->cols();
	weights << "\n";

	//loops through array
	for(int n = 0; n < w1->rows(); n++){
		for(int m = 0; m < w1->cols(); m++){
			weights << (*w1)(n,m) << " ";
		}
		
		//adds line break at end of row
		weights << "\n";
	}
	
	//second matrix input (note starts with line break)
	weights << "\nHidden Layer 2\nsize: ";
	weights << w2->rows();
	weights << " X ";
	weights << w2->cols();
	weights << "\n";

	//loops through array
	for(int n = 0; n < w2->rows(); n++){
		for(int m = 0; m < w2->cols(); m++){
			weights << (*w2)(n,m) << " ";
		}
		
		//adds line break at end of row
		weights << "\n";
	}
	
	//third matrix input (note starts with line break)
	weights << "\nHidden Layer 3\nsize: ";
	weights << w3->rows();
	weights << " X ";
	weights << w3->cols();
	weights << "\n";

	//loops through array
	for(int n = 0; n < w3->rows(); n++){
		for(int m = 0; m < w3->cols(); m++){
			weights << (*w3)(n,m) << " ";
		}
		
		//adds line break at end of row
		weights << "\n";
	}
	
	//output layer matrix values (note different header)
	weights << "\nOutput Layer\nsize: ";
	weights << o4->rows();
	weights << " X ";
	weights << o4->cols();
	weights << "\n";

	//loops through array
	for(int n = 0; n < o4->rows(); n++){
		for(int m = 0; m < o4->cols(); m++){
			weights << (*o4)(n,m) << " ";
		}
		
		//adds line break at end of row
		weights << "\n";
	}
	
	weights.close();
	
	return(0);
}

int file_io::readWeights(Eigen::MatrixXd* w1,
						 Eigen::MatrixXd* w2,
						 Eigen::MatrixXd* w3,
						 Eigen::MatrixXd* o4,
						 std::string f_name){
	//SOME CODE BRO
	return(0);
}

/////////////////////////////////////////////////////////////////////
//basic test to see if a file exists or not
/////////////////////////////////////////////////////////////////////
bool file_io::file_exists(std::string& f_name){
	return(access(f_name.c_str(), F_OK) != -1);
}
