/****************************************************************
*
*	File		: weight_driver.cpp
*	Description	: Methods for the file_io class defined in 
*				  weight_driver.h with full support
*
*	Author		: Adam Loo
*	Last Edited	: Thu May 25 2017
*
****************************************************************/

//includes
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include <sys/stat.h>
#include "weight_driver.h"

//identifies if there is the appropreate .txt ending to a string
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

//creates three 784 X 784 matrices and on 10 X 784
//and initializes them all to random values between [.3, .7]
//(this value may be fun to play with in the future)
int file_io::randomizeWeights(Eigen::MatrixXd* w1,
							  Eigen::MatrixXd* w2,
							  Eigen::MatrixXd* w3,
							  Eigen::MatrixXd* o4){

	Eigen::MatrixXd a(784, 784);
	Eigen::MatrixXd b(784, 784);
	Eigen::MatrixXd c(784, 784);
	Eigen::MatrixXd d(784, 784);
	Eigen::MatrixXd o(10, 784);

	for(int i = 0; i < 784; i++){
		for(int j = 0; j < 784; j++){
			a(i,j) = ((((double)std::rand() / RAND_MAX) / 2.0) + .3);
			b(i,j) = ((((double)std::rand() / RAND_MAX) / 2.0) + .3);
			c(i,j) = ((((double)std::rand() / RAND_MAX) / 2.0) + .3);
			d(i,j) = ((((double)std::rand() / RAND_MAX) / 2.0) + .3);
		}
	}

	for(int i = 0; i < 10; i++){
		for(int j = 0; j < 784; j++){
			o(i,j) = ((((double)std::rand() / RAND_MAX) / 2.0) + .3);
		}
	}

	//setting the passed pointers to the new matrices we just randomized	
	w1 = &a;
	w2 = &b;
	w3 = &c;
	w4 = &d;
	o4 = &o;

	return(0);
}

int file_io::writeWeights(Eigen::MatrixXd* w1,
						  Eigen::MatrixXd* w2,
						  Eigen::MatrixXd* w3,
						  Eigen::MatrixXd* o4,
						  std::string f_name){
	// MORE CODE BROWFLOWSKI
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

std::string formatMatrix(Eigen::MatrixXd*){

}

bool file_io::file_exists(std::string f_name){

	struct stat buffer;
	return(stat (f_name.c_str(), buffer) == 0);
}
