/****************************************************************
*
*	File		: weight_driver.cpp
*	Description	: Methods for the file_io class defined in 
*				  weight_driver.h with full support
*
*	Author		: Adam Loo
*	Last Edited	: Thu Jun 22 2017
*
****************************************************************/

//includes
#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include <sys/stat.h>
#include <unistd.h>
#include <iomanip>
#include <boost/lexical_cast.hpp>
#include <stdlib.h>
#include <time.h>
#include "weight_driver.h"

/////////////////////////////////////////////////////////////////////
//identifies if there is the appropreate .txt ending to a string
/////////////////////////////////////////////////////////////////////
int file_io::validateFileName(std::string f_name){
	
	int i = f_name.length();
	
	//error check file name size
	if(i < 5){
		std::cout << "\nERROR: file name too short. May be "
				  << "missing \".txt\" file descriptor.\n" << std::endl;
		return(1);
	}
	
	if(!(f_name[i-1] == 't' && f_name[i-2] == 'x' &&
	   f_name[i-3] == 't' && f_name[i-4] == '.')){
		std::cout << "ERROR: file input string not \".txt\" type. " 
				  << "Must be of \".txt\" type.\n" << std::endl;
		return(1);
	}
	
	return(0);
}

/////////////////////////////////////////////////////////////////////
//Randomizes the weigths for the matrices. 
// -originally three matrices of 784 x 784 and output of 10 x 784
//	updated design for input layer of 784 input nodes, then a 500
//	neuron layer, then a 1000 neuron layer, then a 50 neuron layer
//	then output layer of 10 that then gets softmax analysis done
//	
// - still initializes all to random values between [-.5, .5]
//	 (this value may be fun to play with in the future)
//
// - Commeted out code is for 4 layer network (500, 1000, 50, 10)
//	 now implementing 3 layer (500, 1000, 10)
/////////////////////////////////////////////////////////////////////
int file_io::randomizeWeights(Eigen::MatrixXd** w1,
							  Eigen::MatrixXd** w2,
							  Eigen::MatrixXd** w3/*,
							  Eigen::MatrixXd** o4*/){

	*w1 = new Eigen::MatrixXd;
	*w2 = new Eigen::MatrixXd;
	*w3 = new Eigen::MatrixXd;
//	*o4 = new Eigen::MatrixXd;

	(*w1)->resize(500, 784);
	(*w2)->resize(1000, 500);
	(*w3)->resize(10, 1000);
//	(*w3)->resize(50, 1000);
//	(*o4)->resize(10, 50);

	//truely randomize
	std::srand(time(NULL));
	
	//randomizing values between -.5 and .5
	for(int i = 0; i < (*w1)->rows(); i++){
		for(int j = 0; j < (*w1)->cols(); j++){
			(**w1)(i,j) = ((((double)std::rand() / RAND_MAX) / 10) - .05);
		}
	}
	for(int i = 0; i < (*w2)->rows(); i++){
		for(int j = 0; j < (*w2)->cols(); j++){
			(**w2)(i,j) = ((((double)std::rand() / RAND_MAX) / 10) - .05);
		}
	}
	for(int i = 0; i < (*w3)->rows(); i++){
		for(int j = 0; j < (*w3)->cols(); j++){
			(**w3)(i,j) = ((((double)std::rand() / RAND_MAX) / 10) - .05);
		}
	}

/*
	for(int i = 0; i < (*o4)->rows(); i++){
		for(int j = 0; j < (*o4)->cols(); j++){
			(**o4)(i,j) = ((((double)std::rand() / RAND_MAX) / 10) - .05);
		}
	}
*/

	return(0);
}

/////////////////////////////////////////////////////////////////////
//file writes data to .txt file according to example.txt doc specs
/////////////////////////////////////////////////////////////////////
int file_io::writeWeights(Eigen::MatrixXd* w1,
						  Eigen::MatrixXd* w2,
						  Eigen::MatrixXd* w3,
						  /*Eigen::MatrixXd* o4,*/
						  std::string f_name){
	
	//first open the file
	std::ofstream weights;
	weights.open(f_name);
	
	//inputting header description
	weights << "Matrix Weights\n";
	
	//first matrix input
	weights << "Hidden Layer 1\nsize: ";
	weights << w1->rows();
	weights << " X ";
	weights << w1->cols();
	weights << "\n";

	//loops through array
	for(int n = 0; n < w1->rows(); n++){
		for(int m = 0; m < w1->cols(); m++){
			weights << (*w1)(n,m) << "|";
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
			weights << (*w2)(n,m) << "|";
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
			weights << (*w3)(n,m) << "|";
		}
		
		//adds line break at end of row
		weights << "\n";
	}

/*
	//output layer matrix values (note different header)
	weights << "\nOutput Layer\nsize: ";
	weights << o4->rows();
	weights << " X ";
	weights << o4->cols();
	weights << "\n";

	//loops through array
	for(int n = 0; n < o4->rows(); n++){
		for(int m = 0; m < o4->cols(); m++){
			weights << (*o4)(n,m) << "|";
		}
		
		//adds line break at end of row
		weights << "\n";
	}
*/
	
	weights.close();
	
	return(0);
}

/////////////////////////////////////////////////////////////////////
//Reads matrices in from txt doc. 
// - Evaluates char by char for weight values then casts to double
// - Creates the matrixes and assigns them to the passed pointer
// - Uses same double pointer implementation that the randomize
//	 method uses
//
//	code admitedly bad. doing it char at a time clumsy and ugly
/////////////////////////////////////////////////////////////////////
int file_io::readWeights(Eigen::MatrixXd** w1,
						 Eigen::MatrixXd** w2,
						 Eigen::MatrixXd** w3,
						 /*Eigen::MatrixXd** w4,*/
						 std::string f_name){
	//create new matrixes
	*w1 = new Eigen::MatrixXd;
	*w2 = new Eigen::MatrixXd;
	*w3 = new Eigen::MatrixXd;
//	*w4 = new Eigen::MatrixXd;

	char tmp[20];
	char c;
	int i, row, col;
	//open file for reading
	std::ifstream weights;
	weights.open(f_name);
	if(weights.is_open()){
		
		//algorithm for parsing though matrix file header and 
		//extracting row and col	
		do{
			weights.get(c);
		}while(c != ':');
		weights.get(c);weights.get(c);
		i=0;
		
		do{
			tmp[i] = c;
			weights.get(c);
			i++;
		}while(c != ' ');
		
		tmp[i] = '\0';
		row = boost::lexical_cast<int>(tmp);
		i=0;
		weights.get(c);weights.get(c);weights.get(c);	//c='X'->' '->'[num]' respectively
		do{
			tmp[i] = c;
			weights.get(c);
			i++;
		}while(c != '\n');
		tmp[i] = '\0';
		col = boost::lexical_cast<int>(tmp);
		weights.get(c);	//reader is at first char of first num

		//resize weights 1 and read in data, ends loop at end of first matrix
		(*w1)->resize(row, col);
		for(int n = 0; n < (*w1)->rows(); n++){
			for(int m = 0; m < (*w1)->cols(); m++){
				//precondition that reader is at first char of first num
				i=0;
				do{
					tmp[i] = c;
					weights.get(c);
					i++;
				}while(c != '|');
				tmp[i]='\0';
				weights.get(c);
				(**w1)(n,m) = strtod(tmp, NULL);	
			}
			weights.get(c);	//moves reader past '\n' char
		}
		
		//repeate above algorithm on second weights
		do{
			weights.get(c);
		}while(c != ':');
		weights.get(c);weights.get(c);
		i=0;
		do{
			tmp[i] = c;
			weights.get(c);
			i++;
		}while(c != ' ');
		tmp[i] = '\0';
		row = boost::lexical_cast<int>(tmp);
		i=0;
		weights.get(c);weights.get(c);weights.get(c);	//c='X'->' '->'[num]' respectively
		do{
			tmp[i] = c;
			weights.get(c);
			i++;
		}while(c != '\n');
		tmp[i] = '\0';
		col = boost::lexical_cast<int>(tmp);
		weights.get(c);	//reader is at first char of first num

		//resize weights 1 and read in data, ends loop at end of first matrix
		(*w2)->resize(row, col);
		for(int n = 0; n < (*w2)->rows(); n++){
			for(int m = 0; m < (*w2)->cols(); m++){
				//precondition that reader is at first char of first num
				i=0;
				do{
					tmp[i] = c;
					weights.get(c);
					i++;
				}while(c != '|');
				tmp[i]='\0';
				weights.get(c);
				(**w2)(n,m) = strtod(tmp, NULL);	
			}
			weights.get(c);	//moves reader past '\n' char
		}

		//read in data from 3rd matrix
		do{
			weights.get(c);
		}while(c != ':');
		weights.get(c);weights.get(c);
		i=0;
		do{
			tmp[i] = c;
			weights.get(c);
			i++;
		}while(c != ' ');
		tmp[i] = '\0';
		row = boost::lexical_cast<int>(tmp);
		i=0;
		weights.get(c);weights.get(c);weights.get(c);	//c='X'->' '->'[num]' respectively
		do{
			tmp[i] = c;
			weights.get(c);
			i++;
		}while(c != '\n');
		tmp[i] = '\0';
		col = boost::lexical_cast<int>(tmp);
		weights.get(c);	//reader is at first char of first num

		//resize weights 1 and read in data, ends loop at end of first matrix
		(*w3)->resize(row, col);
		for(int n = 0; n < (*w3)->rows(); n++){
			for(int m = 0; m < (*w3)->cols(); m++){
				//precondition that reader is at first char of first num
				i=0;
				do{
					tmp[i] = c;
					weights.get(c);
					i++;
				}while(c != '|');
				tmp[i]='\0';
				weights.get(c);
				(**w3)(n,m) = strtod(tmp, NULL);	
			}
			weights.get(c);	//moves reader past '\n' char
		}

/*		
		//read in data from last matrix and then close file
		do{
			weights.get(c);
		}while(c != ':');
		weights.get(c);weights.get(c);
		i=0;
		do{
			tmp[i] = c;
			weights.get(c);
			i++;
		}while(c != ' ');
		tmp[i] = '\0';
		row = boost::lexical_cast<int>(tmp);
		i=0;
		weights.get(c);weights.get(c);weights.get(c);	//c='X'->' '->'[num]' respectively
		do{
			tmp[i] = c;
			weights.get(c);
			i++;
		}while(c != '\n');
		tmp[i] = '\0';
		col = boost::lexical_cast<int>(tmp);
		weights.get(c);	//reader is at first char of first num


		//resize weights 1 and read in data, ends loop at end of first matrix
		(*w4)->resize(row, col);
		for(int n = 0; n < (*w4)->rows(); n++){
			for(int m = 0; m < (*w4)->cols(); m++){
				//precondition that reader is at first char of first num
				i=0;
				do{
					tmp[i] = c;
					weights.get(c);
					i++;
				}while(c != '|');
				tmp[i]='\0';
				weights.get(c);
				(**w4)(n,m) = strtod(tmp, NULL);	
			}
			weights.get(c);	//moves reader past '\n' char
		}
*/
		weights.close();
	}

	return(0);
}

/////////////////////////////////////////////////////////////////////
//basic test to see if a file exists or not
/////////////////////////////////////////////////////////////////////
bool file_io::file_exists(std::string f_name){
	return(access(f_name.c_str(), F_OK) != -1);
}

/////////////////////////////////////////////////////////////////////
//unit test time muddafuddas
//
// -unit test validates users defined file name and prints success
//	of that operation
// -randomizes weights and prints them out to correct file in the 
//	correct format
// -then reads randomized weights back into the unit test and
//	prints success of that operaion
// -operation accepts a string arguemnt from command line and returns
//	standard 0,1
/////////////////////////////////////////////////////////////////////
int file_io::run_unit(std::string path){

	//weights to be randomized and written
	Eigen::MatrixXd* w1=NULL;
	Eigen::MatrixXd* w2=NULL;
	Eigen::MatrixXd* w3=NULL;
//	Eigen::MatrixXd* w4=NULL;
	
	//weights to be read back in
	Eigen::MatrixXd* iw1=NULL;
	Eigen::MatrixXd* iw2=NULL;
	Eigen::MatrixXd* iw3=NULL;
//	Eigen::MatrixXd* iw4=NULL;
	
	//resize 0 matrix
	Eigen::MatrixXd tst;
		
	std::cout << "\n================================";
	std::cout << "\n===In file_io class unit test===\n";
	std::cout << "================================\n";
	
	if(this->validateFileName(path)){
		std::cout << "FAIL: in validate File Name method\n";
		return(1);
	}

	std::cout << "SUCCESS: found ["<<path<<"] a valid file name\n";
	std::cout << "VALUE:   method file_exists("<<path<<") evaluates to ["<<this->file_exists(path)<<"]\n";

	if(this->randomizeWeights(&w1, &w2, &w3/*, &w4*/)){
		std::cout << "FAIL: in randomize weigths method\n";
		return(1);
	}
	
	std::cout << "SUCCESS: randomized pointers ["<<(std::hex)<<w1<<","
								 				 <<(std::hex)<<w2<<","
								 			 	 <<(std::hex)<<w3<<"]\n";
//											 	 <<(std::hex)<<w4<<"]\n";
	if(this->writeWeights(w1, w2, w3, /*w4,*/ path)){
		std::cout << "FAIL: in write weights method\n";
		return(1);
	}
	std::cout << "SUCCESS: wrote matrix values to ["<<path<<"]\n";

	if(this->readWeights(&iw1, &iw2, &iw3, /*&iw4,*/ path)){
		std::cout << "FAIL: in read weights method\n";
		return(1);
	}	
	std::cout << "SUCCESS: passed through read method\n"
			  << "printing read values to io_unit.txt\n";
			
	if(this->writeWeights(iw1, iw2, iw3, /*iw4,*/ "io_unit.txt")){
		std::cout << "FAIL: in read weights method\n";
		return(1);
	}
	
	std::cout <<"==========================="
			  <<"\nCOMPLETED IO FILE UNIT TEST\n";	
	return(0);
}
