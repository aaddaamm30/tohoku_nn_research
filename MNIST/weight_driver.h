/****************************************************************
*
*	File		: weight_driver.h
*	Description	: Class description of weigth input output manager
*				  object. Is in charge of randomizing weights,
*				  outputting weights to user defined txt doc
*				  or reading in weights from user defined txt doc.
*				  throws error if user defined doc is not a .txt
*				  type file or if the file is corrupted or if the
*				  file doesn't exist.
*
*	Author		: Adam Loo
*	Last Edited	: Mon May 29 2017
*
****************************************************************/
#ifndef _WEIGHT_FILE_READER_WRITER_
#define _WEIGHT_FILE_READER_WRITER_

#include <iostream>
#include <Eigen/Dense>
#include <string>

//weight_io class makes very defined weight matrices, three of
//size (784 X 784) and one of size (784 X 10) for the output layer.
//class will be in charge of returning pointers to these matricies
//when asked to either read from a file or randomize and will
//accept four pointers and will be expected to output into a file
//
//this class also handles errors when it comes to validating file
//names and identifiying if there is a file avaliable or not to 
//read from.

class file_io{

	//core componenets of file_io class that nn_engine and nn_driver uses
	public:

		//constructor
		file_io(void){}

		int validateFileName(std::string);
		bool file_exists(std::string);
		int randomizeWeights(Eigen::MatrixXd**,
							 Eigen::MatrixXd**,
							 Eigen::MatrixXd**,
							 Eigen::MatrixXd**);
		int readWeights(Eigen::MatrixXd*,
						Eigen::MatrixXd*,
						Eigen::MatrixXd*,
						Eigen::MatrixXd*, 
						std::string);
		int writeWeights(Eigen::MatrixXd*,
						 Eigen::MatrixXd*,
						 Eigen::MatrixXd*,
						 Eigen::MatrixXd*,
						 std::string);
		//unit test that randomizes weights, outputs 
		//weights to user defined .txt file then reads
		//weights back into program printing success
		//to the terminal then returning 0 to main block
		int run_unit(std::string);
	private:
		
};

#endif
