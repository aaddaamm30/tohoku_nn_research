/****************************************************************
*
*	File		: read_mnist.cpp
*	Description	: Supporting code for data reader functions. Can 
*				  be used to return convenient structures with 
*				  MNIST images and labels using Eigen vector and
*				  matrix objects.
*				  
*	Author		: Adam Loo
*	Last Edited	: Thu June 1 2017
*
****************************************************************/

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include <iomanip>
#include <cstdlib>
#include "read_mnist.h"


//prototypes
int switchIt(int);

///////////////////////////////////////////////////////////////////////////
//constructor function
///////////////////////////////////////////////////////////////////////////
mnist_block::mnist_block(int i){

	//input i will either be 1 to indicate training
	//or a 0 for testing. And the datapath for images
	//and labels will be set appropreately
	std::string pImgData = "";
	std::string pLblData = "";

	if(i == 1){
		if(_setPaths("../train_data/train-images-idx3-ubyte",
						 "../train_data/train-labels-idx1-ubyte"))
			std::cout << "ERROR: at setPaths in mnist_block constructor" << std::endl;
		if(_set_UnitTest("train_data_unit.txt"))
			std::cout << "ERROR: setting unit.txt failure" << std::endl;
	}else if(i == 0){
		if(_setPaths("../test_data/t10k-images-idx3-ubyte",
						 "../test_data/t10k-labels-idx1-ubyte"))
			std::cout << "ERROR: at setPaths in mnist_block constructor" << std::endl;
		if(_set_UnitTest("test_data_unit.txt"))
			std::cout << "ERROR: setting unit.txt failure" << std::endl;
	}

	//function to decode and read data into matrix and label
	if(_readData()){
		std::cout << "ERROR: readData fx failure " << std::endl;
	}

}

///////////////////////////////////////////////////////////////////////////
//public getter methods that returns a single case of a an image vector
///////////////////////////////////////////////////////////////////////////
Eigen::MatrixXd* mnist_block::getImgI(void){
	return(_img);
}

Eigen::VectorXi* mnist_block::getLblI(void){
	return(_lbl);
}

///////////////////////////////////////////////////////////////////////////
//function that reads any amount of data into
//appropreate vector<Eigen::MatrixXd> size
///////////////////////////////////////////////////////////////////////////
int mnist_block::_readData(void){
	
	//variables to hold onto size
	//set and fill the matrix vector array function
	if(_loadUpImgs()){
		std::cout << "ERROR: loudUpImg fx failure " << std::endl;
		return(1);
	}
	
	//set and fill the label vector
	if(_loadUpLbls()){
		std::cout << "ERROR: loadUpLbls fx failure" << std::endl;
		return(1);
	}
	//if successful read
	return(0); 
		
}

///////////////////////////////////////////////////////////////////////////
//function that takes file path input to mnist img data
//and uses encoded information to size appropreate vector
//and then fills the mnist_block class img matrix vectors with
//correct data
///////////////////////////////////////////////////////////////////////////
int mnist_block::_loadUpImgs(void){
		
	//declare varbs
	std::string path = _getImgPath();
	int magic_number = 0;
	int n_images = 0;
	int n_rows = 0;
	int n_cols = 0;
	
	//first open file
	std::ifstream file (path, std::ios::binary);
	if(file.is_open()){

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = switchIt(magic_number);	//get magic number
		file.read((char*)&n_images, sizeof(n_images));
		n_images = switchIt(n_images);			//get number of images
		file.read((char*)&n_rows, sizeof(n_rows));
		n_rows = switchIt(n_rows);				//get number of rows
		file.read((char*)&n_cols, sizeof(n_cols));
		n_cols = switchIt(n_cols);				//get number of colums
	
		//create vector
		Eigen::MatrixXd* m = new Eigen::MatrixXd;
		m->resize((n_cols*n_rows), n_images);
		
		//read through rest of data and input into each matrix
		for(int i = 0; i < n_images; ++i){
			for(int j = 0; j < n_rows; ++j){
				for(int k = 0; k < n_cols; ++k){

					unsigned char tmp = 0;
					file.read((char*)&tmp, sizeof(tmp));
					(*m)(((n_rows*j)+k), i) = (double)tmp;
				}
			}
		}
		
		if(_setImgVec(m)){
			std::cout<<"ERROR: setting class image datablock"<<std::endl;
			return(1);
		}
	}

	//if successfull readthrough
	return(0);
}


///////////////////////////////////////////////////////////////////////////
//function that loads up all labels into vector type
//with the appropreate label values at corrosponding indexes
//of the imgage vector
///////////////////////////////////////////////////////////////////////////
int mnist_block::_loadUpLbls(void){

	//variable declarations
	std::string path = _getLblPath();	
	int magic_number = 0;
	int n_labels = 0;

	//first open file
	std::ifstream file (path, std::ios::binary);
	if(file.is_open()){
		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = switchIt(magic_number);	//get magic number
		file.read((char*)&n_labels, sizeof(n_labels));
		n_labels = switchIt(n_labels);			//get number of labels

		//creating vector
		Eigen::VectorXi* v = new Eigen::VectorXi;
		v->resize(n_labels);

		//read through rest of data and add to vector
		for(int i = 0; i < n_labels; i++){
			
			unsigned char tmp = 0;
			file.read((char*)&tmp, sizeof(tmp));
			(*v)(i) = (int)tmp;
		}

		//std::cout << v << std::endl;
		if(_setLblVec(v)){
			std::cout<<"ERROR: setting class label vector"<<std::endl;
			return(1);
		}
	}
	//if successfull
	return(0);	
}

///////////////////////////////////////////////////////////////////////////
//helper fuction to put into msb
///////////////////////////////////////////////////////////////////////////
int switchIt(int i){

	unsigned char ch1, ch2, ch3, ch4;

	ch1 = i&255;
	ch2 = (i>>8)&255;
	ch3 = (i>>16)&255;
	ch4 = (i>>24)&255;
	int y=((int)ch1<<24)+
		  ((int)ch2<<16)+
		  ((int)ch3<<8)+
		  (int)ch4;
	
	return(y);
}

///////////////////////////////////////////////////////////////////////////
//UNIT TEST TIME :)
// - unit test that creates a txt file output with a single mnist image
// 	 vector and label and also prints out to the console size information
//	 and other helpfull info. Evectively tests the read feature, the number
//	 of values read in, and creates a text file of a vector at random.
///////////////////////////////////////////////////////////////////////////
int mnist_block::run_unit(void){

	//get working vectors and matrixes
	Eigen::MatrixXd* image = getImgI();
	Eigen::VectorXi* label = getLblI();
	
	int stella = std::rand() % image->cols();
	std::cout << "\n====================================";
	std::cout << "\n===In mnist_block class unit test===\n";
	std::cout << "====================================\n";
	std::cout << "Number of image vectors: " << image->cols() << std::endl;
	std::cout << "Size of image vectors: " << image->rows() << std::endl;
	std::cout << "Number of labels: " << label->size() << std::endl;
	
	//print randomized vector and label to txt file
	std::ofstream unitfile;
	unitfile.open(_getFilePath());
	unitfile << "This is unit test output.\n\n"
			 << "Below is 28 X 28 matrix that represents the number " 
			 << (*label)(stella) << std::endl << std::endl;
	
	for(int n = 0; n < 28; n++){
		for(int j = 0; j < 28; j++){
			unitfile << std::setw(3) << (*image)((n*28)+j, stella) << " ";
		}
		unitfile << std::endl;
	}
	
	std::cout << "Printed vector and label of data " 
			  << stella << " to " << _getFilePath() << std::endl;

	return(0);	//be successfull my yung padawan
}

///////////////////////////////////////////////////////////////////////////
//setter declaration
///////////////////////////////////////////////////////////////////////////
int mnist_block::_setPaths(std::string pics, std::string labels){
	_pImgData = pics;
	_pLblData = labels;
	return(0);
}
int mnist_block::_setImgVec(Eigen::MatrixXd* in){
	_img = in;
	return(0);
}
int mnist_block::_setLblVec(Eigen::VectorXi* in){
	_lbl = in;
	return(0);
}
int mnist_block::_set_UnitTest(std::string fileName){
	_unit_file_name = fileName;
	return(0);
}
