/****************************************************************
*
*	File		: read_mnist.cpp
*	Description	: Supporting code for data reader functions. Can 
*				  be used to return convenient structures with 
*				  MNIST images and labels using Eigen vector and
*				  matrix objects.
*				  
*	Author		: Adam Loo
*	Last Edited	: Thu May 25 2017
*
****************************************************************/

#include <iostream>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include "read_mnist.h"


//prototypes
int switchIt(int);

//constructor function
mnist_block::mnist_block(int i){

	//input i will either be 1 to indicate training
	//or a 0 for testing. And the datapath for images
	//and labels will be set appropreately
	std::string pImgData = "";
	std::string pLblData = "";

	if(i == 1){
		if(this->setPaths("../train_data/train-images-idx3-ubyte",
						 "../train_data/train-labels-idx3-ubyte"))
			std::cout << "ERROR: at setPaths in mnist_block constructor" << std::endl;
	}else if(i == 0){
		if(this->setPaths("../test_data/t10k-images-idx3-ubyte",
						 "../test_data/t10k-labels-idx3-ubyte"))
			std::cout << "ERROR: at setPaths in mnist_block constructor" << std::endl;

	}

	//function to decode and read data into matrix and label
	if(this->readData()){
		std::cout << "ERROR: readData fx failure " << std::endl;
	}

}

//public getter methods that returns a single case of a an image vector
Eigen::MatrixXd* mnist_block::getImgI(void){
	return(this->img);
}

Eigen::VectorXi* mnist_block::getLblI(void){
	return(this->lbl);
}

//function that reads any amount of data into
//appropreate vector<Eigen::MatrixXd> size
int mnist_block::readData(void){
	
	//variables to hold onto size
	//set and fill the matrix vector array function
	if(this->loadUpImgs()){
		std::cout << "ERROR: loudUpImg fx failure " << std::endl;
		return(1);
	}
	
	//set and fill the label vector
	if(this->loadUpLbls()){
		std::cout << "ERROR: loadUpLbls fx failure" << std::endl;
		return(1);
	}
	
	//if successful read
	return(0); 
		
}

//function that takes file path input to mnist img data
//and uses encoded information to size appropreate vector
//and then fills the mnist_block class img matrix vectors with
//correct data
int mnist_block::loadUpImgs(void){
		
	//declare varbs
	std::string path = this->getImgPath();
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
		Eigen::MatrixXd m(n_images, (n_rows*n_cols));
	
		//read through rest of data and input into each matrix
		for(int i = 0; i < n_images; ++i){
			for(int j = 0; j < n_rows; ++j){
				for(int k = 0; k < n_cols; ++k){

					unsigned char tmp = 0;
					file.read((char*)&tmp, sizeof(tmp));
					m(i, ((n_rows*j)+k)) = (double)tmp;
				}
			}
		}

		if(this->setImgVec(&m)){
			std::cout<<"ERROR: setting class image datablock"<<std::endl;
			return(1);
		}
	}

	//if successfull readthrough
	return(0);
}

//function that loads up all labels into vector type
//with the appropreate label values at corrosponding indexes
//of the imgage vector
int mnist_block::loadUpLbls(void){

	//variable declarations
	std::string path = this->getLblPath();	
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
		Eigen::VectorXi v(n_labels);

		//read through rest of data and add to vector
		for(int i = 0; i < n_labels; i++){
			
			unsigned char tmp = 0;
			file.read((char*)&tmp, sizeof(tmp));
			v(i) = (int)tmp;
		}

		if(this->setLblVec(&v)){
			std::cout<<"ERROR: setting class label vector"<<std::endl;
			return(1);
		}
	}

	//if successfull
	return(0);	
}

//helper fuction to put into msb
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

//setter declaration
int mnist_block::setPaths(std::string pics, std::string labels){
	this->pImgData = pics;
	this->pLblData = labels;
	return(0);
}
int mnist_block::setImgVec(Eigen::MatrixXd* in){
	this->img = in;
	return(0);
}
int mnist_block::setLblVec(Eigen::VectorXi* in){
	this->lbl = in;
	return(0);
}
