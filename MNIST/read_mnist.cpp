/***************************************
*
*	read_mnist.cpp 
*
*	last updated: 5/15/2017 Adam Loo
* 
*	Description:
*	supporting code for data reader
*	functions. Can be used to return
*	convenient structures with MNIST 
*	images and labels. The type that
*	the input will be defined as is
*	going to be a vector of matricies.
*	Specifically vector types with all
*	image data layed out in a one
*	dimentional array like a vector.
*	Each matrix will be 1X
*
****************************************/

#include <iostream>
#include <vector>
#include <string>
#include <Eigen/Dense>
#include <fstream>
#include "read_mnist.h"

//prototypes
int switchIt(int);
int readData(std::string, std::string);
int loadUpImgs(std::string);
int loadUpLbls(std::string);

//constructor function
mnist_block::mnist_block(int i){

	//input i will either be 1 to indicate training
	//or a 2 for testing. And the datapath for images
	//and labels will be set appropreately
	std::string pImgData = "";
	std::string pLblData = "";

	if(i == 1){
		pImgData = "../train_data/train-images-idx3-ubyte";
		pLblData = "../train_data/train-labels-idx3-ubyte";
	}else if(i == 0){
		pImgData = "../test_data/t10k-images-idx3-ubyte";
		pLblData = "../test_data/t10k-labels-idx3-ubyte";	
	}

	//function to decode and read data into matrix and label
	if(readData(pImgData, pLblData)){
		std::cout << "ERROR: readData fx failure " << std::endl;
	}

}

//function that reads any amount of data into
//appropreate vector<Eigen::MatrixXd> size
int readData(std::string imgPath, std::string lblPath){
	
	//variables to hold onto size
	//set and fill the matrix vector array function
	if(loadUpImgs(imgPath)){
		std::cout << "ERROR: loudUpImg fx failure " << std::endl;
		return(1);
	}
	
	//set and fill the label vector
	if(loadUpLbls(lblPath)){
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
int loadUpImgs(std::string path){
	
	//declare varbs
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
		
		//resize vector
		this->img.resize(n_images, Eigen::MatrixXd m((n_rows*n_cols), 1));
		
		//read through rest of data and input into each matrix
		for(int i = 0; i < n_images; ++i){
			for(int j = 0; j < n_rows; ++j){
				for(int k = 0; k < n_cols; ++k){

					unsigned char tmp = 0;
					file.read((char*)&tmp, sizeof(tmp));
					this->img[i](((n_rows*j)+k), 0) = (double)tmp;
				}
			}
		}
	}

	//if successfull readthrough
	return(0);
}

//function that loads up all labels into vector type
//with the appropreate label values at corrosponding indexes
//of the imgage vector
int loadUpLbls(std::string path){
	
	int magic_number = 0;
	int n_labels = 0;

	//first open file
	std::ifstream file (path, std::ios::binary);
	if(file.is_open()){

		file.read((char*)&magic_number, sizeof(magic_number));
		magic_number = switchIt(magic_number);	//get magic number
		file.read((char*)&n_labels, sizeof(n_labels));
		n_labels = switchIt(n_labels);			//get number of labels

		//resize vector
		this->lbl.resize(n_labels);
		
		//read through rest of data and add to vector
		for(int i = 0; i < n_labels; i++){
			
			unsigned char tmp = 0;
			file.read((char*)&tmp, sizeof(tmp));
			this->lbl[i] = (int)tmp;
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
