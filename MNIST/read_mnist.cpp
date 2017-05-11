/***************************************
*
*	read_mnist.cpp 
*
*	last updated: 5/9/2017 Adam Loo
* 
*	Description:
*	supporting code for data reader
*	functions. Can be used to return
*	convenient structures with MNIST 
*	images and labels. The type that
*	the input will be defined as is
*	going to be a vector of vectors.
*
****************************************/

#include <iostream>
#include <vector>
#include <string>

using namespace std;

//prototypes
int SwitchIt(int);
void readTheDat(int, int, *vector<vector<double>>);

//function that reads any amount of data into
//appropreate vector<vector<double>> size
void readTheDat(string path, int totalImg, int imgDat, vector<vector<double>> &info){
	
	//resize vector to correct amount
	info.resize(totalImg, vector<double>(imgDat));
}



//helper fuction to put into msb
int SwitchIt(int i){

	unsigned char ch1, ch2, ch3, ch4;
	
	cout << "Switched i: " << hex << i << endl;
	
	ch1 = i&255;
	ch2 = (i>>8)&255;
	ch3 = (i>>16)&255;
	ch4 = (i>>24)&255;
	int y=((int)ch1<<24)+
		  ((int)ch2<<16)+
		  ((int)ch3<<8)+
		  (int)ch4;
	
	cout << "to y: " << hex << y << endl << endl;

	return(y);
}
