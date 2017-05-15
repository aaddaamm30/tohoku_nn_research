/****************************************
*
*	eigen practice program 
*
*	last updated: 5/9/2017 Adam Loo
*
*	purpose: practice and learn about
*			 Eigen matrix object and
*			 built in functionality with
*			 focus on how to use in NN
*
****************************************/

#include <iostream>
#include <Eigen/Dense>

using Eigen::MatrixXd;

int main(int argc, char **argv){
	
	MatrixXd m(2,2);
	m(0,0) = 3;
	m(1,0) = 2.5;
	m(0,1) = -1;
	m(1,1) = m(1,0) + m(0,1);

	std::cout << m << std::endl;

	return 0;
}
