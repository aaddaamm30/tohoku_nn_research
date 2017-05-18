##############################
#
# neurnalNetwork Makefile
#
# Last Edited: May thu 18 2017
# Author: Adam Loo
#
##############################

#environment
P="../eigen3/"
CC=g++
CXXFLAGS=-std=c++11 -c -Wall -I $P
LINKARGS=-g
EXEC=neuralNetwork

#may need to add libraries

#filespecs
SOURCES=read_mnist.o nn_driver.o

#make some shit
all: $(SOURCES)  $(EXEC)

$(EXEC): $(SOURCES)
	$(CC) $(LINKARGS) $(SOURCES) -o $(EXEC)

read_mnist.o: read_mnist.cpp read_mnist.h
	$(CC) $(CXXFLAGS) read_mnist.cpp

nn_driver.o: nn_driver.cpp
	$(CC) $(CXXFLAGS) nn_driver.cpp

#clean action
clean: 
	rm -f read_mnist.o nn_driver.o neuralNetwork
