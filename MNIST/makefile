SELL = /bin/sh

#path to eigen headers
P="../eigen3/"

#o files
OBJECTS=read_mnist.o

#final executable
EXEC=mnistNet

#flags
CXXFLAGS=-std=c++11 -c -Wall -I $P

#compiler
CC=g++

#Perform action on all object files
all: $(OBJECTS)
	$(CC) $(CXXFLAGS) -o $(EXEC) $(OBJECTS)
